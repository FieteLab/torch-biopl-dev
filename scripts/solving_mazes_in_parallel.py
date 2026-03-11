import argparse
import copy
import gc
import json
import os
import pickle
from datetime import datetime
from functools import partial
from typing import Optional, Sized

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from tqdm import tqdm
import wandb
import psutil

from bioplnn.models import SpatiallyEmbeddedClassifier
from bioplnn.base_model_configs import BASE_MODEL_CONFIGS
from bioplnn.utils import (
    initialize_dataloader,
    initialize_scheduler,
    manual_seed,
)

# Defaults (override with CLI)
maze_data_path = "./data/mazes"
cabc_data_path = "./data/easy"

checkpoint_path_default = "./train/checkpoints"

# Torch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


def init_distributed_from_env() -> tuple[bool, int, int, int]:
    """Initialize DDP from environment variables if present (torchrun/SLURM).

    Safe to call multiple times in a process.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank

    if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        rank = int(os.environ["RANK"])  # global rank
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class TinyCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = DSConv(16, 32, stride=2)
        self.block2 = DSConv(32, 64, stride=2)
        self.block3 = DSConv(64, 64, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return self.head(x)


class SimpleCNN(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        dropout=0.3,
        conv1_out=64,
        conv2_out=64,
        conv3_out=128,
        fc_dim=512,
        conv_kernel_size=5,
        pool_kernel=2,
        pool_stride=2,
    ):
        super().__init__()
        padding = (
            conv_kernel_size // 2
            if isinstance(conv_kernel_size, int)
            else (conv_kernel_size[0] // 2)
        )
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=conv_kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=conv_kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=conv_kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(pool_kernel, pool_stride)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(conv3_out, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def load_data(dataset, **args):
    train_loader, test_loader = initialize_dataloader(
        seed=42, dataset=dataset, **args
    )
    return train_loader, test_loader


def evaluate(model, model_type, data_loader, criterion, device, num_steps):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, labels in data_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            if "cnn" in model_type:
                if len(x.shape) == 5:
                    x = x[:, :, 0, :, :]
                logits = model(x)
            else:
                logits = model(x, num_steps=num_steps)

            if labels.ndim == 2 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            elif labels.ndim != 1:
                raise ValueError(f"Expected labels shape [N] or [N,1], got {tuple(labels.shape)}")

            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_cpu_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def init_weights_kaiming(m, nonlinearity="relu"):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights_zero(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def rebuild_loader_with_sampler(original_loader: DataLoader, sampler: DistributedSampler) -> DataLoader:
    return DataLoader(
        original_loader.dataset,
        batch_size=original_loader.batch_size,
        sampler=sampler,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        worker_init_fn=original_loader.worker_init_fn,
        generator=original_loader.generator,
        drop_last=False,
        persistent_workers=getattr(original_loader, "persistent_workers", False),
    )


def train(
    model,
    model_type,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    num_steps,
    max_epochs,
    max_gradient,
    train_log_frequency,
    run_name,
    run,
    checkpoint_dir,
    distributed: bool,
    rank: int,
):
    model.train()

    val_accs: list[float] = []
    patience = 200

    save_path = f"{checkpoint_dir}/{run_name}"
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(max_epochs):
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        if distributed and isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
            
        if run is not None and rank == 0:
            run.config.update({"n_epochs": epoch}, allow_val_change=True)
        running_loss, running_correct, running_total = 0.0, 0, 0

        iterator = tqdm(train_loader) if rank == 0 else train_loader
        for i, (x, labels) in enumerate(iterator):
            x = x.to(device)
            labels = labels.to(device)
            
            if i == 0:
                try:
                    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
                except Exception:
                    current_device = 'cpu'
                try:
                    model_device = next(model.parameters()).device if not isinstance(model, DDP) else next(model.module.parameters()).device
                except StopIteration:
                    model_device = current_device
                mem_alloc = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                mem_rsrv = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
                print(
                    f"[rank {rank}] first batch | device={current_device} | model_device={model_device} | "
                    f"x.device={getattr(x, 'device', 'cpu')} | labels.device={getattr(labels, 'device', 'cpu')} | "
                    f"batch_size={labels.size(0)} | mem_alloc={mem_alloc:.1f}MB | mem_reserved={mem_rsrv:.1f}MB"
                )

            try:
                if "cnn" in model_type:
                    if len(x.shape) == 5:
                        x = x[:, :, 0, :, :]
                    logits = model(x)
                else:
                    logits = model(x, num_steps=num_steps, loss_all_timesteps=False)
            except RuntimeError as e:
                if torch.cuda.is_available() and "out of memory" in str(e).lower():
                    try:
                        print(f"[rank {rank}] OOM during forward. Batch {i}, epoch {epoch}.")
                        print(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=False))
                    except Exception:
                        pass
                raise

            if labels.ndim == 2 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            elif labels.ndim != 1:
                raise ValueError(f"Expected labels shape [N] or [N,1], got {tuple(labels.shape)}")

            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            try:
                loss.backward()
            except RuntimeError as e:
                if torch.cuda.is_available() and "out of memory" in str(e).lower():
                    try:
                        print(f"[rank {rank}] OOM during backward. Batch {i}, epoch {epoch}.")
                        print(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=False))
                    except Exception:
                        pass
                raise
            
            grad_norm = get_gradient_norm(model)
            
            if max_gradient is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            predicted = torch.argmax(logits, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            batch_loss = loss.item()

            running_total += batch_total
            running_correct += batch_correct
            running_loss += batch_loss * batch_total

            if rank == 0 and i % train_log_frequency == 0:
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
                print(
                    f"Batch {i} | "
                    + f"Loss: {batch_loss:.4f} | "
                    + f"Acc: {batch_acc:.2%} | "
                    + f"Grad Norm: {grad_norm:.4f} | "
                    + f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                if run is not None:
                    wandb.log({
                        "train_loss": running_loss / running_total if running_total > 0 else 0.0,
                        "train_acc": running_correct / running_total if running_total > 0 else 0.0,
                        "gradient_norm": grad_norm,
                        "step": epoch * len(train_loader) + i,
                        "lr": optimizer.param_groups[0]["lr"]
                    })
        
        epoch_loss = running_loss / running_total if running_total > 0 else 0.0
        epoch_acc = running_correct / running_total if running_total > 0 else 0.0
        
        if rank == 0:
            print(
                f"Training | Epoch: {epoch} | "
                + f"Loss: {epoch_loss:.4f} | "
                + f"Acc: {epoch_acc:.2%} | "
                + f"GPU Memory: {get_gpu_memory():.1f}MB | "
                + f"CPU Memory: {get_cpu_memory():.1f}MB"
            )

        val_loss, val_acc = evaluate(model, model_type, test_loader, criterion, device, num_steps)
        model.train()
        
        if rank == 0 and run is not None:
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            
        if rank == 0 and epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict() if scheduler else None,
            }
            torch.save(checkpoint, f"{save_path}/{epoch}.pth")
        
        val_accs.append(val_acc)
        
        if rank == 0 and len(val_accs) > patience + 10:
            best_val_acc = max(val_accs[:-patience])
            if all(acc <= best_val_acc for acc in val_accs[-patience:]):
                print("Validation accuracy not improving, stopping training")
                print(f"Best validation accuracy: {best_val_acc:.2%}")
                print(val_accs[-patience:])
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'sched_state': scheduler.state_dict() if scheduler else None,
                }
                torch.save(checkpoint, f"{save_path}/{epoch}.pth")
                break

    if rank == 0:
        print(f"Saving final checkpoint to {save_path}/{epoch}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict() if scheduler else None,
        }
        torch.save(checkpoint, f"{save_path}/{epoch}.pth")


def get_scheduler(scheduler_config, optimizer=None, train_loader=None, max_epochs=None, lr=None):
    if scheduler_config is None:
        return None

    kwargs = scheduler_config["kwargs"].copy()

    if "Cycle" in scheduler_config["name"]:
        if lr is None:
            raise ValueError("lr must be provided for Cycle schedulers")
        kwargs["max_lr"] = lr
        if train_loader is not None and max_epochs is not None:
            kwargs["total_steps"] = len(train_loader) * max_epochs
    elif "Lambda" in scheduler_config["name"]:
        if train_loader is not None and "warmup_epochs" in scheduler_config:
            warmup_steps = len(train_loader) * scheduler_config["warmup_epochs"]
            kwargs["lr_lambda"] = lambda step: (step + 1) / warmup_steps if step < warmup_steps else 1.0

    if optimizer is None:
        raise ValueError("optimizer must be provided to create scheduler")

    scheduler = initialize_scheduler(
        class_name=scheduler_config["name"],
        optimizer=optimizer,
        **kwargs,
    )
    return scheduler


def extract_true_model_config(model, model_type):
    base_model = model.module if isinstance(model, DDP) else model

    if model_type == "cnn":
        m = base_model
        return {
            "in_channels": int(m.conv1.in_channels),
            "num_classes": int(m.fc2.out_features),
            "dropout": float(m.dropout.p),
            "conv1_out": int(m.conv1.out_channels),
            "conv2_out": int(m.conv2.out_channels),
            "conv3_out": int(m.conv3.out_channels),
            "fc_dim": int(m.fc1.out_features),
            "conv_kernel_size": int(m.conv1.kernel_size[0]),
            "pool_kernel": int(m.pool.kernel_size),
            "pool_stride": int(m.pool.stride),
        }
    if model_type == "tiny_cnn":
        m = base_model
        return {
            "in_channels": int(m.stem[0].in_channels),
            "num_classes": int(m.head[-1].out_features),
            "dropout": float(m.head[1].p),
        }

    m = base_model
    readout = m.readout
    fc_dim = int(readout[1].out_features)
    dropout = float(readout[3].p) if isinstance(readout[3], nn.Dropout) else None
    num_classes = int(readout[-1].out_features)
    pool_size_classifier = getattr(m.pool, "output_size", (1, 1))
    pool_mode_classifier = "avg" if isinstance(m.pool, nn.AdaptiveAvgPool2d) else "max"

    r = m.rnn
    area_kwargs = []
    for area in r.areas:
        ak = {
            "in_size": list(area.in_size),
            "in_channels": int(area.in_channels),
            "out_channels": int(area.out_channels),
            "num_neuron_types": int(area.num_neuron_types),
            "num_neuron_subtypes": list(area.num_neuron_subtypes),
            "neuron_type_class": list(area.neuron_type_class),
            "neuron_type_density": list(area.neuron_type_density),
            "inter_neuron_type_connectivity": area.inter_neuron_type_connectivity.tolist(),
            "inter_neuron_type_spatial_extents": area.inter_neuron_type_spatial_extents.tolist() if hasattr(area, "inter_neuron_type_spatial_extents") else None,
            "inter_neuron_type_num_subtype_groups": area.inter_neuron_type_num_subtype_groups.tolist(),
            "inter_neuron_type_nonlinearity": area.inter_neuron_type_nonlinearity.tolist(),
            "inter_neuron_type_bias": area.inter_neuron_type_bias.tolist(),
            "tau_mode": list(area.tau_mode),
            "tau_init_fn": list(area.tau_init_fn),
            "in_class": area.in_class,
            "feedback_class": area.feedback_class,
            "default_neuron_state_init_fn": area.default_neuron_state_init_fn,
            "default_feedback_state_init_fn": area.default_feedback_state_init_fn,
            "default_output_state_init_fn": area.default_output_state_init_fn,
        }
        if getattr(area, "use_feedback", False) and int(area.feedback_channels) > 0:
            ak["feedback_channels"] = int(area.feedback_channels)
        area_kwargs.append(ak)

    rnn_kwargs = {
        "num_areas": int(r.num_areas),
        "area_kwargs": area_kwargs,
        "pool_mode": r.pool_mode,
        "batch_first": bool(r.batch_first),
        "area_time_delay": bool(r.area_time_delay),
    }
    if hasattr(r, "inter_area_feedback_connectivity"):
        rnn_kwargs["inter_area_feedback_connectivity"] = r.inter_area_feedback_connectivity.tolist()
        if hasattr(r, "inter_area_feedback_nonlinearity"):
            rnn_kwargs["inter_area_feedback_nonlinearity"] = r.inter_area_feedback_nonlinearity.tolist()
        if hasattr(r, "inter_area_feedback_spatial_extents"):
            rnn_kwargs["inter_area_feedback_spatial_extents"] = r.inter_area_feedback_spatial_extents.tolist()

    return {
        "rnn_kwargs": rnn_kwargs,
        "num_classes": num_classes,
        "fc_dim": fc_dim,
        "dropout": dropout,
        "pool_size_classifier": list(pool_size_classifier) if isinstance(pool_size_classifier, tuple) else pool_size_classifier,
        "pool_mode_classifier": pool_mode_classifier,
    }


def run_experiment(
    model_type,
    model_config,
    dataset="mazes",
    lr=0.001,
    num_steps: "int | None" = 60,
    max_epochs=10,
    batch_size=128,
    max_gradient=None,
    scheduler_config=None,
    init_weights="kaiming",
    neuron_type_nonlinearity="relu",
    data_root: Optional[str] = maze_data_path,
    checkpoints_dir=checkpoint_path_default,
    wandb_project=False,
    seed=42,
    n_frames_range=None,
    dots_kwargs: dict = {},
    samples_per_epoch: "int | None" = None,
    val_samples_per_epoch: "int | None" = None,
    resolution: Optional[int] = None,
):
    manual_seed(seed)

    distributed, rank, world_size, local_rank = init_distributed_from_env()
    ddp_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if model_type == "cnn":
        model = SimpleCNN(**model_config).to(ddp_device)
    elif model_type == "tiny_cnn":
        model = TinyCNN(**model_config).to(ddp_device)
    else:
        model = SpatiallyEmbeddedClassifier(**model_config).to(ddp_device)
    
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Debug: rank to device mapping
    try:
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        print(f"[ddp] rank={rank}/{world_size}, local_rank={local_rank}, device={current_device}, distributed={distributed}")
    except Exception as e:
        print(f"[ddp] rank={rank} debug device print failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    if dataset == "mazes" or dataset == "cabc":
        train_loader, test_loader = load_data(
            dataset,
            batch_size=batch_size,
            root=data_root,
            samples_per_epoch=samples_per_epoch,
            val_samples_per_epoch=val_samples_per_epoch,
            resolution=None if resolution is None else (resolution, resolution),
        )
    elif dataset == "correlated_dots":
        train_loader, test_loader = load_data(dataset, batch_size=batch_size, **dots_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if distributed:
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = rebuild_loader_with_sampler(train_loader, train_sampler)
        if test_loader is not None:
            test_sampler = DistributedSampler(test_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
            test_loader = rebuild_loader_with_sampler(test_loader, test_sampler)

    train_log_frequency = max(1, len(train_loader) // 10)

    if init_weights == "kaiming":
        model.apply(partial(init_weights_kaiming, nonlinearity=neuron_type_nonlinearity))
    elif init_weights == "zero":
        model.apply(init_weights_zero)

    scheduler = get_scheduler(scheduler_config, optimizer=optimizer, train_loader=train_loader, max_epochs=max_epochs, lr=lr) if scheduler_config else None

    true_model_config = extract_true_model_config(model, model_type)
    train_ds = getattr(train_loader, "dataset", None)
    effective_train_samples = (
        len(train_ds) if isinstance(train_ds, Sized) else None
    )
    effective_val_samples = None
    if test_loader is not None:
        val_ds = getattr(test_loader, "dataset", None)
        if isinstance(val_ds, Sized):
            effective_val_samples = len(val_ds)
        
    full_config = {
            "model_type": model_type,
        "model_config": true_model_config,
        "lr": lr,
        "num_steps": num_steps,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "max_gradient": max_gradient,
        "scheduler": scheduler_config,
        "init_weights": init_weights,
        "seed": seed,
        "dots_kwargs": dots_kwargs,
        "dataset": dataset,
    }
    if resolution is not None:
        full_config["resolution"] = int(resolution)

    if dataset != "correlated_dots":
        if samples_per_epoch is None and effective_train_samples is not None:
            full_config["samples_per_epoch"] = int(effective_train_samples)
        elif samples_per_epoch is not None:
            full_config["samples_per_epoch"] = int(samples_per_epoch)
        if effective_val_samples is not None:
            full_config["val_samples_per_epoch"] = int(effective_val_samples)

        run = None
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    if wandb_project and rank == 0:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb-project was True")
        run = wandb.init(project=dataset, config=full_config)
        run_name = run.name

    if rank == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        with open(os.path.join(checkpoints_dir, f"{run_name}.pkl"), "wb") as f:
            pickle.dump(full_config, f)

    train(
        model,
        model_type,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        num_steps,
        max_epochs,
        max_gradient,
        train_log_frequency,
        run_name,
        run,
        checkpoints_dir,
        distributed,
        rank,
    )

    final_val_loss, final_val_acc = (None, None)
    if rank == 0:
        final_val_loss, final_val_acc = evaluate(model, model_type, test_loader, criterion, device, num_steps)
        if run is not None:
            wandb.log({
                "final_val_loss": final_val_loss,
                "final_val_acc": final_val_acc,
            })
        wandb.finish()
    
    if distributed:
        dist.barrier()

    return final_val_loss, final_val_acc


def apply_model_overrides(model_type: str, model_config: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(model_config)
    if "cnn" in model_type:
        return cfg

    fc_dim = overrides.get("fc_dim")
    if fc_dim is not None:
        cfg["fc_dim"] = fc_dim

    areas = cfg.get("rnn_kwargs", {}).get("area_kwargs", [])
    if not areas:
        return cfg

    out_channels_override = overrides.get("out_channels")
    nnst_override = overrides.get("num_neuron_subtypes")
    inter_conn_override = overrides.get("inter_neuron_type_connectivity")
    spatial_extents_override = overrides.get("inter_neuron_type_spatial_extents")
    inter_nt_nl_override = overrides.get("inter_neuron_type_nonlinearity")
    nt_nl_override = overrides.get("neuron_type_nonlinearity")

    for area in areas:
        if out_channels_override is not None:
            area["out_channels"] = out_channels_override

        if nnst_override is not None:
            num_types = area.get("num_neuron_types", None)
            if isinstance(nnst_override, int):
                if num_types is None:
                    raise ValueError("num_neuron_types missing in base config while applying num_neuron_subtypes=int")
                area["num_neuron_subtypes"] = np.ones(int(num_types), dtype=int) * int(nnst_override)
            elif isinstance(nnst_override, (list, tuple, np.ndarray)):
                area["num_neuron_subtypes"] = np.array(list(map(int, nnst_override)))

        if inter_conn_override is not None:
            area["inter_neuron_type_connectivity"] = np.array(inter_conn_override)

        if spatial_extents_override is not None:
            if isinstance(spatial_extents_override, str):
                if spatial_extents_override == "center_excitation":
                    area["inter_neuron_type_spatial_extents"] = np.array([[(5,5), (5,5), (5,5)], [(3,3), (3,3), (3,3)], [(7,7), (7,7), (7,7)]])
                elif spatial_extents_override == "center_inhibition":
                    area["inter_neuron_type_spatial_extents"] = np.array([[(5,5), (5,5), (5,5)], [(5,5), (5,5), (5,5)], [(3,3), (5,5), (5,5)]])
            else:
                area["inter_neuron_type_spatial_extents"] = spatial_extents_override

        if inter_nt_nl_override is not None:
            connectivity = area.get("inter_neuron_type_connectivity")
            if connectivity is None:
                raise ValueError("inter_neuron_type_connectivity missing; cannot derive matrix shape for nonlinearity")
            rows, cols = connectivity.shape
            area["inter_neuron_type_nonlinearity"] = np.full((rows, cols), inter_nt_nl_override)

        if nt_nl_override is not None:
            area["neuron_type_nonlinearity"] = nt_nl_override

    for i in range(1, len(areas)):
        prev_out = areas[i-1].get("out_channels")
        if prev_out is not None:
            areas[i]["in_channels"] = prev_out

    cfg["rnn_kwargs"]["area_kwargs"] = areas
    return cfg


def run_from_checkpoint(wandb_name, epoch, new_params={}, checkpoints_dir=checkpoint_path_default, data_root: Optional[str] = maze_data_path, wandb_project=False):
    distributed, rank, world_size, local_rank = init_distributed_from_env()
    ddp_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    full_config = pickle.load(open(os.path.join(checkpoints_dir, f"{wandb_name}.pkl"), "rb"))
    for param, value in new_params.items():
        full_config[param] = value

    model_config = full_config["model_config"]
    model_type = full_config.get("model_type", "1e1i1a")

    if model_type == "cnn":
        model = SimpleCNN(**model_config).to(ddp_device)
    elif model_type == "tiny_cnn":
        model = TinyCNN(**model_config).to(ddp_device)
    else:
        model = SpatiallyEmbeddedClassifier(**model_config).to(ddp_device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    checkpoint = torch.load(os.path.join(checkpoints_dir, wandb_name, f"{epoch}.pth"), map_location=ddp_device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

    if isinstance(model, DDP):
        try:
            model.module.load_state_dict(state)
        except RuntimeError:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.module.load_state_dict(state)
    else:
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)

    lr = full_config.get("lr", 0.001)
    num_steps = full_config.get("num_steps", 60)
    max_epochs = full_config.get("max_epochs", 10)
    batch_size = full_config.get("batch_size", 128)
    max_gradient = full_config.get("max_gradient", None)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    dataset = full_config.get("dataset", "mazes")
    if dataset == "correlated_dots":
        train_loader, test_loader = load_data(dataset, batch_size=batch_size, root=data_root, **full_config.get("dots_kwargs", {}))
    else:
        train_loader, test_loader = load_data(
            dataset,
            batch_size=batch_size,
            root=data_root,
            samples_per_epoch=full_config.get("samples_per_epoch", None),
            val_samples_per_epoch=full_config.get("val_samples_per_epoch", None),
        )

    if distributed:
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = rebuild_loader_with_sampler(train_loader, train_sampler)
        if test_loader is not None:
            test_sampler = DistributedSampler(test_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
            test_loader = rebuild_loader_with_sampler(test_loader, test_sampler)

    train_log_frequency = max(1, len(train_loader) // 10)

    scheduler = get_scheduler(full_config.get("scheduler", None), optimizer=optimizer, train_loader=train_loader, max_epochs=max_epochs, lr=lr)

    if isinstance(checkpoint, dict) and 'sched_state' in checkpoint and checkpoint['sched_state'] and scheduler:
        try:
            scheduler.load_state_dict(checkpoint['sched_state'])
        except Exception:
            pass

    run = None
    if wandb_project and rank == 0:
        true_model_config = extract_true_model_config(model, model_type)
        full_config["model_config"] = true_model_config

        if dataset != "correlated_dots":
            train_ds = getattr(train_loader, "dataset", None)
            val_ds = getattr(test_loader, "dataset", None) if test_loader is not None else None
            eff_train = len(train_ds) if isinstance(train_ds, Sized) else None
            eff_val = len(val_ds) if isinstance(val_ds, Sized) else None
            if eff_train is not None:
                full_config["samples_per_epoch"] = int(eff_train)
            if eff_val is not None:
                full_config["val_samples_per_epoch"] = int(eff_val)

        run = wandb.init(
            project=dataset,
            config=full_config,
        )

    wandb_name_effective = run.name if run is not None else wandb_name
    if rank == 0:
        with open(os.path.join(checkpoints_dir, f"{wandb_name_effective}.pkl"), "wb") as f:
            pickle.dump(full_config, f)
    
    train(
        model,
        model_type,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        num_steps,
        max_epochs,
        max_gradient,
        train_log_frequency,
        wandb_name_effective,
        run,
        checkpoints_dir,
        distributed,
        rank,
    )

    final_val_loss, final_val_acc = (None, None)
    if rank == 0:
        final_val_loss, final_val_acc = evaluate(model, model_type, test_loader, criterion, device, num_steps)
        if run is not None:
            wandb.log({
                "final_val_loss": final_val_loss,
                "final_val_acc": final_val_acc,
            })
        wandb.finish()
    
    if distributed:
        dist.barrier()

    return final_val_loss, final_val_acc


def parse_tuple(s):
    if "," in s:
        parts = s.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("range must be two numbers: low,high")
        try:
            low, high = map(float, parts)
            if low.is_integer():
                low = int(low)
                high = int(high)
        except ValueError:
            raise argparse.ArgumentTypeError("range must be numbers")
        return (low, high)
    else:
        try:
            val = float(s)
            if val.is_integer():
                return int(val)
            return val
        except ValueError:
            raise argparse.ArgumentTypeError("must be a number or 'low,high'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single model (DDP) on a dataset")

    # Data selection
    parser.add_argument("--dataset", type=str, default="mazes")

    # Model selection
    parser.add_argument("--model-type", type=str, default="cnn",
                        choices=list(BASE_MODEL_CONFIGS.keys()),
                        help="Model preset to use")
    parser.add_argument("--model-config-file", type=str, default=None,
                        help="Path to a JSON file containing the model_config dict. Overrides preset.")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-gradient", type=float, default=None)
    parser.add_argument("--init-weights", type=str, default="kaiming", choices=["kaiming", "zero", "none"])
    parser.add_argument("--fc-dim", type=int, default=None, help="Override fully connected layer width (CNNs and non-CNNs)")
    parser.add_argument("--conv1-out", type=int, default=None, help="CNN conv1 out channels")
    parser.add_argument("--conv2-out", type=int, default=None, help="CNN conv2 out channels")
    parser.add_argument("--conv3-out", type=int, default=None, help="CNN conv3 out channels")
    parser.add_argument("--out-channels", type=int, default=None, help="Override out_channels in area 0 for non-CNN models")
    parser.add_argument("--num-neuron-subtypes", type=str, default=None, help="Override number of neuron subtypes; either an int or comma-separated list like 8,4")
    parser.add_argument("--neuron-type-nonlinearity", type=str, default=None, help="Override neuron_type_nonlinearity for area 0")
    parser.add_argument("--inter-neuron-type-nonlinearity", type=str, default=None, help="Fill the inter-neuron type nonlinearity matrix with a single value (e.g., ReLU)")
    parser.add_argument("--inter-neuron-type-spatial-extents", type=str, default=None, help='Either a tuple like "5,5" or a keyword: center_excitation | center_inhibition')
    parser.add_argument("--seed", type=int, default=42)

    # Scheduler options
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "onecycle", "lambda"],
                        help="LR scheduler to use")
    parser.add_argument("--pct-start", type=float, default=0.15, help="OneCycleLR pct_start")
    parser.add_argument("--div-factor", type=float, default=25.0, help="OneCycleLR div_factor")
    parser.add_argument("--warmup-epochs", type=int, default=50, help="LambdaLR warmup epochs")

    # Paths and logging
    parser.add_argument("--checkpoints-dir", type=str, default=checkpoint_path_default)
    parser.add_argument("--wandb-project", action="store_true", help="If true, enable Weights & Biases logging to this project")

    # Resume options
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--resume-name", type=str, default=None, help="Run name (folder) to resume")
    parser.add_argument("--resume-epoch", type=int, default=None, help="Epoch checkpoint to resume from")

    # Correlated Dots Args
    parser.add_argument("--n-frames", type=parse_tuple, default=40)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--correlation", type=parse_tuple, default=0.5)
    parser.add_argument("--max-speed", type=parse_tuple, default=1)
    parser.add_argument("--samples-per-epoch", type=int, default=None)

    args = parser.parse_args()

    # Build model_config from preset or file
    if args.model_config_file is not None:
        with open(args.model_config_file, "r") as f:
            model_config = json.load(f)
    else:
        model_config = copy.deepcopy(BASE_MODEL_CONFIGS[args.model_type])

    if args.dataset == "mazes":
        data_root = maze_data_path
    elif args.dataset == "cabc":
        data_root = cabc_data_path
    else:
        data_root = None

    # Build scheduler config
    scheduler_cfg = None
    if args.scheduler == "onecycle":
        scheduler_cfg = {"name": "OneCycleLR", "kwargs": {"pct_start": args.pct_start, "anneal_strategy": 'cos', "div_factor": args.div_factor}}
    elif args.scheduler == "lambda":
        scheduler_cfg = {"name": "LambdaLR", "warmup_epochs": args.warmup_epochs, "kwargs": {}}

    try:
        if args.resume:
            if args.resume_name is None or args.resume_epoch is None:
                raise ValueError("--resume requires --resume-name and --resume-epoch")
            run_from_checkpoint(
                wandb_name=args.resume_name,
                epoch=args.resume_epoch,
                new_params={
                    "lr": args.lr,
                    "num_steps": args.num_steps,
                    "max_epochs": args.max_epochs,
                    "batch_size": args.batch_size,
                    "max_gradient": args.max_gradient,
                    "scheduler": scheduler_cfg,
                    "init_weights": args.init_weights,
                    "seed": args.seed,
                },
                checkpoints_dir=args.checkpoints_dir,
                data_root=data_root,
                wandb_project=args.wandb_project,
            )
        else:
            overrides: dict = {}
            if args.fc_dim is not None:
                overrides["fc_dim"] = args.fc_dim
            if args.out_channels is not None:
                overrides["out_channels"] = args.out_channels
            if args.num_neuron_subtypes is not None:
                try:
                    overrides["num_neuron_subtypes"] = int(args.num_neuron_subtypes)
                except ValueError:
                    overrides["num_neuron_subtypes"] = [int(x) for x in args.num_neuron_subtypes.split(",")]
            if args.neuron_type_nonlinearity is not None:
                overrides["neuron_type_nonlinearity"] = args.neuron_type_nonlinearity
            if args.inter_neuron_type_nonlinearity is not None:
                overrides["inter_neuron_type_nonlinearity"] = args.inter_neuron_type_nonlinearity
            if args.inter_neuron_type_spatial_extents is not None:
                if args.inter_neuron_type_spatial_extents in ("center_excitation", "center_inhibition"):
                    overrides["inter_neuron_type_spatial_extents"] = args.inter_neuron_type_spatial_extents
                else:
                    h, w = [int(x) for x in args.inter_neuron_type_spatial_extents.split(",")]
                    overrides["inter_neuron_type_spatial_extents"] = (h, w)

            if "cnn" in args.model_type:
                effective_model_config = model_config
                if args.fc_dim is not None:
                    effective_model_config["fc_dim"] = args.fc_dim
                if args.conv1_out is not None:
                    effective_model_config["conv1_out"] = args.conv1_out
                if args.conv2_out is not None:
                    effective_model_config["conv2_out"] = args.conv2_out
                if args.conv3_out is not None:
                    effective_model_config["conv3_out"] = args.conv3_out
            else:
                effective_model_config = apply_model_overrides(args.model_type, model_config, overrides) if overrides else model_config

            # Adjust input config based on dataset
            if args.dataset == "mazes":
                if "rnn_kwargs" in effective_model_config:
                    areas = effective_model_config["rnn_kwargs"]["area_kwargs"]
                    areas[0]["in_channels"] = 4
                    areas[0]["in_size"] = (48, 48)
                    for i in range(1, len(areas)):
                        areas[i]["in_channels"] = areas[i-1]["out_channels"]
                        areas[i]["in_size"] = areas[i-1]["in_size"]
                else:
                    effective_model_config["in_channels"] = 4
                effective_model_config["num_classes"] = 2

            elif args.dataset == "correlated_dots":
                if "rnn_kwargs" in effective_model_config:
                    areas = effective_model_config["rnn_kwargs"]["area_kwargs"]
                    areas[0]["in_channels"] = 1
                    areas[0]["in_size"] = (args.resolution, args.resolution)
                    for i in range(1, len(areas)):
                        areas[i]["in_channels"] = areas[i-1]["out_channels"]
                        areas[i]["in_size"] = areas[i-1]["in_size"]
                else:
                    if not isinstance(args.n_frames, int):
                        raise NotImplementedError("CNN Not Yet Compatible With N_Frames Range")
                    effective_model_config["in_channels"] = args.n_frames
                effective_model_config["num_classes"] = 8

            elif args.dataset == "cabc":
                if "rnn_kwargs" in effective_model_config:
                    areas = effective_model_config["rnn_kwargs"]["area_kwargs"]
                    areas[0]["in_channels"] = 1
                    areas[0]["in_size"] = (args.resolution, args.resolution) if args.resolution is not None else (350, 350)
                    for i in range(1, len(areas)):
                        areas[i]["in_channels"] = areas[i-1]["out_channels"]
                        areas[i]["in_size"] = areas[i-1]["in_size"]
                else:
                    effective_model_config["in_channels"] = 1
                effective_model_config["num_classes"] = 2

            dots_kwargs: dict = {}
            n_frames_range = None
            if args.dataset == "correlated_dots":
                if isinstance(args.n_frames, tuple):
                    dots_kwargs["n_frames"] = args.n_frames[1]
                    n_frames_range = args.n_frames
                else:
                    dots_kwargs["n_frames"] = args.n_frames
                dots_kwargs["correlation"] = args.correlation
                dots_kwargs["max_speed"] = args.max_speed
                if args.samples_per_epoch is not None:
                    dots_kwargs["samples_per_epoch"] = args.samples_per_epoch
                args.num_steps = None

            run_experiment(
                model_type=args.model_type,
                model_config=effective_model_config,
                dataset=args.dataset,
                lr=args.lr,
                num_steps=args.num_steps,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                max_gradient=args.max_gradient,
                scheduler_config=scheduler_cfg,
                init_weights=args.init_weights,
                neuron_type_nonlinearity=args.neuron_type_nonlinearity if args.neuron_type_nonlinearity is not None else "relu",
                data_root=data_root,
                checkpoints_dir=args.checkpoints_dir,
                wandb_project=args.wandb_project,
                seed=args.seed,
                n_frames_range=n_frames_range,
                dots_kwargs=dots_kwargs,
                samples_per_epoch=args.samples_per_epoch if args.dataset != "correlated_dots" else None,
                val_samples_per_epoch=None,
                resolution=args.resolution,
            )
    finally:
        if dist.is_initialized():
            cleanup_distributed()