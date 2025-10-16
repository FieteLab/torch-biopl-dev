import argparse
import copy
import gc
import json
import os
import pickle
from datetime import datetime
from functools import partial
import random

import GPUtil  # type: ignore
import numpy as np
import psutil
import torch
from torch import nn
from tqdm import tqdm
import wandb  # type: ignore

from bioplnn.models import SpatiallyEmbeddedClassifier
from bioplnn.utils import (
    initialize_dataloader,
    initialize_scheduler,
    manual_seed,
)

# Defaults (override with CLI)
maze_data_path = "./data/mazes"
cabc_data_path = "./data/easy"

checkpoint_path = "./train/checkpoints"

# Torch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

BATCH_SIZE = 32

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
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
        self.block1 = DSConv(16, 32, stride=2)   # /2
        self.block2 = DSConv(32, 64, stride=2)   # /2
        self.block3 = DSConv(64, 64, stride=1)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)       # -> [B, 64, 1, 1]
        return self.head(x)    # -> [B, num_classes]
        
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.gap   = nn.AdaptiveAvgPool2d(1)     # <- NEW
        self.fc1   = nn.Linear(128, 512)          # <- 64 channels only
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.gap(x)                          # [B, 64, 1, 1]
        x = torch.flatten(x, 1)                  # [B, 64]
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

def load_data(dataset, **args):
    train_loader, test_loader = initialize_dataloader(
        seed=42, dataset=dataset, **args
    )

    return train_loader, test_loader

# Define evaluation function
def evaluate(model, model_type, data_loader, criterion, device, num_steps):
    model.eval()
    total_loss = 0
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
                labels = labels.squeeze(1)       # [N, 1] -> [N]
            elif labels.ndim != 1:
                raise ValueError(f"Expected labels shape [N] or [N,1], got {tuple(labels.shape)}")

            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def get_cpu_memory():
    """Get current CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    if torch.cuda.is_available() and GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
    return 0

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

def get_class_distribution(data_loader):
    """Calculate class distribution in the dataset."""
    class_counts = torch.zeros(2)  # Assuming binary classification
    total_samples = 0
    
    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1
        total_samples += len(labels)
    
    return class_counts / total_samples

def get_gradient_norm(model):
    """Calculate the L2 norm of gradients for all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def analyze_logits(logits, labels, num_samples=5, detailed=False):
    """Analyze logits distribution and predictions."""
    probs = torch.softmax(logits, dim=1)
    _, predicted = torch.max(logits, 1)
    
    # Get random samples
    indices = torch.randperm(len(logits))[:num_samples]
    
    print("\nLogits Analysis:")
    print("=" * 50)
    for idx in indices:
        print(f"Sample {idx}:")
        print(f"True label: {labels[idx].item()}")
        print(f"Predicted: {predicted[idx].item()}")
        print(f"Logits: {logits[idx].detach().cpu().numpy()}")
        print(f"Probabilities: {probs[idx].detach().cpu().numpy()}")
        print("-" * 30)
    
    if detailed:
        # Additional statistics
        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        print("\nDetailed Statistics:")
        print(f"Logits mean: {logits_np.mean():.4f}, std: {logits_np.std():.4f}")
        print(f"Logits min: {logits_np.min():.4f}, max: {logits_np.max():.4f}")
        print(f"Probabilities mean: {probs_np.mean():.4f}, std: {probs_np.std():.4f}")
        print(f"Probabilities min: {probs_np.min():.4f}, max: {probs_np.max():.4f}")
        
        # Distribution of predictions
        unique, counts = np.unique(predicted.cpu().numpy(), return_counts=True)
        pred_dist = dict(zip(unique, counts))
        print(f"Prediction distribution: {pred_dist}")

def train(model, model_type, train_loader, test_loader, criterion, optimizer, scheduler, num_steps, max_epochs, max_gradient, train_log_frequency, run_name, run, n_frames_range=None, start_epoch=0, checkpoint_dir=checkpoint_path):
    # Define the training loop
    model.train()

    # # Print initial diagnostics
    # print("\nInitial Dataset Analysis:")
    # print("=" * 50)
    # train_dist = get_class_distribution(train_loader)
    # test_dist = get_class_distribution(test_loader)
    # print(f"Training set class distribution: {train_dist.numpy()}")
    # print(f"Test set class distribution: {test_dist.numpy()}")
    # print("=" * 50)

    gpu_memory = get_gpu_memory()
    cpu_memory = get_cpu_memory()
    gpu_util = get_gpu_utilization()
    print(f"GPU Memory: {gpu_memory:.1f}MB | CPU Memory: {cpu_memory:.1f}MB | GPU Utilization: {gpu_util:.1f}%")

    val_accs = []
    patience = 200

    save_path = f"{checkpoint_dir}/{run_name}"
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(start_epoch, max_epochs):
        if run is not None:
            run.config.update({"n_epochs": epoch}, allow_val_change=True)
        running_loss, running_correct, running_total = 0, 0, 0
        for i, (x, labels) in enumerate(tqdm(train_loader)):
            try:
                x = x.to(device)
            except AttributeError:
                x = [t.to(device) for t in x]
            labels = labels.to(device)

            if n_frames_range is not None and num_steps is None:
                n_frames = random.randint(n_frames_range[0], n_frames_range[1])
                x = x[:, :n_frames]
            
            print(x.shape)
            
            # Forward pass
            if "cnn" in model_type:
                if len(x.shape) == 5:
                    x = x[:, :, 0, :, :]
                logits = model(x)
            else:
                logits = model(x, num_steps=num_steps, loss_all_timesteps=False)

            if labels.ndim == 2 and labels.size(1) == 1:
                labels = labels.squeeze(1)       # [N, 1] -> [N]
            elif labels.ndim != 1:
                raise ValueError(f"Expected labels shape [N] or [N,1], got {tuple(labels.shape)}")

            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient analysis
            grad_norm = get_gradient_norm(model)

            # Gradient clipping
            if max_gradient is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Calculate batch metrics
            predicted = torch.argmax(logits, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            batch_loss = loss.item()

            # Update running metrics
            running_total += batch_total
            running_correct += batch_correct
            running_loss += batch_loss * batch_total  # Weight loss by batch size

            # Log batch metrics if needed
            if i % train_log_frequency == 0:
                batch_acc = batch_correct / batch_total
                print(
                    f"Batch {i} | "
                    + f"Loss: {batch_loss:.4f} | "
                    + f"Acc: {batch_acc:.2%} | "
                    + f"Grad Norm: {grad_norm:.4f} | "
                    + f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                
                # Analyze logits every train_log_frequency batches
                # Use detailed analysis for first few batches
                # analyze_logits(logits, labels, detailed=(i < 3))

                if run is not None:
                    wandb.log({
                        "train_loss": running_loss / running_total,
                        "train_acc": running_correct / running_total,
                        "gradient_norm": grad_norm,
                        "step": epoch * len(train_loader) + i,
                        "lr": optimizer.param_groups[0]["lr"]
                    })
            
        # Calculate epoch metrics
        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        
        print(
            f"Training | Epoch: {epoch} | "
            + f"Loss: {epoch_loss:.4f} | "
            + f"Acc: {epoch_acc:.2%} | "
            + f"GPU Memory: {get_gpu_memory():.1f}MB | "
            + f"CPU Memory: {get_cpu_memory():.1f}MB"
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, model_type, test_loader, criterion, device, num_steps)
        model.train()
        
        if run is not None:
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
        
        if epoch % 10 == 0:
            # Save full checkpoint with model, optimizer, and scheduler state
            checkpoint = {
                'epoch': epoch,
                'model_state': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict() if scheduler else None,
            }
            torch.save(checkpoint, f"{save_path}/{epoch}.pth")
        
        val_accs.append(val_acc)
        
        if len(val_accs) > patience+10:
            best_val_acc = max(val_accs[:-patience])
            if all(acc <= best_val_acc for acc in val_accs[-patience:]):
                print("Validation accuracy not improving, stopping training")
                print(f"Best validation accuracy: {best_val_acc:.2%}")
                print(val_accs[-patience:])
                # Save full checkpoint with model, optimizer, and scheduler state
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'sched_state': scheduler.state_dict() if scheduler else None,
                }
                torch.save(checkpoint, f"{save_path}/{epoch}.pth")
                break

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
        **kwargs)
    return scheduler


def run_experiment(
    model_type,
    model_config,
    dataset="mazes",
    lr=0.001,
    num_steps=60,
    max_epochs=10,
    batch_size=128,
    max_gradient=None,
    scheduler_config=None,
    init_weights="kaiming",
    neuron_type_nonlinearity="relu",
    data_root=maze_data_path,
    checkpoints_dir=checkpoint_path,
    wandb_project=False,
    seed=42,
    n_frames_range=None,
    dots_kwargs = {}
):
    """Run a single experiment with the given model type, config and hyperparameters."""
    manual_seed(seed)

    # Create the model        
    if model_type == "cnn":
        model = SimpleCNN(**model_config).to(device)
    elif model_type == "tiny_cnn":
        model = TinyCNN(**model_config).to(device)
    else:
        model = SpatiallyEmbeddedClassifier(**model_config).to(device)
    
    # Wrap for multi‑GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    if dataset == "mazes" or dataset == "cabc":
        train_loader, test_loader = load_data(dataset, batch_size=batch_size, root=data_root)
    elif dataset == "correlated_dots":
        train_loader, test_loader = load_data(dataset, batch_size=batch_size, **dots_kwargs)

    train_log_frequency = max(1, len(train_loader) // 10)
        
    full_config = {
        "model_type": model_type,
        "model_config": model_config,
        "lr": lr,
        "num_steps": num_steps,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "max_gradient": max_gradient,
        "scheduler": scheduler_config,
        "init_weights": init_weights,
        "seed": seed,
        "dots_kwargs": dots_kwargs
    }

    if init_weights == "kaiming":
        model.apply(partial(init_weights_kaiming, nonlinearity=neuron_type_nonlinearity))
    elif init_weights == "zero":
        model.apply(init_weights_zero)
    
    scheduler = get_scheduler(scheduler_config, optimizer=optimizer, train_loader=train_loader, max_epochs=max_epochs, lr=lr) if scheduler_config else None

    # Logging
    run = None
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    if wandb_project:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb-project was True")
        run = wandb.init(project=dataset, config=full_config)
        run_name = run.name

    # Save run config
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(os.path.join(checkpoints_dir, f"{run_name}.pkl"), "wb") as f:
        pickle.dump(full_config, f)
    
    # Train
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
        n_frames_range=n_frames_range,
        start_epoch=0,
        checkpoint_dir=checkpoints_dir,
    )
    
    # Evaluate final performance
    final_val_loss, final_val_acc = evaluate(model, model_type, test_loader, criterion, device, num_steps)
    if run is not None:
        wandb.log({
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        })
        wandb.finish()
    
    return final_val_loss, final_val_acc

BASE_MODEL_CONFIGS = {
            "1e1i1a": {
                "rnn_kwargs": {
                    "num_areas": 1,
                    "area_kwargs": [
                        {
                            "num_neuron_types": 2,
                            "num_neuron_subtypes": np.array([32, 8]),
                            "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 1, 0], [1, 1, 1], [1, 0, 0]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 4,
                            "out_channels": 32,
                            "inter_neuron_type_nonlinearity": np.array([[None, None, None], [None, None, None], [None, None, None]]),
                            "inter_neuron_type_spatial_extents": (5,5),
                        },
                    ],
                },
                "num_classes": 2,
                "fc_dim": 512,
                "dropout": 0.2,
            },
            "1e1ii1ef1a": {
                "rnn_kwargs": {
                    "num_areas": 2,
                    "area_kwargs": [
                        {
                            "num_neuron_types": 2,
                            "num_neuron_subtypes": np.array([8, 4]),
                            "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1]]
                            ),
                            "in_size": [48, 48],
                            "feedback_channels": 8,
                            "in_channels": 4,
                            "out_channels": 8,
                            "inter_neuron_type_nonlinearity": np.array([[None, None, None], [None, None, None], [None, None, None], [None, None, None]]),
                            "inter_neuron_type_spatial_extents": (5,5),
                        },
                        {
                            "num_neuron_types": 1,
                            "num_neuron_subtypes": np.array([8]),
                            "neuron_type_class": np.array(["excitatory"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 0], [1, 1]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 8,
                            "out_channels": 8,
                            "inter_neuron_type_nonlinearity": np.array([[None, None], [None, None]]),
                            "inter_neuron_type_spatial_extents": (7,7),
                        },
                    ],
                    "inter_area_feedback_connectivity": np.array([[0, 0],[1, 0]])
                },
                "num_classes": 2,
                "fc_dim": 512,
                "dropout": 0.2,
            },
            "1e1ii1a": {
                "rnn_kwargs": {
                    "num_areas": 1,
                    "area_kwargs": [
                        {
                            "in_class": "excitatory"
                            "neuron_type_nonlinearity": "ReLU",
                            "out_nonlinearity": "ReLU",
                            "num_neuron_types": 2,
                            "num_neuron_subtypes": np.array([32, 8]),
                            "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 1, 0], [1, 1, 1], [1, 1, 0]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 4,
                            "out_channels": 32,
                            "inter_neuron_type_nonlinearity": np.array([[None, None, None], [None, None, None], [None, None, None]]),
                            "inter_neuron_type_spatial_extents": (5,5),
                        },
                    ],
                },
                "num_classes": 2,
                "fc_dim": 512,
                "dropout": 0.2,
            },
            "hybrid": {
                "rnn_kwargs": {
                    "num_areas": 1,
                    "area_kwargs": [
                        {
                            "num_neuron_types": 1,
                            "num_neuron_subtypes": np.array([64]),
                            "neuron_type_class": np.array(["hybrid"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 0], [1, 1]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 4,
                            "out_channels": 16,
                            "inter_neuron_type_nonlinearity": np.array([["relu", "relu"], ["relu", "relu"]]),
                            "inter_neuron_type_spatial_extents": (3, 3)
                        },
                    ],
                },
                "num_classes": 2,
                "fc_dim": 64,
                "dropout": 0.1,
            },
            "1h1a": {
                "rnn_kwargs": {
                    "num_areas": 1,
                    "area_kwargs": [
                        {
                            "num_neuron_types": 1,
                            "num_neuron_subtypes": np.array([16]),
                            "neuron_type_class": np.array(["hybrid"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 0], [1, 1]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 4,
                            "out_channels": 32,
                            "inter_neuron_type_nonlinearity": np.array([["sigmoid", "sigmoid"], ["sigmoid", "sigmoid"]]),
                            "inter_neuron_type_spatial_extents": (3, 3)
                        },
                    ],
                },
                "num_classes": 2,
                "fc_dim": 64,
                "dropout": 0.2,
            },
            "hybrid_ei": {
                "rnn_kwargs": {
                    "num_areas": 1,
                    "area_kwargs": [
                        {
                            "num_neuron_types": 2,
                            "num_neuron_subtypes": np.array([64, 64]),
                            "neuron_type_class": np.array(["hybrid", "hybrid"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 1, 0], [1, 1, 1], [1, 1, 0]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 4,
                            "out_channels": 16,
                            "inter_neuron_type_nonlinearity": np.array([["relu", "relu", "relu"], ["relu", "relu", "relu"], ["relu", "relu", "relu"]]),
                            "inter_neuron_type_spatial_extents": (3, 3)
                        },
                    ],
                },
                "num_classes": 2,
                "fc_dim": 64,
                "dropout": 0.1,
            },
            "bio_cnn_hybrid": {
                "rnn_kwargs": {
                    "num_areas": 2,
                    "area_kwargs": [
                        {
                            "num_neuron_types": 1,
                            "num_neuron_subtypes": np.array([64]),
                            "neuron_type_class": np.array(["hybrid"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 0], [0, 1]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 4,
                            "out_channels": 128,
                            "inter_neuron_type_nonlinearity": np.array([["relu", "relu"], ["relu", "relu"]]),
                            "inter_neuron_type_spatial_extents": (5, 5)
                        },
                        {
                            "num_neuron_types": 1,
                            "num_neuron_subtypes": np.array([256]),
                            "neuron_type_class": np.array(["hybrid"]),
                            "inter_neuron_type_connectivity": np.array(
                                [[1, 0], [0, 1]]
                            ),
                            "in_size": [48, 48],
                            "in_channels": 128,
                            "out_channels": 256,
                            "inter_neuron_type_nonlinearity": np.array([["relu", "relu"], ["relu", "relu"]]),
                            "inter_neuron_type_spatial_extents": (5, 5)
                        },
                    ],
                },
                "num_classes": 2,
                "fc_dim": 512,
                "dropout": 0.1,
            },
            "cnn": {
                "in_channels": 4,
                "num_classes": 2,
                "dropout": 0.35,
            },
            "tiny_cnn": {
                "in_channels": 4,
                "num_classes": 2,
                "dropout": 0.1,
            }
        }

def apply_model_overrides(model_type: str, model_config: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(model_config)
    if "cnn" in model_type:
        # Currently only common overrides apply to non-CNN models
        return cfg

    # fc_dim override
    fc_dim = overrides.get("fc_dim")
    if fc_dim is not None:
        cfg["fc_dim"] = fc_dim

    area0 = cfg.get("rnn_kwargs", {}).get("area_kwargs", [{}])[0]
    if not area0:
        return cfg

    # out_channels override
    out_channels = overrides.get("out_channels")
    if out_channels is not None:
        area0["out_channels"] = out_channels

    # num_neuron_subtypes override (int or list)
    nnst = overrides.get("num_neuron_subtypes")
    if nnst is not None:
        num_types = area0.get("num_neuron_types", None)
        if isinstance(nnst, int):
            if num_types is None:
                raise ValueError("num_neuron_types missing in base config while applying num_neuron_subtypes=int")
            area0["num_neuron_subtypes"] = np.ones(int(num_types), dtype=int) * int(nnst)
        elif isinstance(nnst, (list, tuple, np.ndarray)):
            area0["num_neuron_subtypes"] = np.array(list(map(int, nnst)))

    # inter_neuron_type_connectivity - not exposed via CLI in example, keep for completeness
    if "inter_neuron_type_connectivity" in overrides and overrides["inter_neuron_type_connectivity"] is not None:
        area0["inter_neuron_type_connectivity"] = np.array(overrides["inter_neuron_type_connectivity"])

    # inter_neuron_type_spatial_extents
    spatial_extents = overrides.get("inter_neuron_type_spatial_extents")
    if spatial_extents is not None:
        if isinstance(spatial_extents, str):
            if spatial_extents == "center_excitation":
                area0["inter_neuron_type_spatial_extents"] = np.array([[(5,5), (5,5), (5,5)], [(3,3), (3,3), (3,3)], [(7,7), (7,7), (7,7)]])
            elif spatial_extents == "center_inhibition":
                area0["inter_neuron_type_spatial_extents"] = np.array([[(5,5), (5,5), (5,5)], [(5,5), (5,5), (5,5)], [(3,3), (5,5), (5,5)]])
    else:
            # expects tuple like (h, w)
            area0["inter_neuron_type_spatial_extents"] = spatial_extents

    # inter_neuron_type_nonlinearity (single token to fill matrix)
    inter_nt_nl = overrides.get("inter_neuron_type_nonlinearity")
    if inter_nt_nl is not None:
        connectivity = area0.get("inter_neuron_type_connectivity")
        if connectivity is None:
            raise ValueError("inter_neuron_type_connectivity missing; cannot derive matrix shape for nonlinearity")
        rows, cols = connectivity.shape
        area0["inter_neuron_type_nonlinearity"] = np.full((rows, cols), inter_nt_nl)

    # neuron_type_nonlinearity (per-type nonlinearity)
    nt_nl = overrides.get("neuron_type_nonlinearity")
    if nt_nl is not None:
        area0["neuron_type_nonlinearity"] = nt_nl

    # Write back area0
    cfg["rnn_kwargs"]["area_kwargs"][0] = area0
    return cfg

def run_from_checkpoint(wandb_name, epoch, new_params={}, checkpoints_dir=checkpoint_path, data_root=maze_data_path, wandb_project=False):
    full_config = pickle.load(open(os.path.join(checkpoints_dir, f"{wandb_name}.pkl"), "rb"))
    for param, value in new_params.items():
        full_config[param] = value

    model_config = full_config["model_config"]
    model_type = full_config.get("model_type", "1e1i1a")  # Default to bioplnn if not specified
    
    # Create the correct model type
    if model_type == "cnn":
        model = SimpleCNN(**model_config).to(device)
    elif model_type == "tiny_cnn":
        model = TinyCNN(**model_config).to(device)
    else:
        model = SpatiallyEmbeddedClassifier(**model_config).to(device)
    
    # Wrap for multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # # Load checkpoint
    # checkpoint = torch.load(checkpoint_path + f"{wandb_name}/{epoch}.pth")
    
    # # Handle both old format (just state_dict) and new format (full checkpoint)
    # if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    #     # New format - full checkpoint
    #     model.load_state_dict(checkpoint['model_state'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Resuming from epoch {checkpoint['epoch']}, will start at epoch {start_epoch}")
    # else:
    #     # Old format - just state dict
    #     model.load_state_dict(checkpoint)
    #     start_epoch = epoch + 1
    #     print(f"Loaded old format checkpoint from epoch {epoch}, will start at epoch {start_epoch}")
    
    # --- after you instantiate model (and wrap for multi-GPU if you do that) ---
    ckpt = torch.load(os.path.join(checkpoints_dir, wandb_name, f"{epoch}.pth"),
                    map_location=device)  # keep weights_only=False for full ckpt dict

    state = ckpt["model_state"]
    start_epoch = ckpt['epoch'] + 1

    # Load weights robustly across DP / non-DP
    if isinstance(model, nn.DataParallel):
        # checkpoint keys have no 'module.' → load into the underlying module
        try:
            model.module.load_state_dict(state)
        except RuntimeError:
            # If you ever saved with 'module.' keys, strip them:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.module.load_state_dict(state)
    else:
        # Non-DP model: if the checkpoint has 'module.' keys, strip them
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)

    del ckpt
    # del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    
    # Extract hyperparameters
    lr = full_config.get("lr", 0.001)
    num_steps = full_config.get("num_steps", 60)
    max_epochs = full_config.get("max_epochs", 10)
    batch_size = full_config.get("batch_size", 128)
    max_gradient = full_config.get("max_gradient", None)
    
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Load optimizer state if available in checkpoint
    checkpoint = torch.load(os.path.join(checkpoints_dir, f"{wandb_name}/{epoch}.pth"))
    if isinstance(checkpoint, dict) and 'opt_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['opt_state'])
        print("Loaded optimizer state from checkpoint")
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(batch_size=batch_size, root=data_root)

    train_log_frequency = max(1, len(train_loader) // 10)  # How often to log training metrics
    
    # Create scheduler
    scheduler = get_scheduler(full_config.get("scheduler", None), optimizer=optimizer, train_loader=train_loader, max_epochs=max_epochs, lr=lr)
    
    # Load scheduler state if available in checkpoint
    if isinstance(checkpoint, dict) and 'sched_state' in checkpoint and checkpoint['sched_state'] and scheduler:
        scheduler.load_state_dict(checkpoint['sched_state'])
        print("Loaded scheduler state from checkpoint")
    
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    
    run = None
    if wandb_projec:
        run = wandb.init(
            project=dataset,
            config=full_config,
        )

    wandb_name = run.name if run is not None else wandb_name
    #save model config
    with open(os.path.join(checkpoints_dir, f"{wandb_name}.pkl"), "wb") as f:
        pickle.dump(full_config, f)
    
    # Call train with start_epoch parameter
    train(model, model_type, train_loader, test_loader, criterion, optimizer, scheduler, num_steps, max_epochs, max_gradient, train_log_frequency, wandb_name, run, start_epoch, checkpoint_dir=checkpoints_dir)
    
    # Evaluate final performance
    final_val_loss, final_val_acc = evaluate(model, model_type, test_loader, criterion, device, num_steps)
    if run is not None:
        wandb.log({
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        })
        wandb.finish()
    
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
        # Single number
        try:
            val = float(s)
            # If it's an integer-like float, return as int
            if val.is_integer():
                return int(val)
            return val
        except ValueError:
            raise argparse.ArgumentTypeError("must be a number or 'low,high'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single model on a dataset")

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
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-gradient", type=float, default=None)
    parser.add_argument("--init-weights", type=str, default="kaiming", choices=["kaiming", "zero", "none"])
    parser.add_argument("--fc-dim", type=int, default=None, help="Override fully connected layer width for non-CNN models")
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
    parser.add_argument("--data-root", type=str, default=maze_data_path)
    parser.add_argument("--checkpoints-dir", type=str, default=checkpoint_path)
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
    parser.add_argument("--samples-per-epoch", type=int, default=10000)

    args = parser.parse_args()

    # Build model_config from preset or file
    if args.model_config_file is not None:
        with open(args.model_config_file, "r") as f:
            model_config = json.load(f)
    else:
        model_config = copy.deepcopy(BASE_MODEL_CONFIGS[args.model_type])

    # Build scheduler config
    scheduler_cfg = None
    if args.scheduler == "onecycle":
        scheduler_cfg = {"name": "OneCycleLR", "kwargs": {"pct_start": args.pct_start, "anneal_strategy": 'cos', "div_factor": args.div_factor}}
    elif args.scheduler == "lambda":
        scheduler_cfg = {"name": "LambdaLR", "warmup_epochs": args.warmup_epochs, "kwargs": {}}

    # Resume if requested
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
            data_root=args.data_root,
            wandb_project=args.wandb_project,
        )
    else:
        # Prepare model overrides from CLI for non-CNN models
        overrides = {}
        if args.fc_dim is not None: overrides["fc_dim"] = args.fc_dim
        if args.out_channels is not None: overrides["out_channels"] = args.out_channels
        if args.num_neuron_subtypes is not None:
            try:
                # Try to parse as int
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

        # Apply overrides for non-CNN models
        effective_model_config = apply_model_overrides(args.model_type, model_config, overrides) if overrides and "cnn" not in args.model_type else model_config

        # Adjust input config based on dataset
        if args.dataset == "mazes":
            if "rnn_kwargs" in effective_model_config:
                for area in effective_model_config["rnn_kwargs"]["area_kwargs"]:
                    area["in_channels"] = 4
                    area["in_size"] = (48, 48)
            else:
                effective_model_config["in_channels"] = 4  # for CNN presets
            effective_model_config["num_classes"] = 2

        elif args.dataset == "correlated_dots":
            if "rnn_kwargs" in effective_model_config:
                for area in effective_model_config["rnn_kwargs"]["area_kwargs"]:
                    area["in_channels"] = 1
                    area["in_size"] = (args.resolution, args.resolution)
            else:
                if not isinstance(args.n_frames, int):
                    raise NotImplementedError("CNN Not Yet Compatible With N_Frames Range")
                effective_model_config["in_channels"] = args.n_frames
            effective_model_config["num_classes"] = 8

        elif args.dataset == "cabc":
            if "rnn_kwargs" in effective_model_config:
                for area in effective_model_config["rnn_kwargs"]["area_kwargs"]:
                    area["in_channels"] = 1
                    area["in_size"] = (350, 350)
            effective_model_config["num_classes"] = 2
    
        dots_kwargs = {}
        n_frames_range = None
        if args.dataset == "correlated_dots":
            if isinstance(args.n_frames, tuple):
                dots_kwargs["n_frames"] = args.n_frames[1]
                n_frames_range = args.n_frames
            else:
                dots_kwargs["n_frames"] = args.n_frames
            dots_kwargs["resolution"] = None if args.resolution is None else (args.resolution, args.resolution)
            dots_kwargs["correlation"] = args.correlation
            dots_kwargs["max_speed"] = args.max_speed
            dots_kwargs["samples_per_epoch"] = args.samples_per_epoch
            args.num_steps = None

        # Train fresh run
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
            data_root=args.data_root,
            checkpoints_dir=args.checkpoints_dir,
            wandb_project=args.wandb_project,
            seed=args.seed,
            n_frames_range=n_frames_range,
            dots_kwargs=dots_kwargs
        )