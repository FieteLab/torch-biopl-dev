import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pickle
import pandas as pd
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output
import wandb
import psutil
import GPUtil
from functools import partial
import gc
import copy
import datetime
wandb.login(key="a338f755915cccd861b14f29bf68601d8e1ec2c9")

from bioplnn.models import SpatiallyEmbeddedClassifier, SpatiallyEmbeddedAreaConfig, SpatiallyEmbeddedRNN
from bioplnn.datasets import Mazes

from bioplnn.utils import (
    initialize_criterion,
    initialize_dataloader,
    initialize_model,
    initialize_optimizer,
    initialize_scheduler,
    manual_seed,
    manual_seed_deterministic,
    pass_fn,
)

maze_data_path = "/om2/user/jackking/torch-bioplnn-dev/data/mazes"
checkpoint_path = "/om2/user/jackking/torch-bioplnn-dev/train/checkpoints/"
model_csv_path = "/om2/user/jackking/torch-bioplnn-dev/train/models.csv"
model_configs_path = "/om2/user/jackking/torch-bioplnn-dev/train/model_configs.json"

# Torch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

BATCH_SIZE = 32

def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set NCCL environment variables to avoid GPU conflicts
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback interface
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # Ensure we have the right number of GPUs
    assert torch.cuda.device_count() >= world_size, f"Not enough GPUs: {torch.cuda.device_count()} < {world_size}"
    
    # Set device for this process BEFORE initializing process group
    torch.cuda.set_device(rank)
    
    # Print GPU assignment for debugging
    print(f"Rank {rank}: Using GPU {rank} (CUDA device {torch.cuda.current_device()})")
    
    # Initialize the process group with timeout
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30))
    
    # Synchronize all processes
    dist.barrier()

def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, dropout=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, num_steps=None):  # num_steps parameter for compatibility
        # x shape: [batch_size, channels, height, width]
        x = self.pool(self.relu(self.conv1(x)))  # -> [batch_size, 32, 24, 24]
        x = self.pool(self.relu(self.conv2(x)))  # -> [batch_size, 64, 12, 12]
        x = self.pool(self.relu(self.conv3(x)))  # -> [batch_size, 128, 6, 6]
        x = x.view(-1, 512 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_data(batch_size=BATCH_SIZE, num_samples=None, rank=0, world_size=1, distributed=False):
    """Load data with optional distributed sampling."""
    # Get the data loaders
    train_loader, test_loader = initialize_dataloader(
        seed=42, root="./data/mazes/", batch_size=batch_size, dataset="mazes"
    )
    
    if distributed:
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_loader.dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_loader.dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
        
        # Create new data loaders with distributed samplers
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_loader.dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    return train_loader, test_loader

# Define evaluation function
def evaluate(model, data_loader, criterion, device, num_steps, rank=0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, labels in data_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            logits = model(x, num_steps=num_steps)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    # Synchronize metrics across processes
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor(avg_loss).to(device)
        accuracy_tensor = torch.tensor(accuracy).to(device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / dist.get_world_size()
        accuracy = accuracy_tensor.item() / dist.get_world_size()
    
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
    if torch.cuda.is_available():
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

def train(model, train_loader, test_loader, criterion, optimizer, num_steps, max_epochs, train_log_frequency, wandb_name, run, rank=0, world_size=1, distributed=False):
    # Define the training loop
    model.train()

    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=5
    # )

    # Print initial diagnostics only on main process
    if rank == 0:
        print("\nInitial Dataset Analysis:")
        print("=" * 50)
        train_dist = get_class_distribution(train_loader)
        test_dist = get_class_distribution(test_loader)
        print(f"Training set class distribution: {train_dist.numpy()}")
        print(f"Test set class distribution: {test_dist.numpy()}")
        print("=" * 50)

        gpu_memory = get_gpu_memory()
        cpu_memory = get_cpu_memory()
        gpu_util = get_gpu_utilization()
        print(f"GPU Memory: {gpu_memory:.1f}MB | CPU Memory: {cpu_memory:.1f}MB | GPU Utilization: {gpu_util:.1f}%")

    val_accs = []
    patience = 30

    save_path = f"{checkpoint_path}/{wandb_name}"
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(max_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
            
        run.config.update({"n_epochs": epoch}, allow_val_change=True)
        running_loss, running_correct, running_total = 0, 0, 0
        
        # Use tqdm only on main process
        if rank == 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
            
        for i, (x, labels) in enumerate(pbar):
            try:
                x = x.to(device)
            except AttributeError:
                x = [t.to(device) for t in x]
            labels = labels.to(device)
            
            # Forward pass
            logits = model(x, num_steps=num_steps, loss_all_timesteps=False)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Gradient analysis
            grad_norm = get_gradient_norm(model)
            
            optimizer.step()

            # Calculate batch metrics
            predicted = torch.argmax(logits, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            batch_loss = loss.item()

            # Update running metrics
            running_total += batch_total
            running_correct += batch_correct
            running_loss += batch_loss * batch_total  # Weight loss by batch size

            # Log batch metrics if needed (only on main process)
            if rank == 0 and i % train_log_frequency == 0:
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

                wandb.log({
                    "train_loss": running_loss / running_total,
                    "train_acc": running_correct / running_total,
                    "gradient_norm": grad_norm,
                    "step": epoch * len(train_loader) + i,
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        
        if rank == 0:
            print(
                f"Training | Epoch: {epoch} | "
                + f"Loss: {epoch_loss:.4f} | "
                + f"Acc: {epoch_acc:.2%} | "
                + f"GPU Memory: {get_gpu_memory():.1f}MB | "
                + f"CPU Memory: {get_cpu_memory():.1f}MB"
            )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, num_steps, rank)
        model.train()
        
        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            
            if epoch % 10 == 0:
                # Save model (only on main process)
                torch.save(model.module.state_dict() if distributed else model.state_dict(), f"{save_path}/{epoch}.pth")
        
        val_accs.append(val_acc)
        
        if len(val_accs) > patience+10:
            best_val_acc = max(val_accs[:-patience])
            if all(acc <= best_val_acc for acc in val_accs[-patience:]):
                if rank == 0:
                    print("Validation accuracy not improving, stopping training")
                    print(f"Best validation accuracy: {best_val_acc:.2%}")
                    print(val_accs[-patience:])
                # break

def run_experiment(model_type, model_config, hyperparams, rank=0, world_size=1, distributed=False):
    """Run a single experiment with the given model type, config and hyperparameters."""
    # Create the model
    if not model_type == "cnn":
        model = SpatiallyEmbeddedClassifier(**model_config).to(rank)
    elif model_type == "cnn":
        model = SimpleCNN(**model_config).to(rank)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Wrap model with DDP if distributed
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Extract hyperparameters
    lr = hyperparams.get("lr", 0.001)
    num_samples = hyperparams.get("num_samples", 100)
    num_steps = hyperparams.get("num_steps", 60)
    max_epochs = hyperparams.get("max_epochs", 10)
    hyperparams["batch_size"] = 256
    batch_size = hyperparams.get("batch_size", 128)
    
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(batch_size=batch_size, num_samples=num_samples, rank=rank, world_size=world_size, distributed=distributed)

    train_log_frequency = max(1, len(train_loader) // 10)  # How often to log training metrics
        
    full_config = {
            "model_type": model_type,
            "model_config": model_config,
            **hyperparams,
        }
    
    # Only initialize wandb on main process
    if rank == 0:
        run = wandb.init(
            project="mazes",  # Specify your project
            config=full_config,
        )
    else:
        run = None

    if hyperparams.get("init_weights", "kaiming") == "kaiming":
        model.apply(partial(init_weights_kaiming, nonlinearity=hyperparams.get("neuron_type_nonlinearity", "relu")))
    elif hyperparams.get("init_weights", "kaiming") == "zero":
        model.apply(init_weights_zero)

    if rank == 0:
        wandb_name = run.name
        #save model config
        with open(checkpoint_path + f"{wandb_name}.pkl", "wb") as f:
            pickle.dump(full_config, f)
    else:
        wandb_name = None
    
    train(model, train_loader, test_loader, criterion, optimizer, num_steps, max_epochs, train_log_frequency, wandb_name, run, rank, world_size, distributed)
    
    # Evaluate final performance
    final_val_loss, final_val_acc = evaluate(model, test_loader, criterion, device, num_steps, rank)
    
    if rank == 0:
        wandb.log({
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        })

        hyperparams["wandb_name"] = wandb_name
        hyperparams["model_type"] = model_type
        #add hyperparams to model_csv
        models_csv = pd.read_csv(model_csv_path)
        models_csv = pd.concat([models_csv, pd.DataFrame([hyperparams])], ignore_index=True)
        models_csv.to_csv(model_csv_path, index=False)
        
        wandb.finish()
    
    return final_val_loss, final_val_acc

def run_sweep(sweep_config, world_size=1, rank=0):
    """Run a sweep of experiments with different hyperparameters."""
    results = []
    
    # Extract sweep parameters
    model_types = sweep_config.get("model_types", ["bioplnn"])
    # Define base model configs
    base_model_configs = {  #TODO: make these a bunch of yaml files in the model_configs folder
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
                            "inter_neuron_type_spatial_extents": (5, 5)
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
                "dropout": 0.2,
            }
        }

    # Generate hyperparameter combinations
    hyperparams_list = []
    
    # Helper function to generate all combinations of hyperparameters
    def generate_combinations(params_dict, current_combo=None, keys=None):
        if current_combo is None:
            current_combo = {}
            keys = list(params_dict.keys())
        
        if not keys:
            return [current_combo]
        
        combinations = []
        key = keys[0]
        values = params_dict[key]
        
        for value in values:
            new_combo = current_combo.copy()
            new_combo[key] = value
            combinations.extend(generate_combinations(
                params_dict, new_combo, keys[1:]
            ))
        
        return combinations
    
    # Generate all hyperparameter combinations
    if "hyperparams" in sweep_config:
        hyperparams_list = generate_combinations(sweep_config["hyperparams"])
    else:
        # Default hyperparameters if none specified
        hyperparams_list = [{
            "lr": 0.001,
            "num_samples": 100,
            "num_steps": 60,
            "max_epochs": 10,
            "batch_size": 128,
        }]
    
    # Run experiments for all combinations
    for model_type in model_types:
        for hyperparams in hyperparams_list:
            print(f"\nRunning experiment with model_type={model_type}, hyperparams={hyperparams}")
            
            # Create a copy of the base model config
            model_config = copy.deepcopy(base_model_configs[model_type])
            
            # Apply model-specific hyperparameters
            if not model_type == "cnn":
                # Handle fc_dim
                if "fc_dim" in hyperparams:
                    model_config["fc_dim"] = hyperparams["fc_dim"]
                
                if "num_neuron_types" in hyperparams:
                    model_config["rnn_kwargs"]["area_kwargs"][0]["num_neuron_types"] = hyperparams["num_neuron_types"]
                num_neuron_types = model_config["rnn_kwargs"]["area_kwargs"][0]["num_neuron_types"]
                
                # Handle num_neuron_subtypes
                if "num_neuron_subtypes" in hyperparams:
                    # Convert to numpy array with 2 elements (for excitatory and inhibitory)
                    subtypes_value = hyperparams["num_neuron_subtypes"]
                    if isinstance(subtypes_value, int):
                        # If a single integer is provided, use it for both types
                        model_config["rnn_kwargs"]["area_kwargs"][0]["num_neuron_subtypes"] = np.ones(num_neuron_types, dtype=int) * subtypes_value
                    elif isinstance(subtypes_value, (list, tuple)) and len(subtypes_value) == num_neuron_types:
                        # If a list/tuple of 2 values is provided, use them directly
                        model_config["rnn_kwargs"]["area_kwargs"][0]["num_neuron_subtypes"] = np.array(subtypes_value)

                # Handle inter_neuron_type_connectivity
                if "inter_neuron_type_connectivity" in hyperparams:
                    model_config["rnn_kwargs"]["area_kwargs"][0]["inter_neuron_type_connectivity"] = np.array(hyperparams["inter_neuron_type_connectivity"])
                
                # Handle out_channels
                if "out_channels" in hyperparams:
                    model_config["rnn_kwargs"]["area_kwargs"][0]["out_channels"] = hyperparams["out_channels"]
                
                # Handle inter_neuron_type_spatial_extents
                if "inter_neuron_type_spatial_extents" in hyperparams:
                    model_config["rnn_kwargs"]["area_kwargs"][0]["inter_neuron_type_spatial_extents"] = hyperparams["inter_neuron_type_spatial_extents"]
                
                # Handle nonlinearity
                if "inter_neuron_type_nonlinearity" in hyperparams:
                    # Get the connectivity matrix shape
                    connectivity = model_config["rnn_kwargs"]["area_kwargs"][0]["inter_neuron_type_connectivity"]
                    rows, cols = connectivity.shape
                    # Create a matrix of the same shape filled with the chosen nonlinearity
                    nonlinearity_matrix = np.full((rows, cols), hyperparams["inter_neuron_type_nonlinearity"])
                    model_config["rnn_kwargs"]["area_kwargs"][0]["inter_neuron_type_nonlinearity"] = nonlinearity_matrix
                
                if "neuron_type_nonlinearity" in hyperparams:
                    model_config["rnn_kwargs"]["area_kwargs"][0]["neuron_type_nonlinearity"] = hyperparams["neuron_type_nonlinearity"]
                        
            elif model_type == "cnn":
                # Apply any CNN-specific hyperparameters
                if "fc_dim" in hyperparams:
                    # For CNN, we might ignore fc_dim or adapt it somehow
                    pass
            
            # Apply any other model config overrides from sweep_config
            if "model_configs" in sweep_config and model_type in sweep_config["model_configs"]:
                # Deep update the nested dictionary
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                            update_dict(d[k], v)
                        else:
                            d[k] = v
                
                update_dict(model_config, sweep_config["model_configs"][model_type])

            
            try:
                val_loss, val_acc = run_experiment(
                    model_type=model_type,
                    model_config=model_config,
                    hyperparams=hyperparams,
                    rank=rank,
                    world_size=world_size,
                    distributed=(world_size > 1)
                )
                
                # Store results
                results.append({
                    "model_type": model_type,
                    "model_config": model_config,
                    "hyperparams": hyperparams,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })

                print(f"Experiment completed: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                
            except Exception as e:
                print(f"Error in experiment: {e}")
                import traceback
                traceback.print_exc()
                if rank == 0:
                    wandb.finish()
    
    # Find best model
    if results:
        best_result = max(results, key=lambda x: x["val_acc"])
        print("\n" + "="*50)
        print(f"Best model: {best_result['model_type']}")
        print(f"Best hyperparameters: {best_result['hyperparams']}")
        print(f"Validation accuracy: {best_result['val_acc']:.4f}")
        print(f"Validation loss: {best_result['val_loss']:.4f}")
        print("="*50)
    
    return results

def run_from_checkpoint(wandb_name, epoch, rank=0, world_size=1, distributed=False):
    full_config = pickle.load(open(checkpoint_path + f"{wandb_name}.pkl", "rb"))
    model_config = full_config["model_config"]
    num_steps = full_config["num_steps"]
    model = SpatiallyEmbeddedClassifier(**model_config).to(rank)
    
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    state_dict = torch.load(checkpoint_path + f"{wandb_name}/{epoch}.pth", map_location=f'cuda:{rank}')
    if distributed:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    # Extract hyperparameters
    lr = full_config.get("lr", 0.001)
    num_samples = full_config.get("num_samples", 100)
    num_steps = full_config.get("num_steps", 60)
    max_epochs = full_config.get("max_epochs", 10)
    batch_size = full_config.get("batch_size", 128)
    
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(batch_size=batch_size, num_samples=num_samples, rank=rank, world_size=world_size, distributed=distributed)

    train_log_frequency = max(1, len(train_loader) // 10)  # How often to log training metrics
    
    if rank == 0:
        run = wandb.init(
            project="mazes",  # Specify your project
            config=full_config,
        )
    else:
        run = None

    wandb_name = run.name if rank == 0 else None
    #save model config
    if rank == 0:
        with open(checkpoint_path + f"{wandb_name}.pkl", "wb") as f:
            pickle.dump(full_config, f)
    
    train(model, train_loader, test_loader, criterion, optimizer, num_steps, max_epochs, train_log_frequency, wandb_name, run, rank, world_size, distributed)
    
    # Evaluate final performance
    final_val_loss, final_val_acc = evaluate(model, test_loader, criterion, device, num_steps, rank)
    
    if rank == 0:
        wandb.log({
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        })
        
        wandb.finish()
    
    return final_val_loss, final_val_acc

def main_worker(rank, world_size, sweep_config):
    """Main worker function for distributed training."""
    distributed = False
    
    try:
        # Setup distributed training
        if world_size > 1:
            setup_ddp(rank, world_size)
            distributed = True
            
        # Run the sweep
        results = run_sweep(sweep_config, world_size, rank)
        
        # Print results only on main process
        if rank == 0:
            print("Sweep completed!")
            print(f"Results: {results}")
            
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Cleanup distributed training
        if distributed:
            cleanup_ddp()

if __name__ == "__main__":
    print("Running sweep...")
    
    # Define sweep configuration``
    sweep_config = {
        "model_types": ["1e1i1a"],
        "hyperparams": {
            "lr": [0.001],  
            "num_samples": [345600],
            "num_steps": [20],
            "max_epochs": [2000],  
            "batch_size": [128],  # Reduced per-GPU batch size for multi-GPU
            "num_neuron_subtypes": [[32, 8]],
            "fc_dim": [512],
            "out_channels": [32],
            "neuron_type_nonlinearity": ["ReLU"],  
            "inter_neuron_type_nonlinearity": ["ReLU"],
            "inter_neuron_type_spatial_extents": [(7, 7)],
            "init_weights": ["none"],
        },
        "model_configs": {
            "cnn": {
                "dropout": 0.1,
            },
            "bioplnn": {
                "dropout": 0.1,
            }
        }
    }
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    # Multi-GPU training
    if world_size > 1:
        print("Starting multi-GPU training...")
        # Use multiprocessing for multi-GPU training
        mp.spawn(
            main_worker,
            args=(world_size, sweep_config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        results = run_sweep(sweep_config, world_size=1)

    # wandb_name = "celestial-lion-366"
    # run_from_checkpoint(wandb_name, 460)
    # wandb_name = "apricot-durian-360"
    # run_from_checkpoint(wandb_name, 1190)