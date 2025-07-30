from collections.abc import Iterator
from typing import Optional

import torch
from torch import nn

from bioplnn.utils import dataloaders


def initialize_dataloader(**kwargs):
    """Initialize a dataloader based on the dataset name.

    Args:
        **kwargs: Keyword arguments to pass to the dataloader initialization function.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: The train and validation dataloaders.
    """
    dataset = kwargs.pop("dataset")
    if dataset == "mnist":
        train_loader, val_loader = dataloaders.get_mnist_dataloaders(**kwargs)
    elif dataset == "cifar10":
        train_loader, val_loader = dataloaders.get_cifar10_dataloaders(**kwargs)
    elif dataset == "cifar100":
        train_loader, val_loader = dataloaders.get_cifar100_dataloaders(**kwargs)
    elif dataset == "mnist_v1":
        train_loader, val_loader = dataloaders.get_mnist_v1_dataloaders(**kwargs)
    elif dataset == "cifar10_v1":
        train_loader, val_loader = dataloaders.get_cifar10_v1_dataloaders(**kwargs)
    elif dataset == "cifar100_v1":
        train_loader, val_loader = dataloaders.get_cifar100_v1_dataloaders(**kwargs)
    elif dataset == "mazes":
        try:
            train_loader, val_loader = dataloaders.get_mazes_dataloaders(**kwargs)
        except Exception:
            # If validation dataset fails to load, try loading only training dataset
            kwargs["train_only"] = True
            train_loader, _ = dataloaders.get_mazes_dataloaders(**kwargs)
            # Split training dataset into train and validation sets
            train_loader, val_loader = dataloaders.split_train_dataset(
                train_loader,
                val_ratio=0.2,
                seed=kwargs.get("seed"),
            )
    elif dataset == "cabc":
        train_loader, val_loader = dataloaders.get_cabc_dataloaders(**kwargs)
    elif dataset == "qclevr":
        train_loader, val_loader = dataloaders.get_qclevr_dataloaders(**kwargs)
    elif dataset == "correlated_dots":
        train_loader, val_loader = dataloaders.get_correlated_dots_dataloaders(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

    return train_loader, val_loader


def initialize_model(*, class_name: str, **kwargs) -> nn.Module:
    """Initialize a model based on the class name.

    Args:
        class_name (str): The name of the model class to use.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        nn.Module: The initialized model.
    """
    import bioplnn.models

    return getattr(bioplnn.models, class_name)(**kwargs)


def initialize_optimizer(
    *, class_name: str, model_parameters: Iterator[nn.Parameter], **kwargs
) -> torch.optim.Optimizer:
    """Initialize an optimizer for model training.

    Args:
        class_name (str): The name of the optimizer class to use.
        model_parameters (nn.ParameterList): The model parameters to optimize.
        **kwargs: Additional keyword arguments to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    return getattr(torch.optim, class_name)(model_parameters, **kwargs)


def initialize_scheduler(
    *, class_name: str, optimizer: torch.optim.Optimizer, **kwargs
) -> torch.optim.lr_scheduler.LRScheduler:
    """Initialize a learning rate scheduler.

    Args:
        class_name (str): The name of the scheduler class to use.
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        **kwargs: Additional keyword arguments to pass to the scheduler.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The initialized scheduler.
    """
    return getattr(torch.optim.lr_scheduler, class_name)(optimizer, **kwargs)


def initialize_criterion(*, class_name: str, **kwargs) -> torch.nn.Module:
    """Initialize a loss criterion.

    Args:
        class_name (str): The name of the criterion class to use.
        **kwargs: Additional keyword arguments to pass to the criterion.

    Returns:
        torch.nn.Module: The initialized criterion.
    """
    return getattr(torch.nn, class_name)(**kwargs)
