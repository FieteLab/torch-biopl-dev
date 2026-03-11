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
    # Generic samples-per-epoch controls for non-correlated_dots datasets
    # For correlated_dots, samples_per_epoch is handled inside its dataloader
    spe: Optional[int] = None
    val_spe: Optional[int] = None
    if dataset != "correlated_dots":
        spe = kwargs.pop("samples_per_epoch", None)
        val_spe = kwargs.pop("val_samples_per_epoch", None)
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

    # If requested, subset non-correlated datasets to a fixed number of samples per epoch
    if dataset != "correlated_dots" and spe is not None:
        try:
            import random
            from torch.utils.data import Subset, DataLoader

            seed = kwargs.get("seed")
            rng = random.Random(seed)

            # Subset train
            train_dataset = train_loader.dataset
            total_train = len(train_dataset)  # type: ignore
            if spe < total_train:
                train_indices = rng.sample(range(total_train), k=spe)
                train_subset = Subset(train_dataset, train_indices)
                train_loader = DataLoader(
                    train_subset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    pin_memory=train_loader.pin_memory,
                    worker_init_fn=train_loader.worker_init_fn,
                    generator=train_loader.generator,
                )

            # Subset val proportionally (or to provided val_spe)
            if val_loader is not None:
                val_dataset = val_loader.dataset
                total_val = len(val_dataset)  # type: ignore
                if total_train > 0:
                    target_val = (
                        val_spe
                        if val_spe is not None
                        else int(round(min(spe, total_train) * total_val / total_train))
                    )
                else:
                    target_val = val_spe if val_spe is not None else total_val
                target_val = max(0, min(target_val, total_val))
                if target_val < total_val:
                    val_indices = rng.sample(range(total_val), k=target_val)
                    val_subset = Subset(val_dataset, val_indices)
                    val_loader = DataLoader(
                        val_subset,
                        batch_size=val_loader.batch_size,
                        shuffle=False,
                        num_workers=val_loader.num_workers,
                        pin_memory=val_loader.pin_memory,
                        worker_init_fn=val_loader.worker_init_fn,
                        generator=val_loader.generator,
                    )
        except Exception:
            # Fall back silently if any dataset does not support len()/indexing
            pass

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
