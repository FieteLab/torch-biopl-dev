import math
import os
import random
from collections.abc import Mapping
from os import PathLike
from typing import Any, Optional

import numpy as np
import torch
from addict import Dict
from torch import nn
from torch.profiler import ProfilerActivity, profile


class AttrDict(Dict):
    def __missing__(self, key: Any):
        raise KeyError(key)


def pass_fn(*args, **kwargs):
    pass


def without_keys(d: Mapping, keys: list[str]) -> dict:
    return {x: d[x] for x in d if x not in keys}


def is_list_like(x: Any) -> bool:
    if isinstance(x, (str, Mapping)):
        return False
    try:
        iter(x)
        if len(x) > 0:
            x[0]
    except Exception:
        return False
    return True


def manual_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def manual_seed_deterministic(seed: int):
    manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _get_single_activation_class(activation: str | None) -> type[nn.Module]:
    if activation is None:
        return nn.Identity
    else:
        return getattr(nn, activation)


def get_activation_class(
    activation: str,
) -> type[nn.Module] | list[type[nn.Module]]:
    activations = [act.strip() for act in activation.split(",")]
    if len(activations) == 1:
        return _get_single_activation_class(activations[0])
    else:
        return [_get_single_activation_class(act) for act in activations]


def get_activation(activation: str):
    activations = [act.strip() for act in activation.split(",")]
    if len(activations) == 1:
        return _get_single_activation_class(activations[0])()
    else:
        return nn.Sequential(
            *[_get_single_activation_class(act)() for act in activations]
        )


def init_tensor(init_mode: str, *args, **kwargs) -> torch.Tensor:
    match init_mode:
        case "zeros":
            return torch.zeros(*args, **kwargs)
        case "ones":
            return torch.ones(*args, **kwargs)
        case "randn":
            return torch.randn(*args, **kwargs)
        case "rand":
            return torch.rand(*args, **kwargs)
        case _:
            raise ValueError(
                "Invalid mode. Must be 'zeros', 'ones', 'randn', or 'rand'."
            )


def idx_1D_to_2D(x, m, n):
    """
    Convert a 1D index to a 2D index.

    Args:
        x (torch.Tensor): 1D index.

    Returns:
        torch.Tensor: 2D index.
    """
    return torch.stack((x // m, x % n))


def idx_2D_to_1D(x, m, n):
    """
    Convert a 2D index to a 1D index.

    Args:
        x (torch.Tensor): 2D index.

    Returns:
        torch.Tensor: 1D index.
    """
    return x[0] * n + x[1]


def dict_flatten(d, delimiter=".", key=None):
    key = f"{key}{delimiter}" if key is not None else ""
    non_dicts = {
        f"{key}{k}": v for k, v in d.items() if not isinstance(v, dict)
    }
    dicts = {
        f"{key}{k}": v
        for _k, _v in d.items()
        if isinstance(_v, dict)
        for k, v in dict_flatten(_v, delimiter=delimiter, key=_k).items()
    }
    for k in list(dicts.keys()):
        if k in non_dicts:
            raise ValueError(f"Key {k} is used more than once in dict.")
    return non_dicts | dicts


def expand_list(x: Any, n: int, depth: int = 0) -> list[Any]:
    """
    Expand a variable x of type T to a list of x of length n, iff the variable
    is not already an iterable of T of length n.

    Use depth > 0 if the intended type T can be indexed recursively, where
    depth is the maximum number of times x can be recursively indexed if of
    type T. For example, if x is a shallow list, then depth = 1. If T is a
    list of lists or an array or tensor, then depth = 2.

    Args:
        x: The variable to expand.
        n: The number of lists or tuples to expand to.
        depth: The depth x can be recursively indexed. A depth of -1 will
            assume x is of type list[T] and check if x is already of the
            correct length.
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    inner = x
    try:
        for _ in range(depth + 1):
            if isinstance(inner, str):
                raise TypeError
            inner = inner[0]  # type: ignore
    except TypeError:
        return [x] * n  # type: ignore

    if x is None:
        assert depth == -1
        raise ValueError("x cannot be None if depth is -1.")

    if len(x) != n:
        raise ValueError(f"x must have length {n}.")

    return x


def expand_array_2d(x: Any, m: int, n: int, depth: int = 0) -> np.ndarray:
    """
    Expand a variable x of type T to a 2D array of shape (m, n) if the
    variable is not already an array of T of shape (m, n).

    Use depth > 0 if the intended type T can be indexed recursively, where
    depth is the maximum number of times x can be recursively indexed if of
    type T. For example, if x is a shallow list, then depth = 1. If x is a
    list of lists or an array or tensor, then depth = 2.

    Args:
        x: The variable to expand.
        m: The number of rows in the expanded array.
        n: The number of columns in the expanded array.
        depth: The depth x can be recursively indexed. A depth of -1 will
            assume x is of type list[T] and check if x is already of the
            correct shape.
    """

    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")

    inner = x
    try:
        for _ in range(depth + 2):
            inner = inner[0]  # type: ignore
    except TypeError:
        array = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                array[i, j] = x
        return array

    if x is None:
        assert depth == -1
        raise ValueError("x cannot be None if depth is -1.")

    x = np.array(x, dtype=object)

    if x.shape != (m, n):
        raise ValueError(f"x must have shape ({m}, {n}).")

    return x


def print_cuda_mem_stats():
    f, t = torch.cuda.mem_get_info()
    print(f"Free/Total: {f / (1024**3):.2f}GB/{t / (1024**3):.2f}GB")


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        num_params = (
            param._nnz()
            if param.layout
            in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
            else param.numel()
        )
        total_params += num_params
    return total_params


def profile_fn(fn, kwargs, sort_by="cuda_time_total", row_limit=50):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn(kwargs)
    return prof.key_averages.table(sort_by=sort_by, row_limit=row_limit)


def create_random_topographic_hh_connectivity(
    sheet_size: tuple[int, int],
    synapse_std: float,
    synapses_per_neuron: int,
    self_recurrence: bool,
) -> torch.Tensor:
    """
    Generates random connectivity matrices for the TRNN.

    Args:
        sheet_size (tuple[int, int]): Size of the sheet-like topology.
        synapse_std (float): Standard deviation for random synapse initialization.
        synapses_per_neuron (int): Number of synapses per neuron.
        self_recurrence (bool): Whether to include self-recurrent connections.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Connectivity matrices for input-to-hidden and hidden-to-hidden connections.
    """
    # Generate random connectivity for hidden-to-hidden connections
    num_neurons = sheet_size[0] * sheet_size[1]

    idx_1d = torch.arange(num_neurons)
    idx = idx_1D_to_2D(idx_1d, sheet_size[0], sheet_size[1]).t()

    synapses = (
        torch.randn(num_neurons, 2, synapses_per_neuron) * synapse_std
        + idx.unsqueeze(-1)
    ).long()

    if self_recurrence:
        synapses = torch.cat([synapses, idx.unsqueeze(-1)], dim=2)

    synapses = synapses.clamp(
        torch.zeros(2).view(1, 2, 1),
        torch.tensor((sheet_size[0] - 1, sheet_size[1] - 1)).view(1, 2, 1),
    )
    synapses = idx_2D_to_1D(
        synapses.transpose(0, 1).flatten(1), sheet_size[0], sheet_size[1]
    ).view(num_neurons, -1)

    synapse_root = idx_1d.unsqueeze(-1).expand(-1, synapses.shape[1])

    indices_hh = torch.stack((synapses, synapse_root)).flatten(1)

    ## He initialization of values (synapses_per_neuron is the fan_in)
    values_hh = torch.randn(indices_hh.shape[1]) * math.sqrt(
        2 / synapses_per_neuron
    )

    connectivity_hh = torch.sparse_coo_tensor(
        indices_hh,
        values_hh,
        (num_neurons, num_neurons),
        check_invariants=True,
    ).coalesce()

    return connectivity_hh


def create_identity_ih_connectivity(
    input_size: int,
    num_neurons: int,
    input_indices: Optional[torch.Tensor | PathLike] = None,
) -> torch.Tensor:
    # Generate identity connectivity for input-to-hidden connections
    indices_ih = torch.stack(
        (
            input_indices
            if input_indices is not None
            else torch.arange(input_size),
            torch.arange(input_size),
        )  # type: ignore
    )

    values_ih = torch.ones(indices_ih.shape[1])

    connectivity_ih = torch.sparse_coo_tensor(
        indices_ih,
        values_ih,
        (num_neurons, input_size),
        check_invariants=True,
    ).coalesce()

    return connectivity_ih
