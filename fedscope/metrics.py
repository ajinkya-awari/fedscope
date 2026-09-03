"""Deterministic, offline metrics for FedScope strategy contracts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def client_drift(
    client_weights: Sequence[np.ndarray], global_weights: Sequence[np.ndarray]
) -> float:
    """Return the Euclidean distance across corresponding model parameters."""
    if len(client_weights) != len(global_weights):
        raise ValueError("client and global parameter counts must match")

    squared_distance = 0.0
    for client, global_parameter in zip(client_weights, global_weights, strict=True):
        client_array = np.asarray(client, dtype=float)
        global_array = np.asarray(global_parameter, dtype=float)
        if client_array.shape != global_array.shape:
            raise ValueError("client and global parameter shapes must match")
        squared_distance += float(np.sum((client_array - global_array) ** 2))
    return float(np.sqrt(squared_distance))


def comm_cost_bytes(
    n_params: int,
    n_clients: int,
    has_control_variate: bool = False,
    dtype_bytes: int = 4,
) -> int:
    """Return bidirectional per-round payload bytes for parameter transmission.

    A control-variate method sends an additional model-sized vector in both
    directions. ServerMomentum passes ``False`` because it has no such vector.
    """
    if n_params < 0 or n_clients < 0 or dtype_bytes <= 0:
        raise ValueError("parameter count, client count, and dtype bytes must be valid")
    directions = 4 if has_control_variate else 2
    return int(n_params * n_clients * dtype_bytes * directions)
