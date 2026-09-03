"""Data loading and partitioning contracts."""

from .loader import flatten_labels, get_pathmnist_dataset
from .partition import build_partition_manifest, dirichlet_partition, validate_partition

__all__ = [
    "dirichlet_partition",
    "build_partition_manifest",
    "flatten_labels",
    "get_pathmnist_dataset",
    "validate_partition",
]
