"""Lazy PathMNIST construction and label normalization."""

from pathlib import Path

import numpy as np


def flatten_labels(labels) -> np.ndarray:
    """Return integer class labels as a flat array for PathMNIST's nine classes."""
    values = np.asarray(labels).squeeze()
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional after squeeze")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("labels must be numeric")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("labels must contain integer class IDs")
    values = values.astype(np.int64, copy=False)
    if np.any((values < 0) | (values > 8)):
        raise ValueError("labels must contain class IDs in 0..8")
    return values


def get_pathmnist_dataset(
    split: str,
    root: str | Path,
    size: int = 28,
    download: bool = False,
):
    """Construct PathMNIST without importing MedMNIST until this function is called."""
    if size != 28:
        raise ValueError("Project 05 is limited to PathMNIST size 28")

    from medmnist import PathMNIST
    from torchvision import transforms

    dataset = PathMNIST(
        split=split,
        root=str(root),
        size=size,
        transform=transforms.ToTensor(),
        download=download,
    )
    dataset.labels = flatten_labels(np.asarray(dataset.labels).squeeze())
    return dataset
