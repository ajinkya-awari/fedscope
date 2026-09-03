"""Seeded Dirichlet partitioning and its auditable gate."""

import numpy as np

from .loader import flatten_labels


def _check_partition_arguments(n_clients, alpha, min_client_size):
    if not isinstance(n_clients, (int, np.integer)) or n_clients < 1:
        raise ValueError("n_clients must be a positive integer")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be positive") from error
    if not np.isfinite(alpha_value) or alpha_value <= 0:
        raise ValueError("alpha must be positive")
    if not isinstance(min_client_size, (int, np.integer)) or min_client_size < 0:
        raise ValueError("min_client_size must be a nonnegative integer")


def dirichlet_partition(
    labels,
    n_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = 1,
    require_all_classes: bool = False,
) -> dict[int, np.ndarray]:
    """Partition sample indices by class using a local seeded Dirichlet draw."""
    _check_partition_arguments(n_clients, alpha, min_client_size)
    flat_labels = flatten_labels(labels)
    if flat_labels.size < n_clients * min_client_size:
        raise ValueError("minimum client-size constraint cannot be met")

    rng = np.random.default_rng(seed)
    clients = {client_id: [] for client_id in range(n_clients)}
    for class_id in range(9):
        class_indices = np.flatnonzero(flat_labels == class_id)
        if require_all_classes and class_indices.size < n_clients:
            raise ValueError(
                f"require_all_classes requires at least {n_clients} samples per class"
            )
        if class_indices.size == 0:
            continue
        shuffled = rng.permutation(class_indices)
        reserved = n_clients if require_all_classes else 0
        for client_id in range(reserved):
            clients[client_id].append(int(shuffled[client_id]))
        remainder = shuffled[reserved:]
        if remainder.size:
            proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
            counts = rng.multinomial(remainder.size, proportions)
            start = 0
            for client_id, count in enumerate(counts):
                clients[client_id].extend(
                    int(index) for index in remainder[start : start + count]
                )
                start += count

    _rebalance_minimum(clients, flat_labels, min_client_size, require_all_classes)
    return {
        client_id: np.asarray(sorted(indices), dtype=np.int64)
        for client_id, indices in clients.items()
    }


def _rebalance_minimum(clients, labels, minimum, require_all_classes):
    while True:
        undersized = [client for client, indices in clients.items() if len(indices) < minimum]
        if not undersized:
            return
        target = undersized[0]
        donors = sorted(clients, key=lambda client: len(clients[client]), reverse=True)
        moved = False
        for donor in donors:
            if donor == target or len(clients[donor]) <= minimum:
                continue
            for position, index in enumerate(clients[donor]):
                if require_all_classes:
                    class_id = labels[index]
                    if sum(labels[item] == class_id for item in clients[donor]) <= 1:
                        continue
                clients[target].append(clients[donor].pop(position))
                moved = True
                break
            if moved:
                break
        if not moved:
            raise ValueError("minimum client-size constraint cannot be met")


def validate_partition(
    partition,
    labels,
    n_clients: int,
    min_client_size: int,
    require_all_classes: bool,
) -> dict:
    """Validate a partition and return sizes, class coverage, and exclusions."""
    _check_partition_arguments(n_clients, 1.0, min_client_size)
    flat_labels = flatten_labels(labels)
    expected = set(range(n_clients))
    actual = set(partition)
    if actual != expected:
        raise ValueError("partition has missing client IDs")

    seen = []
    sizes = {}
    coverage = {}
    for client_id in range(n_clients):
        indices = np.asarray(partition[client_id])
        if indices.ndim != 1:
            raise ValueError("partition indices must be one-dimensional integers")
        if indices.size == 0:
            indices = indices.astype(np.int64)
        elif not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("partition indices must be one-dimensional integers")
        if np.any(indices < 0) or np.any(indices >= flat_labels.size):
            raise ValueError("partition index is out of bounds")
        if len(indices) < min_client_size:
            raise ValueError("minimum client-size constraint failed")
        seen.extend(int(index) for index in indices)
        sizes[client_id] = int(len(indices))
        coverage[client_id] = sorted(set(int(value) for value in flat_labels[indices]))

    if len(seen) != len(set(seen)) or set(seen) != set(range(flat_labels.size)):
        raise ValueError("partition must assign every sample exactly once")
    if require_all_classes and any(classes != list(range(9)) for classes in coverage.values()):
        raise ValueError("class-coverage constraint failed")
    return {"client_sizes": sizes, "class_coverage": coverage, "exclusions": []}


def build_partition_manifest(
    partition,
    labels,
    n_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = 1,
    require_all_classes: bool = False,
    exclusions: list[dict] | None = None,
) -> dict:
    """Validate a partition and persist the reproducibility fields beside its gates."""
    _check_partition_arguments(n_clients, alpha, min_client_size)
    flat_labels = flatten_labels(labels)
    manifest = validate_partition(
        partition,
        flat_labels,
        n_clients=n_clients,
        min_client_size=min_client_size,
        require_all_classes=require_all_classes,
    )
    manifest.update(
        {
            "schema_version": 1,
            "dataset": "PathMNIST",
            "size": 28,
            "n_clients": n_clients,
            "alpha": float(alpha),
            "seed": int(seed),
            "min_client_size": min_client_size,
            "require_all_classes": bool(require_all_classes),
            "sample_count": int(flat_labels.size),
            "exclusions": list(exclusions or manifest["exclusions"]),
            "client_indices": {
                int(client_id): np.asarray(indices, dtype=np.int64).tolist()
                for client_id, indices in partition.items()
            },
        }
    )
    return manifest
