from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from fedscope.data.loader import flatten_labels, get_pathmnist_dataset
from fedscope.data.partition import (
    build_partition_manifest,
    dirichlet_partition,
    validate_partition,
)


def test_flatten_labels_accepts_medmnist_column_labels_and_rejects_bad_ids():
    labels = flatten_labels(np.array([[0], [4], [8]]))

    assert labels.shape == (3,)
    assert labels.dtype.kind in "iu"
    np.testing.assert_array_equal(labels, np.array([0, 4, 8]))
    np.testing.assert_array_equal(flatten_labels(np.array([[4]])), np.array([4]))

    with pytest.raises(ValueError, match="one-dimensional"):
        flatten_labels(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="0..8"):
        flatten_labels(np.array([0, 9]))


def test_loader_is_lazy_and_passes_pathmnist_size_transform_and_download(monkeypatch, tmp_path):
    calls = {}

    class FakeToTensor:
        pass

    def fake_pathmnist(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(labels=np.array([[0], [1]]))

    medmnist = ModuleType("medmnist")
    medmnist.PathMNIST = fake_pathmnist
    torchvision = ModuleType("torchvision")
    torchvision.transforms = SimpleNamespace(ToTensor=FakeToTensor)
    monkeypatch.setitem(__import__("sys").modules, "medmnist", medmnist)
    monkeypatch.setitem(__import__("sys").modules, "torchvision", torchvision)

    dataset = get_pathmnist_dataset("train", tmp_path, size=28, download=True)

    assert dataset.labels.shape == (2,)
    assert calls == {
        "split": "train",
        "root": str(tmp_path),
        "size": 28,
        "transform": calls["transform"],
        "download": True,
    }
    assert isinstance(calls["transform"], FakeToTensor)


def test_loader_rejects_non_28_pathmnist_size_before_construction(tmp_path):
    with pytest.raises(ValueError, match="size 28"):
        get_pathmnist_dataset("train", tmp_path, size=224, download=False)


def test_dirichlet_partition_is_seeded_complete_and_validated():
    labels = np.tile(np.arange(9, dtype=np.int64), 5)

    first = dirichlet_partition(
        labels, n_clients=3, alpha=0.5, seed=42, min_client_size=5,
        require_all_classes=True,
    )
    second = dirichlet_partition(
        labels, n_clients=3, alpha=0.5, seed=42, min_client_size=5,
        require_all_classes=True,
    )

    assert first.keys() == {0, 1, 2}
    for client_id in first:
        np.testing.assert_array_equal(first[client_id], second[client_id])
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(first.values()))), np.arange(labels.size)
    )
    manifest = validate_partition(first, labels, 3, 5, True)
    assert sum(manifest["client_sizes"].values()) == labels.size
    assert all(size >= 5 for size in manifest["client_sizes"].values())
    assert all(set(classes) == set(range(9)) for classes in manifest["class_coverage"].values())
    assert manifest["exclusions"] == []


def test_partition_gate_fails_clearly_when_constraints_are_impossible():
    labels = np.arange(9, dtype=np.int64)

    with pytest.raises(ValueError, match="requires at least 2 samples per class"):
        dirichlet_partition(labels, 2, 0.5, 7, require_all_classes=True)

    with pytest.raises(ValueError, match="missing client IDs"):
        validate_partition({0: np.array([0, 1])}, labels, 2, 1, False)


def test_partition_manifest_persists_seed_alpha_counts_coverage_and_exclusions():
    labels = np.tile(np.arange(9, dtype=np.int64), 5)
    partition = dirichlet_partition(labels, 3, 0.5, 123, min_client_size=5)

    manifest = build_partition_manifest(
        partition,
        labels,
        n_clients=3,
        alpha=0.5,
        seed=123,
        min_client_size=5,
        require_all_classes=False,
    )

    assert manifest["dataset"] == "PathMNIST"
    assert manifest["size"] == 28
    assert manifest["seed"] == 123
    assert manifest["alpha"] == pytest.approx(0.5)
    assert manifest["sample_count"] == 45
    assert set(manifest["client_sizes"]) == {0, 1, 2}
    assert set(manifest["class_coverage"]) == {0, 1, 2}
    assert sum(len(indices) for indices in manifest["client_indices"].values()) == 45
    assert manifest["exclusions"] == []

    excluded = build_partition_manifest(
        partition,
        labels,
        n_clients=3,
        alpha=0.5,
        seed=123,
        min_client_size=5,
        require_all_classes=False,
        exclusions=[{"alpha": 0.1, "reason": "client below minimum"}],
    )
    assert excluded["exclusions"][0]["reason"] == "client below minimum"


def test_partition_rejects_non_numeric_alpha_with_value_error():
    with pytest.raises(ValueError, match="alpha"):
        dirichlet_partition(np.arange(9), 3, "bad", 42)


def test_partition_manifest_integer_keys_become_strings_after_json_round_trip():
    """Break caught: manifest consumer must handle both int and str keys after JSON serialize."""
    import json

    labels = np.tile(np.arange(9, dtype=np.int64), 5)
    partition = dirichlet_partition(labels, 3, 0.5, 42, min_client_size=5)
    manifest = build_partition_manifest(partition, labels, n_clients=3, alpha=0.5, seed=42)

    assert all(isinstance(k, int) for k in manifest["client_indices"].keys()), (
        "build_partition_manifest must produce integer client_indices keys"
    )
    assert all(isinstance(k, int) for k in manifest["client_sizes"].keys()), (
        "build_partition_manifest must produce integer client_sizes keys"
    )

    round_tripped = json.loads(json.dumps(manifest))

    assert all(isinstance(k, str) for k in round_tripped["client_indices"].keys()), (
        "client_indices keys must become strings after JSON round-trip"
    )
    assert all(isinstance(k, str) for k in round_tripped["client_sizes"].keys()), (
        "client_sizes keys must become strings after JSON round-trip"
    )
    assert sum(len(v) for v in round_tripped["client_indices"].values()) == manifest["sample_count"]
