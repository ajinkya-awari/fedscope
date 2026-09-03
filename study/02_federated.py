"""Federated-run guard and minimal dependency-free orchestration contract."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path

from fedscope.gates import require_centralized_gate


def run_federated(
    *,
    strategy,
    parameters,
    baseline_gate,
    partition_manifest=None,
    seed: int | None = None,
    alpha: float | None = None,
    output_path: str | Path | None = None,
    server_round: int = 1,
    results=None,
    failures=(),
):
    """Guard the baseline before configuring or aggregating any federated work."""
    baseline_result = require_centralized_gate(baseline_gate)
    _require_run_provenance(partition_manifest, seed=seed, alpha=alpha, output_path=output_path)
    output = Path(output_path)
    instructions = strategy.configure_fit(server_round, parameters)
    if results is None:
        _write_run_artifact(
            output,
            strategy=strategy,
            server_round=server_round,
            status="configured",
            baseline_result=baseline_result,
            partition_manifest=partition_manifest,
        )
        return instructions
    aggregated = strategy.aggregate_fit(server_round, results, failures)
    _write_run_artifact(
        output,
        strategy=strategy,
        server_round=server_round,
        status="aggregated",
        baseline_result=baseline_result,
        partition_manifest=partition_manifest,
    )
    return aggregated


def _require_run_provenance(partition_manifest, *, seed, alpha, output_path) -> None:
    if not isinstance(partition_manifest, dict):
        raise ValueError("partition manifest is required before federated execution")
    if seed is None or alpha is None or output_path is None:
        raise ValueError("partition manifest, seed, alpha, and output path are required")
    required = {
        "dataset",
        "size",
        "n_clients",
        "alpha",
        "seed",
        "sample_count",
        "client_sizes",
        "class_coverage",
        "client_indices",
        "exclusions",
    }
    missing = sorted(required - set(partition_manifest))
    if missing:
        raise ValueError(f"partition manifest is missing fields: {', '.join(missing)}")
    if partition_manifest["dataset"] != "PathMNIST" or partition_manifest["size"] != 28:
        raise ValueError("partition manifest must identify PathMNIST size 28")
    if not isinstance(partition_manifest["sample_count"], Integral) or isinstance(
        partition_manifest["sample_count"], bool
    ):
        raise ValueError("partition manifest sample_count must be an integer")
    if not isinstance(partition_manifest["n_clients"], Integral) or isinstance(
        partition_manifest["n_clients"], bool
    ):
        raise ValueError("partition manifest n_clients must be an integer")
    if not isinstance(partition_manifest["seed"], Integral) or isinstance(
        partition_manifest["seed"], bool
    ):
        raise ValueError("partition manifest seed must be an integer")
    if not isinstance(partition_manifest["alpha"], Real) or isinstance(
        partition_manifest["alpha"], bool
    ):
        raise ValueError("partition manifest alpha must be numeric")
    if not isinstance(alpha, Real) or isinstance(alpha, bool):
        raise ValueError("run alpha must be numeric")
    manifest_alpha = float(partition_manifest["alpha"])
    requested_alpha = float(alpha)
    if not math.isfinite(manifest_alpha) or manifest_alpha <= 0:
        raise ValueError("partition manifest alpha must be positive and finite")
    if not isinstance(partition_manifest["client_sizes"], dict):
        raise ValueError("partition manifest client_sizes must be a mapping")
    if not isinstance(partition_manifest["client_indices"], dict):
        raise ValueError("partition manifest client_indices must be a mapping")
    if not isinstance(partition_manifest["class_coverage"], dict):
        raise ValueError("partition manifest class_coverage must be a mapping")
    if not isinstance(partition_manifest["exclusions"], list):
        raise ValueError("partition manifest exclusions must be a list")
    if len(partition_manifest["class_coverage"]) != 5:
        raise ValueError("partition manifest class_coverage must contain five clients")
    for client_id in range(5):
        coverage = partition_manifest["class_coverage"].get(
            client_id, partition_manifest["class_coverage"].get(str(client_id))
        )
        if not isinstance(coverage, list) or any(
            not isinstance(class_id, Integral)
            or isinstance(class_id, bool)
            or class_id < 0
            or class_id > 8
            for class_id in coverage
        ):
            raise ValueError("partition manifest class_coverage must contain class ID lists")
    if any(not isinstance(exclusion, dict) for exclusion in partition_manifest["exclusions"]):
        raise ValueError("partition manifest exclusions must contain mappings")
    if partition_manifest["n_clients"] != 5 or len(partition_manifest["client_sizes"]) != 5:
        raise ValueError("partition manifest must describe exactly five clients")
    if partition_manifest["sample_count"] <= 0:
        raise ValueError("partition manifest sample_count must be positive")
    if partition_manifest["seed"] != seed or manifest_alpha != requested_alpha:
        raise ValueError("run seed and alpha must match the partition manifest")
    try:
        output = Path(output_path)
    except TypeError as error:
        raise ValueError("output path must be a JSON path") from error
    if not output.is_absolute() or output.suffix.lower() != ".json":
        raise ValueError("output path must be an absolute JSON path")
    if not output.parent.is_dir():
        raise ValueError("output path parent directory must already exist")
    client_indices = partition_manifest["client_indices"]
    client_sizes = partition_manifest["client_sizes"]
    if len(client_indices) != 5:
        raise ValueError("partition manifest client_indices must contain five clients")
    all_indices = []
    for client_id in range(5):
        indices = client_indices.get(client_id, client_indices.get(str(client_id)))
        size = client_sizes.get(client_id, client_sizes.get(str(client_id)))
        if not isinstance(indices, list) or size != len(indices):
            raise ValueError("partition manifest client_indices do not match client sizes")
        if not isinstance(size, Integral) or isinstance(size, bool) or size <= 0:
            raise ValueError("partition manifest client_sizes must contain positive integers")
        if any(not isinstance(index, Integral) or isinstance(index, bool) or index < 0 for index in indices):
            raise ValueError("partition manifest client_indices must contain nonnegative integers")
        all_indices.extend(indices)
    if sorted(all_indices) != list(range(partition_manifest["sample_count"])):
        raise ValueError("partition manifest client_indices must cover each sample exactly once")


def _write_run_artifact(output: Path, *, strategy, server_round, status, baseline_result, partition_manifest) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "algorithm": str(strategy.algorithm_name),
        "server_round": int(server_round),
        "baseline_gate": {
            "metric": float(baseline_result.metric),
            "threshold": float(baseline_result.threshold),
            "sample_count": int(baseline_result.sample_count),
            "seed": int(baseline_result.seed),
            "device": baseline_result.device,
        },
        "provenance": partition_manifest,
    }
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
