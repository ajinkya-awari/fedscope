"""Offline contract tests for the FedScope model, gate, client, and strategies."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from torch import nn

from fedscope.client import FedScopeClient
from fedscope.gates import (
    CentralizedGateResult,
    evaluate_centralized_gate,
    macro_f1,
    require_centralized_gate,
)
from fedscope.models import FedScopeCNN
from fedscope.strategies import FedAvgStrategy, FedProxStrategy


def _linear_model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 28 * 28, 9))


class _AlwaysZero(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((inputs.shape[0], 9), dtype=inputs.dtype)
        logits[:, 0] = 1.0
        return logits


def _four_sample_loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (torch.zeros((2, 3, 28, 28)), torch.tensor([[0], [1]])),
        (torch.ones((2, 3, 28, 28)), torch.tensor([[0], [1]])),
    ]


def test_cnn_returns_nine_logits_for_ten_rgb_28px_samples() -> None:
    model = FedScopeCNN()

    logits = model(torch.zeros((10, 3, 28, 28)))

    assert logits.shape == (10, 9)


def test_macro_f1_uses_flat_labels_and_zero_division_zero() -> None:
    score = macro_f1([0, 0, 1, 1, 2], [0, 1, 1, 1, 1])

    assert score == pytest.approx(((2.0 / 3.0) + (2.0 / 3.0)) / 9.0)


def test_macro_f1_includes_all_nine_pathmnist_classes() -> None:
    score = macro_f1([0], [0])

    assert score == pytest.approx(1.0 / 9.0)


def test_centralized_gate_reports_failed_four_sample_macro_f1_decision() -> None:
    result = evaluate_centralized_gate(
        _AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.40
    )

    assert result == CentralizedGateResult(
        passed=False,
        metric=pytest.approx((2.0 / 3.0) / 9.0),
        threshold=0.40,
        sample_count=4,
        seed=42,
        device="cpu",
    )


def test_require_centralized_gate_rejects_missing_and_failed_gate(tmp_path) -> None:
    failed = CentralizedGateResult(False, 0.69, 0.70, 4, 42, "cpu")

    with pytest.raises(RuntimeError, match="centralized gate.*not found"):
        require_centralized_gate(tmp_path / "missing-gate.json")
    with pytest.raises(RuntimeError, match="centralized gate"):
        require_centralized_gate(failed)


def test_require_centralized_gate_returns_passed_gate(tmp_path) -> None:
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")

    loaded = require_centralized_gate(path)
    assert loaded.passed is True


def test_require_centralized_gate_revalidates_passed_result_integrity() -> None:
    forged = CentralizedGateResult(True, 0.90, 0.70, 1, 42, "cpu")

    with pytest.raises(RuntimeError, match="centralized gate"):
        require_centralized_gate(forged)


def test_client_fit_uses_flat_labels_and_returns_positive_tau() -> None:
    client = FedScopeClient(_linear_model(), _four_sample_loader(), _four_sample_loader())

    parameters, sample_count, metrics = client.fit(
        client.get_parameters(),
        {"local_epochs": 1, "learning_rate": 0.01, "proximal_mu": 0.0},
    )

    assert len(parameters) == len(client.get_parameters())
    assert sample_count == 4
    assert metrics["tau"] == 2


def test_client_requires_explicit_proximal_mu_config() -> None:
    client = FedScopeClient(_linear_model(), _four_sample_loader(), _four_sample_loader())

    with pytest.raises(KeyError, match="proximal_mu"):
        client.fit(client.get_parameters(), {"local_epochs": 1, "learning_rate": 0.01})


def test_client_evaluate_returns_macro_f1_for_four_validation_samples() -> None:
    client = FedScopeClient(_linear_model(), _four_sample_loader(), _four_sample_loader())

    loss, sample_count, metrics = client.evaluate(client.get_parameters(), {})

    assert loss >= 0.0
    assert sample_count == 4
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_fedavg_aggregates_five_parameter_sets_by_sample_count() -> None:
    strategy = FedAvgStrategy(num_clients=2)
    strategy.configure_fit(1, [np.array([0.0, 0.0])])
    results = [
        ([np.array([1.0, 3.0])], 1, {}),
        ([np.array([3.0, 9.0])], 3, {}),
    ]

    aggregated, metrics = strategy.aggregate_fit(1, results, [])

    assert aggregated[0].tolist() == [2.5, 7.5]
    assert metrics["sample_count"] == 4
    assert len(strategy.drift_history) == 2


def test_fedprox_preserves_mu_in_each_client_instruction() -> None:
    strategy = FedProxStrategy(proximal_mu=0.25, num_clients=2)

    instructions = strategy.configure_fit(3, [np.array([1.0])])

    assert len(instructions) == 2
    assert all(item.config["proximal_mu"] == 0.25 for item in instructions)
    assert strategy.algorithm_name == "FedProx"


def test_run_federated_blocks_before_strategy_work_for_failed_gate(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")

    class ExplodingStrategy:
        def configure_fit(self, server_round, parameters):
            raise AssertionError("strategy work must not start")

    failed = CentralizedGateResult(False, 0.69, 0.70, 4, 42, "cpu")
    with pytest.raises(RuntimeError, match="centralized gate"):
        runner.run_federated(
            strategy=ExplodingStrategy(),
            parameters=[np.array([1.0])],
            baseline_gate=tmp_path / "missing-baseline.json",
        )


def test_run_federated_requires_five_client_partition_provenance(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    baseline_path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")

    with pytest.raises(ValueError, match="partition manifest"):
        runner.run_federated(
            strategy=FedAvgStrategy(num_clients=5),
            parameters=[np.array([1.0])],
            baseline_gate=baseline_path,
            output_path=tmp_path / "run.json",
            seed=42,
            alpha=0.5,
        )


def test_run_federated_rejects_fabricated_or_ambiguous_manifest(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    baseline_path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")
    fabricated = {
        "dataset": "PathMNIST",
        "size": 28,
        "n_clients": 5,
        "alpha": 0.5,
        "seed": 42,
        "sample_count": 5,
        "client_sizes": {index: 1 for index in range(5)},
        "class_coverage": {index: [0] for index in range(5)},
        "exclusions": [],
    }

    with pytest.raises(ValueError, match="client_indices|manifest"):
        runner.run_federated(
            strategy=FedAvgStrategy(num_clients=5),
            parameters=[np.array([1.0])],
            baseline_gate=baseline_path,
            partition_manifest=fabricated,
            output_path=tmp_path / "run.json",
            seed=42,
            alpha=0.5,
        )

    with pytest.raises(ValueError, match="output path"):
        runner.run_federated(
            strategy=FedAvgStrategy(num_clients=5),
            parameters=[np.array([1.0])],
            baseline_gate=baseline_path,
            partition_manifest={**fabricated, "client_indices": {index: [index] for index in range(5)}},
            output_path=object(),
            seed=42,
            alpha=0.5,
        )


def test_run_federated_requires_exact_manifest_alpha(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    baseline_path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")
    manifest = {
        "dataset": "PathMNIST",
        "size": 28,
        "n_clients": 5,
        "alpha": 0.5,
        "seed": 42,
        "sample_count": 5,
        "client_sizes": {index: 1 for index in range(5)},
        "class_coverage": {index: [0] for index in range(5)},
        "client_indices": {index: [index] for index in range(5)},
        "exclusions": [],
    }

    with pytest.raises(ValueError, match="seed and alpha"):
        runner.run_federated(
            strategy=FedAvgStrategy(num_clients=5),
            parameters=[np.array([1.0])],
            baseline_gate=baseline_path,
            partition_manifest=manifest,
            output_path=tmp_path / "run.json",
            seed=42,
            alpha=0.5000000001,
        )


def test_run_federated_rejects_malformed_manifest_shapes(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    baseline_path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")
    malformed = {
        "dataset": "PathMNIST",
        "size": 28,
        "n_clients": 5,
        "alpha": 0.5,
        "seed": 42,
        "sample_count": 5,
        "client_sizes": [],
        "class_coverage": {index: [0] for index in range(5)},
        "client_indices": {index: [index] for index in range(5)},
        "exclusions": [],
    }

    with pytest.raises(ValueError, match="client_sizes"):
        runner.run_federated(
            strategy=FedAvgStrategy(num_clients=5),
            parameters=[np.array([1.0])],
            baseline_gate=baseline_path,
            partition_manifest=malformed,
            output_path=tmp_path / "run.json",
            seed=42,
            alpha=0.5,
        )


def test_run_federated_rejects_malformed_manifest_values(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    baseline_path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")
    malformed = {
        "dataset": "PathMNIST",
        "size": 28,
        "n_clients": 5,
        "alpha": "bad",
        "seed": 42,
        "sample_count": "bad",
        "client_sizes": {index: 1 for index in range(5)},
        "class_coverage": [],
        "client_indices": {index: [index] for index in range(5)},
        "exclusions": "none",
    }

    with pytest.raises(ValueError, match="manifest"):
        runner.run_federated(
            strategy=FedAvgStrategy(num_clients=5),
            parameters=[np.array([1.0])],
            baseline_gate=baseline_path,
            partition_manifest=malformed,
            output_path=tmp_path / "run.json",
            seed=42,
            alpha=0.5,
        )


def test_run_federated_rejects_nested_manifest_values_and_numeric_alpha_strings(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    baseline_path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")
    malformed = {
        "dataset": "PathMNIST",
        "size": 28,
        "n_clients": 5,
        "alpha": "0.5",
        "seed": 42,
        "sample_count": 5,
        "client_sizes": {index: 1 for index in range(5)},
        "class_coverage": {index: [0, "bad"] for index in range(5)},
        "client_indices": {index: [index] for index in range(5)},
        "exclusions": [{"reason": "ok"}, "bad"],
    }

    with pytest.raises(ValueError, match="alpha|class_coverage|exclusions"):
        runner.run_federated(
            strategy=FedAvgStrategy(num_clients=5),
            parameters=[np.array([1.0])],
            baseline_gate=baseline_path,
            partition_manifest=malformed,
            output_path=tmp_path / "run.json",
            seed=42,
            alpha=0.5,
        )


def test_run_federated_persists_a_non_result_run_artifact(tmp_path) -> None:
    runner = importlib.import_module("study.02_federated")
    passed = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)
    baseline = importlib.import_module("study.01_baseline")
    baseline_path = baseline.save_baseline_gate(passed, tmp_path / "baseline.json")
    manifest = {
        "dataset": "PathMNIST",
        "size": 28,
        "n_clients": 5,
        "alpha": 0.5,
        "seed": 42,
        "sample_count": 5,
        "client_sizes": {index: 1 for index in range(5)},
        "class_coverage": {index: [0] for index in range(5)},
        "client_indices": {index: [index] for index in range(5)},
        "exclusions": [],
    }
    output_path = tmp_path / "run.json"

    runner.run_federated(
        strategy=FedAvgStrategy(num_clients=5),
        parameters=[np.array([1.0])],
        baseline_gate=baseline_path,
        partition_manifest=manifest,
        output_path=output_path,
        seed=42,
        alpha=0.5,
    )

    payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "configured"
    assert payload["provenance"]["sample_count"] == 5


def test_baseline_gate_can_persist_a_provenance_bearing_decision(tmp_path) -> None:
    runner = importlib.import_module("study.01_baseline")
    result = evaluate_centralized_gate(_AlwaysZero(), _four_sample_loader(), seed=42, threshold=0.0)

    path = runner.save_baseline_gate(result, tmp_path / "baseline_gate.json")

    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["sample_count"] == 4
    assert payload["seed"] == 42
    assert require_centralized_gate(path).passed is True


def test_gate_file_without_integrity_hash_is_rejected(tmp_path) -> None:
    path = tmp_path / "baseline_gate.json"
    path.write_text(
        __import__("json").dumps(
            {"passed": True, "metric": 0.9, "threshold": 0.7, "sample_count": 1, "seed": 42, "device": "cpu"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="integrity"):
        require_centralized_gate(path)


def test_non_object_gate_file_is_rejected_cleanly(tmp_path) -> None:
    path = tmp_path / "baseline_gate.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid"):
        require_centralized_gate(path)


def test_validate_gate_result_rejects_inconsistent_passed_true_with_low_metric() -> None:
    """Break caught: passed=True with metric < threshold is an inconsistent gate state."""
    from fedscope.gates import validate_gate_result

    inconsistent = CentralizedGateResult(True, 0.60, 0.70, 4, 42, "cpu")

    with pytest.raises(RuntimeError, match="did not pass"):
        validate_gate_result(inconsistent)
