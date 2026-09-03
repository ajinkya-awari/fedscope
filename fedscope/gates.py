"""Centralized macro-F1 gate used to block federated work until it passes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CentralizedGateResult:
    passed: bool
    metric: float
    threshold: float
    sample_count: int
    seed: int
    device: str


def gate_payload(result: CentralizedGateResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "metric": result.metric,
        "threshold": result.threshold,
        "sample_count": result.sample_count,
        "seed": result.seed,
        "device": result.device,
    }


def gate_integrity_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def macro_f1(y_true, y_pred) -> float:
    """Return unweighted per-class F1, assigning zero to undefined classes."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("macro_f1 requires flat integer labels")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.issubdtype(actual.dtype, np.integer) or not np.issubdtype(
        predicted.dtype, np.integer
    ):
        raise ValueError("macro_f1 requires integer labels")
    if actual.size == 0:
        return 0.0
    if np.any((actual < 0) | (actual > 8)) or np.any((predicted < 0) | (predicted > 8)):
        raise ValueError("macro_f1 labels must be in class range 0..8")

    scores: list[float] = []
    for label in range(9):
        true_positive = np.sum((actual == label) & (predicted == label))
        false_positive = np.sum((actual != label) & (predicted == label))
        false_negative = np.sum((actual == label) & (predicted != label))
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else float(2 * true_positive / denominator))
    return float(np.mean(scores))


def evaluate_centralized_gate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
    threshold: float = 0.70,
    device: str = "cpu",
) -> CentralizedGateResult:
    """Evaluate a model without training and record the reproducible gate decision."""
    run_device = torch.device(device)
    was_training = model.training
    model.to(run_device)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(run_device))
            predictions = logits.argmax(dim=1).detach().cpu().reshape(-1).tolist()
            flattened_labels = labels.detach().cpu().reshape(-1).tolist()
            y_true.extend(int(value) for value in flattened_labels)
            y_pred.extend(int(value) for value in predictions)
    if was_training:
        model.train()
    metric = macro_f1(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64))
    return CentralizedGateResult(
        passed=metric >= threshold,
        metric=metric,
        threshold=threshold,
        sample_count=len(y_true),
        seed=seed,
        device=str(run_device),
    )


def _read_gate(path: Path) -> CentralizedGateResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"centralized gate result not found: {path}") from error
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(f"centralized gate result is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"centralized gate result is invalid: {path}")
    integrity = payload.pop("integrity_sha256", None)
    if not isinstance(integrity, str) or integrity != gate_integrity_digest(payload):
        raise RuntimeError(f"centralized gate integrity is missing or invalid: {path}")
    try:
        return CentralizedGateResult(
            passed=bool(payload["passed"]),
            metric=float(payload["metric"]),
            threshold=float(payload["threshold"]),
            sample_count=int(payload["sample_count"]),
            seed=int(payload["seed"]),
            device=str(payload["device"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"centralized gate result is invalid: {path}") from error


def validate_gate_result(result: CentralizedGateResult) -> CentralizedGateResult:
    """Validate an evaluation result before it is persisted or consumed."""
    if not isinstance(result, CentralizedGateResult):
        raise RuntimeError("centralized gate result has an invalid type")
    valid_result = (
        result.passed
        and np.isfinite(result.metric)
        and np.isfinite(result.threshold)
        and 0.0 <= result.metric <= 1.0
        and 0.0 <= result.threshold <= 1.0
        and result.sample_count > 0
        and result.metric >= result.threshold
    )
    if not valid_result:
        raise RuntimeError(
            "centralized gate did not pass or is invalid "
            f"(macro_f1={result.metric:.4f}, threshold={result.threshold:.4f})"
        )
    return result


def require_centralized_gate(path_or_result: str | Path | CentralizedGateResult) -> CentralizedGateResult:
    """Return a passing persisted gate or raise before any federated action begins."""
    if not isinstance(path_or_result, (str, Path)):
        raise RuntimeError("centralized gate path is required; pass a persisted gate file")
    return validate_gate_result(_read_gate(Path(path_or_result)))
