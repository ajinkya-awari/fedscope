"""Centralized evaluation entry point for the baseline gate."""

from __future__ import annotations

import json
from pathlib import Path

from fedscope.gates import (
    CentralizedGateResult,
    evaluate_centralized_gate,
    gate_integrity_digest,
    gate_payload,
    require_centralized_gate,
    validate_gate_result,
)


def run_baseline_gate(
    model,
    loader,
    seed: int,
    threshold: float = 0.70,
    device: str = "cpu",
    output_path: str | Path | None = None,
) -> CentralizedGateResult:
    result = evaluate_centralized_gate(model, loader, seed=seed, threshold=threshold, device=device)
    if output_path is not None:
        save_baseline_gate(result, output_path)
    return result


def save_baseline_gate(result: CentralizedGateResult, output_path: str | Path) -> Path:
    """Persist the gate decision and its provenance without creating parent directories."""
    validate_gate_result(result)
    path = Path(output_path)
    payload = gate_payload(result)
    payload["integrity_sha256"] = gate_integrity_digest(payload)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
