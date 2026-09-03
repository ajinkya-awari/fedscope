"""Validated loading of precomputed FedScope dashboard artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_REQUIRED_SCHEMA = {
    "schema_version": int,
    "summary": dict,
    "convergence": list,
    "client_drift": list,
    "communication": list,
    "provenance": dict,
}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_REQUIRED_PNGS = {"convergence", "client_drift", "communication"}


def load_artifacts(
    json_path: str | Path,
    png_paths: Mapping[str, str | Path],
    allowed_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load only schema-valid JSON and PNG-signature-verified artifact paths."""
    resolved_json = _require_file(json_path, ".json", "JSON", allowed_root)
    try:
        payload = json.loads(resolved_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {resolved_json}") from error
    _validate_schema(payload)

    missing_pngs = sorted(_REQUIRED_PNGS - set(png_paths))
    if missing_pngs:
        raise ValueError(f"missing required PNG artifacts: {', '.join(missing_pngs)}")
    resolved_pngs: dict[str, str] = {}
    for name, path in png_paths.items():
        resolved_png = _require_file(path, ".png", "PNG", allowed_root)
        if not resolved_png.read_bytes().startswith(_PNG_SIGNATURE):
            raise ValueError(f"invalid PNG artifact: {resolved_png}")
        resolved_pngs[str(name)] = str(resolved_png)

    payload["artifact_paths"] = {"json": str(resolved_json), "png": resolved_pngs}
    return payload


def _require_file(path: str | Path, suffix: str, label: str, allowed_root: str | Path | None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != suffix or not resolved.is_file():
        raise ValueError(f"missing required {label} artifact: {resolved}")
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"{label} artifact is outside the approved artifact root: {resolved}")
    return resolved


def _validate_schema(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("invalid artifact schema: expected a JSON object")
    for key, expected_type in _REQUIRED_SCHEMA.items():
        if key not in payload or not isinstance(payload[key], expected_type):
            raise ValueError(f"invalid artifact schema: missing or invalid {key}")
    if payload["schema_version"] != 1:
        raise ValueError("invalid artifact schema: unsupported schema_version")
    provenance = payload["provenance"]
    required_provenance = {
        "dataset",
        "size",
        "seed",
        "alpha",
        "sample_count",
        "algorithm",
        "metric_definition",
        "exclusions",
    }
    missing = sorted(required_provenance - set(provenance))
    if missing:
        raise ValueError(f"invalid artifact provenance: missing {', '.join(missing)}")
    if provenance["dataset"] != "PathMNIST" or provenance["size"] != 28:
        raise ValueError("invalid artifact provenance: expected PathMNIST size 28")
    if not isinstance(provenance["seed"], int) or not isinstance(provenance["sample_count"], int):
        raise ValueError("invalid artifact provenance: seed and sample_count must be integers")
    if provenance["sample_count"] <= 0 or not math.isfinite(float(provenance["alpha"])) or provenance["alpha"] <= 0:
        raise ValueError("invalid artifact provenance: sample_count and alpha must be positive")
    if not isinstance(provenance["algorithm"], str) or not isinstance(provenance["metric_definition"], str):
        raise ValueError("invalid artifact provenance: algorithm and metric_definition must be strings")
    if not isinstance(provenance["exclusions"], list):
        raise ValueError("invalid artifact provenance: exclusions must be a list")
