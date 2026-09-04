"""Offline analysis of FedScope benchmark artifacts.

Loads precomputed JSON artifacts from a completed approved-seeds run and
produces a summary table (mean ± std across seeds) for each algorithm.
No PathMNIST data, no Flower, no training; display-only.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ALGORITHMS = ("FedAvg", "FedProx", "FedNova", "ServerMomentum")
SEEDS = (42, 123, 456)


def load_artifacts(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all artifacts.json files for each (seed, algorithm) pair."""
    results: dict[str, list[dict[str, Any]]] = {algo: [] for algo in ALGORITHMS}
    for seed in SEEDS:
        for algo in ALGORITHMS:
            artifact_path = output_root / f"seed_{seed}" / algo / "artifacts.json"
            if not artifact_path.is_file():
                continue
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            results[algo].append(payload)
    return results


def final_round_f1(payload: dict[str, Any]) -> float | None:
    convergence = payload.get("convergence", [])
    if not convergence:
        return None
    last = max(convergence, key=lambda row: row["round"])
    return float(last["macro_f1"])


def mean_drift(payload: dict[str, Any]) -> float | None:
    drift_records = payload.get("client_drift", [])
    if not drift_records:
        return None
    values = [float(row["drift"]) for row in drift_records if "drift" in row]
    return statistics.mean(values) if values else None


def total_comm_bytes(payload: dict[str, Any]) -> int | None:
    comm = payload.get("communication", [])
    if not comm:
        return None
    return sum(int(row["bytes"]) for row in comm)


def summarise(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for algo in ALGORITHMS:
        payloads = results[algo]
        if not payloads:
            rows.append({"algorithm": algo, "seeds_run": 0, "note": "no artifacts found"})
            continue
        f1_values = [v for p in payloads if (v := final_round_f1(p)) is not None]
        drift_values = [v for p in payloads if (v := mean_drift(p)) is not None]
        comm_values = [v for p in payloads if (v := total_comm_bytes(p)) is not None]
        row: dict[str, Any] = {
            "algorithm": algo,
            "seeds_run": len(payloads),
            "final_macro_f1_mean": round(statistics.mean(f1_values), 4) if f1_values else None,
            "final_macro_f1_std": round(statistics.stdev(f1_values), 4) if len(f1_values) > 1 else None,
            "mean_client_drift_mean": round(statistics.mean(drift_values), 6) if drift_values else None,
            "total_comm_bytes": comm_values[0] if comm_values else None,
            "device": payloads[0].get("summary", {}).get("device"),
            "centralized_f1": round(payloads[0]["summary"]["centralized_macro_f1"], 4) if payloads else None,
        }
        rows.append(row)
    return rows


def print_table(summary: list[dict[str, Any]]) -> None:
    header = f"{'Algorithm':<20} {'Seeds':>5} {'Final F1 mean':>14} {'±std':>7} {'Client drift':>13} {'Comm (MB)':>10}"
    print(header)
    print("-" * len(header))
    for row in summary:
        if "note" in row:
            print(f"{row['algorithm']:<20}  {row['note']}")
            continue
        f1_mean = f"{row['final_macro_f1_mean']:.4f}" if row["final_macro_f1_mean"] is not None else "    —"
        f1_std = f"{row['final_macro_f1_std']:.4f}" if row["final_macro_f1_std"] is not None else "   —"
        drift = f"{row['mean_client_drift_mean']:.4f}" if row["mean_client_drift_mean"] is not None else "           —"
        comm = f"{row['total_comm_bytes'] / 1e6:.2f}" if row["total_comm_bytes"] is not None else "         —"
        print(f"{row['algorithm']:<20} {row['seeds_run']:>5} {f1_mean:>14} {f1_std:>7} {drift:>13} {comm:>10}")


def main(output_root: Path | None = None) -> None:
    if output_root is None:
        output_root = Path("/kaggle/working/fedscope_artifacts")
    if not output_root.is_dir():
        raise SystemExit(f"Artifact directory not found: {output_root}\nRun the approved-seeds Kaggle job first.")
    results = load_artifacts(output_root)
    summary = summarise(results)
    print(f"\nFedScope Benchmark Summary — PathMNIST size 28, 5 clients, alpha=0.5")
    print(f"Seeds: {SEEDS} | Rounds: 3 | Device: {summary[0].get('device', 'unknown') if summary else '?'}")
    print(f"Centralized baseline F1: {summary[0].get('centralized_f1', '?') if summary else '?'}\n")
    print_table(summary)
    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    main(args.output_root)
