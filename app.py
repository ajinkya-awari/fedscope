"""Display-only FedScope dashboard for verified, precomputed artifacts."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from fedscope.artifacts import load_artifacts


def main(
    json_path: str | Path = "results/benchmark_table.json",
    png_paths: dict[str, str | Path] | None = None,
) -> None:
    """Render recorded artifacts; visibly stop when any required artifact is invalid."""
    paths = png_paths or {
        "convergence": "results/convergence.png",
        "client_drift": "results/client_drift.png",
        "communication": "results/communication.png",
    }
    try:
        artifacts = load_artifacts(json_path, paths, allowed_root=Path(json_path).expanduser().resolve().parent)
    except ValueError as error:
        st.error(f"FedScope dashboard cannot load required artifacts: {error}")
        st.stop()
        return

    summary_tab, convergence_tab, drift_tab, communication_tab = st.tabs(
        ["Summary", "Convergence", "Client Drift", "Communication"]
    )
    with summary_tab:
        st.write(artifacts["summary"])
    with convergence_tab:
        st.write(artifacts["convergence"])
        st.image(artifacts["artifact_paths"]["png"]["convergence"])
    with drift_tab:
        st.write(artifacts["client_drift"])
        if "client_drift" in artifacts["artifact_paths"]["png"]:
            st.image(artifacts["artifact_paths"]["png"]["client_drift"])
    with communication_tab:
        st.write(artifacts["communication"])
        if "communication" in artifacts["artifact_paths"]["png"]:
            st.image(artifacts["artifact_paths"]["png"]["communication"])


if __name__ == "__main__":
    main()
