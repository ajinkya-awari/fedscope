import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


@dataclass
class FakeFitResult:
    client_id: str
    parameters: list[np.ndarray]
    num_examples: int
    metrics: dict[str, float]


class FakeStreamlit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.tab_names: list[str] = []
        self.stopped = False
        self.writes: list[object] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def stop(self) -> None:
        self.stopped = True
        raise RuntimeError("streamlit stopped")

    def tabs(self, names: list[str]) -> list[object]:
        self.tab_names = names
        return [FakeTab() for _ in names]

    def write(self, value: object) -> None:
        self.writes.append(value)

    def image(self, value: object) -> None:
        self.writes.append(value)


class FakeTab:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _import_attr(module_name: str, attribute: str):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ModuleNotFoundError, AttributeError) as error:
        pytest.fail(f"Missing required interface {module_name}.{attribute}: {error}")


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "summary": {"algorithm": "FedNova", "macro_f1": 0.5},
        "convergence": [{"round": 1, "macro_f1": 0.5}],
        "client_drift": [{"round": 1, "client_id": "client-0", "drift": 1.0}],
        "communication": [{"round": 1, "bytes": 400}],
        "provenance": {
            "dataset": "PathMNIST",
            "size": 28,
            "seed": 42,
            "alpha": 0.5,
            "sample_count": 5,
            "algorithm": "FedNova",
            "metric_definition": "macro_f1_zero_division_0",
            "exclusions": [],
        },
    }


def _write_png(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")


def test_fednova_normalizes_unequal_tau_against_pre_round_weights() -> None:
    """Break caught: replacing normalized updates with a final-weight average."""
    strategy_class = _import_attr("fedscope.strategies.fednova", "FedNovaStrategy")
    strategy = strategy_class()
    global_weights = [np.array([1.0, 2.0])]
    strategy.configure_fit(server_round=1, parameters=global_weights)
    results = [
        FakeFitResult("client-a", [np.array([3.0, 4.0])], 2, {"tau": 2.0}),
        FakeFitResult("client-b", [np.array([2.0, 8.0])], 1, {"tau": 4.0}),
    ]

    weights, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    np.testing.assert_allclose(weights[0], np.array([3.0, 46.0 / 9.0]))
    assert metrics["tau_eff"] == pytest.approx(8.0 / 3.0)
    assert strategy.algorithm_name == "FedNova"
    assert [record["round"] for record in strategy.drift_history] == [1, 1]
    assert [record["drift"] for record in strategy.drift_history] == pytest.approx([np.sqrt(8.0), np.sqrt(37.0)])


def test_strategy_package_exports_all_four_algorithm_contracts() -> None:
    strategies = importlib.import_module("fedscope.strategies")

    assert strategies.FedAvgStrategy.algorithm_name == "FedAvg"
    assert strategies.FedProxStrategy.algorithm_name == "FedProx"
    assert strategies.FedNovaStrategy.algorithm_name == "FedNova"
    assert strategies.ServerMomentumStrategy.algorithm_name == "ServerMomentum"


def test_normalized_strategies_return_client_instructions_with_explicit_proximal_mu() -> None:
    strategies = importlib.import_module("fedscope.strategies")
    parameters = [np.array([0.0])]

    for strategy in (
        strategies.FedNovaStrategy(num_clients=2),
        strategies.ServerMomentumStrategy(num_clients=2),
    ):
        instructions = strategy.configure_fit(1, parameters)
        assert len(instructions) == 2
        assert all(instruction.config["proximal_mu"] == 0.0 for instruction in instructions)


def test_normalized_strategy_cannot_override_explicit_proximal_mu_default() -> None:
    strategy = importlib.import_module("fedscope.strategies").FedNovaStrategy(
        fit_config={"proximal_mu": 0.9}
    )

    assert strategy.configure_fit(1, [np.array([0.0])])[0].config["proximal_mu"] == 0.0


@pytest.mark.parametrize("tau", [0.0, -1.0, float("inf"), float("nan")])
def test_fednova_rejects_non_positive_or_non_finite_tau(tau: float) -> None:
    """Break caught: silently treating invalid FedNova tau as FedAvg."""
    strategy_class = _import_attr("fedscope.strategies.fednova", "FedNovaStrategy")
    strategy = strategy_class()
    strategy.configure_fit(1, [np.array([1.0])])

    with pytest.raises(ValueError, match="tau"):
        strategy.aggregate_fit(
            1,
            [FakeFitResult("client-a", [np.array([2.0])], 1, {"tau": tau})],
            [],
        )


def test_server_momentum_lazily_initializes_ema_and_records_pre_round_drift() -> None:
    """Break caught: eagerly initialized EMA or drift measured from post-round weights."""
    strategy_class = _import_attr("fedscope.strategies.server_momentum", "ServerMomentumStrategy")
    strategy = strategy_class(momentum=0.5)
    assert strategy.ema_parameters is None
    strategy.configure_fit(1, [np.array([0.0, 0.0])])

    weights, _ = strategy.aggregate_fit(
        1,
        [
            FakeFitResult("client-a", [np.array([2.0, 0.0])], 1, {}),
            FakeFitResult("client-b", [np.array([0.0, 4.0])], 1, {}),
        ],
        [],
    )

    np.testing.assert_allclose(weights[0], np.array([1.0, 2.0]))
    assert strategy.algorithm_name == "ServerMomentum"
    assert [record["drift"] for record in strategy.drift_history] == pytest.approx([2.0, 4.0])


def test_client_drift_and_server_momentum_communication_are_exact() -> None:
    """Break caught: wrong norm or an unintended control-variate surcharge."""
    client_drift = _import_attr("fedscope.metrics", "client_drift")
    comm_cost_bytes = _import_attr("fedscope.metrics", "comm_cost_bytes")

    assert client_drift([np.array([3.0, 4.0])], [np.array([0.0, 0.0])]) == pytest.approx(5.0)
    assert comm_cost_bytes(10, 5) == 400
    assert comm_cost_bytes(10, 5, has_control_variate=True) == 800
    assert comm_cost_bytes(10, 5, has_control_variate=False) == 400


def test_artifact_loader_requires_valid_json_png_and_schema(tmp_path: Path) -> None:
    """Break caught: dashboard accepts absent, malformed, or unverified artifacts."""
    load_artifacts = _import_attr("fedscope.artifacts", "load_artifacts")
    json_path = tmp_path / "benchmark_table.json"
    png_paths = {
        name: tmp_path / f"{name}.png"
        for name in ("convergence", "client_drift", "communication")
    }
    json_path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    for path in png_paths.values():
        _write_png(path)

    loaded = load_artifacts(json_path, png_paths, allowed_root=tmp_path)

    assert loaded["summary"]["algorithm"] == "FedNova"
    assert loaded["artifact_paths"]["json"] == str(json_path.resolve())
    assert loaded["artifact_paths"]["png"]["convergence"] == str(png_paths["convergence"].resolve())
    with pytest.raises(ValueError, match="missing|required"):
        load_artifacts(tmp_path / "missing.json", png_paths, allowed_root=tmp_path)
    png_paths["convergence"].write_bytes(b"not a png")
    with pytest.raises(ValueError, match="PNG"):
        load_artifacts(json_path, png_paths, allowed_root=tmp_path)


def test_artifact_loader_requires_provenance_and_all_three_png_views(tmp_path: Path) -> None:
    load_artifacts = _import_attr("fedscope.artifacts", "load_artifacts")
    payload = _valid_payload()
    payload.pop("provenance")
    json_path = tmp_path / "benchmark_table.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        load_artifacts(
            json_path,
            {"convergence": tmp_path / "convergence.png"},
            allowed_root=tmp_path,
        )


def test_display_only_dashboard_stops_visibly_when_artifacts_are_missing(tmp_path: Path, monkeypatch) -> None:
    """Break caught: dashboard proceeds, trains, or downloads after artifact loading fails."""
    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    sys.modules.pop("app", None)
    app = importlib.import_module("app")

    with pytest.raises(RuntimeError, match="stopped"):
        app.main(
            tmp_path / "missing.json",
            {
                name: tmp_path / f"{name}.png"
                for name in ("convergence", "client_drift", "communication")
            },
        )

    assert fake_streamlit.errors
    assert fake_streamlit.stopped is True


def test_display_only_dashboard_exposes_four_artifact_tabs(tmp_path: Path, monkeypatch) -> None:
    """Break caught: dashboard omits an analysis view or uses a non-artifact data path."""
    json_path = tmp_path / "benchmark_table.json"
    png_paths = {
        name: tmp_path / f"{name}.png"
        for name in ("convergence", "client_drift", "communication")
    }
    json_path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    for path in png_paths.values():
        _write_png(path)
    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    sys.modules.pop("app", None)
    app = importlib.import_module("app")

    app.main(json_path, png_paths)

    assert fake_streamlit.tab_names == ["Summary", "Convergence", "Client Drift", "Communication"]
    assert not fake_streamlit.errors


def test_five_client_synthetic_smoke_round_records_all_client_drifts() -> None:
    """Break caught: a multi-client round omits clients or silently ignores tau."""
    strategy_class = _import_attr("fedscope.strategies.fednova", "FedNovaStrategy")
    strategy = strategy_class()
    strategy.configure_fit(1, [np.array([0.0])])
    results = [
        FakeFitResult(f"client-{index}", [np.array([float(index + 1)])], index + 1, {"tau": float(index + 1)})
        for index in range(5)
    ]

    weights, _ = strategy.aggregate_fit(1, results, [])

    assert len(results) == 5
    assert len(strategy.drift_history) == 5
    assert weights[0][0] == pytest.approx(11.0 / 3.0)
