from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest


def _load_entrypoint():
    path = Path(__file__).parents[1] / "kaggle" / "run_fedscope.py"
    spec = importlib.util.spec_from_file_location("fedscope_kaggle_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kaggle_plan_is_bounded_to_pathmnist_and_approved_seeds() -> None:
    module = _load_entrypoint()

    spec = module.RunSpec()

    assert spec.dataset == "PathMNIST"
    assert spec.image_size == 28
    assert spec.n_clients == 5
    assert spec.seeds == (42, 123, 456)


def test_kaggle_plan_rejects_large_or_unapproved_runs() -> None:
    module = _load_entrypoint()

    with pytest.raises(ValueError, match="exactly five clients"):
        module.RunSpec(n_clients=4)
    with pytest.raises(ValueError, match="seeds must be exactly"):
        module.RunSpec(seeds=(42,))
    with pytest.raises(ValueError, match="size 28"):
        module.RunSpec(image_size=224)


def test_kaggle_execution_plan_keeps_smoke_separate_from_three_seed_run() -> None:
    module = _load_entrypoint()

    smoke_spec, smoke_seeds = module.resolve_execution_plan(module.RunSpec(), "smoke")
    full_spec, full_seeds = module.resolve_execution_plan(module.RunSpec(), "approved-seeds")

    assert smoke_spec.rounds == 1
    assert smoke_seeds == (42,)
    assert full_spec.rounds == module.RunSpec().rounds
    assert full_seeds == (42, 123, 456)
    with pytest.raises(ValueError, match="execution mode"):
        module.resolve_execution_plan(module.RunSpec(), "three-seed-gpu")


def test_kaggle_device_resolution_does_not_claim_cuda_without_availability() -> None:
    module = _load_entrypoint()

    assert module.resolve_device("cpu") == "cpu"
    assert module.resolve_device("auto") in {"cpu", "cuda"}


def test_kaggle_flower_strategies_receive_explicit_initial_parameters(monkeypatch, tmp_path) -> None:
    """Break caught: Flower simulation starts without an auditable initial model state."""
    module = _load_entrypoint()

    initial_parameter_payload = object()

    def fake_ndarrays_to_parameters(values):
        assert values
        return initial_parameter_payload

    flwr = ModuleType("flwr")
    common = ModuleType("flwr.common")
    common.ndarrays_to_parameters = fake_ndarrays_to_parameters
    common.parameters_to_ndarrays = lambda parameters: parameters
    client = ModuleType("flwr.client")

    class FakeNumPyClient:
        pass

    class FakeFedAvg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFedProx(FakeFedAvg):
        def __init__(self, proximal_mu, **kwargs):
            super().__init__(**kwargs)
            self.proximal_mu = proximal_mu

    client.NumPyClient = FakeNumPyClient
    strategy = ModuleType("flwr.server.strategy")
    strategy.FedAvg = FakeFedAvg
    strategy.FedProx = FakeFedProx
    server = ModuleType("flwr.server")
    server.strategy = strategy

    monkeypatch.setitem(sys.modules, "flwr", flwr)
    monkeypatch.setitem(sys.modules, "flwr.common", common)
    monkeypatch.setitem(sys.modules, "flwr.client", client)
    monkeypatch.setitem(sys.modules, "flwr.server", server)
    monkeypatch.setitem(sys.modules, "flwr.server.strategy", strategy)

    _, make_strategy, initial_parameters = module._flower_components(
        module.RunSpec(),
        seed=42,
        device="cpu",
        train_dataset=SimpleNamespace(root=tmp_path),
        partition={client_id: [] for client_id in range(5)},
        val_loader=[],
    )

    fedavg = make_strategy("FedAvg", lambda *_: (0.0, {}))
    fedprox = make_strategy("FedProx", lambda *_: (0.0, {}))

    fednova = make_strategy("FedNova", lambda *_: (0.0, {}))
    server_momentum = make_strategy("ServerMomentum", lambda *_: (0.0, {}))

    assert initial_parameters is initial_parameter_payload
    assert fedavg.kwargs["initial_parameters"] is initial_parameter_payload
    assert fedprox.kwargs["initial_parameters"] is initial_parameter_payload
    assert fednova.kwargs["initial_parameters"] is initial_parameter_payload
    assert server_momentum.kwargs["initial_parameters"] is initial_parameter_payload


def test_kaggle_runner_seeds_shuffled_loaders_and_cuda_backend() -> None:
    """Break caught: Kaggle smoke uses implicit DataLoader/CUDA randomness."""
    module = _load_entrypoint()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "torch.Generator()" in source
    assert "generator=generator if shuffle else None" in source
    assert "seed=seed + client_id" in source
    assert "torch.backends.cudnn.deterministic = True" in source
    assert "torch.backends.cudnn.benchmark = False" in source


def test_kaggle_runner_records_loaded_dataset_lengths() -> None:
    """Break caught: expected split counts are presented as measured evidence."""
    module = _load_entrypoint()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "PATHMNIST_COUNTS" not in source
    assert '"train": len(train_dataset)' in source
    assert '"val": len(val_dataset)' in source
