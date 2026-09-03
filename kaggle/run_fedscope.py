"""Bounded Kaggle Flower simulation entry point for the approved FedScope study.

The script is intentionally self-contained at the orchestration boundary: it downloads
only PathMNIST size 28, persists the centralized gate before federated work, runs exactly
five clients, and writes provenance-bearing display artifacts. The default smoke mode runs one
round for seed 42; approved-seeds mode runs seeds 42/123/456.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fedscope.data.loader import get_pathmnist_dataset
from fedscope.data.partition import build_partition_manifest, dirichlet_partition
from fedscope.gates import (
    evaluate_centralized_gate,
    gate_integrity_digest,
    gate_payload,
    require_centralized_gate,
)
from fedscope.metrics import client_drift, comm_cost_bytes
from fedscope.models.cnn import FedScopeCNN
from fedscope.client import FedScopeClient


ALGORITHMS = ("FedAvg", "FedProx", "FedNova", "ServerMomentum")
EXECUTION_MODES = ("smoke", "approved-seeds")


@dataclass(frozen=True)
class RunSpec:
    dataset: str = "PathMNIST"
    image_size: int = 28
    n_clients: int = 5
    alpha: float = 0.5
    min_client_size: int = 10
    seeds: tuple[int, ...] = (42, 123, 456)
    rounds: int = 3
    local_epochs: int = 1
    central_epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.01
    threshold: float = 0.70
    proximal_mu: float = 0.01
    server_momentum: float = 0.9

    def __post_init__(self) -> None:
        if self.dataset != "PathMNIST" or self.image_size != 28:
            raise ValueError("the approved run is limited to PathMNIST size 28")
        if self.n_clients != 5:
            raise ValueError("the approved run requires exactly five clients")
        if self.seeds != (42, 123, 456):
            raise ValueError("seeds must be exactly (42, 123, 456)")
        if self.rounds < 1 or self.local_epochs < 1 or self.central_epochs < 1:
            raise ValueError("round and epoch counts must be positive")
        if self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("batch size and learning rate must be positive")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def resolve_execution_plan(spec: RunSpec, mode: str) -> tuple[RunSpec, tuple[int, ...]]:
    if mode == "smoke":
        return replace(spec, rounds=1), (spec.seeds[0],)
    if mode == "approved-seeds":
        return spec, spec.seeds
    raise ValueError("unsupported execution mode")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, indices=None, *, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    source = dataset if indices is None else Subset(dataset, [int(index) for index in indices])
    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("shuffled loaders require an explicit seed")
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def _train_centrally(model: torch.nn.Module, loader: DataLoader, *, epochs: int, device: str, lr: float) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            logits = model(inputs.to(device))
            loss = criterion(logits, labels.to(device).squeeze().reshape(-1))
            loss.backward()
            optimizer.step()


def _save_gate(result, path: Path) -> None:
    payload = gate_payload(result)
    payload["integrity_sha256"] = gate_integrity_digest(payload)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _set_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    state = model.state_dict()
    if len(state) != len(parameters):
        raise ValueError("parameter count does not match the model")
    restored = {
        name: torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        for (name, reference), value in zip(state.items(), parameters, strict=True)
    }
    model.load_state_dict(restored, strict=True)


def _initial_flower_parameters(converter) -> Any:
    return converter(
        [value.detach().cpu().numpy().copy() for value in FedScopeCNN().state_dict().values()]
    )


def _flower_components(spec: RunSpec, *, seed: int, device: str, train_dataset, partition, val_loader):
    """Create Flower client and strategy factories lazily after dependency installation."""
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.client import NumPyClient
    from flwr.server.strategy import FedAvg, FedProx

    class FedScopeFlowerClient(NumPyClient):
        def __init__(self, cid: str):
            client_id = int(cid)
            dataset = get_pathmnist_dataset(
                "train", root=train_dataset.root, size=spec.image_size, download=False
            )
            local_loader = _loader(
                dataset,
                partition[client_id],
                batch_size=spec.batch_size,
                shuffle=True,
                seed=seed + client_id,
            )
            model = FedScopeCNN()
            self.client = FedScopeClient(model, local_loader, (), device=device)

        def get_parameters(self, config=None):
            del config
            return self.client.get_parameters()

        def fit(self, parameters, config):
            return self.client.fit(parameters, dict(config))

        def evaluate(self, parameters, config):
            del parameters, config
            return 0.0, 0, {}

    class DriftCaptureMixin:
        algorithm_name = ""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pre_round_global_parameters = None
            self.drift_history = []

        def configure_fit(self, server_round, parameters, client_manager):
            self.pre_round_global_parameters = parameters_to_ndarrays(parameters)
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            self._capture_drift(server_round, results)
            return super().aggregate_fit(server_round, results, failures)

        def _capture_drift(self, server_round, results):
            if self.pre_round_global_parameters is None:
                raise RuntimeError("configure_fit must snapshot global parameters")
            for index, (_, fit_res) in enumerate(results):
                self.drift_history.append(
                    {
                        "round": server_round,
                        "client_id": str(index),
                        "drift": client_drift(
                            parameters_to_ndarrays(fit_res.parameters),
                            self.pre_round_global_parameters,
                        ),
                    }
                )

    class DriftFedAvg(DriftCaptureMixin, FedAvg):
        algorithm_name = "FedAvg"

    class DriftFedProx(DriftCaptureMixin, FedProx):
        algorithm_name = "FedProx"

    class FedNovaFlower(DriftCaptureMixin, FedAvg):
        algorithm_name = "FedNova"

        def aggregate_fit(self, server_round, results, failures):
            del failures
            if not results or self.pre_round_global_parameters is None:
                raise RuntimeError("FedNova requires a pre-round snapshot and client results")
            global_parameters = self.pre_round_global_parameters
            sample_counts = np.asarray([float(fit.num_examples) for _, fit in results], dtype=float)
            if not np.all(np.isfinite(sample_counts)) or np.any(sample_counts <= 0):
                raise ValueError("FedNova requires positive client sample counts")
            weights = sample_counts / sample_counts.sum()
            taus = np.asarray(
                [float(fit.metrics.get("tau", 0.0)) for _, fit in results], dtype=float
            )
            if not np.all(np.isfinite(taus)) or np.any(taus <= 0):
                raise ValueError("FedNova requires positive finite tau values")
            tau_eff = float(np.dot(weights, taus))
            normalized = [np.zeros_like(value, dtype=float) for value in global_parameters]
            self._capture_drift(server_round, results)
            for weight, (_, fit), tau in zip(weights, results, taus, strict=True):
                client_parameters = parameters_to_ndarrays(fit.parameters)
                for index, (client, global_value) in enumerate(
                    zip(client_parameters, global_parameters, strict=True)
                ):
                    normalized[index] += weight * (client - global_value) / tau
            updated = [
                global_value + tau_eff * update
                for global_value, update in zip(global_parameters, normalized, strict=True)
            ]
            return ndarrays_to_parameters(updated), {"tau_eff": tau_eff}

    class ServerMomentumFlower(DriftCaptureMixin, FedAvg):
        algorithm_name = "ServerMomentum"

        def __init__(self, *args, momentum: float, **kwargs):
            super().__init__(*args, **kwargs)
            self.momentum = momentum
            self.ema_parameters = None

        def aggregate_fit(self, server_round, results, failures):
            del failures
            if not results or self.pre_round_global_parameters is None:
                raise RuntimeError("ServerMomentum requires a pre-round snapshot and results")
            global_parameters = self.pre_round_global_parameters
            counts = np.asarray([float(fit.num_examples) for _, fit in results], dtype=float)
            if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
                raise ValueError("ServerMomentum requires positive client sample counts")
            weights = counts / counts.sum()
            averaged = [np.zeros_like(value, dtype=float) for value in global_parameters]
            self._capture_drift(server_round, results)
            for weight, (_, fit) in zip(weights, results, strict=True):
                for index, value in enumerate(parameters_to_ndarrays(fit.parameters)):
                    averaged[index] += weight * value
            if self.ema_parameters is None:
                self.ema_parameters = [value.copy() for value in averaged]
            else:
                self.ema_parameters = [
                    self.momentum * previous + (1.0 - self.momentum) * current
                    for previous, current in zip(self.ema_parameters, averaged, strict=True)
                ]
            return ndarrays_to_parameters(self.ema_parameters), {}

    def client_fn(cid: str):
        return FedScopeFlowerClient(cid).to_client()

    initial_parameters = _initial_flower_parameters(ndarrays_to_parameters)

    def make_strategy(name: str, evaluate_fn):
        fit_config = lambda server_round: {
            "server_round": server_round,
            "local_epochs": spec.local_epochs,
            "learning_rate": spec.learning_rate,
            "proximal_mu": spec.proximal_mu if name == "FedProx" else 0.0,
        }
        common = dict(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=spec.n_clients,
            min_available_clients=spec.n_clients,
            on_fit_config_fn=fit_config,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )
        if name == "FedAvg":
            return DriftFedAvg(**common)
        if name == "FedProx":
            return DriftFedProx(proximal_mu=spec.proximal_mu, **common)
        if name == "FedNova":
            return FedNovaFlower(**common)
        if name == "ServerMomentum":
            return ServerMomentumFlower(momentum=spec.server_momentum, **common)
        raise ValueError(f"unsupported algorithm: {name}")

    return client_fn, make_strategy, initial_parameters


def _write_plots(output: Path, convergence: list[dict], drift: list[dict], communication: list[dict]) -> None:
    import matplotlib.pyplot as plt

    views = {
        "convergence": ([row["round"] for row in convergence], [row["macro_f1"] for row in convergence], "macro F1"),
        "client_drift": ([row["round"] for row in drift], [row["drift"] for row in drift], "client drift"),
        "communication": ([row["round"] for row in communication], [row["bytes"] for row in communication], "bytes"),
    }
    for name, (x_values, y_values, label) in views.items():
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(x_values, y_values, marker="o")
        axis.set_xlabel("server round")
        axis.set_ylabel(label)
        axis.set_title(name)
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=120)
        plt.close(figure)


def run_seed(spec: RunSpec, *, seed: int, root: Path, output_root: Path, device: str) -> None:
    from flwr.simulation import start_simulation
    from flwr.server import ServerConfig

    seed_everything(seed)
    seed_root = output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    train_dataset = get_pathmnist_dataset("train", root=root, size=spec.image_size, download=True)
    val_dataset = get_pathmnist_dataset("val", root=root, size=spec.image_size, download=True)
    sample_counts = {"train": len(train_dataset), "val": len(val_dataset)}
    partition = dirichlet_partition(
        train_dataset.labels,
        n_clients=spec.n_clients,
        alpha=spec.alpha,
        seed=seed,
        min_client_size=spec.min_client_size,
        require_all_classes=True,
    )
    manifest = build_partition_manifest(
        partition,
        train_dataset.labels,
        n_clients=spec.n_clients,
        alpha=spec.alpha,
        seed=seed,
        min_client_size=spec.min_client_size,
        require_all_classes=True,
    )
    (seed_root / "partition_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    baseline_model = FedScopeCNN()
    _train_centrally(
        baseline_model,
        _loader(train_dataset, batch_size=spec.batch_size, shuffle=True, seed=seed),
        epochs=spec.central_epochs,
        device=device,
        lr=spec.learning_rate,
    )
    baseline = evaluate_centralized_gate(
        baseline_model,
        _loader(val_dataset, batch_size=spec.batch_size, shuffle=False),
        seed=seed,
        threshold=spec.threshold,
        device=device,
    )
    baseline_path = seed_root / "baseline_gate.json"
    if baseline.passed:
        _save_gate(baseline, baseline_path)
    else:
        _save_gate(baseline, seed_root / "baseline_gate_blocked.json")
        print(f"BLOCKED seed={seed}: centralized macro_f1={baseline.metric:.6f} < {baseline.threshold:.6f}")
        return
    require_centralized_gate(baseline_path)

    val_loader = _loader(val_dataset, batch_size=spec.batch_size, shuffle=False)
    for algorithm in ALGORITHMS:
        run_root = seed_root / algorithm
        run_root.mkdir(parents=True, exist_ok=True)
        client_fn, make_strategy, initial_parameters = _flower_components(
            spec,
            seed=seed,
            device=device,
            train_dataset=train_dataset,
            partition=partition,
            val_loader=val_loader,
        )
        convergence: list[dict[str, Any]] = []

        def evaluate_fn(server_round, parameters, config=None):
            del config
            model = FedScopeCNN()
            _set_parameters(model, parameters)
            result = evaluate_centralized_gate(
                model,
                val_loader,
                seed=seed,
                threshold=0.0,
                device=device,
            )
            convergence.append({"round": int(server_round), "macro_f1": result.metric})
            return 1.0 - result.metric, {"macro_f1": result.metric}

        strategy = make_strategy(algorithm, evaluate_fn)
        history = start_simulation(
            client_fn=client_fn,
            num_clients=spec.n_clients,
            config=ServerConfig(num_rounds=spec.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 1.0 if device == "cuda" else 0.0},
            ray_init_args={"include_dashboard": False, "num_cpus": spec.n_clients},
        )
        del history, initial_parameters
        drift = list(strategy.drift_history)
        parameter_count = sum(value.numel() for value in FedScopeCNN().parameters())
        communication = [
            {
                "round": round_id,
                "bytes": comm_cost_bytes(parameter_count, spec.n_clients),
            }
            for round_id in range(1, spec.rounds + 1)
        ]
        provenance = {
            **manifest,
            "algorithm": algorithm,
            "metric_definition": "nine-class macro F1; absent classes score zero",
            "device": device,
            "sample_count": sample_counts["train"],
            "sample_counts": sample_counts,
            "centralized_gate_path": str(baseline_path),
        }
        payload = {
            "schema_version": 1,
            "summary": {
                "algorithm": algorithm,
                "seed": seed,
                "device": device,
                "centralized_macro_f1": baseline.metric,
            },
            "convergence": convergence,
            "client_drift": drift,
            "communication": communication,
            "provenance": provenance,
        }
        (run_root / "artifacts.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        _write_plots(run_root, convergence, drift, communication)
        print(f"COMPLETED seed={seed} algorithm={algorithm} device={device} samples={sample_counts['train']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/working/medmnist"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/fedscope_artifacts"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mode", choices=EXECUTION_MODES, default="smoke")
    args = parser.parse_args()
    spec, seeds = resolve_execution_plan(RunSpec(), args.mode)
    device = resolve_device(args.device)
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {"spec": spec.__dict__, "device": device, "mode": args.mode, "seeds": seeds},
            sort_keys=True,
            default=list,
        )
    )
    for seed in seeds:
        run_seed(spec, seed=seed, root=args.data_root, output_root=args.output_root, device=device)


if __name__ == "__main__":
    main()
