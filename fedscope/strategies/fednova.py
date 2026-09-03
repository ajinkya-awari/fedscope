"""Dependency-free FedNova aggregation for offline contract tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fedscope.metrics import client_drift
from .fedavg import FitInstruction


class FedNovaStrategy:
    algorithm_name = "FedNova"

    def __init__(self, num_clients: int = 1, fit_config: dict[str, Any] | None = None) -> None:
        if num_clients < 1:
            raise ValueError("num_clients must be positive")
        self.num_clients = num_clients
        self.fit_config = dict(fit_config or {})
        self.pre_round_global_parameters: list[np.ndarray] | None = None
        self.drift_history: list[dict[str, float | int | str]] = []

    def configure_fit(
        self, server_round: int, parameters: Sequence[np.ndarray]
    ) -> list[FitInstruction]:
        """Snapshot pre-round global parameters before client fitting begins."""
        self.pre_round_global_parameters = [np.asarray(value, dtype=float).copy() for value in parameters]
        config = {**self.fit_config, "server_round": server_round, "proximal_mu": 0.0}
        return [
            FitInstruction([value.copy() for value in self.pre_round_global_parameters], dict(config))
            for _ in range(self.num_clients)
        ]

    def aggregate_fit(
        self, server_round: int, results: Sequence[Any], failures: Sequence[Any]
    ) -> tuple[list[np.ndarray], dict[str, float]]:
        """Apply FedNova's weighted, tau-normalized client updates.

        Unlike FedAvgStrategy, which accepts ``(parameters, n, metrics)`` tuples,
        each element of ``results`` must expose ``.parameters``, ``.num_examples``,
        and ``.metrics`` attributes (Flower-style FitResult or a compatible dataclass).
        """
        del failures
        global_weights = self._require_global_weights()
        if not results:
            raise ValueError("FedNova requires at least one client result")

        sample_counts = np.asarray([float(result.num_examples) for result in results], dtype=float)
        if not np.all(np.isfinite(sample_counts)) or np.any(sample_counts <= 0):
            raise ValueError("client sample counts must be positive and finite")
        weights = sample_counts / sample_counts.sum()
        taus = np.asarray([self._tau_from(result) for result in results], dtype=float)
        tau_eff = float(np.dot(weights, taus))

        normalized_updates = [np.zeros_like(parameter, dtype=float) for parameter in global_weights]
        for index, result in enumerate(results):
            client_parameters = self._parameters_from(result, global_weights)
            self.drift_history.append(
                {
                    "round": server_round,
                    "client_id": str(getattr(result, "client_id", index)),
                    "drift": client_drift(client_parameters, global_weights),
                }
            )
            for parameter_index, (client, global_parameter) in enumerate(
                zip(client_parameters, global_weights, strict=True)
            ):
                normalized_updates[parameter_index] += (
                    weights[index] * (client - global_parameter) / taus[index]
                )

        updated_parameters = [
            global_parameter + tau_eff * normalized_update
            for global_parameter, normalized_update in zip(
                global_weights, normalized_updates, strict=True
            )
        ]
        return updated_parameters, {"tau_eff": tau_eff}

    def _require_global_weights(self) -> list[np.ndarray]:
        if self.pre_round_global_parameters is None:
            raise ValueError("configure_fit must snapshot global parameters before aggregation")
        return [value.copy() for value in self.pre_round_global_parameters]

    @staticmethod
    def _tau_from(result: Any) -> float:
        try:
            tau = float(result.metrics["tau"])
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("FedNova requires a positive finite tau per client") from error
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("FedNova requires a positive finite tau per client")
        return tau

    @staticmethod
    def _parameters_from(result: Any, global_weights: Sequence[np.ndarray]) -> list[np.ndarray]:
        try:
            client_parameters = [np.asarray(value, dtype=float) for value in result.parameters]
        except AttributeError as error:
            raise ValueError("client result must provide parameters") from error
        if len(client_parameters) != len(global_weights):
            raise ValueError("client parameter count must match global parameters")
        for client, global_parameter in zip(client_parameters, global_weights, strict=True):
            if client.shape != global_parameter.shape:
                raise ValueError("client parameter shapes must match global parameters")
        return client_parameters
