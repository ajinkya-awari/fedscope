"""Dependency-free server-side EMA aggregation for offline contract tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fedscope.metrics import client_drift
from .fedavg import FitInstruction


class ServerMomentumStrategy:
    algorithm_name = "ServerMomentum"

    def __init__(
        self,
        momentum: float = 0.9,
        num_clients: int = 1,
        fit_config: dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if num_clients < 1:
            raise ValueError("num_clients must be positive")
        self.momentum = momentum
        self.num_clients = num_clients
        self.fit_config = dict(fit_config or {})
        self.ema_parameters: list[np.ndarray] | None = None
        self.pre_round_global_parameters: list[np.ndarray] | None = None
        self.drift_history: list[dict[str, float | int | str]] = []

    def configure_fit(
        self, server_round: int, parameters: Sequence[np.ndarray]
    ) -> list[FitInstruction]:
        """Snapshot global parameters before this round's client updates."""
        self.pre_round_global_parameters = [np.asarray(value, dtype=float).copy() for value in parameters]
        config = {**self.fit_config, "server_round": server_round, "proximal_mu": 0.0}
        return [
            FitInstruction([value.copy() for value in self.pre_round_global_parameters], dict(config))
            for _ in range(self.num_clients)
        ]

    def aggregate_fit(
        self, server_round: int, results: Sequence[Any], failures: Sequence[Any]
    ) -> tuple[list[np.ndarray], dict[str, float]]:
        """Average clients, then lazily initialize or advance server-side EMA.

        Like FedNovaStrategy, each element of ``results`` must expose ``.parameters``,
        ``.num_examples``, and ``.metrics`` attributes (Flower-style FitResult or a
        compatible dataclass), not the ``(parameters, n, metrics)`` tuples accepted
        by FedAvgStrategy.
        """
        del failures
        global_weights = self._require_global_weights()
        if not results:
            raise ValueError("ServerMomentum requires at least one client result")

        sample_counts = np.asarray([float(result.num_examples) for result in results], dtype=float)
        if not np.all(np.isfinite(sample_counts)) or np.any(sample_counts <= 0):
            raise ValueError("client sample counts must be positive and finite")
        proportions = sample_counts / sample_counts.sum()
        averaged = [np.zeros_like(value, dtype=float) for value in global_weights]

        for index, result in enumerate(results):
            client_parameters = self._parameters_from(result, global_weights)
            self.drift_history.append(
                {
                    "round": server_round,
                    "client_id": str(getattr(result, "client_id", index)),
                    "drift": client_drift(client_parameters, global_weights),
                }
            )
            for parameter_index, client_parameter in enumerate(client_parameters):
                averaged[parameter_index] += proportions[index] * client_parameter

        if self.ema_parameters is None:
            self.ema_parameters = [value.copy() for value in averaged]
        else:
            self.ema_parameters = [
                self.momentum * previous + (1.0 - self.momentum) * current
                for previous, current in zip(self.ema_parameters, averaged, strict=True)
            ]
        return [value.copy() for value in self.ema_parameters], {}

    def _require_global_weights(self) -> list[np.ndarray]:
        if self.pre_round_global_parameters is None:
            raise ValueError("configure_fit must snapshot global parameters before aggregation")
        return [value.copy() for value in self.pre_round_global_parameters]

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
