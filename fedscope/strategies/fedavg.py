"""Sample-weighted FedAvg strategy contract without a Flower dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fedscope.metrics import client_drift


@dataclass(frozen=True)
class FitInstruction:
    parameters: list[np.ndarray]
    config: dict[str, Any]


class FedAvgStrategy:
    algorithm_name = "FedAvg"

    def __init__(self, num_clients: int = 1, fit_config: dict[str, Any] | None = None) -> None:
        if num_clients < 1:
            raise ValueError("num_clients must be positive")
        self.num_clients = num_clients
        self.fit_config = dict(fit_config or {})
        self.pre_round_global_parameters: list[np.ndarray] | None = None
        self.drift_history: list[dict[str, float | int | str]] = []

    def configure_fit(self, server_round: int, parameters: Sequence[np.ndarray]) -> list[FitInstruction]:
        self.pre_round_global_parameters = [np.asarray(value).copy() for value in parameters]
        config = {**self.fit_config, "server_round": server_round, "proximal_mu": 0.0}
        return [
            FitInstruction([value.copy() for value in self.pre_round_global_parameters], dict(config))
            for _ in range(self.num_clients)
        ]

    def aggregate_fit(self, server_round: int, results: Sequence[tuple], failures: Sequence[object]):
        """Aggregate `(parameters, num_examples, metrics)` tuples by sample count."""
        if self.pre_round_global_parameters is None:
            raise ValueError("configure_fit must snapshot global parameters before aggregation")
        global_weights = [value.copy() for value in self.pre_round_global_parameters]
        if not results:
            return None, {"failure_count": len(failures), "sample_count": 0}
        unpacked = [self._unpack(result) for result in results]
        total_samples = sum(sample_count for _, sample_count, _ in unpacked)
        if total_samples <= 0:
            raise ValueError("FedAvg requires a positive total sample count")
        parameter_count = len(unpacked[0][0])
        if any(len(parameters) != parameter_count for parameters, _, _ in unpacked):
            raise ValueError("FedAvg results have different parameter counts")
        aggregated = []
        for index in range(parameter_count):
            weighted = sum(
                np.asarray(parameters[index], dtype=np.float64) * sample_count
                for parameters, sample_count, _ in unpacked
            )
            aggregated.append(weighted / total_samples)
        for client_index, (parameters, _, _) in enumerate(unpacked):
            self.drift_history.append(
                {
                    "round": server_round,
                    "client_id": str(client_index),
                    "drift": client_drift(parameters, global_weights),
                }
            )
        return aggregated, {
            "failure_count": len(failures),
            "sample_count": total_samples,
            "drift_count": len(self.drift_history),
        }

    @staticmethod
    def _unpack(result: tuple) -> tuple[Sequence[np.ndarray], int, dict]:
        if len(result) != 3:
            raise ValueError("fit results must be (parameters, num_examples, metrics)")
        parameters, sample_count, metrics = result
        if int(sample_count) < 0:
            raise ValueError("num_examples must not be negative")
        return parameters, int(sample_count), dict(metrics)
