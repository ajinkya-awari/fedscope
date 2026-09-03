"""FedProx strategy contract that carries the proximal coefficient to clients."""

from __future__ import annotations

from typing import Any

from .fedavg import FedAvgStrategy


class FedProxStrategy(FedAvgStrategy):
    algorithm_name = "FedProx"

    def __init__(
        self,
        proximal_mu: float,
        num_clients: int = 1,
        fit_config: dict[str, Any] | None = None,
    ) -> None:
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")
        super().__init__(num_clients=num_clients, fit_config=fit_config)
        self.proximal_mu = float(proximal_mu)

    def configure_fit(self, server_round: int, parameters):
        instructions = super().configure_fit(server_round, parameters)
        return [
            type(instruction)(
                parameters=instruction.parameters,
                config={**instruction.config, "proximal_mu": self.proximal_mu},
            )
            for instruction in instructions
        ]
