"""Dependency-free strategy contracts compatible with Flower-style clients."""

from .fedavg import FedAvgStrategy, FitInstruction
from .fednova import FedNovaStrategy
from .fedprox import FedProxStrategy
from .server_momentum import ServerMomentumStrategy

__all__ = [
    "FedAvgStrategy",
    "FedNovaStrategy",
    "FedProxStrategy",
    "FitInstruction",
    "ServerMomentumStrategy",
]
