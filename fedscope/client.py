"""Flower-like, dependency-free client contract for offline FedScope tests."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from .gates import macro_f1


class FedScopeClient:
    """Shared local-SGD client with Flower-style parameter and result tuples."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        validation_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = torch.device(device)
        self.model.to(self.device)

    def get_parameters(self) -> list[np.ndarray]:
        return [value.detach().cpu().numpy().copy() for value in self.model.state_dict().values()]

    def _set_parameters(self, parameters: list[np.ndarray]) -> None:
        state = self.model.state_dict()
        if len(parameters) != len(state):
            raise ValueError("parameter count does not match the model")
        restored = {}
        for (name, reference), value in zip(state.items(), parameters, strict=True):
            tensor = torch.as_tensor(value, dtype=reference.dtype, device=self.device)
            if tensor.shape != reference.shape:
                raise ValueError(f"parameter shape does not match for {name}")
            restored[name] = tensor.clone()
        self.model.load_state_dict(restored, strict=True)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
        """Train locally with SGD and return `(parameters, examples, metrics)`.

        ``proximal_mu`` is intentionally indexed rather than defaulted: FedProx
        callers must state its value, including the explicit FedAvg value of zero.
        """
        proximal_mu = float(config["proximal_mu"])
        local_epochs = int(config.get("local_epochs", 1))
        learning_rate = float(config.get("learning_rate", 0.01))
        if local_epochs < 1:
            raise ValueError("local_epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")

        self._set_parameters(parameters)
        reference_parameters = [parameter.detach().clone() for parameter in self.model.parameters()]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        tau = 0
        sample_count = 0
        self.model.train()
        for epoch in range(local_epochs):
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).squeeze().reshape(-1)
                optimizer.zero_grad()
                loss = criterion(self.model(inputs), labels)
                if proximal_mu:
                    proximal_term = sum(
                        torch.sum((parameter - reference) ** 2)
                        for parameter, reference in zip(self.model.parameters(), reference_parameters, strict=True)
                    )
                    loss = loss + (0.5 * proximal_mu * proximal_term)
                loss.backward()
                optimizer.step()
                tau += 1
                if epoch == 0:
                    sample_count += int(labels.numel())
        if tau == 0:
            raise ValueError("local training produced no batches, so tau is not positive")
        return self.get_parameters(), sample_count, {"tau": tau}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
        """Evaluate local validation data with a loss and macro-F1 metric."""
        del config
        self._set_parameters(parameters)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        sample_count = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).squeeze().reshape(-1)
                logits = self.model(inputs)
                total_loss += float(criterion(logits, labels).item())
                sample_count += int(labels.numel())
                y_true.extend(int(value) for value in labels.detach().cpu().reshape(-1).tolist())
                y_pred.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().tolist())
        if sample_count == 0:
            raise ValueError("validation loader is empty")
        return total_loss / sample_count, sample_count, {
            "macro_f1": macro_f1(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64))
        }
