"""Small RGB classifier used by the centralized gate and offline clients."""

from __future__ import annotations

from torch import Tensor, nn


class FedScopeCNN(nn.Module):
    """A compact CNN that maps 28px RGB images to class logits."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs).flatten(1))
