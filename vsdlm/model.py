from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Lightweight CNN configuration for VSDLM."""

    base_channels: int = 32
    num_blocks: int = 4
    dropout: float = 0.3


class _SepConvBlock(nn.Module):
    """Depthwise separable convolution block with optional residual."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_residual:
            x = x + identity
        x = F.relu(x, inplace=True)
        return x


class VSDLM(nn.Module):
    """Compact mouth state classifier that outputs logits."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        base = self.config.base_channels
        num_blocks = max(1, self.config.num_blocks)

        stems = [
            nn.Conv2d(3, base, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        ]
        self.stem = nn.Sequential(*stems)

        channels = base
        blocks = []
        for idx in range(num_blocks):
            stride = 2 if idx % 2 == 0 and idx > 0 else 1
            next_channels = channels * (2 if stride == 2 else 1)
            blocks.append(_SepConvBlock(channels, next_channels, stride=stride))
            channels = next_channels

        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity(),
            nn.Linear(channels, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        logits = self.head(x)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
