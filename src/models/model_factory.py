"""Model factory for ReciclAR Vision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models


BackboneName = Literal["mobilenet_v3_small", "efficientnet_b0"]


@dataclass
class ModelConfig:
    num_classes: int
    backbone: BackboneName = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2


def create_model(config: ModelConfig) -> nn.Module:
    if config.backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if config.pretrained else None)
        in_features = model.classifier[3].in_features
        classifier = [
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, config.num_classes),
        ]
        model.classifier[3] = nn.Identity()
        model.classifier.extend(classifier)
    elif config.backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if config.pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, config.num_classes),
        )
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    return model


def freeze_backbone(model: nn.Module, train_classifier_only: bool = True) -> None:
    if not train_classifier_only:
        return
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
