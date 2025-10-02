"""Inference helpers for serving ReciclAR Vision models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from ..data.transforms import get_eval_transforms
from .model_factory import ModelConfig, create_model


@dataclass
class InferenceConfig:
    num_classes: int
    backbone: str
    image_size: int
    device: str
    class_names: List[str]


class Predictor:
    def __init__(self, weights_path: str | Path) -> None:
        checkpoint = torch.load(weights_path, map_location="cpu")
        config_dict = checkpoint.get("config")
        if config_dict is None:
            raise KeyError("Checkpoint missing 'config' entry.")
        class_names = config_dict.get("class_names") or []
        if isinstance(class_names, tuple):
            class_names = list(class_names)
        self.config = InferenceConfig(
            num_classes=config_dict["num_classes"],
            backbone=config_dict.get("backbone", "efficientnet_b0"),
            image_size=config_dict.get("image_size", 300),
            device=config_dict.get("device", "cpu"),
            class_names=class_names,
        )
        model_config = ModelConfig(
            num_classes=self.config.num_classes,
            backbone=self.config.backbone,  # type: ignore[arg-type]
            pretrained=False,
        )
        self.model = create_model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transforms = get_eval_transforms(image_size=self.config.image_size)
        self.class_names: List[str] = self.config.class_names

    @torch.inference_mode()
    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0)
