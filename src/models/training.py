"""Training utilities for ReciclAR Vision."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model_factory import ModelConfig, create_model, freeze_backbone


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


@dataclass
class TrainingConfig:
    num_classes: int
    class_names: list[str]
    image_size: int
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_classifier_only: bool = True
    output_dir: str = "artifacts"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=1)
        self.best_score: float = -float("inf")
        self.best_path: Optional[Path] = None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, config.epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_validation(val_loader)
            self.scheduler.step(val_acc)
            tqdm.write(
                f"Epoch {epoch}/{config.epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
            )
            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_path = output_dir / "best_model.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": config.__dict__,
                    },
                    self.best_path,
                )
        if self.best_path is None:
            raise RuntimeError("Training finished without improving validation accuracy.")
        return self.best_path

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        for batch in tqdm(loader, leave=False):
            batch = _to_device(batch, self.device)
            logits = self.model(batch["image"])
            loss = self.criterion(logits, batch["label"])
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * batch["image"].size(0)
        return total_loss / len(loader.dataset)

    def _run_validation(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                logits = self.model(batch["image"])
                loss = self.criterion(logits, batch["label"])
                total_loss += loss.item() * batch["image"].size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch["label"]).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)


def build_trainer(config: TrainingConfig) -> Trainer:
    model_config = ModelConfig(
        num_classes=config.num_classes,
        backbone=config.backbone,  # type: ignore[arg-type]
        pretrained=config.pretrained,
        dropout=config.dropout,
    )
    model = create_model(model_config)
    freeze_backbone(model, train_classifier_only=config.train_classifier_only)
    return Trainer(model, config)
