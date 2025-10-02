"""Training entry point for ReciclAR Vision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import DataLoader

from src.data.dataset import WasteClassificationDataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.training import TrainingConfig, build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReciclAR Vision model")
    parser.add_argument("data_dir", type=Path, help="Directorio raíz con las carpetas de clases")
    parser.add_argument("--split-json", type=Path, help="Ruta a un JSON con la asignación de imágenes a train/val/test")
    parser.add_argument("--backbone", choices=["mobilenet_v3_small", "efficientnet_b0"], default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-classifier-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_datasets(
    data_dir: Path,
    image_size: int,
    split_json: Path | None,
) -> tuple[WasteClassificationDataset, WasteClassificationDataset]:
    if split_json is None:
        train_dataset = WasteClassificationDataset(
            data_dir / "train",
            transforms=get_train_transforms(image_size=image_size),
        )
        val_dataset = WasteClassificationDataset(
            data_dir / "val",
            transforms=get_eval_transforms(image_size=image_size),
            class_to_idx={cls: idx for idx, cls in enumerate(train_dataset.classes)},
        )
        return train_dataset, val_dataset

    split_data: Dict[str, Dict[str, List[str]]] = json.loads(split_json.read_text(encoding="utf-8"))
    classes = sorted(split_data["train"].keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def expand_paths(section: str) -> List[str]:
        expanded: List[str] = []
        for cls, paths in split_data[section].items():
            for rel_path in paths:
                expanded.append(str((data_dir / Path(rel_path)).resolve()))
        return expanded

    train_paths = expand_paths("train")
    val_paths = expand_paths("val")

    train_dataset = WasteClassificationDataset(
        data_dir,
        transforms=get_train_transforms(image_size=image_size),
        class_to_idx=class_to_idx,
        file_paths=train_paths,
    )
    val_dataset = WasteClassificationDataset(
        data_dir,
        transforms=get_eval_transforms(image_size=image_size),
        class_to_idx=class_to_idx,
        file_paths=val_paths,
    )
    return train_dataset, val_dataset


def main() -> None:
    args = parse_args()
    train_dataset, val_dataset = build_datasets(args.data_dir, args.image_size, args.split_json)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    config = TrainingConfig(
        num_classes=len(train_dataset.classes),
        class_names=train_dataset.classes,
        image_size=args.image_size,
        backbone=args.backbone,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=str(args.output_dir),
        device=args.device,
        train_classifier_only=args.train_classifier_only,
    )
    trainer = build_trainer(config)
    best_path = trainer.fit(train_loader, val_loader)
    print("Best model saved to", best_path)


if __name__ == "__main__":
    main()
