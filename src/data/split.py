"""Utilities to create stratified train/val/test splits."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.model_selection import train_test_split

from .dataset import ALLOWED_EXTENSIONS


Split = Dict[str, List[str]]


def _gather_samples(root: Path) -> Tuple[List[Path], List[str]]:
    images: List[Path] = []
    labels: List[str] = []
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS:
                images.append(image_path.relative_to(root))
                labels.append(class_dir.name)
    if not images:
        raise ValueError(f"No images found in '{root}'.")
    return images, labels


def create_stratified_split(
    root: str | Path,
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 17,
) -> Dict[str, Split]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root)

    images, labels = _gather_samples(root_path)
    test_size = 1.0 - train_size
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    relative_val_size = val_size / (1 - train_size)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images,
        temp_labels,
        test_size=1 - relative_val_size,
        stratify=temp_labels,
        random_state=seed,
    )

    def group_by_class(items: Sequence[Path], classes: Sequence[str]) -> Split:
        grouped: Split = {}
        for path, label in zip(items, classes):
            grouped.setdefault(label, []).append(str(path))
        return grouped

    return {
        "train": group_by_class(train_images, train_labels),
        "val": group_by_class(val_images, val_labels),
        "test": group_by_class(test_images, test_labels),
    }


def save_split(split: Dict[str, Split], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)


def class_distribution(split: Dict[str, Split]) -> Dict[str, Counter]:
    return {
        part: Counter({cls: len(paths) for cls, paths in grouped.items()})
        for part, grouped in split.items()
    }
