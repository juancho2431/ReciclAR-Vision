"""Dataset utilities for ReciclAR Vision."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS


def _ensure_class_mapping(class_to_idx: Optional[Dict[str, int]], classes: Sequence[str]) -> Dict[str, int]:
    if class_to_idx is not None:
        return class_to_idx
    return {cls: idx for idx, cls in enumerate(sorted(classes))}


class WasteClassificationDataset(Dataset):
    """PyTorch ``Dataset`` reading images from either folders or explicit lists."""

    def __init__(
        self,
        root: str | Path,
        transforms: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        file_paths: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory '{self.root}' not found")
        if not self.root.is_dir():
            raise NotADirectoryError(f"'{self.root}' is not a directory")

        self.transforms = transforms
        if file_paths is not None:
            derived_classes = {Path(path).parent.name for path in file_paths}
            self.class_to_idx = _ensure_class_mapping(class_to_idx, derived_classes)
            self.samples = []
            for path in file_paths:
                rel_path = Path(path)
                label_name = rel_path.parent.name
                if label_name not in self.class_to_idx:
                    raise ValueError(f"Unknown class '{label_name}' in file list")
                absolute_path = (self.root / rel_path).resolve()
                self.samples.append((absolute_path, self.class_to_idx[label_name]))
        else:
            classes = [d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if not classes:
                raise ValueError(f"No class folders found in '{self.root}'")
            self.class_to_idx = _ensure_class_mapping(class_to_idx, classes)
            self.samples = []
            for cls, idx in self.class_to_idx.items():
                class_dir = self.root / cls
                if not class_dir.exists():
                    raise ValueError(f"Class '{cls}' does not exist in {self.root}")
                for file in sorted(class_dir.rglob("*")):
                    if file.is_file() and _is_image_file(file):
                        self.samples.append((file.resolve(), idx))
        if not self.samples:
            raise ValueError(f"No image files found in '{self.root}'")

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        if self.transforms is not None:
            transformed = self.transforms(image=image_np)
            tensor = transformed["image"]
        else:
            tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        return {"image": tensor, "label": torch.tensor(label, dtype=torch.long), "path": str(path)}

    @property
    def classes(self) -> List[str]:
        idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        return [idx_to_class[idx] for idx in range(len(idx_to_class))]
