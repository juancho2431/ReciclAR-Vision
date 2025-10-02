from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.data.dataset import WasteClassificationDataset
from src.data.split import create_stratified_split
from src.data.transforms import get_eval_transforms, get_train_transforms


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> Path:
    classes = ["paper", "plastic"]
    rng = np.random.default_rng(0)
    for cls in classes:
        class_dir = tmp_path / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            array = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
            Image.fromarray(array).save(class_dir / f"img_{i}.jpg")
    return tmp_path


def test_transforms_produce_tensor(tmp_dataset: Path) -> None:
    dataset = WasteClassificationDataset(tmp_dataset, transforms=get_train_transforms(image_size=64))
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape[1:] == (64, 64)


def test_eval_transform_is_deterministic(tmp_dataset: Path) -> None:
    dataset = WasteClassificationDataset(tmp_dataset, transforms=get_eval_transforms(image_size=64))
    sample_a = dataset[0]["image"]
    sample_b = dataset[0]["image"]
    assert torch.allclose(sample_a, sample_b)


def test_stratified_split(tmp_dataset: Path) -> None:
    split = create_stratified_split(tmp_dataset, train_size=0.5, val_size=0.25, seed=42)
    for part, groups in split.items():
        assert set(groups.keys()) == {"paper", "plastic"}


def test_dataset_from_split_json(tmp_dataset: Path, tmp_path: Path) -> None:
    split = create_stratified_split(tmp_dataset, train_size=0.5, val_size=0.25, seed=1)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    from src.data.dataset import WasteClassificationDataset

    train_paths = []
    for cls, paths in split["train"].items():
        train_paths.extend(paths)
    dataset = WasteClassificationDataset(tmp_dataset, file_paths=train_paths)
    assert len(dataset) == len(train_paths)
