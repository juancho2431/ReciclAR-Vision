"""Image transformation utilities for ReciclAR Vision datasets."""
from __future__ import annotations

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _normalize_stats(dataset: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return mean and std tuples for supported datasets.

    Parameters
    ----------
    dataset:
        Name of the dataset. Currently only ``"reciclar"`` is supported, but
        the function is extensible to make it explicit where statistics are
        defined.
    """

    if dataset != "reciclar":
        raise ValueError(f"Unsupported dataset '{dataset}'.")
    # Statistics based on ImageNet to leverage pre-trained backbones.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return imagenet_mean, imagenet_std


def get_train_transforms(
    image_size: int = 300,
    dataset: str = "reciclar",
) -> A.Compose:
    """Return the data augmentation pipeline used during training.

    The transformations follow the requirements defined in the project plan:
    - Resize to the desired ``image_size`` keeping the aspect ratio by padding.
    - Apply horizontal flips, rotations, colour jittering and slight blur to
      simulate variations in the capture conditions.
    - Normalise using ImageNet statistics so that pre-trained backbones work
      as expected.
    """

    mean, std = _normalize_stats(dataset)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.8,
            ),
            A.RandomBrightnessContrast(p=0.7),
            A.HueSaturationValue(p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(
    image_size: int = 300,
    dataset: str = "reciclar",
) -> A.Compose:
    """Return deterministic transforms for validation/test/inference."""

    mean, std = _normalize_stats(dataset)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
