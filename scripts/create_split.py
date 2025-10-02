"""CLI to create stratified dataset splits."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.split import class_distribution, create_stratified_split, save_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test split")
    parser.add_argument("data_dir", type=Path, help="Directory with class subfolders")
    parser.add_argument("output", type=Path, help="Path to save the split JSON")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = create_stratified_split(args.data_dir, args.train_size, args.val_size, args.seed)
    save_split(split, args.output)
    dist = class_distribution(split)
    print("Split saved to", args.output)
    for part, counter in dist.items():
        print(part, dict(counter))


if __name__ == "__main__":
    main()
