"""Split labeled YOLO training pairs into train/val folders.

Moves only complete image-label pairs and only when --apply is provided.
Use --dry-run to preview the selected files without moving anything.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move YOLO train pairs into validation split.")
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Validation ratio (default: 0.25)")
    parser.add_argument("--dry-run", action="store_true", help="Preview files without moving them")
    parser.add_argument("--apply", action="store_true", help="Actually move selected files")
    return parser.parse_args()


def count_files(split: str) -> tuple[int, int]:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    image_count = sum(
        1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_count = sum(1 for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt")
    return image_count, label_count


def train_pairs() -> list[tuple[Path, Path]]:
    image_dir = DATASET_ROOT / "images" / "train"
    label_dir = DATASET_ROOT / "labels" / "train"
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def print_counts(title: str) -> None:
    train_images, train_labels = count_files("train")
    val_images, val_labels = count_files("val")
    print(title)
    print(f"  train images: {train_images}")
    print(f"  train labels: {train_labels}")
    print(f"  val images: {val_images}")
    print(f"  val labels: {val_labels}")


def main() -> None:
    args = parse_args()
    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.apply and args.dry_run:
        raise ValueError("Use either --apply or --dry-run, not both")

    print_counts("Before:")
    pairs = train_pairs()
    move_count = max(1, round(len(pairs) * args.val_ratio)) if pairs else 0
    rng = random.Random(RANDOM_SEED)
    selected = sorted(rng.sample(pairs, move_count), key=lambda pair: pair[0].name)

    print(f"\nComplete train image-label pairs: {len(pairs)}")
    print(f"Selected for validation: {len(selected)}")
    for image_path, label_path in selected:
        print(f"  {image_path.name} + {label_path.name}")

    if args.apply:
        val_image_dir = DATASET_ROOT / "images" / "val"
        val_label_dir = DATASET_ROOT / "labels" / "val"
        val_image_dir.mkdir(parents=True, exist_ok=True)
        val_label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in selected:
            shutil.move(str(image_path), val_image_dir / image_path.name)
            shutil.move(str(label_path), val_label_dir / label_path.name)
        print("\nMoved selected pairs to validation split.")
    else:
        print("\nDry run only. Re-run with --apply to move selected pairs.")

    print_counts("\nAfter:")


if __name__ == "__main__":
    main()
