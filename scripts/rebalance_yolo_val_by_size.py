"""Rebalance YOLO validation examples by object size bucket.

Moves complete image/label pairs between train and val only when --apply is used.
The test split is never touched.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_VAL_COUNTS = {"small": 2, "medium": 3, "large": 2}


@dataclass(frozen=True)
class Pair:
    split: str
    bucket: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance YOLO val split by size bucket.")
    parser.add_argument("--dry-run", action="store_true", help="Preview moves without changing files")
    parser.add_argument("--apply", action="store_true", help="Actually move selected pairs")
    return parser.parse_args()


def bucket_for_area(area: float) -> str:
    if area < 0.015:
        return "small"
    if area < 0.06:
        return "medium"
    return "large"


def first_label_bucket(label_path: Path) -> str | None:
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"WARNING: skipping malformed label: {label_path.relative_to(ROOT)}")
            return None
        try:
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            print(f"WARNING: skipping non-numeric label: {label_path.relative_to(ROOT)}")
            return None
        return bucket_for_area(width * height)
    print(f"WARNING: skipping empty label: {label_path.relative_to(ROOT)}")
    return None


def find_image(image_dir: Path, stem: str) -> Path | None:
    matches = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0] if matches else None


def load_pairs(split: str) -> list[Pair]:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    pairs: list[Pair] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = find_image(image_dir, label_path.stem)
        if image_path is None:
            continue
        bucket = first_label_bucket(label_path)
        if bucket is None:
            continue
        pairs.append(Pair(split=split, bucket=bucket, image_path=image_path, label_path=label_path))
    return sorted(pairs, key=lambda pair: pair.image_path.name)


def bucket_counts(pairs: list[Pair]) -> dict[str, int]:
    return {bucket: sum(1 for pair in pairs if pair.bucket == bucket) for bucket in TARGET_VAL_COUNTS}


def print_counts(title: str, train_pairs: list[Pair], val_pairs: list[Pair]) -> None:
    print(title)
    print("  train:")
    for bucket, count in bucket_counts(train_pairs).items():
        print(f"    {bucket}: {count}")
    print("  val:")
    for bucket, count in bucket_counts(val_pairs).items():
        print(f"    {bucket}: {count}")


def plan_moves(train_pairs: list[Pair], val_pairs: list[Pair]) -> tuple[list[Pair], list[Pair]]:
    train_to_val: list[Pair] = []
    val_to_train: list[Pair] = []

    for bucket, target in TARGET_VAL_COUNTS.items():
        current_val = [pair for pair in val_pairs if pair.bucket == bucket]
        if len(current_val) > target:
            val_to_train.extend(current_val[target:])
        elif len(current_val) < target:
            needed = target - len(current_val)
            available_train = [pair for pair in train_pairs if pair.bucket == bucket]
            train_to_val.extend(available_train[:needed])

    return train_to_val, val_to_train


def simulated_pairs(
    train_pairs: list[Pair], val_pairs: list[Pair], train_to_val: list[Pair], val_to_train: list[Pair]
) -> tuple[list[Pair], list[Pair]]:
    train_to_val_names = {pair.image_path.name for pair in train_to_val}
    val_to_train_names = {pair.image_path.name for pair in val_to_train}
    simulated_train = [pair for pair in train_pairs if pair.image_path.name not in train_to_val_names]
    simulated_val = [pair for pair in val_pairs if pair.image_path.name not in val_to_train_names]
    simulated_train.extend(val_to_train)
    simulated_val.extend(train_to_val)
    return simulated_train, simulated_val


def move_pair(pair: Pair, target_split: str) -> None:
    target_image_dir = DATASET_ROOT / "images" / target_split
    target_label_dir = DATASET_ROOT / "labels" / target_split
    target_image = target_image_dir / pair.image_path.name
    target_label = target_label_dir / pair.label_path.name
    if target_image.exists() or target_label.exists():
        raise FileExistsError(f"Refusing to overwrite existing pair for {pair.image_path.stem}")
    shutil.move(str(pair.image_path), target_image)
    shutil.move(str(pair.label_path), target_label)


def print_move_plan(title: str, pairs: list[Pair], target_split: str) -> None:
    print(title)
    if not pairs:
        print("  none")
        return
    for pair in pairs:
        print(f"  {pair.image_path.name} ({pair.bucket}) -> {target_split}")


def main() -> None:
    args = parse_args()
    if args.apply and args.dry_run:
        raise ValueError("Use either --apply or --dry-run, not both")

    train_pairs = load_pairs("train")
    val_pairs = load_pairs("val")
    print_counts("Before:", train_pairs, val_pairs)

    train_to_val, val_to_train = plan_moves(train_pairs, val_pairs)
    print_move_plan("\nPlanned train -> val moves:", train_to_val, "val")
    print_move_plan("\nPlanned val -> train moves:", val_to_train, "train")

    if args.apply:
        for pair in train_to_val:
            move_pair(pair, "val")
        for pair in val_to_train:
            move_pair(pair, "train")
        print("\nApplied rebalance moves.")
        after_train = load_pairs("train")
        after_val = load_pairs("val")
        print_counts("\nAfter:", after_train, after_val)
    else:
        simulated_train, simulated_val = simulated_pairs(train_pairs, val_pairs, train_to_val, val_to_train)
        print("\nDry run only. Re-run with --apply to move pairs.")
        print_counts("\nAfter (planned):", simulated_train, simulated_val)


if __name__ == "__main__":
    main()
