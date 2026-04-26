"""Validate the future YOLO dataset scaffold.

YOLO labels must be created manually or exported from a labeling tool.
This script only counts files and warns about images missing labels.
It does not create fake labels.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def image_files(split: str) -> list[Path]:
    image_dir = DATASET_ROOT / "images" / split
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_files(split: str) -> list[Path]:
    label_dir = DATASET_ROOT / "labels" / split
    return sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt")


def main() -> None:
    total_missing = 0
    print(f"YOLO dataset root: {DATASET_ROOT}")
    print("Class 0: golf_ball")

    for split in SPLITS:
        images = image_files(split)
        labels = label_files(split)
        label_names = {label.stem for label in labels}
        missing = [image for image in images if image.stem not in label_names]
        total_missing += len(missing)

        print(f"\n{split}:")
        print(f"  images: {len(images)}")
        print(f"  labels: {len(labels)}")
        print(f"  missing labels: {len(missing)}")
        for image in missing:
            expected = DATASET_ROOT / "labels" / split / f"{image.stem}.txt"
            print(f"  WARNING: missing label for {image.relative_to(ROOT)} -> {expected.relative_to(ROOT)}")

    if total_missing:
        print(f"\nDone with warnings: {total_missing} image(s) are missing labels.")
    else:
        print("\nDone: every image has a matching label file.")


if __name__ == "__main__":
    main()
