"""Audit YOLO image/label pairs and bounding-box annotations."""

from __future__ import annotations

from pathlib import Path
from statistics import mean


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


def warn(message: str) -> None:
    print(f"  WARNING: {message}")


def parse_label_file(label_path: Path) -> tuple[list[float], list[float], list[float], int]:
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    warnings = 0

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            warn(f"{label_path.relative_to(ROOT)}:{line_number} malformed line: expected 5 values")
            warnings += 1
            continue

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            warn(f"{label_path.relative_to(ROOT)}:{line_number} malformed numeric value")
            warnings += 1
            continue

        if class_id != 0:
            warn(f"{label_path.relative_to(ROOT)}:{line_number} class_id is {class_id}, expected 0")
            warnings += 1

        if any(value < 0 or value > 1 for value in values):
            warn(f"{label_path.relative_to(ROOT)}:{line_number} normalized value outside 0..1")
            warnings += 1

        width = values[2]
        height = values[3]
        widths.append(width)
        heights.append(height)
        areas.append(width * height)

    return widths, heights, areas, warnings


def print_stats(title: str, values: list[float]) -> None:
    if not values:
        print(f"  {title}: no boxes")
        return
    print(
        f"  {title}: min={min(values):.4f}, max={max(values):.4f}, "
        f"mean={mean(values):.4f}"
    )


def main() -> None:
    all_widths: list[float] = []
    all_heights: list[float] = []
    all_areas: list[float] = []
    total_warnings = 0

    print(f"YOLO dataset root: {DATASET_ROOT}")
    print("Expected class 0: golf_ball")

    for split in SPLITS:
        images = image_files(split)
        labels = label_files(split)
        image_stems = {image.stem for image in images}
        label_stems = {label.stem for label in labels}

        missing_labels = [image for image in images if image.stem not in label_stems]
        missing_images = [label for label in labels if label.stem not in image_stems]

        print(f"\n{split}:")
        print(f"  images: {len(images)}")
        print(f"  labels: {len(labels)}")
        print(f"  images missing labels: {len(missing_labels)}")
        print(f"  labels missing images: {len(missing_images)}")

        for image in missing_labels:
            warn(f"missing label for {image.relative_to(ROOT)}")
            total_warnings += 1
        for label in missing_images:
            warn(f"missing image for {label.relative_to(ROOT)}")
            total_warnings += 1

        split_widths: list[float] = []
        split_heights: list[float] = []
        split_areas: list[float] = []
        for label in labels:
            widths, heights, areas, warnings = parse_label_file(label)
            split_widths.extend(widths)
            split_heights.extend(heights)
            split_areas.extend(areas)
            total_warnings += warnings

        print_stats("bbox width", split_widths)
        print_stats("bbox height", split_heights)
        print_stats("bbox area", split_areas)

        all_widths.extend(split_widths)
        all_heights.extend(split_heights)
        all_areas.extend(split_areas)

    print("\nOverall bounding-box statistics:")
    print_stats("bbox width", all_widths)
    print_stats("bbox height", all_heights)
    print_stats("bbox area", all_areas)

    if total_warnings:
        print(f"\nDone with warnings: {total_warnings}")
    else:
        print("\nDone: dataset audit passed with no warnings.")


if __name__ == "__main__":
    main()
