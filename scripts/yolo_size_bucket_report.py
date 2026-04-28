"""Report YOLO object size buckets by dataset split."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "yolo"
SPLITS = ("train", "val", "test")


def bucket_for_area(area: float) -> str:
    if area < 0.015:
        return "small"
    if area < 0.06:
        return "medium"
    return "large"


def parse_label_file(label_path: Path) -> list[tuple[float, str]]:
    objects: list[tuple[float, str]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"WARNING: skipping malformed line {label_path.relative_to(ROOT)}:{line_number}")
            continue
        try:
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            print(f"WARNING: skipping non-numeric line {label_path.relative_to(ROOT)}:{line_number}")
            continue
        area = width * height
        objects.append((area, bucket_for_area(area)))
    return objects


def main() -> None:
    val_buckets_present: set[str] = set()

    print(f"YOLO dataset root: {DATASET_ROOT}")
    print("Buckets: small < 0.015, medium 0.015-0.06, large >= 0.06")

    for split in SPLITS:
        label_dir = DATASET_ROOT / "labels" / split
        bucket_files: dict[str, list[str]] = defaultdict(list)
        bucket_counts: dict[str, int] = {"small": 0, "medium": 0, "large": 0}

        for label_path in sorted(label_dir.glob("*.txt")):
            for area, bucket in parse_label_file(label_path):
                bucket_counts[bucket] += 1
                bucket_files[bucket].append(f"{label_path.stem} (area={area:.4f})")

        if split == "val":
            val_buckets_present = {bucket for bucket, count in bucket_counts.items() if count > 0}

        print(f"\n{split}:")
        for bucket in ("small", "medium", "large"):
            print(f"  {bucket}: {bucket_counts[bucket]}")
            for filename in bucket_files[bucket]:
                print(f"    - {filename}")

    if len(val_buckets_present) == 1:
        only_bucket = next(iter(val_buckets_present))
        print(f"\nWARNING: validation set only contains one size bucket: {only_bucket}")
    elif not val_buckets_present:
        print("\nWARNING: validation set has no labeled objects.")


if __name__ == "__main__":
    main()
