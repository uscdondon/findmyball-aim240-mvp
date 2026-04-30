from pathlib import Path
from collections import defaultdict
import os

ROOT = Path(".").resolve()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".HEIC", ".JPG", ".JPEG", ".PNG"}
VIDEO_EXTS = {".mov", ".MOV", ".mp4", ".MP4"}
LABEL_EXTS = {".txt"}

IGNORE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
}

def is_ignored(path: Path) -> bool:
    return any(part in IGNORE_PARTS for part in path.parts)

def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)

def category(path: Path) -> str:
    p = str(path)

    if p.startswith("raw_media/"):
        return "raw_media_originals"
    if p.startswith("input/"):
        return "input_converted_or_detector_inputs"
    if p.startswith("videos/"):
        return "videos"
    if p.startswith("data/yolo_v2/"):
        if "/images/" in p:
            return "yolo_v2_images"
        if "/labels/" in p:
            return "yolo_v2_labels"
        return "yolo_v2_other"
    if p.startswith("data/yolo/"):
        if "/images/" in p:
            return "yolo_v1_images"
        if "/labels/" in p:
            return "yolo_v1_labels"
        return "yolo_v1_other"
    if p.startswith("evidence/"):
        return "evidence"
    if p.startswith("runs/"):
        return "runs_training_outputs"
    if p.startswith("tmp/"):
        return "tmp_scratch"
    if p.startswith("output/"):
        return "output_detector_outputs"
    return "other"

files = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT)
    if is_ignored(rel):
        continue

    suffix = path.suffix
    if suffix in IMAGE_EXTS or suffix in VIDEO_EXTS or suffix in LABEL_EXTS:
        files.append(rel)

by_cat = defaultdict(list)
for rel in sorted(files):
    by_cat[category(rel)].append(rel)

report_path = ROOT / "reports" / "image_inventory.md"

with report_path.open("w") as f:
    f.write("# FindMyBall Image / Video / Label Inventory\n\n")
    f.write("Generated from project root.\n\n")

    total_size = sum(size_mb(ROOT / rel) for rel in files)
    f.write(f"Total tracked/scanned media-like files: {len(files)}\n\n")
    f.write(f"Approx total size: {total_size:.2f} MB\n\n")

    f.write("## Summary by Category\n\n")
    f.write("| Category | Files | Size MB |\n")
    f.write("|---|---:|---:|\n")
    for cat in sorted(by_cat):
        cat_files = by_cat[cat]
        cat_size = sum(size_mb(ROOT / rel) for rel in cat_files)
        f.write(f"| {cat} | {len(cat_files)} | {cat_size:.2f} |\n")

    f.write("\n## Detailed Inventory\n\n")
    for cat in sorted(by_cat):
        f.write(f"## {cat}\n\n")
        f.write("| File | Ext | Size MB |\n")
        f.write("|---|---:|---:|\n")
        for rel in by_cat[cat]:
            f.write(f"| `{rel}` | `{rel.suffix}` | {size_mb(ROOT / rel):.2f} |\n")
        f.write("\n")

print(f"Wrote inventory report: {report_path}")
print()
print("Summary:")
for cat in sorted(by_cat):
    cat_files = by_cat[cat]
    cat_size = sum(size_mb(ROOT / rel) for rel in cat_files)
    print(f"{cat:35} files={len(cat_files):4d} size={cat_size:8.2f} MB")
