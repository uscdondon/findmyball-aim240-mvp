"""Seed YOLO image folders from still images or sampled video frames.

This script prepares images for later manual labeling. It does not create
YOLO label files and does not create fake annotations.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}
VALID_SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed YOLO images from images or videos.")
    parser.add_argument("inputs", nargs="+", help="Image or video paths to add")
    parser.add_argument("--every", type=int, default=15, help="For videos, save every Nth frame")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to save per video")
    parser.add_argument("--split", default="train", choices=sorted(VALID_SPLITS), help="YOLO split")
    return parser.parse_args()


def clean_stem(path: Path) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in path.stem).strip("_")


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter:03d}{suffix}"
        counter += 1
    return candidate


def copy_image(path: Path, output_dir: Path) -> int:
    out_path = unique_path(output_dir, clean_stem(path), path.suffix.lower())
    shutil.copy2(path, out_path)
    print(f"Added image: {out_path.relative_to(ROOT)}")
    return 1


def extract_video_frames(path: Path, output_dir: Path, every: int, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"WARNING: could not open video: {path}")
        return 0

    added = 0
    frame_index = 0
    stem = clean_stem(path)
    while added < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % every == 0:
            out_path = unique_path(output_dir, f"{stem}_frame_{frame_index:06d}", ".jpg")
            cv2.imwrite(str(out_path), frame)
            print(f"Added frame: {out_path.relative_to(ROOT)}")
            added += 1
        frame_index += 1

    cap.release()
    return added


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        raise ValueError("--every must be a positive integer")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer")

    output_dir = DATASET_ROOT / "images" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    total_added = 0
    for raw_input in args.inputs:
        path = Path(raw_input)
        if not path.exists():
            print(f"WARNING: missing input: {path}")
            continue

        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            total_added += copy_image(path, output_dir)
        elif suffix in VIDEO_EXTENSIONS:
            total_added += extract_video_frames(path, output_dir, args.every, args.max_frames)
        else:
            print(f"WARNING: unsupported file type: {path}")

    print(f"Done. Added {total_added} image(s) to {output_dir.relative_to(ROOT)}.")
    print("No YOLO labels were created. Label files must be added manually or exported from a labeling tool.")


if __name__ == "__main__":
    main()
