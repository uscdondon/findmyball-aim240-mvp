"""Extract every Nth frame from a video into output/frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Save every Nth frame (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        raise ValueError("--every must be a positive integer")

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames_dir = ROOT / "output" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_count = 0
    stem = video_path.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.every == 0:
            out_path = frames_dir / f"{stem}_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
        frame_index += 1

    cap.release()
    print(
        f"Done. Saved {saved_count} frame(s) to {frames_dir} "
        f"from {frame_index} total frame(s)."
    )


if __name__ == "__main__":
    main()

