"""Run baseline golf ball detection across sampled video frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.detector import annotate_image, detect_ball  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline detection across video frames.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Stop after this many processed frames (default: 60)",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=960,
        help="Resize frames to this width before detection (default: 960)",
    )
    return parser.parse_args()


def detection_row(video_path: Path, frame_index: int, timestamp: float, det, frame_path: Path) -> dict:
    radius = 0.5 * max(det.w, det.h)
    return {
        "video_path": str(video_path),
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp, 3),
        "label": det.label,
        "confidence": float(det.confidence),
        "x": int(det.x),
        "y": int(det.y),
        "w": int(det.w),
        "h": int(det.h),
        "radius": round(float(radius), 2),
        "detector_type": det.detector_type,
        "selection_method": getattr(det, "selection_method", ""),
        "annotated_frame_path": str(frame_path),
    }


def empty_row(video_path: Path, frame_index: int, timestamp: float, frame_path: Path) -> dict:
    return {
        "video_path": str(video_path),
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp, 3),
        "label": "",
        "confidence": "",
        "x": "",
        "y": "",
        "w": "",
        "h": "",
        "radius": "",
        "detector_type": "",
        "selection_method": "",
        "annotated_frame_path": str(frame_path),
    }


def resize_frame(frame, target_width: int):
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame
    scale = target_width / float(frame.shape[1])
    target_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        raise ValueError("--every must be a positive integer")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer")

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_dir = ROOT / "output"
    frames_dir = output_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.resize_width > 0 and source_width > args.resize_width:
        writer_width = args.resize_width
        writer_height = int(source_height * (args.resize_width / float(source_width)))
    else:
        writer_width = source_width
        writer_height = source_height

    annotated_video_path = output_dir / f"{stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (writer_width, writer_height),
    )
    if not writer.isOpened():
        writer = None

    rows: list[dict] = []
    processed_frames = 0
    frame_index = 0

    while processed_frames < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        output_frame = resize_frame(frame, args.resize_width)
        if frame_index % args.every == 0:
            print(f"Processing frame {frame_index}...")
            timestamp = frame_index / fps
            resized_frame = resize_frame(frame, args.resize_width)
            detections = detect_ball(resized_frame)
            output_frame = annotate_image(resized_frame, detections)
            frame_path = frames_dir / f"{stem}_frame_{frame_index:06d}_annotated.jpg"
            cv2.imwrite(str(frame_path), output_frame)

            if detections:
                for det in detections:
                    rows.append(detection_row(video_path, frame_index, timestamp, det, frame_path))
            else:
                rows.append(empty_row(video_path, frame_index, timestamp, frame_path))
            processed_frames += 1

        if writer is not None:
            writer.write(output_frame)
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    json_path = output_dir / f"{stem}_video_detections.json"
    csv_path = output_dir / f"{stem}_video_detections.csv"
    summary = {
        "video_path": str(video_path),
        "processed_every_n_frames": args.every,
        "max_processed_frames": args.max_frames,
        "resize_width": args.resize_width,
        "total_frames": frame_index,
        "processed_frames": processed_frames,
        "annotated_video_path": str(annotated_video_path) if annotated_video_path.exists() else "",
        "detections": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "video_path",
        "frame_index",
        "timestamp_seconds",
        "label",
        "confidence",
        "x",
        "y",
        "w",
        "h",
        "radius",
        "detector_type",
        "selection_method",
        "annotated_frame_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

