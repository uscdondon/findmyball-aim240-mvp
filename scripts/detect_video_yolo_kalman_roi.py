"""Experimental YOLO + Kalman ROI golf-ball tracker.

This script is an optional experiment for controlled capstone evaluation. It is
not production-grade tracking; it combines full-frame YOLO recovery with a
Kalman-predicted region of interest to reduce frame-to-frame search area.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi"}
CSV_COLUMNS = [
    "frame_index",
    "source_name",
    "detected",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
    "detection_mode",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
]


@dataclass
class Detection:
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def box_width(self) -> float:
        return self.x2 - self.x1

    @property
    def box_height(self) -> float:
        return self.y2 - self.y1


@dataclass
class CropWindow:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1


@dataclass
class FrameResult:
    detection: Detection | None
    detection_mode: str
    crop: CropWindow | None
    predicted_center: tuple[float, float] | None


class KalmanTracker:
    """Small constant-velocity Kalman tracker around OpenCV's KalmanFilter."""

    def __init__(self) -> None:
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.filter.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.7
        self.filter.errorCovPost = np.eye(4, dtype=np.float32)
        self.active = False
        self.misses = 0

    def initialize(self, center_x: float, center_y: float) -> None:
        state = np.array([[center_x], [center_y], [0.0], [0.0]], dtype=np.float32)
        self.filter.statePre = state.copy()
        self.filter.statePost = state.copy()
        self.active = True
        self.misses = 0

    def predict(self) -> tuple[float, float]:
        prediction = self.filter.predict()
        return float(prediction[0, 0]), float(prediction[1, 0])

    def correct(self, center_x: float, center_y: float) -> None:
        if not self.active:
            self.initialize(center_x, center_y)
            return
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
        self.filter.correct(measurement)
        self.misses = 0

    def mark_miss(self, max_misses: int) -> None:
        self.misses += 1
        if self.misses > max_misses:
            self.active = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experimental YOLO + Kalman ROI tracking on a video or frame folder."
    )
    parser.add_argument("--source", required=True, help="Path to an input video or image folder.")
    parser.add_argument("--model", required=True, help="Path to trained YOLO model weights.")
    parser.add_argument("--output-dir", required=True, help="Directory for annotated outputs and metrics.")
    parser.add_argument("--conf", type=float, default=0.25, help="Full-frame YOLO confidence threshold.")
    parser.add_argument("--roi-conf", type=float, default=0.15, help="ROI YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--roi-size", type=int, default=512, help="Square ROI crop size in pixels.")
    parser.add_argument(
        "--max-misses-before-full",
        type=int,
        default=3,
        help="ROI misses allowed before trying full-frame recovery.",
    )
    parser.add_argument(
        "--full-every",
        type=int,
        default=8,
        help="Force full-frame detection every N frames while tracking.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    source = Path(args.source)
    model_path = Path(args.model)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.roi_size <= 0:
        raise ValueError("--roi-size must be positive")
    if args.max_misses_before_full < 1:
        raise ValueError("--max-misses-before-full must be at least 1")
    if args.full_every < 1:
        raise ValueError("--full-every must be at least 1")
    for name in ("conf", "roi_conf"):
        value = getattr(args, name)
        if value < 0 or value > 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def image_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.iterdir() if path.is_file() and is_image_file(path))


def clamp_detection(detection: Detection, width: int, height: int) -> Detection:
    return Detection(
        confidence=detection.confidence,
        x1=float(np.clip(detection.x1, 0, max(width - 1, 0))),
        y1=float(np.clip(detection.y1, 0, max(height - 1, 0))),
        x2=float(np.clip(detection.x2, 0, max(width - 1, 0))),
        y2=float(np.clip(detection.y2, 0, max(height - 1, 0))),
    )


def best_golf_ball_detection(model: YOLO, frame: np.ndarray, conf: float, imgsz: int) -> Detection | None:
    if frame.size == 0:
        return None

    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return None

    best: Detection | None = None
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id != 0:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detection = Detection(confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2)
        if best is None or confidence > best.confidence:
            best = detection

    if best is None:
        return None

    height, width = frame.shape[:2]
    return clamp_detection(best, width, height)


def crop_around(center: tuple[float, float], frame_shape: tuple[int, ...], roi_size: int) -> CropWindow:
    height, width = frame_shape[:2]
    half = roi_size / 2.0
    raw_x1 = int(round(center[0] - half))
    raw_y1 = int(round(center[1] - half))
    raw_x2 = raw_x1 + roi_size
    raw_y2 = raw_y1 + roi_size

    x1 = max(0, raw_x1)
    y1 = max(0, raw_y1)
    x2 = min(width, raw_x2)
    y2 = min(height, raw_y2)
    return CropWindow(x1=x1, y1=y1, x2=x2, y2=y2)


def map_crop_detection(detection: Detection, crop: CropWindow, frame_shape: tuple[int, ...]) -> Detection:
    mapped = Detection(
        confidence=detection.confidence,
        x1=detection.x1 + crop.x1,
        y1=detection.y1 + crop.y1,
        x2=detection.x2 + crop.x1,
        y2=detection.y2 + crop.y1,
    )
    height, width = frame_shape[:2]
    return clamp_detection(mapped, width, height)


def detect_in_crop(
    model: YOLO,
    frame: np.ndarray,
    crop: CropWindow,
    conf: float,
    imgsz: int,
) -> Detection | None:
    if not crop.is_valid:
        return None
    cropped = frame[crop.y1 : crop.y2, crop.x1 : crop.x2]
    detection = best_golf_ball_detection(model, cropped, conf, imgsz)
    if detection is None:
        return None
    return map_crop_detection(detection, crop, frame.shape)


def track_frame(
    frame: np.ndarray,
    frame_index: int,
    model: YOLO,
    tracker: KalmanTracker,
    conf: float,
    roi_conf: float,
    imgsz: int,
    roi_size: int,
    max_misses_before_full: int,
    full_every: int,
) -> FrameResult:
    predicted_center: tuple[float, float] | None = None
    crop: CropWindow | None = None
    detection: Detection | None = None
    detection_mode = "none"

    force_full = tracker.active and frame_index % full_every == 0
    if tracker.active:
        predicted_center = tracker.predict()
        crop = crop_around(predicted_center, frame.shape, roi_size)

    use_full_frame = not tracker.active or force_full or tracker.misses >= max_misses_before_full
    if use_full_frame:
        detection = best_golf_ball_detection(model, frame, conf, imgsz)
        detection_mode = "full" if detection is not None else "none"
    elif crop is not None:
        detection = detect_in_crop(model, frame, crop, roi_conf, imgsz)
        if detection is not None:
            detection_mode = "roi"
        else:
            projected_misses = tracker.misses + 1
            if projected_misses >= max_misses_before_full:
                detection = best_golf_ball_detection(model, frame, conf, imgsz)
                detection_mode = "full" if detection is not None else "none"

    if detection is not None:
        if tracker.active:
            tracker.correct(detection.center_x, detection.center_y)
        else:
            tracker.initialize(detection.center_x, detection.center_y)
    else:
        tracker.mark_miss(max_misses_before_full)

    return FrameResult(
        detection=detection,
        detection_mode=detection_mode,
        crop=crop,
        predicted_center=predicted_center,
    )


def empty_row(frame_index: int, source_name: str, mode: str, crop: CropWindow | None) -> dict[str, Any]:
    crop_values = crop_to_values(crop)
    return {
        "frame_index": frame_index,
        "source_name": source_name,
        "detected": False,
        "confidence": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "center_x": "",
        "center_y": "",
        "box_width": "",
        "box_height": "",
        "detection_mode": mode,
        **crop_values,
    }


def detection_row(
    frame_index: int,
    source_name: str,
    detection: Detection,
    mode: str,
    crop: CropWindow | None,
) -> dict[str, Any]:
    crop_values = crop_to_values(crop)
    return {
        "frame_index": frame_index,
        "source_name": source_name,
        "detected": True,
        "confidence": round(detection.confidence, 4),
        "x1": round(detection.x1, 2),
        "y1": round(detection.y1, 2),
        "x2": round(detection.x2, 2),
        "y2": round(detection.y2, 2),
        "center_x": round(detection.center_x, 2),
        "center_y": round(detection.center_y, 2),
        "box_width": round(detection.box_width, 2),
        "box_height": round(detection.box_height, 2),
        "detection_mode": mode,
        **crop_values,
    }


def crop_to_values(crop: CropWindow | None) -> dict[str, Any]:
    if crop is None:
        return {"crop_x1": "", "crop_y1": "", "crop_x2": "", "crop_y2": ""}
    return {
        "crop_x1": crop.x1,
        "crop_y1": crop.y1,
        "crop_x2": crop.x2,
        "crop_y2": crop.y2,
    }


def draw_path(frame: np.ndarray, centers: list[tuple[int, int]]) -> None:
    if len(centers) < 2:
        return
    for start, end in zip(centers[:-1], centers[1:]):
        cv2.line(frame, start, end, (0, 255, 255), 2)


def annotate_frame(
    frame: np.ndarray,
    result: FrameResult,
    centers: list[tuple[int, int]],
) -> np.ndarray:
    annotated = frame.copy()
    draw_path(annotated, centers)

    if result.crop is not None and result.crop.is_valid:
        cv2.rectangle(
            annotated,
            (result.crop.x1, result.crop.y1),
            (result.crop.x2, result.crop.y2),
            (255, 0, 0),
            2,
        )

    if result.predicted_center is not None:
        predicted = (
            int(round(result.predicted_center[0])),
            int(round(result.predicted_center[1])),
        )
        cv2.drawMarker(
            annotated,
            predicted,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    if result.detection is None:
        cv2.putText(
            annotated,
            "no detection",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    detection = result.detection
    x1 = int(round(detection.x1))
    y1 = int(round(detection.y1))
    x2 = int(round(detection.x2))
    y2 = int(round(detection.y2))
    center = (int(round(detection.center_x)), int(round(detection.center_y)))
    label = f"{result.detection_mode} {detection.confidence:.2f}"

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(annotated, center, 5, (0, 0, 255), -1)
    cv2.putText(
        annotated,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def rough_direction(delta_x: float | None, delta_y: float | None) -> str:
    if delta_x is None or delta_y is None:
        return "insufficient detections"
    distance = float(np.hypot(delta_x, delta_y))
    if distance < 5:
        return "minimal movement"

    x_direction = "right" if delta_x > 0 else "left"
    y_direction = "down" if delta_y > 0 else "up"
    abs_x = abs(delta_x)
    abs_y = abs(delta_y)

    if abs_x >= abs_y * 2:
        return f"mostly {x_direction}"
    if abs_y >= abs_x * 2:
        return f"mostly {y_direction}"
    return f"{y_direction}-{x_direction}"


def build_summary(
    rows: list[dict[str, Any]],
    source: Path,
    model_path: Path,
    conf: float,
    roi_conf: float,
    roi_size: int,
    max_misses_before_full: int,
    full_every: int,
) -> dict[str, Any]:
    detected_rows = [row for row in rows if row["detected"]]
    confidences = [float(row["confidence"]) for row in detected_rows]

    start_center = None
    end_center = None
    delta_x = None
    delta_y = None
    if detected_rows:
        first = detected_rows[0]
        last = detected_rows[-1]
        start_center = [float(first["center_x"]), float(first["center_y"])]
        end_center = [float(last["center_x"]), float(last["center_y"])]
        delta_x = end_center[0] - start_center[0]
        delta_y = end_center[1] - start_center[1]

    total_frames = len(rows)
    frames_with_detection = len(detected_rows)
    frames_without_detection = total_frames - frames_with_detection

    return {
        "source": str(source),
        "model": str(model_path),
        "confidence_threshold": conf,
        "roi_confidence_threshold": roi_conf,
        "roi_size": roi_size,
        "max_misses_before_full": max_misses_before_full,
        "full_every": full_every,
        "total_frames_processed": total_frames,
        "frames_with_detection": frames_with_detection,
        "frames_without_detection": frames_without_detection,
        "detection_rate": round(frames_with_detection / total_frames, 4) if total_frames else 0,
        "first_detection_frame": int(detected_rows[0]["frame_index"]) if detected_rows else None,
        "last_detection_frame": int(detected_rows[-1]["frame_index"]) if detected_rows else None,
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "max_confidence": round(max(confidences), 4) if confidences else None,
        "min_confidence": round(min(confidences), 4) if confidences else None,
        "full_frame_detections": sum(1 for row in rows if row["detection_mode"] == "full"),
        "roi_detections": sum(1 for row in rows if row["detection_mode"] == "roi"),
        "start_center": start_center,
        "end_center": end_center,
        "delta_x": round(delta_x, 2) if delta_x is not None else None,
        "delta_y": round(delta_y, 2) if delta_y is not None else None,
        "rough_direction": rough_direction(delta_x, delta_y),
        "notes": (
            "Experimental YOLO + Kalman ROI tracker for controlled capstone evaluation. "
            "Full-frame YOLO is used for initialization and recovery; ROI detections are "
            "mapped back to full-frame coordinates. This is not production-grade tracking."
        ),
    }


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_frame(
    frame: np.ndarray,
    frame_index: int,
    source_name: str,
    model: YOLO,
    tracker: KalmanTracker,
    centers: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    result = track_frame(
        frame=frame,
        frame_index=frame_index,
        model=model,
        tracker=tracker,
        conf=args.conf,
        roi_conf=args.roi_conf,
        imgsz=args.imgsz,
        roi_size=args.roi_size,
        max_misses_before_full=args.max_misses_before_full,
        full_every=args.full_every,
    )

    if result.detection is None:
        row = empty_row(frame_index, source_name, result.detection_mode, result.crop)
    else:
        center = (
            int(round(result.detection.center_x)),
            int(round(result.detection.center_y)),
        )
        centers.append(center)
        row = detection_row(
            frame_index,
            source_name,
            result.detection,
            result.detection_mode,
            result.crop,
        )

    return row, annotate_frame(frame, result, centers)


def process_image_folder(source: Path, model: YOLO, output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    annotated_dir = output_dir / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    paths = image_files(source)
    if not paths:
        raise RuntimeError(f"No .jpg, .jpeg, or .png images found in folder: {source}")

    tracker = KalmanTracker()
    rows: list[dict[str, Any]] = []
    centers: list[tuple[int, int]] = []
    for frame_index, image_path in enumerate(paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            rows.append(empty_row(frame_index, image_path.name, "none", None))
            continue

        row, annotated = process_frame(frame, frame_index, image_path.name, model, tracker, centers, args)
        rows.append(row)
        out_path = annotated_dir / f"{frame_index:06d}_{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)

    return rows


def process_video(source: Path, model: YOLO, output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    annotated_dir = output_dir / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    writer_fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotated_video_path = output_dir / "annotated_video.mp4"

    writer = None
    if width > 0 and height > 0:
        writer = cv2.VideoWriter(
            str(annotated_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            writer_fps,
            (width, height),
        )
        if not writer.isOpened():
            print("Could not open video writer; annotated_video.mp4 will not be saved.")
            writer = None

    tracker = KalmanTracker()
    rows: list[dict[str, Any]] = []
    centers: list[tuple[int, int]] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        row, annotated = process_frame(frame, frame_index, source.name, model, tracker, centers, args)
        rows.append(row)

        if writer is not None:
            writer.write(annotated)
        frame_path = annotated_dir / f"{source.stem}_frame_{frame_index:06d}_annotated.jpg"
        cv2.imwrite(str(frame_path), annotated)
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    return rows


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print("Experimental YOLO + Kalman ROI tracking complete.")
    print(f"Frames processed: {summary['total_frames_processed']}")
    print(f"Frames with detection: {summary['frames_with_detection']}")
    print(f"Detection rate: {summary['detection_rate']}")
    print(f"Full-frame detections: {summary['full_frame_detections']}")
    print(f"ROI detections: {summary['roi_detections']}")
    print(f"Rough direction: {summary['rough_direction']}")
    print(f"Wrote CSV: {output_dir / 'detections.csv'}")
    print(f"Wrote summary: {output_dir / 'trajectory_summary.json'}")
    print(f"Wrote annotated frames folder: {output_dir / 'annotated_frames'}")


def supported_source(source: Path) -> bool:
    return source.is_dir() or (source.is_file() and is_video_file(source))


def main() -> None:
    args = parse_args()
    validate_args(args)

    source = Path(args.source)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    if not supported_source(source):
        raise ValueError(f"Source must be a video file or folder of images: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    if source.is_dir():
        rows = process_image_folder(source, model, output_dir, args)
    else:
        rows = process_video(source, model, output_dir, args)

    summary = build_summary(
        rows=rows,
        source=source,
        model_path=model_path,
        conf=args.conf,
        roi_conf=args.roi_conf,
        roi_size=args.roi_size,
        max_misses_before_full=args.max_misses_before_full,
        full_every=args.full_every,
    )

    csv_path = output_dir / "detections.csv"
    summary_path = output_dir / "trajectory_summary.json"
    write_csv(rows, csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
