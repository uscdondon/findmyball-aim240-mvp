"""Detect and lightly track a golf ball across video frames with YOLOv8."""

import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_MODEL = "runs/detect/runs/findmyball/yolo_v2_clean_batch_01/weights/best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi"}
CSV_COLUMNS = [
    "frame_index",
    "timestamp_seconds",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO golf-ball detection and simple centroid tracking on video or frames."
    )
    parser.add_argument("--source", required=True, help="Path to an input video or image folder.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to trained YOLO model weights.")
    parser.add_argument("--output-dir", default="output/yolo_video_tracking")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.4, help="YOLO IoU threshold.")
    parser.add_argument(
        "--save-every-frame",
        action="store_true",
        help="In video mode, also save each annotated frame as a JPG.",
    )
    return parser.parse_args()


def validate_paths(source: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def image_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.iterdir() if path.is_file() and is_image_file(path))


def best_golf_ball_detection(model: YOLO, frame, conf: float, iou: float) -> dict | None:
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)
    if not results:
        return None

    best = None
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id != 0:
            continue

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detection = {
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
            "box_width": x2 - x1,
            "box_height": y2 - y1,
        }
        if best is None or confidence > best["confidence"]:
            best = detection

    return best


def empty_row(frame_index: int, timestamp_seconds, source_name: str) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
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
    }


def detection_row(frame_index: int, timestamp_seconds, source_name: str, detection: dict) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "source_name": source_name,
        "detected": True,
        "confidence": round(detection["confidence"], 4),
        "x1": round(detection["x1"], 2),
        "y1": round(detection["y1"], 2),
        "x2": round(detection["x2"], 2),
        "y2": round(detection["y2"], 2),
        "center_x": round(detection["center_x"], 2),
        "center_y": round(detection["center_y"], 2),
        "box_width": round(detection["box_width"], 2),
        "box_height": round(detection["box_height"], 2),
    }


def draw_path(frame, centers: list[tuple[int, int]]) -> None:
    if len(centers) < 2:
        return
    for start, end in zip(centers[:-1], centers[1:]):
        cv2.line(frame, start, end, (0, 255, 255), 2)


def annotate_frame(frame, detection: dict | None, centers: list[tuple[int, int]]):
    annotated = frame.copy()
    draw_path(annotated, centers)

    if detection is None:
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

    x1 = int(round(detection["x1"]))
    y1 = int(round(detection["y1"]))
    x2 = int(round(detection["x2"]))
    y2 = int(round(detection["y2"]))
    center = (int(round(detection["center_x"])), int(round(detection["center_y"])))
    label = f"golf_ball {detection['confidence']:.2f}"

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
    if (delta_x * delta_x + delta_y * delta_y) ** 0.5 < 5:
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


def build_summary(rows: list[dict], source: Path, model_path: Path, conf: float) -> dict:
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
        "total_frames_processed": total_frames,
        "frames_with_detection": frames_with_detection,
        "frames_without_detection": frames_without_detection,
        "detection_rate": round(frames_with_detection / total_frames, 4) if total_frames else 0,
        "first_detection_frame": int(detected_rows[0]["frame_index"]) if detected_rows else None,
        "last_detection_frame": int(detected_rows[-1]["frame_index"]) if detected_rows else None,
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "max_confidence": round(max(confidences), 4) if confidences else None,
        "min_confidence": round(min(confidences), 4) if confidences else None,
        "start_center": start_center,
        "end_center": end_center,
        "delta_x": round(delta_x, 2) if delta_x is not None else None,
        "delta_y": round(delta_y, 2) if delta_y is not None else None,
        "rough_direction": rough_direction(delta_x, delta_y),
        "notes": (
            "Simple frame-by-frame YOLO centroid tracking for controlled iPhone putt/chip MVP "
            "testing. This is not production-grade tracking."
        ),
    }


def write_csv(rows: list[dict], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_image_folder(source: Path, model: YOLO, output_dir: Path, conf: float, iou: float) -> list[dict]:
    annotated_dir = output_dir / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    paths = image_files(source)
    if not paths:
        raise RuntimeError(f"No .jpg, .jpeg, or .png images found in folder: {source}")

    rows = []
    centers: list[tuple[int, int]] = []
    for frame_index, image_path in enumerate(paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            rows.append(empty_row(frame_index, "", image_path.name))
            continue

        detection = best_golf_ball_detection(model, frame, conf, iou)
        if detection is None:
            rows.append(empty_row(frame_index, "", image_path.name))
        else:
            center = (int(round(detection["center_x"])), int(round(detection["center_y"])))
            centers.append(center)
            rows.append(detection_row(frame_index, "", image_path.name, detection))

        annotated = annotate_frame(frame, detection, centers)
        out_path = annotated_dir / f"{frame_index:06d}_{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)

    return rows


def process_video(
    source: Path,
    model: YOLO,
    output_dir: Path,
    conf: float,
    iou: float,
    save_every_frame: bool,
) -> tuple[list[dict], Path | None]:
    annotated_dir = output_dir / "annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_for_timestamps = fps if fps and fps > 0 else None
    writer_fps = fps_for_timestamps if fps_for_timestamps else 30.0
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

    rows = []
    centers: list[tuple[int, int]] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = round(frame_index / fps_for_timestamps, 3) if fps_for_timestamps else ""
        detection = best_golf_ball_detection(model, frame, conf, iou)
        if detection is None:
            rows.append(empty_row(frame_index, timestamp, source.name))
        else:
            center = (int(round(detection["center_x"])), int(round(detection["center_y"])))
            centers.append(center)
            rows.append(detection_row(frame_index, timestamp, source.name, detection))

        annotated = annotate_frame(frame, detection, centers)
        if writer is not None:
            writer.write(annotated)
        if save_every_frame:
            frame_path = annotated_dir / f"{source.stem}_frame_{frame_index:06d}_annotated.jpg"
            cv2.imwrite(str(frame_path), annotated)

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
        return rows, annotated_video_path

    return rows, None


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotated_frames").mkdir(parents=True, exist_ok=True)

    validate_paths(source, model_path)
    model = YOLO(str(model_path))

    if source.is_dir():
        rows = process_image_folder(source, model, output_dir, args.conf, args.iou)
        annotated_video_path = None
    elif source.is_file() and is_video_file(source):
        rows, annotated_video_path = process_video(
            source,
            model,
            output_dir,
            args.conf,
            args.iou,
            args.save_every_frame,
        )
    else:
        raise ValueError(f"Source must be a video file or folder of images: {source}")

    csv_path = output_dir / "detections.csv"
    summary_path = output_dir / "trajectory_summary.json"
    summary = build_summary(rows, source, model_path, args.conf)

    write_csv(rows, csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("YOLO video/frame tracking complete.")
    print(f"Frames processed: {summary['total_frames_processed']}")
    print(f"Frames with detection: {summary['frames_with_detection']}")
    print(f"Detection rate: {summary['detection_rate']}")
    print(f"Rough direction: {summary['rough_direction']}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote annotated frames folder: {output_dir / 'annotated_frames'}")
    if annotated_video_path is not None:
        print(f"Wrote annotated video: {annotated_video_path}")


if __name__ == "__main__":
    main()
