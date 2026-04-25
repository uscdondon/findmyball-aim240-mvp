"""CLI script to detect ball candidates in an image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.detector import annotate_image, detect_ball_with_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline detection on an image.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")

    detections, debug_mask = detect_ball_with_mask(image)
    annotated = annotate_image(image, detections)

    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    annotated_path = output_dir / f"{stem}_annotated.png"
    json_path = output_dir / f"{stem}_detections.json"
    mask_path = output_dir / f"{stem}_mask.png"

    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(mask_path), debug_mask)

    payload = {
        "image_path": str(image_path),
        "annotated_image_path": str(annotated_path),
        "debug_mask_path": str(mask_path),
        "detections_json_path": str(json_path),
        "count": len(detections),
        "detections": [
            {
                "x": int(d.x),
                "y": int(d.y),
                "w": int(d.w),
                "h": int(d.h),
                "confidence": float(d.confidence),
                "method": d.method,
            }
            for d in detections
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

