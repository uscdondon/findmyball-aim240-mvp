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

    detections, colored_mask, white_mask, combined_mask, hough_candidates = detect_ball_with_mask(image)
    annotated = annotate_image(image, detections)

    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    annotated_path = output_dir / f"{stem}_annotated.png"
    json_path = output_dir / f"{stem}_detections.json"
    colored_mask_path = output_dir / f"{stem}_colored_mask.png"
    white_mask_path = output_dir / f"{stem}_white_mask.png"
    combined_mask_path = output_dir / f"{stem}_combined_mask.png"
    hough_candidates_path = output_dir / f"{stem}_hough_candidates.png"

    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(colored_mask_path), colored_mask)
    cv2.imwrite(str(white_mask_path), white_mask)
    cv2.imwrite(str(combined_mask_path), combined_mask)
    cv2.imwrite(str(hough_candidates_path), hough_candidates)

    payload = {
        "image_path": str(image_path),
        "annotated_image_path": str(annotated_path),
        "colored_mask_path": str(colored_mask_path),
        "white_mask_path": str(white_mask_path),
        "combined_mask_path": str(combined_mask_path),
        "hough_candidates_path": str(hough_candidates_path),
        "detections_json_path": str(json_path),
        "count": len(detections),
        "detections": [
            {
                "x": int(d.x),
                "y": int(d.y),
                "w": int(d.w),
                "h": int(d.h),
                "label": d.label,
                "confidence": float(d.confidence),
                "circularity": float(d.circularity),
                "area": float(d.area),
                "detector_type": d.detector_type,
            }
            for d in detections
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

