"""Baseline classical CV detector for FindMyBall MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class DetectionResult:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    method: str


def _orange_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (7, 7), 0)
    # Broader ranges to tolerate red-orange/orange under varied lighting.
    lower_orange_1 = np.array([0, 55, 40], dtype=np.uint8)
    upper_orange_1 = np.array([18, 255, 255], dtype=np.uint8)
    lower_orange_2 = np.array([18, 45, 40], dtype=np.uint8)
    upper_orange_2 = np.array([35, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(blur, lower_orange_1, upper_orange_1)
    mask2 = cv2.inRange(blur, lower_orange_2, upper_orange_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def _detect_orange_contours(mask: np.ndarray) -> list[DetectionResult]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[DetectionResult] = []
    image_h, image_w = mask.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h else 0.0
        touches_edge = x <= 1 or y <= 1 or (x + w) >= (image_w - 1) or (y + h) >= (image_h - 1)
        if not touches_edge and not (0.55 <= aspect_ratio <= 1.8):
            continue

        aspect_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio))
        area_score = min(area / float(image_h * image_w) * 8.0, 1.0)
        conf = float(
            min(
                max(0.45 * circularity + 0.25 * aspect_score + 0.30 * area_score, 0.0),
                1.0,
            )
        )
        results.append(
            DetectionResult(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                confidence=round(conf, 3),
                method="hsv_orange_shape",
            )
        )
    # Prefer larger confident candidates for the near-field ball baseline.
    results.sort(key=lambda d: (d.confidence, d.w * d.h), reverse=True)
    return results[:5]


def detect_ball(image_bgr: np.ndarray) -> list[DetectionResult]:
    """Detect orange golf ball candidates in a BGR image."""
    mask = _orange_mask(image_bgr)
    return _detect_orange_contours(mask)


def detect_ball_with_mask(image_bgr: np.ndarray) -> tuple[list[DetectionResult], np.ndarray]:
    """Detect orange golf ball candidates and return the debug mask."""
    mask = _orange_mask(image_bgr)
    return _detect_orange_contours(mask), mask


def annotate_image(image_bgr: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    annotated = image_bgr.copy()
    for det in detections:
        p1 = (det.x, det.y)
        p2 = (det.x + det.w, det.y + det.h)
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 4)
        center = (det.x + det.w // 2, det.y + det.h // 2)
        radius = max(8, int(0.5 * max(det.w, det.h)))
        cv2.circle(annotated, center, radius, (255, 255, 0), 3)
        label = f"{det.method}:{det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (det.x, max(15, det.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def run_detection(image_path: str | Path) -> dict[str, Any]:
    """Load image, run detector, return serializable output."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = detect_ball(image)
    return {
        "image_path": str(image_path),
        "detections": [asdict(d) for d in detections],
        "count": len(detections),
    }

