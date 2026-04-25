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
    label: str
    circularity: float
    area: float
    detector_type: str


def _resize_for_processing(image_bgr: np.ndarray, max_dim: int = 1400) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image_bgr, 1.0
    scale = max_dim / float(largest)
    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _colored_mask_from_hsv(hsv_blur: np.ndarray) -> np.ndarray:
    # Red, red-orange, orange, and pink ranges.
    ranges = [
        (np.array([0, 45, 35], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)),
        (np.array([12, 40, 35], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
        (np.array([170, 45, 35], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
        (np.array([145, 35, 45], dtype=np.uint8), np.array([170, 255, 255], dtype=np.uint8)),
    ]
    mask = np.zeros(hsv_blur.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_blur, low, high))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return cv2.dilate(mask, kernel, iterations=1)


def _white_mask_from_hsv(hsv_blur: np.ndarray) -> np.ndarray:
    # Cool-tinted white allowed: low saturation + bright value.
    lower = np.array([30, 0, 85], dtype=np.uint8)
    upper = np.array([125, 95, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_blur, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _circle_stats(
    hsv: np.ndarray,
    v_channel: np.ndarray,
    colored_mask: np.ndarray,
    white_mask: np.ndarray,
    cx: int,
    cy: int,
    r: int,
) -> tuple[float, float, float]:
    inside = np.zeros(v_channel.shape, dtype=np.uint8)
    cv2.circle(inside, (cx, cy), r, 255, -1)
    ring = np.zeros(v_channel.shape, dtype=np.uint8)
    cv2.circle(ring, (cx, cy), int(r * 1.35), 255, -1)
    cv2.circle(ring, (cx, cy), int(r * 1.05), 0, -1)

    inside_pixels = max(1, cv2.countNonZero(inside))
    ring_pixels = max(1, cv2.countNonZero(ring))
    colored_ratio = cv2.countNonZero(cv2.bitwise_and(colored_mask, inside)) / float(inside_pixels)
    white_ratio = cv2.countNonZero(cv2.bitwise_and(white_mask, inside)) / float(inside_pixels)

    inside_v = float(cv2.mean(v_channel, mask=inside)[0])
    ring_v = float(cv2.mean(v_channel, mask=ring)[0])
    contrast = min(abs(inside_v - ring_v) / 90.0, 1.0)
    _ = hsv  # keep signature explicit for future tweaks
    return colored_ratio, white_ratio, contrast


def _detect_hough_candidates(
    image_bgr: np.ndarray,
    colored_mask: np.ndarray,
    white_mask: np.ndarray,
) -> tuple[list[DetectionResult], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.2)
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    min_r = max(30, int(min_dim * 0.03))
    max_r = min(300, int(min_dim * 0.48))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(35, int(min_r * 0.75)),
        param1=120,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )

    hough_vis = image_bgr.copy()
    results: list[DetectionResult] = []
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    if circles is None:
        return results, hough_vis

    for cx_f, cy_f, r_f in np.round(circles[0, :]).astype(int):
        cx = int(np.clip(cx_f, 0, w - 1))
        cy = int(np.clip(cy_f, 0, h - 1))
        r = int(np.clip(r_f, min_r, max_r))
        cv2.circle(hough_vis, (cx, cy), r, (255, 255, 0), 2)

        colored_ratio, white_ratio, contrast = _circle_stats(
            hsv, v_channel, colored_mask, white_mask, cx, cy, r
        )
        radius_score = min(max((r - min_r) / float(max(1, (max_r - min_r))), 0.0), 1.0)
        circularity = 1.0  # Hough already enforces circle geometry.

        if colored_ratio >= white_ratio and colored_ratio > 0.08:
            label = "colored_golf_ball_candidate"
            detector_type = "hough_plus_colored_mask"
            color_score = colored_ratio
        elif white_ratio > 0.10:
            label = "white_golf_ball_candidate"
            detector_type = "hough_plus_white_mask"
            color_score = white_ratio
        else:
            label = "unknown_ball_candidate"
            detector_type = "hough_shape_only"
            color_score = max(colored_ratio, white_ratio)

        confidence = min(0.45 * circularity + 0.35 * color_score + 0.20 * contrast + 0.10 * radius_score, 1.0)
        if color_score < 0.04 and contrast < 0.08:
            continue

        x = max(0, cx - r)
        y = max(0, cy - r)
        w_box = min(w - x, r * 2)
        h_box = min(h - y, r * 2)
        area = float(np.pi * (r**2))
        results.append(
            DetectionResult(
                x=int(x),
                y=int(y),
                w=int(w_box),
                h=int(h_box),
                confidence=round(float(confidence), 3),
                label=label,
                circularity=round(float(circularity), 3),
                area=round(area, 1),
                detector_type=detector_type,
            )
        )

    return results, hough_vis


def _iou(a: DetectionResult, b: DetectionResult) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    union = (a.w * a.h) + (b.w * b.h) - inter_area
    return inter_area / float(max(1, union))


def _dedupe_overlapping_candidates(
    candidates: list[DetectionResult], iou_threshold: float = 0.45
) -> list[DetectionResult]:
    kept: list[DetectionResult] = []
    for cand in sorted(candidates, key=lambda d: (d.confidence, d.w * d.h), reverse=True):
        if any(_iou(cand, existing) >= iou_threshold for existing in kept):
            continue
        kept.append(cand)
    return kept


def detect_ball(image_bgr: np.ndarray) -> list[DetectionResult]:
    """Detect visible golf ball candidates using Hough circles + mask scoring."""
    detections, _, _, _, _ = detect_ball_with_mask(image_bgr)
    return detections


def detect_ball_with_mask(
    image_bgr: np.ndarray,
) -> tuple[list[DetectionResult], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect candidates and return colored/white/combined masks + Hough preview."""
    proc, scale = _resize_for_processing(image_bgr)
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (7, 7), 0)
    colored = _colored_mask_from_hsv(hsv_blur)
    white = _white_mask_from_hsv(hsv_blur)
    combined = cv2.bitwise_or(colored, white)

    candidates, hough_vis = _detect_hough_candidates(proc, colored, white)
    candidates = _dedupe_overlapping_candidates(candidates)
    candidates.sort(key=lambda d: (d.confidence, d.w * d.h), reverse=True)
    candidates = candidates[:3]

    if scale != 1.0:
        inv = 1.0 / scale
        for det in candidates:
            det.x = int(round(det.x * inv))
            det.y = int(round(det.y * inv))
            det.w = int(round(det.w * inv))
            det.h = int(round(det.h * inv))
            det.area = round(det.area * (inv**2), 1)
        out_h, out_w = image_bgr.shape[:2]
        colored = cv2.resize(colored, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        white = cv2.resize(white, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.resize(combined, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        hough_vis = cv2.resize(hough_vis, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return candidates, colored, white, combined, hough_vis


def annotate_image(image_bgr: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    annotated = image_bgr.copy()
    for det in detections:
        p1 = (det.x, det.y)
        p2 = (det.x + det.w, det.y + det.h)
        color = (0, 140, 255) if det.label == "colored_golf_ball_candidate" else (255, 255, 255)
        text_color = (0, 140, 255) if det.label == "colored_golf_ball_candidate" else (220, 220, 220)
        cv2.rectangle(annotated, p1, p2, color, 4)
        center = (det.x + det.w // 2, det.y + det.h // 2)
        radius = max(8, int(0.5 * max(det.w, det.h)))
        cv2.circle(annotated, center, radius, color, 3)
        label = f"{det.label}:{det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (det.x, max(15, det.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
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

