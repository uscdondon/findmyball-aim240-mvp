# Chip Arc 06 Red/White YOLO v4 Improvement Evidence

This folder preserves the before/after evidence for a targeted AIM240 chip-shot improvement experiment using fresh red-ball and white-ball iPhone chip-shot videos.

## Baseline: YOLO v2 Clean Model

At confidence 0.25:

- Red chip video: 4 / 17 frames detected = 23.53%
- White chip video: 6 / 19 frames detected = 31.58%

The YOLO v2 clean model produced intermittent real-world chip-shot detections but did not provide continuous enough detection for a reliable centroid path.

## Improvement: YOLO v4 Chip-Arc Model

After hand-labeling a small curated set of visible red and white chip-shot frames and retraining a YOLO v4 chip-arc model:

- Red chip video: 17 / 17 frames detected = 100%
- White chip video: 19 / 19 frames detected = 100%
- Red average confidence: 0.9123
- White average confidence: 0.8523

This demonstrates that targeted chip-arc labeling substantially improved detection continuity on the curated red/white chip-shot test clips.

## Scope Note

This is a small, targeted MVP improvement experiment. It demonstrates that the model can be improved through focused data collection, hand labeling, retraining, and before/after measurement. It does not claim production readiness, calibrated real-world trajectory, GPS tracking, or generalization to unseen chip-shot videos.

The next best-practice step is to evaluate the YOLO v4 chip-arc model on a separate unseen chip-shot clip.
