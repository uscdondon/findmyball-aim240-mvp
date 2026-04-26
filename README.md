## AIM240 FindMyBall MVP

## Project Overview

FindMyBall AIM240 MVP is a computer vision capstone prototype for detecting golf ball candidates in still images and short putt/chip-shot videos. The current system is a working baseline intended for class demonstration and iterative improvement, not a production detector.

## Current MVP Result

The current MVP can:

- detect a red/orange golf ball in a still image
- detect a white golf ball in a still image
- process a short red micro-putt iPhone video
- save annotated outputs
- save JSON/CSV detection logs
- save evidence artifacts in `evidence/video/`

## Current Method

The baseline method currently uses:

- HSV segmentation for red/orange/pink golf balls
- brightness/low-saturation mask for white golf balls
- Hough circle detection
- largest-mask-component plus Hough-circle selection
- one dominant detection candidate per still image
- sampled frame processing for video
- per-frame center-point extraction (`center_x`, `center_y`) plus basic trajectory summary across sampled frames

The video pipeline now outputs an early trajectory summary from detected center points. `trajectory_summary` includes first/last detection frame, start/end center, `delta_x`, `delta_y`, total pixel displacement, and a simple movement direction label. This is not final resting-location estimation yet; it is an early bridge from frame detection toward putt/chip tracking.

## Evidence

Current evidence artifacts include:

- `evidence/video/red_micro_putt_best_frame_screenshot.png`
- `evidence/video/red_micro_putt_video_detections.json`
- `evidence/video/red_micro_putt_video_detections.csv`
- `evidence/video/red_micro_putt_annotated.mp4` (if present)

## Current Project Structure

```text
app/
  __init__.py
  main.py
  api/
    __init__.py
    endpoints.py
  schemas/
    __init__.py
    prediction.py
  services/
    __init__.py
    detector.py
scripts/
  detect_image.py
  detect_video.py
  prepare_yolo_dataset.py
  seed_yolo_images.py
  split_yolo_dataset.py
  extract_frames.py
data/
  yolo/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    dataset.yaml
    README.md
input/
output/
videos/
models/
frontend/
requirements.txt
README.md
.gitignore
```

## How to Run

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/detect_image.py input/red.png
python scripts/detect_image.py input/white.png
python scripts/detect_video.py videos/red_micro_putt.MOV --every 15 --max-frames 20 --resize-width 720
python scripts/prepare_yolo_dataset.py
```

## YOLO Dataset Structure

This is the next layer after the baseline detector. The baseline detector proves the computer vision pipeline, while the YOLO dataset scaffold prepares the project for future trained object detection. YOLO training has not been run yet.

Place images in:

- `data/yolo/images/train`
- `data/yolo/images/val`
- `data/yolo/images/test`

Place matching label files in:

- `data/yolo/labels/train`
- `data/yolo/labels/val`
- `data/yolo/labels/test`

Each label file uses YOLO format:

```text
class_id x_center y_center width height
```

All coordinate values are normalized from `0` to `1`. Class `0` means `golf_ball`.

Use this utility to count images/labels and warn about missing labels:

```bash
python scripts/prepare_yolo_dataset.py
```

Use this utility to copy still images or sampled video frames into a YOLO image split for later manual labeling:

```bash
python scripts/seed_yolo_images.py input/red.png input/white.png videos/red_micro_putt_trimmed.mp4 --every 15 --max-frames 30 --split train
```

`seed_yolo_images.py` does not create label files or fake annotations. It only prepares images for later manual labeling.

After labels exist, use this utility to move a reproducible portion of complete image-label pairs from train to validation:

```bash
python scripts/split_yolo_dataset.py --dry-run
python scripts/split_yolo_dataset.py --apply
```

## Initial YOLO Seed Dataset

- The project now includes an initial labeled YOLO seed dataset.
- The dataset currently contains 21 labeled training images.
- The class list has one class: `golf_ball`.
- Labels were created manually using [makesense.ai](https://www.makesense.ai/).
- This dataset is a seed for future YOLO training, not a production-scale dataset.
- Red/orange and white ball examples are included.

## YOLO Smoke Training

The project now has a small labeled YOLO seed dataset and can run a tiny YOLOv8n smoke training run. This is not a final accurate model; it is a validation that the dataset structure, labels, and training pipeline work.

- YOLOv8n trained successfully for 3 epochs.
- The dataset used 16 training images and 5 validation images.
- Training produced `best.pt` and `last.pt` locally.
- Training artifacts live under `runs/` and are ignored because they can become large.

```bash
pip install ultralytics

yolo detect train \
  model=yolov8n.pt \
  data=data/yolo/dataset.yaml \
  epochs=3 \
  imgsz=640 \
  batch=4 \
  project=runs/findmyball \
  name=yolo_smoke_test

yolo detect predict \
  model=runs/detect/runs/findmyball/yolo_smoke_test/weights/best.pt \
  source=data/yolo/images/val \
  project=runs/findmyball \
  name=yolo_smoke_predictions \
  save=True
```

This smoke test validates the YOLO training pipeline but does not prove robust golf-ball detection yet. More labeled images, more diverse frames, and longer training will be required.

## Known Limitations

- Baseline only; not YOLO-based yet
- Controlled close-up video only so far
- Bounding boxes are intentionally generous
- May fail with fast motion, distant balls, motion blur, or cluttered scenes
- Trajectory summary is basic and based on sampled detections only
- No final resting-location estimate yet
- No FastAPI/Docker/frontend layer yet

## Next Steps

- test white-ball micro-putt video
- strengthen trajectory modeling beyond basic center-point displacement
- prepare YOLO dataset structure
- train YOLO detector
- wrap model in FastAPI/Docker later
- optionally add a simple frontend demo
