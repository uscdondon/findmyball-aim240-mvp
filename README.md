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
  yolo_audit_dataset.py
  yolo_predict_compare.py
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

Future YOLO training images should be compressed JPGs, not full-size PNGs. The image scene should not be cropped tightly around the ball; keep the full training context and make only the label bounding box tight around the ball.

```bash
python scripts/prepare_training_images.py --input-dir raw_images --output-dir data/yolo/images/train
```

After labels exist, use this utility to move a reproducible portion of complete image-label pairs from train to validation:

```bash
python scripts/split_yolo_dataset.py --dry-run
python scripts/split_yolo_dataset.py --apply
```

Use this utility to audit image-label pairing, label formatting, class IDs, normalized coordinates, and bounding-box size statistics:

```bash
python scripts/yolo_audit_dataset.py
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

The first YOLO smoke model trained successfully and produced model weights, but prediction on the 5-image validation set produced no detections. This is acceptable for the smoke-test stage because the purpose was to validate dataset structure, labels, and training execution. Improving YOLO detection quality requires more labeled images, more varied frames, and longer training.

YOLO smoke-test evidence artifacts (if present):

- `evidence/yolo_smoke_test/results.csv`
- `evidence/yolo_smoke_test/labels.jpg`
- `evidence/yolo_smoke_test/results.png`

### Batch 02 Training Result

Batch 02 trained for 15 epochs on the enlarged labeled dataset and produced local `best.pt` and `last.pt` weights. This is still experimental work, not a production-grade detector.

- Reported validation metrics improved on the small 5-image validation set.
- Visual prediction at default confidence did not show reliable boxes.
- A low-confidence prediction run with `conf=0.05` produced one correct golf-ball detection at confidence around 0.08.

Together, these observations suggest the model has begun to learn the `golf_ball` class, but the detector remains weak and not robust. This is not reliable detection under normal thresholds and should not be interpreted as production performance.

The next improvement is more labeled examples, especially varied video frames, distances, lighting, and ball colors.

Batch 02 evidence (if present):

- `evidence/yolo_batch_02/yolo_batch_02_low_conf_prediction_example.jpg`
- `evidence/yolo_batch_02/results.csv`
- `evidence/yolo_batch_02/results.png`
- `evidence/yolo_batch_02/labels.jpg`

### YOLO Batch 02 Interpretation

Batch 02 produced one correct low-confidence golf-ball detection at approximately 0.08 confidence. This indicates the YOLO model has started learning the `golf_ball` class, but the detector is still weak and not robust. Low confidence means the next priority is not model integration; it is improving the dataset.

The next improvement path is:

- audit the current YOLO dataset
- add more labeled video frames
- rebalance train/validation splits after adding data
- retrain as Batch 03 with more data
- compare predictions at `conf=0.05`, `conf=0.10`, and `conf=0.25`

For visual comparison across confidence thresholds:

```bash
python scripts/yolo_predict_compare.py \
  --model runs/findmyball/yolo_batch_02/weights/best.pt \
  --source data/yolo/images/val \
  --name yolo_batch_02_compare
```

### YOLO Batch 04 Rebalanced Validation Note

Batch 04 rebalanced validation exposed a scale-specific failure mode. The model detects small and medium golf balls confidently, but large close-up white golf balls such as `IMG_7044` and `IMG_7045` are only detected at low confidence.

At `conf=0.05`, `IMG_7044` produced a `golf_ball` detection around `0.21` and `IMG_7045` around `0.11`, but these detections do not survive normal confidence thresholds. This indicates the model is learning the pattern but still needs more large close-up examples and improved localization.

Next data priority:

- large close-up white balls
- large close-up red/orange balls
- tight labels around the visible ball
- more varied lighting and angles
- keep `IMG_7044` and `IMG_7045` as hard validation examples

## YOLO v2 Clean Dataset Result

After visual inspection showed that some first-pass still-image bounding boxes were too loose, the project moved from that dataset to a cleaner second-pass YOLO dataset.

The second-pass dataset uses full-scene compressed JPG images with tight `golf_ball` bounding boxes. Grass and other surroundings stay visible for background context, while labels stay constrained to the visible golf ball only.

YOLOv8n was trained for **30 epochs** on the clean v2 dataset. Training completed successfully and produced saved model weights.

On the **small** v2 validation set, Ultralytics validation reported:

- Precision: **0.972**
- Recall: **1.000**
- mAP50: **0.995**
- mAP50-95: **0.995**

That validation split is still small, so treat these metrics as a strong controlled prototype checkpoint, **not** evidence of robust real-world behavior or production readiness.

**Next step:** run inference on fresh, unseen photos and video frames that were not used for training or validation.

## Current Status: End-to-End Pipeline Working, Clean Dataset Pass In Progress

FindMyBall is an AIM240 computer vision capstone project for golf ball detection and tracking. The project now has an end-to-end ML prototype pipeline working:

`data collection -> frame extraction -> manual YOLO labeling -> dataset validation -> train/validation split -> YOLOv8 training -> saved model weights -> test predictions`

The current model is not production-quality, and the main current focus is data quality. Visual inspection showed that some first-pass still-image bounding boxes were too loose. Rather than overclaiming performance, the project is moving to a cleaner second-pass YOLO dataset with compressed JPG images and tighter labels.

This reflects a core ML engineering lesson: dataset quality and label consistency are required before model metrics are meaningful. The project does not claim robust generalization or production readiness at this stage.

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
