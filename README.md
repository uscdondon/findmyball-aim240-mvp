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

Visual prediction checks on that small validation set also looked strong overall: golf-ball hits were typically **~0.98–1.00** confidence. One useful case was different: an image triggered a **false positive**—a finger was labeled as `golf_ball` at **lower confidence**. That is diagnostic information, not a sign the run failed; it flags the next dataset focus: **hard-negative** examples—scenes or frames with grass-only background, fingers, shadows, glare, clubs, leaves, tees, or other clearly non-ball objects—so training helps the model learn what **not** to call `golf_ball`. The v2 detector is progressing, but remains a prototype and **not production-ready**.

An additional unseen still-image test was run using a new compressed JPG containing both a white golf ball and a red/orange golf ball on grass. This image was not part of the YOLO v2 training or validation set. The model detected both balls successfully, with approximately **0.99** confidence on the white ball and **0.98** confidence on the red/orange ball, and the bounding boxes appeared tight around the visible balls. This is a useful detector generalization check before moving to sequential iPhone video tracking, but it is still only one unseen still-image test and **not** a production-readiness claim.

**Next step:** run inference on fresh, unseen photos and video frames that were not used for training or validation.

## YOLO Video Centroid Tracking

The project includes a simple frame-by-frame YOLO tracking script for controlled iPhone putt/chip MVP testing. It uses YOLO detections to find the `golf_ball` bounding box in each frame, then tracks the bounding-box centroid over time as a lightweight 2D trajectory.

This is not production-grade tracking and does not use ByteTrack or a multi-object tracker. It is intended for short, controlled videos or extracted frame folders where the capstone goal is to show detection plus basic movement summary.

Outputs include `detections.csv`, `trajectory_summary.json`, annotated frames, and `annotated_video.mp4` when the input is a video and video writing succeeds.

## Fresh iPhone Putt Video Tracking Test

The YOLO centroid tracking script was tested on a fresh iPhone white-ball putt video. The video was converted into sequential frames, and the script ran YOLOv8 detection frame-by-frame. It produced `detections.csv`, `trajectory_summary.json`, and annotated frames with bounding boxes, centroids, and a simple path line. In this first video test, **8** frames were processed and the ball was detected in **5** frames, for a **62.5%** detection rate. Average confidence was **0.3133**, with confidence ranging from **0.2695** to **0.3442**. The centroid summary estimated the ball moving **mostly left**, with `delta_x = -906.25` and `delta_y = -30.73`.

This shows the MVP tracking pipeline working on a real iPhone putt video, while also showing that detection confidence stayed **below** typical still-image validation results from the compact v2 set and several frames had **no detection**. Stronger repeatable tracking will need **more labeled video frames** of small, moving white golf balls, including harder cases such as **motion blur**, **distance**, and **near the edge of the frame**.

## Confidence Threshold Sweep for Video Tracking

Still-image tests produced high-confidence detections around **0.98 to 1.00**. The iPhone putt-video frames were more difficult because the white ball was **tiny**, **moving**, **compressed**, and sometimes near the **frame edge**.

A confidence-threshold sweep was run on that white-ball putt frame sequence using the **YOLO v2** model. At **`conf=0.25`**, visual detections were **conservative** and looked **clean enough** for a respectable capstone demo. At **`conf=0.15`**, **recall** improved: the model detected the ball in **7 of 8** frames in the CSV output. That makes **`0.15`** a useful option for centroid-tracking continuity, but it is a **recall-oriented** choice and needs **visual inspection** for potential false positives.

This illustrates the **precision/recall tradeoff**: higher thresholds are stricter, while lower thresholds can recover more tiny moving-ball frames. This is prototyping and experimentation for the MVP—**not** a production readiness claim.

## Video Frame Retraining Experiment

After the first white-ball putt video tracking test, the model detected the ball in **5 of 8** frames at **`conf=0.25`**. To target the missed frames, the **8 extracted video frames were labeled**, and a **YOLO v3** dataset was built by **adding** those frames to the clean **v2** dataset. The video-frame labels were visually inspected and **tight**, but the golf ball remained **very small** in the image.

YOLOv8n was **retrained for 30 epochs** on the v3 dataset. On the same putt video sequence at the **normal** tracking threshold, the v3 model **did not improve** the frame-level detection outcome (it was not a clear win over v2 in this run). A **train-split** evaluation still looked partially strong but showed a **recall gap**:

- Precision: **1.000**
- Recall: **0.703**
- mAP50: **0.800**
- mAP50-95: **0.694**

**Interpretation:** adding only **8** tiny, difficult video-frame examples was **not enough** to make small moving golf balls reliably detectable in this setup. The **current best MVP tracking result** on that first white-ball putt video remains the **v2** model.

**Next improvement:** add **more** labeled small moving-ball frames, prefer **tighter camera framing** or **closer** capture when possible, and explore **ROI/crop-based training** or a **lower confidence threshold** for video tracking when false positives can be tolerated.

## Demo Commands

Brief commands below assume Ultralytics CLI is installed (`pip install ultralytics`) and paths are substituted for your unseen white/red JPG and folder of sequentially extracted white-ball putt frames. Outputs are prototyping artifacts—not a finished product.

Still-image prediction on the unseen white/red dual-ball photo (`best.pt` is the trained v2 weights):

```bash
yolo detect predict \
  model=runs/detect/runs/findmyball/yolo_v2_clean_batch_01/weights/best.pt \
  source=YOUR_UNSEEN_WHITE_RED_DUAL_BALL.jpg \
  conf=0.25 \
  project=runs/findmyball \
  name=demo_white_red_still \
  save=True \
  exist_ok=True
```

Centroid tracking on the white-ball putt frame sequence (**v2** model, **`conf=0.25`**):

```bash
python scripts/detect_video_yolo.py \
  --model runs/detect/runs/findmyball/yolo_v2_clean_batch_01/weights/best.pt \
  --source PATH/TO/YOUR_PUTT_FRAMES_FOLDER \
  --conf 0.25 \
  --output-dir output/yolo_video_tracking/demo_putt_conf025
```

Same sequence with **`conf=0.15`** (recall-oriented; check annotated frames for false positives):

```bash
python scripts/detect_video_yolo.py \
  --model runs/detect/runs/findmyball/yolo_v2_clean_batch_01/weights/best.pt \
  --source PATH/TO/YOUR_PUTT_FRAMES_FOLDER \
  --conf 0.15 \
  --output-dir output/yolo_video_tracking/demo_putt_conf015
```

Tracking runs write annotated frames plus `detections.csv` and `trajectory_summary.json` under `--output-dir` (`annotated_video.mp4` when the `--source` is a video file and writing succeeds).

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
