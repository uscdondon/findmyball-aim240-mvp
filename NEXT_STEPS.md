# FindMyBall AIM240 - Next Steps

## Current Status

The project has an end-to-end smoke-test ML pipeline:

data capture -> frame extraction -> YOLO labeling -> dataset validation -> train/val split -> YOLO training -> inference

The current YOLO smoke model trained successfully and produced local weights, but it did not yet produce reliable detections on validation images. The pipeline works; the model quality is still early-stage.

## Completed

- Baseline computer vision detector for red/orange and white golf balls
- Still-image detection outputs
- Sampled video detection pipeline
- Red micro-putt video evidence
- Basic detection logs and evidence artifacts
- YOLO dataset scaffold
- 21 labeled YOLO seed images
- Train/val split
- YOLOv8n smoke training for 3 epochs
- YOLO inference command tested
- README updated with smoke-test status

## Next Engineering Priorities

Next immediate priorities:

- Run dataset audit.
- Add more labeled video frames.
- Rebalance train/val after adding data.
- Train Batch 03 with more data.
- Compare predictions at conf=0.05, 0.10, 0.25.

1. Add more labeled data
   - More red/orange ball frames
   - More white ball frames
   - More lighting conditions
   - More distances and camera angles

2. Improve YOLO training
   - Try more epochs after adding data
   - Keep using yolov8n for fast iteration
   - Compare validation predictions before claiming improvement

3. Improve video pipeline
   - Add better trajectory summary
   - Detect ball center across frames
   - Estimate simple movement direction
   - Eventually estimate final resting region

4. Later integration
   - FastAPI wrapper
   - Docker
   - Simple frontend or Streamlit demo
   - AIM230 service layer integration

## Next Session First Command

cd /Users/doninouye/Desktop/aim240-capstone/findmyball-aim240-mvp
source .venv/bin/activate
git status
python scripts/prepare_yolo_dataset.py

## Do Not Do First

- Do not train again before adding data.
- Do not build frontend before model/pipeline is stronger.
- Do not overclaim model accuracy.
- Do not delete runs/ unless sure.

## 2026-04-26 Evening Note

New red and white labeled still-image data was added. Additional red/white videos exist in raw_media/2026-04-26 and should be processed later into YOLO frame batches. Do not train again until the video frames are seeded, labeled, and dataset validation is clean.
