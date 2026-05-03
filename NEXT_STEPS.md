# FindMyBall AIM240 - Next Steps

## Public Deployment Links

- **FindMyBall.io** — public landing page for the golf ball computer vision MVP: https://findmyball.io/
- **Don M. Inouye** — professional AI / ML portfolio site: https://donminouye.com/

## Current Status

The project has an end-to-end smoke-test ML pipeline:

data capture -> frame extraction -> YOLO labeling -> dataset validation -> train/val split -> YOLO training -> inference

The current YOLO workflow trains successfully and produces local weights. Batch 03 now detects validation images strongly, but confidence and localization still vary by object scale. The pipeline works; the model quality is still early-stage.

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
- Balance close, medium, and far golf-ball examples.
- Rebalance validation so small, medium, and large examples are represented.

1. Add more labeled data
   - More red/orange ball frames
   - More white ball frames
   - More lighting conditions
   - More distances and camera angles
   - Balanced close, medium, and far ball examples

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

## 2026-04-27 Size Balance Note

The size bucket report showed the validation set had no large golf-ball examples. Before retraining or evaluating Batch 03-style runs, rebalance validation so small, medium, and large golf-ball examples are represented.

## Batch 04 Rebalanced Validation Note

Batch 04 rebalanced validation exposed a scale-specific failure mode: small and medium examples detect strongly, but large close-up white golf balls (notably `IMG_7044` and `IMG_7045`) are still weak at normal confidence thresholds. Next data priority is balanced large close-up white/red-orange examples, tight visible-ball labels, and more lighting/angle variation while keeping hard large-example validation images in place.

## Parking Note - Label Cleanup Needed

Current issue:
The project has many folders in play, and some YOLO ground-truth labels appear too loose, especially large close-up golf-ball examples such as IMG_7044 and IMG_7045.

Do not continue relabeling while tired.

Next session plan:
1. Start with git status.
2. Run:
   python scripts/yolo_audit_dataset.py
   python scripts/prepare_yolo_dataset.py
   python scripts/visualize_yolo_labels.py --split train --max-images 80
   python scripts/visualize_yolo_labels.py --split val --max-images 20
3. Open:
   evidence/label_visual_checks/train
   evidence/label_visual_checks/val
4. Make one simple list:
   tmp/bad_label_stems.txt
5. Add only image stems whose GROUND-TRUTH label preview is visibly wrong.
6. Relabel only those images in MakeSense.
7. Copy corrected labels back into the correct split.
8. Validate again.
9. Commit only after visual confirmation.

Important:
- Do not move images between train and val tonight.
- Do not train again tonight.
- Do not process videos tonight.
- Do not delete raw_media.
- Do not delete data/yolo.
- Do not rewrite Git history.
