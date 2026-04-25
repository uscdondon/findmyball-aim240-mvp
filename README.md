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
  extract_frames.py
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
```

## Known Limitations

- Baseline only; not YOLO-based yet
- Controlled close-up video only so far
- Bounding boxes are intentionally generous
- May fail with fast motion, distant balls, motion blur, or cluttered scenes
- No trajectory estimate yet
- No final resting-location estimate yet
- No FastAPI/Docker/frontend layer yet

## Next Steps

- test white-ball micro-putt video
- add trajectory estimate from detected center points
- prepare YOLO dataset structure
- train YOLO detector
- wrap model in FastAPI/Docker later
- optionally add a simple frontend demo
