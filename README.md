## AIM240 FindMyBall MVP

This is a **small, runnable, class-ready capstone MVP** for AIM240.

Today it implements a baseline **classical computer vision detector** (OpenCV + NumPy) and a working command-line pipeline:

- detect a ball-like object in an image
- save an annotated image to `output/`
- save detection JSON to `output/`
- extract every Nth frame from a video into `output/frames/`

This is intentionally minimal so you can demo quickly and iterate.

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

## Quick Start

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/detect_image.py input/sample.png
python scripts/detect_video.py videos/sample.mov --every 5
python scripts/extract_frames.py videos/sample.mov --every 10
```

## What Works Today

### 1) Baseline Detector (`app/services/detector.py`)

- Uses OpenCV classical CV methods focused on **orange golf ball candidates**:
  - HSV orange/red-orange color segmentation
  - mask cleanup with blur + morphological open/close
  - contour filtering by area, circularity, and aspect ratio
- Sorts by confidence and returns only top candidates for readable annotations.

## Current Result

The current MVP successfully detects the orange golf ball in the sample image using HSV color segmentation plus contour-based shape filtering (area, circularity, and aspect ratio).

## Known Limitation

This baseline can also detect orange UI elements or thumbnails, because it detects orange circular/compact regions rather than semantic golf balls. That limitation is why a trained YOLO model is planned as future work.

### 2) Image Detection CLI (`scripts/detect_image.py`)

- Input: image path
- Output files:
  - `output/<image_stem>_annotated.png`
  - `output/<image_stem>_detections.json`
- Also prints detection JSON to terminal.

### 3) Frame Extraction CLI (`scripts/extract_frames.py`)

- Input: video path + `--every N`
- Output files:
  - `output/frames/<video_stem>_frame_000000.jpg` (and so on)

### 4) Video Batch Detection CLI (`scripts/detect_video.py`)

Still images are useful detector tests, but the real capstone direction is putt/chip-shot video. `detect_video.py` runs the current baseline detector across sampled video frames, saves annotated frames, writes JSON/CSV detection summaries, and attempts an annotated MP4 output.

Future work will add tracking across frames and final resting-location estimation.

## Minimal FastAPI Skeleton Included

FastAPI files are scaffolded (`app/main.py`, `app/api/endpoints.py`, `app/schemas/`) so the project can grow cleanly, but the full API is intentionally not built yet.

## Future Work (Not Implemented Today)

- YOLO-based training and inference pipeline
- Full FastAPI upload/prediction endpoints
- Dockerized deployment
- Simple frontend for uploads and result visualization

## Notes

- Keep sample media in `input/` and `videos/`.
- Generated outputs go to `output/`.
- Model weights and large media are ignored by git.
