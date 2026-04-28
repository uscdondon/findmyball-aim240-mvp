"""Run YOLO prediction at several confidence thresholds for visual comparison."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


THRESHOLDS = (0.05, 0.10, 0.25)
PROJECT_DIR = Path("runs") / "findmyball"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO predictions at multiple confidence thresholds.")
    parser.add_argument("--model", required=True, help="Path to YOLO model weights, such as best.pt")
    parser.add_argument("--source", required=True, help="Image, folder, or video source for prediction")
    parser.add_argument("--name", default="predict_compare", help="Base run name for comparison outputs")
    return parser.parse_args()


def run_predict(model: str, source: str, base_name: str, threshold: float) -> Path:
    suffix = f"conf_{int(threshold * 100):02d}"
    run_name = f"{base_name}_{suffix}"
    command = [
        "yolo",
        "detect",
        "predict",
        f"model={model}",
        f"source={source}",
        f"conf={threshold}",
        f"project={PROJECT_DIR}",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
    ]
    print(f"\nRunning prediction at conf={threshold:.2f}")
    print(" ".join(command))
    subprocess.run(command, check=True)
    return PROJECT_DIR / run_name


def main() -> None:
    args = parse_args()
    output_dirs: list[Path] = []
    for threshold in THRESHOLDS:
        output_dirs.append(run_predict(args.model, args.source, args.name, threshold))

    print("\nPrediction comparison outputs:")
    for output_dir in output_dirs:
        print(f"  {output_dir}")


if __name__ == "__main__":
    main()
