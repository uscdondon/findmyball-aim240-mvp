import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run detect_video_yolo.py across multiple confidence thresholds."
    )
    parser.add_argument(
        "--source",
        default="input/video_tests/putt_test_01_white_ball_frames",
        help="Video file or folder of extracted frames.",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/runs/findmyball/yolo_v2_clean_batch_01/weights/best.pt",
        help="YOLO model path.",
    )
    parser.add_argument(
        "--output-root",
        default="output/yolo_video_tracking/confidence_sweep_putt_test_01",
        help="Root folder for confidence sweep outputs.",
    )
    parser.add_argument(
        "--confs",
        nargs="+",
        type=float,
        default=[0.50, 0.35, 0.25, 0.20, 0.15, 0.10],
        help="Confidence thresholds to test.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    model = Path(args.model)
    output_root = Path(args.output_root)

    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    if not model.exists():
        raise SystemExit(f"Model does not exist: {model}")

    output_root.mkdir(parents=True, exist_ok=True)

    results = []

    for conf in args.confs:
        conf_name = str(conf).replace(".", "p")
        output_dir = output_root / f"conf_{conf_name}"

        cmd = [
            sys.executable,
            "scripts/detect_video_yolo.py",
            "--source",
            str(source),
            "--model",
            str(model),
            "--output-dir",
            str(output_dir),
            "--conf",
            str(conf),
        ]

        run_command(cmd)

        summary_path = output_dir / "trajectory_summary.json"
        if not summary_path.exists():
            print(f"Missing summary: {summary_path}")
            continue

        summary = json.loads(summary_path.read_text())

        results.append(
            {
                "conf": conf,
                "output_dir": str(output_dir),
                "total_frames_processed": summary.get("total_frames_processed"),
                "frames_with_detection": summary.get("frames_with_detection"),
                "frames_without_detection": summary.get("frames_without_detection"),
                "detection_rate": summary.get("detection_rate"),
                "average_confidence": summary.get("average_confidence"),
                "min_confidence": summary.get("min_confidence"),
                "max_confidence": summary.get("max_confidence"),
                "rough_direction": summary.get("rough_direction"),
                "delta_x": summary.get("delta_x"),
                "delta_y": summary.get("delta_y"),
            }
        )

    print("\nCONFIDENCE SWEEP SUMMARY")
    print("-" * 120)
    print(
        f"{'conf':>6} | {'detected':>10} | {'rate':>8} | {'avg_conf':>8} | "
        f"{'min':>8} | {'max':>8} | {'direction':>14} | output"
    )
    print("-" * 120)

    for row in results:
        detected = f"{row['frames_with_detection']}/{row['total_frames_processed']}"
        rate = row["detection_rate"]
        avg = row["average_confidence"]
        mn = row["min_confidence"]
        mx = row["max_confidence"]

        print(
            f"{row['conf']:>6.2f} | "
            f"{detected:>10} | "
            f"{rate if rate is not None else '':>8} | "
            f"{avg if avg is not None else '':>8} | "
            f"{mn if mn is not None else '':>8} | "
            f"{mx if mx is not None else '':>8} | "
            f"{str(row['rough_direction']):>14} | "
            f"{row['output_dir']}"
        )

    print("\nHow to choose:")
    print("  Use the highest confidence threshold that still gives a believable ball path.")
    print("  0.50 or 0.35 is stricter/respectable, but may miss tiny video balls.")
    print("  0.25 is conservative and common for object detection demos.")
    print("  0.20 or 0.15 may be useful for tiny moving balls if boxes are visually correct.")
    print("  0.10 is recall-oriented; use only if annotated frames still look clean.")
    print("  Do not choose by numbers alone. Open annotated_frames and inspect the path.")


if __name__ == "__main__":
    main()
