from pathlib import Path
import argparse
import cv2

CLASS_NAMES = {0: "golf_ball"}

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def draw_labels(split: str, max_images: int | None = None):
    root = Path("data/yolo")
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    out_dir = Path("evidence") / "label_visual_checks" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    if max_images:
        label_files = label_files[:max_images]

    written = 0

    for label_path in label_files:
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            print(f"Missing image for {label_path}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read {image_path}")
            continue

        h, w = img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                print(f"Malformed line in {label_path}: {line}")
                continue

            class_id = int(float(parts[0]))
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h

            x1 = int(round(x_center - box_w / 2))
            y1 = int(round(y_center - box_h / 2))
            x2 = int(round(x_center + box_w / 2))
            y2 = int(round(y_center + box_h / 2))

            label = CLASS_NAMES.get(class_id, f"class_{class_id}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(
                img,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        out_path = out_dir / f"{image_path.stem}_labels.jpg"
        cv2.imwrite(str(out_path), img)
        written += 1

    print(f"Wrote {written} label preview image(s) to {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO ground-truth labels.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    draw_labels(args.split, args.max_images)

if __name__ == "__main__":
    main()
