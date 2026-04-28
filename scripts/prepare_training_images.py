"""Prepare YOLO training images as compressed JPGs without cropping."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw images into YOLO-ready JPG files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing source images")
    parser.add_argument("--output-dir", required=True, help="Directory to write prepared JPG images")
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum output width (default: 1280)")
    parser.add_argument("--quality", type=int, default=85, help="JPG quality from 1 to 95 (default: 85)")
    return parser.parse_args()


def output_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width, height
    scale = max_width / float(width)
    return max_width, int(round(height * scale))


def prepare_image(source: Path, output_dir: Path, max_width: int, quality: int) -> bool:
    try:
        with Image.open(source) as image:
            original_size = image.size
            new_size = output_size(image.width, image.height, max_width)
            prepared = image.convert("RGB")
            if new_size != original_size:
                prepared = prepared.resize(new_size, Image.Resampling.LANCZOS)

            output_path = output_dir / f"{source.stem}.jpg"
            prepared.save(output_path, "JPEG", quality=quality, optimize=True)
            print(
                f"{source} -> {output_path} | "
                f"original={original_size[0]}x{original_size[1]} "
                f"new={new_size[0]}x{new_size[1]}"
            )
            return True
    except UnidentifiedImageError:
        if source.suffix.lower() == ".heic":
            print(f"Could not read HEIC file: {source}")
            print("Convert HEIC first with: sips -s format jpeg input.HEIC --out output.jpg")
        else:
            print(f"Could not read image file: {source}")
    return False


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if not (1 <= args.quality <= 95):
        raise ValueError("--quality must be between 1 and 95")

    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for source in sorted(input_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if prepare_image(source, output_dir, args.max_width, args.quality):
            converted += 1

    print(f"Done. Converted {converted} image(s) to {output_dir}.")
    print("Labels were not modified.")


if __name__ == "__main__":
    main()
