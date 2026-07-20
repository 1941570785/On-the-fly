#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAX_RESOLUTION = 1_500_000


def target_image_size(width: int, height: int) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if width * height <= MAX_RESOLUTION:
        return width, height
    downsampling = math.sqrt((width * height) / MAX_RESOLUTION)
    return int(round(width / downsampling)), int(round(height / downsampling))


def processed_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    if width * height > MAX_RESOLUTION:
        downsampling = math.sqrt((width * height) / MAX_RESOLUTION)
        image = cv2.resize(
            image,
            (0, 0),
            fx=1.0 / downsampling,
            fy=1.0 / downsampling,
            interpolation=cv2.INTER_AREA,
        )
        height, width = image.shape[:2]
    return int(width), int(height)


def first_image(source: Path) -> Path:
    images_dir = source / "images"
    candidates = sorted(
        (path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda path: (len(path.stem), path.stem),
    )
    if not candidates:
        raise FileNotFoundError(f"No images in {images_dir}")
    return candidates[0]


def remove_invalid_cache(path: Path) -> None:
    if not path.exists():
        return
    try:
        torch.jit.load(str(path), map_location="cpu")
    except Exception as error:
        print(f"Removing invalid cache {path}: {error}", flush=True)
        path.unlink()


def prewarm(sources: list[Path], *, top_k: int = 6144) -> list[tuple[int, int]]:
    from poses.feature_detector import Detector
    from scene.dense_extractor import DenseExtractor

    sizes = []
    for source in sources:
        size = processed_image_size(first_image(source))
        if size not in sizes:
            sizes.append(size)
    cache_dir = ROOT / "models" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for width, height in sizes:
        dense_path = cache_dir / f"dense_extractor_{width}_{height}.pt"
        detector_path = cache_dir / f"xfeat_{width}_{height}_{int(top_k)}.pt"
        remove_invalid_cache(dense_path)
        remove_invalid_cache(detector_path)
        print(f"Prewarming model caches for {width}x{height}", flush=True)
        dense = DenseExtractor(width, height)
        detector = Detector(int(top_k), width, height)
        del dense, detector
        torch.cuda.empty_cache()
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--top_k", type=int, default=6144)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = prewarm(list(args.sources), top_k=args.top_k)
    print("prewarmed=" + ",".join(f"{width}x{height}" for width, height in sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
