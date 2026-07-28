#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from args import get_args
from scene.scene_model import SceneModel
from tools.run_b_internal_ablation import (
    B_SIGNAL_MODES,
    SCENES,
    Scene,
    XYZ_SCENE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract renders and dominant visible-Gaussian IDs for the "
            "B-internal ablation."
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument(
        "--scene",
        choices=tuple(SCENES),
        default=XYZ_SCENE.name,
    )
    parser.add_argument("--frame-name", default="001441.png")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _scene_path(
    root: Path,
    mode: str,
    repeat: int,
    scene: Scene,
) -> Path:
    dataset_directory = scene.dataset.replace(" ", "_")
    return (
        root
        / mode
        / f"repeat_{repeat}"
        / dataset_directory
        / scene.name
    )


def _find_keyframe_id(scene_model: SceneModel, frame_name: str) -> int:
    matches = [
        index
        for index, keyframe in enumerate(scene_model.keyframes)
        if Path(keyframe.info["name"]).name == frame_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one keyframe named {frame_name}, found {matches}"
        )
    return matches[0]


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    return rgba[..., :3], rgba[..., 3] > 0


def _load_render(
    render_path: Path,
    package: dict[str, torch.Tensor],
) -> tuple[np.ndarray, str]:
    if render_path.is_file():
        return (
            np.asarray(
                Image.open(render_path).convert("RGB"),
                dtype=np.uint8,
            ),
            "saved_test_image",
        )
    render = (
        package["render"]
        .detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return render, "rasterizer"


@torch.inference_mode()
def extract_scene(
    scene_path: Path,
    source_image: Path,
    frame_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with (scene_path / "metadata.json").open(
        "r", encoding="utf-8"
    ) as source:
        metadata = json.load(source)
    inference_args = get_args(
        [
            "-s",
            str(source_image.parent.parent),
            "-m",
            str(scene_path),
            "--viewer_mode",
            "none",
        ]
    )
    scene_model = SceneModel.from_scene(str(scene_path), inference_args)
    scene_model.f = float(metadata["config"]["f"])
    scene_model.init_intrinsics()
    keyframe_id = _find_keyframe_id(scene_model, frame_name)
    package = scene_model.render_from_id(keyframe_id, pyr_lvl=0)
    render_path = scene_path / "test_images" / frame_name
    render, render_source = _load_render(render_path, package)
    id_map = package["mainGaussID"][0].int().cpu().numpy()
    if render.shape[:2] != id_map.shape:
        raise ValueError("render and Gaussian-ID map shapes differ")
    valid_ids = id_map[id_map >= 0]
    record = {
        "scene_dir": str(scene_path),
        "keyframe_id": keyframe_id,
        "keyframe_name": frame_name,
        "restored_focal_pixels": scene_model.f,
        "render_source": render_source,
        "render_path": str(render_path) if render_path.is_file() else None,
        "visible_unique_gaussians": int(np.unique(valid_ids).size),
        "total_blended_gaussians": int(scene_model.xyz.shape[0]),
    }
    del package, scene_model
    gc.collect()
    torch.cuda.empty_cache()
    return render, id_map, record


def main() -> int:
    args = parse_args()
    scene = SCENES[args.scene]
    ground_truth, valid_mask = load_ground_truth(args.source_image)
    arrays: dict[str, np.ndarray] = {
        "ground_truth": ground_truth,
        "valid_mask": valid_mask,
    }
    manifest = []
    for mode in B_SIGNAL_MODES:
        for repeat in range(1, args.repeat + 1):
            scene_path = _scene_path(
                args.experiment_root,
                mode,
                repeat,
                scene,
            )
            if not (scene_path / "metadata.json").is_file():
                raise FileNotFoundError(scene_path / "metadata.json")
            render, id_map, record = extract_scene(
                scene_path,
                args.source_image,
                args.frame_name,
            )
            prefix = f"{mode}_repeat_{repeat}"
            arrays[f"{prefix}_render"] = render
            arrays[f"{prefix}_ids"] = id_map
            manifest.append(
                {
                    "signal_mode": mode,
                    "repeat": repeat,
                    **record,
                }
            )
            print(
                f"{mode} repeat {repeat}: "
                f"{record['visible_unique_gaussians']} visible Gaussians",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
