#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_b_internal_ablation import SCENES, Scene, XYZ_SCENE


EVIDENCE_MODES = ("base", "r_e_d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract final renders, dominant Gaussian IDs, and exact "
            "B-module sampling probabilities for Base and Ours."
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument(
        "--scene",
        choices=tuple(SCENES),
        default=XYZ_SCENE.name,
    )
    parser.add_argument("--frame-name", required=True)
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
    if mode not in EVIDENCE_MODES:
        raise ValueError(f"unsupported evidence mode: {mode}")
    dataset_directory = scene.dataset.replace(" ", "_")
    return (
        root
        / mode
        / f"repeat_{repeat}"
        / dataset_directory
        / scene.name
    )


def _load_sampling_trace(
    scene_path: Path,
    frame_name: str,
) -> dict[str, np.ndarray]:
    trace_path = (
        Path(scene_path)
        / "sampling_trace"
        / f"{Path(frame_name).stem}.npz"
    )
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    with np.load(trace_path) as archive:
        arrays = {
            key: np.asarray(archive[key]).copy()
            for key in (
                "base_probability",
                "final_probability",
                "sample_mask",
            )
        }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent trace shapes: {shapes}")
    final = arrays["final_probability"]
    if float(final.min(initial=0.0)) < 0.0 or float(
        final.max(initial=0.0)
    ) > 1.0:
        raise ValueError("traced Bernoulli probabilities must lie in [0, 1]")
    return arrays


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    return rgba[..., :3], rgba[..., 3] > 0


def _find_keyframe_id(scene_model: Any, frame_name: str) -> int:
    matches = [
        index
        for index, keyframe in enumerate(scene_model.keyframes)
        if Path(keyframe.info["name"]).name == Path(frame_name).name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one keyframe named {frame_name}, found {matches}"
        )
    return matches[0]


def extract_scene(
    scene_path: Path,
    source_image: Path,
    frame_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import torch

    from args import get_args
    from scene.scene_model import SceneModel

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
    with torch.inference_mode():
        scene_model = SceneModel.from_scene(str(scene_path), inference_args)
        scene_model.f = float(metadata["config"]["f"])
        scene_model.init_intrinsics()
        keyframe_id = _find_keyframe_id(scene_model, frame_name)
        package = scene_model.render_from_id(keyframe_id, pyr_lvl=0)
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
    id_map = package["mainGaussID"][0].int().cpu().numpy()
    trace = _load_sampling_trace(scene_path, frame_name)
    if render.shape[:2] != id_map.shape:
        raise ValueError("render and Gaussian-ID map shapes differ")
    if render.shape[:2] != trace["final_probability"].shape:
        raise ValueError("render and probability-map shapes differ")
    valid_ids = id_map[id_map >= 0]
    arrays = {
        "render": render,
        "ids": id_map,
        **trace,
    }
    record = {
        "scene_dir": str(scene_path),
        "keyframe_id": keyframe_id,
        "keyframe_name": Path(frame_name).name,
        "restored_focal_pixels": scene_model.f,
        "visible_unique_gaussians": int(np.unique(valid_ids).size),
        "total_blended_gaussians": int(scene_model.xyz.shape[0]),
        "full_frame_expected_samples": float(
            trace["final_probability"].sum(dtype=np.float64)
        ),
        "full_frame_realized_samples": int(trace["sample_mask"].sum()),
    }
    del package, scene_model
    gc.collect()
    torch.cuda.empty_cache()
    return arrays, record


def main() -> int:
    args = parse_args()
    scene = SCENES[args.scene]
    ground_truth, valid_mask = load_ground_truth(args.source_image)
    output_arrays: dict[str, np.ndarray] = {
        "ground_truth": ground_truth,
        "valid_mask": valid_mask,
    }
    manifest = []
    for mode in EVIDENCE_MODES:
        for repeat in range(1, args.repeat + 1):
            scene_path = _scene_path(
                args.experiment_root,
                mode,
                repeat,
                scene,
            )
            arrays, record = extract_scene(
                scene_path,
                args.source_image,
                args.frame_name,
            )
            prefix = f"{mode}_repeat_{repeat}"
            for name, array in arrays.items():
                output_arrays[f"{prefix}_{name}"] = array
            manifest.append(
                {
                    "signal_mode": mode,
                    "repeat": repeat,
                    **record,
                }
            )
            print(
                f"{mode} repeat {repeat}: "
                f"{record['visible_unique_gaussians']} visible, "
                f"mu={record['full_frame_expected_samples']:.3f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_arrays)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
