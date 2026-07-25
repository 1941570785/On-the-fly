#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMPARISON_TOOLS = Path("/data2/zxd/3D_Reconstruction/comparison_tools")
if str(COMPARISON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMPARISON_TOOLS))

from paper_table5_pose_eval import compute_paper_pose_metrics  # noqa: E402
from poses.pose_verification import interpolate_world_to_camera_pose  # noqa: E402
from tools.evaluate_a_v2_baseline_pose_experiment import (  # noqa: E402
    POSE_FIELDS,
    _dataset_name,
    _evaluation_names,
    _invert_pose_map,
    _load_metadata,
    _load_stage_trace,
    reference_scene_spec,
)
from tools.evaluate_official_pose_benchmark import load_reference  # noqa: E402


SCENES = (
    "bonsai",
    "counter",
    "garden",
    "forest1",
    "forest2",
    "university2",
    "desk",
    "xyz",
    "long_office",
)


def model_dir(run_root: Path, scene: str, variant: str) -> Path:
    scene_root = run_root / scene
    if not scene_root.exists():
        scene_root = run_root
    return (
        scene_root
        / "repeat_01"
        / variant
        / scene
        / "model"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def blend_trajectories(
    post_a: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    final_weight: float,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in post_a.keys() & final.keys():
        post_w2c = torch.as_tensor(
            np.linalg.inv(post_a[name]), dtype=torch.float64
        )
        final_w2c = torch.as_tensor(
            np.linalg.inv(final[name]), dtype=torch.float64
        )
        blended_w2c = interpolate_world_to_camera_pose(
            post_w2c,
            final_w2c,
            final_weight,
        ).numpy()
        output[name] = np.linalg.inv(blended_w2c)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for scene in args.scenes:
        directory = model_dir(args.run_root, scene, args.variant)
        final, test_names = _load_metadata(directory)
        initial, post_a, _, _, _ = _load_stage_trace(directory)
        reference = _invert_pose_map(load_reference(reference_scene_spec(scene)))
        names = _evaluation_names(scene, test_names)
        stage_trajectories = [
            ("initial", initial),
            ("post_a", post_a),
            ("final", final),
        ]
        stage_trajectories.extend(
            (
                f"final_weight_{weight:.2f}",
                blend_trajectories(post_a, final, weight),
            )
            for weight in (0.25, 0.50, 0.75)
        )
        for stage, trajectory in stage_trajectories:
            metrics = compute_paper_pose_metrics(
                trajectory,
                reference,
                names,
            )
            rows.append(
                {
                    "dataset": _dataset_name(scene),
                    "scene": scene,
                    "stage": stage,
                    **{field: float(metrics[field]) for field in POSE_FIELDS},
                }
            )
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["stage"]))].append(row)
    macro = []
    for (dataset, stage), selected in sorted(grouped.items()):
        macro.append(
            {
                "dataset": dataset,
                "stage": stage,
                **{
                    field: float(np.mean([float(row[field]) for row in selected]))
                    for field in POSE_FIELDS
                },
            }
        )
    write_csv(args.output / "stage_scene.csv", rows)
    write_csv(args.output / "stage_dataset_macro.csv", macro)
    for row in macro:
        print(
            row["dataset"],
            row["stage"],
            *(f"{float(row[field]):.6f}" for field in POSE_FIELDS),
        )


if __name__ == "__main__":
    main()
