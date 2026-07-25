#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMPARISON_TOOLS = Path("/data2/zxd/3D_Reconstruction/comparison_tools")
if str(COMPARISON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMPARISON_TOOLS))

from paper_table5_pose_eval import (  # noqa: E402
    _apply_similarity_c2w,
    _fit_similarity_c2w,
    _rotation_error_rad,
    canonical_frame_id,
    compute_paper_pose_metrics,
)
from tools.evaluate_a_v2_baseline_pose_experiment import (  # noqa: E402
    _dataset_name,
    _evaluation_names,
    reference_scene_spec,
)
from tools.evaluate_official_pose_benchmark import (  # noqa: E402
    load_reference,
    valid_pose,
)


POSE_FIELDS = ("T_APE", "R_APE", "T_RPE", "R_RPE")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(
        np.clip((np.trace(first.T @ second) - 1.0) * 0.5, -1.0, 1.0)
    )
    return math.acos(cosine)


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    return skew * (0.5 * angle / max(math.sin(angle), 1e-8))


def _trajectory_scales(poses: list[np.ndarray]) -> tuple[float, float]:
    translation_steps = [
        float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
        for previous, current in zip(poses[:-1], poses[1:])
    ]
    rotation_steps = [
        _rotation_angle(previous[:3, :3], current[:3, :3])
        for previous, current in zip(poses[:-1], poses[1:])
    ]
    return (
        max(float(np.median(translation_steps)), 1e-8),
        max(float(np.median(rotation_steps)), math.radians(0.01)),
    )


def _bidirectional_consistency(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    *,
    previous_gap: int,
    following_gap: int,
    translation_scale: float,
    rotation_scale: float,
) -> float:
    previous_dt = float(max(previous_gap, 1))
    following_dt = float(max(following_gap, 1))
    velocity_before = (current[:3, 3] - previous[:3, 3]) / previous_dt
    velocity_after = (following[:3, 3] - current[:3, 3]) / following_dt
    translation_acceleration = float(
        np.linalg.norm(velocity_after - velocity_before)
    )
    rotation_before = _so3_log(
        previous[:3, :3].T @ current[:3, :3]
    ) / previous_dt
    rotation_after = _so3_log(
        current[:3, :3].T @ following[:3, :3]
    ) / following_dt
    rotation_acceleration = float(np.linalg.norm(rotation_after - rotation_before))
    return math.sqrt(
        (translation_acceleration / translation_scale) ** 2
        + (rotation_acceleration / rotation_scale) ** 2
    )


def _fixed_absolute_errors(
    pose_c2w: np.ndarray,
    reference_c2w: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, float]:
    aligned = _apply_similarity_c2w(
        pose_c2w[None],
        scale=scale,
        rotation=rotation,
        translation=translation,
    )[0]
    translation_error = float(
        np.linalg.norm(aligned[:3, 3] - reference_c2w[:3, 3])
    )
    rotation_error = float(
        _rotation_error_rad(
            aligned[:3, :3][None],
            reference_c2w[:3, :3][None],
        )[0]
    )
    return translation_error, rotation_error


def load_scene(trace_path: Path, scene: str) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    raw_events = [
        event for event in trace.get("events", []) if isinstance(event, dict)
    ]
    events: list[dict[str, Any]] = []
    for event in raw_events:
        initial_w2c = valid_pose(event.get("initial_Rt"))
        candidate_w2c = valid_pose(event.get("candidate_Rt"))
        name = str(event.get("image_name", ""))
        if initial_w2c is None or not name:
            continue
        events.append(
            {
                **event,
                "_name": name,
                "_frame_id": canonical_frame_id(name),
                "_initial_c2w": np.linalg.inv(initial_w2c),
                "_candidate_c2w": (
                    np.linalg.inv(candidate_w2c)
                    if candidate_w2c is not None
                    else None
                ),
            }
        )
    events.sort(key=lambda item: int(item.get("position", 0)))
    initial_poses = [event["_initial_c2w"] for event in events]
    translation_scale, rotation_scale = _trajectory_scales(initial_poses)
    initial_by_name = {
        event["_frame_id"]: event["_initial_c2w"] for event in events
    }
    test_names = [
        event["_name"] for event in events if bool(event.get("is_test", False))
    ]
    evaluation_names = _evaluation_names(scene, test_names)
    reference_w2c = load_reference(reference_scene_spec(scene))
    reference = {
        canonical_frame_id(name): np.linalg.inv(pose)
        for name, pose in reference_w2c.items()
    }
    evaluation_ids = [
        canonical_frame_id(name)
        for name in evaluation_names
        if canonical_frame_id(name) in initial_by_name
        and canonical_frame_id(name) in reference
    ]
    initial_eval = np.stack([initial_by_name[name] for name in evaluation_ids])
    reference_eval = np.stack([reference[name] for name in evaluation_ids])
    alignment = _fit_similarity_c2w(initial_eval, reference_eval)
    initial_metrics = compute_paper_pose_metrics(
        initial_by_name,
        reference,
        evaluation_names,
    )

    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        candidate = event["_candidate_c2w"]
        if candidate is None:
            continue
        initial = event["_initial_c2w"]
        temporal_initial = float("inf")
        temporal_candidate = float("inf")
        temporal_ratio = float("inf")
        if 0 < index < len(events) - 1:
            previous = events[index - 1]
            following = events[index + 1]
            previous_gap = int(event.get("position", index)) - int(
                previous.get("position", index - 1)
            )
            following_gap = int(following.get("position", index + 1)) - int(
                event.get("position", index)
            )
            temporal_initial = _bidirectional_consistency(
                previous["_initial_c2w"],
                initial,
                following["_initial_c2w"],
                previous_gap=previous_gap,
                following_gap=following_gap,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
            )
            temporal_candidate = _bidirectional_consistency(
                previous["_initial_c2w"],
                candidate,
                following["_initial_c2w"],
                previous_gap=previous_gap,
                following_gap=following_gap,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
            )
            temporal_ratio = temporal_candidate / max(temporal_initial, 1e-8)

        frame_id = event["_frame_id"]
        initial_gt_translation = float("nan")
        candidate_gt_translation = float("nan")
        initial_gt_rotation = float("nan")
        candidate_gt_rotation = float("nan")
        if frame_id in reference:
            initial_gt_translation, initial_gt_rotation = _fixed_absolute_errors(
                initial,
                reference[frame_id],
                scale=alignment[0],
                rotation=alignment[1],
                translation=alignment[2],
            )
            candidate_gt_translation, candidate_gt_rotation = (
                _fixed_absolute_errors(
                    candidate,
                    reference[frame_id],
                    scale=alignment[0],
                    rotation=alignment[1],
                    translation=alignment[2],
                )
            )
        candidate_debug = event.get("candidate_pose_debug", {})
        if not isinstance(candidate_debug, dict):
            candidate_debug = {}
        correction_translation = float(
            event.get("correction_translation", float("inf"))
        )
        correction_rotation = math.radians(
            float(event.get("correction_rotation_deg", float("inf")))
        )
        rows.append(
            {
                "dataset": _dataset_name(scene),
                "scene": scene,
                "position": int(event.get("position", index)),
                "frame_id": frame_id,
                "is_test": bool(event.get("is_test", False)),
                "original_accepted": bool(event.get("accepted", False)),
                "validation_support": int(event.get("validation_support", 0) or 0),
                "median_gain": float(
                    event.get("relative_median_improvement", float("-inf"))
                ),
                "mean_ratio": float(event.get("mean_ratio", float("inf"))),
                "p90_ratio": float(event.get("p90_ratio", float("inf"))),
                "correction_translation": correction_translation,
                "correction_rotation_rad": correction_rotation,
                "correction_translation_step_ratio": correction_translation
                / translation_scale,
                "correction_rotation_step_ratio": correction_rotation
                / rotation_scale,
                "temporal_initial": temporal_initial,
                "temporal_candidate": temporal_candidate,
                "temporal_ratio": temporal_ratio,
                "pnp_inliers": int(
                    candidate_debug.get("num_pnp_inliers", 0) or 0
                ),
                "miniba_inliers": int(
                    candidate_debug.get("num_miniba_inliers", 0) or 0
                ),
                "correspondences": int(
                    candidate_debug.get("num_2d3d_correspondences", 0) or 0
                ),
                "initial_gt_translation": initial_gt_translation,
                "candidate_gt_translation": candidate_gt_translation,
                "initial_gt_rotation": initial_gt_rotation,
                "candidate_gt_rotation": candidate_gt_rotation,
                "gt_translation_improved": candidate_gt_translation
                < initial_gt_translation,
                "gt_rotation_improved": candidate_gt_rotation
                < initial_gt_rotation,
                "gt_both_improved": candidate_gt_translation
                < initial_gt_translation
                and candidate_gt_rotation < initial_gt_rotation,
                "_candidate_c2w": candidate,
            }
        )
    return {
        "scene": scene,
        "events": events,
        "rows": rows,
        "initial_by_name": initial_by_name,
        "reference": reference,
        "evaluation_names": evaluation_names,
        "initial_metrics": initial_metrics,
    }


def _candidate_passes(row: dict[str, Any], gate: tuple[float, ...]) -> bool:
    (
        min_median_gain,
        max_mean_ratio,
        max_p90_ratio,
        max_translation_step_ratio,
        max_rotation_step_ratio,
        max_temporal_ratio,
    ) = gate
    return bool(
        row["is_test"]
        and row["validation_support"] >= 24
        and row["median_gain"] >= min_median_gain
        and row["mean_ratio"] <= max_mean_ratio
        and row["p90_ratio"] <= max_p90_ratio
        and row["correction_translation_step_ratio"]
        <= max_translation_step_ratio
        and row["correction_rotation_step_ratio"] <= max_rotation_step_ratio
        and row["temporal_ratio"] <= max_temporal_ratio
    )


def replay_scene(
    scene_data: dict[str, Any],
    gate: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trajectory = dict(scene_data["initial_by_name"])
    selected = [
        row for row in scene_data["rows"] if _candidate_passes(row, gate)
    ]
    for row in selected:
        trajectory[row["frame_id"]] = row["_candidate_c2w"]
    metrics = compute_paper_pose_metrics(
        trajectory,
        scene_data["reference"],
        scene_data["evaluation_names"],
    )
    return metrics, selected


def gate_grid() -> Iterable[tuple[float, ...]]:
    return itertools.product(
        (0.03, 0.05, 0.08, 0.10, 0.15),
        (0.99, 0.98, 0.95, 0.90),
        (1.01, 1.00, 0.98, 0.95),
        (0.25, 0.50, 1.00, 2.00),
        (0.25, 0.50, 1.00, 2.00),
        (0.75, 0.90, 1.00, 1.10),
    )


def summarize_gates(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in gate_grid():
        metric_ratios: list[float] = []
        selected_rows: list[dict[str, Any]] = []
        output: dict[str, Any] = {
            "min_median_gain": gate[0],
            "max_mean_ratio": gate[1],
            "max_p90_ratio": gate[2],
            "max_translation_step_ratio": gate[3],
            "max_rotation_step_ratio": gate[4],
            "max_temporal_ratio": gate[5],
        }
        for scene_data in scenes:
            replay, selected = replay_scene(scene_data, gate)
            selected_rows.extend(selected)
            scene = str(scene_data["scene"])
            initial = scene_data["initial_metrics"]
            output[f"{scene}_accepted_test"] = len(selected)
            for field in POSE_FIELDS:
                ratio = float(replay[field]) / max(float(initial[field]), 1e-12)
                metric_ratios.append(ratio)
                output[f"{scene}_{field}"] = float(replay[field])
                output[f"{scene}_{field}_ratio"] = ratio
        output["accepted_test"] = len(selected_rows)
        output["both_gt_improved_rate"] = (
            float(np.mean([bool(row["gt_both_improved"]) for row in selected_rows]))
            if selected_rows
            else 0.0
        )
        output["mean_metric_ratio"] = float(np.mean(metric_ratios))
        output["worst_metric_ratio"] = float(np.max(metric_ratios))
        output["all_metrics_improved"] = all(value < 1.0 for value in metric_ratios)
        rows.append(output)
    rows.sort(
        key=lambda row: (
            not bool(row["all_metrics_improved"]),
            float(row["worst_metric_ratio"]),
            float(row["mean_metric_ratio"]),
            -int(row["accepted_test"]),
        )
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        nargs=2,
        action="append",
        metavar=("SCENE", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        load_scene(Path(path), scene) for scene, path in args.trace
    ]
    candidate_rows: list[dict[str, Any]] = []
    for scene in scenes:
        for row in scene["rows"]:
            candidate_rows.append(
                {key: value for key, value in row.items() if not key.startswith("_")}
            )
    gates = summarize_gates(scenes)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "candidate_rows.csv", candidate_rows)
    _write_csv(args.output / "gate_grid.csv", gates[:250])
    summary = {
        "scenes": {
            scene["scene"]: {
                "initial_metrics": {
                    field: scene["initial_metrics"][field] for field in POSE_FIELDS
                },
                "candidate_count": len(scene["rows"]),
                "test_candidate_count": sum(
                    bool(row["is_test"]) for row in scene["rows"]
                ),
                "original_accepted_test": sum(
                    bool(row["is_test"]) and bool(row["original_accepted"])
                    for row in scene["rows"]
                ),
            }
            for scene in scenes
        },
        "top_gate": gates[0] if gates else None,
        "all_metrics_improved_gate_count": sum(
            bool(row["all_metrics_improved"]) for row in gates
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
