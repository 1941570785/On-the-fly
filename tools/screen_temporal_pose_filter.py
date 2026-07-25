#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_a_v2_baseline_pose_experiment import (  # noqa: E402
    EVALUATION_SCENE_DIRS,
    POSE_FIELDS,
    _dataset_name,
    _evaluation_names,
    _invert_pose_map,
    _load_metadata,
    reference_scene_spec,
)
from tools.evaluate_official_pose_benchmark import load_reference  # noqa: E402
from paper_table5_pose_eval import (  # noqa: E402
    canonical_frame_id,
    compute_paper_pose_metrics,
)


DEFAULT_RUN_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/comparison_results/"
    "a_v21_strict_pilot_20260723"
)
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
BASELINE = {
    "MipNeRF360": (11.4, 0.035, 16.6, 0.047),
    "StaticHikes": (22.859, 0.0462, 2.267, 0.0058),
    "TUM": (40.2, 0.313, 5.7, 0.045),
}


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    angle = math.acos(float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    return vector * (0.5 * angle / math.sin(angle))


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _ordered_pose_items(
    trajectory: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    def key(item: tuple[str, np.ndarray]) -> tuple[int, str]:
        frame_id = canonical_frame_id(item[0])
        digits = "".join(character for character in frame_id if character.isdigit())
        return (int(digits) if digits else 10**12, frame_id)

    return sorted(trajectory.items(), key=key)


def _triggered_frame_ids(model_dir: Path) -> set[str]:
    path = model_dir / "pose_initialization_risk_trace.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: set[str] = set()
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        verification = event.get("verification", {})
        if not isinstance(verification, dict):
            verification = {}
        if bool(
            event.get("verification_trigger", False)
            or verification.get("triggered", False)
        ):
            output.add(canonical_frame_id(str(event.get("image_name", ""))))
    return output


def _predict_next(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    predicted = np.eye(4, dtype=np.float64)
    predicted[:3, 3] = current[:3, 3] + (
        current[:3, 3] - previous[:3, 3]
    )
    rotation_step = previous[:3, :3].T @ current[:3, :3]
    predicted[:3, :3] = current[:3, :3] @ rotation_step
    return predicted


def _blend_pose(current: np.ndarray, predicted: np.ndarray, weight: float) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = (
        (1.0 - weight) * current[:3, 3]
        + weight * predicted[:3, 3]
    )
    delta = current[:3, :3].T @ predicted[:3, :3]
    output[:3, :3] = current[:3, :3] @ _so3_exp(
        weight * _so3_log(delta)
    )
    return output


def _motion_residual(
    previous: np.ndarray,
    current: np.ndarray,
    history: list[tuple[float, float]],
) -> float:
    translation = float(
        np.linalg.norm(current[:3, 3] - previous[:3, 3])
    )
    rotation = float(
        np.linalg.norm(_so3_log(previous[:3, :3].T @ current[:3, :3]))
    )
    if len(history) < 4:
        return 0.0
    recent = np.asarray(history[-9:], dtype=np.float64)
    median = np.median(recent, axis=0)
    mad = np.median(np.abs(recent - median), axis=0)
    scale = np.maximum(1.4826 * mad, np.array([1e-8, 1e-6]))
    z = np.abs(np.array([translation, rotation]) - median) / scale
    return float(np.max(z))


def causal_temporal_filter(
    trajectory: dict[str, np.ndarray],
    *,
    weight: float,
    scope: str,
    triggered_ids: set[str],
    motion_z: float,
) -> dict[str, np.ndarray]:
    ordered = _ordered_pose_items(trajectory)
    filtered: dict[str, np.ndarray] = {}
    history: list[tuple[float, float]] = []
    previous_outputs: list[np.ndarray] = []
    for name, raw_pose in ordered:
        frame_id = canonical_frame_id(name)
        current = np.asarray(raw_pose, dtype=np.float64)
        output = current.copy()
        residual = (
            _motion_residual(previous_outputs[-1], current, history)
            if previous_outputs
            else 0.0
        )
        active = scope == "all"
        if scope == "triggered":
            active = frame_id in triggered_ids
        elif scope == "motion":
            active = residual >= motion_z
        elif scope == "triggered_or_motion":
            active = frame_id in triggered_ids or residual >= motion_z
        if active and len(previous_outputs) >= 2:
            predicted = _predict_next(
                previous_outputs[-2],
                previous_outputs[-1],
            )
            output = _blend_pose(current, predicted, weight)
        if previous_outputs:
            history.append(
                (
                    float(
                        np.linalg.norm(
                            output[:3, 3] - previous_outputs[-1][:3, 3]
                        )
                    ),
                    float(
                        np.linalg.norm(
                            _so3_log(
                                previous_outputs[-1][:3, :3].T
                                @ output[:3, :3]
                            )
                        )
                    ),
                )
            )
        previous_outputs.append(output)
        filtered[frame_id] = output
    return filtered


def prepare_scenes(run_root: Path) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for scene in SCENES:
        model_dir = (
            run_root
            / scene
            / "repeat_01"
            / "a_v21_strict"
            / scene
            / "model"
        )
        trajectory, test_names = _load_metadata(model_dir)
        prepared.append(
            {
                "scene": scene,
                "dataset": _dataset_name(scene),
                "trajectory": trajectory,
                "triggered_ids": _triggered_frame_ids(model_dir),
                "reference": _invert_pose_map(
                    load_reference(reference_scene_spec(scene))
                ),
                "evaluation_names": _evaluation_names(scene, test_names),
            }
        )
    return prepared


def evaluate_configuration(
    prepared_scenes: list[dict[str, object]],
    *,
    scope: str,
    weight: float,
    motion_z: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prepared in prepared_scenes:
        scene = str(prepared["scene"])
        filtered = causal_temporal_filter(
            prepared["trajectory"],
            weight=weight,
            scope=scope,
            triggered_ids=prepared["triggered_ids"],
            motion_z=motion_z,
        )
        metrics = compute_paper_pose_metrics(
            filtered,
            prepared["reference"],
            prepared["evaluation_names"],
        )
        rows.append(
            {
                "scope": scope,
                "weight": weight,
                "motion_z": motion_z,
                "dataset": prepared["dataset"],
                "scene": scene,
                **{field: float(metrics[field]) for field in POSE_FIELDS},
            }
        )
    return rows


def dataset_macro(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    output: list[dict[str, object]] = []
    for dataset, selected in grouped.items():
        output.append(
            {
                "scope": selected[0]["scope"],
                "weight": selected[0]["weight"],
                "motion_z": selected[0]["motion_z"],
                "dataset": dataset,
                **{
                    field: float(np.mean([float(row[field]) for row in selected]))
                    for field in POSE_FIELDS
                },
            }
        )
    return output


def score_configuration(rows: list[dict[str, object]]) -> tuple[float, float]:
    ratios: list[float] = []
    for row in rows:
        baseline = BASELINE[str(row["dataset"])]
        ratios.extend(
            float(row[field]) / baseline[index]
            for index, field in enumerate(POSE_FIELDS)
        )
    return float(np.mean(ratios)), float(np.max(ratios))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN_ROOT / "temporal_filter_screen",
    )
    args = parser.parse_args()

    prepared_scenes = prepare_scenes(args.run_root)
    all_scene_rows: list[dict[str, object]] = []
    all_macro_rows: list[dict[str, object]] = []
    ranking: list[dict[str, object]] = []
    for scope in ("all", "triggered", "motion", "triggered_or_motion"):
        for weight in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            for motion_z in ((2.0, 3.0, 4.0) if "motion" in scope else (3.0,)):
                scene_rows = evaluate_configuration(
                    prepared_scenes,
                    scope=scope,
                    weight=weight,
                    motion_z=motion_z,
                )
                macro_rows = dataset_macro(scene_rows)
                mean_ratio, worst_ratio = score_configuration(macro_rows)
                all_scene_rows.extend(scene_rows)
                all_macro_rows.extend(macro_rows)
                ranking.append(
                    {
                        "scope": scope,
                        "weight": weight,
                        "motion_z": motion_z,
                        "mean_baseline_ratio": mean_ratio,
                        "worst_baseline_ratio": worst_ratio,
                    }
                )
    ranking.sort(
        key=lambda row: (
            float(row["mean_baseline_ratio"]),
            float(row["worst_baseline_ratio"]),
        )
    )
    write_csv(args.output / "scene.csv", all_scene_rows)
    write_csv(args.output / "dataset_macro.csv", all_macro_rows)
    write_csv(args.output / "ranking.csv", ranking)
    print(json.dumps(ranking[:10], indent=2))


if __name__ == "__main__":
    main()
