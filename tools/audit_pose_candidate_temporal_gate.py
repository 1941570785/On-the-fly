#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
)
from poses.pose_verification import (  # noqa: E402
    interpolate_world_to_camera_pose,
    validation_candidate_rank,
)
from tools.evaluate_a_v2_baseline_pose_experiment import (  # noqa: E402
    reference_scene_spec,
)
from tools.evaluate_official_pose_benchmark import (  # noqa: E402
    load_reference,
    valid_pose,
)


def _camera_centre(w2c: np.ndarray) -> np.ndarray:
    return -(w2c[:3, :3].T @ w2c[:3, 3])


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(
        np.clip(
            (np.trace(first @ second.T) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
    )
    return math.acos(cosine)


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    angle = _rotation_angle(rotation, np.eye(3, dtype=np.float64))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    return vector * (0.5 * angle / max(math.sin(angle), 1e-8))


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
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


def _predict_w2c(
    history: list[tuple[int, np.ndarray]],
    frame_id: int,
) -> np.ndarray | None:
    if len(history) < 2:
        return None
    previous_id, previous = history[-2]
    last_id, last = history[-1]
    previous_c2w = np.linalg.inv(previous)
    last_c2w = np.linalg.inv(last)
    history_gap = max(last_id - previous_id, 1)
    current_gap = max(frame_id - last_id, 1)
    ratio = float(current_gap) / float(history_gap)
    predicted_c2w = np.eye(4, dtype=np.float64)
    predicted_c2w[:3, 3] = last_c2w[:3, 3] + ratio * (
        last_c2w[:3, 3] - previous_c2w[:3, 3]
    )
    rotation_step = previous_c2w[:3, :3].T @ last_c2w[:3, :3]
    predicted_c2w[:3, :3] = (
        last_c2w[:3, :3] @ _so3_exp(ratio * _so3_log(rotation_step))
    )
    return np.linalg.inv(predicted_c2w)


def _history_scales(
    history: list[tuple[int, np.ndarray]],
) -> tuple[float, float]:
    translations: list[float] = []
    rotations: list[float] = []
    for (_, previous), (_, current) in zip(history[-10:-1], history[-9:]):
        translations.append(
            float(
                np.linalg.norm(
                    _camera_centre(current) - _camera_centre(previous)
                )
            )
        )
        rotations.append(
            _rotation_angle(current[:3, :3], previous[:3, :3])
        )
    return (
        max(float(np.median(translations)) if translations else 0.0, 1e-4),
        max(
            float(np.median(rotations)) if rotations else 0.0,
            math.radians(0.05),
        ),
    )


def _temporal_score(
    pose: np.ndarray,
    prediction: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> tuple[float, float, float]:
    translation = float(
        np.linalg.norm(_camera_centre(pose) - _camera_centre(prediction))
    )
    rotation = _rotation_angle(pose[:3, :3], prediction[:3, :3])
    score = math.sqrt(
        (translation / translation_scale) ** 2
        + (rotation / rotation_scale) ** 2
    )
    return score, translation, rotation


def _aligned_errors(
    pose_w2c: np.ndarray,
    reference_w2c: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, float]:
    aligned = _apply_similarity_c2w(
        np.linalg.inv(pose_w2c)[None],
        scale=scale,
        rotation=rotation,
        translation=translation,
    )[0]
    reference_c2w = np.linalg.inv(reference_w2c)
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


def audit_trace(
    trace_path: Path,
    scene: str,
) -> list[dict[str, Any]]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    reference = {
        canonical_frame_id(name): pose
        for name, pose in load_reference(reference_scene_spec(scene)).items()
    }
    events = [
        event
        for event in trace.get("events", [])
        if isinstance(event, dict)
        and isinstance(event.get("verification"), dict)
    ]
    anchors: list[tuple[np.ndarray, np.ndarray]] = []
    for event in events:
        frame_id = canonical_frame_id(str(event.get("image_name", "")))
        initial = valid_pose(
            event.get("initial_estimated_Rt")
            or event["verification"].get("initial_Rt")
        )
        if initial is not None and frame_id in reference:
            anchors.append((np.linalg.inv(initial), np.linalg.inv(reference[frame_id])))
    if len(anchors) < 3:
        return []
    scale, rotation, translation = _fit_similarity_c2w(
        np.stack([item[0] for item in anchors]),
        np.stack([item[1] for item in anchors]),
    )

    rows: list[dict[str, Any]] = []
    history: list[tuple[int, np.ndarray]] = []
    for event in events:
        verification = event["verification"]
        initial = valid_pose(
            event.get("initial_estimated_Rt") or verification.get("initial_Rt")
        )
        post_a = valid_pose(
            event.get("post_a_Rt")
            or event.get("estimated_Rt")
            or verification.get("final_Rt")
        )
        numeric_frame_id = int(
            verification.get("frame_id", event.get("frame_id", -1))
        )
        canonical_id = canonical_frame_id(str(event.get("image_name", "")))
        prediction = _predict_w2c(history, numeric_frame_id)
        if (
            initial is not None
            and prediction is not None
            and canonical_id in reference
            and bool(verification.get("attempted", False))
        ):
            translation_scale, rotation_scale = _history_scales(history)
            initial_temporal = _temporal_score(
                initial,
                prediction,
                translation_scale,
                rotation_scale,
            )
            initial_gt = _aligned_errors(
                initial,
                reference[canonical_id],
                scale=scale,
                rotation=rotation,
                translation=translation,
            )
            solver_candidates = {
                str(item.get("name")): valid_pose(item.get("pose"))
                for item in verification.get("solver_candidates", [])
                if isinstance(item, dict)
            }
            pre = {
                field: verification.get(f"pre_reprojection_{field}")
                for field in ("valid_count", "mean", "median", "p90", "max")
            }
            for candidate_row in verification.get("step_candidates", []):
                if not isinstance(candidate_row, dict):
                    continue
                if not bool(candidate_row.get("accepted", False)):
                    continue
                source = str(candidate_row.get("source", ""))
                base = solver_candidates.get(source)
                if base is None:
                    continue
                alpha = float(candidate_row.get("alpha", 1.0))
                candidate = (
                    interpolate_world_to_camera_pose(
                        torch.as_tensor(initial, dtype=torch.float64),
                        torch.as_tensor(base, dtype=torch.float64),
                        alpha,
                    )
                    .numpy()
                )
                candidate_temporal = _temporal_score(
                    candidate,
                    prediction,
                    translation_scale,
                    rotation_scale,
                )
                candidate_gt = _aligned_errors(
                    candidate,
                    reference[canonical_id],
                    scale=scale,
                    rotation=rotation,
                    translation=translation,
                )
                post = {
                    field: candidate_row.get(f"post_{field}")
                    for field in ("valid_count", "mean", "median", "p90", "max")
                }
                rank = validation_candidate_rank(pre, post)
                rows.append(
                    {
                        "scene": scene,
                        "frame_id": numeric_frame_id,
                        "source": source,
                        "alpha": alpha,
                        "selected": (
                            source
                            == str(
                                verification.get(
                                    "selected_candidate_source", ""
                                )
                            )
                            and abs(
                                alpha
                                - float(
                                    verification.get(
                                        "selected_step_alpha", -1.0
                                    )
                                )
                            )
                            < 1e-8
                        ),
                        "robust_rank": rank[0],
                        "temporal_score_initial": initial_temporal[0],
                        "temporal_score_candidate": candidate_temporal[0],
                        "temporal_score_ratio": candidate_temporal[0]
                        / max(initial_temporal[0], 1e-8),
                        "temporal_translation_initial": initial_temporal[1],
                        "temporal_translation_candidate": candidate_temporal[1],
                        "temporal_rotation_initial": initial_temporal[2],
                        "temporal_rotation_candidate": candidate_temporal[2],
                        "gt_translation_initial": initial_gt[0],
                        "gt_translation_candidate": candidate_gt[0],
                        "gt_rotation_initial": initial_gt[1],
                        "gt_rotation_candidate": candidate_gt[1],
                        "gt_translation_improved": candidate_gt[0] < initial_gt[0],
                        "gt_rotation_improved": candidate_gt[1] < initial_gt[1],
                        "gt_both_improved": (
                            candidate_gt[0] < initial_gt[0]
                            and candidate_gt[1] < initial_gt[1]
                        ),
                    }
                )
        if post_a is not None:
            history.append((numeric_frame_id, post_a))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ratio_limit in (0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50):
        selected = [
            row
            for row in rows
            if float(row["temporal_score_ratio"]) <= ratio_limit
        ]
        if not selected:
            continue
        best_by_frame: dict[tuple[str, int], dict[str, Any]] = {}
        for row in selected:
            key = (str(row["scene"]), int(row["frame_id"]))
            if key not in best_by_frame or float(row["robust_rank"]) < float(
                best_by_frame[key]["robust_rank"]
            ):
                best_by_frame[key] = row
        chosen = list(best_by_frame.values())
        output.append(
            {
                "temporal_ratio_limit": ratio_limit,
                "accepted_frames": len(chosen),
                "translation_improved_rate": float(
                    np.mean(
                        [bool(row["gt_translation_improved"]) for row in chosen]
                    )
                ),
                "rotation_improved_rate": float(
                    np.mean(
                        [bool(row["gt_rotation_improved"]) for row in chosen]
                    )
                ),
                "both_improved_rate": float(
                    np.mean([bool(row["gt_both_improved"]) for row in chosen])
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        action="append",
        nargs=2,
        metavar=("SCENE", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for scene, path in args.trace:
        rows.extend(audit_trace(Path(path), scene))
    gate_rows = summarize_gates(rows)
    _write_csv(args.output / "candidate_rows.csv", rows)
    _write_csv(args.output / "temporal_gate_summary.csv", gate_rows)

    selected = [row for row in rows if bool(row["selected"])]
    print(
        json.dumps(
            {
                "candidate_rows": len(rows),
                "selected_rows": len(selected),
                "selected_both_improved_rate": (
                    float(
                        np.mean(
                            [bool(row["gt_both_improved"]) for row in selected]
                        )
                    )
                    if selected
                    else 0.0
                ),
                "gates": gate_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
