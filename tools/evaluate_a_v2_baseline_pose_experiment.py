#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    paper_test_names,
    sorted_image_names,
    summarize_errors,
)
from tools.evaluate_official_pose_benchmark import (  # noqa: E402
    SCENES as POSE_REFERENCE_SCENES,
    load_reference,
    valid_pose,
)
from tools.run_a_v2_baseline_pose_experiment import (  # noqa: E402
    SCENES,
    VARIANTS,
)


POSE_FIELDS = ("T_APE", "R_APE", "T_RPE", "R_RPE")
REFERENCE_ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly/datasets")
EVALUATION_SCENE_DIRS = {
    "bonsai": REFERENCE_ROOT / "MipNeRF360/bonsai",
    "counter": REFERENCE_ROOT / "MipNeRF360/counter",
    "garden": REFERENCE_ROOT / "MipNeRF360/garden",
    "forest1": REFERENCE_ROOT / "StaticHikes/forest1",
    "forest2": REFERENCE_ROOT / "StaticHikes/forest2",
    "university2": REFERENCE_ROOT / "StaticHikes/university2",
    "desk": REFERENCE_ROOT / "TUM/desk1",
    "xyz": REFERENCE_ROOT / "TUM/desk2",
    "long_office": REFERENCE_ROOT / "TUM/long_office_household",
}


def reference_scene_spec(scene_name: str) -> Any:
    if scene_name not in POSE_REFERENCE_SCENES:
        raise ValueError(f"unknown pose-reference scene: {scene_name}")
    return POSE_REFERENCE_SCENES[scene_name]


def _canonical_pose_map(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, value in values.items():
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            output[canonical_frame_id(name)] = pose
    return output


def _fixed_stage_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    aligned = _apply_similarity_c2w(
        predicted,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    ape_translation = np.linalg.norm(
        aligned[:, :3, 3] - reference[:, :3, 3], axis=1
    )
    ape_rotation = _rotation_error_rad(
        aligned[:, :3, :3], reference[:, :3, :3]
    )
    pred_relative = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    gt_relative = np.linalg.inv(reference[:-1]) @ reference[1:]
    relative_error = np.linalg.inv(gt_relative) @ pred_relative
    rpe_translation = np.linalg.norm(relative_error[:, :3, 3], axis=1)
    identity = np.repeat(
        np.eye(3, dtype=np.float64)[None], len(relative_error), axis=0
    )
    rpe_rotation = _rotation_error_rad(relative_error[:, :3, :3], identity)
    ape = summarize_errors(
        translation=ape_translation,
        rotation_rad=ape_rotation,
    )
    rpe = summarize_errors(
        translation=rpe_translation,
        rotation_rad=rpe_rotation,
    )
    metrics: dict[str, float | int] = {
        "T_APE": ape["translation_paper_rmse"],
        "R_APE": ape["rotation_rad_rmse"],
        "T_RPE": rpe["translation_paper_rmse"],
        "R_RPE": rpe["rotation_rad_rmse"],
        "T_APE_p95": 100.0 * float(np.percentile(ape_translation, 95)),
        "R_APE_p95": float(np.percentile(ape_rotation, 95)),
        "T_APE_max": 100.0 * float(np.max(ape_translation)),
        "R_APE_max": float(np.max(ape_rotation)),
        "ape_frames": int(len(aligned)),
        "rpe_pairs": int(len(relative_error)),
    }
    return metrics, ape_translation, ape_rotation


def compute_fixed_alignment_stage_report(
    reference_by_name: dict[str, np.ndarray],
    initial_by_name: dict[str, np.ndarray],
    post_a_by_name: dict[str, np.ndarray],
    final_by_name: dict[str, np.ndarray],
    evaluation_names: Iterable[str],
    *,
    accepted_frame_ids: set[str] | None = None,
) -> dict[str, Any]:
    reference = _canonical_pose_map(reference_by_name)
    initial = _canonical_pose_map(initial_by_name)
    post_a = _canonical_pose_map(post_a_by_name)
    final = _canonical_pose_map(final_by_name)
    accepted = {canonical_frame_id(value) for value in (accepted_frame_ids or set())}
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for name in evaluation_names:
        frame_id = canonical_frame_id(name)
        if frame_id in seen:
            continue
        if all(frame_id in values for values in (reference, initial, post_a, final)):
            seen.add(frame_id)
            ordered_ids.append(frame_id)
    if len(ordered_ids) < 3:
        raise ValueError("stage evaluation requires at least three common poses")

    alignment_ids = [frame_id for frame_id in ordered_ids if frame_id not in accepted]
    alignment_fallback = "none"
    if len(alignment_ids) < 3:
        alignment_ids = list(ordered_ids)
        alignment_fallback = "all_evaluation_frames"
    initial_anchor = np.stack([initial[frame_id] for frame_id in alignment_ids])
    reference_anchor = np.stack([reference[frame_id] for frame_id in alignment_ids])
    scale, rotation, translation = _fit_similarity_c2w(
        initial_anchor,
        reference_anchor,
    )
    gt = np.stack([reference[frame_id] for frame_id in ordered_ids])
    stage_maps = {"initial": initial, "post_a": post_a, "final": final}
    stage_metrics: dict[str, dict[str, float | int]] = {}
    frame_errors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stage, values in stage_maps.items():
        prediction = np.stack([values[frame_id] for frame_id in ordered_ids])
        metrics, translation_error, rotation_error = _fixed_stage_metrics(
            prediction,
            gt,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        stage_metrics[stage] = metrics
        frame_errors[stage] = (translation_error, rotation_error)

    accepted_indices = [
        index for index, frame_id in enumerate(ordered_ids) if frame_id in accepted
    ]
    both_improved = [
        index
        for index in accepted_indices
        if frame_errors["post_a"][0][index] < frame_errors["initial"][0][index]
        and frame_errors["post_a"][1][index] < frame_errors["initial"][1][index]
    ]
    retained = [
        index
        for index in both_improved
        if frame_errors["final"][0][index] < frame_errors["initial"][0][index]
        and frame_errors["final"][1][index] < frame_errors["initial"][1][index]
    ]
    initial_translation = frame_errors["initial"][0]
    initial_rotation = frame_errors["initial"][1]
    translation_median = float(np.median(initial_translation))
    rotation_median = float(np.median(initial_rotation))
    translation_mad = float(np.median(np.abs(initial_translation - translation_median)))
    rotation_mad = float(np.median(np.abs(initial_rotation - rotation_median)))
    translation_catastrophic = max(
        5.0 * translation_median,
        translation_median + 6.0 * 1.4826 * translation_mad,
        1e-12,
    )
    rotation_catastrophic = max(
        5.0 * rotation_median,
        rotation_median + 6.0 * 1.4826 * rotation_mad,
        math.radians(1.0),
    )
    for stage, (translation_error, rotation_error) in frame_errors.items():
        catastrophic = (translation_error > translation_catastrophic) | (
            rotation_error > rotation_catastrophic
        )
        stage_metrics[stage]["catastrophic_failure_rate"] = float(
            np.mean(catastrophic)
        )

    return {
        "evaluation_frame_ids": ordered_ids,
        "alignment_frame_ids": alignment_ids,
        "alignment_fallback": alignment_fallback,
        "alignment_scale": float(scale),
        "accepted_count": len(accepted_indices),
        "accepted_both_improved_count": len(both_improved),
        "accepted_both_improved_rate": len(both_improved)
        / max(len(accepted_indices), 1),
        "retained_count": len(retained),
        "retention_rate": len(retained) / max(len(both_improved), 1),
        "translation_catastrophic_threshold_native": translation_catastrophic,
        "rotation_catastrophic_threshold_rad": rotation_catastrophic,
        "stages": stage_metrics,
    }


def aggregate_repeat_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda values: tuple(str(value) for value in values)):
        selected = groups[key]
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["n"] = len(selected)
        for metric in metric_fields:
            values = np.asarray(
                [float(row[metric]) for row in selected], dtype=np.float64
            )
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(summary)
    return output


def aggregate_stage_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda values: tuple(str(value) for value in values)):
        selected = groups[key]
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["n"] = len(selected)
        for field in (
            "trace_attempts",
            "trace_accepts",
            "accepted_count",
            "accepted_both_improved_count",
            "retained_count",
        ):
            summary[field] = int(sum(int(row.get(field, 0) or 0) for row in selected))
        attempts = int(summary["trace_attempts"])
        accepted = int(summary["accepted_count"])
        both_improved = int(summary["accepted_both_improved_count"])
        retained = int(summary["retained_count"])
        summary["accept_rate"] = accepted / max(attempts, 1)
        summary["accepted_both_improved_rate"] = both_improved / max(accepted, 1)
        summary["retention_rate"] = retained / max(both_improved, 1)
        output.append(summary)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _invert_pose_map(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        canonical_frame_id(name): np.linalg.inv(np.asarray(pose, dtype=np.float64))
        for name, pose in values.items()
    }


def _load_metadata(model_dir: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    metadata = _read_json(model_dir / "metadata.json")
    trajectory: dict[str, np.ndarray] = {}
    test_names: list[str] = []
    for keyframe in metadata.get("keyframes", []):
        if not isinstance(keyframe, dict):
            continue
        info = keyframe.get("info", {})
        if not isinstance(info, dict):
            continue
        name = str(info.get("name", info.get("image_name", "")))
        pose_w2c = valid_pose(keyframe.get("Rt"))
        if not name or pose_w2c is None:
            continue
        frame_id = canonical_frame_id(name)
        trajectory[frame_id] = np.linalg.inv(pose_w2c)
        if bool(info.get("is_test", False)):
            test_names.append(name)
    return trajectory, test_names


def _load_stage_trace(
    model_dir: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    set[str],
    dict[str, Any],
    list[str],
]:
    path = model_dir / "pose_initialization_risk_trace.json"
    if not path.exists():
        return {}, {}, set(), {}, []
    trace = _read_json(path)
    initial: dict[str, np.ndarray] = {}
    post_a: dict[str, np.ndarray] = {}
    accepted: set[str] = set()
    ordered_frame_ids: list[str] = []
    for event in trace.get("events", []):
        if not isinstance(event, dict):
            continue
        name = str(event.get("image_name", ""))
        frame_id = canonical_frame_id(name)
        verification = event.get("verification", {})
        if not isinstance(verification, dict):
            verification = {}
        initial_w2c = valid_pose(
            event.get("initial_estimated_Rt") or verification.get("initial_Rt")
        )
        post_w2c = valid_pose(
            event.get("post_a_Rt")
            or event.get("estimated_Rt")
            or verification.get("final_Rt")
        )
        if not frame_id or initial_w2c is None or post_w2c is None:
            continue
        if frame_id not in initial:
            ordered_frame_ids.append(frame_id)
        initial[frame_id] = np.linalg.inv(initial_w2c)
        post_a[frame_id] = np.linalg.inv(post_w2c)
        if bool(event.get("verification_accepted", False)):
            accepted.add(frame_id)
    return initial, post_a, accepted, trace.get("summary", {}), ordered_frame_ids


def _dataset_name(scene_name: str) -> str:
    raw = str(SCENES[scene_name].dataset)
    return "TUM" if raw.startswith("TUM") else raw


def _evaluation_names(scene_name: str, metadata_test_names: list[str]) -> list[str]:
    dataset = _dataset_name(scene_name)
    if dataset in {"MipNeRF360", "TUM"}:
        image_names = sorted_image_names(EVALUATION_SCENE_DIRS[scene_name] / "images")
        return paper_test_names(image_names, dataset=dataset)
    return list(metadata_test_names)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _macro_rows(scene_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        groups[(int(row["repeat"]), str(row["variant"]), str(row["dataset"]))].append(
            row
        )
    output: list[dict[str, Any]] = []
    for (repeat, variant, dataset), rows in sorted(groups.items()):
        output.append(
            {
                "repeat": repeat,
                "variant": variant,
                "dataset": dataset,
                "scene_count": len(rows),
                **{
                    field: float(np.mean([float(row[field]) for row in rows]))
                    for field in POSE_FIELDS
                },
            }
        )
    return output


def evaluate_experiment(
    run_root: Path,
    output_dir: Path,
    *,
    repeats: Sequence[int],
    variants: Sequence[str],
    scene_names: Sequence[str],
) -> dict[str, Any]:
    final_scene_rows: list[dict[str, Any]] = []
    stage_scene_rows: list[dict[str, Any]] = []
    stage_reports: dict[str, Any] = {}
    for repeat in repeats:
        for variant in variants:
            for scene_name in scene_names:
                model_dir = (
                    Path(run_root)
                    / f"repeat_{int(repeat):02d}"
                    / variant
                    / scene_name
                    / "model"
                )
                if not (model_dir / "metadata.json").exists():
                    continue
                final, metadata_test_names = _load_metadata(model_dir)
                reference_w2c = load_reference(reference_scene_spec(scene_name))
                reference = _invert_pose_map(reference_w2c)
                evaluation_names = _evaluation_names(scene_name, metadata_test_names)
                final_metrics = compute_paper_pose_metrics(
                    final,
                    reference,
                    evaluation_names,
                )
                final_scene_rows.append(
                    {
                        "repeat": int(repeat),
                        "variant": variant,
                        "dataset": _dataset_name(scene_name),
                        "scene": scene_name,
                        **{field: final_metrics[field] for field in POSE_FIELDS},
                        "ape_frames": final_metrics["ape_frames"],
                        "rpe_pairs": final_metrics["rpe_pairs"],
                        "alignment_scale": final_metrics["alignment_scale"],
                        "model_dir": str(model_dir),
                    }
                )
                initial, post_a, accepted, trace_summary, stage_names = _load_stage_trace(
                    model_dir
                )
                if initial and post_a:
                    stage = compute_fixed_alignment_stage_report(
                        reference,
                        initial,
                        post_a,
                        final,
                        stage_names,
                        accepted_frame_ids=accepted,
                    )
                    report_key = f"repeat_{int(repeat):02d}/{variant}/{scene_name}"
                    stage_reports[report_key] = stage
                    row: dict[str, Any] = {
                        "repeat": int(repeat),
                        "variant": variant,
                        "dataset": _dataset_name(scene_name),
                        "scene": scene_name,
                        "stage_frame_scope": "all_trace_common_frames",
                        "accepted_count": stage["accepted_count"],
                        "accepted_both_improved_count": stage[
                            "accepted_both_improved_count"
                        ],
                        "accepted_both_improved_rate": stage[
                            "accepted_both_improved_rate"
                        ],
                        "retained_count": stage["retained_count"],
                        "retention_rate": stage["retention_rate"],
                        "alignment_fallback": stage["alignment_fallback"],
                        "trace_eligible": trace_summary.get("eligible", 0),
                        "trace_candidates": trace_summary.get(
                            "verification_candidates", 0
                        ),
                        "trace_attempts": trace_summary.get("verification_attempts", 0),
                        "trace_accepts": trace_summary.get("verification_accepted", 0),
                    }
                    for stage_name, metrics in stage["stages"].items():
                        for field, value in metrics.items():
                            row[f"{stage_name}_{field}"] = value
                    for field in POSE_FIELDS:
                        row[f"delta_post_a_minus_initial_{field}"] = float(
                            stage["stages"]["post_a"][field]
                        ) - float(stage["stages"]["initial"][field])
                        row[f"delta_final_minus_initial_{field}"] = float(
                            stage["stages"]["final"][field]
                        ) - float(stage["stages"]["initial"][field])
                    stage_scene_rows.append(row)

    macro = _macro_rows(final_scene_rows)
    scene_repeat_summary = aggregate_repeat_rows(
        final_scene_rows,
        group_fields=("dataset", "scene", "variant"),
        metric_fields=POSE_FIELDS,
    )
    repeat_summary = aggregate_repeat_rows(
        macro,
        group_fields=("dataset", "variant"),
        metric_fields=POSE_FIELDS,
    )
    stage_variant_summary = aggregate_stage_rows(
        stage_scene_rows,
        group_fields=("variant",),
    )
    stage_scene_summary = aggregate_stage_rows(
        stage_scene_rows,
        group_fields=("dataset", "scene", "variant"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "final_scene.csv", final_scene_rows)
    _write_csv(output_dir / "final_scene_repeat_summary.csv", scene_repeat_summary)
    _write_csv(output_dir / "final_dataset_macro_by_repeat.csv", macro)
    _write_csv(output_dir / "final_dataset_repeat_summary.csv", repeat_summary)
    _write_csv(output_dir / "stage_scene.csv", stage_scene_rows)
    _write_csv(output_dir / "stage_variant_summary.csv", stage_variant_summary)
    _write_csv(output_dir / "stage_scene_summary.csv", stage_scene_summary)
    report = {
        "protocol": {
            "final_alignment": "per-scene Sim(3) on paper test views",
            "final_statistics": "per-scene RMSE and dataset macro mean",
            "translation": "100 x native unit",
            "rotation": "radians",
            "stage_alignment": "one Sim(3) fitted on initial poses not accepted by A",
            "stage_frames": "all ordered frames common to trace, final trajectory, and GT",
        },
        "final_scene": final_scene_rows,
        "final_scene_repeat_summary": scene_repeat_summary,
        "final_dataset_macro_by_repeat": macro,
        "final_dataset_repeat_summary": repeat_summary,
        "stage_scene": stage_scene_rows,
        "stage_variant_summary": stage_variant_summary,
        "stage_scene_summary": stage_scene_summary,
        "stage_reports": stage_reports,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# A-v2 Baseline Pose Experiment",
        "",
        "Final metrics use the unchanged paper Table 5 formulas.",
        "",
        "| Dataset | Variant | n | T.APE | R.APE | T.RPE | R.RPE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in repeat_summary:
        values = [
            f"{float(row[f'{field}_mean']):.6f} +/- {float(row[f'{field}_std']):.6f}"
            for field in POSE_FIELDS
        ]
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['n']} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "| Variant | Attempts | Accepted | Accept rate | Both GT improved | Retained |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stage_variant_summary:
        lines.append(
            f"| {row['variant']} | {row['trace_attempts']} | {row['accepted_count']} | "
            f"{float(row['accept_rate']):.4f} | "
            f"{float(row['accepted_both_improved_rate']):.4f} | "
            f"{float(row['retention_rate']):.4f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _discover_repeats(run_root: Path) -> list[int]:
    output: list[int] = []
    for path in run_root.glob("repeat_*"):
        match = re.fullmatch(r"repeat_(\d+)", path.name)
        if match:
            output.append(int(match.group(1)))
    return sorted(set(output))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--repeats", nargs="*", type=int, default=[])
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=[])
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repeats = list(args.repeats) if args.repeats else _discover_repeats(args.run_root)
    if not repeats:
        raise ValueError(f"no repeat directories found under {args.run_root}")
    variants = list(args.variants) if args.variants else list(VARIANTS)
    scene_names = list(args.scenes) if args.scenes else list(SCENES)
    output_dir = args.output_dir or args.run_root / "evaluation"
    evaluate_experiment(
        args.run_root,
        output_dir,
        repeats=repeats,
        variants=variants,
        scene_names=scene_names,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
