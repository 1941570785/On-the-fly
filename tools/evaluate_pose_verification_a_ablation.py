#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_official_pose_benchmark import (
    REFERENCE_ROOT,
    SCENES,
    _aligned_metrics,
    _spearman,
    canonical_frame_id,
    load_metadata_trajectory,
    load_reference,
    metadata_summary,
    natural_key,
    pose_errors,
    read_json,
    valid_pose,
)


METHODS = ("v31_control", "a_observe", "a_full")
PRIMARY_POSE_METRICS = (
    "ape_trans_rmse",
    "ape_rot_deg_mean",
    "rpe_trans_rmse",
    "rpe_rot_deg_mean",
)
PRIMARY_RENDER_METRICS = ("PSNR", "SSIM", "LPIPS", "time")


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _sum(values: list[float]) -> float:
    return float(sum(float(value) for value in values if value is not None and math.isfinite(float(value))))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_scene(
    reference: dict[str, np.ndarray],
    methods: dict[str, dict[str, np.ndarray]],
    *,
    rpe_delta: int = 1,
) -> dict[str, Any]:
    missing = set(METHODS) - set(methods)
    if missing:
        raise ValueError(f"Missing methods: {sorted(missing)}")
    reference_ids = set(reference)
    common = set(reference_ids)
    for method in METHODS:
        common &= set(methods[method])
    common_ids = sorted(common, key=natural_key)
    if len(common_ids) < 3:
        raise ValueError("At least three common reference-valid frames are required")
    result: dict[str, Any] = {
        "reference_frames": len(reference),
        "common_frames": len(common_ids),
        "common_frame_ids": common_ids,
        "rpe_delta": max(1, int(rpe_delta)),
        "methods": {},
    }
    for method in METHODS:
        metrics, _ = _aligned_metrics(
            reference,
            methods[method],
            common_ids,
            rpe_delta=result["rpe_delta"],
        )
        available = len(reference_ids & set(methods[method]))
        metrics.update(
            {
                "trajectory_frames": len(methods[method]),
                "reference_matched_frames": available,
                "coverage": available / len(reference) if reference else 0.0,
            }
        )
        result["methods"][method] = metrics
    return result


def _event_error(
    pose: np.ndarray,
    reference_pose: np.ndarray,
    similarity: Any,
) -> tuple[float, float]:
    aligned = similarity.apply(pose[None])
    error = pose_errors(aligned, reference_pose[None])
    return float(error["trans_mean"]), float(error["rot_deg_mean"])


def evaluate_a_trace(
    trace_path: Path,
    reference: dict[str, np.ndarray],
    alignment_trajectory: dict[str, np.ndarray],
    alignment_frame_ids: list[str],
) -> dict[str, Any]:
    trace = read_json(trace_path)
    fit_ids = [
        frame_id
        for frame_id in alignment_frame_ids
        if frame_id in reference and frame_id in alignment_trajectory
    ]
    _, similarity = _aligned_metrics(reference, alignment_trajectory, fit_ids, rpe_delta=1)
    rows: list[dict[str, Any]] = []
    for event in trace.get("events", []):
        if not isinstance(event, dict):
            continue
        frame_id = canonical_frame_id(event.get("image_name"))
        if not frame_id or frame_id not in reference:
            continue
        verification = event.get("verification", {})
        verification = verification if isinstance(verification, dict) else {}
        initial = valid_pose(
            event.get("initial_estimated_Rt")
            or verification.get("initial_Rt")
            or event.get("estimated_Rt")
        )
        final = valid_pose(
            event.get("estimated_Rt")
            or verification.get("final_Rt")
            or event.get("initial_estimated_Rt")
        )
        if initial is None or final is None:
            continue
        initial_trans, initial_rot = _event_error(initial, reference[frame_id], similarity)
        final_trans, final_rot = _event_error(final, reference[frame_id], similarity)
        rows.append(
            {
                "frame_id": frame_id,
                "risk_score": float(event.get("risk_score", 0.0)),
                "candidate": bool(event.get("verification_candidate", False)),
                "attempted": bool(event.get("verification_attempted", False)),
                "accepted": bool(event.get("verification_accepted", False)),
                "initial_translation_error": initial_trans,
                "initial_rotation_error_deg": initial_rot,
                "final_translation_error": final_trans,
                "final_rotation_error_deg": final_rot,
                "fallback_exact": bool(np.allclose(initial, final, atol=1e-7, rtol=0.0)),
                "match_count_total": _numeric(event.get("match_count_total")),
                "num_2d3d_correspondences": _numeric(event.get("num_2d3d_correspondences")),
                "num_pnp_inliers": _numeric(event.get("num_pnp_inliers")),
                "num_miniba_inliers": _numeric(event.get("num_miniba_inliers")),
                "pnp_inlier_ratio": _numeric(event.get("pnp_inlier_ratio")),
                "miniba_inlier_ratio": _numeric(event.get("miniba_inlier_ratio")),
                "pre_mean": _numeric(verification.get("pre_reprojection_mean")),
                "post_mean": _numeric(verification.get("post_reprojection_mean")),
                "pre_median": _numeric(verification.get("pre_reprojection_median")),
                "post_median": _numeric(verification.get("post_reprojection_median")),
                "pre_p90": _numeric(verification.get("pre_reprojection_p90")),
                "post_p90": _numeric(verification.get("post_reprojection_p90")),
                "runtime": _numeric(verification.get("runtime_seconds")),
                "rotation_correction_deg": _numeric(verification.get("rotation_correction_deg")),
                "translation_correction": _numeric(verification.get("translation_correction")),
            }
        )

    candidates = [row for row in rows if row["candidate"]]
    noncandidates = [row for row in rows if not row["candidate"]]
    attempts = [row for row in rows if row["attempted"]]
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in attempts if not row["accepted"]]
    risks = [row["risk_score"] for row in rows]
    initial_trans = [row["initial_translation_error"] for row in rows]
    initial_rot = [row["initial_rotation_error_deg"] for row in rows]
    candidate_trans = _mean([row["initial_translation_error"] for row in candidates])
    noncandidate_trans = _mean([row["initial_translation_error"] for row in noncandidates])
    candidate_rot = _mean([row["initial_rotation_error_deg"] for row in candidates])
    noncandidate_rot = _mean([row["initial_rotation_error_deg"] for row in noncandidates])

    def reduction(before_key: str, after_key: str) -> float | None:
        values = [
            float(row[before_key]) - float(row[after_key])
            for row in attempts
            if row[before_key] is not None and row[after_key] is not None
        ]
        return _mean(values)

    return {
        "evaluated_events": len(rows),
        "candidate_count": len(candidates),
        "attempt_count": len(attempts),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accept_rate": len(accepted) / max(len(attempts), 1),
        "candidate_frame_ids": [row["frame_id"] for row in candidates],
        "risk_translation_spearman": _spearman(risks, initial_trans),
        "risk_rotation_spearman": _spearman(risks, initial_rot),
        "candidate_translation_error_mean": candidate_trans,
        "noncandidate_translation_error_mean": noncandidate_trans,
        "candidate_translation_enrichment": _ratio(candidate_trans, noncandidate_trans),
        "candidate_rotation_error_mean": candidate_rot,
        "noncandidate_rotation_error_mean": noncandidate_rot,
        "candidate_rotation_enrichment": _ratio(candidate_rot, noncandidate_rot),
        "accepted_initial_translation_error_mean": _mean(
            [row["initial_translation_error"] for row in accepted]
        ),
        "accepted_final_translation_error_mean": _mean(
            [row["final_translation_error"] for row in accepted]
        ),
        "accepted_initial_rotation_error_mean": _mean(
            [row["initial_rotation_error_deg"] for row in accepted]
        ),
        "accepted_final_rotation_error_mean": _mean(
            [row["final_rotation_error_deg"] for row in accepted]
        ),
        "accepted_translation_improved_count": sum(
            row["final_translation_error"] < row["initial_translation_error"]
            for row in accepted
        ),
        "accepted_rotation_improved_count": sum(
            row["final_rotation_error_deg"] < row["initial_rotation_error_deg"]
            for row in accepted
        ),
        "accepted_translation_harmed_count": sum(
            row["final_translation_error"] > row["initial_translation_error"]
            for row in accepted
        ),
        "accepted_rotation_harmed_count": sum(
            row["final_rotation_error_deg"] > row["initial_rotation_error_deg"]
            for row in accepted
        ),
        "rejected_exact_fallback_count": sum(row["fallback_exact"] for row in rejected),
        "reprojection_mean_reduction": reduction("pre_mean", "post_mean"),
        "reprojection_median_reduction": reduction("pre_median", "post_median"),
        "reprojection_p90_reduction": reduction("pre_p90", "post_p90"),
        "runtime_seconds_total": _sum([row["runtime"] for row in attempts]),
        "runtime_seconds_mean": _mean([row["runtime"] for row in attempts]),
        "translation_correction_mean": _mean(
            [row["translation_correction"] for row in attempts]
        ),
        "rotation_correction_deg_mean": _mean(
            [row["rotation_correction_deg"] for row in attempts]
        ),
        "match_count_mean": _mean([row["match_count_total"] for row in rows]),
        "correspondence_count_mean": _mean(
            [row["num_2d3d_correspondences"] for row in rows]
        ),
        "pnp_inlier_count_mean": _mean([row["num_pnp_inliers"] for row in rows]),
        "miniba_inlier_count_mean": _mean([row["num_miniba_inliers"] for row in rows]),
        "pnp_inlier_ratio_mean": _mean([row["pnp_inlier_ratio"] for row in rows]),
        "miniba_inlier_ratio_mean": _mean([row["miniba_inlier_ratio"] for row in rows]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _average_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: _mean([_numeric(row.get(key)) for row in rows]) for key in keys}


def evaluate_run(
    run_root: Path,
    output_dir: Path,
    *,
    scene_names: list[str],
    reference_root: Path = REFERENCE_ROOT,
    rpe_delta: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"run_root": str(run_root), "scenes": {}}
    pose_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for scene_name in scene_names:
        spec = SCENES[scene_name]
        reference = load_reference(spec, reference_root)
        model_dirs = {method: run_root / method / scene_name / "model" for method in METHODS}
        trajectories = {
            method: load_metadata_trajectory(path) for method, path in model_dirs.items()
        }
        pose_report = evaluate_scene(reference, trajectories, rpe_delta=rpe_delta)
        observe_diagnostic = evaluate_a_trace(
            model_dirs["a_observe"] / "pose_initialization_risk_trace.json",
            reference,
            trajectories["a_observe"],
            pose_report["common_frame_ids"],
        )
        full_diagnostic = evaluate_a_trace(
            model_dirs["a_full"] / "pose_initialization_risk_trace.json",
            reference,
            trajectories["a_full"],
            pose_report["common_frame_ids"],
        )
        observe_ids = set(observe_diagnostic["candidate_frame_ids"])
        full_ids = set(full_diagnostic["candidate_frame_ids"])
        candidate_union = observe_ids | full_ids
        candidate_agreement = {
            "candidate_common_count": len(observe_ids & full_ids),
            "candidate_union_count": len(candidate_union),
            "candidate_jaccard": (
                len(observe_ids & full_ids) / len(candidate_union) if candidate_union else 1.0
            ),
        }
        report["scenes"][scene_name] = {
            "dataset": spec.dataset,
            "reference_type": spec.reference_type,
            "pose": pose_report,
            "observe_diagnostic": observe_diagnostic,
            "full_diagnostic": full_diagnostic,
            **candidate_agreement,
        }
        for method in METHODS:
            pose_rows.append(
                {
                    "dataset": spec.dataset,
                    "scene": scene_name,
                    "method": method,
                    "reference_type": spec.reference_type,
                    "reference_frames": pose_report["reference_frames"],
                    "common_frames": pose_report["common_frames"],
                    **pose_report["methods"][method],
                }
            )
            render_rows.append(
                {
                    "dataset": spec.dataset,
                    "scene": scene_name,
                    "method": method,
                    **metadata_summary(model_dirs[method]),
                }
            )
        control_pose = pose_report["methods"]["v31_control"]
        full_pose = pose_report["methods"]["a_full"]
        control_render = metadata_summary(model_dirs["v31_control"])
        full_render = metadata_summary(model_dirs["a_full"])
        delta_rows.append(
            {
                "dataset": spec.dataset,
                "scene": scene_name,
                **{
                    f"delta_{key}_full_minus_control": float(full_pose[key])
                    - float(control_pose[key])
                    for key in PRIMARY_POSE_METRICS
                },
                **{
                    f"delta_{key}_full_minus_control": float(full_render[key])
                    - float(control_render[key])
                    for key in PRIMARY_RENDER_METRICS
                },
            }
        )
        diagnostic_rows.append(
            {
                "dataset": spec.dataset,
                "scene": scene_name,
                **candidate_agreement,
                **{
                    f"observe_{key}": value
                    for key, value in observe_diagnostic.items()
                    if key != "candidate_frame_ids"
                },
                **{
                    f"full_{key}": value
                    for key, value in full_diagnostic.items()
                    if key != "candidate_frame_ids"
                },
                "observe_candidate_frames": ";".join(
                    observe_diagnostic["candidate_frame_ids"]
                ),
                "full_candidate_frames": ";".join(full_diagnostic["candidate_frame_ids"]),
            }
        )

    dataset_macro_rows: list[dict[str, Any]] = []
    for dataset in dict.fromkeys(row["dataset"] for row in render_rows):
        for method in METHODS:
            selected_render = [
                row for row in render_rows if row["dataset"] == dataset and row["method"] == method
            ]
            selected_pose = [
                row for row in pose_rows if row["dataset"] == dataset and row["method"] == method
            ]
            dataset_macro_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "scene_count": len(selected_render),
                    **_average_rows(selected_render, PRIMARY_RENDER_METRICS),
                    **_average_rows(selected_pose, PRIMARY_POSE_METRICS),
                }
            )
    nine_scene_rows: list[dict[str, Any]] = []
    for method in METHODS:
        selected_render = [row for row in render_rows if row["method"] == method]
        selected_pose = [row for row in pose_rows if row["method"] == method]
        nine_scene_rows.append(
            {
                "method": method,
                "scene_count": len(selected_render),
                **_average_rows(selected_render, PRIMARY_RENDER_METRICS),
                **_average_rows(selected_pose, PRIMARY_POSE_METRICS),
            }
        )

    _write_csv(output_dir / "pose_scene.csv", pose_rows)
    _write_csv(output_dir / "render_scene.csv", render_rows)
    _write_csv(output_dir / "full_minus_control.csv", delta_rows)
    _write_csv(output_dir / "a_diagnostics_scene.csv", diagnostic_rows)
    _write_csv(output_dir / "dataset_macro.csv", dataset_macro_rows)
    _write_csv(output_dir / "nine_scene_macro.csv", nine_scene_rows)
    (output_dir / "pose_verification_a_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--rpe_delta", type=int, default=1)
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scene_names = list(args.scenes) if args.scenes else list(SCENES)
    evaluate_run(
        args.run_root,
        args.output_dir,
        scene_names=scene_names,
        reference_root=args.reference_root,
        rpe_delta=max(1, int(args.rpe_delta)),
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
