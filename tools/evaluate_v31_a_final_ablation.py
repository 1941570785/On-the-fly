#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_official_pose_benchmark import (
    REFERENCE_ROOT,
    SCENES,
    _aligned_metrics,
    load_metadata_trajectory,
    load_reference,
    metadata_summary,
    natural_key,
)


METHODS = (
    "baseline",
    "w_o_pose_risk_a",
    "w_o_response_sampling",
    "w_o_extra_optimization",
    "full",
)
RENDER_METRICS = ("PSNR", "SSIM", "LPIPS", "time")
POSE_METRICS = (
    "ape_trans_rmse",
    "ape_rot_deg_mean",
    "rpe_trans_rmse",
    "rpe_rot_deg_mean",
)


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def extract_trigger_summary(
    metadata: dict[str, Any],
    risk_trace: dict[str, Any],
) -> dict[str, Any]:
    sampling = metadata.get("pose_render_texture_sampling", {})
    extra = metadata.get("pose_render_extra_optimization", {})
    sampling = sampling if isinstance(sampling, dict) else {}
    extra = extra if isinstance(extra, dict) else {}
    events = risk_trace.get("events", [])
    events = events if isinstance(events, list) else []
    valid_events = [event for event in events if isinstance(event, dict)]
    return {
        "sampling_mode": sampling.get("mode", "off"),
        "sampling_events": int(sampling.get("events", 0) or 0),
        "sampling_applied": int(sampling.get("applied", 0) or 0),
        "sampling_budget_shift_abs_sum": float(
            sampling.get("budget_shift_abs_sum", 0.0) or 0.0
        ),
        "extra_optimization_mode": extra.get("mode", "off"),
        "extra_optimization_events": int(extra.get("events", 0) or 0),
        "extra_optimization_applied": int(extra.get("applied", 0) or 0),
        "extra_iterations": int(extra.get("extra_iterations_sum", 0) or 0),
        "risk_events": len(valid_events),
        "risk_candidates": sum(bool(event.get("risk_candidate", False)) for event in valid_events),
        "quarantined": sum(
            bool(event.get("pose_reference_quarantined", False))
            for event in valid_events
        ),
    }


def aggregate_rendering(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = [method for method in METHODS if any(row.get("method") == method for row in rows)]
    methods.extend(
        method
        for method in dict.fromkeys(str(row.get("method")) for row in rows)
        if method not in methods
    )
    output: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in rows if row.get("method") == method]
        if not selected:
            continue
        output.append(
            {
                "method": method,
                "scenes": len(selected),
                **{
                    metric: sum(float(row[metric]) for row in selected) / len(selected)
                    for metric in RENDER_METRICS
                },
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_dir_for(
    method: str,
    scene: str,
    run_root: Path,
    baseline_root: Path,
) -> Path:
    if method == "baseline":
        return baseline_root / scene / "model"
    return run_root / method / scene / "model"


def attach_render_deltas(rows: list[dict[str, Any]]) -> None:
    full = {
        row["scene"]: row
        for row in rows
        if row.get("method") == "full"
    }
    for row in rows:
        reference = full[row["scene"]]
        for metric in RENDER_METRICS:
            row[f"delta_{metric}_to_full"] = float(row[metric]) - float(reference[metric])


def attach_pose_deltas(rows: list[dict[str, Any]]) -> None:
    full = {
        row["scene"]: row
        for row in rows
        if row.get("method") == "full"
    }
    for row in rows:
        reference = full[row["scene"]]
        for metric in POSE_METRICS:
            row[f"delta_{metric}_to_full"] = float(row[metric]) - float(reference[metric])


def evaluate(
    run_root: Path,
    baseline_root: Path,
    output_dir: Path,
    *,
    scene_names: list[str],
    reference_root: Path = REFERENCE_ROOT,
    rpe_delta: int = 1,
) -> dict[str, Any]:
    rendering_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "run_root": str(run_root),
        "baseline_root": str(baseline_root),
        "scenes": {},
    }
    for scene_name in scene_names:
        spec = SCENES[scene_name]
        reference = load_reference(spec, reference_root)
        model_dirs = {
            method: model_dir_for(method, scene_name, run_root, baseline_root)
            for method in METHODS
        }
        trajectories = {
            method: load_metadata_trajectory(model_dir)
            for method, model_dir in model_dirs.items()
        }
        common_ids = set(reference)
        for trajectory in trajectories.values():
            common_ids &= set(trajectory)
        ordered_ids = sorted(common_ids, key=natural_key)
        if len(ordered_ids) < 3:
            raise ValueError(f"{scene_name} has fewer than three five-way common frames")
        scene_report: dict[str, Any] = {
            "dataset": spec.dataset,
            "reference_type": spec.reference_type,
            "reference_frames": len(reference),
            "common_frames": len(ordered_ids),
            "methods": {},
        }
        for method in METHODS:
            model_dir = model_dirs[method]
            metadata = read_json_if_present(model_dir / "metadata.json")
            rendering = metadata_summary(model_dir)
            rendering_row = {
                "dataset": spec.dataset,
                "scene": scene_name,
                "method": method,
                **rendering,
            }
            rendering_rows.append(rendering_row)
            metrics, _ = _aligned_metrics(
                reference,
                trajectories[method],
                ordered_ids,
                rpe_delta=rpe_delta,
            )
            matched = len(set(reference) & set(trajectories[method]))
            pose_row = {
                "dataset": spec.dataset,
                "scene": scene_name,
                "reference_type": spec.reference_type,
                "method": method,
                "reference_frames": len(reference),
                "common_frames": len(ordered_ids),
                **metrics,
                "trajectory_frames": len(trajectories[method]),
                "reference_matched_frames": matched,
                "coverage": matched / len(reference) if reference else 0.0,
            }
            pose_rows.append(pose_row)
            risk_trace = read_json_if_present(model_dir / "pose_risk_utility_trace.json")
            trigger_row = {
                "dataset": spec.dataset,
                "scene": scene_name,
                "method": method,
                **extract_trigger_summary(metadata, risk_trace),
            }
            trigger_rows.append(trigger_row)
            scene_report["methods"][method] = {
                "rendering": rendering,
                "pose": metrics,
                "triggers": trigger_row,
            }
        report["scenes"][scene_name] = scene_report
    attach_render_deltas(rendering_rows)
    attach_pose_deltas(pose_rows)
    aggregate_rows = aggregate_rendering(rendering_rows)
    full_aggregate = next(row for row in aggregate_rows if row["method"] == "full")
    for row in aggregate_rows:
        for metric in RENDER_METRICS:
            row[f"delta_{metric}_to_full"] = float(row[metric]) - float(full_aggregate[metric])
    write_csv(output_dir / "rendering_ablation.csv", rendering_rows)
    write_csv(output_dir / "rendering_macro.csv", aggregate_rows)
    write_csv(output_dir / "pose_ablation.csv", pose_rows)
    write_csv(output_dir / "component_triggers.csv", trigger_rows)
    report["rendering_macro"] = aggregate_rows
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--rpe_delta", type=int, default=1)
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scene_names = list(args.scenes) if args.scenes else list(SCENES)
    evaluate(
        args.run_root,
        args.baseline_root,
        args.output_dir,
        scene_names=scene_names,
        reference_root=args.reference_root,
        rpe_delta=max(1, args.rpe_delta),
    )
    print(args.output_dir / "ablation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
