#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.standard_pose_eval_report import (
        fit_similarity_w2c,
        pose_errors,
        relative_pose_errors,
    )
except ModuleNotFoundError:
    from standard_pose_eval_report import (
        fit_similarity_w2c,
        pose_errors,
        relative_pose_errors,
    )


REFERENCE_DIRS = {
    "bonsai": {
        "v31": Path(
            "/data2/zxd/3D_Reconstruction/On_the_fly_pose_render_coupling_v2/"
            "results/BRANCH_EXPERIMENTS_20260703/"
            "baseline_render_lock_intra_frame_v31_full9_20260706_135504/"
            "bonsai/model"
        ),
    },
    "forest1": {
        "v31": Path(
            "/data2/zxd/3D_Reconstruction/On_the_fly_pose_render_coupling_v2/"
            "results/BRANCH_EXPERIMENTS_20260703/"
            "baseline_render_lock_intra_frame_v31_full9_20260706_135504/"
            "forest1/model"
        ),
    },
}

# Preserved full-precision results behind the paper tables. The corresponding
# baseline trajectory artifacts were later overwritten, so these values are
# authoritative only for quality, runtime, and keyframe count.
TABLE_BASELINE_METRICS = {
    "bonsai": {
        "PSNR": 24.254085183143616,
        "SSIM": 0.8119161486625671,
        "LPIPS": 0.24992872774600983,
        "time": 67.05928325653076,
        "num_keyframes": 223,
    },
    "forest1": {
        "PSNR": 17.78855973482132,
        "SSIM": 0.48459342680871487,
        "LPIPS": 0.4262222908437252,
        "time": 109.03564667701721,
        "num_keyframes": 269,
    },
}

QUALITY_METRICS = ("PSNR", "SSIM", "LPIPS")
POSE_METRICS = ("APE_t", "APE_R_deg", "RPE_t", "RPE_R_deg")


def canonical_frame_id(name: Any) -> str:
    stem = Path(str(name or "")).stem
    if stem.isdigit():
        return str(int(stem))
    return stem.lower()


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _pose(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return pose


def metadata_trajectory(model_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    metadata = _read_json(model_dir / "metadata.json")
    estimated: dict[str, np.ndarray] = {}
    gt: dict[str, np.ndarray] = {}
    for keyframe in metadata.get("keyframes", []):
        if not isinstance(keyframe, dict):
            continue
        info = keyframe.get("info", {})
        if not isinstance(info, dict):
            continue
        frame_id = canonical_frame_id(info.get("name"))
        est_pose = _pose(keyframe.get("Rt"))
        gt_pose = _pose(info.get("gt_Rt"))
        if not frame_id or est_pose is None or gt_pose is None:
            continue
        estimated[frame_id] = est_pose
        gt[frame_id] = gt_pose
    return {"estimated": estimated, "gt": gt}


def risk_trace_trajectory(model_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    trace = _read_json(model_dir / "pose_risk_utility_trace.json")
    estimated: dict[str, np.ndarray] = {}
    gt: dict[str, np.ndarray] = {}
    for event in trace.get("events", []):
        if not isinstance(event, dict) or not event.get("baseline_selected", False):
            continue
        frame_id = canonical_frame_id(event.get("image_name"))
        est_pose = _pose(event.get("estimated_Rt"))
        gt_pose = _pose(event.get("gt_Rt"))
        if not frame_id or est_pose is None or gt_pose is None:
            continue
        estimated[frame_id] = est_pose
        gt[frame_id] = gt_pose
    return {"estimated": estimated, "gt": gt}


def evaluate_pose_methods(
    trajectories: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    gt_source_method: str,
    rpe_delta: int = 1,
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    if gt_source_method not in trajectories:
        raise ValueError("GT source method must be present in trajectories")
    common = None
    for trajectory in trajectories.values():
        valid = set(trajectory["estimated"]) & set(trajectory["gt"])
        common = valid if common is None else common & valid
    common_ids = sorted(common or set(), key=_natural_key)
    if len(common_ids) < 3:
        raise ValueError("At least three common pose frames are required")

    gt_source = trajectories[gt_source_method]["gt"]
    gt = np.stack([gt_source[frame_id] for frame_id in common_ids])
    output: dict[str, Any] = {
        "common_frame_count": int(len(common_ids)),
        "common_frame_ids": common_ids,
        "rpe_delta": int(max(1, rpe_delta)),
    }
    for method, trajectory in trajectories.items():
        estimated = np.stack(
            [trajectory["estimated"][frame_id] for frame_id in common_ids]
        )
        similarity = fit_similarity_w2c(estimated, gt)
        aligned = similarity.apply(estimated)
        ape = pose_errors(aligned, gt)
        rpe = relative_pose_errors(aligned, gt, delta=max(1, int(rpe_delta)))
        output[method] = {
            "ape_trans_mean": ape["trans_mean"],
            "ape_trans_rmse": ape["trans_rmse"],
            "ape_rot_deg_mean": ape["rot_deg_mean"],
            "ape_rot_deg_median": ape["rot_deg_median"],
            "rpe_trans_mean": rpe["trans_mean"],
            "rpe_trans_rmse": rpe["trans_rmse"],
            "rpe_rot_deg_mean": rpe["rot_deg_mean"],
            "rpe_rot_deg_median": rpe["rot_deg_median"],
            "alignment_scale": float(similarity.scale),
        }
    return output


def evaluate_three_way_pose(
    trajectories: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    rpe_delta: int = 1,
) -> dict[str, Any]:
    if set(trajectories) != {"baseline", "v31", "new"}:
        raise ValueError("Trajectories must contain baseline, v31, and new")
    return evaluate_pose_methods(
        trajectories,
        gt_source_method="baseline",
        rpe_delta=rpe_delta,
    )


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quality_rows(model_dir: Path) -> dict[str, dict[str, float]]:
    path = model_dir / "frame_metrics.csv"
    if not path.exists():
        return {}
    output: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame_id = canonical_frame_id(
                row.get("original_image_name")
                or row.get("render_image_name")
                or row.get("image_name")
            )
            values = {
                "PSNR": _to_float(row.get("psnr", row.get("PSNR"))),
                "SSIM": _to_float(row.get("ssim", row.get("SSIM"))),
                "LPIPS": _to_float(row.get("lpips", row.get("LPIPS"))),
            }
            if frame_id and all(value is not None for value in values.values()):
                output[frame_id] = {
                    key: float(value) for key, value in values.items() if value is not None
                }
    return output


def aligned_quality_means(
    rows_by_method: dict[str, dict[str, dict[str, float]]]
) -> dict[str, Any]:
    common = None
    for rows in rows_by_method.values():
        common = set(rows) if common is None else common & set(rows)
    common_ids = sorted(common or set(), key=_natural_key)
    if not common_ids:
        raise ValueError("No common rendered evaluation frames")
    result: dict[str, Any] = {
        "common_frame_count": int(len(common_ids)),
        "common_frame_ids": common_ids,
    }
    for method, rows in rows_by_method.items():
        result[method] = {
            metric: float(np.mean([rows[frame_id][metric] for frame_id in common_ids]))
            for metric in QUALITY_METRICS
        }
    return result


def compare_scene(scene: str, new_model_dir: Path, *, rpe_delta: int = 1) -> list[dict[str, Any]]:
    references = REFERENCE_DIRS[scene]
    method_dirs = {
        "v31": references["v31"],
        "new": new_model_dir,
    }
    trajectories = {
        "v31": metadata_trajectory(method_dirs["v31"]),
        "new": metadata_trajectory(method_dirs["new"]),
    }
    pose = evaluate_pose_methods(
        trajectories,
        gt_source_method="v31",
        rpe_delta=rpe_delta,
    )
    quality = aligned_quality_means(
        {method: quality_rows(path) for method, path in method_dirs.items()}
    )
    metadata = {
        method: _read_json(path / "metadata.json") for method, path in method_dirs.items()
    }
    utility_trace = _read_json(new_model_dir / "pose_risk_utility_trace.json")
    utility_summary = utility_trace.get("summary", {})
    if not isinstance(utility_summary, dict):
        utility_summary = {}

    rows: list[dict[str, Any]] = []
    for method in ("baseline", "v31", "new"):
        if method == "baseline":
            quality_values = TABLE_BASELINE_METRICS[scene]
            method_metadata = quality_values
            pose_values: dict[str, Any] = {}
            model_dir = "reported:On-the-fly-NVS"
        else:
            quality_values = quality[method]
            method_metadata = metadata[method]
            pose_values = pose[method]
            model_dir = str(method_dirs[method])
        row = {
            "scene": scene,
            "method": method,
            "quality_common_frames": (
                "table" if method == "baseline" else quality["common_frame_count"]
            ),
            "pose_common_frames": (
                "" if method == "baseline" else pose["common_frame_count"]
            ),
            "PSNR": quality_values["PSNR"],
            "SSIM": quality_values["SSIM"],
            "LPIPS": quality_values["LPIPS"],
            "time": method_metadata.get("time", ""),
            "num_keyframes": method_metadata.get(
                "num_keyframes",
                method_metadata.get("num keyframes", ""),
            ),
            "APE_t": pose_values.get("ape_trans_mean", ""),
            "APE_R_deg": pose_values.get("ape_rot_deg_mean", ""),
            "RPE_t": pose_values.get("rpe_trans_mean", ""),
            "RPE_R_deg": pose_values.get("rpe_rot_deg_mean", ""),
            "risk_candidates": utility_summary.get("risk_candidates", 0)
            if method == "new"
            else 0,
            "review_admit": utility_summary.get("review_admit", 0)
            if method == "new"
            else 0,
            "isolate_low_utility": utility_summary.get("isolate_low_utility", 0)
            if method == "new"
            else 0,
            "pose_reference_quarantined": utility_summary.get(
                "pose_reference_quarantined", 0
            )
            if method == "new"
            else 0,
            "quarantine_cooldown_admit": utility_summary.get(
                "quarantine_cooldown_admit", 0
            )
            if method == "new"
            else 0,
            "model_dir": model_dir,
        }
        rows.append(row)

    baseline_row = rows[0]
    v31_row = rows[1]
    for row in rows:
        for metric in (*QUALITY_METRICS, *POSE_METRICS, "time"):
            row[f"delta_{metric}_to_baseline"] = _difference(
                row.get(metric), baseline_row.get(metric)
            )
            row[f"delta_{metric}_to_v31"] = _difference(
                row.get(metric), v31_row.get(metric)
            )
    return rows


def _difference(value: Any, reference: Any) -> float | str:
    left = _to_float(value)
    right = _to_float(reference)
    return left - right if left is not None and right is not None else ""


def _format_number(value: Any, digits: int) -> str:
    number = _to_float(value)
    return "--" if number is None else f"{number:.{digits}f}"


def _format_signed(value: Any, digits: int) -> str:
    number = _to_float(value)
    return "--" if number is None else f"{number:+.{digits}f}"


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (output_dir / "three_way_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "three_way_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V31 Risk-Utility Three-Way Comparison",
        "",
        "On-the-fly-NVS quality and time are the preserved full-precision values behind the paper tables. Its original trajectory artifact is unavailable, so baseline APE/RPE are intentionally not reported. V31 and the new model use one common pose frame set per scene after independent Sim(3) alignment. Translation units follow each dataset coordinate scale.",
        "",
        "| Scene | Method | Quality/Pose Frames | PSNR | SSIM | LPIPS | APE-t | APE-R (deg) | RPE-t | RPE-R (deg) | Time (s) | Review/Drop/Pose-Q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        frame_label = (
            f"{row['quality_common_frames']}/"
            f"{row['pose_common_frames'] if row['pose_common_frames'] != '' else '--'}"
        )
        lines.append(
            f"| {row['scene']} | {row['method']} | {frame_label} | "
            f"{_format_number(row['PSNR'], 4)} | "
            f"{_format_number(row['SSIM'], 4)} | "
            f"{_format_number(row['LPIPS'], 4)} | "
            f"{_format_number(row['APE_t'], 5)} | "
            f"{_format_number(row['APE_R_deg'], 4)} | "
            f"{_format_number(row['RPE_t'], 5)} | "
            f"{_format_number(row['RPE_R_deg'], 4)} | "
            f"{_format_number(row['time'], 3)} | "
            f"{row['review_admit']}/{row['isolate_low_utility']}/"
            f"{row['pose_reference_quarantined']} |"
        )
    lines.extend(
        [
            "",
            "## New Model Deltas",
            "",
            "Negative LPIPS, APE, RPE, and time deltas indicate improvement.",
            "",
            "| Scene | Reference | dPSNR | dSSIM | dLPIPS | dAPE-t | dAPE-R | dRPE-t | dRPE-R | dTime (s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["method"] != "new":
            continue
        for reference in ("baseline", "v31"):
            lines.append(
                f"| {row['scene']} | {reference} | "
                f"{_format_signed(row[f'delta_PSNR_to_{reference}'], 4)} | "
                f"{_format_signed(row[f'delta_SSIM_to_{reference}'], 4)} | "
                f"{_format_signed(row[f'delta_LPIPS_to_{reference}'], 4)} | "
                f"{_format_signed(row[f'delta_APE_t_to_{reference}'], 5)} | "
                f"{_format_signed(row[f'delta_APE_R_deg_to_{reference}'], 4)} | "
                f"{_format_signed(row[f'delta_RPE_t_to_{reference}'], 5)} | "
                f"{_format_signed(row[f'delta_RPE_R_deg_to_{reference}'], 4)} | "
                f"{_format_signed(row[f'delta_time_to_{reference}'], 3)} |"
            )
    (output_dir / "three_way_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_root", required=True)
    parser.add_argument("--new_variant", default="V31_RU_active")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--rpe_delta", type=int, default=1)
    parser.add_argument(
        "--scenes", nargs="*", choices=sorted(REFERENCE_DIRS), default=[]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    new_root = Path(args.new_root)
    output_dir = Path(args.out_dir) if args.out_dir else new_root / "comparison"
    scenes = list(args.scenes) if args.scenes else ["bonsai", "forest1"]
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        rows.extend(
            compare_scene(
                scene,
                new_root / scene / args.new_variant / "model",
                rpe_delta=max(1, int(args.rpe_delta)),
            )
        )
    write_report(output_dir, rows)
    print(output_dir / "three_way_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
