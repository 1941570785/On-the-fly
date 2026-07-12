#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloaders.read_write_model import read_images_binary  # noqa: E402
from tools.standard_pose_eval_report import (  # noqa: E402
    Similarity,
    fit_similarity_w2c,
    pose_errors,
    relative_pose_errors,
)


REFERENCE_ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly/datasets")
METHODS = ("baseline", "v31", "v31_a")


@dataclass(frozen=True)
class SceneSpec:
    name: str
    dataset: str
    reference_type: str
    reference_path: str


SCENES: dict[str, SceneSpec] = {
    "bonsai": SceneSpec("bonsai", "MipNeRF360", "COLMAP pseudo-GT", "MipNeRF360/bonsai/sparse/0/images.bin"),
    "counter": SceneSpec("counter", "MipNeRF360", "COLMAP pseudo-GT", "MipNeRF360/counter/sparse/0/images.bin"),
    "garden": SceneSpec("garden", "MipNeRF360", "COLMAP pseudo-GT", "MipNeRF360/garden/sparse/0/images.bin"),
    "forest1": SceneSpec("forest1", "StaticHikes", "COLMAP pseudo-GT", "StaticHikes/forest1/sparse/0/images.bin"),
    "forest2": SceneSpec("forest2", "StaticHikes", "COLMAP pseudo-GT", "StaticHikes/forest2/sparse/0/images.bin"),
    "university2": SceneSpec("university2", "StaticHikes", "COLMAP pseudo-GT", "StaticHikes/university2/sparse/0/images.bin"),
    "desk": SceneSpec("desk", "TUM", "official mocap GT", "TUM/desk1/sparse/GT"),
    "xyz": SceneSpec("xyz", "TUM", "official mocap GT", "TUM/desk2/sparse/GT"),
    "long_office": SceneSpec("long_office", "TUM", "official mocap GT", "TUM/long_office_household/sparse/GT"),
}


def canonical_frame_id(name: Any) -> str:
    stem = Path(str(name or "")).stem
    if stem.isdigit():
        return str(int(stem))
    return stem.lower()


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def valid_pose(value: Any) -> np.ndarray | None:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return pose


def load_colmap_reference(images_path: Path) -> dict[str, np.ndarray]:
    if not images_path.exists():
        raise FileNotFoundError(images_path)
    output: dict[str, np.ndarray] = {}
    for image in read_images_binary(str(images_path)).values():
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = image.qvec2rotmat()
        pose[:3, 3] = image.tvec
        output[canonical_frame_id(image.name)] = pose
    if len(output) < 3:
        raise ValueError(f"Reference model has fewer than three poses: {images_path}")
    return output


def load_tum_reference(gt_dir: Path) -> dict[str, np.ndarray]:
    poses_path = gt_dir / "poses_w2c.npy"
    mask_path = gt_dir / "valid_mask.npy"
    map_path = gt_dir / "frame_map.csv"
    for path in (poses_path, mask_path, map_path):
        if not path.exists():
            raise FileNotFoundError(path)
    poses = np.load(poses_path)
    valid_mask = np.asarray(np.load(mask_path), dtype=bool)
    with map_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if poses.shape != (len(rows), 4, 4) or valid_mask.shape != (len(rows),):
        raise ValueError(f"Inconsistent TUM reference arrays in {gt_dir}")
    output: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        pose = valid_pose(poses[index])
        if valid_mask[index] and pose is not None:
            output[canonical_frame_id(row["image_name"])] = pose
    if len(output) < 3:
        raise ValueError(f"TUM reference has fewer than three valid poses: {gt_dir}")
    return output


def load_reference(spec: SceneSpec, reference_root: Path = REFERENCE_ROOT) -> dict[str, np.ndarray]:
    path = reference_root / spec.reference_path
    if spec.dataset == "TUM":
        return load_tum_reference(path)
    return load_colmap_reference(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_metadata_trajectory(model_dir: Path) -> dict[str, np.ndarray]:
    metadata = read_json(model_dir / "metadata.json")
    output: dict[str, np.ndarray] = {}
    for keyframe in metadata.get("keyframes", []):
        if not isinstance(keyframe, dict):
            continue
        info = keyframe.get("info", {})
        if not isinstance(info, dict):
            continue
        frame_id = canonical_frame_id(info.get("name"))
        pose = valid_pose(keyframe.get("Rt"))
        if frame_id and pose is not None:
            output[frame_id] = pose
    if len(output) < 3:
        raise ValueError(f"Model has fewer than three valid estimated poses: {model_dir}")
    return output


def _aligned_metrics(
    reference: dict[str, np.ndarray],
    estimated: dict[str, np.ndarray],
    frame_ids: list[str],
    *,
    rpe_delta: int,
) -> tuple[dict[str, Any], Similarity]:
    if len(frame_ids) < 3:
        raise ValueError("At least three common reference-valid frames are required")
    gt = np.stack([reference[frame_id] for frame_id in frame_ids])
    est = np.stack([estimated[frame_id] for frame_id in frame_ids])
    similarity = fit_similarity_w2c(est, gt)
    aligned = similarity.apply(est)
    ape = pose_errors(aligned, gt)
    rpe = relative_pose_errors(aligned, gt, delta=max(1, int(rpe_delta)))
    metrics = {
        "ape_count": int(ape["count"]),
        "ape_trans_mean": ape["trans_mean"],
        "ape_trans_rmse": ape["trans_rmse"],
        "ape_trans_median": ape["trans_median"],
        "ape_trans_p90": ape["trans_p90"],
        "ape_rot_deg_mean": ape["rot_deg_mean"],
        "ape_rot_deg_median": ape["rot_deg_median"],
        "rpe_count": int(rpe["count"]),
        "rpe_trans_mean": rpe["trans_mean"],
        "rpe_trans_rmse": rpe["trans_rmse"],
        "rpe_rot_deg_mean": rpe["rot_deg_mean"],
        "rpe_rot_deg_median": rpe["rot_deg_median"],
        "alignment_scale": float(similarity.scale),
    }
    return metrics, similarity


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
    result: dict[str, Any] = {
        "reference_frames": len(reference),
        "three_way_common_frames": len(common_ids),
        "three_way_common_frame_ids": common_ids,
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

    pair_ids = sorted(reference_ids & set(methods["v31"]) & set(methods["v31_a"]), key=natural_key)
    pair_methods: dict[str, Any] = {
        "common_frames": len(pair_ids),
        "common_frame_ids": pair_ids,
    }
    for method in ("v31", "v31_a"):
        metrics, _ = _aligned_metrics(
            reference,
            methods[method],
            pair_ids,
            rpe_delta=result["rpe_delta"],
        )
        pair_methods[method] = metrics
    result["v31_vs_a"] = pair_methods
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    x = _rankdata(np.asarray(left, dtype=np.float64))
    y = _rankdata(np.asarray(right, dtype=np.float64))
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _percentile_rank(values: list[float], value: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(100.0 * np.mean(array <= float(value))) if len(array) else 0.0


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-15:
        return None
    return float(numerator / denominator)


def evaluate_risk_trace(
    trace_path: Path,
    reference: dict[str, np.ndarray],
    alignment_trajectory: dict[str, np.ndarray],
    alignment_frame_ids: list[str],
) -> dict[str, Any]:
    trace = read_json(trace_path)
    alignment_ids = [
        frame_id
        for frame_id in alignment_frame_ids
        if frame_id in reference and frame_id in alignment_trajectory
    ]
    _, similarity = _aligned_metrics(
        reference,
        alignment_trajectory,
        alignment_ids,
        rpe_delta=1,
    )
    rows: list[dict[str, Any]] = []
    for event in trace.get("events", []):
        if not isinstance(event, dict):
            continue
        frame_id = canonical_frame_id(event.get("image_name"))
        pose = valid_pose(event.get("estimated_Rt"))
        if not frame_id or pose is None or frame_id not in reference:
            continue
        aligned = similarity.apply(pose[None])
        error = pose_errors(aligned, reference[frame_id][None])
        rows.append(
            {
                "frame_id": frame_id,
                "risk_score": float(event.get("risk_score", 0.0)),
                "risk_candidate": bool(event.get("risk_candidate", False)),
                "quarantined": bool(event.get("pose_reference_quarantined", False)),
                "translation_error": float(error["trans_mean"]),
                "rotation_error_deg": float(error["rot_deg_mean"]),
            }
        )
    trans = [row["translation_error"] for row in rows]
    rot = [row["rotation_error_deg"] for row in rows]
    risks = [row["risk_score"] for row in rows]
    flagged = [row for row in rows if row["risk_candidate"]]
    unflagged = [row for row in rows if not row["risk_candidate"]]
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        if not row["quarantined"]:
            continue
        quarantined.append(
            {
                **row,
                "translation_percentile": _percentile_rank(trans, row["translation_error"]),
                "rotation_percentile": _percentile_rank(rot, row["rotation_error_deg"]),
            }
        )
    flagged_trans = _mean([row["translation_error"] for row in flagged])
    unflagged_trans = _mean([row["translation_error"] for row in unflagged])
    flagged_rot = _mean([row["rotation_error_deg"] for row in flagged])
    unflagged_rot = _mean([row["rotation_error_deg"] for row in unflagged])
    return {
        "evaluated_events": len(rows),
        "risk_candidates": len(flagged),
        "quarantined_count": len(quarantined),
        "risk_translation_spearman": _spearman(risks, trans),
        "risk_rotation_spearman": _spearman(risks, rot),
        "candidate_translation_mean": flagged_trans,
        "noncandidate_translation_mean": unflagged_trans,
        "candidate_translation_enrichment": _ratio(flagged_trans, unflagged_trans),
        "candidate_rotation_mean": flagged_rot,
        "noncandidate_rotation_mean": unflagged_rot,
        "candidate_rotation_enrichment": _ratio(flagged_rot, unflagged_rot),
        "quarantined": quarantined,
    }


def metadata_summary(model_dir: Path) -> dict[str, Any]:
    metadata = read_json(model_dir / "metadata.json")
    return {
        "PSNR": metadata.get("PSNR"),
        "SSIM": metadata.get("SSIM"),
        "LPIPS": metadata.get("LPIPS"),
        "time": metadata.get("time"),
        "num_keyframes": metadata.get("num keyframes", metadata.get("num_keyframes")),
        "num_anchors": metadata.get("num anchors", metadata.get("num_anchors")),
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


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "--" if not math.isfinite(number) else f"{number:.{digits}f}"


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
    pair_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for scene_name in scene_names:
        spec = SCENES[scene_name]
        reference = load_reference(spec, reference_root)
        model_dirs = {method: run_root / method / scene_name / "model" for method in METHODS}
        trajectories = {method: load_metadata_trajectory(path) for method, path in model_dirs.items()}
        pose_report = evaluate_scene(reference, trajectories, rpe_delta=rpe_delta)
        diagnostic = evaluate_risk_trace(
            model_dirs["v31_a"] / "pose_risk_utility_trace.json",
            reference,
            trajectories["v31_a"],
            pose_report["three_way_common_frame_ids"],
        )
        report["scenes"][scene_name] = {
            "dataset": spec.dataset,
            "reference_type": spec.reference_type,
            "pose": pose_report,
            "a_module": diagnostic,
        }
        for method in METHODS:
            metrics = pose_report["methods"][method]
            pose_rows.append(
                {
                    "dataset": spec.dataset,
                    "scene": scene_name,
                    "reference_type": spec.reference_type,
                    "method": method,
                    "reference_frames": pose_report["reference_frames"],
                    "common_frames": pose_report["three_way_common_frames"],
                    **{key: value for key, value in metrics.items() if key != "alignment_scale"},
                    "alignment_scale": metrics["alignment_scale"],
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
        pair = pose_report["v31_vs_a"]
        row: dict[str, Any] = {
            "dataset": spec.dataset,
            "scene": scene_name,
            "reference_type": spec.reference_type,
            "common_frames": pair["common_frames"],
        }
        for method in ("v31", "v31_a"):
            for key, value in pair[method].items():
                row[f"{key}_{method}"] = value
        for key in ("ape_trans_rmse", "ape_rot_deg_mean", "rpe_trans_rmse", "rpe_rot_deg_mean"):
            row[f"delta_{key}"] = pair["v31_a"][key] - pair["v31"][key]
        pair_rows.append(row)
        diagnostic_rows.append(
            {
                "dataset": spec.dataset,
                "scene": scene_name,
                **{key: value for key, value in diagnostic.items() if key != "quarantined"},
                "quarantined_frames": ";".join(item["frame_id"] for item in diagnostic["quarantined"]),
            }
        )

    _write_csv(output_dir / "pose_three_way.csv", pose_rows)
    _write_csv(output_dir / "pose_v31_vs_a.csv", pair_rows)
    _write_csv(output_dir / "rendering_summary.csv", render_rows)
    _write_csv(output_dir / "a_module_diagnostics.csv", diagnostic_rows)
    (output_dir / "official_pose_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Official-Protocol Pose Benchmark",
        "",
        "TUM uses official mocap ground truth. MipNeRF360 and StaticHikes use COLMAP pseudo-ground truth. All pose metrics use independently fitted Sim(3) alignment on one three-way common frame set per scene.",
        "",
        "| Dataset | Scene | Method | Frames | Coverage | ATE-RMSE | APE-R deg | RPE-t RMSE | RPE-R deg |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pose_rows:
        lines.append(
            f"| {row['dataset']} | {row['scene']} | {row['method']} | "
            f"{row['common_frames']} | {_fmt(row['coverage'], 4)} | "
            f"{_fmt(row['ape_trans_rmse'])} | {_fmt(row['ape_rot_deg_mean'], 4)} | "
            f"{_fmt(row['rpe_trans_rmse'])} | {_fmt(row['rpe_rot_deg_mean'], 4)} |"
        )
    (output_dir / "official_pose_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--rpe_delta", type=int, default=1)
    parser.add_argument("--scenes", nargs="*", choices=sorted(SCENES), default=[])
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
    print(args.output_dir / "official_pose_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
