from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/data2/zxd/3D_Reconstruction/On_the_fly")

DATASETS = [
    {
        "name": "bonsai",
        "source": "datasets/MipNeRF360/bonsai",
        "test_hold": 8,
        "baseline": "results/MipNeRF360/bonsai",
    },
    {
        "name": "counter",
        "source": "datasets/MipNeRF360/counter",
        "test_hold": 8,
        "baseline": "results/MipNeRF360/counter",
    },
    {
        "name": "garden",
        "source": "datasets/MipNeRF360/garden",
        "test_hold": 8,
        "baseline": "results/MipNeRF360/garden",
    },
    {
        "name": "forest1",
        "source": "datasets/StaticHikes/forest1",
        "test_hold": 10,
        "baseline": "results/StaticHikes/forest1/compatible",
    },
    {
        "name": "forest2",
        "source": "datasets/StaticHikes/forest2",
        "test_hold": 10,
        "baseline": "results/StaticHikes/forest2",
    },
    {
        "name": "university2",
        "source": "datasets/StaticHikes/university2",
        "test_hold": 10,
        "baseline": "results/StaticHikes/university2/compatible",
    },
    {
        "name": "desk1",
        "source": "datasets/TUM/desk1",
        "test_hold": 30,
        "baseline": "results/TUM/desk1",
    },
    {
        "name": "desk2",
        "source": "datasets/TUM/desk2",
        "test_hold": 30,
        "baseline": "results/TUM/desk2",
    },
    {
        "name": "long_office_household",
        "source": "datasets/TUM/long_office_household",
        "test_hold": 30,
        "baseline": "results/TUM/long_office_household/compatible",
    },
]

QUALITY_METRICS = ["psnr", "ssim", "lpips"]


@dataclass(frozen=True)
class Similarity:
    rotation: np.ndarray
    translation: np.ndarray
    scale: float

    def apply(self, poses_w2c: np.ndarray) -> np.ndarray:
        poses_w2c = _pose_array(poses_w2c)
        c2w = np.linalg.inv(poses_w2c)
        aligned_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(c2w), axis=0)
        aligned_c2w[:, :3, :3] = self.rotation[None] @ c2w[:, :3, :3]
        aligned_c2w[:, :3, 3] = (
            self.scale * (self.rotation[None] @ c2w[:, :3, 3, None])[:, :, 0]
            + self.translation[None]
        )
        return np.linalg.inv(aligned_c2w)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--method_override",
        action="append",
        default=[],
        help="Dataset-specific method model dir override, formatted as dataset=/path/to/model.",
    )
    parser.add_argument("--rpe_delta", type=int, default=1)
    args = parser.parse_args()

    method_root = Path(args.method_root)
    out_dir = Path(args.out_dir)
    overrides = _parse_overrides(args.method_override)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in DATASETS:
        rows.append(
            summarize_dataset(
                item,
                method_root=method_root,
                method_overrides=overrides,
                rpe_delta=max(int(args.rpe_delta), 1),
            )
        )
    write_summary(out_dir, rows)
    return 0


def summarize_dataset(
    item: dict[str, str | int],
    *,
    method_root: Path,
    method_overrides: dict[str, Path],
    rpe_delta: int,
) -> dict[str, Any]:
    name = str(item["name"])
    source = REPO / str(item["source"])
    test_hold = int(item["test_hold"])
    baseline_dir = REPO / str(item["baseline"])
    method_dir = method_overrides.get(name, method_root / name / "model")

    baseline_meta = read_metadata(baseline_dir)
    method_meta = read_metadata(method_dir)
    pose_result = evaluate_metadata_pair(baseline_meta, method_meta, rpe_delta=rpe_delta)

    eval_names = official_eval_names(source, test_hold)
    quality_rows = align_metric_rows(
        eval_names,
        read_frame_metrics(baseline_dir),
        read_frame_metrics(method_dir),
    )
    quality = aligned_quality_means(quality_rows)

    row: dict[str, Any] = {
        "dataset": name,
        "test_hold": test_hold,
        "baseline_eval_frames": len(eval_names),
        "matched_eval_frames": len(quality_rows),
        "quality_coverage": len(quality_rows) / len(eval_names) if eval_names else 0.0,
        "pose_eval_frames": pose_result["pose_eval_frames"],
        "pose_common_frames": pose_result["common_frames"],
        "pose_alignment_frames": pose_result["alignment_frames"],
        "rpe_delta": rpe_delta,
        "kf_base": int(_meta_value(baseline_meta, "num keyframes", 0) or 0),
        "kf_method": int(_meta_value(method_meta, "num keyframes", 0) or 0),
        "time_base": _to_float(_meta_value(baseline_meta, "time", "")),
        "time_method": _to_float(_meta_value(method_meta, "time", "")),
        "baseline_dir": str(baseline_dir),
        "method_dir": str(method_dir),
    }
    for metric in QUALITY_METRICS:
        row[f"{metric.upper()}_base"] = quality["baseline"].get(metric)
        row[f"{metric.upper()}_method"] = quality["method"].get(metric)
    for side in ["baseline", "method"]:
        prefix = "base" if side == "baseline" else "method"
        for key, value in pose_result[side].items():
            row[f"{key}_{prefix}"] = value
    return row


def evaluate_metadata_pair(
    baseline_metadata: dict[str, Any],
    method_metadata: dict[str, Any],
    *,
    rpe_delta: int = 1,
) -> dict[str, Any]:
    baseline = _keyframes_by_name(baseline_metadata)
    method = _keyframes_by_name(method_metadata)
    common_names = [
        name
        for name in sorted(set(baseline) & set(method), key=_natural_key)
        if baseline[name]["gt"] is not None and method[name]["gt"] is not None
    ]
    if len(common_names) < 3:
        raise ValueError("At least three common GT-valid frames are required")

    gt_eval = _stack_pose(baseline, common_names, "gt")
    baseline_est = _stack_pose(baseline, common_names, "estimated")
    method_est = _stack_pose(method, common_names, "estimated")

    baseline_similarity = fit_similarity_w2c(baseline_est, gt_eval)
    method_similarity = fit_similarity_w2c(method_est, gt_eval)
    baseline_aligned = baseline_similarity.apply(baseline_est)
    method_aligned = method_similarity.apply(method_est)

    baseline_ape = pose_errors(baseline_aligned, gt_eval)
    method_ape = pose_errors(method_aligned, gt_eval)
    baseline_rpe = relative_pose_errors(baseline_aligned, gt_eval, delta=rpe_delta)
    method_rpe = relative_pose_errors(method_aligned, gt_eval, delta=rpe_delta)

    return {
        "common_frames": len(common_names),
        "alignment_frames": len(common_names),
        "pose_eval_frames": len(common_names),
        "baseline_keyframes": len(baseline),
        "method_keyframes": len(method),
        "baseline": _pose_summary(baseline_ape, baseline_rpe, baseline_similarity),
        "method": _pose_summary(method_ape, method_rpe, method_similarity),
    }


def fit_similarity_w2c(estimated_w2c: np.ndarray, gt_w2c: np.ndarray) -> Similarity:
    estimated_w2c = _pose_array(estimated_w2c)
    gt_w2c = _pose_array(gt_w2c)
    if estimated_w2c.shape != gt_w2c.shape or len(estimated_w2c) < 3:
        raise ValueError("At least three paired poses with identical shape are required")

    estimated_centers = np.linalg.inv(estimated_w2c)[:, :3, 3]
    gt_centers = np.linalg.inv(gt_w2c)[:, :3, 3]
    estimated_mean = estimated_centers.mean(axis=0)
    gt_mean = gt_centers.mean(axis=0)
    estimated_centered = estimated_centers - estimated_mean
    gt_centered = gt_centers - gt_mean
    variance = float(np.mean(np.sum(estimated_centered**2, axis=1)))
    if variance <= 1e-15:
        raise ValueError("Estimated camera centers are degenerate")

    covariance = gt_centered.T @ estimated_centered / len(estimated_centers)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / variance)
    translation = gt_mean - scale * (rotation @ estimated_mean)
    return Similarity(rotation=rotation, translation=translation, scale=scale)


def pose_errors(aligned_w2c: np.ndarray, gt_w2c: np.ndarray) -> dict[str, Any]:
    aligned_c2w = np.linalg.inv(_pose_array(aligned_w2c))
    gt_c2w = np.linalg.inv(_pose_array(gt_w2c))
    trans = np.linalg.norm(aligned_c2w[:, :3, 3] - gt_c2w[:, :3, 3], axis=1)
    rot_rad = _rotation_error_rad(aligned_c2w[:, :3, :3], gt_c2w[:, :3, :3])
    return _error_summary(trans, rot_rad)


def relative_pose_errors(
    aligned_w2c: np.ndarray,
    gt_w2c: np.ndarray,
    *,
    delta: int = 1,
) -> dict[str, Any]:
    aligned_c2w = np.linalg.inv(_pose_array(aligned_w2c))
    gt_c2w = np.linalg.inv(_pose_array(gt_w2c))
    delta = max(int(delta), 1)
    if len(aligned_c2w) <= delta:
        empty = np.asarray([], dtype=np.float64)
        return _error_summary(empty, empty)
    estimated_rel = np.linalg.inv(aligned_c2w[:-delta]) @ aligned_c2w[delta:]
    gt_rel = np.linalg.inv(gt_c2w[:-delta]) @ gt_c2w[delta:]
    error = np.linalg.inv(gt_rel) @ estimated_rel
    trans = np.linalg.norm(error[:, :3, 3], axis=1)
    rot_rad = _rotation_error_rad(
        error[:, :3, :3],
        np.repeat(np.eye(3, dtype=np.float64)[None], len(error), axis=0),
    )
    return _error_summary(trans, rot_rad)


def _pose_summary(ape: dict[str, Any], rpe: dict[str, Any], similarity: Similarity) -> dict[str, float | int | None]:
    return {
        "ape_count": ape["count"],
        "ape_trans_mean": ape["trans_mean"],
        "ape_trans_median": ape["trans_median"],
        "ape_trans_rmse": ape["trans_rmse"],
        "ape_trans_p90": ape["trans_p90"],
        "ape_rot_rad_mean": ape["rot_rad_mean"],
        "ape_rot_deg_mean": ape["rot_deg_mean"],
        "ape_rot_deg_median": ape["rot_deg_median"],
        "rpe_count": rpe["count"],
        "rpe_trans_mean": rpe["trans_mean"],
        "rpe_trans_median": rpe["trans_median"],
        "rpe_trans_rmse": rpe["trans_rmse"],
        "rpe_rot_rad_mean": rpe["rot_rad_mean"],
        "rpe_rot_deg_mean": rpe["rot_deg_mean"],
        "rpe_rot_deg_median": rpe["rot_deg_median"],
        "alignment_scale": similarity.scale,
    }


def _error_summary(trans: np.ndarray, rot_rad: np.ndarray) -> dict[str, Any]:
    rot_deg = np.degrees(rot_rad)
    return {
        "count": int(len(trans)),
        "trans_mean": _mean(trans),
        "trans_median": _median(trans),
        "trans_rmse": _rmse(trans),
        "trans_p90": _percentile(trans, 90),
        "rot_rad_mean": _mean(rot_rad),
        "rot_rad_median": _median(rot_rad),
        "rot_deg_mean": _mean(rot_deg),
        "rot_deg_median": _median(rot_deg),
        "rot_deg_p90": _percentile(rot_deg, 90),
    }


def official_eval_names(source: Path, test_hold: int) -> list[str]:
    names = get_image_names(source / "images")
    return names[:: int(test_hold)] if test_hold > 0 else []


def get_image_names(path: Path) -> list[str]:
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.suffix in suffixes], key=_natural_key)


def read_frame_metrics(result_dir: Path) -> list[dict[str, Any]]:
    path = result_dir / "frame_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_metadata(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def align_metric_rows(
    eval_names: list[str],
    baseline_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = _rows_by_image_name(baseline_rows)
    method = _rows_by_image_name(method_rows)
    aligned = []
    for order, image_name in enumerate(eval_names):
        if image_name not in baseline or image_name not in method:
            continue
        aligned.append(
            {
                "eval_order": order,
                "image_name": image_name,
                "baseline": baseline[image_name],
                "method": method[image_name],
            }
        )
    return aligned


def aligned_quality_means(aligned: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        side: {
            metric: _mean(
                np.asarray(
                    [
                        _to_float(row[side].get(metric))
                        for row in aligned
                        if _to_float(row[side].get(metric)) is not None
                    ],
                    dtype=np.float64,
                )
            )
            for metric in QUALITY_METRICS
        }
        for side in ["baseline", "method"]
    }


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = out_dir / "standard_pose_eval_summary.csv"
    json_path = out_dir / "standard_pose_eval_summary.json"
    md_path = out_dir / "standard_pose_eval_summary.md"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_summary_markdown(rows), encoding="utf-8")


def _summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Standard pose evaluation. Delta = method - baseline; LPIPS, APE, RPE, and time are lower-is-better.",
        "",
        "| dataset | eval match | kf base -> method | PSNR | SSIM | LPIPS | T.APE | R.APE(rad/deg) | T.RPE | R.RPE(rad/deg) | time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {matched}/{total} | {kf} | {psnr} | {ssim} | {lpips} | {tape} | {rape} | {trpe} | {rrpe} | {time} |".format(
                dataset=row["dataset"],
                matched=row["matched_eval_frames"],
                total=row["baseline_eval_frames"],
                kf=_fmt_pair(row["kf_base"], row["kf_method"], precision=0),
                psnr=_fmt_pair(row.get("PSNR_base"), row.get("PSNR_method")),
                ssim=_fmt_pair(row.get("SSIM_base"), row.get("SSIM_method")),
                lpips=_fmt_pair(row.get("LPIPS_base"), row.get("LPIPS_method")),
                tape=_fmt_pair(row.get("ape_trans_mean_base"), row.get("ape_trans_mean_method")),
                rape=_fmt_pose_rot(row, "ape"),
                trpe=_fmt_pair(row.get("rpe_trans_mean_base"), row.get("rpe_trans_mean_method")),
                rrpe=_fmt_pose_rot(row, "rpe"),
                time=_fmt_time(row.get("time_base"), row.get("time_method")),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt_pair(base: Any, method: Any, *, precision: int = 4) -> str:
    b = _to_float(base)
    m = _to_float(method)
    if b is None or m is None:
        return "NA"
    if precision == 0:
        return f"{int(round(b))} -> {int(round(m))} ({int(round(m - b)):+d})"
    return f"{b:.{precision}f} -> {m:.{precision}f} ({m - b:+.{precision}f})"


def _fmt_pose_rot(row: dict[str, Any], prefix: str) -> str:
    rad = _fmt_pair(row.get(f"{prefix}_rot_rad_mean_base"), row.get(f"{prefix}_rot_rad_mean_method"))
    deg = _fmt_pair(row.get(f"{prefix}_rot_deg_mean_base"), row.get(f"{prefix}_rot_deg_mean_method"))
    return f"{rad}; {deg}deg"


def _fmt_time(base: Any, method: Any) -> str:
    b = _to_float(base)
    m = _to_float(method)
    if b is None or m is None:
        return "NA"
    return f"{b:.2f}s -> {m:.2f}s ({m - b:+.2f}s)"


def _keyframes_by_name(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for keyframe in metadata.get("keyframes", []) or []:
        info = keyframe.get("info", {}) or {}
        name = str(info.get("name", "")).strip()
        estimated = keyframe.get("Rt")
        if not name or estimated is None:
            continue
        gt = info.get("gt_Rt", info.get("Rt"))
        out[name] = {
            "estimated": np.asarray(estimated, dtype=np.float64),
            "gt": None if gt is None else np.asarray(gt, dtype=np.float64),
            "is_test": bool(info.get("is_test", False)),
        }
    return out


def _rows_by_image_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if not _truthy(row.get("is_test_view")):
            continue
        name = str(row.get("original_image_name", "")).strip()
        if name:
            out[name] = row
    return out


def _stack_pose(records: dict[str, dict[str, Any]], names: list[str], key: str) -> np.ndarray:
    return np.stack([records[name][key] for name in names]).astype(np.float64)


def _pose_array(value: Any) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 3 or out.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose array [N,4,4], got {out.shape}")
    return out


def _rotation_error_rad(rotation: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = rotation @ np.swapaxes(target, -1, -2)
    trace = np.trace(diff, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _parse_overrides(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid override {value!r}; expected dataset=/path/to/model")
        dataset, path = value.split("=", 1)
        out[dataset.strip()] = Path(path.strip())
    return out


def _meta_value(meta: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in meta:
        return meta[key]
    if key == "num keyframes" and "num_keyframes" in meta:
        return meta["num_keyframes"]
    return default


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str) and value.lower() == "nan":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(np.mean(values))


def _median(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(np.median(values))


def _rmse(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(np.sqrt(np.mean(values**2)))


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return None if len(values) == 0 else float(np.percentile(values, percentile))


if __name__ == "__main__":
    raise SystemExit(main())
