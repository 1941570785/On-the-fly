from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any


FRAME_METRIC_FIELDS = [
    "dataset_name",
    "scene_name",
    "frame_idx",
    "original_frame_idx",
    "stream_frame_idx",
    "original_image_name",
    "render_image_name",
    "sequence_order",
    "is_test_view",
    "is_keyframe",
    "is_registered",
    "registration_status",
    "pose_stage",
    "pose_format",
    "split",
    "est_r00",
    "est_r01",
    "est_r02",
    "est_r10",
    "est_r11",
    "est_r12",
    "est_r20",
    "est_r21",
    "est_r22",
    "est_tx",
    "est_ty",
    "est_tz",
    "psnr",
    "ssim",
    "lpips",
    "abs_trans_error",
    "abs_rot_error_deg",
    "rel_trans_error",
    "rel_rot_error_deg",
    "output_dir",
]


def infer_dataset_scene_from_output_dir(output_dir: str | Path) -> tuple[str, str]:
    parts = Path(str(output_dir)).parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 2 < len(parts):
            return parts[idx + 1], parts[idx + 2]
    return "", ""


def build_frame_metric_row(
    *,
    dataset_name: str,
    scene_name: str,
    frame_idx: int,
    original_image_name: str,
    sequence_order: int,
    is_test_view: bool,
    is_keyframe: bool,
    is_registered: bool,
    est_rt: Any,
    quality: dict[str, Any] | None,
    pose_error: dict[str, Any] | None,
    output_dir: str | Path,
    stream_frame_idx: int | None = None,
) -> dict[str, Any]:
    quality = quality or {}
    pose_error = pose_error or {}
    row = {field: None for field in FRAME_METRIC_FIELDS}
    row.update(
        {
            "dataset_name": dataset_name,
            "scene_name": scene_name,
            "frame_idx": int(frame_idx),
            "original_frame_idx": frame_index_from_name(original_image_name),
            "stream_frame_idx": int(
                stream_frame_idx if stream_frame_idx is not None else sequence_order
            ),
            "original_image_name": original_image_name,
            "render_image_name": original_image_name if is_test_view else "",
            "sequence_order": int(sequence_order),
            "is_test_view": bool(is_test_view),
            "is_keyframe": bool(is_keyframe),
            "is_registered": bool(is_registered),
            "registration_status": "registered" if is_registered else "unregistered",
            "pose_stage": "final",
            "pose_format": "w2c",
            "split": "test" if is_test_view else "train",
            "psnr": _first_float(quality, "psnr", "PSNR"),
            "ssim": _first_float(quality, "ssim", "SSIM"),
            "lpips": _first_float(quality, "lpips", "LPIPS"),
            "abs_trans_error": _first_float(
                pose_error,
                "abs_trans_error",
                "absolute_relative_translation_error",
                "t",
            ),
            "abs_rot_error_deg": _first_float(
                pose_error,
                "abs_rot_error_deg",
                "absolute_relative_rotation_error",
                "R_deg",
                "R°",
            ),
            "rel_trans_error": _first_float(pose_error, "rel_trans_error"),
            "rel_rot_error_deg": _first_float(pose_error, "rel_rot_error_deg"),
            "output_dir": str(output_dir),
        }
    )
    row.update(_flatten_estimated_pose(est_rt))
    return row


def frame_index_from_name(image_name: str) -> int:
    stem = Path(str(image_name)).stem
    matches = list(re.finditer(r"\d+", stem))
    if not matches:
        return -1
    return int(matches[-1].group(0))


def write_frame_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _encode_cell(row.get(field)) for field in FRAME_METRIC_FIELDS})


def _flatten_estimated_pose(est_rt: Any) -> dict[str, Any]:
    out = {}
    keys = [
        "est_r00",
        "est_r01",
        "est_r02",
        "est_r10",
        "est_r11",
        "est_r12",
        "est_r20",
        "est_r21",
        "est_r22",
        "est_tx",
        "est_ty",
        "est_tz",
    ]
    for key in keys:
        out[key] = None
    if est_rt is None:
        return out

    try:
        rows = est_rt.tolist() if hasattr(est_rt, "tolist") else est_rt
        out.update(
            {
                "est_r00": _to_float(rows[0][0]),
                "est_r01": _to_float(rows[0][1]),
                "est_r02": _to_float(rows[0][2]),
                "est_r10": _to_float(rows[1][0]),
                "est_r11": _to_float(rows[1][1]),
                "est_r12": _to_float(rows[1][2]),
                "est_r20": _to_float(rows[2][0]),
                "est_r21": _to_float(rows[2][1]),
                "est_r22": _to_float(rows[2][2]),
                "est_tx": _to_float(rows[0][3]),
                "est_ty": _to_float(rows[1][3]),
                "est_tz": _to_float(rows[2][3]),
            }
        )
    except Exception:
        return out
    return out


def _first_float(values: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(values.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def _encode_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value
