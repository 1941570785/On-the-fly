from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_DROP_THRESHOLDS = {
    "psnr": 1.0,
    "ssim": 0.03,
    "lpips": 0.03,
    "abs_trans_error": 0.5,
    "abs_rot_error_deg": 5.0,
}

METRIC_ALIASES = {
    "psnr": ("psnr", "PSNR"),
    "ssim": ("ssim", "SSIM"),
    "lpips": ("lpips", "LPIPS"),
    "abs_trans_error": (
        "abs_trans_error",
        "absolute_relative_translation_error",
        "translation_error",
        "t",
    ),
    "abs_rot_error_deg": (
        "abs_rot_error_deg",
        "absolute_relative_rotation_error",
        "absolute_relative_rotation_error_deg",
        "R_deg",
        "R°",
    ),
}

BASELINE_ALIASES = {
    "psnr": ("PSNR", "psnr"),
    "ssim": ("SSIM", "ssim"),
    "lpips": ("LPIPS", "lpips"),
    "abs_trans_error": ("t", "abs_trans_error", "absolute_relative_translation_error"),
    "abs_rot_error_deg": ("R°", "R_deg", "abs_rot_error_deg", "absolute_relative_rotation_error"),
}


INTERVAL_FIELDS = [
    "interval",
    "interval_lo",
    "interval_hi",
    "frame_count",
    "eval_frame_count",
    "psnr_mean",
    "ssim_mean",
    "lpips_mean",
    "abs_trans_error_mean",
    "abs_rot_error_deg_mean",
    "psnr_delta_from_previous",
    "ssim_delta_from_previous",
    "lpips_delta_from_previous",
    "abs_trans_error_delta_from_previous",
    "abs_rot_error_deg_delta_from_previous",
    "quality_drop_from_previous",
    "quality_drop_score",
    "drop_reasons",
    "psnr_delta_to_baseline",
    "ssim_delta_to_baseline",
    "lpips_delta_to_baseline",
    "abs_trans_error_delta_to_baseline",
    "abs_rot_error_deg_delta_to_baseline",
]


def build_interval_metric_evaluation(
    frame_metric_rows: list[dict[str, Any]],
    *,
    interval_size: int = 100,
    baseline: dict[str, Any] | None = None,
    drop_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    interval_size = max(1, int(interval_size))
    thresholds = dict(DEFAULT_DROP_THRESHOLDS)
    if drop_thresholds:
        thresholds.update({k: float(v) for k, v in drop_thresholds.items()})

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_metric_rows:
        frame_idx = _frame_index(row)
        if frame_idx is None:
            continue
        grouped[(frame_idx // interval_size) * interval_size].append(row)

    baseline_values = _normalize_baseline(baseline or {})
    interval_rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for lo in sorted(grouped):
        hi = lo + interval_size
        rows = grouped[lo]
        metrics = {name: _mean_metric(rows, name) for name in METRIC_ALIASES}
        eval_count = sum(1 for row in rows if _metric_value(row, "psnr") is not None)
        out: dict[str, Any] = {
            "interval": f"{lo}-{hi}",
            "interval_lo": lo,
            "interval_hi": hi,
            "frame_count": len(rows),
            "eval_frame_count": eval_count,
            "psnr_mean": metrics["psnr"],
            "ssim_mean": metrics["ssim"],
            "lpips_mean": metrics["lpips"],
            "abs_trans_error_mean": metrics["abs_trans_error"],
            "abs_rot_error_deg_mean": metrics["abs_rot_error_deg"],
        }
        _attach_previous_drops(out, previous, thresholds)
        _attach_baseline_deltas(out, baseline_values)
        interval_rows.append(out)
        previous = out

    worst = _worst_drop(interval_rows)
    return {
        "interval_size": interval_size,
        "interval_rows": interval_rows,
        "worst_quality_drop_interval": worst.get("interval", "") if worst else "",
        "worst_quality_drop_score": worst.get("quality_drop_score", 0.0) if worst else 0.0,
        "drop_thresholds": thresholds,
        "baseline": baseline_values,
    }


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] = INTERVAL_FIELDS) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_interval_report(evaluation: dict[str, Any]) -> str:
    lines = [
        "# Interval Metric Evaluation",
        "",
        f"- interval_size: {evaluation.get('interval_size')}",
        f"- worst_quality_drop_interval: {evaluation.get('worst_quality_drop_interval', '')}",
        f"- worst_quality_drop_score: {_fmt(evaluation.get('worst_quality_drop_score'))}",
        "",
        "| interval | eval frames | PSNR | SSIM | LPIPS | t err | R err | drop | reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in evaluation.get("interval_rows", []):
        lines.append(
            "| {interval} | {n} | {psnr} | {ssim} | {lpips} | {t} | {r} | {drop} | {reasons} |".format(
                interval=row.get("interval", ""),
                n=row.get("eval_frame_count", 0),
                psnr=_fmt(row.get("psnr_mean")),
                ssim=_fmt(row.get("ssim_mean")),
                lpips=_fmt(row.get("lpips_mean")),
                t=_fmt(row.get("abs_trans_error_mean")),
                r=_fmt(row.get("abs_rot_error_deg_mean")),
                drop=str(bool(row.get("quality_drop_from_previous"))),
                reasons=row.get("drop_reasons", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _frame_index(row: dict[str, Any]) -> int | None:
    for key in ("stream_frame_idx", "sequence_order", "frame_id", "source_frame_id", "frame_idx", "original_frame_idx"):
        value = _to_float(row.get(key))
        if value is not None:
            return int(value)
    name = str(row.get("original_image_name") or row.get("image_name") or row.get("render_image_name") or "")
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [_metric_value(row, metric) for row in rows]
    vals = [float(v) for v in values if v is not None]
    return float(mean(vals)) if vals else None


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    for key in METRIC_ALIASES[metric]:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _normalize_baseline(baseline: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric, aliases in BASELINE_ALIASES.items():
        for key in aliases:
            value = _to_float(baseline.get(key))
            if value is not None:
                out[metric] = value
                break
    return out


def _attach_previous_drops(out: dict[str, Any], previous: dict[str, Any] | None, thresholds: dict[str, float]) -> None:
    reasons: list[str] = []
    score = 0.0
    for metric, out_key in (
        ("psnr", "psnr_mean"),
        ("ssim", "ssim_mean"),
        ("lpips", "lpips_mean"),
        ("abs_trans_error", "abs_trans_error_mean"),
        ("abs_rot_error_deg", "abs_rot_error_deg_mean"),
    ):
        delta_key = out_key.replace("_mean", "_delta_from_previous")
        cur = out.get(out_key)
        prev = previous.get(out_key) if previous else None
        delta = None if cur is None or prev is None else float(cur) - float(prev)
        out[delta_key] = delta
        if delta is None:
            continue
        threshold = float(thresholds[metric])
        if metric in {"psnr", "ssim"} and delta <= -threshold:
            reasons.append(f"{metric}_drop")
            score += abs(delta) / threshold
        elif metric == "lpips" and delta >= threshold:
            reasons.append("lpips_increase")
            score += delta / threshold
        elif metric == "abs_trans_error" and delta >= threshold:
            reasons.append("translation_error_increase")
            score += delta / threshold
        elif metric == "abs_rot_error_deg" and delta >= threshold:
            reasons.append("rotation_error_increase")
            score += delta / threshold
    out["quality_drop_from_previous"] = bool(reasons)
    out["quality_drop_score"] = float(score)
    out["drop_reasons"] = ";".join(reasons)


def _attach_baseline_deltas(out: dict[str, Any], baseline_values: dict[str, float]) -> None:
    mapping = {
        "psnr": "psnr_mean",
        "ssim": "ssim_mean",
        "lpips": "lpips_mean",
        "abs_trans_error": "abs_trans_error_mean",
        "abs_rot_error_deg": "abs_rot_error_deg_mean",
    }
    for metric, out_key in mapping.items():
        cur = out.get(out_key)
        base = baseline_values.get(metric)
        out[out_key.replace("_mean", "_delta_to_baseline")] = None if cur is None or base is None else float(cur) - float(base)


def _worst_drop(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [row for row in rows if row.get("quality_drop_from_previous")]
    if not flagged:
        return {}
    return max(flagged, key=lambda row: float(row.get("quality_drop_score", 0.0) or 0.0))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
