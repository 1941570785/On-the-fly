from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path("/data2/zxd/3D_Reconstruction/On_the_fly")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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

METRICS = ["psnr", "ssim", "lpips", "abs_rot_error_deg", "abs_trans_error"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    method_root = Path(args.method_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for item in DATASETS:
        row = summarize_dataset(item, method_root, out_dir)
        summary_rows.append(row)
    write_summary(summary_rows, out_dir)
    return 0


def summarize_dataset(item: dict[str, str | int], method_root: Path, out_dir: Path) -> dict[str, Any]:
    name = str(item["name"])
    source = REPO / str(item["source"])
    test_hold = int(item["test_hold"])
    baseline_dir = REPO / str(item["baseline"])
    method_dir = method_root / name / "model"
    eval_names = official_eval_names(source, test_hold)
    baseline_rows = read_frame_metrics(baseline_dir)
    method_rows = read_frame_metrics(method_dir)
    aligned = align_metric_rows(eval_names, baseline_rows, method_rows)
    aligned_eval_names = [row["image_name"] for row in aligned]
    baseline_series = series_for_names(aligned_eval_names, baseline_rows)
    method_series = series_for_names(aligned_eval_names, method_rows)
    draw_quality_svg(
        baseline_series,
        out_dir / "plots" / f"{name}_baseline_quality.svg",
        title=f"{name} baseline test_hold={test_hold}",
    )
    draw_quality_svg(
        method_series,
        out_dir / "plots" / f"{name}_method_quality.svg",
        title=f"{name} method aligned to baseline test_hold={test_hold}",
    )
    draw_combined_quality_svg(
        baseline_series,
        method_series,
        out_dir / "plots" / f"{name}_combined_quality.svg",
        title=f"{name} baseline vs method aligned test_hold={test_hold}",
    )
    write_aligned_csv(out_dir / f"{name}_aligned_eval_frames.csv", aligned)
    base_meta = read_metadata(baseline_dir)
    method_meta = read_metadata(method_dir)
    means = aligned_means(aligned)
    return {
        "dataset": name,
        "test_hold": test_hold,
        "baseline_eval_frames": len(eval_names),
        "matched_eval_frames": len(aligned),
        "coverage": len(aligned) / len(eval_names) if eval_names else 0.0,
        "kf_base": int(_meta_value(base_meta, "num keyframes", 0) or 0),
        "kf_method": int(_meta_value(method_meta, "num keyframes", 0) or 0),
        "PSNR_base": means["baseline"]["psnr"],
        "PSNR_method": means["method"]["psnr"],
        "SSIM_base": means["baseline"]["ssim"],
        "SSIM_method": means["method"]["ssim"],
        "LPIPS_base": means["baseline"]["lpips"],
        "LPIPS_method": means["method"]["lpips"],
        "Rdeg_base": means["baseline"]["abs_rot_error_deg"],
        "Rdeg_method": means["method"]["abs_rot_error_deg"],
        "t_base": means["baseline"]["abs_trans_error"],
        "t_method": means["method"]["abs_trans_error"],
        "time_base": _to_float(_meta_value(base_meta, "time", "")),
        "time_method": _to_float(_meta_value(method_meta, "time", "")),
        "baseline_dir": str(baseline_dir),
        "method_dir": str(method_dir),
    }


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
                "frame_label": frame_label(image_name),
                "baseline": baseline[image_name],
                "method": method[image_name],
            }
        )
    return aligned


def official_eval_names(source: Path, test_hold: int) -> list[str]:
    names = get_image_names(source / "images")
    return names[:: int(test_hold)] if test_hold > 0 else []


def get_image_names(path: Path) -> list[str]:
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted(
        [p.name for p in Path(path).iterdir() if p.suffix in suffixes],
        key=_natural_key,
    )


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


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


def _rows_by_image_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if not _truthy(row.get("is_test_view")):
            continue
        name = str(row.get("original_image_name", "")).strip()
        if name:
            out[name] = row
    return out


def series_for_names(eval_names: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = _rows_by_image_name(rows)
    series = []
    for order, image_name in enumerate(eval_names):
        row = by_name.get(image_name)
        if row is None:
            continue
        series.append(
            {
                "eval_order": order,
                "image_name": image_name,
                "frame_label": frame_label(image_name),
                "psnr": _to_float(row.get("psnr")),
                "ssim": _to_float(row.get("ssim")),
                "lpips": _to_float(row.get("lpips")),
            }
        )
    return series


def aligned_means(aligned: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    out = {"baseline": {}, "method": {}}
    for side in out:
        for metric in METRICS:
            values = [
                _to_float(row[side].get(metric))
                for row in aligned
                if _to_float(row[side].get(metric)) is not None
            ]
            out[side][metric] = sum(values) / len(values) if values else None
    return out


def write_aligned_csv(path: Path, aligned: list[dict[str, Any]]) -> None:
    fields = [
        "eval_order",
        "image_name",
        "frame_label",
        "baseline_psnr",
        "method_psnr",
        "baseline_ssim",
        "method_ssim",
        "baseline_lpips",
        "method_lpips",
        "baseline_abs_rot_error_deg",
        "method_abs_rot_error_deg",
        "baseline_abs_trans_error",
        "method_abs_trans_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in aligned:
            out = {
                "eval_order": row["eval_order"],
                "image_name": row["image_name"],
                "frame_label": row["frame_label"],
            }
            for metric in METRICS:
                out[f"baseline_{metric}"] = row["baseline"].get(metric, "")
                out[f"method_{metric}"] = row["method"].get(metric, "")
            writer.writerow(out)


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "dataset",
        "test_hold",
        "baseline_eval_frames",
        "matched_eval_frames",
        "coverage",
        "kf_base",
        "kf_method",
        "PSNR_base",
        "PSNR_method",
        "SSIM_base",
        "SSIM_method",
        "LPIPS_base",
        "LPIPS_method",
        "Rdeg_base",
        "Rdeg_method",
        "t_base",
        "t_method",
        "time_base",
        "time_method",
        "baseline_dir",
        "method_dir",
    ]
    with (out_dir / "baseline_compatible_eval_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "baseline_compatible_eval_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = [
        "指标对比，delta = method - baseline；LPIPS/Rdeg/t/time 越低越好。只统计 baseline test_hold 帧与新模型实际覆盖帧的交集。",
        "",
        "| dataset | eval match | kf base -> method | PSNR base -> method | SSIM base -> method | LPIPS base -> method | Rdeg base -> method | t base -> method | time base -> method |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md.append(
            "| {dataset} | {matched}/{total} ({cov}) | {kf} | {psnr} | {ssim} | {lpips} | {rdeg} | {t} | {time} |".format(
                dataset=row["dataset"],
                matched=row["matched_eval_frames"],
                total=row["baseline_eval_frames"],
                cov=_fmt_pct(row["coverage"]),
                kf=_fmt_pair(row["kf_base"], row["kf_method"], 0),
                psnr=_fmt_pair(row["PSNR_base"], row["PSNR_method"], 4),
                ssim=_fmt_pair(row["SSIM_base"], row["SSIM_method"], 4),
                lpips=_fmt_pair(row["LPIPS_base"], row["LPIPS_method"], 4),
                rdeg=_fmt_pair(row["Rdeg_base"], row["Rdeg_method"], 4),
                t=_fmt_pair(row["t_base"], row["t_method"], 4),
                time=_fmt_pair(row["time_base"], row["time_method"], 2, suffix="s"),
            )
        )
    (out_dir / "baseline_compatible_eval_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )


def draw_quality_svg(series: list[dict[str, Any]], path: Path, title: str) -> None:
    width = 1200
    panel_height = 240
    top = 50
    left = 70
    right = 30
    bottom = 55
    height = top + 3 * panel_height + bottom
    colors = {"psnr": "#1f77b4", "ssim": "#2ca02c", "lpips": "#d62728"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{_esc(title)}</text>',
    ]
    for idx, metric in enumerate(["psnr", "ssim", "lpips"]):
        y0 = top + idx * panel_height
        parts.extend(_svg_panel(series, metric, colors[metric], left, y0, width - left - right, panel_height - 35))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def draw_combined_quality_svg(
    baseline_series: list[dict[str, Any]],
    method_series: list[dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    width = 1200
    panel_height = 240
    top = 70
    left = 70
    right = 30
    bottom = 55
    height = top + 3 * panel_height + bottom
    colors = {"baseline": "#1f77b4", "method": "#ff7f0e"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{_esc(title)}</text>',
        f'<line x1="{width/2-115:.1f}" y1="50" x2="{width/2-75:.1f}" y2="50" stroke="{colors["baseline"]}" stroke-width="3"/>',
        f'<text x="{width/2-68:.1f}" y="54" font-family="Arial" font-size="13">Baseline</text>',
        f'<line x1="{width/2+25:.1f}" y1="50" x2="{width/2+65:.1f}" y2="50" stroke="{colors["method"]}" stroke-width="3"/>',
        f'<text x="{width/2+72:.1f}" y="54" font-family="Arial" font-size="13">Method</text>',
    ]
    for idx, metric in enumerate(["psnr", "ssim", "lpips"]):
        y0 = top + idx * panel_height
        parts.extend(
            _svg_combined_panel(
                baseline_series,
                method_series,
                metric,
                colors,
                left,
                y0,
                width - left - right,
                panel_height - 35,
            )
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _svg_panel(series, metric, color, x0, y0, w, h):
    values = [row[metric] for row in series if row.get(metric) is not None]
    labels = [row["frame_label"] for row in series]
    if not values:
        values = [0.0, 1.0]
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    pad = (vmax - vmin) * 0.08
    vmin -= pad
    vmax += pad

    def sx(i):
        if len(series) <= 1:
            return x0 + w / 2
        return x0 + i * w / (len(series) - 1)

    def sy(v):
        return y0 + h - (float(v) - vmin) * h / (vmax - vmin)

    pts = [
        (sx(i), sy(row[metric]))
        for i, row in enumerate(series)
        if row.get(metric) is not None
    ]
    path_d = " ".join(
        ("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}"
        for i, (x, y) in enumerate(pts)
    )
    out = [
        f'<text x="{x0}" y="{y0+14}" font-family="Arial" font-size="15" font-weight="700">{metric.upper()}</text>',
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{x0-8}" y="{y0+5}" text-anchor="end" font-family="Arial" font-size="11">{vmax:.3g}</text>',
        f'<text x="{x0-8}" y="{y0+h}" text-anchor="end" font-family="Arial" font-size="11">{vmin:.3g}</text>',
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.2"/>',
    ]
    for i, row in enumerate(series):
        value = row.get(metric)
        if value is None:
            continue
        x, y = sx(i), sy(value)
        out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{color}"/>')
    label_font_size = 6 if len(labels) > 72 else 7 if len(labels) > 48 else 8
    for i, label in enumerate(labels):
        x = sx(i)
        out.append(
            f'<text x="{x:.2f}" y="{y0+h+14}" transform="rotate(65 {x:.2f} {y0+h+14})" font-family="Arial" font-size="{label_font_size}" text-anchor="start">{_esc(label)}</text>'
        )
    return out


def _svg_combined_panel(baseline_series, method_series, metric, colors, x0, y0, w, h):
    series_by_name = {"baseline": baseline_series, "method": method_series}
    labels = _combined_frame_labels(baseline_series, method_series)
    values = []
    for series in series_by_name.values():
        values.extend(row[metric] for row in series if row.get(metric) is not None)
    if not values:
        values = [0.0, 1.0]
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    pad = (vmax - vmin) * 0.08
    vmin -= pad
    vmax += pad
    n = max(len(baseline_series), len(method_series), 1)

    def sx(i):
        if n <= 1:
            return x0 + w / 2
        return x0 + i * w / (n - 1)

    def sy(v):
        return y0 + h - (float(v) - vmin) * h / (vmax - vmin)

    out = [
        f'<text x="{x0}" y="{y0+14}" font-family="Arial" font-size="15" font-weight="700">{metric.upper()}</text>',
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{x0-8}" y="{y0+5}" text-anchor="end" font-family="Arial" font-size="11">{vmax:.3g}</text>',
        f'<text x="{x0-8}" y="{y0+h}" text-anchor="end" font-family="Arial" font-size="11">{vmin:.3g}</text>',
    ]
    for name, series in series_by_name.items():
        pts = [
            (sx(i), sy(row[metric]))
            for i, row in enumerate(series)
            if row.get(metric) is not None
        ]
        path_d = " ".join(
            ("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}"
            for i, (x, y) in enumerate(pts)
        )
        if path_d:
            out.append(
                f'<path data-series="{name}-{metric}" d="{path_d}" fill="none" stroke="{colors[name]}" stroke-width="2.2"/>'
            )
        for i, row in enumerate(series):
            value = row.get(metric)
            if value is None:
                continue
            x, y = sx(i), sy(value)
            out.append(
                f'<circle data-series="{name}-{metric}" cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="{colors[name]}"/>'
            )
    label_font_size = 6 if len(labels) > 72 else 7 if len(labels) > 48 else 8
    for i, label in enumerate(labels):
        x = sx(i)
        out.append(
            f'<text x="{x:.2f}" y="{y0+h+14}" transform="rotate(65 {x:.2f} {y0+h+14})" font-family="Arial" font-size="{label_font_size}" text-anchor="start">{_esc(label)}</text>'
        )
    return out


def _combined_frame_labels(
    baseline_series: list[dict[str, Any]],
    method_series: list[dict[str, Any]],
) -> list[str]:
    labels = []
    n = max(len(baseline_series), len(method_series))
    for i in range(n):
        label = ""
        if i < len(baseline_series):
            label = str(baseline_series[i].get("frame_label", ""))
        if not label and i < len(method_series):
            label = str(method_series[i].get("frame_label", ""))
        labels.append(label)
    return labels


def frame_label(image_name: str) -> str:
    stem = Path(str(image_name)).stem
    matches = re.findall(r"\d+", stem)
    return matches[-1] if matches else stem


def _meta_value(meta: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in meta:
        return meta[key]
    if key == "Rdeg_base":
        return meta.get("R°", default)
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    value = _to_float(value)
    return "" if value is None else f"{100*value:.1f}%"


def _fmt_pair(base: Any, method: Any, digits: int, suffix: str = "") -> str:
    base_f = _to_float(base)
    method_f = _to_float(method)
    if base_f is None or method_f is None:
        return ""
    delta = method_f - base_f
    fmt = f"{{:.{digits}f}}"
    if digits == 0:
        fmt = "{:.0f}"
    return f"{fmt.format(base_f)}{suffix} -> {fmt.format(method_f)}{suffix} ({delta:+.{digits}f}{suffix})"


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
