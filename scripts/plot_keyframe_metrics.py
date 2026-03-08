import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


METRIC_SPECS = {
    "psnr": {"label": "PSNR", "y_min": 10.0, "y_max": 35.0, "filename": "PSNR.svg"},
    "ssim": {"label": "SSIM", "y_min": 0.0, "y_max": 1.0, "filename": "SSIM.svg"},
    "lpips": {"label": "LPIPS", "y_min": 0.0, "y_max": 1.0, "filename": "LPIPS.svg"},
}

DEFAULT_ROOTS = ["results/MipNeRF360", "results/StaticHikes", "results/TUM"]
DEFAULT_COMPARE_ROOT_A = ["results/MipNeRF360", "results/StaticHikes", "results/TUM"]
DEFAULT_COMPARE_ROOT_B = [
    "results/ablation/NoJoint/MipNeRF360",
    "results/ablation/NoJoint/StaticHikes",
    "results/ablation/NoJoint/TUM",
]
DEFAULT_COMPARE_OUTPUT = "contrast"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-frame PSNR/SSIM/LPIPS as SVG figures. Supports single-run plotting and "
            "normal-vs-ablation comparison plotting."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default="single",
        help="single: plot each dataset independently; compare: overlay two experiment trees.",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_ROOTS,
        help="Dataset root directories for single mode.",
    )
    parser.add_argument(
        "--compare-roots-a",
        nargs="+",
        default=DEFAULT_COMPARE_ROOT_A,
        help="Base experiment roots for compare mode.",
    )
    parser.add_argument(
        "--compare-roots-b",
        nargs="+",
        default=DEFAULT_COMPARE_ROOT_B,
        help="Secondary experiment roots for compare mode.",
    )
    parser.add_argument(
        "--compare-label-a",
        default="Normal",
        help="Legend label for compare-roots-a.",
    )
    parser.add_argument(
        "--compare-label-b",
        default="NoJoint",
        help="Legend label for compare-roots-b.",
    )
    parser.add_argument(
        "--compare-output-root",
        default=DEFAULT_COMPARE_OUTPUT,
        help="Output directory root for compare mode.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "train", "test"],
        default="all",
        help="Filter frames by is_test when reading keyframe_metrics.csv.",
    )
    parser.add_argument(
        "--ext",
        choices=["svg"],
        default="svg",
        help="Output figure format. Currently only svg is supported without external dependencies.",
    )
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _fmt(value: float) -> str:
    if value != value or math.isinf(value):
        return "nan"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _nice_step(n: int, target_ticks: int = 10) -> int:
    if n <= 1:
        return 1
    raw = max(1, int(math.ceil(n / target_ticks)))
    scale = 10 ** int(math.floor(math.log10(raw)))
    normalized = raw / scale
    if normalized <= 1:
        snapped = 1
    elif normalized <= 2:
        snapped = 2
    elif normalized <= 5:
        snapped = 5
    else:
        snapped = 10
    return int(snapped * scale)


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _read_rows(csv_path: Path, only: str) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                is_test = _as_bool(row.get("is_test", "false"))
                if only == "test" and not is_test:
                    continue
                if only == "train" and is_test:
                    continue
                rows.append(
                    {
                        "keyframe_id": int(float(row.get("keyframe_id", "0"))),
                        "frame_id": int(float(row.get("frame_id", "-1"))),
                        "image_name": row.get("image_name", ""),
                        "is_test": is_test,
                        "psnr": float(row.get("psnr", "nan")),
                        "ssim": float(row.get("ssim", "nan")),
                        "lpips": float(row.get("lpips", "nan")),
                    }
                )
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda item: (item["keyframe_id"], item["frame_id"]))
    return rows


def _read_metadata_metrics(metadata_path: Path) -> dict[str, float]:
    with metadata_path.open("r") as handle:
        metadata = json.load(handle)
    return {
        "psnr": float(metadata.get("PSNR", float("nan"))),
        "ssim": float(metadata.get("SSIM", float("nan"))),
        "lpips": float(metadata.get("LPIPS", float("nan"))),
    }


def _dataset_dirs(roots: Iterable[str]) -> list[Path]:
    matched = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for metrics_csv in sorted(root_path.glob("*/keyframe_metrics.csv")):
            dataset_dir = metrics_csv.parent
            if (dataset_dir / "metadata.json").exists():
                matched.append(dataset_dir)
    return matched


def _relative_dataset_map(roots: Iterable[str]) -> dict[str, Path]:
    dataset_map: dict[str, Path] = {}
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for dataset_dir in _dataset_dirs([root]):
            rel = dataset_dir.relative_to(root_path).as_posix()
            dataset_map[rel] = dataset_dir
    return dataset_map


def _series_payload(rows: list[dict], metric_name: str) -> dict:
    return {
        "xs": list(range(1, len(rows) + 1)),
        "ys": [row[metric_name] for row in rows],
    }


def _base_svg_style() -> str:
    return """
        .title { font: 700 28px Arial, sans-serif; fill: #111827; }
        .subtitle { font: 400 18px Arial, sans-serif; fill: #4b5563; }
        .axis { stroke: #111827; stroke-width: 2; }
        .grid { stroke: #e5e7eb; stroke-width: 1; }
        .tick { font: 15px Arial, sans-serif; fill: #374151; }
        .axis-label { font: 18px Arial, sans-serif; fill: #111827; }
        .legend-text { font: 16px Arial, sans-serif; fill: #111827; }
        .legend-box { fill: #ffffff; stroke: #d1d5db; stroke-width: 1; }
    """


def _write_empty_svg(out_path: Path, message: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="200">'
        f'<text x="20" y="40" font-size="24">{_svg_escape(message)}</text>'
        "</svg>",
        encoding="utf-8",
    )


def _write_svg_lineplot(
    out_path: Path,
    title: str,
    metric_label: str,
    xs: list[int],
    ys: list[float],
    y_min: float,
    y_max: float,
    mean_value: float,
):
    width = 1600
    height = 720
    pad_left = 110
    pad_right = 40
    pad_top = 80
    pad_bottom = 90
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    valid_points = [(x, y) for x, y in zip(xs, ys) if y == y and not math.isinf(y)]
    if not valid_points:
        _write_empty_svg(out_path, "No valid points")
        return

    x0 = min(xs)
    x1 = max(xs)
    if x1 == x0:
        x1 += 1

    def tx(x: float) -> float:
        return pad_left + (x - x0) * plot_width / (x1 - x0)

    def ty(y: float) -> float:
        clamped = min(max(y, y_min), y_max)
        return pad_top + (y_max - clamped) * plot_height / (y_max - y_min)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>",
        _base_svg_style()
        + """
        .line { fill: none; stroke: #2563eb; stroke-width: 2.5; }
        .point { fill: #1d4ed8; stroke: #ffffff; stroke-width: 1.5; }
        .mean-line { stroke: #dc2626; stroke-width: 2.5; stroke-dasharray: 10 6; }
        .mean-label { font: 700 16px Arial, sans-serif; fill: #dc2626; }
        .below-mean { fill: rgba(220, 38, 38, 0.08); }
        """,
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{pad_left}" y="42">{_svg_escape(title)} | {metric_label}</text>',
        f'<text class="subtitle" x="{pad_left}" y="68">Blue line: frame metric | Red dashed line: final mean from metadata.json</text>',
    ]

    if mean_value == mean_value and not math.isinf(mean_value):
        mean_y = ty(mean_value)
        bottom_y = ty(y_min)
        if mean_y < bottom_y:
            shade_y = mean_y
            shade_h = bottom_y - mean_y
        else:
            shade_y = bottom_y
            shade_h = max(0.0, mean_y - bottom_y)
        if shade_h > 0:
            svg.append(
                f'<rect class="below-mean" x="{pad_left}" y="{shade_y:.2f}" width="{plot_width}" height="{shade_h:.2f}"/>'
            )
        svg.append(
            f'<line class="mean-line" x1="{pad_left}" y1="{mean_y:.2f}" x2="{pad_left + plot_width}" y2="{mean_y:.2f}"/>'
        )
        svg.append(
            f'<text class="mean-label" x="{pad_left + plot_width - 210}" y="{mean_y - 10:.2f}">Mean = {_fmt(mean_value)}</text>'
        )

    for i in range(6):
        y_value = y_min + (y_max - y_min) * i / 5.0
        y_pos = ty(y_value)
        svg.append(f'<line class="grid" x1="{pad_left}" y1="{y_pos:.2f}" x2="{pad_left + plot_width}" y2="{y_pos:.2f}"/>')
        svg.append(f'<line class="axis" x1="{pad_left - 8}" y1="{y_pos:.2f}" x2="{pad_left}" y2="{y_pos:.2f}"/>')
        svg.append(f'<text class="tick" x="{pad_left - 58}" y="{y_pos + 5:.2f}">{_fmt(y_value)}</text>')

    tick_step = _nice_step(len(xs), target_ticks=12)
    x_ticks = list(range(1, len(xs) + 1, tick_step))
    if x_ticks[-1] != len(xs):
        x_ticks.append(len(xs))
    for x_value in x_ticks:
        x_pos = tx(x_value)
        svg.append(f'<line class="grid" x1="{x_pos:.2f}" y1="{pad_top}" x2="{x_pos:.2f}" y2="{pad_top + plot_height}"/>')
        svg.append(f'<line class="axis" x1="{x_pos:.2f}" y1="{pad_top + plot_height}" x2="{x_pos:.2f}" y2="{pad_top + plot_height + 8}"/>')
        svg.append(f'<text class="tick" x="{x_pos - 10:.2f}" y="{pad_top + plot_height + 28}">{x_value}</text>')

    svg.append(f'<line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_height}"/>')
    svg.append(f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_height}" x2="{pad_left + plot_width}" y2="{pad_top + plot_height}"/>')
    svg.append(f'<text class="axis-label" x="{pad_left + plot_width / 2 - 50:.2f}" y="{height - 25}">Frame Count</text>')
    svg.append(f'<text class="axis-label" transform="translate(32 {pad_top + plot_height / 2:.2f}) rotate(-90)">{metric_label}</text>')

    polyline = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in valid_points)
    svg.append(f'<polyline class="line" points="{polyline}"/>')
    for x, y in valid_points:
        svg.append(f'<circle class="point" cx="{tx(x):.2f}" cy="{ty(y):.2f}" r="4.5"/>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def _write_svg_comparison_plot(
    out_path: Path,
    title: str,
    metric_label: str,
    y_min: float,
    y_max: float,
    series_a: dict,
    mean_a: float,
    label_a: str,
    series_b: dict,
    mean_b: float,
    label_b: str,
):
    width = 1700
    height = 760
    pad_left = 110
    pad_right = 50
    pad_top = 95
    pad_bottom = 90
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    colors = {
        "a_line": "#2563eb",
        "a_point": "#1d4ed8",
        "a_mean": "#1e40af",
        "b_line": "#f59e0b",
        "b_point": "#d97706",
        "b_mean": "#b45309",
    }

    valid_a = [(x, y) for x, y in zip(series_a["xs"], series_a["ys"]) if y == y and not math.isinf(y)]
    valid_b = [(x, y) for x, y in zip(series_b["xs"], series_b["ys"]) if y == y and not math.isinf(y)]
    if not valid_a and not valid_b:
        _write_empty_svg(out_path, "No valid points")
        return

    max_len = max(series_a["xs"][-1] if series_a["xs"] else 0, series_b["xs"][-1] if series_b["xs"] else 0)
    x0 = 1
    x1 = max(2, max_len)

    def tx(x: float) -> float:
        return pad_left + (x - x0) * plot_width / (x1 - x0)

    def ty(y: float) -> float:
        clamped = min(max(y, y_min), y_max)
        return pad_top + (y_max - clamped) * plot_height / (y_max - y_min)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>",
        _base_svg_style()
        + """
        .line-a { fill: none; stroke-width: 2.6; }
        .line-b { fill: none; stroke-width: 2.6; }
        .point-a { stroke: #ffffff; stroke-width: 1.4; }
        .point-b { stroke: #ffffff; stroke-width: 1.4; }
        .mean-a { stroke-width: 2.5; stroke-dasharray: 12 6; }
        .mean-b { stroke-width: 2.5; stroke-dasharray: 4 5; }
        .mean-label-a { font: 700 15px Arial, sans-serif; }
        .mean-label-b { font: 700 15px Arial, sans-serif; }
        """,
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{pad_left}" y="42">{_svg_escape(title)} | {metric_label}</text>',
        f'<text class="subtitle" x="{pad_left}" y="68">Overlay comparison: points, lines and mean lines use different colors for each experiment</text>',
    ]

    for i in range(6):
        y_value = y_min + (y_max - y_min) * i / 5.0
        y_pos = ty(y_value)
        svg.append(f'<line class="grid" x1="{pad_left}" y1="{y_pos:.2f}" x2="{pad_left + plot_width}" y2="{y_pos:.2f}"/>')
        svg.append(f'<line class="axis" x1="{pad_left - 8}" y1="{y_pos:.2f}" x2="{pad_left}" y2="{y_pos:.2f}"/>')
        svg.append(f'<text class="tick" x="{pad_left - 58}" y="{y_pos + 5:.2f}">{_fmt(y_value)}</text>')

    tick_step = _nice_step(x1, target_ticks=12)
    x_ticks = list(range(1, x1 + 1, tick_step))
    if x_ticks[-1] != x1:
        x_ticks.append(x1)
    for x_value in x_ticks:
        x_pos = tx(x_value)
        svg.append(f'<line class="grid" x1="{x_pos:.2f}" y1="{pad_top}" x2="{x_pos:.2f}" y2="{pad_top + plot_height}"/>')
        svg.append(f'<line class="axis" x1="{x_pos:.2f}" y1="{pad_top + plot_height}" x2="{x_pos:.2f}" y2="{pad_top + plot_height + 8}"/>')
        svg.append(f'<text class="tick" x="{x_pos - 10:.2f}" y="{pad_top + plot_height + 28}">{x_value}</text>')

    svg.append(f'<line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_height}"/>')
    svg.append(f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_height}" x2="{pad_left + plot_width}" y2="{pad_top + plot_height}"/>')
    svg.append(f'<text class="axis-label" x="{pad_left + plot_width / 2 - 50:.2f}" y="{height - 25}">Frame Count</text>')
    svg.append(f'<text class="axis-label" transform="translate(32 {pad_top + plot_height / 2:.2f}) rotate(-90)">{metric_label}</text>')

    legend_x = width - 330
    legend_y = 26
    svg.append(f'<rect class="legend-box" x="{legend_x}" y="{legend_y}" width="290" height="84" rx="8"/>')
    svg.append(f'<line x1="{legend_x + 18}" y1="{legend_y + 28}" x2="{legend_x + 60}" y2="{legend_y + 28}" stroke="{colors["a_line"]}" stroke-width="3"/>')
    svg.append(f'<circle cx="{legend_x + 39}" cy="{legend_y + 28}" r="5" fill="{colors["a_point"]}" stroke="#ffffff" stroke-width="1.2"/>')
    svg.append(f'<line x1="{legend_x + 18}" y1="{legend_y + 54}" x2="{legend_x + 60}" y2="{legend_y + 54}" stroke="{colors["a_mean"]}" stroke-width="3" stroke-dasharray="12 6"/>')
    svg.append(f'<text class="legend-text" x="{legend_x + 72}" y="{legend_y + 33}">{_svg_escape(label_a)} frames</text>')
    svg.append(f'<text class="legend-text" x="{legend_x + 72}" y="{legend_y + 59}">{_svg_escape(label_a)} mean</text>')
    svg.append(f'<line x1="{legend_x + 160}" y1="{legend_y + 28}" x2="{legend_x + 202}" y2="{legend_y + 28}" stroke="{colors["b_line"]}" stroke-width="3"/>')
    svg.append(f'<circle cx="{legend_x + 181}" cy="{legend_y + 28}" r="5" fill="{colors["b_point"]}" stroke="#ffffff" stroke-width="1.2"/>')
    svg.append(f'<line x1="{legend_x + 160}" y1="{legend_y + 54}" x2="{legend_x + 202}" y2="{legend_y + 54}" stroke="{colors["b_mean"]}" stroke-width="3" stroke-dasharray="4 5"/>')
    svg.append(f'<text class="legend-text" x="{legend_x + 214}" y="{legend_y + 33}">{_svg_escape(label_b)} frames</text>')
    svg.append(f'<text class="legend-text" x="{legend_x + 214}" y="{legend_y + 59}">{_svg_escape(label_b)} mean</text>')

    if mean_a == mean_a and not math.isinf(mean_a):
        mean_y_a = ty(mean_a)
        svg.append(f'<line class="mean-a" x1="{pad_left}" y1="{mean_y_a:.2f}" x2="{pad_left + plot_width}" y2="{mean_y_a:.2f}" stroke="{colors["a_mean"]}"/>')
        svg.append(f'<text class="mean-label-a" x="{pad_left + plot_width - 300}" y="{mean_y_a - 12:.2f}" fill="{colors["a_mean"]}">{_svg_escape(label_a)} mean = {_fmt(mean_a)}</text>')

    if mean_b == mean_b and not math.isinf(mean_b):
        mean_y_b = ty(mean_b)
        svg.append(f'<line class="mean-b" x1="{pad_left}" y1="{mean_y_b:.2f}" x2="{pad_left + plot_width}" y2="{mean_y_b:.2f}" stroke="{colors["b_mean"]}"/>')
        offset = 18 if mean_a == mean_a and abs(mean_y_b - ty(mean_a)) < 18 else -12
        svg.append(f'<text class="mean-label-b" x="{pad_left + plot_width - 300}" y="{mean_y_b + offset:.2f}" fill="{colors["b_mean"]}">{_svg_escape(label_b)} mean = {_fmt(mean_b)}</text>')

    if valid_a:
        poly_a = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in valid_a)
        svg.append(f'<polyline class="line-a" points="{poly_a}" stroke="{colors["a_line"]}"/>')
        for x, y in valid_a:
            svg.append(f'<circle class="point-a" cx="{tx(x):.2f}" cy="{ty(y):.2f}" r="4.4" fill="{colors["a_point"]}"/>')

    if valid_b:
        poly_b = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in valid_b)
        svg.append(f'<polyline class="line-b" points="{poly_b}" stroke="{colors["b_line"]}"/>')
        for x, y in valid_b:
            svg.append(f'<circle class="point-b" cx="{tx(x):.2f}" cy="{ty(y):.2f}" r="4.4" fill="{colors["b_point"]}"/>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def _plot_dataset(dataset_dir: Path, only: str):
    metrics_csv = dataset_dir / "keyframe_metrics.csv"
    metadata_json = dataset_dir / "metadata.json"
    rows = _read_rows(metrics_csv, only=only)
    if not rows:
        raise RuntimeError(f"No rows found after filtering in {metrics_csv}")

    metadata_metrics = _read_metadata_metrics(metadata_json)
    dataset_title = f"{dataset_dir.parent.name}/{dataset_dir.name}"

    for metric_name, spec in METRIC_SPECS.items():
        series = _series_payload(rows, metric_name)
        _write_svg_lineplot(
            out_path=dataset_dir / spec["filename"],
            title=dataset_title,
            metric_label=spec["label"],
            xs=series["xs"],
            ys=series["ys"],
            y_min=spec["y_min"],
            y_max=spec["y_max"],
            mean_value=metadata_metrics[metric_name],
        )

    summary_path = dataset_dir / "keyframe_plot_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"dataset={dataset_title}",
                f"filter={only}",
                f"frames={len(rows)}",
                f"mean_psnr={_fmt(metadata_metrics['psnr'])}",
                f"mean_ssim={_fmt(metadata_metrics['ssim'])}",
                f"mean_lpips={_fmt(metadata_metrics['lpips'])}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _plot_comparisons(
    roots_a: Iterable[str],
    roots_b: Iterable[str],
    output_root: str,
    only: str,
    label_a: str,
    label_b: str,
):
    dataset_map_a = _relative_dataset_map(roots_a)
    dataset_map_b = _relative_dataset_map(roots_b)
    shared_keys = sorted(set(dataset_map_a) & set(dataset_map_b))
    if not shared_keys:
        raise SystemExit("No matching datasets were found between the two experiment trees.")

    output_root_path = Path(output_root)
    print(f"Found {len(shared_keys)} matching datasets.")
    for rel_key in shared_keys:
        dataset_dir_a = dataset_map_a[rel_key]
        dataset_dir_b = dataset_map_b[rel_key]
        print(f"Comparing {rel_key} ...")
        rows_a = _read_rows(dataset_dir_a / "keyframe_metrics.csv", only=only)
        rows_b = _read_rows(dataset_dir_b / "keyframe_metrics.csv", only=only)
        if not rows_a or not rows_b:
            raise RuntimeError(f"Filtered rows are empty for dataset {rel_key}")

        metrics_a = _read_metadata_metrics(dataset_dir_a / "metadata.json")
        metrics_b = _read_metadata_metrics(dataset_dir_b / "metadata.json")

        path_parts = Path(rel_key).parts
        group_name = dataset_dir_a.parent.name
        dataset_name = path_parts[-1]
        title = f"{group_name}/{dataset_name}"
        out_dir = output_root_path / group_name / dataset_name

        for metric_name, spec in METRIC_SPECS.items():
            _write_svg_comparison_plot(
                out_path=out_dir / spec["filename"],
                title=title,
                metric_label=spec["label"],
                y_min=spec["y_min"],
                y_max=spec["y_max"],
                series_a=_series_payload(rows_a, metric_name),
                mean_a=metrics_a[metric_name],
                label_a=label_a,
                series_b=_series_payload(rows_b, metric_name),
                mean_b=metrics_b[metric_name],
                label_b=label_b,
            )

        (out_dir / "contrast_summary.txt").write_text(
            "\n".join(
                [
                    f"dataset={title}",
                    f"filter={only}",
                    f"{label_a}_frames={len(rows_a)}",
                    f"{label_b}_frames={len(rows_b)}",
                    f"{label_a}_mean_psnr={_fmt(metrics_a['psnr'])}",
                    f"{label_b}_mean_psnr={_fmt(metrics_b['psnr'])}",
                    f"{label_a}_mean_ssim={_fmt(metrics_a['ssim'])}",
                    f"{label_b}_mean_ssim={_fmt(metrics_b['ssim'])}",
                    f"{label_a}_mean_lpips={_fmt(metrics_a['lpips'])}",
                    f"{label_b}_mean_lpips={_fmt(metrics_b['lpips'])}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def main():
    args = parse_args()
    if args.mode == "single":
        dataset_dirs = _dataset_dirs(args.roots)
        if not dataset_dirs:
            raise SystemExit("No dataset directories with keyframe_metrics.csv and metadata.json were found.")
        print(f"Found {len(dataset_dirs)} dataset directories.")
        for dataset_dir in dataset_dirs:
            print(f"Plotting {dataset_dir} ...")
            _plot_dataset(dataset_dir, only=args.only)
    else:
        _plot_comparisons(
            roots_a=args.compare_roots_a,
            roots_b=args.compare_roots_b,
            output_root=args.compare_output_root,
            only=args.only,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
        )
    print("Done.")


if __name__ == "__main__":
    main()

