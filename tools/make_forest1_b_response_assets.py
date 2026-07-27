#!/usr/bin/env python3
"""Create reproducible local B-module ablation assets for forest1 frame 11."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter


METHODS = ("base", "r", "r_e", "r_d", "r_e_d")
METHOD_LABELS = {
    "base": "Base",
    "r": "R",
    "r_e": "R + E",
    "r_d": "R + D",
    "r_e_d": "R + E + D",
}
METHOD_COLORS = {
    "base": "#B7B7B7",
    "r": "#7598C8",
    "r_e": "#58A7A0",
    "r_d": "#78A96B",
    "r_e_d": "#D95F59",
}
METHOD_EDGES = {
    "base": "#747474",
    "r": "#456F9F",
    "r_e": "#2B7772",
    "r_d": "#4E7C42",
    "r_e_d": "#A83B38",
}
ROI_STYLES = (
    ("red", "#D62728"),
    ("blue", "#1976D2"),
)

PHOTOMETRIC_WEIGHT = 0.65
EDGE_WEIGHT = 0.35
GUIDE_MIN = 0.25
GUIDE_MAX = 4.0


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select two high-response regions and export the real B-module "
            "internal ablation density maps and local metrics."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--response-source",
        choices=METHODS,
        default="base",
        help=(
            "Variant whose render defines the pre-intervention response "
            "used for ROI selection."
        ),
    )
    parser.add_argument("--window-width", type=int, default=256)
    parser.add_argument("--window-height", type=int, default=144)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--minimum-support", type=float, default=0.65)
    parser.add_argument(
        "--minimum-center-distance",
        type=float,
        default=280.0,
        help=(
            "Non-maximum-suppression distance in pixels so the two ROIs "
            "represent spatially distinct response modes."
        ),
    )
    parser.add_argument("--density-sigma", type=float, default=1.8)
    return parser.parse_args()


def _disc_kernel(radius: int = 3) -> torch.Tensor:
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = (torch.sqrt(x**2 + y**2) <= radius + 0.5).float()
    return (kernel / kernel.sum())[None, None]


def compute_response_components(
    ground_truth: np.ndarray,
    render: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the final B response computation used by the model."""

    if ground_truth.shape != render.shape:
        raise ValueError(
            f"ground truth and render shapes differ: "
            f"{ground_truth.shape} vs {render.shape}"
        )
    if ground_truth.ndim != 3 or ground_truth.shape[2] != 3:
        raise ValueError("expected H x W x 3 RGB arrays")

    gt = torch.from_numpy(ground_truth.copy()).permute(2, 0, 1).float() / 255.0
    predicted = torch.from_numpy(render.copy()).permute(2, 0, 1).float() / 255.0
    height, width = ground_truth.shape[:2]

    # Match the level-zero keyframe image construction in SceneModel.
    gt_half = F.avg_pool2d(gt[None], kernel_size=2, stride=2)
    gt_level_zero = F.interpolate(
        gt_half,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )[0]
    residual = torch.abs(predicted - gt_level_zero).mean(dim=0)

    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )[None, None]
    laplacian = F.conv2d(
        residual[None, None],
        laplacian_kernel,
        padding="same",
    ).abs()
    laplacian[..., :, 0] = 0
    laplacian[..., :, -1] = 0
    laplacian[..., 0, :] = 0
    laplacian[..., -1, :] = 0
    edge = F.conv2d(
        laplacian,
        _disc_kernel(),
        padding="same",
    )[0, 0].clamp(0, 1)

    response = PHOTOMETRIC_WEIGHT * residual + EDGE_WEIGHT * edge
    guide = torch.clamp(
        response / torch.clamp(response.mean(), min=1e-12),
        min=GUIDE_MIN,
        max=GUIDE_MAX,
    )
    return (
        residual.numpy(),
        edge.numpy(),
        response.numpy(),
        guide.numpy(),
    )


def compute_response_guide(
    ground_truth: np.ndarray,
    render: np.ndarray,
) -> np.ndarray:
    return compute_response_components(ground_truth, render)[-1]


def boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def select_response_rois(
    guide: np.ndarray,
    support: np.ndarray,
    *,
    window_width: int,
    window_height: int,
    stride: int,
    margin: int,
    minimum_support: float,
    count: int,
    minimum_center_distance: float = 0.0,
) -> list[dict[str, object]]:
    """Select deterministic, non-overlapping windows by mean response."""

    if guide.shape != support.shape:
        raise ValueError("guide and support maps must have the same shape")
    height, width = guide.shape
    if window_width > width or window_height > height:
        raise ValueError("ROI window is larger than the image")
    if stride < 1:
        raise ValueError("stride must be positive")

    last_x = width - margin - window_width
    last_y = height - margin - window_height
    x_positions = list(range(margin, last_x + 1, stride))
    y_positions = list(range(margin, last_y + 1, stride))
    if x_positions and x_positions[-1] != last_x:
        x_positions.append(last_x)
    if y_positions and y_positions[-1] != last_y:
        y_positions.append(last_y)

    candidates: list[dict[str, object]] = []
    for y0 in y_positions:
        for x0 in x_positions:
            x1 = x0 + window_width
            y1 = y0 + window_height
            local_support = float(support[y0:y1, x0:x1].mean())
            if local_support < minimum_support:
                continue
            candidates.append(
                {
                    "box": (x0, y0, x1, y1),
                    "score": float(guide[y0:y1, x0:x1].mean()),
                    "support": local_support,
                }
            )

    candidates.sort(
        key=lambda record: (
            -float(record["score"]),
            record["box"][1],
            record["box"][0],
        )
    )
    selected: list[dict[str, object]] = []
    for candidate in candidates:
        if any(
            boxes_overlap(candidate["box"], existing["box"])
            for existing in selected
        ):
            continue
        candidate_x0, candidate_y0, candidate_x1, candidate_y1 = (
            candidate["box"]
        )
        candidate_center = (
            0.5 * (candidate_x0 + candidate_x1),
            0.5 * (candidate_y0 + candidate_y1),
        )
        if any(
            math.hypot(
                candidate_center[0]
                - 0.5 * (existing["box"][0] + existing["box"][2]),
                candidate_center[1]
                - 0.5 * (existing["box"][1] + existing["box"][3]),
            )
            < minimum_center_distance
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"found only {len(selected)} non-overlapping response regions"
        )
    return selected


def local_psnr(
    ground_truth: np.ndarray,
    render: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = box
    difference = (
        ground_truth[y0:y1, x0:x1].astype(np.float64)
        - render[y0:y1, x0:x1].astype(np.float64)
    )
    mse = float(np.mean(difference**2))
    return float(10.0 * np.log10((255.0**2) / max(mse, 1e-12)))


def local_gaussian_count(
    id_map: np.ndarray,
    box: tuple[int, int, int, int],
) -> int:
    x0, y0, x1, y1 = box
    identifiers = id_map[y0:y1, x0:x1]
    return int(np.unique(identifiers[identifiers >= 0]).size)


def footprint_density(
    id_map: np.ndarray,
    box: tuple[int, int, int, int],
    sigma: float,
) -> np.ndarray:
    """Spread unit mass over each dominant Gaussian's local footprint."""

    x0, y0, x1, y1 = box
    local_ids = id_map[y0:y1, x0:x1]
    valid = local_ids >= 0
    if not np.any(valid):
        return np.zeros(local_ids.shape, dtype=np.float64)

    identifiers = local_ids[valid]
    unique_ids, inverse = np.unique(identifiers, return_inverse=True)
    footprint_sizes = np.bincount(inverse)
    histogram = np.zeros(local_ids.shape, dtype=np.float64)
    histogram[valid] = 1.0 / footprint_sizes[inverse]
    density = gaussian_filter(histogram, sigma=sigma, mode="reflect")
    if density.sum() > 0:
        density *= unique_ids.size / density.sum()
    if not np.isclose(density.sum(), unique_ids.size, rtol=1e-6):
        raise RuntimeError("footprint density does not preserve Gaussian mass")
    return density


def _muted_background(crop: np.ndarray) -> np.ndarray:
    rgb = crop.astype(np.float64) / 255.0
    gray = np.sum(rgb * np.asarray([0.2126, 0.7152, 0.0722]), axis=2)
    muted = 0.32 * rgb + 0.68 * gray[..., None]
    return np.clip(0.78 * muted + 0.22, 0.0, 1.0)


def save_density_overlay(
    background: np.ndarray,
    density: np.ndarray,
    output_path: Path,
    norm: Normalize,
) -> None:
    height, width = density.shape
    figure = plt.figure(
        figsize=(width / 100.0, height / 100.0),
        dpi=300,
        frameon=False,
    )
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(_muted_background(background), interpolation="nearest")
    scaled = np.asarray(norm(density), dtype=float)
    rgba = plt.get_cmap("viridis")(scaled)
    rgba[..., 3] = np.where(
        density > 0,
        0.08 + 0.67 * np.power(np.clip(scaled, 0.0, 1.0), 0.58),
        0.0,
    )
    axis.imshow(rgba, interpolation="bilinear")
    levels = [
        float(norm.vmax) * fraction
        for fraction in (0.38, 0.70)
        if float(norm.vmax) * fraction < float(density.max())
    ]
    if levels:
        axis.contour(
            density,
            levels=levels,
            colors="white",
            linewidths=0.35,
            alpha=0.74,
        )
    axis.set_axis_off()
    figure.savefig(
        output_path,
        dpi=300,
        facecolor="white",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)


def _save_figure_formats(
    figure: plt.Figure,
    stem: Path,
) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(
            stem.with_suffix(suffix),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )


def save_bar_chart(
    values: np.ndarray,
    *,
    ylabel: str,
    output_stem: Path,
    metric: str,
) -> None:
    positions = np.arange(len(METHODS))
    colors = [METHOD_COLORS[method] for method in METHODS]
    edges = [METHOD_EDGES[method] for method in METHODS]
    figure, axis = plt.subplots(figsize=(5.25, 2.75), dpi=180)
    bars = axis.bar(
        positions,
        values,
        width=0.58,
        color=colors,
        edgecolor=edges,
        linewidth=0.9,
    )
    axis.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.65, alpha=0.9)
    axis.set_axisbelow(True)

    if metric == "count":
        axis.set_ylim(0, max(float(values.max()) * 1.24, 1.0))
        labels = [f"{int(value)}" for value in values]
    else:
        spread = max(float(values.max() - values.min()), 0.25)
        lower = math.floor((float(values.min()) - 0.24 * spread) * 2.0) / 2.0
        upper = math.ceil((float(values.max()) + 0.36 * spread) * 2.0) / 2.0
        if upper <= lower:
            upper = lower + 1.0
        axis.set_ylim(lower, upper)
        labels = [f"{value:.3f}" for value in values]

    y0, y1 = axis.get_ylim()
    offset = 0.025 * (y1 - y0)
    for method, bar, label in zip(METHODS, bars, labels):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold" if method == "r_e_d" else "normal",
            color="#A83B38" if method == "r_e_d" else "#242424",
        )
    figure.tight_layout(pad=0.55)
    _save_figure_formats(figure, output_stem)
    plt.close(figure)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def save_annotated_ground_truth(
    ground_truth: np.ndarray,
    rois: list[dict[str, object]],
    output_path: Path,
) -> None:
    canvas = Image.fromarray(ground_truth).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(5, round(canvas.width / 220))
    label_radius = max(17, round(canvas.width / 58))
    label_font = _font(max(19, round(canvas.width / 48)))
    for index, (roi, (_, color)) in enumerate(zip(rois, ROI_STYLES), start=1):
        x0, y0, x1, y1 = roi["box"]
        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline=color,
            width=line_width,
        )
        center = (
            x0 + label_radius + line_width,
            y0 + label_radius + line_width,
        )
        draw.ellipse(
            (
                center[0] - label_radius,
                center[1] - label_radius,
                center[0] + label_radius,
                center[1] + label_radius,
            ),
            fill="white",
            outline=color,
            width=max(3, line_width // 2),
        )
        text = str(index)
        bounds = draw.textbbox((0, 0), text, font=label_font)
        draw.text(
            (
                center[0] - (bounds[2] - bounds[0]) / 2,
                center[1] - (bounds[3] - bounds[1]) / 2 - 1,
            ),
            text,
            fill=color,
            font=label_font,
        )
    canvas.save(output_path, quality=98)


def save_response_audit(
    ground_truth: np.ndarray,
    guide: np.ndarray,
    rois: list[dict[str, object]],
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(ground_truth.shape[1] / 100.0, ground_truth.shape[0] / 100.0),
        dpi=180,
        frameon=False,
    )
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(ground_truth)
    overlay = axis.imshow(
        guide,
        cmap="inferno",
        vmin=GUIDE_MIN,
        vmax=GUIDE_MAX,
        alpha=0.55,
    )
    for roi, (_, color) in zip(rois, ROI_STYLES):
        x0, y0, x1, y1 = roi["box"]
        axis.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=color,
                linewidth=3.0,
            )
        )
    axis.set_axis_off()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)


def _percentile_rank(values: np.ndarray, score: float) -> float:
    return float(100.0 * np.mean(values <= score))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    ground_truth = arrays["ground_truth"]
    guides = []
    residuals = []
    edges = []
    responses = []
    support_maps = []
    for repeat in range(1, args.repeat + 1):
        prefix = f"{args.response_source}_repeat_{repeat}"
        residual, edge, response, guide = compute_response_components(
            ground_truth,
            arrays[f"{prefix}_render"],
        )
        residuals.append(residual)
        edges.append(edge)
        responses.append(response)
        guides.append(guide)
        support_maps.append(arrays[f"{prefix}_ids"] >= 0)

    mean_guide = np.mean(guides, axis=0)
    mean_support = np.mean(support_maps, axis=0)
    rois = select_response_rois(
        mean_guide,
        mean_support,
        window_width=args.window_width,
        window_height=args.window_height,
        stride=args.stride,
        margin=args.margin,
        minimum_support=args.minimum_support,
        count=2,
        minimum_center_distance=args.minimum_center_distance,
    )
    supported_values = mean_guide[mean_support >= args.minimum_support]
    for roi, (name, color) in zip(rois, ROI_STYLES):
        x0, y0, x1, y1 = roi["box"]
        roi["name"] = name
        roi["color"] = color
        roi["mean_residual"] = float(
            np.mean(residuals, axis=0)[y0:y1, x0:x1].mean()
        )
        roi["mean_edge"] = float(
            np.mean(edges, axis=0)[y0:y1, x0:x1].mean()
        )
        roi["mean_raw_response"] = float(
            np.mean(responses, axis=0)[y0:y1, x0:x1].mean()
        )
        roi["response_percentile"] = _percentile_rank(
            supported_values,
            float(roi["score"]),
        )

    save_annotated_ground_truth(
        ground_truth,
        rois,
        args.output_dir / "forest1_frame11_gt_two_response_rois.png",
    )
    save_response_audit(
        ground_truth,
        mean_guide,
        rois,
        args.output_dir / "forest1_frame11_response_selection_audit.png",
    )
    Image.fromarray(ground_truth).save(
        args.output_dir / "forest1_frame11_gt.png"
    )

    rows: list[dict[str, object]] = []
    density_maps: dict[tuple[str, str], np.ndarray] = {}
    for roi in rois:
        x0, y0, x1, y1 = roi["box"]
        crop = ground_truth[y0:y1, x0:x1]
        Image.fromarray(crop).save(
            args.output_dir / f"region_{roi['name']}_original_crop.png"
        )
        for method in METHODS:
            counts = []
            psnrs = []
            densities = []
            for repeat in range(1, args.repeat + 1):
                prefix = f"{method}_repeat_{repeat}"
                render = arrays[f"{prefix}_render"]
                id_map = arrays[f"{prefix}_ids"]
                counts.append(local_gaussian_count(id_map, roi["box"]))
                psnrs.append(local_psnr(ground_truth, render, roi["box"]))
                densities.append(
                    footprint_density(
                        id_map,
                        roi["box"],
                        sigma=args.density_sigma,
                    )
                )
            density_maps[(roi["name"], method)] = np.mean(densities, axis=0)
            rows.append(
                {
                    "region": roi["name"],
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "box_x0": x0,
                    "box_y0": y0,
                    "box_x1": x1,
                    "box_y1": y1,
                    "response_score": roi["score"],
                    "response_percentile": roi["response_percentile"],
                    "support": roi["support"],
                    "gaussian_count_mean_raw": float(np.mean(counts)),
                    "gaussian_count_mean_floor": int(
                        math.floor(float(np.mean(counts)))
                    ),
                    "local_psnr_mean": float(np.mean(psnrs)),
                    "gaussian_count_runs": json.dumps(counts),
                    "local_psnr_runs": json.dumps(psnrs),
                }
            )

    positive_density = np.concatenate(
        [density[density > 0] for density in density_maps.values()]
    )
    density_vmax = float(np.percentile(positive_density, 99.5))
    density_norm = Normalize(vmin=0.0, vmax=max(density_vmax, 1e-12))
    for roi in rois:
        x0, y0, x1, y1 = roi["box"]
        crop = ground_truth[y0:y1, x0:x1]
        for method in METHODS:
            save_density_overlay(
                crop,
                density_maps[(roi["name"], method)],
                args.output_dir
                / f"region_{roi['name']}_density_{method}.png",
                density_norm,
            )

    for roi in rois:
        region_rows = [
            next(
                row
                for row in rows
                if row["region"] == roi["name"]
                and row["method"] == method
            )
            for method in METHODS
        ]
        count_values = np.asarray(
            [row["gaussian_count_mean_floor"] for row in region_rows],
            dtype=float,
        )
        psnr_values = np.asarray(
            [row["local_psnr_mean"] for row in region_rows],
            dtype=float,
        )
        save_bar_chart(
            count_values,
            ylabel="Number of Gaussians",
            output_stem=(
                args.output_dir
                / f"region_{roi['name']}_gaussian_numbers"
            ),
            metric="count",
        )
        save_bar_chart(
            psnr_values,
            ylabel="Local PSNR (dB)",
            output_stem=(
                args.output_dir / f"region_{roi['name']}_local_psnr"
            ),
            metric="psnr",
        )

    base_by_region = {
        region: next(
            row
            for row in rows
            if row["region"] == region and row["method"] == "base"
        )
        for region, _ in ROI_STYLES
    }
    for row in rows:
        base = base_by_region[row["region"]]
        row["delta_gaussian_numbers"] = (
            int(row["gaussian_count_mean_floor"])
            - int(base["gaussian_count_mean_floor"])
        )
        row["delta_local_psnr"] = (
            float(row["local_psnr_mean"])
            - float(base["local_psnr_mean"])
        )

    csv_path = args.output_dir / "forest1_frame11_local_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "source_npz": str(args.input.resolve()),
        "response_definition": {
            "selection_source_method": args.response_source,
            "photometric_weight": PHOTOMETRIC_WEIGHT,
            "edge_weight": EDGE_WEIGHT,
            "guide_clip": [GUIDE_MIN, GUIDE_MAX],
            "coverage_role": (
                "D is the scalar coverage-deficit gate used by B; it does "
                "not alter the pixel-wise ordering of the R+E guide."
            ),
        },
        "selection": {
            "repeat_count": args.repeat,
            "window_width": args.window_width,
            "window_height": args.window_height,
            "stride": args.stride,
            "margin": args.margin,
            "minimum_support": args.minimum_support,
            "minimum_center_distance": args.minimum_center_distance,
            "regions": rois,
        },
        "density": {
            "definition": (
                "Each unique dominant visible Gaussian contributes unit "
                "mass distributed over its local screen-space footprint."
            ),
            "sigma": args.density_sigma,
            "shared_vmin": 0.0,
            "shared_vmax_percentile_99_5": density_vmax,
            "repeat_aggregation": "pixel-wise arithmetic mean",
        },
        "bar_statistics": {
            "repeat_aggregation": "arithmetic mean over complete runs",
            "error_bars": False,
            "displayed_gaussian_count": "floor of the run mean",
            "displayed_psnr": "raw run mean",
        },
    }
    with (args.output_dir / "forest1_frame11_audit.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(audit, output, indent=2)
    np.savez_compressed(
        args.output_dir / "forest1_frame11_analysis_arrays.npz",
        mean_response_guide=mean_guide,
        mean_render_support=mean_support,
        **{
            f"density_{region}_{method}": density
            for (region, method), density in density_maps.items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
