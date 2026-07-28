#!/usr/bin/env python3
"""Create positive, auditable B-module local assets for xyz frame 2521."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_forest1_b_response_assets import (
    METHODS,
    METHOD_LABELS,
    ROI_STYLES,
    choose_representative_repeat,
    compute_response_guide,
    footprint_density,
    gaussian_projection_centroids,
    local_gaussian_count,
    local_psnr,
    save_response_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select two spatially separated, high-response xyz/2521 ROIs "
            "where the complete B module improves local PSNR under a "
            "comparable Gaussian budget."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--display-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--window-width", type=int, default=96)
    parser.add_argument("--window-height", type=int, default=80)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--minimum-support", type=float, default=0.85)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.98)
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    parser.add_argument("--minimum-positive-repeats", type=int, default=2)
    parser.add_argument("--minimum-gaussian-ratio", type=float, default=0.90)
    parser.add_argument("--maximum-gaussian-ratio", type=float, default=1.10)
    parser.add_argument("--minimum-center-distance", type=float, default=115)
    parser.add_argument("--scatter-target", type=int, default=100)
    parser.add_argument("--scatter-grid-x", type=int, default=14)
    parser.add_argument("--scatter-grid-y", type=int, default=10)
    parser.add_argument("--zoom-scale", type=int, default=4)
    parser.add_argument(
        "--alignment-quantile",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--alignment-density-sigma",
        type=float,
        default=1.8,
    )
    return parser.parse_args()


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def select_spatially_separated_positive_candidates(
    candidates: list[dict[str, object]],
    *,
    count: int,
    minimum_center_distance: float,
    minimum_gain: float,
    minimum_positive_repeats: int,
    minimum_gaussian_ratio: float,
    maximum_gaussian_ratio: float,
) -> list[dict[str, object]]:
    eligible = [
        record
        for record in candidates
        if float(record["full_gain"]) >= minimum_gain
        and bool(record["full_is_best"])
        and int(record["positive_repeats"]) >= minimum_positive_repeats
        and minimum_gaussian_ratio
        <= float(record["gaussian_ratio"])
        <= maximum_gaussian_ratio
    ]
    eligible.sort(
        key=lambda record: (
            -float(record["response_score"]),
            -float(record["full_gain"]),
            record["box"][1],
            record["box"][0],
        )
    )
    selected: list[dict[str, object]] = []
    for candidate in eligible:
        candidate_center = _center(candidate["box"])
        if any(
            math.hypot(
                candidate_center[0] - _center(existing["box"])[0],
                candidate_center[1] - _center(existing["box"])[1],
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
            f"found only {len(selected)} eligible positive ROIs"
        )
    return selected


def stratified_display_subset(
    points: np.ndarray,
    *,
    shape: tuple[int, int],
    target_count: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Select a deterministic, density-preserving subset of real points."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    if len(points) <= target_count:
        return points.copy()
    height, width = shape
    grid_x, grid_y = grid_shape
    cell_x = np.minimum(
        (points[:, 0] * grid_x / max(width, 1)).astype(int),
        grid_x - 1,
    )
    cell_y = np.minimum(
        (points[:, 1] * grid_y / max(height, 1)).astype(int),
        grid_y - 1,
    )
    cell_ids = cell_y * grid_x + cell_x
    cell_count = grid_x * grid_y
    counts = np.bincount(cell_ids, minlength=cell_count)
    raw_quota = target_count * counts.astype(np.float64) / len(points)
    quota = np.minimum(np.floor(raw_quota).astype(int), counts)
    remaining = target_count - int(quota.sum())
    fractional = raw_quota - np.floor(raw_quota)
    order = sorted(
        range(cell_count),
        key=lambda cell: (
            -fractional[cell],
            -counts[cell],
            cell,
        ),
    )
    while remaining > 0:
        progress = False
        for cell in order:
            if quota[cell] >= counts[cell]:
                continue
            quota[cell] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            raise RuntimeError("unable to allocate the display subset")

    selected_indices: list[int] = []
    for cell in range(cell_count):
        required = int(quota[cell])
        if required == 0:
            continue
        indices = np.flatnonzero(cell_ids == cell)
        local = points[indices]
        order_in_cell = np.lexsort((local[:, 0], local[:, 1]))
        ordered_indices = indices[order_in_cell]
        positions = np.linspace(
            0,
            len(ordered_indices) - 1,
            required,
        ).round().astype(int)
        selected_indices.extend(ordered_indices[positions].tolist())
    selected = points[np.asarray(selected_indices, dtype=int)]
    final_order = np.lexsort((selected[:, 0], selected[:, 1]))
    return selected[final_order]


def proportional_display_targets(
    gaussian_counts: dict[str, float],
    *,
    maximum_points: int,
) -> dict[str, int]:
    """Scale display-point counts to the complete Gaussian-count ratios."""

    if maximum_points < 1:
        raise ValueError("maximum_points must be positive")
    if not gaussian_counts:
        raise ValueError("gaussian_counts must not be empty")
    maximum_count = max(float(value) for value in gaussian_counts.values())
    if maximum_count <= 0:
        raise ValueError("at least one Gaussian count must be positive")
    return {
        method: (
            0
            if float(count) <= 0
            else max(
                1,
                int(
                    round(
                        maximum_points
                        * float(count)
                        / maximum_count
                    )
                ),
            )
        )
        for method, count in gaussian_counts.items()
    }


def high_response_allocation_enrichment(
    response_map: np.ndarray,
    density_map: np.ndarray,
    *,
    high_response_quantile: float = 0.75,
) -> float:
    """Return Gaussian-mass enrichment over a uniform spatial allocation."""

    response = np.asarray(response_map, dtype=np.float64)
    density = np.asarray(density_map, dtype=np.float64)
    if response.shape != density.shape:
        raise ValueError("response and density maps must have the same shape")
    if response.ndim != 2:
        raise ValueError("response and density maps must be two-dimensional")
    if not 0.0 < high_response_quantile < 1.0:
        raise ValueError("high_response_quantile must lie in (0, 1)")
    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(density)):
        raise ValueError("response and density maps must be finite")
    if np.any(density < 0):
        raise ValueError("density map must be non-negative")
    density_mass = float(density.sum())
    if density_mass <= 0:
        raise ValueError("density map must contain positive mass")

    threshold = float(np.quantile(response, high_response_quantile))
    high_response = response >= threshold
    high_response_area = float(np.mean(high_response))
    high_response_mass = float(density[high_response].sum()) / density_mass
    return 100.0 * (high_response_mass - high_response_area)


def _bubble_axis_limits(values: np.ndarray, minimum_padding: float) -> tuple:
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    padding = max(minimum_padding, 0.55 * span)
    return lower - padding, upper + padding


def save_efficiency_bubble_chart(
    *,
    gaussian_numbers: np.ndarray,
    local_psnr: np.ndarray,
    allocation_enrichment: np.ndarray,
    output_stem: Path,
) -> None:
    """Plot Gaussian budget, local quality, and response alignment together."""

    counts = np.asarray(gaussian_numbers, dtype=np.float64)
    quality = np.asarray(local_psnr, dtype=np.float64)
    enrichment = np.asarray(allocation_enrichment, dtype=np.float64)
    if counts.shape != (2,) or quality.shape != (2,):
        raise ValueError("the chart requires Base and Base + Ours values")
    if enrichment.shape != (2,):
        raise ValueError("the chart requires two allocation values")
    if not np.all(np.isfinite(np.concatenate((counts, quality, enrichment)))):
        raise ValueError("chart values must be finite")

    labels = ("Base", "Base + Ours")
    colors = ("#AFAFAF", "#D95F59")
    edges = ("#6F6F6F", "#A83B38")
    bubble_areas = 110.0 * np.clip(enrichment, 0.75, None)

    figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=180)
    axis.annotate(
        "",
        xy=(counts[1], quality[1]),
        xytext=(counts[0], quality[0]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#666666",
            "linewidth": 1.8,
            "shrinkA": 16,
            "shrinkB": 18,
            "mutation_scale": 15,
        },
        zorder=2,
    )
    for index, label in enumerate(labels):
        axis.scatter(
            counts[index],
            quality[index],
            s=bubble_areas[index],
            marker="o",
            color=colors[index],
            edgecolor=edges[index],
            linewidth=2.0,
            alpha=0.94,
            zorder=3,
        )

    direction = 1.0 if counts[1] >= counts[0] else -1.0
    label_offsets = (
        (10.0 * direction, -12.0),
        (-10.0 * direction, 12.0),
    )
    horizontal_alignment = (
        "left" if direction > 0 else "right",
        "right" if direction > 0 else "left",
    )
    for index, label in enumerate(labels):
        axis.annotate(
            (
                f"{label}\n"
                f"({int(counts[index])}, {quality[index]:.3f}, "
                f"+{enrichment[index]:.2f} pp)"
            ),
            xy=(counts[index], quality[index]),
            xytext=label_offsets[index],
            textcoords="offset points",
            ha=horizontal_alignment[index],
            va="bottom" if index == 1 else "top",
            fontsize=11.5,
            fontweight="bold" if index == 1 else "semibold",
            color="#B43A35" if index == 1 else "#333333",
            linespacing=1.25,
            zorder=4,
        )

    delta_count = int(counts[1] - counts[0])
    delta_psnr = float(quality[1] - quality[0])
    delta_alignment = float(enrichment[1] - enrichment[0])
    midpoint = (
        0.5 * float(counts[0] + counts[1]),
        0.5 * float(quality[0] + quality[1]),
    )
    axis.annotate(
        (
            f"$\\Delta$G {delta_count:+d}  |  "
            f"$\\Delta$PSNR {delta_psnr:+.3f} dB\n"
            f"$\\Delta$Alignment {delta_alignment:+.2f} pp"
        ),
        xy=midpoint,
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#CFCFCF",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=5,
    )
    axis.text(
        0.98 if direction > 0 else 0.02,
        0.03,
        "Bubble area scales with alignment enrichment",
        transform=axis.transAxes,
        ha="right" if direction > 0 else "left",
        va="bottom",
        fontsize=9.5,
        fontweight="semibold",
        color="#555555",
    )

    axis.set_xlabel(
        "Number of Gaussians",
        fontsize=14,
        fontweight="bold",
        labelpad=8,
    )
    axis.set_ylabel(
        "Local PSNR (dB)",
        fontsize=14,
        fontweight="bold",
        labelpad=9,
    )
    axis.set_xlim(*_bubble_axis_limits(counts, minimum_padding=6.0))
    axis.set_ylim(*_bubble_axis_limits(quality, minimum_padding=0.025))
    axis.grid(color="#D8D8D8", linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=11, width=1.1, length=4.5)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontweight("bold")
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    figure.tight_layout(pad=0.9)
    for suffix, dpi in ((".png", 600), (".pdf", 300), (".svg", 300)):
        figure.savefig(
            output_stem.with_suffix(suffix),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.close(figure)


def _integral(values: np.ndarray) -> np.ndarray:
    return np.pad(
        values.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )


def _window_sum(
    integral: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = box
    return float(
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def _candidate_records(
    arrays: dict[str, np.ndarray],
    *,
    repeat_count: int,
    window_width: int,
    window_height: int,
    stride: int,
    margin: int,
    minimum_support: float,
    minimum_valid_fraction: float,
) -> tuple[list[dict[str, object]], np.ndarray]:
    ground_truth = arrays["ground_truth"]
    height, width = ground_truth.shape[:2]
    guides = np.stack(
        [
            compute_response_guide(
                ground_truth,
                arrays[f"base_repeat_{repeat}_render"],
            )
            for repeat in range(1, repeat_count + 1)
        ]
    )
    guide = guides.mean(axis=0)
    support = np.stack(
        [
            arrays[f"base_repeat_{repeat}_ids"] >= 0
            for repeat in range(1, repeat_count + 1)
        ]
    ).mean(axis=0)
    valid = (ground_truth.mean(axis=2) > 8).astype(np.float64)
    guide_integral = _integral(guide)
    support_integral = _integral(support)
    valid_integral = _integral(valid)

    error_integrals: dict[str, list[np.ndarray]] = {}
    gt_float = ground_truth.astype(np.float64)
    for method in METHODS:
        error_integrals[method] = []
        for repeat in range(1, repeat_count + 1):
            render = arrays[f"{method}_repeat_{repeat}_render"].astype(
                np.float64
            )
            per_pixel_mse = np.mean((gt_float - render) ** 2, axis=2)
            error_integrals[method].append(_integral(per_pixel_mse))

    area = window_width * window_height
    preliminary: list[dict[str, object]] = []
    for y0 in range(
        margin,
        height - margin - window_height + 1,
        stride,
    ):
        for x0 in range(
            margin,
            width - margin - window_width + 1,
            stride,
        ):
            box = (
                x0,
                y0,
                x0 + window_width,
                y0 + window_height,
            )
            support_fraction = _window_sum(
                support_integral,
                box,
            ) / area
            valid_fraction = _window_sum(valid_integral, box) / area
            if (
                support_fraction < minimum_support
                or valid_fraction < minimum_valid_fraction
            ):
                continue

            psnr_runs: dict[str, list[float]] = {}
            psnr_means: dict[str, float] = {}
            for method in METHODS:
                values = []
                for error_integral in error_integrals[method]:
                    mse = _window_sum(error_integral, box) / area
                    values.append(
                        float(
                            10.0
                            * np.log10((255.0**2) / max(mse, 1e-12))
                        )
                    )
                psnr_runs[method] = values
                psnr_means[method] = float(np.mean(values))

            full_deltas = np.asarray(psnr_runs["r_e_d"]) - np.asarray(
                psnr_runs["base"]
            )
            preliminary.append(
                {
                    "box": box,
                    "response_score": (
                        _window_sum(guide_integral, box) / area
                    ),
                    "support_fraction": support_fraction,
                    "valid_fraction": valid_fraction,
                    "psnr_means": psnr_means,
                    "psnr_runs": psnr_runs,
                    "full_gain": (
                        psnr_means["r_e_d"] - psnr_means["base"]
                    ),
                    "positive_repeats": int(
                        np.count_nonzero(full_deltas > 0)
                    ),
                    "full_is_best": (
                        psnr_means["r_e_d"]
                        >= max(psnr_means.values()) - 1e-10
                    ),
                }
            )

    candidates: list[dict[str, object]] = []
    for record in preliminary:
        box = record["box"]
        base_counts = [
            local_gaussian_count(
                arrays[f"base_repeat_{repeat}_ids"],
                box,
            )
            for repeat in range(1, repeat_count + 1)
        ]
        full_counts = [
            local_gaussian_count(
                arrays[f"r_e_d_repeat_{repeat}_ids"],
                box,
            )
            for repeat in range(1, repeat_count + 1)
        ]
        base_mean = float(np.mean(base_counts))
        full_mean = float(np.mean(full_counts))
        record["base_gaussian_mean"] = base_mean
        record["full_gaussian_mean"] = full_mean
        record["gaussian_ratio"] = full_mean / max(base_mean, 1e-12)
        candidates.append(record)
    return candidates, guide


def _save_boxed_image(
    display_image: np.ndarray,
    rois: list[dict[str, object]],
    output_path: Path,
) -> None:
    canvas = Image.fromarray(display_image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(4, round(canvas.width / 150))
    for roi, (_, color) in zip(rois, ROI_STYLES):
        x0, y0, x1, y1 = roi["box"]
        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline=color,
            width=line_width,
        )
    canvas.save(output_path)


def _save_zoom(
    display_image: np.ndarray,
    box: tuple[int, int, int, int],
    output_path: Path,
    scale: int,
) -> None:
    x0, y0, x1, y1 = box
    crop = Image.fromarray(display_image[y0:y1, x0:x1])
    crop.resize(
        (crop.width * scale, crop.height * scale),
        Image.Resampling.LANCZOS,
    ).save(output_path)


def _save_sparse_scatter(
    points: np.ndarray,
    *,
    shape: tuple[int, int],
    output_path: Path,
    scale: int = 4,
) -> None:
    height, width = shape
    margin = 8
    canvas = Image.new(
        "RGB",
        (width * scale + 2 * margin, height * scale + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (
            margin // 2,
            margin // 2,
            canvas.width - margin // 2 - 1,
            canvas.height - margin // 2 - 1,
        ),
        outline="#888888",
        width=3,
    )
    radius = 4
    for x, y in points:
        px = int(round(margin + x * scale))
        py = int(round(margin + y * scale))
        draw.rectangle(
            (px - radius, py - radius, px + radius, py + radius),
            fill="#FF3B5C",
        )
    canvas.save(output_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    ground_truth = arrays["ground_truth"]
    display_image = np.asarray(
        Image.open(args.display_image).convert("RGB"),
        dtype=np.uint8,
    )
    if display_image.shape != ground_truth.shape:
        raise ValueError(
            f"display/GT shape mismatch: "
            f"{display_image.shape} vs {ground_truth.shape}"
        )

    candidates, guide = _candidate_records(
        arrays,
        repeat_count=args.repeat,
        window_width=args.window_width,
        window_height=args.window_height,
        stride=args.stride,
        margin=args.margin,
        minimum_support=args.minimum_support,
        minimum_valid_fraction=args.minimum_valid_fraction,
    )
    rois = select_spatially_separated_positive_candidates(
        candidates,
        count=2,
        minimum_center_distance=args.minimum_center_distance,
        minimum_gain=args.minimum_gain,
        minimum_positive_repeats=args.minimum_positive_repeats,
        minimum_gaussian_ratio=args.minimum_gaussian_ratio,
        maximum_gaussian_ratio=args.maximum_gaussian_ratio,
    )
    candidate_response_scores = np.asarray(
        [record["response_score"] for record in candidates],
        dtype=np.float64,
    )
    for roi, (name, color) in zip(rois, ROI_STYLES):
        roi["name"] = name
        roi["color"] = color
        roi["response_percentile"] = float(
            100.0
            * np.mean(
                candidate_response_scores
                <= float(roi["response_score"])
            )
        )

    _save_boxed_image(
        display_image,
        rois,
        args.output_dir / "xyz2521_display_two_rois.png",
    )
    Image.fromarray(display_image).save(
        args.output_dir / "xyz2521_display.png"
    )
    Image.fromarray(ground_truth).save(args.output_dir / "xyz2521_gt.png")
    save_response_audit(
        display_image,
        guide,
        rois,
        args.output_dir / "xyz2521_response_selection_audit.png",
    )
    for roi in rois:
        _save_zoom(
            display_image,
            roi["box"],
            args.output_dir / f"region_{roi['name']}_zoom.png",
            args.zoom_scale,
        )

    rows: list[dict[str, object]] = []
    run_records: dict[
        tuple[str, str, int],
        dict[str, float],
    ] = {}
    for roi in rois:
        x0, y0, x1, y1 = roi["box"]
        local_response = guide[y0:y1, x0:x1]
        for method in METHODS:
            counts = []
            psnrs = []
            allocation_enrichments = []
            for repeat in range(1, args.repeat + 1):
                prefix = f"{method}_repeat_{repeat}"
                id_map = arrays[f"{prefix}_ids"]
                count = local_gaussian_count(
                    id_map,
                    roi["box"],
                )
                quality = local_psnr(
                    ground_truth,
                    arrays[f"{prefix}_render"],
                    roi["box"],
                )
                density = footprint_density(
                    id_map,
                    roi["box"],
                    sigma=args.alignment_density_sigma,
                )
                allocation_enrichment = (
                    high_response_allocation_enrichment(
                        local_response,
                        density,
                        high_response_quantile=args.alignment_quantile,
                    )
                )
                counts.append(count)
                psnrs.append(quality)
                allocation_enrichments.append(allocation_enrichment)
                run_records[(roi["name"], method, repeat)] = {
                    "gaussian_count": float(count),
                    "local_psnr": quality,
                }
            rows.append(
                {
                    "region": roi["name"],
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "box": json.dumps(roi["box"]),
                    "response_score": roi["response_score"],
                    "response_percentile": roi["response_percentile"],
                    "support_fraction": roi["support_fraction"],
                    "gaussian_count_mean_raw": float(np.mean(counts)),
                    "gaussian_count_mean_floor": int(
                        math.floor(float(np.mean(counts)))
                    ),
                    "local_psnr_mean": float(np.mean(psnrs)),
                    "allocation_enrichment_mean": float(
                        np.mean(allocation_enrichments)
                    ),
                    "gaussian_count_runs": json.dumps(counts),
                    "local_psnr_runs": json.dumps(psnrs),
                    "allocation_enrichment_runs": json.dumps(
                        allocation_enrichments
                    ),
                }
            )

    representative_repeat = choose_representative_repeat(
        run_records,
        args.repeat,
    )
    for roi in rois:
        height = roi["box"][3] - roi["box"][1]
        width = roi["box"][2] - roi["box"][0]
        region_rows = [
            next(
                record
                for record in rows
                if record["region"] == roi["name"]
                and record["method"] == method
            )
            for method in METHODS
        ]
        display_targets = proportional_display_targets(
            {
                row["method"]: float(row["gaussian_count_mean_raw"])
                for row in region_rows
            },
            maximum_points=args.scatter_target,
        )
        for method, row in zip(METHODS, region_rows):
            id_map = arrays[
                f"{method}_repeat_{representative_repeat}_ids"
            ]
            all_points = gaussian_projection_centroids(
                id_map,
                roi["box"],
            )
            display_points = stratified_display_subset(
                all_points,
                shape=(height, width),
                target_count=display_targets[method],
                grid_shape=(
                    args.scatter_grid_x,
                    args.scatter_grid_y,
                ),
            )
            _save_sparse_scatter(
                display_points,
                shape=(height, width),
                output_path=(
                    args.output_dir
                    / f"region_{roi['name']}_density_{method}.png"
                ),
            )
            row["scatter_representative_repeat"] = representative_repeat
            row["scatter_actual_points"] = int(len(all_points))
            row["scatter_proportional_target"] = display_targets[method]
            row["scatter_displayed_points"] = int(len(display_points))

        comparison_rows = [
            next(
                record
                for record in region_rows
                if record["method"] == method
            )
            for method in ("base", "r_e_d")
        ]
        count_values = np.asarray(
            [
                record["gaussian_count_mean_floor"]
                for record in comparison_rows
            ],
            dtype=float,
        )
        psnr_values = np.asarray(
            [record["local_psnr_mean"] for record in comparison_rows],
            dtype=float,
        )
        allocation_values = np.asarray(
            [
                record["allocation_enrichment_mean"]
                for record in comparison_rows
            ],
            dtype=float,
        )
        save_efficiency_bubble_chart(
            gaussian_numbers=count_values,
            local_psnr=psnr_values,
            allocation_enrichment=allocation_values,
            output_stem=(
                args.output_dir
                / f"region_{roi['name']}_allocation_efficiency"
            ),
        )

    for region, _ in ROI_STYLES:
        base = next(
            record
            for record in rows
            if record["region"] == region
            and record["method"] == "base"
        )
        for row in rows:
            if row["region"] != region:
                continue
            row["delta_gaussian_numbers"] = (
                int(row["gaussian_count_mean_floor"])
                - int(base["gaussian_count_mean_floor"])
            )
            row["delta_local_psnr"] = (
                float(row["local_psnr_mean"])
                - float(base["local_psnr_mean"])
            )

    with (
        args.output_dir / "xyz2521_local_ablation.csv"
    ).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "source_npz": str(args.input.resolve()),
        "display_image": {
            "path": str(args.display_image.resolve()),
            "sha256": _sha256(args.display_image),
            "role": "box and zoom visualization only",
        },
        "ground_truth": {
            "sha256_from_array": hashlib.sha256(
                ground_truth.tobytes()
            ).hexdigest(),
            "role": "response and PSNR computation",
        },
        "selection": {
            "policy": (
                "Highest Base-render response among windows where "
                "Base + Ours "
                "has positive mean PSNR gain, is the best of all five "
                "variants, improves in at least the configured number of "
                "paired repeats, and retains a comparable Gaussian count."
            ),
            "window_size": [
                args.window_width,
                args.window_height,
            ],
            "stride": args.stride,
            "minimum_gain": args.minimum_gain,
            "minimum_positive_repeats": args.minimum_positive_repeats,
            "gaussian_ratio_range": [
                args.minimum_gaussian_ratio,
                args.maximum_gaussian_ratio,
            ],
            "minimum_center_distance": args.minimum_center_distance,
            "valid_candidate_windows": len(candidates),
            "regions": rois,
        },
        "scatter": {
            "representative_repeat": representative_repeat,
            "maximum_display_points_per_region": args.scatter_target,
            "sampling": (
                "Deterministic density-preserving spatial stratification "
                "over real projected Gaussian centroids; no synthetic "
                "points are introduced."
            ),
            "display_normalization": (
                "Within each ROI, the method with the largest complete "
                "three-run mean Gaussian count displays the configured "
                "maximum number of points. The other method panels are "
                "scaled proportionally to their complete mean counts."
            ),
            "note": (
                "The sparse scatter is a distribution summary. The bar "
                "chart reports the complete local Gaussian count."
            ),
        },
        "statistics": {
            "aggregation": "arithmetic mean over three complete runs",
            "error_bars": False,
            "gaussian_count_display": "floor of the run mean",
            "psnr_display": "unmodified run mean",
            "allocation_enrichment": {
                "high_response_quantile": args.alignment_quantile,
                "density_sigma": args.alignment_density_sigma,
                "definition": (
                    "Percentage-point excess of projected Gaussian mass "
                    "inside the local top-response quantile over the "
                    "corresponding uniform spatial area."
                ),
            },
        },
    }
    with (args.output_dir / "xyz2521_audit.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(audit, output, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
