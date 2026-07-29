#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


METHODS = ("base", "r_e_d")
METHOD_LABELS = ("Base", "Ours")
METHOD_COLORS = ("#777777", "#C83E3E")
RED_COLOR = "#D62728"
BLUE_COLOR = "#1F77B4"
ORANGE_COLOR = "#E69F00"
PURPLE_COLOR = "#9467BD"
OURS_SAMPLING_COLOR = "#66A866"
BUBBLE_BASE_AREA = 1700.0
BUBBLE_CONTRAST_EXPONENT = 4.0
SAMPLING_SHARE_BUBBLE_MAX_AREA = 760.0
SAMPLING_SHARE_BUBBLE_CONTRAST_EXPONENT = 2.5
ROI_ARROW_CURVATURES = {
    "red": -2.00,
    "blue": 0.04,
    "orange": -2.45,
    "purple": 0.42,
}
ROI_BOX_HALO_LINEWIDTH = 6.2
ROI_BOX_COLOR_LINEWIDTH = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create formula-aligned Base/Ours local Gaussian allocation "
            "evidence from a traced B-module frame."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--window-width", type=int, default=96)
    parser.add_argument("--window-height", type=int, default=80)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.97)
    parser.add_argument("--minimum-psnr-gain", type=float, default=0.05)
    parser.add_argument("--minimum-positive-repeats", type=int, default=2)
    parser.add_argument("--minimum-mu-change", type=float, default=3.0)
    parser.add_argument(
        "--minimum-gaussian-reduction",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--minimum-lower-gaussian-repeats",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--minimum-center-distance",
        type=float,
        default=180.0,
    )
    return parser.parse_args()


def roi_expected_samples(
    probability: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = box
    return float(
        np.asarray(probability, dtype=np.float64)[y0:y1, x0:x1].sum()
    )


def roi_sampling_share(
    probability: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    values = np.asarray(probability, dtype=np.float64)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("frame sampling mass must be positive")
    return 100.0 * roi_expected_samples(values, box) / total


def normalized_probability_change(
    arrays: dict[str, np.ndarray],
    *,
    repeat: int,
) -> np.ndarray:
    changes = []
    for repetition in range(1, repeat + 1):
        base = np.asarray(
            arrays[f"base_repeat_{repetition}_final_probability"],
            dtype=np.float64,
        )
        ours = np.asarray(
            arrays[f"r_e_d_repeat_{repetition}_final_probability"],
            dtype=np.float64,
        )
        if base.shape != ours.shape or base.ndim != 2:
            raise ValueError("Base and Ours probability maps must align")
        base_total = float(base.sum())
        ours_total = float(ours.sum())
        if base_total <= 0 or ours_total <= 0:
            raise ValueError("probability maps must have positive mass")
        changes.append(ours / ours_total - base / base_total)
    if not changes:
        raise ValueError("at least one repetition is required")
    return np.mean(np.stack(changes, axis=0), axis=0)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _center_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_center = _center(first)
    second_center = _center(second)
    return math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )


def choose_redistribution_rois(
    candidates: Iterable[dict[str, object]],
    *,
    minimum_center_distance: float,
    minimum_psnr_gain: float,
    minimum_positive_repeats: int,
    minimum_mu_change: float,
    minimum_gaussian_reduction: float,
    minimum_lower_gaussian_repeats: int,
) -> tuple[dict[str, object], dict[str, object]]:
    records = list(candidates)
    red_candidates = [
        record
        for record in records
        if float(record["psnr_gain"]) >= minimum_psnr_gain
        and int(record["positive_psnr_repeats"])
        >= minimum_positive_repeats
        and float(record["mu_delta"]) >= minimum_mu_change
    ]
    red_candidates.sort(
        key=lambda record: (
            -float(record["mu_relative_change"])
            * float(record["psnr_gain"])
            * math.sqrt(
                max(float(record["redistribution_l1"]), 1e-9)
            ),
            record["box"][1],
            record["box"][0],
        )
    )
    if not red_candidates:
        raise RuntimeError("no positive-probability ROI satisfies the rules")

    for red in red_candidates:
        blue_candidates = [
            record
            for record in records
            if float(record["psnr_gain"]) >= minimum_psnr_gain
            and int(record["positive_psnr_repeats"])
            >= minimum_positive_repeats
            and float(record["mu_delta"]) <= -minimum_mu_change
            and float(record["gaussian_reduction"])
            >= minimum_gaussian_reduction
            and int(record["lower_gaussian_repeats"])
            >= minimum_lower_gaussian_repeats
            and _center_distance(red["box"], record["box"])
            >= minimum_center_distance
        ]
        blue_candidates.sort(
            key=lambda record: (
                -(
                    -float(record["mu_relative_change"])
                    * float(record["psnr_gain"])
                    * math.sqrt(float(record["gaussian_reduction"]))
                    * math.sqrt(
                        max(float(record["redistribution_l1"]), 1e-9)
                    )
                ),
                record["box"][1],
                record["box"][0],
            )
        )
        if blue_candidates:
            return red, blue_candidates[0]
    raise RuntimeError("no separated lower-budget ROI satisfies the rules")


def _boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return (
        max(first[0], second[0]) < min(first[2], second[2])
        and max(first[1], second[1]) < min(first[3], second[3])
    )


def choose_additional_redistribution_rois(
    candidates: Iterable[dict[str, object]],
    *,
    protected_boxes: Iterable[tuple[int, int, int, int]],
    minimum_center_distance: float,
    minimum_psnr_gain: float,
    minimum_negative_psnr_gain: float,
    minimum_positive_repeats: int,
    minimum_mu_change: float,
    minimum_sampling_share: float,
    maximum_sampling_share: float,
    maximum_positive_gaussian_change: float,
    minimum_gaussian_reduction: float,
    minimum_lower_gaussian_repeats: int,
) -> tuple[dict[str, object], dict[str, object]]:
    records = list(candidates)
    protected = tuple(protected_boxes)

    def separated(
        record: dict[str, object],
        boxes: Iterable[tuple[int, int, int, int]],
    ) -> bool:
        return all(
            not _boxes_overlap(record["box"], box)
            and _center_distance(record["box"], box)
            >= minimum_center_distance
            for box in boxes
        )

    def mean_value(
        record: dict[str, object],
        field: str,
        method: str,
    ) -> float:
        return float(np.mean(record[field][method]))

    def shares_are_local(record: dict[str, object]) -> bool:
        shares = (
            mean_value(record, "sampling_share", "base"),
            mean_value(record, "sampling_share", "r_e_d"),
        )
        return (
            min(shares) >= minimum_sampling_share
            and max(shares) <= maximum_sampling_share
        )

    positive_candidates = []
    for record in records:
        gaussian_change = (
            mean_value(record, "gaussian_numbers", "r_e_d")
            - mean_value(record, "gaussian_numbers", "base")
        )
        if (
            float(record["psnr_gain"]) >= minimum_psnr_gain
            and int(record["positive_psnr_repeats"])
            >= minimum_positive_repeats
            and float(record["mu_delta"]) >= minimum_mu_change
            and abs(gaussian_change)
            <= maximum_positive_gaussian_change
            and shares_are_local(record)
            and separated(record, protected)
        ):
            positive_candidates.append(record)
    positive_candidates.sort(
        key=lambda record: (
            -(
                float(record["psnr_gain"])
                * float(record["mu_relative_change"])
                * math.sqrt(
                    max(float(record["redistribution_l1"]), 1e-9)
                )
            ),
            record["box"][1],
            record["box"][0],
        )
    )
    if not positive_candidates:
        raise RuntimeError("no additional positive-mass ROI satisfies the rules")
    orange = positive_candidates[0]

    protected_with_orange = protected + (orange["box"],)
    negative_candidates = [
        record
        for record in records
        if float(record["psnr_gain"])
        >= max(minimum_psnr_gain, minimum_negative_psnr_gain)
        and int(record["positive_psnr_repeats"])
        >= minimum_positive_repeats
        and float(record["mu_delta"]) <= -minimum_mu_change
        and float(record["gaussian_reduction"])
        >= minimum_gaussian_reduction
        and int(record["lower_gaussian_repeats"])
        >= minimum_lower_gaussian_repeats
        and shares_are_local(record)
        and separated(record, protected_with_orange)
    ]
    negative_candidates.sort(
        key=lambda record: (
            -(
                -float(record["mu_relative_change"])
                * float(record["psnr_gain"])
                * math.sqrt(float(record["gaussian_reduction"]))
                * math.sqrt(
                    max(float(record["redistribution_l1"]), 1e-9)
                )
            ),
            record["box"][1],
            record["box"][0],
        )
    )
    if not negative_candidates:
        raise RuntimeError("no additional negative-mass ROI satisfies the rules")
    return orange, negative_candidates[0]


def _axis_limits(
    values: np.ndarray,
    *,
    minimum_padding: float,
    relative_padding: float,
) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = max(minimum_padding, relative_padding * (upper - lower))
    return lower - padding, upper + padding


def contrast_enhanced_bubble_areas(
    expected_samples: np.ndarray,
    *,
    base_area: float = BUBBLE_BASE_AREA,
    exponent: float = BUBBLE_CONTRAST_EXPONENT,
) -> np.ndarray:
    expected = np.asarray(expected_samples, dtype=np.float64)
    if expected.shape != (2,) or np.any(expected <= 0):
        raise ValueError("expected sample counts must contain two positives")
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected sample counts must be finite")
    if not math.isfinite(base_area) or base_area <= 0:
        raise ValueError("base bubble area must be positive")
    if not math.isfinite(exponent) or exponent <= 0:
        raise ValueError("bubble contrast exponent must be positive")
    normalized = expected / expected[0]
    return base_area * np.power(normalized, exponent)


def sampling_share_bubble_areas(
    sampling_share: np.ndarray,
    *,
    maximum_area: float = 620.0,
    contrast_exponent: float = 1.0,
) -> np.ndarray:
    shares = np.asarray(sampling_share, dtype=np.float64)
    if shares.ndim != 1 or shares.size == 0 or np.any(shares <= 0):
        raise ValueError("sampling shares must be positive")
    if not np.all(np.isfinite(shares)):
        raise ValueError("sampling shares must be finite")
    if not math.isfinite(maximum_area) or maximum_area <= 0:
        raise ValueError("maximum bubble area must be positive")
    if (
        not math.isfinite(contrast_exponent)
        or contrast_exponent <= 0
    ):
        raise ValueError("bubble contrast exponent must be positive")
    normalized = shares / float(np.max(shares))
    return maximum_area * np.power(normalized, contrast_exponent)


def relative_sampling_share_bubble_areas(
    sampling_share: np.ndarray,
    *,
    base_area: float = 360.0,
    contrast_exponent: float = 2.5,
) -> np.ndarray:
    shares = np.asarray(sampling_share, dtype=np.float64)
    if shares.shape != (2,) or np.any(shares <= 0):
        raise ValueError("sampling shares must contain Base and Ours positives")
    if not np.all(np.isfinite(shares)):
        raise ValueError("sampling shares must be finite")
    if not math.isfinite(base_area) or base_area <= 0:
        raise ValueError("base bubble area must be positive")
    if not math.isfinite(contrast_exponent) or contrast_exponent <= 0:
        raise ValueError("bubble contrast exponent must be positive")
    return base_area * np.power(shares / shares[0], contrast_exponent)


def save_efficiency_bubble_chart(
    *,
    gaussian_numbers: np.ndarray,
    local_psnr: np.ndarray,
    expected_samples: np.ndarray,
    output_stem: Path,
) -> None:
    counts = np.asarray(gaussian_numbers, dtype=np.float64)
    quality = np.asarray(local_psnr, dtype=np.float64)
    expected = np.asarray(expected_samples, dtype=np.float64)
    if counts.shape != (2,) or quality.shape != (2,):
        raise ValueError("the chart requires Base and Ours values")
    if expected.shape != (2,) or np.any(expected <= 0):
        raise ValueError("expected sample counts must contain two positives")
    if not np.all(np.isfinite(np.concatenate((counts, quality, expected)))):
        raise ValueError("bubble-chart values must be finite")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(5.4, 4.5))
    areas = contrast_enhanced_bubble_areas(expected)
    axis.annotate(
        "",
        xy=(counts[1], quality[1]),
        xytext=(counts[0], quality[0]),
        arrowprops={
            "arrowstyle": "->",
            "color": "#4A4A4A",
            "linewidth": 1.8,
            "linestyle": "--",
            "shrinkA": 19,
            "shrinkB": 19,
        },
        zorder=1,
    )
    for index, (label, color) in enumerate(
        zip(METHOD_LABELS, METHOD_COLORS)
    ):
        axis.scatter(
            counts[index],
            quality[index],
            s=areas[index],
            c=color,
            edgecolors="#222222",
            linewidths=1.4,
            alpha=0.86,
            zorder=3,
        )
        horizontal = -14 if index == 0 else 14
        vertical = -19 if index == 0 else 19
        relative_change = 100.0 * (expected[index] / expected[0] - 1.0)
        change_text = "reference" if index == 0 else f"{relative_change:+.1f}%"
        axis.annotate(
            (
                f"{label}\n"
                f"$\\mu(\\mathcal{{R}})$ = {expected[index]:.1f}\n"
                f"({change_text})"
            ),
            (counts[index], quality[index]),
            xytext=(horizontal, vertical),
            textcoords="offset points",
            ha="right" if index == 0 else "left",
            va="top" if index == 0 else "bottom",
            color="#222222" if index == 0 else "#A82424",
            fontweight="bold",
            fontsize=13,
            zorder=4,
        )

    axis.set_xlabel("Gaussian Numbers")
    axis.set_ylabel("Local PSNR (dB)")
    axis.set_xlim(
        _axis_limits(
            counts,
            minimum_padding=6.0,
            relative_padding=0.55,
        )
    )
    axis.set_ylim(
        _axis_limits(
            quality,
            minimum_padding=0.10,
            relative_padding=0.42,
        )
    )
    axis.grid(True, color="#D7D7D7", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.set_title(
        (
            "Bubble area $\\propto "
            "[\\mu(\\mathcal{R})/\\mu_{\\mathrm{Base}}(\\mathcal{R})]^4$"
        ),
        loc="left",
        fontsize=10.5,
        color="#444444",
        pad=8.0,
    )
    figure.tight_layout(pad=0.8)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def save_combined_roi_efficiency_chart(
    *,
    red_gaussian_numbers: np.ndarray,
    red_local_psnr: np.ndarray,
    red_sampling_share: np.ndarray,
    blue_gaussian_numbers: np.ndarray,
    blue_local_psnr: np.ndarray,
    blue_sampling_share: np.ndarray,
    output_stem: Path,
    orange_gaussian_numbers: np.ndarray | None = None,
    orange_local_psnr: np.ndarray | None = None,
    orange_sampling_share: np.ndarray | None = None,
    purple_gaussian_numbers: np.ndarray | None = None,
    purple_local_psnr: np.ndarray | None = None,
    purple_sampling_share: np.ndarray | None = None,
    relative_share_area: bool = False,
) -> None:
    regions = [
        {
            "counts": np.asarray(
                red_gaussian_numbers,
                dtype=np.float64,
            ),
            "quality": np.asarray(red_local_psnr, dtype=np.float64),
            "share": np.asarray(red_sampling_share, dtype=np.float64),
            "colors": ("#E9A09A", "#C83E3E"),
            "arrow": "#B52D2D",
            "arrow_curvature": ROI_ARROW_CURVATURES["red"],
            "label_offsets": ((-8, -14), (-2, 15)),
            "base_value_offset": (13, -2),
            "base_value_alignment": "left",
            "value_offset": (14, 1),
            "value_alignment": "left",
        },
        {
            "counts": np.asarray(
                blue_gaussian_numbers,
                dtype=np.float64,
            ),
            "quality": np.asarray(blue_local_psnr, dtype=np.float64),
            "share": np.asarray(blue_sampling_share, dtype=np.float64),
            "colors": ("#9FC7E3", "#2678B8"),
            "arrow": "#1F6FA9",
            "arrow_curvature": ROI_ARROW_CURVATURES["blue"],
            "label_offsets": ((0, -14), (0, 15)),
            "base_value_offset": (-13, -1),
            "base_value_alignment": "right",
            "value_offset": (-13, 1),
            "value_alignment": "right",
        },
    ]
    optional_regions = (
        (
            (
                orange_gaussian_numbers,
                orange_local_psnr,
                orange_sampling_share,
            ),
            {
                "colors": ("#F4C979", ORANGE_COLOR),
                "arrow": "#B97700",
                "arrow_curvature": ROI_ARROW_CURVATURES["orange"],
                "label_offsets": ((-5, -14), (0, 15)),
                "base_value_offset": (13, -2),
                "base_value_alignment": "left",
                "value_offset": (14, 1),
                "value_alignment": "left",
            },
        ),
        (
            (
                purple_gaussian_numbers,
                purple_local_psnr,
                purple_sampling_share,
            ),
            {
                "colors": ("#C9B5DC", PURPLE_COLOR),
                "arrow": "#76509A",
                "arrow_curvature": ROI_ARROW_CURVATURES["purple"],
                "label_offsets": ((0, -14), (0, 15)),
                "base_value_offset": (13, -2),
                "base_value_alignment": "left",
                "value_offset": (14, 1),
                "value_alignment": "left",
            },
        ),
    )
    for values, style in optional_regions:
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(
                "each optional region requires counts, PSNR, and share"
            )
        regions.append(
            {
                "counts": np.asarray(values[0], dtype=np.float64),
                "quality": np.asarray(values[1], dtype=np.float64),
                "share": np.asarray(values[2], dtype=np.float64),
                **style,
            }
        )
    for metrics in regions:
        counts = metrics["counts"]
        quality = metrics["quality"]
        share = metrics["share"]
        if counts.shape != (2,) or quality.shape != (2,):
            raise ValueError("each region requires Base and Ours values")
        if share.shape != (2,) or np.any(share <= 0):
            raise ValueError("sampling shares must contain two positives")
        if not np.all(
            np.isfinite(np.concatenate((counts, quality, share)))
        ):
            raise ValueError("chart values must be finite")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.8))
    all_counts = np.concatenate(
        [metrics["counts"] for metrics in regions]
    )
    all_quality = np.concatenate(
        [metrics["quality"] for metrics in regions]
    )
    if relative_share_area:
        region_areas = [
            relative_sampling_share_bubble_areas(metrics["share"])
            for metrics in regions
        ]
    else:
        all_shares = np.concatenate(
            [metrics["share"] for metrics in regions]
        )
        all_areas = sampling_share_bubble_areas(
            all_shares,
            maximum_area=SAMPLING_SHARE_BUBBLE_MAX_AREA,
            contrast_exponent=SAMPLING_SHARE_BUBBLE_CONTRAST_EXPONENT,
        )
        region_areas = [
            all_areas[index : index + 2]
            for index in range(0, all_areas.size, 2)
        ]
    area_label = "Bubble area = ROI sampling share (%)"
    x_limits = (
        float(np.min(all_counts)) - 24.0,
        float(np.max(all_counts)) + 32.0,
    )
    y_limits = (
        float(np.min(all_quality)) - 1.10,
        float(np.max(all_quality)) + 1.15,
    )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    metric_annotations = []
    for metrics, areas in zip(regions, region_areas):
        counts = metrics["counts"]
        quality = metrics["quality"]
        arrow_color = metrics["arrow"]
        base_radius = math.sqrt(float(areas[0]) / math.pi)
        ours_radius = math.sqrt(float(areas[1]) / math.pi)
        arrow = axis.annotate(
            "",
            xy=(counts[1], quality[1]),
            xytext=(counts[0], quality[0]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": arrow_color,
                "linewidth": 3.4,
                "shrinkA": max(base_radius * 0.72, 2.0),
                "shrinkB": max(ours_radius * 0.72, 2.0),
                "mutation_scale": 22,
                "connectionstyle": (
                    f"arc3,rad={metrics['arrow_curvature']}"
                ),
            },
            zorder=2,
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=5.4,
                        foreground="white",
                    ),
                    path_effects.Normal(),
                ]
            )

        for index, method in enumerate(METHOD_LABELS):
            axis.scatter(
                counts[index],
                quality[index],
                s=areas[index],
                c=metrics["colors"][index],
                edgecolors=arrow_color if index else "#4A4A4A",
                linewidths=2.0 if index else 1.5,
                alpha=0.93,
                zorder=3,
            )
            metric_annotations.append(axis.annotate(
                method,
                (counts[index], quality[index]),
                xytext=metrics["label_offsets"][index],
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=13,
                fontfamily="Times New Roman",
                fontweight="bold",
                color=arrow_color if index else "#202020",
                annotation_clip=True,
                clip_on=True,
                zorder=4,
            ))

        metric_annotations.append(axis.annotate(
            f"{counts[0]:.0f}, {quality[0]:.2f} dB",
            (counts[0], quality[0]),
            xytext=metrics["base_value_offset"],
            textcoords="offset points",
            ha=metrics["base_value_alignment"],
            va="center",
            fontsize=13,
            fontfamily="Times New Roman",
            fontweight="bold",
            color="#303030",
            annotation_clip=True,
            clip_on=True,
            zorder=4,
        ))
        metric_annotations.append(axis.annotate(
            f"{counts[1]:.0f}, {quality[1]:.2f} dB",
            (counts[1], quality[1]),
            xytext=metrics["value_offset"],
            textcoords="offset points",
            ha=metrics["value_alignment"],
            va="center",
            fontsize=14,
            fontfamily="Times New Roman",
            fontweight="bold",
            color=arrow_color,
            annotation_clip=True,
            clip_on=True,
            zorder=4,
        ))

    axis.set_xlabel("Gaussian Numbers")
    axis.set_ylabel("Local PSNR (dB)")
    axis.set_title(
        "Local Gaussian Sampling Redistribution",
        fontsize=18,
        fontfamily="Times New Roman",
        fontweight="bold",
        pad=14,
    )
    area_note = axis.text(
        0.985,
        0.970,
        area_label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        fontfamily="Times New Roman",
        fontweight="bold",
        color="#303030",
        bbox={
            "boxstyle": "square,pad=0.28",
            "facecolor": "#F2F2F2",
            "edgecolor": "#B8B8B8",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
        zorder=5,
    )
    axis.grid(
        True,
        color="#D7D7D7",
        linestyle=":",
        linewidth=0.75,
        alpha=0.65,
    )
    axis.set_axisbelow(True)
    for tick_label in (
        list(axis.get_xticklabels()) + list(axis.get_yticklabels())
    ):
        tick_label.set_fontname("Times New Roman")
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#333333")
    figure.tight_layout(pad=1.0)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_bounds = axis.get_window_extent(renderer=renderer)
    for annotation in metric_annotations + [area_note]:
        bounds = annotation.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < axis_bounds.x0 - 1.0
            or bounds.y0 < axis_bounds.y0 - 1.0
            or bounds.x1 > axis_bounds.x1 + 1.0
            or bounds.y1 > axis_bounds.y1 + 1.0
        ):
            raise RuntimeError("chart annotation exceeds the coordinate box")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def mean_normalized_probability(
    arrays: dict[str, np.ndarray],
    *,
    method: str,
    repeat: int,
) -> np.ndarray:
    distributions = []
    for repetition in range(1, repeat + 1):
        values = np.asarray(
            arrays[f"{method}_repeat_{repetition}_final_probability"],
            dtype=np.float64,
        )
        total = float(values.sum())
        if values.ndim != 2 or total <= 0:
            raise ValueError("probability maps must be positive 2D arrays")
        distributions.append(values / total)
    if not distributions:
        raise ValueError("at least one repetition is required")
    return np.mean(np.stack(distributions, axis=0), axis=0)


def deterministic_probability_points(
    probability: np.ndarray,
    *,
    count: int,
    seed: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("probability must be a finite 2D array")
    if count <= 0:
        raise ValueError("point count must be positive")
    eligible = values > 0
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError("valid mask must match probability shape")
        eligible &= valid
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        raise ValueError("probability map has no eligible pixels")
    count = min(count, int(indices.size))
    weights = values.ravel()[indices]
    random = np.random.default_rng(seed)
    scores = np.log(weights) + random.gumbel(size=weights.size)
    selected = indices[np.argpartition(scores, -count)[-count:]]
    y_coordinates, x_coordinates = np.unravel_index(
        selected,
        values.shape,
    )
    return x_coordinates.astype(np.float64), y_coordinates.astype(np.float64)


def _add_probability_roi_boxes(
    axis: plt.Axes,
    *,
    red_box: tuple[int, int, int, int],
    blue_box: tuple[int, int, int, int],
    orange_box: tuple[int, int, int, int] | None = None,
    purple_box: tuple[int, int, int, int] | None = None,
) -> None:
    regions = [(red_box, RED_COLOR), (blue_box, BLUE_COLOR)]
    if orange_box is not None:
        regions.append((orange_box, ORANGE_COLOR))
    if purple_box is not None:
        regions.append((purple_box, PURPLE_COLOR))
    for box, color in regions:
        x0, y0, x1, y1 = box
        axis.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="white",
                linewidth=ROI_BOX_HALO_LINEWIDTH,
                zorder=5,
            )
        )
        axis.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=color,
                linewidth=ROI_BOX_COLOR_LINEWIDTH,
                zorder=6,
            )
        )


def _style_probability_axis(
    axis: plt.Axes,
    *,
    width: int,
    height: int,
    title: str,
) -> None:
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")
    axis.set_title(title, fontsize=13, fontweight="bold", pad=8)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#777777")


def _export_probability_figure(
    figure: plt.Figure,
    output_stem: Path,
    *,
    dpi: int,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def save_global_probability_redistribution_maps(
    *,
    base_probability: np.ndarray,
    ours_probability: np.ndarray,
    valid_mask: np.ndarray,
    red_box: tuple[int, int, int, int],
    blue_box: tuple[int, int, int, int],
    output_dir: Path,
    orange_box: tuple[int, int, int, int] | None = None,
    purple_box: tuple[int, int, int, int] | None = None,
    output_suffix: str = "",
    display_budget: int = 1000,
    dpi: int = 600,
) -> None:
    base = np.asarray(base_probability, dtype=np.float64)
    ours = np.asarray(ours_probability, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if base.shape != ours.shape or base.shape != valid.shape:
        raise ValueError("Base, Ours, and valid mask must align")
    if base.ndim != 2:
        raise ValueError("probability maps must be 2D")
    for legacy_stem in (
        "full_frame_probability_redistribution_points",
        "full_frame_probability_change_points",
    ):
        for extension in (".png", ".pdf", ".svg"):
            legacy_path = output_dir / f"{legacy_stem}{extension}"
            if legacy_path.is_file():
                legacy_path.unlink()
    height, width = base.shape

    base_points = deterministic_probability_points(
        base,
        count=display_budget,
        seed=713,
        valid_mask=valid,
    )
    ours_points = deterministic_probability_points(
        ours,
        count=display_budget,
        seed=713,
        valid_mask=valid,
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    panels = (
        ("Base Sampling Probability", base_points, "#777777", "s"),
        (
            "Ours Sampling Probability",
            ours_points,
            OURS_SAMPLING_COLOR,
            "s",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.35))
    for axis, (title, points, color, marker) in zip(axes, panels):
        axis.scatter(
            points[0],
            points[1],
            s=10.5,
            marker=marker,
            c=color,
            edgecolors="none",
            alpha=0.9,
            rasterized=True,
        )
        _add_probability_roi_boxes(
            axis,
            red_box=red_box,
            blue_box=blue_box,
            orange_box=orange_box,
            purple_box=purple_box,
        )
        _style_probability_axis(
            axis,
            width=width,
            height=height,
            title=title,
        )

    figure.suptitle(
        "Full-frame Sampling Probability Distribution",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.045,
        (
            f"Both panels use the same display budget ($N={display_budget}$). "
            "Points visualize normalized sampling probability, "
            "not Gaussian primitives."
        ),
        ha="center",
        va="center",
        fontsize=9.3,
    )
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.84,
        bottom=0.14,
        wspace=0.10,
    )
    _export_probability_figure(
        figure,
        output_dir
        / f"full_frame_sampling_probability_comparison{output_suffix}",
        dpi=dpi,
    )

    separate_panels = (
        (
            "Base Sampling Distribution",
            base_points,
            "#777777",
            "s",
            "full_frame_base_sampling_points",
        ),
        (
            "Ours Sampling Distribution",
            ours_points,
            OURS_SAMPLING_COLOR,
            "s",
            "full_frame_ours_sampling_points",
        ),
    )
    for title, points, color, marker, name in separate_panels:
        panel_figure, panel_axis = plt.subplots(figsize=(7.2, 5.55))
        panel_axis.scatter(
            points[0],
            points[1],
            s=22.0,
            marker=marker,
            c=color,
            edgecolors="none",
            alpha=0.92,
            rasterized=True,
        )
        _add_probability_roi_boxes(
            panel_axis,
            red_box=red_box,
            blue_box=blue_box,
            orange_box=orange_box,
            purple_box=purple_box,
        )
        _style_probability_axis(
            panel_axis,
            width=width,
            height=height,
            title=title,
        )
        panel_figure.text(
            0.5,
            0.035,
            f"Fixed display budget: $N={display_budget}$",
            ha="center",
            fontsize=11,
        )
        panel_figure.subplots_adjust(
            left=0.03,
            right=0.985,
            top=0.92,
            bottom=0.09,
        )
        _export_probability_figure(
            panel_figure,
            output_dir / f"{name}{output_suffix}",
            dpi=dpi,
        )


def _integral_image(values: np.ndarray) -> np.ndarray:
    integral = np.asarray(values, dtype=np.float64).cumsum(0).cumsum(1)
    return np.pad(integral, ((1, 0), (1, 0)))


def _integral_sum(
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


def _local_gaussian_count(
    id_map: np.ndarray,
    box: tuple[int, int, int, int],
) -> int:
    x0, y0, x1, y1 = box
    identifiers = np.asarray(id_map)[y0:y1, x0:x1]
    identifiers = identifiers[identifiers >= 0]
    return int(np.unique(identifiers).size)


def build_candidates(
    arrays: dict[str, np.ndarray],
    *,
    repeat: int,
    window_width: int,
    window_height: int,
    stride: int,
    margin: int,
    minimum_valid_fraction: float,
) -> list[dict[str, object]]:
    ground_truth = arrays["ground_truth"].astype(np.float64) / 255.0
    valid = arrays["valid_mask"].astype(np.float64)
    height, width = valid.shape
    valid_integral = _integral_image(valid)
    error_integrals: dict[tuple[str, int], np.ndarray] = {}
    probability_integrals: dict[tuple[str, int], np.ndarray] = {}
    probability_totals: dict[tuple[str, int], float] = {}
    redistribution_integrals: dict[int, np.ndarray] = {}
    for method in METHODS:
        for repetition in range(1, repeat + 1):
            prefix = f"{method}_repeat_{repetition}"
            render = arrays[f"{prefix}_render"].astype(np.float64) / 255.0
            squared_error = ((ground_truth - render) ** 2).mean(axis=2)
            error_integrals[method, repetition] = _integral_image(
                squared_error * valid
            )
            probability = arrays[f"{prefix}_final_probability"]
            probability_integrals[method, repetition] = _integral_image(
                probability
            )
            probability_totals[method, repetition] = float(
                np.asarray(probability, dtype=np.float64).sum()
            )
    for repetition in range(1, repeat + 1):
        prefix = f"r_e_d_repeat_{repetition}"
        redistribution_integrals[repetition] = _integral_image(
            np.abs(
                arrays[f"{prefix}_final_probability"].astype(np.float64)
                - arrays[f"{prefix}_base_probability"].astype(np.float64)
            )
        )

    candidates = []
    for y0 in range(
        margin,
        height - window_height - margin + 1,
        stride,
    ):
        for x0 in range(
            margin,
            width - window_width - margin + 1,
            stride,
        ):
            box = (
                x0,
                y0,
                x0 + window_width,
                y0 + window_height,
            )
            valid_count = _integral_sum(valid_integral, box)
            if (
                valid_count / float(window_width * window_height)
                < minimum_valid_fraction
            ):
                continue
            psnr_values: dict[str, list[float]] = {
                method: [] for method in METHODS
            }
            mu_values: dict[str, list[float]] = {
                method: [] for method in METHODS
            }
            share_values: dict[str, list[float]] = {
                method: [] for method in METHODS
            }
            for method in METHODS:
                for repetition in range(1, repeat + 1):
                    mse = (
                        _integral_sum(
                            error_integrals[method, repetition],
                            box,
                        )
                        / valid_count
                    )
                    psnr_values[method].append(
                        -10.0 * math.log10(max(mse, 1e-12))
                    )
                    roi_mass = _integral_sum(
                        probability_integrals[method, repetition],
                        box,
                    )
                    mu_values[method].append(roi_mass)
                    share_values[method].append(
                        100.0
                        * roi_mass
                        / max(
                            probability_totals[method, repetition],
                            1e-12,
                        )
                    )
            psnr_gain = float(
                np.mean(psnr_values["r_e_d"])
                - np.mean(psnr_values["base"])
            )
            positive_psnr_repeats = sum(
                ours > base
                for ours, base in zip(
                    psnr_values["r_e_d"],
                    psnr_values["base"],
                )
            )
            mu_delta = float(
                np.mean(mu_values["r_e_d"])
                - np.mean(mu_values["base"])
            )
            base_mu = float(np.mean(mu_values["base"]))
            mu_relative_change = mu_delta / max(base_mu, 1e-12)
            if abs(mu_delta) < 1e-12 or positive_psnr_repeats == 0:
                continue
            gaussian_values: dict[str, list[int]] = {
                method: [
                    _local_gaussian_count(
                        arrays[f"{method}_repeat_{repetition}_ids"],
                        box,
                    )
                    for repetition in range(1, repeat + 1)
                ]
                for method in METHODS
            }
            gaussian_reduction = float(
                np.mean(gaussian_values["base"])
                - np.mean(gaussian_values["r_e_d"])
            )
            candidates.append(
                {
                    "box": box,
                    "local_psnr": psnr_values,
                    "expected_samples": mu_values,
                    "sampling_share": share_values,
                    "gaussian_numbers": gaussian_values,
                    "psnr_gain": psnr_gain,
                    "positive_psnr_repeats": positive_psnr_repeats,
                    "mu_delta": mu_delta,
                    "mu_relative_change": mu_relative_change,
                    "redistribution_l1": float(
                        np.mean(
                            [
                                _integral_sum(
                                    redistribution_integrals[repetition],
                                    box,
                                )
                                for repetition in range(1, repeat + 1)
                            ]
                        )
                    ),
                    "gaussian_reduction": gaussian_reduction,
                    "lower_gaussian_repeats": sum(
                        ours < base
                        for ours, base in zip(
                            gaussian_values["r_e_d"],
                            gaussian_values["base"],
                        )
                    ),
                }
            )
    return candidates


def _metric_summary(
    name: str,
    record: dict[str, object],
) -> list[dict[str, object]]:
    output = []
    for method, label in zip(METHODS, METHOD_LABELS):
        gaussian_mean = float(
            np.mean(record["gaussian_numbers"][method])
        )
        psnr_mean = float(np.mean(record["local_psnr"][method]))
        mu_mean = float(np.mean(record["expected_samples"][method]))
        share_mean = float(np.mean(record["sampling_share"][method]))
        output.append(
            {
                "ROI": name,
                "Method": label,
                "Gaussian Numbers": int(math.floor(gaussian_mean)),
                "Gaussian Mean (raw)": gaussian_mean,
                "Local PSNR (dB)": psnr_mean,
                "Expected Samples in ROI": mu_mean,
                "Sampling Share (%)": share_mean,
            }
        )
    return output


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def box_image_with_rois(
    image: Image.Image,
    *,
    red_box: tuple[int, int, int, int],
    blue_box: tuple[int, int, int, int],
    orange_box: tuple[int, int, int, int] | None = None,
    purple_box: tuple[int, int, int, int] | None = None,
    width: int = 8,
) -> Image.Image:
    if width <= 0:
        raise ValueError("ROI box width must be positive")
    boxed = image.copy()
    drawing = ImageDraw.Draw(boxed)
    regions = [(red_box, RED_COLOR), (blue_box, BLUE_COLOR)]
    if orange_box is not None:
        regions.append((orange_box, ORANGE_COLOR))
    if purple_box is not None:
        regions.append((purple_box, PURPLE_COLOR))
    for box, color in regions:
        x0, y0, x1, y1 = box
        if x0 < 0 or y0 < 0 or x1 > image.width or y1 > image.height:
            raise ValueError("ROI box must stay inside the image")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("ROI box must have positive area")
        drawing.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline=color,
            width=width,
        )
    return boxed


def main() -> int:
    args = parse_args()
    with np.load(args.input) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    candidates = build_candidates(
        arrays,
        repeat=args.repeat,
        window_width=args.window_width,
        window_height=args.window_height,
        stride=args.stride,
        margin=args.margin,
        minimum_valid_fraction=args.minimum_valid_fraction,
    )
    red, blue = choose_redistribution_rois(
        candidates,
        minimum_center_distance=args.minimum_center_distance,
        minimum_psnr_gain=args.minimum_psnr_gain,
        minimum_positive_repeats=args.minimum_positive_repeats,
        minimum_mu_change=args.minimum_mu_change,
        minimum_gaussian_reduction=args.minimum_gaussian_reduction,
        minimum_lower_gaussian_repeats=(
            args.minimum_lower_gaussian_repeats
        ),
    )
    orange, purple = choose_additional_redistribution_rois(
        candidates,
        protected_boxes=(red["box"], blue["box"]),
        minimum_center_distance=120.0,
        minimum_psnr_gain=args.minimum_psnr_gain,
        minimum_negative_psnr_gain=1.25,
        minimum_positive_repeats=args.minimum_positive_repeats,
        minimum_mu_change=args.minimum_mu_change,
        minimum_sampling_share=2.0,
        maximum_sampling_share=10.0,
        maximum_positive_gaussian_change=50.0,
        minimum_gaussian_reduction=args.minimum_gaussian_reduction,
        minimum_lower_gaussian_repeats=(
            args.minimum_lower_gaussian_repeats
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(arrays["ground_truth"].astype(np.uint8), mode="RGB")
    boxed = box_image_with_rois(
        image,
        red_box=red["box"],
        blue_box=blue["box"],
    )
    boxed.save(args.output_dir / "xyz_002882_gt_red_blue_boxes.png")
    selected_render = Image.fromarray(
        arrays["base_repeat_1_render"].astype(np.uint8),
        mode="RGB",
    )
    boxed_render = box_image_with_rois(
        selected_render,
        red_box=red["box"],
        blue_box=blue["box"],
    )
    boxed_render.save(
        args.output_dir / "xyz_002882_render_red_blue_boxes.png"
    )
    boxed_four_rois = box_image_with_rois(
        selected_render,
        red_box=red["box"],
        blue_box=blue["box"],
        orange_box=orange["box"],
        purple_box=purple["box"],
    )
    boxed_four_rois.save(
        args.output_dir / "xyz_002882_render_four_roi_boxes.png"
    )
    image.crop(red["box"]).save(args.output_dir / "red_roi_gt.png")
    image.crop(blue["box"]).save(args.output_dir / "blue_roi_gt.png")
    base_probability = mean_normalized_probability(
        arrays,
        method="base",
        repeat=args.repeat,
    )
    ours_probability = mean_normalized_probability(
        arrays,
        method="r_e_d",
        repeat=args.repeat,
    )
    frame_probability_change = ours_probability - base_probability
    save_global_probability_redistribution_maps(
        base_probability=base_probability,
        ours_probability=ours_probability,
        valid_mask=arrays["valid_mask"].astype(bool),
        red_box=red["box"],
        blue_box=blue["box"],
        output_dir=args.output_dir,
    )
    save_global_probability_redistribution_maps(
        base_probability=base_probability,
        ours_probability=ours_probability,
        valid_mask=arrays["valid_mask"].astype(bool),
        red_box=red["box"],
        blue_box=blue["box"],
        orange_box=orange["box"],
        purple_box=purple["box"],
        output_dir=args.output_dir,
        output_suffix="_four_rois",
    )

    rows = []
    region_rows: dict[str, list[dict[str, object]]] = {}
    for name, record in (
        ("Red ROI", red),
        ("Blue ROI", blue),
        ("Orange ROI", orange),
        ("Purple ROI", purple),
    ):
        selected_rows = _metric_summary(name, record)
        region_rows[name] = selected_rows
        rows.extend(selected_rows)
        save_efficiency_bubble_chart(
            gaussian_numbers=np.array(
                [row["Gaussian Numbers"] for row in selected_rows],
                dtype=np.float64,
            ),
            local_psnr=np.array(
                [row["Local PSNR (dB)"] for row in selected_rows],
                dtype=np.float64,
            ),
            expected_samples=np.array(
                [
                    row["Expected Samples in ROI"]
                    for row in selected_rows
                ],
                dtype=np.float64,
            ),
            output_stem=(
                args.output_dir
                / f"{name.lower().replace(' ', '_')}_base_ours_bubble"
            ),
        )
    def region_values(region_name: str, field: str) -> np.ndarray:
        return np.array(
            [row[field] for row in region_rows[region_name]],
            dtype=np.float64,
        )

    combined_arguments = {
        "red_gaussian_numbers": region_values(
            "Red ROI",
            "Gaussian Numbers",
        ),
        "red_local_psnr": region_values(
            "Red ROI",
            "Local PSNR (dB)",
        ),
        "red_sampling_share": region_values(
            "Red ROI",
            "Sampling Share (%)",
        ),
        "blue_gaussian_numbers": region_values(
            "Blue ROI",
            "Gaussian Numbers",
        ),
        "blue_local_psnr": region_values(
            "Blue ROI",
            "Local PSNR (dB)",
        ),
        "blue_sampling_share": region_values(
            "Blue ROI",
            "Sampling Share (%)",
        ),
    }
    save_combined_roi_efficiency_chart(
        **combined_arguments,
        output_stem=(
            args.output_dir / "combined_red_blue_roi_efficiency"
        ),
    )
    save_combined_roi_efficiency_chart(
        **combined_arguments,
        orange_gaussian_numbers=region_values(
            "Orange ROI",
            "Gaussian Numbers",
        ),
        orange_local_psnr=region_values(
            "Orange ROI",
            "Local PSNR (dB)",
        ),
        orange_sampling_share=region_values(
            "Orange ROI",
            "Sampling Share (%)",
        ),
        purple_gaussian_numbers=region_values(
            "Purple ROI",
            "Gaussian Numbers",
        ),
        purple_local_psnr=region_values(
            "Purple ROI",
            "Local PSNR (dB)",
        ),
        purple_sampling_share=region_values(
            "Purple ROI",
            "Sampling Share (%)",
        ),
        output_stem=(
            args.output_dir / "combined_four_roi_efficiency"
        ),
        relative_share_area=True,
    )

    csv_path = args.output_dir / "roi_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "third_dimension": {
            "name": "Expected Samples in ROI",
            "symbol": "mu_t(R)",
            "definition": "sum of final Bernoulli probabilities inside ROI",
            "bubble_area_encoding": (
                "A = A_base * (mu / mu_base)^4; exact mu and relative "
                "change are annotated"
            ),
            "combined_chart_encoding": (
                "red-blue chart uses one global monotonic power mapping; "
                "the four-ROI chart encodes each Ours bubble relative to "
                "its ROI-specific Base sampling share (Base area = 360 "
                "pt^2; exponent 2.5)"
            ),
            "combined_chart_quantity": {
                "name": "ROI Sampling Share",
                "unit": "%",
                "definition": (
                    "100 * sum of final sampling probabilities inside ROI "
                    "/ sum over the full frame"
                ),
            },
        },
        "selection_rules": {
            "minimum_psnr_gain": args.minimum_psnr_gain,
            "minimum_positive_repeats": args.minimum_positive_repeats,
            "minimum_mu_change": args.minimum_mu_change,
            "minimum_gaussian_reduction": (
                args.minimum_gaussian_reduction
            ),
            "minimum_lower_gaussian_repeats": (
                args.minimum_lower_gaussian_repeats
            ),
            "minimum_center_distance": args.minimum_center_distance,
            "additional_roi_rules": {
                "minimum_center_distance": 120.0,
                "minimum_negative_psnr_gain": 1.25,
                "minimum_sampling_share": 2.0,
                "maximum_sampling_share": 10.0,
                "maximum_positive_gaussian_change": 50.0,
            },
        },
        "red_roi": _json_ready(red),
        "blue_roi": _json_ready(blue),
        "orange_roi": _json_ready(orange),
        "purple_roi": _json_ready(purple),
        "full_frame_probability_change": {
            "normalized_sum": float(frame_probability_change.sum()),
            "positive_mass": float(
                np.clip(frame_probability_change, 0.0, None).sum()
            ),
            "negative_mass": float(
                np.clip(-frame_probability_change, 0.0, None).sum()
            ),
            "visual_encoding": {
                "base_and_ours_display_budget": 1000,
                "increase_markers": 420,
                "decrease_markers": 420,
                "increase_marker": "red square",
                "decrease_marker": "blue circle",
                "selection": (
                    "deterministic weighted sampling without replacement"
                ),
            },
        },
        "reported_rows": rows,
    }
    (args.output_dir / "selection_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "red": red["box"],
                "blue": blue["box"],
                "orange": orange["box"],
                "purple": purple["box"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
