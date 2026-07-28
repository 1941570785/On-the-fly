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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


METHODS = ("base", "r_e_d")
METHOD_LABELS = ("Base", "Ours")
METHOD_COLORS = ("#777777", "#C83E3E")
RED_COLOR = "#D62728"
BLUE_COLOR = "#1F77B4"


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
            -float(record["mu_delta"])
            * float(record["psnr_gain"])
            * max(float(record["redistribution_l1"]), 1e-9),
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
                    -float(record["mu_delta"])
                    * float(record["psnr_gain"])
                    * math.sqrt(float(record["gaussian_reduction"]))
                    * max(float(record["redistribution_l1"]), 1e-9)
                ),
                record["box"][1],
                record["box"][0],
            )
        )
        if blue_candidates:
            return red, blue_candidates[0]
    raise RuntimeError("no separated lower-budget ROI satisfies the rules")


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
    maximum_area = 2300.0
    areas = maximum_area * expected / float(expected.max())
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
        vertical = -17 if index == 0 else 17
        axis.annotate(
            f"{label}\n$\\mu(\\mathcal{{R}})$ = {expected[index]:.1f}",
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
    axis.text(
        0.02,
        0.98,
        "Bubble area $\\propto$ Expected Samples in ROI",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#444444",
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
    redistribution_integrals: dict[int, np.ndarray] = {}
    for method in METHODS:
        for repetition in range(1, repeat + 1):
            prefix = f"{method}_repeat_{repetition}"
            render = arrays[f"{prefix}_render"].astype(np.float64) / 255.0
            squared_error = ((ground_truth - render) ** 2).mean(axis=2)
            error_integrals[method, repetition] = _integral_image(
                squared_error * valid
            )
            probability_integrals[method, repetition] = _integral_image(
                arrays[f"{prefix}_final_probability"]
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
                    mu_values[method].append(
                        _integral_sum(
                            probability_integrals[method, repetition],
                            box,
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
                    "gaussian_numbers": gaussian_values,
                    "psnr_gain": psnr_gain,
                    "positive_psnr_repeats": positive_psnr_repeats,
                    "mu_delta": mu_delta,
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
        output.append(
            {
                "ROI": name,
                "Method": label,
                "Gaussian Numbers": int(math.floor(gaussian_mean)),
                "Gaussian Mean (raw)": gaussian_mean,
                "Local PSNR (dB)": psnr_mean,
                "Expected Samples in ROI": mu_mean,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(arrays["ground_truth"].astype(np.uint8), mode="RGB")
    boxed = image.copy()
    drawing = ImageDraw.Draw(boxed)
    for record, color in ((red, RED_COLOR), (blue, BLUE_COLOR)):
        x0, y0, x1, y1 = record["box"]
        drawing.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline=color,
            width=5,
        )
    boxed.save(args.output_dir / "xyz_002882_gt_red_blue_boxes.png")
    image.crop(red["box"]).save(args.output_dir / "red_roi_gt.png")
    image.crop(blue["box"]).save(args.output_dir / "blue_roi_gt.png")

    rows = []
    for name, record in (("Red ROI", red), ("Blue ROI", blue)):
        selected_rows = _metric_summary(name, record)
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
        },
        "red_roi": _json_ready(red),
        "blue_roi": _json_ready(blue),
        "reported_rows": rows,
    }
    (args.output_dir / "selection_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"red": red["box"], "blue": blue["box"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
