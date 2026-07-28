#!/usr/bin/env python3
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


METHODS = ("base", "r", "r_e", "r_d", "r_e_d")
VARIANTS = {
    "base": ("Base", False, False, False),
    "r": ("Photo.", True, False, False),
    "r_e": ("Photo.+Struct.", True, True, False),
    "r_d": ("Photo.+Cov.", True, False, True),
    "r_e_d": ("Full", True, True, True),
}
DEFAULT_RED_BOX = (492, 88, 588, 168)
DEFAULT_BLUE_BOX = (524, 272, 620, 352)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create separate B-module internal-ablation tables for two "
            "fixed local regions."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--red-box",
        type=int,
        nargs=4,
        default=DEFAULT_RED_BOX,
    )
    parser.add_argument(
        "--blue-box",
        type=int,
        nargs=4,
        default=DEFAULT_BLUE_BOX,
    )
    return parser.parse_args()


def _validate_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> None:
    x0, y0, x1, y1 = box
    height, width = shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"ROI {box} lies outside image shape {shape}")


def _local_gaussian_count(
    id_map: np.ndarray,
    box: tuple[int, int, int, int],
) -> int:
    x0, y0, x1, y1 = box
    identifiers = np.asarray(id_map).squeeze()[y0:y1, x0:x1]
    identifiers = identifiers[identifiers >= 0]
    return int(np.unique(identifiers).size)


def _local_psnr(
    ground_truth: np.ndarray,
    render: np.ndarray,
    valid_mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = box
    reference = ground_truth[y0:y1, x0:x1].astype(np.float64) / 255.0
    estimate = render[y0:y1, x0:x1].astype(np.float64) / 255.0
    valid = valid_mask[y0:y1, x0:x1].astype(bool)
    if not np.any(valid):
        raise ValueError(f"ROI {box} contains no valid pixels")
    squared_error = np.mean((reference - estimate) ** 2, axis=2)
    mse = float(np.mean(squared_error[valid]))
    return -10.0 * math.log10(max(mse, 1e-12))


def summarize_roi(
    arrays: dict[str, np.ndarray],
    *,
    box: tuple[int, int, int, int],
    repeat_count: int,
) -> list[dict[str, object]]:
    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    ground_truth = np.asarray(arrays["ground_truth"])
    _validate_box(box, ground_truth.shape[:2])
    valid_mask = np.asarray(
        arrays.get(
            "valid_mask",
            np.mean(ground_truth, axis=2) > 8,
        )
    )
    records: list[dict[str, object]] = []
    for method in METHODS:
        label, photo, structure, coverage = VARIANTS[method]
        gaussian_runs = []
        psnr_runs = []
        for repetition in range(1, repeat_count + 1):
            prefix = f"{method}_repeat_{repetition}"
            render_key = f"{prefix}_render"
            ids_key = f"{prefix}_ids"
            if render_key not in arrays or ids_key not in arrays:
                raise KeyError(f"missing arrays for {prefix}")
            gaussian_runs.append(
                _local_gaussian_count(arrays[ids_key], box)
            )
            psnr_runs.append(
                _local_psnr(
                    ground_truth,
                    np.asarray(arrays[render_key]),
                    valid_mask,
                    box,
                )
            )
        gaussian_mean = float(np.mean(gaussian_runs))
        records.append(
            {
                "variant": label,
                "photo": photo,
                "structure": structure,
                "coverage": coverage,
                "gaussian_numbers": int(math.floor(gaussian_mean)),
                "gaussian_mean_raw": gaussian_mean,
                "local_psnr": float(np.mean(psnr_runs)),
                "gaussian_runs": gaussian_runs,
                "psnr_runs": psnr_runs,
            }
        )
    base_gaussians = int(records[0]["gaussian_numbers"])
    base_psnr = float(records[0]["local_psnr"])
    for record in records:
        record["delta_gaussian_numbers"] = (
            int(record["gaussian_numbers"]) - base_gaussians
        )
        record["delta_psnr"] = (
            float(record["local_psnr"]) - base_psnr
        )
    return records


def _signed_integer(value: int, *, base: bool) -> str:
    if base:
        return "—"
    return f"{value:+d}".replace("-", "−")


def _signed_decimal(value: float, *, base: bool) -> str:
    if base:
        return "—"
    return f"{value:+.3f}".replace("-", "−")


def _component_mark(enabled: bool) -> str:
    return "✓" if enabled else "×"


def _display_rows(
    rows: list[dict[str, object]],
) -> list[list[str]]:
    output = []
    for index, row in enumerate(rows):
        output.append(
            [
                str(row["variant"]),
                _component_mark(bool(row["photo"])),
                _component_mark(bool(row["structure"])),
                _component_mark(bool(row["coverage"])),
                str(int(row["gaussian_numbers"])),
                _signed_integer(
                    int(row["delta_gaussian_numbers"]),
                    base=index == 0,
                ),
                f"{float(row['local_psnr']):.3f}",
                _signed_decimal(
                    float(row["delta_psnr"]),
                    base=index == 0,
                ),
            ]
        )
    return output


def _write_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    fieldnames = [
        "Variant",
        "Photo.",
        "Struct.",
        "Cov.",
        "Gaussian Numbers",
        "Delta Gaussian Numbers",
        "Local PSNR (dB)",
        "Delta PSNR",
        "Gaussian Runs",
        "PSNR Runs",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "Variant": row["variant"],
                    "Photo.": int(bool(row["photo"])),
                    "Struct.": int(bool(row["structure"])),
                    "Cov.": int(bool(row["coverage"])),
                    "Gaussian Numbers": row["gaussian_numbers"],
                    "Delta Gaussian Numbers": (
                        ""
                        if index == 0
                        else row["delta_gaussian_numbers"]
                    ),
                    "Local PSNR (dB)": (
                        f"{float(row['local_psnr']):.6f}"
                    ),
                    "Delta PSNR": (
                        ""
                        if index == 0
                        else f"{float(row['delta_psnr']):+.6f}"
                    ),
                    "Gaussian Runs": json.dumps(
                        row["gaussian_runs"],
                        separators=(",", ":"),
                    ),
                    "PSNR Runs": json.dumps(
                        row["psnr_runs"],
                        separators=(",", ":"),
                    ),
                }
            )


def _write_latex(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    lines = [
        r"\begin{tabular}{lcccrrrr}",
        r"\toprule",
        (
            r"Variant & Photo. & Struct. & Cov. & \#Gauss. & "
            r"$\Delta$\#G & PSNR$\uparrow$ & $\Delta$PSNR \\"
        ),
        r"\midrule",
    ]
    display_rows = _display_rows(rows)
    for index, row in enumerate(display_rows):
        marks = [
            r"\checkmark" if value == "✓" else r"$\times$"
            for value in row[1:4]
        ]
        values = [row[0], *marks, *row[4:]]
        line = " & ".join(values) + r" \\"
        if index == len(display_rows) - 1:
            lines.append(r"\midrule")
            line = r"\textbf{" + values[0] + "} & " + " & ".join(
                values[1:4]
            ) + " & " + " & ".join(
                r"\textbf{" + value + "}"
                for value in values[4:]
            ) + r" \\"
        lines.append(line)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _save_table_figure(
    rows: list[dict[str, object]],
    *,
    roi_name: str,
    accent: str,
    output_stem: Path,
    repeat_count: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.8, 2.55))
    axis.axis("off")
    columns = [
        "Variant",
        "Photo.",
        "Struct.",
        "Cov.",
        "#Gauss.",
        "Δ#G",
        "PSNR↑",
        "ΔPSNR",
    ]
    table = axis.table(
        cellText=_display_rows(rows),
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.19, 0.10, 0.11, 0.09, 0.14, 0.12, 0.13, 0.13],
        bbox=[0.015, 0.18, 0.97, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    last_row = len(rows)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.get_text().set_fontfamily("Times New Roman")
        if row_index > 0 and 1 <= column_index <= 3:
            cell.get_text().set_fontfamily("DejaVu Sans")
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0.0)
        if row_index == 0:
            cell.set_facecolor("#F0F0F0")
            cell.get_text().set_weight("bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor("#333333")
            cell.set_linewidth(0.9)
        elif row_index == last_row:
            cell.set_facecolor(f"{accent}18")
            cell.get_text().set_weight("bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor("#555555")
            cell.set_linewidth(0.8)
            if column_index >= 4:
                cell.get_text().set_color(accent)
        elif row_index == 1:
            cell.visible_edges = "B"
            cell.set_edgecolor("#B0B0B0")
            cell.set_linewidth(0.6)
    axis.set_title(
        f"{roi_name}: B-module Internal Ablation",
        fontsize=15,
        fontfamily="Times New Roman",
        fontweight="bold",
        pad=5,
    )
    axis.text(
        0.5,
        0.055,
        (
            f"Mean of {repeat_count} complete runs; "
            "Δ values are relative to Base."
        ),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        fontfamily="Times New Roman",
        color="#333333",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 600), (".pdf", 300), (".svg", 300)):
        figure.savefig(
            output_stem.with_suffix(suffix),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(figure)


def export_roi_table(
    rows: list[dict[str, object]],
    *,
    roi_name: str,
    accent: str,
    output_stem: Path,
    repeat_count: int,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_stem.with_suffix(".csv"))
    _write_latex(rows, output_stem.with_suffix(".tex"))
    _save_table_figure(
        rows,
        roi_name=roi_name,
        accent=accent,
        output_stem=output_stem,
        repeat_count=repeat_count,
    )


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    with np.load(args.input, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    regions = (
        (
            "Red ROI",
            tuple(args.red_box),
            "#B52D2D",
            "red_roi_b_internal_ablation",
        ),
        (
            "Blue ROI",
            tuple(args.blue_box),
            "#1F6FA9",
            "blue_roi_b_internal_ablation",
        ),
    )
    summary = {}
    combined_rows = []
    for roi_name, box, accent, output_name in regions:
        rows = summarize_roi(
            arrays,
            box=box,
            repeat_count=args.repeat,
        )
        export_roi_table(
            rows,
            roi_name=roi_name,
            accent=accent,
            output_stem=args.output_dir / output_name,
            repeat_count=args.repeat,
        )
        summary[roi_name] = {"box": list(box), "rows": rows}
        for row in rows:
            combined_rows.append(
                {
                    "ROI": roi_name,
                    "Variant": row["variant"],
                    "Photo.": int(bool(row["photo"])),
                    "Struct.": int(bool(row["structure"])),
                    "Cov.": int(bool(row["coverage"])),
                    "Gaussian Numbers": row["gaussian_numbers"],
                    "Delta Gaussian Numbers": (
                        row["delta_gaussian_numbers"]
                    ),
                    "Local PSNR (dB)": row["local_psnr"],
                    "Delta PSNR": row["delta_psnr"],
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (
        args.output_dir / "two_roi_b_internal_ablation.csv"
    ).open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(combined_rows[0]),
        )
        writer.writeheader()
        writer.writerows(combined_rows)
    with (
        args.output_dir / "two_roi_b_internal_ablation.json"
    ).open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
