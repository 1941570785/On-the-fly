#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_extra_optimization_round_ablation import ACTIVE_SCENES


QUALITY_METRICS = ("PSNR", "SSIM", "LPIPS")
SUMMARY_METRICS = (
    "PSNR",
    "SSIM",
    "LPIPS",
    "time",
    "wall_time_seconds",
    "extra_events",
    "extra_applied",
    "extra_iterations",
    "extra_iterations_mean",
    "num_keyframes",
    "num_anchors",
)
PAIR_METRICS = ("PSNR_gain", "SSIM_gain", "LPIPS_gain", "time_overhead")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value!r}")
    return result


def normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["budget"] = int(row["budget"])
        row["seed"] = int(row["seed"])
        row["returncode"] = int(row.get("returncode", 1))
        row["dry_run"] = _as_bool(row.get("dry_run", False))
        for metric in SUMMARY_METRICS:
            if row.get(metric, "") in (None, ""):
                if metric in QUALITY_METRICS or metric == "time":
                    raise ValueError(f"missing {metric} for {row.get('job_id', '')}")
                row[metric] = 0.0
            else:
                row[metric] = _as_float(row[metric], field=metric)
        result.append(row)
    return result


def validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no experiment rows provided")
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        if int(row["returncode"]) != 0 or bool(row["dry_run"]):
            raise ValueError(f"incomplete experiment row: {row.get('job_id', '')}")
        key = (str(row["scene"]), int(row["seed"]), int(row["budget"]))
        if key in seen:
            raise ValueError(f"duplicate scene/seed/budget row: {key}")
        seen.add(key)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"manifest must contain a list of objects: {path}")
        return [dict(row) for row in value]
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _stats(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "std": statistics.stdev(items) if len(items) > 1 else 0.0,
        "min": min(items),
        "max": max(items),
    }


def _add_stats(target: dict[str, Any], name: str, values: Iterable[float]) -> None:
    for statistic, value in _stats(values).items():
        target[f"{name}_{statistic}"] = value


def scene_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["scene"]), int(row["budget"]))].append(row)
    result: list[dict[str, Any]] = []
    for (dataset, scene, budget), group in groups.items():
        item: dict[str, Any] = {
            "dataset": dataset,
            "scene": scene,
            "budget": budget,
            "runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            _add_stats(item, metric, (float(row[metric]) for row in group))
        result.append(item)
    return sorted(result, key=lambda row: (str(row["dataset"]), str(row["scene"]), int(row["budget"])))


def active_macro_summary(
    scene_rows: Sequence[dict[str, Any]],
    *,
    active_scenes: Sequence[str] = ACTIVE_SCENES,
) -> list[dict[str, Any]]:
    active = set(active_scenes)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        if str(row["scene"]) in active:
            groups[int(row["budget"])].append(row)
    result: list[dict[str, Any]] = []
    for budget, group in sorted(groups.items()):
        present = {str(row["scene"]) for row in group}
        if present != active:
            raise ValueError(
                f"budget {budget} active scenes mismatch: missing={sorted(active - present)}"
            )
        item: dict[str, Any] = {
            "budget": budget,
            "scenes": len(group),
            "active_scenes": ";".join(sorted(active)),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[f"{metric}_median"]) for row in group]
            item[f"{metric}_macro"] = statistics.fmean(values)
            item[f"{metric}_scene_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        result.append(item)
    if not result:
        raise ValueError("no active-scene rows found")
    return result


def pair_against_zero(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(row["scene"]), int(row["seed"]), int(row["budget"])): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for (scene, seed, budget), row in sorted(indexed.items()):
        if budget == 0:
            continue
        baseline = indexed.get((scene, seed, 0))
        if baseline is None:
            raise ValueError(f"missing zero-budget pair for {(scene, seed, budget)}")
        result.append(
            {
                "dataset": row["dataset"],
                "scene": scene,
                "seed": seed,
                "budget": budget,
                "PSNR_gain": float(row["PSNR"]) - float(baseline["PSNR"]),
                "SSIM_gain": float(row["SSIM"]) - float(baseline["SSIM"]),
                "LPIPS_gain": float(baseline["LPIPS"]) - float(row["LPIPS"]),
                "time_overhead": float(row["time"]) - float(baseline["time"]),
                "extra_iterations": float(row["extra_iterations"]),
                "extra_applied": float(row["extra_applied"]),
            }
        )
    return result


def scene_stratified_bootstrap(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_719,
) -> tuple[float, float]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene"])].append(float(row[field]))
    if not groups:
        raise ValueError("cannot bootstrap empty rows")
    scenes = sorted(groups)
    scene_means = np.asarray(
        [statistics.fmean(groups[scene]) for scene in scenes], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(scenes), size=(replicates, len(scenes)))
    estimates = scene_means[samples].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_budget_summary(
    paired_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
) -> dict[int, dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        groups[int(row["budget"])].append(row)
    result: dict[int, dict[str, Any]] = {}
    for budget, group in sorted(groups.items()):
        item: dict[str, Any] = {
            "budget": budget,
            "comparisons": len(group),
            "scenes": len({str(row["scene"]) for row in group}),
        }
        for metric in PAIR_METRICS:
            values = [float(row[metric]) for row in group]
            _add_stats(item, metric, values)
            low, high = scene_stratified_bootstrap(
                group,
                metric,
                replicates=bootstrap_replicates,
                seed=20_260_719 + budget,
            )
            item[f"{metric}_ci95_low"] = low
            item[f"{metric}_ci95_high"] = high
            item[f"{metric}_wins"] = sum(value > 0.0 for value in values)
        result[budget] = item
    return result


def _paired_candidate_vs_max(
    rows: Sequence[dict[str, Any]],
    *,
    candidate: int,
    maximum: int,
    active_scenes: Sequence[str],
) -> list[dict[str, Any]]:
    active = set(active_scenes)
    indexed = {
        (str(row["scene"]), int(row["seed"]), int(row["budget"])): row
        for row in rows
        if str(row["scene"]) in active
    }
    keys = sorted(
        (scene, seed)
        for scene, seed, budget in indexed
        if budget == maximum
    )
    result: list[dict[str, Any]] = []
    for scene, seed in keys:
        candidate_row = indexed.get((scene, seed, candidate))
        maximum_row = indexed.get((scene, seed, maximum))
        if candidate_row is None or maximum_row is None:
            raise ValueError(
                f"missing candidate/maximum pair for {(scene, seed, candidate, maximum)}"
            )
        result.append(
            {
                "scene": scene,
                "seed": seed,
                "PSNR_difference": float(candidate_row["PSNR"])
                - float(maximum_row["PSNR"]),
            }
        )
    return result


def select_budget(
    rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    *,
    active_scenes: Sequence[str] = ACTIVE_SCENES,
    bootstrap_replicates: int = 10_000,
    tolerance_db: float = 0.02,
) -> dict[str, Any]:
    if not curve_rows:
        raise ValueError("empty round curve")
    curve = {int(row["budget"]): row for row in curve_rows}
    maximum_row = max(
        curve_rows,
        key=lambda row: (float(row["PSNR_macro"]), -int(row["budget"])),
    )
    maximum_budget = int(maximum_row["budget"])
    maximum_psnr = float(maximum_row["PSNR_macro"])
    candidates: list[dict[str, Any]] = []
    for budget in sorted(value for value in curve if value <= maximum_budget):
        row = curve[budget]
        gap = maximum_psnr - float(row["PSNR_macro"])
        if budget == maximum_budget:
            low, high = 0.0, 0.0
        else:
            differences = _paired_candidate_vs_max(
                rows,
                candidate=budget,
                maximum=maximum_budget,
                active_scenes=active_scenes,
            )
            low, high = scene_stratified_bootstrap(
                differences,
                "PSNR_difference",
                replicates=bootstrap_replicates,
                seed=20_260_719 + budget * 101 + maximum_budget,
            )
        within_tolerance = gap <= float(tolerance_db) + 1e-12
        ci_overlaps_zero = low <= 0.0 <= high
        ssim_not_worse = float(row["SSIM_macro"]) >= float(maximum_row["SSIM_macro"])
        lpips_not_worse = float(row["LPIPS_macro"]) <= float(maximum_row["LPIPS_macro"])
        not_both_degrade = ssim_not_worse or lpips_not_worse
        equivalent = within_tolerance and ci_overlaps_zero and not_both_degrade
        candidates.append(
            {
                "budget": budget,
                "psnr_gap_to_max": gap,
                "psnr_difference_ci95_low": low,
                "psnr_difference_ci95_high": high,
                "within_0.02_db": within_tolerance,
                "ci_overlaps_zero": ci_overlaps_zero,
                "ssim_not_worse": ssim_not_worse,
                "lpips_not_worse": lpips_not_worse,
                "not_both_degrade": not_both_degrade,
                "equivalent": equivalent,
            }
        )
    equivalent = [item["budget"] for item in candidates if item["equivalent"]]
    selected = min(equivalent) if equivalent else maximum_budget
    return {
        "selected_budget": int(selected),
        "max_psnr_budget": maximum_budget,
        "max_psnr": maximum_psnr,
        "tolerance_db": float(tolerance_db),
        "bootstrap_replicates": int(bootstrap_replicates),
        "active_scenes": list(active_scenes),
        "candidates": candidates,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _selected_scene_rows(
    scene_rows: Sequence[dict[str, Any]], selected_budget: int
) -> list[dict[str, Any]]:
    return [
        row for row in scene_rows if int(row["budget"]) in {0, int(selected_budget)}
    ]


def _dataset_selected_summary(
    selected_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        groups[(str(row["dataset"]), int(row["budget"]))].append(row)
    result: list[dict[str, Any]] = []
    for (dataset, budget), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "dataset": dataset,
            "budget": budget,
            "scenes": len(group),
        }
        for metric in ("PSNR", "SSIM", "LPIPS", "time"):
            item[metric] = statistics.fmean(
                float(row[f"{metric}_median"]) for row in group
            )
        result.append(item)
    return result


def _plot_curve(
    curve_rows: Sequence[dict[str, Any]], selected_budget: int, output_dir: Path
) -> None:
    rows = sorted(curve_rows, key=lambda row: int(row["budget"]))
    budgets = [int(row["budget"]) for row in rows]
    panels = (
        ("PSNR_macro", "PSNR_scene_std", "PSNR (dB)", True),
        ("SSIM_macro", "SSIM_scene_std", "SSIM", True),
        ("LPIPS_macro", "LPIPS_scene_std", "LPIPS", False),
        ("time_macro", "time_scene_std", "Time (s)", False),
    )
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.2), constrained_layout=True)
    for axis, (field, error_field, label, higher) in zip(axes, panels):
        values = [float(row[field]) for row in rows]
        errors = [float(row[error_field]) for row in rows]
        axis.errorbar(
            budgets,
            values,
            yerr=errors,
            color="#2F6B9A",
            marker="o",
            linewidth=1.5,
            capsize=2.5,
        )
        axis.axvline(selected_budget, color="#C23B22", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Extra-round budget K")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_title("higher is better" if higher else "lower is better", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"extra_round_quality_curve.{suffix}", dpi=240)
    plt.close(fig)


def _plot_pareto(
    curve_rows: Sequence[dict[str, Any]], selected_budget: int, output_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    for row in sorted(curve_rows, key=lambda item: int(item["budget"])):
        budget = int(row["budget"])
        color = "#C23B22" if budget == selected_budget else "#2F6B9A"
        axis.scatter(float(row["time_macro"]), float(row["PSNR_macro"]), color=color)
        axis.annotate(
            f"K={budget}",
            (float(row["time_macro"]), float(row["PSNR_macro"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Reconstruction time (s)")
    axis.set_ylabel("PSNR (dB)")
    axis.grid(alpha=0.25, linewidth=0.6)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"extra_round_quality_time_pareto.{suffix}", dpi=240)
    plt.close(fig)


def _write_results_markdown(
    output_dir: Path,
    curve_rows: Sequence[dict[str, Any]],
    paired_summaries: dict[int, dict[str, Any]],
    selection: dict[str, Any],
    dataset_rows: Sequence[dict[str, Any]],
) -> None:
    selected = int(selection["selected_budget"])
    lines = [
        "# Extra-Optimization Round Ablation Results",
        "",
        f"Selected budget: **K={selected}**.",
        "",
        "## Active-Scene Sensitivity",
        "",
        "| K | PSNR | SSIM | LPIPS | Time (s) | Realized extra iterations |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(curve_rows, key=lambda item: int(item["budget"])):
        lines.append(
            f"| {int(row['budget'])} | {float(row['PSNR_macro']):.5f} | "
            f"{float(row['SSIM_macro']):.5f} | {float(row['LPIPS_macro']):.5f} | "
            f"{float(row['time_macro']):.2f} | "
            f"{float(row['extra_iterations_macro']):.2f} |"
        )
    lines.extend(["", "## Paired Effect Versus K=0", ""])
    paired = paired_summaries.get(selected)
    if paired is not None:
        lines.extend(
            [
                f"- PSNR gain: {float(paired['PSNR_gain_mean']):+.5f} dB "
                f"(95% CI {float(paired['PSNR_gain_ci95_low']):+.5f}, "
                f"{float(paired['PSNR_gain_ci95_high']):+.5f}).",
                f"- SSIM gain: {float(paired['SSIM_gain_mean']):+.6f}.",
                f"- LPIPS gain: {float(paired['LPIPS_gain_mean']):+.6f} "
                "(positive means lower LPIPS).",
                f"- Time overhead: {float(paired['time_overhead_mean']):+.3f} s.",
            ]
        )
    lines.extend(
        [
            "",
            "## Selected Full-Scene Dataset Summary",
            "",
            "| Dataset | K | PSNR | SSIM | LPIPS | Time (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {int(row['budget'])} | "
            f"{float(row['PSNR']):.5f} | {float(row['SSIM']):.5f} | "
            f"{float(row['LPIPS']):.5f} | {float(row['time']):.2f} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    active_scenes: Sequence[str] = ACTIVE_SCENES,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    normalized = normalize_rows(rows)
    validate_rows(normalized)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_rows = scene_summary(normalized)
    curve_rows = active_macro_summary(scene_rows, active_scenes=active_scenes)
    paired_rows = pair_against_zero(
        [row for row in normalized if str(row["scene"]) in set(active_scenes)]
    )
    paired_summaries = paired_budget_summary(
        paired_rows, bootstrap_replicates=bootstrap_replicates
    )
    selection = select_budget(
        normalized,
        curve_rows,
        active_scenes=active_scenes,
        bootstrap_replicates=bootstrap_replicates,
    )
    selected_rows = _selected_scene_rows(scene_rows, int(selection["selected_budget"]))
    dataset_rows = _dataset_selected_summary(selected_rows)

    _write_csv(output_dir / "round_scene_summary.csv", scene_rows)
    _write_csv(output_dir / "round_active_macro_summary.csv", curve_rows)
    _write_csv(output_dir / "round_paired_vs_zero.csv", paired_rows)
    _write_csv(
        output_dir / "round_paired_vs_zero_summary.csv",
        list(paired_summaries.values()),
    )
    _write_csv(output_dir / "full9_selected_comparison.csv", selected_rows)
    _write_csv(output_dir / "full9_selected_dataset_summary.csv", dataset_rows)
    _write_json(output_dir / "selection.json", selection)
    _plot_curve(curve_rows, int(selection["selected_budget"]), output_dir)
    _plot_pareto(curve_rows, int(selection["selected_budget"]), output_dir)
    _write_results_markdown(
        output_dir, curve_rows, paired_summaries, selection, dataset_rows
    )
    return selection


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the paired extra-optimization round ablation."
    )
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-scenes", nargs="*", default=list(ACTIVE_SCENES))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    for manifest in args.manifests:
        rows.extend(read_manifest(manifest))
    selection = evaluate_rows(
        rows,
        output_dir=args.output_dir.resolve(),
        active_scenes=tuple(args.active_scenes),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(
        f"selected_budget={selection['selected_budget']} output={args.output_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
