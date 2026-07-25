#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("T_APE", "R_APE", "T_RPE", "R_RPE")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
) -> list[dict[str, Any]]:
    indexed = {
        (int(row["repeat"]), str(row["scene"]), str(row["variant"])): row
        for row in rows
    }
    presets = sorted({str(row["variant"]) for row in rows} - {baseline})
    scenes = sorted({str(row["scene"]) for row in rows})
    repeats = sorted({int(row["repeat"]) for row in rows})
    output: list[dict[str, Any]] = []
    for preset in presets:
        cell_ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
        for repeat in repeats:
            for scene in scenes:
                base = indexed.get((repeat, scene, baseline))
                candidate = indexed.get((repeat, scene, preset))
                if base is None or candidate is None:
                    continue
                for metric in METRICS:
                    denominator = max(float(base[metric]), 1e-12)
                    cell_ratios[(scene, metric)].append(
                        float(candidate[metric]) / denominator
                    )
        cell_means = {
            key: float(np.mean(values)) for key, values in cell_ratios.items()
        }
        all_ratios = np.asarray(
            [value for values in cell_ratios.values() for value in values],
            dtype=np.float64,
        )
        wins = sum(value < 1.0 for value in cell_means.values())
        regressions = sum(value > 1.02 for value in cell_means.values())
        output.append(
            {
                "preset": preset,
                "paired_observations": int(all_ratios.size),
                "mean_ratio": float(np.mean(all_ratios)) if all_ratios.size else 0.0,
                "median_ratio": (
                    float(np.median(all_ratios)) if all_ratios.size else 0.0
                ),
                "ratio_std": float(np.std(all_ratios)) if all_ratios.size else 0.0,
                "worst_cell_ratio": max(cell_means.values(), default=0.0),
                "mean_cell_wins": int(wins),
                "mean_cell_regressions_over_2pct": int(regressions),
                "passes_screen": bool(wins >= 7 and regressions == 0),
                "cell_mean_ratios": {
                    f"{scene}:{metric}": value
                    for (scene, metric), value in sorted(cell_means.items())
                },
            }
        )
    return sorted(
        output,
        key=lambda row: (
            not bool(row["passes_screen"]),
            int(row["mean_cell_regressions_over_2pct"]),
            float(row["mean_ratio"]),
            float(row["ratio_std"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_scene_csv", type=Path)
    parser.add_argument("--baseline", default="a_off")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(_read_rows(args.final_scene_csv), baseline=args.baseline)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
