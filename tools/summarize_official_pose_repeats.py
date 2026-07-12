#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_official_pose_benchmark import (  # noqa: E402
    REFERENCE_ROOT,
    SCENES,
    evaluate_scene,
    load_metadata_trajectory,
    load_reference,
    metadata_summary,
    read_json,
)


METRIC_DIRECTIONS = {
    "PSNR": "higher",
    "SSIM": "higher",
    "LPIPS": "lower",
    "time": "lower",
    "ATE_RMSE": "lower",
    "APE_R_deg": "lower",
    "RPE_t_RMSE": "lower",
    "RPE_R_deg": "lower",
}


def _sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def aggregate_records(
    records: list[dict[str, Any]],
    metric_directions: dict[str, str] = METRIC_DIRECTIONS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["scene"]), int(record["repeat"]), str(record["method"]))
        grouped[key] = record
    scenes = sorted({str(record["scene"]) for record in records})
    output: list[dict[str, Any]] = []
    for scene in scenes:
        repeats = sorted(
            {
                int(record["repeat"])
                for record in records
                if str(record["scene"]) == scene
            }
        )
        paired_repeats = [
            repeat
            for repeat in repeats
            if (scene, repeat, "v31") in grouped and (scene, repeat, "v31_a") in grouped
        ]
        for metric, direction in metric_directions.items():
            pairs: list[tuple[float, float]] = []
            used_repeats: list[int] = []
            for repeat in paired_repeats:
                left = grouped[(scene, repeat, "v31")].get(metric)
                right = grouped[(scene, repeat, "v31_a")].get(metric)
                if left is None or right is None:
                    continue
                left_value = float(left)
                right_value = float(right)
                if not math.isfinite(left_value) or not math.isfinite(right_value):
                    continue
                pairs.append((left_value, right_value))
                used_repeats.append(repeat)
            if not pairs:
                continue
            v31_values = [pair[0] for pair in pairs]
            a_values = [pair[1] for pair in pairs]
            deltas = [right - left for left, right in pairs]
            if direction == "higher":
                wins = sum(delta > 0 for delta in deltas)
            elif direction == "lower":
                wins = sum(delta < 0 for delta in deltas)
            else:
                raise ValueError(f"Unknown metric direction: {direction}")
            output.append(
                {
                    "scene": scene,
                    "metric": metric,
                    "direction": direction,
                    "n": len(pairs),
                    "repeat_ids": ";".join(str(value) for value in used_repeats),
                    "v31_mean": float(np.mean(v31_values)),
                    "v31_std": _sample_std(v31_values),
                    "v31_a_mean": float(np.mean(a_values)),
                    "v31_a_std": _sample_std(a_values),
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": _sample_std(deltas),
                    "a_better_count": int(wins),
                }
            )
    return output


def collect_run_records(
    run_root: Path,
    *,
    repeat: int,
    scene_names: list[str],
    reference_root: Path = REFERENCE_ROOT,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene_name in scene_names:
        spec = SCENES[scene_name]
        reference = load_reference(spec, reference_root)
        model_dirs = {
            method: run_root / method / scene_name / "model"
            for method in ("v31", "v31_a")
        }
        trajectories = {
            method: load_metadata_trajectory(path)
            for method, path in model_dirs.items()
        }
        pose_report = evaluate_scene(
            reference,
            {
                "baseline": trajectories["v31"],
                "v31": trajectories["v31"],
                "v31_a": trajectories["v31_a"],
            },
        )
        for method in ("v31", "v31_a"):
            pose = pose_report["methods"][method]
            metadata = metadata_summary(model_dirs[method])
            record = {
                "dataset": spec.dataset,
                "reference_type": spec.reference_type,
                "scene": scene_name,
                "repeat": repeat,
                "method": method,
                "common_frames": pose_report["three_way_common_frames"],
                "coverage": pose["coverage"],
                "PSNR": metadata["PSNR"],
                "SSIM": metadata["SSIM"],
                "LPIPS": metadata["LPIPS"],
                "time": metadata["time"],
                "ATE_RMSE": pose["ape_trans_rmse"],
                "APE_R_deg": pose["ape_rot_deg_mean"],
                "RPE_t_RMSE": pose["rpe_trans_rmse"],
                "RPE_R_deg": pose["rpe_rot_deg_mean"],
                "risk_candidates": 0,
                "quarantined": 0,
                "model_dir": str(model_dirs[method]),
            }
            if method == "v31_a":
                trace = read_json(model_dirs[method] / "pose_risk_utility_trace.json")
                summary = trace.get("summary", {})
                if isinstance(summary, dict):
                    record["risk_candidates"] = int(summary.get("risk_candidates", 0))
                    record["quarantined"] = int(summary.get("pose_reference_quarantined", 0))
            records.append(record)
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def write_reports(output_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_records(records)
    _write_csv(output_dir / "repeat_raw.csv", records)
    _write_csv(output_dir / "repeat_summary.csv", summary)
    payload = {"records": records, "summary": summary}
    (output_dir / "repeat_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V31 vs V31+A Paired Repeats",
        "",
        "Delta is V31+A minus V31. Positive is favorable only for PSNR and SSIM; negative is favorable for all other metrics.",
        "",
        "| Scene | Metric | V31 mean +/- std | V31+A mean +/- std | Paired delta | A wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scene']} | {row['metric']} | "
            f"{_format(row['v31_mean'])} +/- {_format(row['v31_std'])} | "
            f"{_format(row['v31_a_mean'])} +/- {_format(row['v31_a_std'])} | "
            f"{_format(row['delta_mean'])} +/- {_format(row['delta_std'])} | "
            f"{row['a_better_count']}/{row['n']} |"
        )
    (output_dir / "repeat_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--scenes", nargs="*", choices=sorted(SCENES), default=["forest1", "long_office"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records: list[dict[str, Any]] = []
    for repeat, run_root in enumerate(args.run_root, start=1):
        records.extend(
            collect_run_records(
                run_root,
                repeat=repeat,
                scene_names=list(args.scenes),
                reference_root=args.reference_root,
            )
        )
    write_reports(args.output_dir, records)
    print(args.output_dir / "repeat_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
