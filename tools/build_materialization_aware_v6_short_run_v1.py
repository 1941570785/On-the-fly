#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly")
HELPER = ROOT / "tools" / "build_commit_materialization_contract_fix_v1.py"
V5_BASELINE_POSE_FAIL = 166
V5_SHORT500_MATERIALIZATION_RATE = 0.2663


def load_helper():
    spec = importlib.util.spec_from_file_location("contract_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--short300_model", required=True)
    parser.add_argument("--short300_terminal", required=True)
    parser.add_argument("--short500_model", required=True)
    parser.add_argument("--short500_terminal", required=True)
    args = parser.parse_args()
    helper = load_helper()
    root = Path(args.root).resolve()
    short_root = root / "short_run"
    run300 = helper.summarize_run(
        "live_short_300",
        Path(args.short300_model).resolve(),
        Path(args.short300_terminal).resolve(),
        short_root / "live_short_300",
    )
    run500 = helper.summarize_run(
        "live_short_500",
        Path(args.short500_model).resolve(),
        Path(args.short500_terminal).resolve(),
        short_root / "live_short_500",
    )
    growth = int(run500["final_keyframe_count"]) - int(run300["final_keyframe_count"])
    mat500 = read_csv(short_root / "live_short_500" / "recovery_commit_materialization_trace.csv")
    pose_fail_300_500 = [
        row
        for row in mat500
        if 300 <= int(row.get("source_frame_id", -1) or -1) < 500
        and "inliers_too_few" in str(row.get("failure_reason", ""))
    ]
    materialized_300_500 = [
        row
        for row in mat500
        if 300 <= int(row.get("source_frame_id", -1) or -1) < 500
        and str(row.get("materialized", "")).lower() == "true"
    ]
    actual_kf_300_500 = 0
    for row in read_csv(short_root / "live_short_500" / "keyframe_timeline.csv"):
        sid = int(row.get("source_frame_id", row.get("frame_id", -1)) or -1)
        if 300 <= sid < 500:
            actual_kf_300_500 += 1
    reason_counts = Counter(str(row.get("failure_reason", "")) for row in pose_fail_300_500)
    run500["pose_failures_300_500"] = len(pose_fail_300_500)
    run500["materialized_commit_count_300_500"] = len(materialized_300_500)
    run500["actual_keyframe_added_300_500"] = actual_kf_300_500
    run500["pnp_miniba_too_few_failures"] = sum(
        1 for row in mat500 if "inliers_too_few" in str(row.get("failure_reason", ""))
    )
    run500["keyframe_growth_plateau"] = growth < 40
    run500["starvation_risk"] = bool(run500["keyframes_per_100_frames"] < 28.0 or run500["keyframe_growth_plateau"])
    comparison_rows = [
        {"run": "live_short_300", **run300},
        {"run": "live_short_500", **run500},
    ]
    write_csv(short_root / "materialization_aware_v6_short_run_comparison.csv", comparison_rows)
    ready = bool(
        run300["short_run_stable"]
        and run500["short_run_stable"]
        and run500["materialization_rate"] > V5_SHORT500_MATERIALIZATION_RATE
        and len(materialized_300_500) > 0
        and actual_kf_300_500 > 0
        and run500["pnp_miniba_too_few_failures"] < V5_BASELINE_POSE_FAIL
        and int(run300["final_keyframe_count"]) >= 90
        and int(run500["final_keyframe_count"]) >= 140
        and growth >= 40
        and 28.0 <= float(run500["keyframes_per_100_frames"]) <= 45.0
        and not bool(run500["keyframe_growth_plateau"])
        and not bool(run500["starvation_risk"])
        and float(run500["main_chain_gap_p90"]) <= 5.0
        and float(run500["main_chain_gap_p95"]) <= 7.0
        and float(run500["main_chain_gap_max"]) <= 20.0
    )
    comparison = {
        "live_short_300": run300,
        "live_short_500": run500,
        "keyframe_growth_300_to_500": growth,
        "pose_failure_reason_counts_300_500": dict(reason_counts),
        "ready_for_materialization_aware_v6_full_run": ready,
    }
    write_json(short_root / "materialization_aware_v6_short_run_comparison.json", comparison)
    write_json(short_root / "ready_for_materialization_aware_v6_full_run.json", comparison)
    (short_root / "paper_aligned_materialization_aware_v6_short_run_report.md").write_text(
        "\n".join(
            [
                "# materialization-aware v6 short run report",
                "",
                f"- short300 keyframes/density/materialization_rate: {run300['final_keyframe_count']}/{run300['keyframes_per_100_frames']:.3f}/{run300['materialization_rate']:.4f}",
                f"- short500 keyframes/density/materialization_rate: {run500['final_keyframe_count']}/{run500['keyframes_per_100_frames']:.3f}/{run500['materialization_rate']:.4f}",
                f"- growth 300->500: {growth}",
                f"- 300-500 materialized/actual_keyframe: {len(materialized_300_500)}/{actual_kf_300_500}",
                f"- pnp/miniba failures total: {run500['pnp_miniba_too_few_failures']} (v5={V5_BASELINE_POSE_FAIL})",
                f"- ready_for_materialization_aware_v6_full_run: {ready}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
