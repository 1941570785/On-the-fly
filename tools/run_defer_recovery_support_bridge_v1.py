#!/usr/bin/env python3
"""Launch PAPER_ALIGNED_DEFER_RECOVERY_SUPPORT_BRIDGE_V1 short runs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/zxd/miniconda3/envs/otf/bin/python"
OUT = REPO / "results/StaticHikes/forest1/PAPER_ALIGNED_DEFER_RECOVERY_SUPPORT_BRIDGE_V1"
LIFECYCLE = REPO / "results/StaticHikes/forest1/lifecycle/all_input_frame_lifecycle.csv"
DATASET = REPO / "datasets/StaticHikes/forest1"
BUILD = REPO / "tools/build_defer_recovery_support_bridge_v1.py"


def run_one(name: str, model_path: Path, extra_args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(REPO / "train.py"),
        "-s",
        str(DATASET),
        "-m",
        str(model_path),
        "--test_hold",
        "10",
        "--test_frequency",
        "20",
        *extra_args,
    ]
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"=== {name} {stamp} ===\n")
        logf.flush()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
        logf.write(f"exit_code: {proc.returncode}\n")
    return int(proc.returncode)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="short800", choices=["short800", "short1000", "all", "audit"])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    common = [
        "--risk_admission_mode",
        "paper_aligned_semantic_v1",
        "--paper_aligned_recovery_commit_bridge",
        "true_source_commit",
        "--paper_aligned_recovery_commit_control",
        "recovery_commit_early_seed_v7",
        "--paper_aligned_direct_update_prev_desc_on_hold",
        "off",
        "--paper_aligned_lifecycle_csv",
        str(LIFECYCLE),
        "--paper_aligned_direct_density_control",
        "target_band_v2_2_2_1",
        "--paper_aligned_defer_recovery_support_bridge",
        "v1",
    ]

    if args.only == "audit":
        return subprocess.call([PYTHON, str(BUILD), "--output_root", str(OUT)], cwd=str(REPO))

    if args.only in {"short800", "all"}:
        rd = OUT / "support_bridge_short800"
        code = run_one(
            "support_bridge_short800",
            rd,
            [
                *common,
                "--max_frames",
                "800",
                "--paper_aligned_contract_trace_path",
                str(rd / "model" / "semantic_trace.json"),
            ],
            rd / "train.log",
        )
        subprocess.run(
            [PYTHON, str(BUILD), "--output_root", str(OUT), "--run", "short800"],
            cwd=str(REPO),
        )
        if code != 0:
            return code

    if args.only in {"short1000", "all"}:
        summary = OUT / "support_bridge_short800_summary.json"
        if summary.exists():
            import json

            s = json.loads(summary.read_text())
            if not s.get("gate", {}).get("proceed_short1000", False):
                print("short800 gate failed; skip short1000", file=sys.stderr)
                return 1
        rd = OUT / "support_bridge_short1000"
        code = run_one(
            "support_bridge_short1000",
            rd,
            [
                *common,
                "--max_frames",
                "1000",
                "--paper_aligned_contract_trace_path",
                str(rd / "model" / "semantic_trace.json"),
            ],
            rd / "train.log",
        )
        subprocess.run(
            [PYTHON, str(BUILD), "--output_root", str(OUT), "--run", "short1000"],
            cwd=str(REPO),
        )
        return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
