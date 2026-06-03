#!/usr/bin/env python3
"""Launch baseline + v2.2.2.1 short800/1000 then protective short500."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/zxd/miniconda3/envs/otf/bin/python"
OUT = REPO / "results/StaticHikes/forest1/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_1_GAP_FIX_V1"
LIFECYCLE = REPO / "results/StaticHikes/forest1/lifecycle/all_input_frame_lifecycle.csv"
BUILD = REPO / "tools/build_direct_density_rebalance_v2_2_2_1.py"
DATASET = REPO / "datasets/StaticHikes/forest1"


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
        env["CUDA_VISIBLE_DEVICES"] = "7"
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        logf.write(f"exit_code: {proc.returncode}\n")
    return int(proc.returncode)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument(
        "--only",
        default="",
        help="baseline_guard|short800|short1000|short500|post_run|all",
    )
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seq_log = OUT / "sequential_train.log"
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
    ]

    if not args.skip_baseline and (not args.only or args.only == "baseline_guard"):
        bg = OUT / "baseline_guard"
        code = run_one(
            "baseline_guard",
            bg,
            [
                "--max_frames",
                "500",
                "--paper_aligned_contract_trace_path",
                str(bg / "model" / "semantic_trace.json"),
            ],
            bg / "train.log",
        )
        if code != 0:
            return code
        subprocess.run(
            [PYTHON, str(BUILD), "--phase", "baseline_guard", "--output_root", str(OUT)],
            cwd=str(REPO),
        )

    def run_short(key: str, nf: int) -> int:
        rd = OUT / f"direct_density_v2_2_2_1_{key}"
        return run_one(
            f"v2_2_2_1_{key}",
            rd,
            [
                *common,
                "--max_frames",
                str(nf),
                "--paper_aligned_contract_trace_path",
                str(rd / "model" / "semantic_trace.json"),
            ],
            rd / "train.log",
        )

    run800 = not args.only or args.only in {"short800", "all"}
    run1000 = not args.only or args.only in {"short1000", "all"}
    run500 = not args.only or args.only in {"short500", "all"}

    if run800 and args.only != "short500":
        code = run_short("short800", 800)
        with seq_log.open("a", encoding="utf-8") as f:
            f.write(f"=== short800 ec={code} ===\n")
        if code != 0:
            return code

    if run1000 and args.only != "short500":
        code = run_short("short1000", 1000)
        with seq_log.open("a", encoding="utf-8") as f:
            f.write(f"=== short1000 ec={code} ===\n")
        if code != 0:
            return code

    if args.only in {"short800", "short1000"}:
        return 0

    if not args.only or args.only == "all":
        gate = subprocess.run(
            [PYTHON, str(BUILD), "--phase", "short800_1000_gate", "--output_root", str(OUT)],
            cwd=str(REPO),
        )
        if gate.returncode != 0:
            subprocess.run(
                [PYTHON, str(BUILD), "--phase", "post_run", "--output_root", str(OUT)],
                cwd=str(REPO),
            )
            return 0

    if run500:
        code = run_short("short500", 500)
        if code != 0:
            return code

    if not args.only or args.only in {"post_run", "all", "short500"}:
        post = subprocess.run(
            [PYTHON, str(BUILD), "--phase", "post_run", "--output_root", str(OUT)],
            cwd=str(REPO),
        )
        return int(post.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
