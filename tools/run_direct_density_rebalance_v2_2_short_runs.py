#!/usr/bin/env python3
"""Launch baseline guard + v2.2 short500/800/1000 live trains."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/zxd/miniconda3/envs/otf/bin/python"
OUT = REPO / "results/StaticHikes/forest1/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2"
LIFECYCLE = REPO / "results/StaticHikes/forest1/lifecycle/all_input_frame_lifecycle.csv"
BUILD = REPO / "tools/build_direct_density_rebalance_v2_2.py"
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
        logf.write(f"=== END {name} ec={proc.returncode} ===\n")
    return int(proc.returncode)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--only", default="", help="baseline_guard|short500|short800|short1000|post_run")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seq_log = OUT / "sequential_train.log"
    common_pnp = [
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
    ]
    jobs = []
    if not args.skip_baseline and (not args.only or args.only == "baseline_guard"):
        bg = OUT / "baseline_guard"
        jobs.append(
            (
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
        )
    v22_common = [
        *common_pnp,
        "--paper_aligned_direct_density_control",
        "target_band_v2_2",
    ]
    for label, nf in (
        ("short500", 500),
        ("short800", 800),
        ("short1000", 1000),
    ):
        key = f"short{nf}"
        if args.only and args.only not in {key, "all"}:
            continue
        run_dir = OUT / f"direct_density_v2_2_{key}"
        jobs.append(
            (
                f"v2_2_{key}",
                run_dir,
                [
                    *v22_common,
                    "--max_frames",
                    str(nf),
                    "--paper_aligned_contract_trace_path",
                    str(run_dir / "model" / "semantic_trace.json"),
                ],
                run_dir / "train.log",
            )
        )

    ec = 0
    for name, model_path, extra, log_path in jobs:
        code = run_one(name, model_path, extra, log_path)
        with seq_log.open("a", encoding="utf-8") as f:
            f.write(f"=== {name} ec={code} ===\n")
        if code != 0:
            print(f"{name} failed with exit {code}", file=sys.stderr)
            return code
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if not args.only or args.only in {"post_run", "all"}:
        post = subprocess.run(
            [PYTHON, str(BUILD), "--phase", "post_run", "--output_root", str(OUT)],
            cwd=str(REPO),
        )
        return int(post.returncode)
    return ec


if __name__ == "__main__":
    raise SystemExit(main())
