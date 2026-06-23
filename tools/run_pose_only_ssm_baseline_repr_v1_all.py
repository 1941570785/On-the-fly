#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path("/data2/zxd/3D_Reconstruction/On_the_fly")
PYTHON = Path("/home/zxd/miniconda3/envs/otf/bin/python")
EXPERIMENT_LABEL = os.environ.get(
    "OTF_EXPERIMENT_LABEL", "pose_only_ssm_baseline_repr_v1_all"
)
OUT_ROOT = REPO / "results/BRANCH_EXPERIMENTS_20260623" / EXPERIMENT_LABEL
SUMMARY_CSV = OUT_ROOT / f"{EXPERIMENT_LABEL}_run_summary.csv"
SUMMARY_JSON = OUT_ROOT / f"{EXPERIMENT_LABEL}_run_summary.json"
MASTER_LOG = OUT_ROOT / "logs/master.log"

DATASETS = [
    ("bonsai", "datasets/MipNeRF360/bonsai", 8),
    ("counter", "datasets/MipNeRF360/counter", 8),
    ("garden", "datasets/MipNeRF360/garden", 8),
    ("forest1", "datasets/StaticHikes/forest1", 10),
    ("forest2", "datasets/StaticHikes/forest2", 10),
    ("university2", "datasets/StaticHikes/university2", 10),
    ("desk1", "datasets/TUM/desk1", 30),
    ("desk2", "datasets/TUM/desk2", 30),
    ("long_office_household", "datasets/TUM/long_office_household", 30),
]

FIELDNAMES = [
    "dataset",
    "returncode",
    "test_hold",
    "model_dir",
    "log_path",
    "elapsed_wall_sec",
    "num anchors",
    "num keyframes",
    "time",
    "FPS",
    "PSNR",
    "SSIM",
    "LPIPS",
    "R_deg",
    "t",
    "branch",
    "commit",
]


def command_for(source: str, model_dir: Path, test_hold: int) -> list[str]:
    return [
        str(PYTHON),
        "train.py",
        "-s",
        source,
        "-m",
        str(model_dir.relative_to(REPO)),
        "--test_hold",
        str(test_hold),
        "--test_frequency",
        "20",
        "--paper_aligned_contract_trace_path",
        str((model_dir / "semantic_trace.json").relative_to(REPO)),
        "--risk_admission_mode",
        "on_the_fly_innovation_v1",
        "--paper_aligned_recovery_commit_bridge",
        "off",
        "--paper_aligned_defer_recovery_support_bridge",
        "off",
        "--paper_aligned_recovery_commit_control",
        "off",
        "--paper_aligned_direct_density_control",
        "pose_only_ssm_baseline_repr_v1",
        "--paper_aligned_direct_update_prev_desc_on_hold",
        "light",
        "--paper_aligned_direct_hold_tracking_bridge_mode",
        "light",
    ]


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    branch = git_value(["branch", "--show-current"])
    commit = git_value(["rev-parse", "--short", "HEAD"])
    log(f"RUN_START branch={branch} commit={commit}")
    for dataset, source, test_hold in DATASETS:
        model_dir = OUT_ROOT / dataset / "model"
        log_path = model_dir / "train.log"
        if (model_dir / "metadata.json").exists() and log_path.exists():
            log(f"SKIP {dataset} already complete")
            rows.append(summarize_existing(dataset, test_hold, model_dir, log_path))
            write_outputs(rows)
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for(source, model_dir, test_hold)
        (model_dir / "command.json").write_text(
            json.dumps(cmd, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"START {dataset} test_hold={test_hold}")
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log_file, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        row = summarize_existing(dataset, test_hold, model_dir, log_path)
        row["returncode"] = proc.returncode
        row["elapsed_wall_sec"] = elapsed
        rows.append(row)
        write_outputs(rows)
        log(
            f"DONE {dataset} rc={proc.returncode} "
            f"time={row.get('time')} wall={elapsed:.1f}"
        )
        if proc.returncode != 0:
            return proc.returncode
    log("ALL_DONE")
    return 0


def summarize_existing(dataset: str, test_hold: int, model_dir: Path, log_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "returncode": 0,
        "test_hold": test_hold,
        "model_dir": str(model_dir.relative_to(REPO)),
        "log_path": str(log_path.relative_to(REPO)),
        "elapsed_wall_sec": "",
    }
    row.update(load_metadata(model_dir))
    row["branch"] = git_value(["branch", "--show-current"])
    row["commit"] = git_value(["rev-parse", "--short", "HEAD"])
    return row


def load_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    row: dict[str, Any] = {}
    mapping = {
        "num anchors": "num anchors",
        "num keyframes": "num keyframes",
        "time": "time",
        "FPS": "FPS",
        "PSNR": "PSNR",
        "SSIM": "SSIM",
        "LPIPS": "LPIPS",
        "R_deg": "R\u00b0",
        "t": "t",
    }
    for out_key, meta_key in mapping.items():
        if meta_key in meta:
            row[out_key] = meta[meta_key]
    return row


def write_outputs(rows: list[dict[str, Any]]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception:
        return ""


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
