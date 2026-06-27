#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path("/data2/zxd/3D_Reconstruction/On_the_fly")
PYTHON = Path("/home/zxd/miniconda3/envs/otf/bin/python")
PADDED_DATASET_ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly_padded_datasets")
EXPERIMENT_LABEL = os.environ.get(
    "OTF_EXPERIMENT_LABEL", "pose_safe_streaming_memory_v1_all"
)
OUT_ROOT = REPO / "results/BRANCH_EXPERIMENTS_20260627" / EXPERIMENT_LABEL
SUMMARY_CSV = OUT_ROOT / f"{EXPERIMENT_LABEL}_run_summary.csv"
SUMMARY_JSON = OUT_ROOT / f"{EXPERIMENT_LABEL}_run_summary.json"
MASTER_LOG = OUT_ROOT / "logs/master.log"

DATASETS = [
    ("bonsai", PADDED_DATASET_ROOT / "MipNerf360/bonsai", 8),
    ("counter", PADDED_DATASET_ROOT / "MipNerf360/counter", 8),
    ("garden", PADDED_DATASET_ROOT / "MipNerf360/garden", 8),
    ("forest1", PADDED_DATASET_ROOT / "StaticHikes/forest1", 10),
    ("forest2", PADDED_DATASET_ROOT / "StaticHikes/forest2", 10),
    ("university2", PADDED_DATASET_ROOT / "StaticHikes/university2", 10),
    ("desk1", PADDED_DATASET_ROOT / "TUM/rgbd_dataset_freiburg1_desk", 30),
    ("desk2", PADDED_DATASET_ROOT / "TUM/rgbd_dataset_freiburg2_xyz", 30),
    (
        "long_office_household",
        PADDED_DATASET_ROOT / "TUM/rgbd_dataset_freiburg3_long_office_household",
        30,
    ),
]

FIELDNAMES = [
    "dataset",
    "returncode",
    "test_hold",
    "source",
    "model_dir",
    "trace_path",
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


def command_for(source: Path, model_dir: Path, test_hold: int) -> list[str]:
    return [
        str(PYTHON),
        "train.py",
        "-s",
        str(source),
        "-m",
        str(model_dir.relative_to(REPO)),
        "--test_hold",
        str(test_hold),
        "--test_frequency",
        "20",
        "--enable_reboot",
        "--downsampling",
        "1.0",
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
        "pose_safe_streaming_memory_v1",
        "--paper_aligned_direct_update_prev_desc_on_hold",
        "light",
        "--paper_aligned_direct_hold_tracking_bridge_mode",
        "light",
    ]


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    branch = git_value(["branch", "--show-current"])
    commit = git_value(["rev-parse", "--short", "HEAD"])
    rows: list[dict[str, Any]] = []
    log(f"RUN_START branch={branch} commit={commit} out_root={OUT_ROOT}")
    for dataset, source, test_hold in DATASETS:
        if not source.exists():
            raise FileNotFoundError(source)
        model_dir = OUT_ROOT / dataset / "model"
        log_path = model_dir / "train.log"
        if (model_dir / "metadata.json").exists() and log_path.exists():
            log(f"SKIP {dataset} already complete")
            rows.append(summarize_existing(dataset, source, test_hold, model_dir, log_path))
            write_outputs(rows)
            continue
        if model_dir.exists():
            shutil.rmtree(model_dir)
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
        row = summarize_existing(dataset, source, test_hold, model_dir, log_path)
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


def summarize_existing(
    dataset: str,
    source: Path,
    test_hold: int,
    model_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "returncode": 0,
        "test_hold": test_hold,
        "source": str(source),
        "model_dir": str(model_dir.relative_to(REPO)),
        "trace_path": str((model_dir / "semantic_trace.json").relative_to(REPO)),
        "log_path": str(log_path.relative_to(REPO)),
        "elapsed_wall_sec": "",
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "--short", "HEAD"]),
    }
    row.update(load_metadata(model_dir))
    return row


def load_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
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
    return {out_key: meta[meta_key] for out_key, meta_key in mapping.items() if meta_key in meta}


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
