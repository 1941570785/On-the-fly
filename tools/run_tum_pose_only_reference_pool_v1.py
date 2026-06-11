#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path("/data2/zxd/3D_Reconstruction/On_the_fly")
PYTHON = Path("/home/zxd/miniconda3/envs/otf/bin/python")
OUT_ROOT = REPO / "results/BRANCH_EXPERIMENTS_20260611/pose-only-reference-pool-v1_tum"
SUMMARY_CSV = OUT_ROOT / "pose_only_reference_pool_v1_tum_run_summary.csv"
SUMMARY_JSON = OUT_ROOT / "pose_only_reference_pool_v1_tum_run_summary.json"
TRACE_CSV = OUT_ROOT / "pose_only_reference_pool_v1_tum_trace_stats.csv"
TRACE_JSON = OUT_ROOT / "pose_only_reference_pool_v1_tum_trace_stats.json"
MASTER_LOG = OUT_ROOT / "logs/master.log"
MODE = "pose_rep_active_memory_v1"

DATASETS = [
    ("desk1", "datasets/TUM/desk1", 613),
    ("desk2", "datasets/TUM/desk2", 3669),
    ("long_office_household", "datasets/TUM/long_office_household", 2585),
]

FIELDNAMES = [
    "dataset",
    "returncode",
    "max_frames",
    "model_dir",
    "log_path",
    "elapsed_wall_sec",
    "num anchors",
    "num keyframes",
    "keyframes",
    "time",
    "time_sec",
    "FPS",
    "PSNR",
    "SSIM",
    "LPIPS",
    "R_deg",
    "t",
    "direct_density_event_count",
    "value_hold_count",
    "pose_only_tracking_hold_count",
    "active_memory_context_count",
    "active_memory_tracking_only_count",
    "active_memory_representation_count",
    "active_memory_marginal_value_mean",
    "active_memory_redundancy_pressure_mean",
    "pose_only_reference_registered_count",
    "pose_only_reference_selected_query_count",
    "pose_only_reference_selected_total",
    "pose_only_reference_used_for_pnp_count",
    "pose_only_reference_used_for_miniba_count",
    "pose_only_reference_pool_size_final",
    "long_stream_low_growth_count",
    "high_recent_growth_guard_count",
    "branch",
    "commit",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception:
        return ""


def command_for(source: str, model_dir: Path, max_frames: int) -> list[str]:
    return [
        str(PYTHON),
        "train.py",
        "-s",
        source,
        "-m",
        str(model_dir.relative_to(REPO)),
        "--test_hold",
        "10",
        "--test_frequency",
        "20",
        "--max_frames",
        str(max_frames),
        "--paper_aligned_contract_trace_path",
        str((model_dir / "semantic_trace.json").relative_to(REPO)),
        "--risk_admission_mode",
        "on_the_fly_innovation_v1",
        "--paper_aligned_recovery_commit_bridge",
        "true_source_commit",
        "--paper_aligned_defer_recovery_support_bridge",
        "v1",
        "--paper_aligned_recovery_commit_control",
        "off",
        "--paper_aligned_direct_density_control",
        MODE,
        "--paper_aligned_direct_update_prev_desc_on_hold",
        "light",
    ]


def parse_train_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"num anchors:\s*(?P<anchors>\d+),\s*num keyframes:\s*(?P<keyframes>\d+),\s*"
        r"time:\s*(?P<time>[-+0-9.]+),\s*FPS:\s*(?P<FPS>[-+0-9.]+),\s*"
        r"PSNR:\s*(?P<PSNR>[-+0-9.]+),\s*SSIM:\s*(?P<SSIM>[-+0-9.]+),\s*"
        r"LPIPS:\s*(?P<LPIPS>[-+0-9.]+),\s*R\S*:\s*(?P<R_deg>[-+0-9.]+),\s*"
        r"t:\s*(?P<t>[-+0-9.]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    match = matches[-1]
    row: dict[str, Any] = {
        "num anchors": int(match.group("anchors")),
        "num keyframes": int(match.group("keyframes")),
        "keyframes": int(match.group("keyframes")),
    }
    for key in ["time", "FPS", "PSNR", "SSIM", "LPIPS", "R_deg", "t"]:
        row[key] = float(match.group(key))
    row["time_sec"] = row["time"]
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
    if "num keyframes" in row:
        row["keyframes"] = row["num keyframes"]
    if "time" in row:
        row["time_sec"] = row["time"]
    return row


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def trace_stats(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "semantic_trace.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        trace = json.load(f)
    events = trace.get("direct_density_control_events", [])
    pool_events = trace.get("pose_reference_pool_events", [])
    pool_summary = trace.get("pose_only_reference_pool_summary", {}) or {}
    marginal: list[float] = []
    pressure: list[float] = []
    for event in events:
        try:
            marginal.append(float(event.get("active_memory_marginal_value", 0.0)))
            pressure.append(float(event.get("active_memory_redundancy_pressure", 0.0)))
        except Exception:
            pass
    return {
        "direct_density_event_count": len(events),
        "value_hold_count": sum(bool(e.get("hold_low_representation_value")) for e in events),
        "pose_only_tracking_hold_count": sum(
            bool(e.get("held_frame_used_for_tracking_bridge")) for e in events
        ),
        "active_memory_context_count": sum(bool(e.get("active_memory_context")) for e in events),
        "active_memory_tracking_only_count": sum(
            e.get("active_memory_frame_role") == "tracking_only" for e in events
        ),
        "active_memory_representation_count": sum(
            e.get("active_memory_frame_role") == "representation" for e in events
        ),
        "active_memory_marginal_value_mean": _mean(marginal),
        "active_memory_redundancy_pressure_mean": _mean(pressure),
        "pose_only_reference_registered_count": sum(
            e.get("event_type") == "pose_only_register" and bool(e.get("registered"))
            for e in pool_events
        ),
        "pose_only_reference_selected_query_count": sum(
            e.get("event_type") == "pose_only_select_summary"
            and int(e.get("pose_only_selected_count", 0) or 0) > 0
            for e in pool_events
        ),
        "pose_only_reference_selected_total": sum(
            int(e.get("pose_only_selected_count", 0) or 0)
            for e in pool_events
            if e.get("event_type") == "pose_only_select_summary"
        ),
        "pose_only_reference_used_for_pnp_count": sum(
            e.get("reference_commit_origin") == "pose_only_reference"
            and bool(e.get("reference_used_for_pnp"))
            for e in pool_events
        ),
        "pose_only_reference_used_for_miniba_count": sum(
            e.get("reference_commit_origin") == "pose_only_reference"
            and bool(e.get("reference_used_for_miniba"))
            for e in pool_events
        ),
        "pose_only_reference_pool_size_final": int(pool_summary.get("pool_size", 0) or 0),
        "long_stream_low_growth_count": sum(
            bool(e.get("long_stream_low_growth_context")) for e in events
        ),
        "high_recent_growth_guard_count": sum(
            bool(e.get("high_recent_growth_representation_guard")) for e in events
        ),
    }


def summarize_existing(dataset: str, max_frames: int, model_dir: Path, log_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "returncode": 0,
        "max_frames": max_frames,
        "model_dir": str(model_dir.relative_to(REPO)),
        "log_path": str(log_path.relative_to(REPO)),
        "elapsed_wall_sec": "",
    }
    row.update(parse_train_log(log_path))
    row.update(load_metadata(model_dir))
    row.update(trace_stats(model_dir))
    row["branch"] = git_value(["branch", "--show-current"])
    row["commit"] = git_value(["rev-parse", "--short", "HEAD"])
    return row


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    trace_rows = [
        {
            key: row.get(key, "")
            for key in [
                "dataset",
                "direct_density_event_count",
                "value_hold_count",
                "pose_only_tracking_hold_count",
                "active_memory_context_count",
                "active_memory_tracking_only_count",
                "active_memory_representation_count",
                "active_memory_marginal_value_mean",
                "active_memory_redundancy_pressure_mean",
                "pose_only_reference_registered_count",
                "pose_only_reference_selected_query_count",
                "pose_only_reference_selected_total",
                "pose_only_reference_used_for_pnp_count",
                "pose_only_reference_used_for_miniba_count",
                "pose_only_reference_pool_size_final",
                "long_stream_low_growth_count",
                "high_recent_growth_guard_count",
            ]
        }
        for row in rows
    ]
    with TRACE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace_rows)
    TRACE_JSON.write_text(json.dumps(trace_rows, indent=2), encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    branch = git_value(["branch", "--show-current"])
    commit = git_value(["rev-parse", "--short", "HEAD"])
    log(f"RUN_START branch={branch} commit={commit} mode={MODE}")
    for dataset, source, max_frames in DATASETS:
        model_dir = OUT_ROOT / dataset / "model"
        log_path = model_dir / "train.log"
        if log_path.exists() and (model_dir / "metadata.json").exists():
            log(f"SKIP {dataset} already complete")
            rows.append(summarize_existing(dataset, max_frames, model_dir, log_path))
            write_outputs(rows)
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for(source, model_dir, max_frames)
        (model_dir / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
        log(f"START {dataset} max_frames={max_frames}")
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log_file, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        row = summarize_existing(dataset, max_frames, model_dir, log_path)
        row["returncode"] = proc.returncode
        row["elapsed_wall_sec"] = elapsed
        rows.append(row)
        write_outputs(rows)
        log(f"DONE {dataset} rc={proc.returncode} time={row.get('time_sec')} wall={elapsed:.1f}")
        if proc.returncode != 0:
            return proc.returncode
    log("ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
