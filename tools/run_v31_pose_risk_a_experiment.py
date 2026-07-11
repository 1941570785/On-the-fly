#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SCENE_SOURCES = {
    "bonsai": "/data2/zxd/3D_Reconstruction/On_the_fly_padded_datasets/MipNerf360/bonsai",
    "forest1": "/data2/zxd/3D_Reconstruction/On_the_fly_padded_datasets/StaticHikes/forest1",
}

V31_ARGS = (
    "--risk_admission_mode",
    "on_the_fly_innovation_v1",
    "--paper_aligned_pose_render_assimilation_profile",
    "baseline_render_lock_intra_frame_v31",
)

A_CONFIG_ARGS = (
    "--pose_initialization_risk_absolute_threshold",
    "0.10",
    "--pose_initialization_risk_adaptive_sigma",
    "2.0",
    "--pose_initialization_risk_warmup",
    "8",
    "--pose_initialization_risk_history_size",
    "64",
    "--pose_initialization_risk_cooldown_frames",
    "12",
)

VARIANTS: list[tuple[str, tuple[str, ...]]] = [
    (
        "V31_control",
        (*V31_ARGS, "--pose_initialization_risk_mode", "off"),
    ),
    (
        "V31_A_observe",
        (
            *V31_ARGS,
            "--pose_initialization_risk_mode",
            "observe_v1",
            *A_CONFIG_ARGS,
        ),
    ),
    (
        "V31_A_isolate",
        (
            *V31_ARGS,
            "--pose_initialization_risk_mode",
            "isolate_v1",
            *A_CONFIG_ARGS,
        ),
    ),
]

SUMMARY_FIELDS = [
    "scene",
    "variant",
    "returncode",
    "PSNR",
    "SSIM",
    "LPIPS",
    "time",
    "num anchors",
    "num keyframes",
    "R_deg",
    "t",
    "delta_psnr_to_control",
    "delta_ssim_to_control",
    "delta_lpips_to_control",
    "delta_time_to_control",
    "delta_psnr_to_observe",
    "delta_ssim_to_observe",
    "delta_lpips_to_observe",
    "delta_time_to_observe",
    "risk_events",
    "risk_eligible",
    "risk_isolated",
    "risk_isolated_ratio",
    "risk_score_mean",
    "pose_uncertainty_mean",
    "state_support_gap_mean",
    "temporal_degradation_mean",
    "model_dir",
    "log_path",
]


@dataclass(frozen=True)
class ExperimentSpec:
    scene: str
    variant: str
    source: str
    model_dir: Path
    extra_args: tuple[str, ...]


def build_specs(output_root: Path, scenes: list[str]) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            scene=scene,
            variant=variant,
            source=SCENE_SOURCES[scene],
            model_dir=output_root / scene / variant / "model",
            extra_args=extra_args,
        )
        for scene in scenes
        for variant, extra_args in VARIANTS
    ]


def build_command(spec: ExperimentSpec, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "-s",
        spec.source,
        "-m",
        str(spec.model_dir),
        "--test_hold",
        str(args.test_hold),
        "--test_frequency",
        str(args.test_frequency),
        *spec.extra_args,
    ]
    if int(args.max_frames) > 0:
        command.extend(["--max_frames", str(int(args.max_frames))])
    return command


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _frame_pose_means(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    totals = {"R_deg": 0.0, "t": 0.0}
    counts = {"R_deg": 0, "t": 0}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for output_key, csv_key in (
                ("R_deg", "abs_rot_error_deg"),
                ("t", "abs_trans_error"),
            ):
                raw = row.get(csv_key, "")
                if raw == "":
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                totals[output_key] += value
                counts[output_key] += 1
    return {
        key: totals[key] / counts[key]
        for key in totals
        if counts[key] > 0
    }


def summarize_run(spec: ExperimentSpec, returncode: int) -> dict[str, Any]:
    metadata = _load_json(spec.model_dir / "metadata.json")
    pose_means = _frame_pose_means(spec.model_dir / "frame_metrics.csv")
    risk_trace = _load_json(
        spec.model_dir / "pose_initialization_risk_trace.json"
    )
    risk_summary = risk_trace.get("summary", {})
    if not isinstance(risk_summary, dict):
        risk_summary = {}
    return {
        "scene": spec.scene,
        "variant": spec.variant,
        "returncode": int(returncode),
        "PSNR": metadata.get("PSNR", ""),
        "SSIM": metadata.get("SSIM", ""),
        "LPIPS": metadata.get("LPIPS", ""),
        "time": metadata.get("time", ""),
        "num anchors": metadata.get("num anchors", ""),
        "num keyframes": metadata.get("num keyframes", ""),
        "R_deg": metadata.get("R_deg", metadata.get("R°", pose_means.get("R_deg", ""))),
        "t": metadata.get("t", pose_means.get("t", "")),
        "risk_events": risk_summary.get("events", 0),
        "risk_eligible": risk_summary.get("eligible", 0),
        "risk_isolated": risk_summary.get("isolated", 0),
        "risk_isolated_ratio": risk_summary.get("isolated_ratio", 0.0),
        "risk_score_mean": risk_summary.get("risk_score_mean", 0.0),
        "pose_uncertainty_mean": risk_summary.get("pose_uncertainty_mean", 0.0),
        "state_support_gap_mean": risk_summary.get("state_support_gap_mean", 0.0),
        "temporal_degradation_mean": risk_summary.get("temporal_degradation_mean", 0.0),
        "model_dir": str(spec.model_dir),
        "log_path": str(spec.model_dir / "train.log"),
    }


def _delta(value: Any, reference: Any) -> float | str:
    try:
        return float(value) - float(reference)
    except (TypeError, ValueError):
        return ""


def attach_control_deltas(rows: list[dict[str, Any]]) -> None:
    controls = {
        row["scene"]: row
        for row in rows
        if row.get("variant") == "V31_control"
    }
    observes = {
        row["scene"]: row
        for row in rows
        if row.get("variant") == "V31_A_observe"
    }
    for row in rows:
        control = controls.get(row["scene"], {})
        observe = observes.get(row["scene"], {})
        row["delta_psnr_to_control"] = _delta(row.get("PSNR"), control.get("PSNR"))
        row["delta_ssim_to_control"] = _delta(row.get("SSIM"), control.get("SSIM"))
        row["delta_lpips_to_control"] = _delta(row.get("LPIPS"), control.get("LPIPS"))
        row["delta_time_to_control"] = _delta(row.get("time"), control.get("time"))
        row["delta_psnr_to_observe"] = _delta(row.get("PSNR"), observe.get("PSNR"))
        row["delta_ssim_to_observe"] = _delta(row.get("SSIM"), observe.get("SSIM"))
        row["delta_lpips_to_observe"] = _delta(row.get("LPIPS"), observe.get("LPIPS"))
        row["delta_time_to_observe"] = _delta(row.get("time"), observe.get("time"))


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    attach_control_deltas(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def run_one(spec: ExperimentSpec, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_existing and (spec.model_dir / "metadata.json").exists():
        return summarize_run(spec, 0)
    if spec.model_dir.exists() and not args.skip_existing:
        shutil.rmtree(spec.model_dir)
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(spec, args)
    (spec.model_dir / "command.json").write_text(
        json.dumps(command, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        return summarize_run(spec, 0)
    with (spec.model_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return summarize_run(spec, process.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="")
    parser.add_argument("--only_scene", nargs="*", choices=sorted(SCENE_SOURCES), default=[])
    parser.add_argument("--only_variant", nargs="*", default=[])
    parser.add_argument("--test_hold", type=int, default=8)
    parser.add_argument("--test_frequency", type=int, default=-1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else (
        ROOT
        / "results"
        / "BRANCH_EXPERIMENTS_20260703"
        / f"v31_pose_risk_a_bonsai_forest1_{timestamp}"
    )
    scenes = list(args.only_scene) if args.only_scene else ["bonsai", "forest1"]
    specs = build_specs(output_root, scenes)
    if args.only_variant:
        requested = set(args.only_variant)
        specs = [spec for spec in specs if spec.variant in requested]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[{spec.scene}/{spec.variant}] {' '.join(build_command(spec, args))}", flush=True)
        row = run_one(spec, args)
        rows.append(row)
        write_summary(output_root, rows)
        print(
            f"[{spec.scene}/{spec.variant}] returncode={row['returncode']} "
            f"PSNR={row['PSNR']} SSIM={row['SSIM']} LPIPS={row['LPIPS']} "
            f"isolated={row['risk_isolated']}",
            flush=True,
        )
    return 0 if all(int(row["returncode"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
