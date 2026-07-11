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
    "--pose_initialization_risk_mode",
    "observe_v1",
)

JOINT_CONFIG_ARGS = (
    "--pose_initialization_risk_absolute_threshold",
    "0.10",
    "--pose_initialization_risk_adaptive_sigma",
    "2.0",
    "--pose_initialization_risk_warmup",
    "8",
    "--pose_initialization_risk_history_size",
    "64",
    "--pose_risk_utility_threshold",
    "0.24",
    "--pose_risk_utility_selectivity_reference",
    "1.8",
    "--pose_risk_utility_probe_downsample",
    "4",
    "--pose_risk_utility_isolation_risk_margin",
    "0.04",
    "--pose_risk_utility_isolation_cooldown_frames",
    "24",
    "--pose_risk_utility_quarantine_risk_margin",
    "0.08",
    "--pose_risk_utility_quarantine_cooldown_frames",
    "64",
    "--pose_risk_utility_review_iterations",
    "2",
    "--pose_risk_utility_review_min_coverage",
    "0.15",
    "--pose_risk_utility_review_max_rotation_deg",
    "1.5",
    "--pose_risk_utility_review_max_translation",
    "0.05",
)

VARIANTS: list[tuple[str, tuple[str, ...]]] = [
    (
        "V31_RU_observe",
        (
            *V31_ARGS,
            "--pose_risk_utility_admission_mode",
            "observe_v1",
            *JOINT_CONFIG_ARGS,
        ),
    ),
    (
        "V31_RU_active",
        (
            *V31_ARGS,
            "--pose_risk_utility_admission_mode",
            "active_v1",
            *JOINT_CONFIG_ARGS,
        ),
    ),
    (
        "V31_RU_active_no_review",
        (
            *V31_ARGS,
            "--pose_risk_utility_admission_mode",
            "active_v1",
            *JOINT_CONFIG_ARGS,
            "--pose_risk_utility_review_iterations",
            "0",
        ),
    ),
    (
        "V31_RU_pose_quarantine",
        (
            *V31_ARGS,
            "--pose_risk_utility_admission_mode",
            "pose_quarantine_v1",
            *JOINT_CONFIG_ARGS,
            "--pose_risk_utility_review_iterations",
            "0",
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
    "risk_candidates",
    "review_admit",
    "isolate_low_utility",
    "admit_conservative",
    "cooldown_admit",
    "quarantine_cooldown_admit",
    "pose_reference_quarantined",
    "utility_score_mean",
    "coverage_deficit_mean",
    "residual_selectivity_mean",
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def summarize_run(spec: ExperimentSpec, returncode: int) -> dict[str, Any]:
    metadata = _load_json(spec.model_dir / "metadata.json")
    trace = _load_json(spec.model_dir / "pose_risk_utility_trace.json")
    summary = trace.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
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
        "R_deg": metadata.get("R_deg", metadata.get("R°", "")),
        "t": metadata.get("t", ""),
        "risk_candidates": summary.get("risk_candidates", 0),
        "review_admit": summary.get("review_admit", 0),
        "isolate_low_utility": summary.get("isolate_low_utility", 0),
        "admit_conservative": summary.get("admit_conservative", 0),
        "cooldown_admit": summary.get("cooldown_admit", 0),
        "pose_reference_quarantined": summary.get(
            "pose_reference_quarantined", 0
        ),
        "utility_score_mean": summary.get("utility_score_mean", 0.0),
        "coverage_deficit_mean": summary.get("coverage_deficit_mean", 0.0),
        "residual_selectivity_mean": summary.get(
            "residual_selectivity_mean", 0.0
        ),
        "model_dir": str(spec.model_dir),
        "log_path": str(spec.model_dir / "train.log"),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
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
    parser.add_argument(
        "--only_scene", nargs="*", choices=sorted(SCENE_SOURCES), default=[]
    )
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
        / f"v31_risk_utility_bonsai_forest1_{timestamp}"
    )
    scenes = list(args.only_scene) if args.only_scene else ["bonsai", "forest1"]
    specs = build_specs(output_root, scenes)
    if args.only_variant:
        requested = set(args.only_variant)
        specs = [spec for spec in specs if spec.variant in requested]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(
            f"[{spec.scene}/{spec.variant}] {' '.join(build_command(spec, args))}",
            flush=True,
        )
        row = run_one(spec, args)
        rows.append(row)
        write_summary(output_root, rows)
        print(
            f"[{spec.scene}/{spec.variant}] returncode={row['returncode']} "
            f"PSNR={row['PSNR']} SSIM={row['SSIM']} LPIPS={row['LPIPS']} "
            f"review={row['review_admit']} isolated={row['isolate_low_utility']}",
            flush=True,
        )
    return 0 if all(int(row["returncode"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
