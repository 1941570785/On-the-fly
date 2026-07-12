#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REPO = Path("/data2/zxd/3D_Reconstruction/On_the_fly_main_true_baseline_20260703")
DEFAULT_PYTHON = Path("/home/zxd/miniconda3/envs/otf/bin/python")
RESULTS_PARENT = ROOT / "results" / "BRANCH_EXPERIMENTS_20260703"


@dataclass(frozen=True)
class SceneSpec:
    name: str
    dataset: str
    source: Path
    test_hold: int


@dataclass(frozen=True)
class ExperimentSpec:
    variant: str
    scene: SceneSpec
    repo: Path
    model_dir: Path
    run_dir: Path


PADDED_ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly_padded_datasets")
SCENES: dict[str, SceneSpec] = {
    "bonsai": SceneSpec("bonsai", "MipNeRF360", PADDED_ROOT / "MipNerf360/bonsai", 8),
    "counter": SceneSpec("counter", "MipNeRF360", PADDED_ROOT / "MipNerf360/counter", 8),
    "garden": SceneSpec("garden", "MipNeRF360", PADDED_ROOT / "MipNerf360/garden", 8),
    "forest1": SceneSpec("forest1", "StaticHikes", PADDED_ROOT / "StaticHikes/forest1", 10),
    "forest2": SceneSpec("forest2", "StaticHikes", PADDED_ROOT / "StaticHikes/forest2", 10),
    "university2": SceneSpec("university2", "StaticHikes", PADDED_ROOT / "StaticHikes/university2", 10),
    "desk": SceneSpec("desk", "TUM", PADDED_ROOT / "TUM/rgbd_dataset_freiburg1_desk", 30),
    "xyz": SceneSpec("xyz", "TUM", PADDED_ROOT / "TUM/rgbd_dataset_freiburg2_xyz", 30),
    "long_office": SceneSpec(
        "long_office",
        "TUM",
        PADDED_ROOT / "TUM/rgbd_dataset_freiburg3_long_office_household",
        30,
    ),
}


V31_PROFILE_ARGS = (
    "--risk_admission_mode",
    "on_the_fly_innovation_v1",
    "--paper_aligned_pose_render_assimilation_profile",
    "baseline_render_lock_intra_frame_v31",
)

A_MODULE_ARGS = (
    "--pose_initialization_risk_mode",
    "observe_v1",
    "--pose_risk_utility_admission_mode",
    "pose_quarantine_v1",
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
    "0",
    "--pose_risk_utility_review_min_coverage",
    "0.15",
    "--pose_risk_utility_review_max_rotation_deg",
    "1.5",
    "--pose_risk_utility_review_max_translation",
    "0.05",
)

VARIANT_REPOS = {
    "baseline": BASELINE_REPO,
    "v31": ROOT,
    "v31_a": ROOT,
}


def validate_single_gpu(value: str) -> str:
    devices = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(devices) != 1:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one device for this benchmark"
        )
    return devices[0]


def build_specs(
    output_root: Path,
    variants: list[str],
    scene_names: list[str],
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for variant in variants:
        if variant not in VARIANT_REPOS:
            raise ValueError(f"Unknown variant: {variant}")
        for scene_name in scene_names:
            if scene_name not in SCENES:
                raise ValueError(f"Unknown scene: {scene_name}")
            run_dir = output_root / variant / scene_name
            specs.append(
                ExperimentSpec(
                    variant=variant,
                    scene=SCENES[scene_name],
                    repo=VARIANT_REPOS[variant],
                    model_dir=run_dir / "model",
                    run_dir=run_dir,
                )
            )
    return specs


def build_command(spec: ExperimentSpec, *, python: Path = DEFAULT_PYTHON) -> list[str]:
    command = [
        str(python),
        "train.py",
        "-s",
        str(spec.scene.source),
        "-m",
        str(spec.model_dir),
        "--viewer_mode",
        "none",
        "--test_hold",
        str(spec.scene.test_hold),
        "--test_frequency",
        "-1",
    ]
    if spec.variant == "baseline":
        command.append("--eval_poses")
    elif spec.variant == "v31":
        command.extend(
            [
                *V31_PROFILE_ARGS,
                "--pose_initialization_risk_mode",
                "off",
                "--pose_risk_utility_admission_mode",
                "off",
            ]
        )
    elif spec.variant == "v31_a":
        command.extend([*V31_PROFILE_ARGS, *A_MODULE_ARGS])
    else:
        raise ValueError(f"Unknown variant: {spec.variant}")
    return command


def git_commit(repo: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def preflight(specs: list[ExperimentSpec], python: Path) -> None:
    if not python.exists():
        raise FileNotFoundError(python)
    for spec in specs:
        if not spec.scene.source.is_dir():
            raise FileNotFoundError(spec.scene.source)
        train_path = spec.repo / "train.py"
        if not train_path.exists():
            raise FileNotFoundError(train_path)


def read_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def status_path(spec: ExperimentSpec) -> Path:
    return spec.run_dir / "run_status.json"


def is_complete(spec: ExperimentSpec) -> bool:
    path = status_path(spec)
    if not path.exists() or not (spec.model_dir / "metadata.json").exists():
        return False
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(status.get("returncode", 1)) == 0


def summarize(spec: ExperimentSpec, returncode: int, skipped: bool = False) -> dict[str, Any]:
    metadata = read_metadata(spec.model_dir)
    return {
        "variant": spec.variant,
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "test_hold": spec.scene.test_hold,
        "returncode": int(returncode),
        "skipped": bool(skipped),
        "PSNR": metadata.get("PSNR"),
        "SSIM": metadata.get("SSIM"),
        "LPIPS": metadata.get("LPIPS"),
        "time": metadata.get("time"),
        "num_keyframes": metadata.get("num keyframes", metadata.get("num_keyframes")),
        "model_dir": str(spec.model_dir),
        "log_path": str(spec.run_dir / "train.log"),
    }


def _safe_remove_model(spec: ExperimentSpec, output_root: Path) -> None:
    if not spec.model_dir.exists():
        return
    resolved_model = spec.model_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_model == resolved_root or resolved_root not in resolved_model.parents:
        raise ValueError(f"Refusing to remove model outside output root: {resolved_model}")
    shutil.rmtree(resolved_model)


def run_one(
    spec: ExperimentSpec,
    *,
    output_root: Path,
    python: Path,
    skip_existing: bool,
    dry_run: bool,
    gpu: str,
) -> dict[str, Any]:
    command = build_command(spec, python=python)
    print(f"[{spec.variant}/{spec.scene.name}] {' '.join(command)}", flush=True)
    if dry_run:
        return summarize(spec, 0, skipped=True)
    if skip_existing and is_complete(spec):
        print(f"[{spec.variant}/{spec.scene.name}] existing complete run", flush=True)
        return summarize(spec, 0, skipped=True)
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    _safe_remove_model(spec, output_root)
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "variant": spec.variant,
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "test_hold": spec.scene.test_hold,
        "source": str(spec.scene.source),
        "repo": str(spec.repo),
        "repo_commit": git_commit(spec.repo),
        "cuda_visible_devices": gpu,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (spec.run_dir / "command.json").write_text(
        json.dumps(context, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    started = time.time()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    with (spec.run_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=spec.repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {
        **context,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_time_seconds": time.time() - started,
        "returncode": int(process.returncode),
        "metadata_exists": (spec.model_dir / "metadata.json").exists(),
    }
    status_path(spec).write_text(
        json.dumps(status, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    row = summarize(spec, process.returncode)
    print(
        f"[{spec.variant}/{spec.scene.name}] returncode={row['returncode']} "
        f"PSNR={row['PSNR']} SSIM={row['SSIM']} LPIPS={row['LPIPS']}",
        flush=True,
    )
    return row


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0]) if rows else []
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANT_REPOS), default=[])
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--repeat_index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gpu = validate_single_gpu(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or RESULTS_PARENT / f"official_pose_full9_{timestamp}"
    if args.repeat_index > 0 and args.output_root is None:
        output_root = RESULTS_PARENT / f"official_pose_repeat{args.repeat_index}_{timestamp}"
    variants = list(args.variants) if args.variants else list(VARIANT_REPOS)
    scene_names = list(args.scenes) if args.scenes else list(SCENES)
    specs = build_specs(output_root, variants, scene_names)
    preflight(specs, args.python)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_one(
            spec,
            output_root=output_root,
            python=args.python,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            gpu=gpu,
        )
        rows.append(row)
        if not args.dry_run:
            write_manifest(output_root, rows)
        if int(row["returncode"]) != 0:
            return int(row["returncode"]) or 1
    if args.dry_run:
        print(f"dry-run jobs={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
