#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_pose_verification_a_ablation import (
    A_SHARED_ARGS,
    DEFAULT_PYTHON,
    SCENES,
    V31_PROFILE_ARGS,
    SceneSpec,
)


K16_ARGS = (
    "--paper_aligned_pose_render_extra_optimization_fraction",
    "0.51666666666666672",
    "--paper_aligned_pose_render_extra_optimization_max_extra",
    "16",
)


@dataclass(frozen=True)
class Preset:
    pose_mode: str
    extra_args: tuple[str, ...] = ()


PRESETS: dict[str, Preset] = {
    "a_off": Preset("off"),
    "a_observe": Preset("observe_v1"),
    "a_isolate": Preset("isolate_v1"),
    "a_isolate_conservative": Preset(
        "isolate_v1",
        (
            "--pose_initialization_risk_absolute_threshold",
            "0.12",
            "--pose_initialization_risk_cooldown_frames",
            "20",
        ),
    ),
    "a_isolate_severe": Preset(
        "isolate_v1",
        (
            "--pose_initialization_risk_absolute_threshold",
            "0.15",
            "--pose_initialization_risk_cooldown_frames",
            "20",
        ),
    ),
    "a_utility_observe": Preset(
        "observe_v1",
        (
            "--pose_risk_utility_admission_mode",
            "observe_v1",
        ),
    ),
    "a_utility_observe_verify": Preset(
        "observe_v1",
        (
            "--pose_risk_utility_admission_mode",
            "observe_v1",
            "--pose_risk_utility_use_verification_candidates",
        ),
    ),
    "a_quarantine": Preset(
        "observe_v1",
        (
            "--pose_risk_utility_admission_mode",
            "pose_quarantine_v1",
        ),
    ),
    "a_quarantine_utility": Preset(
        "observe_v1",
        (
            "--pose_risk_utility_admission_mode",
            "pose_quarantine_utility_v1",
            "--pose_risk_utility_quarantine_risk_margin",
            "0.04",
        ),
    ),
    "a_quarantine_severe": Preset(
        "observe_v1",
        (
            "--pose_risk_utility_admission_mode",
            "pose_quarantine_severe_v1",
            "--pose_risk_utility_use_verification_candidates",
            "--pose_risk_utility_quarantine_cooldown_frames",
            "12",
        ),
    ),
    "a_v2": Preset("verify_v2"),
    "photo_i2_g001": Preset(
        "verify_v2",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_iterations",
            "2",
            "--pose_verification_photometric_min_relative_improvement",
            "0.001",
            "--pose_verification_photometric_min_support_ratio",
            "0.95",
        ),
    ),
    "photo_i4_g002": Preset(
        "verify_v2",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_iterations",
            "4",
            "--pose_verification_photometric_min_relative_improvement",
            "0.002",
            "--pose_verification_photometric_min_support_ratio",
            "0.95",
        ),
    ),
    "photo_i4_g005": Preset(
        "verify_v2",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_iterations",
            "4",
            "--pose_verification_photometric_min_relative_improvement",
            "0.005",
            "--pose_verification_photometric_min_support_ratio",
            "0.98",
        ),
    ),
    "photo_test_i2_g001": Preset(
        "verify_v2",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_scope",
            "test_only",
            "--pose_verification_photometric_iterations",
            "2",
            "--pose_verification_photometric_min_relative_improvement",
            "0.001",
            "--pose_verification_photometric_min_support_ratio",
            "0.95",
        ),
    ),
    "photo_test_i4_g002": Preset(
        "verify_v2",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_scope",
            "test_only",
            "--pose_verification_photometric_iterations",
            "4",
            "--pose_verification_photometric_min_relative_improvement",
            "0.002",
            "--pose_verification_photometric_min_support_ratio",
            "0.95",
        ),
    ),
    "photo_raw_i4_g002": Preset(
        "observe_v1",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_seed",
            "raw",
            "--pose_verification_photometric_iterations",
            "4",
            "--pose_verification_photometric_min_relative_improvement",
            "0.002",
            "--pose_verification_photometric_min_support_ratio",
            "0.95",
        ),
    ),
    "photo_raw_test_i4_g002": Preset(
        "observe_v1",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_seed",
            "raw",
            "--pose_verification_photometric_scope",
            "test_only",
            "--pose_verification_photometric_iterations",
            "4",
            "--pose_verification_photometric_min_relative_improvement",
            "0.002",
            "--pose_verification_photometric_min_support_ratio",
            "0.95",
        ),
    ),
    "photo_raw_i8_lr5_g005": Preset(
        "observe_v1",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_seed",
            "raw",
            "--pose_verification_photometric_iterations",
            "8",
            "--pose_verification_photometric_lr_scale",
            "5.0",
            "--pose_verification_photometric_min_relative_improvement",
            "0.005",
            "--pose_verification_photometric_min_support_ratio",
            "0.98",
        ),
    ),
    "photo_raw_test_i8_lr5_g005": Preset(
        "observe_v1",
        (
            "--pose_verification_photometric_review",
            "--pose_verification_photometric_seed",
            "raw",
            "--pose_verification_photometric_scope",
            "test_only",
            "--pose_verification_photometric_iterations",
            "8",
            "--pose_verification_photometric_lr_scale",
            "5.0",
            "--pose_verification_photometric_min_relative_improvement",
            "0.005",
            "--pose_verification_photometric_min_support_ratio",
            "0.98",
        ),
    ),
}

PHOTOMETRIC_SAFETY_ARGS = (
    "--pose_verification_photometric_min_coverage",
    "0.15",
    "--pose_verification_photometric_max_rotation_deg",
    "0.5",
    "--pose_verification_photometric_max_translation",
    "0.01",
)


@dataclass(frozen=True)
class ExperimentSpec:
    repeat: int
    preset: str
    scene: SceneSpec
    run_dir: Path
    model_dir: Path

    @property
    def job_id(self) -> str:
        return f"repeat{self.repeat:02d}:{self.scene.name}:{self.preset}"


def validate_gpus(values: Sequence[str]) -> list[str]:
    gpus = [str(value).strip() for value in values if str(value).strip()]
    if not gpus:
        raise ValueError("at least one physical GPU identifier is required")
    if len(gpus) > 4:
        raise ValueError("at most four physical GPUs are allowed")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU identifiers must be unique")
    return gpus


def _rotated(values: list[str], repeat: int) -> list[str]:
    offset = (int(repeat) - 1) % max(len(values), 1)
    return values[offset:] + values[:offset]


def build_specs(
    *,
    output_root: Path,
    presets: Sequence[str],
    scene_names: Sequence[str],
    repeats: Sequence[int],
) -> list[ExperimentSpec]:
    preset_names = [str(value) for value in presets]
    scenes = [str(value) for value in scene_names]
    repeat_values = [int(value) for value in repeats]
    unknown_presets = set(preset_names) - set(PRESETS)
    unknown_scenes = set(scenes) - set(SCENES)
    if unknown_presets:
        raise ValueError(f"unknown presets: {sorted(unknown_presets)}")
    if unknown_scenes:
        raise ValueError(f"unknown scenes: {sorted(unknown_scenes)}")
    if not preset_names or not scenes or not repeat_values:
        raise ValueError("presets, scenes, and repeats must be non-empty")
    if len(set(preset_names)) != len(preset_names):
        raise ValueError("presets must be unique")
    if len(set(scenes)) != len(scenes):
        raise ValueError("scenes must be unique")
    if len(set(repeat_values)) != len(repeat_values) or min(repeat_values) <= 0:
        raise ValueError("repeat indices must be unique positive integers")

    specs: list[ExperimentSpec] = []
    for repeat in repeat_values:
        for scene_name in scenes:
            for preset_name in _rotated(preset_names, repeat):
                run_dir = (
                    Path(output_root)
                    / f"repeat_{repeat:02d}"
                    / preset_name
                    / scene_name
                )
                specs.append(
                    ExperimentSpec(
                        repeat=repeat,
                        preset=preset_name,
                        scene=SCENES[scene_name],
                        run_dir=run_dir,
                        model_dir=run_dir / "model",
                    )
                )
    return specs


def paired_gpu_assignments(
    specs: Sequence[ExperimentSpec],
    gpus: Sequence[str],
) -> dict[str, str]:
    physical_gpus = validate_gpus(gpus)
    group_to_gpu: dict[tuple[int, str], str] = {}
    assignments: dict[str, str] = {}
    for spec in specs:
        group = (int(spec.repeat), str(spec.scene.name))
        if group not in group_to_gpu:
            group_to_gpu[group] = physical_gpus[
                len(group_to_gpu) % len(physical_gpus)
            ]
        assignments[spec.job_id] = group_to_gpu[group]
    return assignments


def build_command(
    spec: ExperimentSpec,
    *,
    python: Path = DEFAULT_PYTHON,
) -> list[str]:
    preset = PRESETS[spec.preset]
    command = [
        str(python),
        "train.py",
        "-s",
        str(spec.scene.source),
        "-m",
        str(spec.model_dir),
        "--viewer_mode",
        "none",
        "--enable_reboot",
        "--test_hold",
        str(spec.scene.test_hold),
        "--test_frequency",
        "-1",
        *V31_PROFILE_ARGS,
        "--pose_initialization_risk_mode",
        preset.pose_mode,
        *A_SHARED_ARGS,
        "--pose_verification_v2_min_improvement",
        "0.0",
        *K16_ARGS,
    ]
    if "--pose_verification_photometric_review" in preset.extra_args:
        command.extend(PHOTOMETRIC_SAFETY_ARGS)
    if preset.extra_args:
        command.extend(preset.extra_args)
    return command


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_complete(spec: ExperimentSpec) -> bool:
    status = _read_json(spec.run_dir / "run_status.json")
    return bool(
        int(status.get("returncode", 1)) == 0
        and (spec.model_dir / "metadata.json").is_file()
    )


def _safe_reset_model(spec: ExperimentSpec, output_root: Path) -> None:
    if not spec.model_dir.exists():
        return
    model = spec.model_dir.resolve()
    root = Path(output_root).resolve()
    if model == root or root not in model.parents:
        raise ValueError(f"refusing to remove model outside output root: {model}")
    shutil.rmtree(model)


def _git_context(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "repo_commit": head.stdout.strip() if head.returncode == 0 else "",
        "repo_dirty": bool(status.stdout.strip()) if status.returncode == 0 else True,
        "git_status": status.stdout.splitlines() if status.returncode == 0 else [],
        "git_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
    }


def _job_summary(
    spec: ExperimentSpec,
    *,
    gpu: str,
    returncode: int,
    wall_time_seconds: float,
    skipped: bool = False,
) -> dict[str, Any]:
    trace = _read_json(spec.model_dir / "pose_initialization_risk_trace.json")
    events = [event for event in trace.get("events", []) if isinstance(event, dict)]
    return {
        "job_id": spec.job_id,
        "repeat": spec.repeat,
        "preset": spec.preset,
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "gpu": str(gpu),
        "returncode": int(returncode),
        "skipped": bool(skipped),
        "wall_time_seconds": float(wall_time_seconds),
        "geometric_attempts": sum(
            bool(event.get("verification_attempted", False)) for event in events
        ),
        "geometric_accepts": sum(
            bool(event.get("verification_accepted", False)) for event in events
        ),
        "photometric_attempts": sum(
            bool(event.get("photometric_verification_attempted", False))
            for event in events
        ),
        "photometric_accepts": sum(
            bool(event.get("photometric_verification_accepted", False))
            for event in events
        ),
        "model_dir": str(spec.model_dir),
        "log_path": str(spec.run_dir / "train.log"),
    }


def _write_manifest(output_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: str(row["job_id"]))
    _write_json_atomic(output_root / "manifest.json", ordered)
    if not ordered:
        return
    fields = list(ordered[0])
    temporary = output_root / "manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(output_root / "manifest.csv")


def run_one(
    spec: ExperimentSpec,
    *,
    root: Path,
    output_root: Path,
    python: Path,
    gpu: str,
    skip_existing: bool,
) -> dict[str, Any]:
    if skip_existing and _is_complete(spec):
        return _job_summary(
            spec,
            gpu=gpu,
            returncode=0,
            wall_time_seconds=0.0,
            skipped=True,
        )
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    _safe_reset_model(spec, output_root)
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(spec, python=python)
    context = {
        "job_id": spec.job_id,
        "gpu": str(gpu),
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json_atomic(spec.run_dir / "command.json", context)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with (spec.run_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time = time.time() - started
    _write_json_atomic(
        spec.run_dir / "run_status.json",
        {
            **context,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_time_seconds": wall_time,
            "returncode": int(process.returncode),
            "metadata_exists": (spec.model_dir / "metadata.json").is_file(),
        },
    )
    return _job_summary(
        spec,
        gpu=gpu,
        returncode=process.returncode,
        wall_time_seconds=wall_time,
    )


def run_parallel(
    specs: Sequence[ExperimentSpec],
    *,
    root: Path,
    output_root: Path,
    python: Path,
    gpus: Sequence[str],
    skip_existing: bool,
    paired_gpu_affinity: bool = False,
) -> list[dict[str, Any]]:
    shared_pending: queue.Queue[ExperimentSpec] = queue.Queue()
    pending_by_gpu: dict[str, queue.Queue[ExperimentSpec]] = {
        str(gpu): queue.Queue() for gpu in gpus
    }
    if paired_gpu_affinity:
        assignments = paired_gpu_assignments(specs, gpus)
        for spec in specs:
            pending_by_gpu[assignments[spec.job_id]].put(spec)
    else:
        for spec in specs:
            shared_pending.put(spec)
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        pending = pending_by_gpu[str(gpu)] if paired_gpu_affinity else shared_pending
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            try:
                row = run_one(
                    spec,
                    root=root,
                    output_root=output_root,
                    python=python,
                    gpu=gpu,
                    skip_existing=skip_existing,
                )
            except Exception as error:
                row = _job_summary(
                    spec,
                    gpu=gpu,
                    returncode=1,
                    wall_time_seconds=0.0,
                )
                row["runner_error"] = repr(error)
            with lock:
                rows.append(row)
                _write_manifest(output_root, rows)
                print(
                    f"[{row['job_id']} gpu={gpu}] returncode={row['returncode']} "
                    f"photo={row['photometric_accepts']}/{row['photometric_attempts']}",
                    flush=True,
                )
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    return rows


def _gpu_inventory() -> list[str]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in process.stdout.splitlines() if line.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--presets", nargs="+", choices=tuple(PRESETS), required=True)
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES), required=True)
    parser.add_argument("--repeats", nargs="+", type=int, required=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--paired_gpu_affinity",
        action="store_true",
        help="Keep every preset for one scene/repeat on the same physical GPU.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    gpus = validate_gpus(args.gpus)
    missing_gpus = set(gpus) - set(_gpu_inventory())
    if missing_gpus:
        raise ValueError(f"unknown GPU identifiers: {sorted(missing_gpus)}")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    specs = build_specs(
        output_root=args.output_root,
        presets=args.presets,
        scene_names=args.scenes,
        repeats=args.repeats,
    )
    for spec in specs:
        if not spec.scene.source.is_dir():
            raise FileNotFoundError(spec.scene.source)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        args.output_root / "experiment_context.json",
        {
            "root": str(root),
            "output_root": str(args.output_root),
            "python": str(args.python),
            "gpus": gpus,
            "presets": list(args.presets),
            "scenes": list(args.scenes),
            "repeats": list(args.repeats),
            "paired_gpu_affinity": bool(args.paired_gpu_affinity),
            "job_count": len(specs),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **_git_context(root),
        },
    )
    rows = run_parallel(
        specs,
        root=root,
        output_root=args.output_root,
        python=args.python,
        gpus=gpus,
        skip_existing=bool(args.skip_existing),
        paired_gpu_affinity=bool(args.paired_gpu_affinity),
    )
    failures = [row for row in rows if int(row["returncode"]) != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
