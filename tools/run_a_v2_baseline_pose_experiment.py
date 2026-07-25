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
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_pose_verification_a_ablation import (  # noqa: E402
    A_SHARED_ARGS,
    DEFAULT_PYTHON,
    SCENES,
    V31_PROFILE_ARGS,
    SceneSpec,
)


RESULTS_PARENT = Path(
    "/data2/zxd/3D_Reconstruction/comparison_results/a_v2_baseline_pose_20260722"
)
VARIANTS: dict[str, str] = {
    "a_off": "off",
    "a_current": "verify_v1",
    "a_v2": "verify_v2",
    "a_v21": "verify_v2",
    "a_v21_strict": "verify_v2",
    "a_v22_epipolar": "verify_v2",
    "a_v21_sensitive": "verify_v2",
    "a_v21_sensitive_g005": "verify_v2",
    "a_v23_multihypothesis": "verify_v2",
    "a_v24_multiview": "verify_v2",
    "a_v25_anchor_freeze": "verify_v2",
    "a_v26_frozen_sparse": "verify_v2",
    "a_v27_frozen_verification": "verify_v2",
    "a_v28_stable_anchor": "verify_v2",
    "a_v29_reference_guard": "verify_v2",
    "a_v210_high_risk_reference_guard": "verify_v2",
    "a_v211_trigger_aligned_reference_guard": "verify_v2",
    "a_v212_temporal_safe_reference_guard": "verify_v2",
    "a_v213_deterministic_pose_sampling": "verify_v2",
    "a_v214_deterministic_opencv_registration": "verify_v2",
    "a_v215_fixed_async_pose_budget": "verify_v2",
    "a_v216_combined_reference_support": "verify_v2",
    "a_v217_coherent_frozen_geometry": "verify_v2",
    "a_v218_guarded_frozen_geometry": "verify_v2",
    "a_v219_guarded_frozen_live_pose": "verify_v2",
    "a_v220_homogeneous_frozen_geometry": "verify_v2",
    "a_v221_frozen_global_support_guard": "verify_v2",
    "a_v222_pose_safe_v18": "observe_v1",
}
K16_ARGS = (
    "--paper_aligned_pose_render_extra_optimization_fraction",
    "0.51666666666666672",
    "--paper_aligned_pose_render_extra_optimization_max_extra",
    "16",
)


@dataclass(frozen=True)
class ExperimentSpec:
    repeat: int
    variant: str
    scene: SceneSpec
    run_dir: Path
    model_dir: Path

    @property
    def job_id(self) -> str:
        return f"repeat{self.repeat:02d}:{self.scene.name}:{self.variant}"


def validate_single_gpu(value: str) -> str:
    devices = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if len(devices) != 1:
        raise ValueError("exactly one physical GPU identifier is required")
    return devices[0]


def _rotated(values: list[str], repeat: int) -> list[str]:
    if not values:
        return []
    offset = (int(repeat) - 1) % len(values)
    return values[offset:] + values[:offset]


def build_specs(
    *,
    output_root: Path,
    variants: Sequence[str],
    scene_names: Sequence[str],
    repeats: Sequence[int],
) -> list[ExperimentSpec]:
    variant_names = [str(value) for value in variants]
    scene_values = [str(value) for value in scene_names]
    repeat_values = [int(value) for value in repeats]
    if not variant_names or len(set(variant_names)) != len(variant_names):
        raise ValueError("variants must be non-empty and unique")
    if not scene_values or len(set(scene_values)) != len(scene_values):
        raise ValueError("scenes must be non-empty and unique")
    if not repeat_values or len(set(repeat_values)) != len(repeat_values):
        raise ValueError("repeat indices must be non-empty and unique")
    if any(value <= 0 for value in repeat_values):
        raise ValueError("repeat indices must be positive")
    unknown_variants = set(variant_names) - set(VARIANTS)
    unknown_scenes = set(scene_values) - set(SCENES)
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")
    if unknown_scenes:
        raise ValueError(f"unknown scenes: {sorted(unknown_scenes)}")

    specs: list[ExperimentSpec] = []
    for repeat in repeat_values:
        ordered_variants = _rotated(variant_names, repeat)
        for scene_name in scene_values:
            for variant in ordered_variants:
                run_dir = (
                    Path(output_root)
                    / f"repeat_{repeat:02d}"
                    / variant
                    / scene_name
                )
                specs.append(
                    ExperimentSpec(
                        repeat=repeat,
                        variant=variant,
                        scene=SCENES[scene_name],
                        run_dir=run_dir,
                        model_dir=run_dir / "model",
                    )
                )
    return specs


def build_command(
    spec: ExperimentSpec,
    *,
    python: Path = DEFAULT_PYTHON,
) -> list[str]:
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
        VARIANTS[spec.variant],
        *A_SHARED_ARGS,
        "--pose_verification_v2_min_improvement",
        (
            "0.005"
            if spec.variant == "a_v21_sensitive_g005"
            else (
                "0.02"
                if spec.variant
                in {
                    "a_v21_strict",
                    "a_v22_epipolar",
                    "a_v21_sensitive",
                    "a_v23_multihypothesis",
                    "a_v24_multiview",
                    "a_v25_anchor_freeze",
                    "a_v26_frozen_sparse",
                    "a_v27_frozen_verification",
                    "a_v28_stable_anchor",
                    "a_v29_reference_guard",
                    "a_v210_high_risk_reference_guard",
                    "a_v211_trigger_aligned_reference_guard",
                    "a_v212_temporal_safe_reference_guard",
                    "a_v213_deterministic_pose_sampling",
                    "a_v214_deterministic_opencv_registration",
                    "a_v215_fixed_async_pose_budget",
                    "a_v216_combined_reference_support",
                    "a_v217_coherent_frozen_geometry",
                    "a_v218_guarded_frozen_geometry",
                    "a_v219_guarded_frozen_live_pose",
                    "a_v220_homogeneous_frozen_geometry",
                    "a_v221_frozen_global_support_guard",
                }
                else "0.0"
            )
        ),
        *K16_ARGS,
    ]
    if spec.variant == "a_v22_epipolar":
        command.extend(
            [
                "--pose_verification_candidate_mode",
                "balanced_epipolar_v22",
            ]
        )
    elif spec.variant == "a_v23_multihypothesis":
        command.extend(
            [
                "--pose_verification_candidate_mode",
                "multihypothesis_v23",
            ]
        )
    elif spec.variant == "a_v24_multiview":
        command.extend(
            [
                "--pose_verification_candidate_mode",
                "multiview_relative_v24",
                "--pose_verification_max_temporal_score_ratio",
                "0.90",
            ]
        )
    elif spec.variant in {
        "a_v21",
        "a_v21_strict",
        "a_v21_sensitive",
        "a_v21_sensitive_g005",
        "a_v25_anchor_freeze",
        "a_v26_frozen_sparse",
        "a_v27_frozen_verification",
        "a_v28_stable_anchor",
        "a_v29_reference_guard",
        "a_v210_high_risk_reference_guard",
        "a_v211_trigger_aligned_reference_guard",
        "a_v212_temporal_safe_reference_guard",
        "a_v213_deterministic_pose_sampling",
        "a_v214_deterministic_opencv_registration",
        "a_v215_fixed_async_pose_budget",
        "a_v216_combined_reference_support",
        "a_v217_coherent_frozen_geometry",
        "a_v218_guarded_frozen_geometry",
        "a_v219_guarded_frozen_live_pose",
        "a_v220_homogeneous_frozen_geometry",
        "a_v221_frozen_global_support_guard",
    }:
        command.extend(
            [
                "--pose_verification_candidate_mode",
                "balanced_step_v21",
            ]
        )
    if spec.variant in {"a_v21_sensitive", "a_v21_sensitive_g005"}:
        command.extend(
            [
                "--pose_initialization_risk_adaptive_sigma",
                "0.0",
            ]
        )
    if spec.variant == "a_v25_anchor_freeze":
        command.extend(
            [
                "--pose_verification_geometry_anchor_mode",
                "freeze_v1",
            ]
        )
    if spec.variant == "a_v26_frozen_sparse":
        command.extend(
            [
                "--pose_verification_reference_geometry_mode",
                "frozen_first_valid_v1",
            ]
        )
    if spec.variant == "a_v217_coherent_frozen_geometry":
        command.extend(
            [
                "--pose_verification_geometry_anchor_mode",
                "freeze_v1",
                "--pose_verification_reference_geometry_mode",
                "frozen_first_valid_v1",
            ]
        )
    if spec.variant == "a_v218_guarded_frozen_geometry":
        command.extend(
            [
                "--pose_verification_reference_geometry_mode",
                "guarded_frozen_v3",
                "--pose_verification_frozen_min_match_support",
                "24",
                "--pose_verification_frozen_min_live_ratio",
                "0.50",
            ]
        )
    if spec.variant == "a_v219_guarded_frozen_live_pose":
        command.extend(
            [
                "--pose_verification_reference_geometry_mode",
                "guarded_frozen_live_pose_v4",
                "--pose_verification_frozen_min_match_support",
                "24",
                "--pose_verification_frozen_min_live_ratio",
                "0.50",
            ]
        )
    if spec.variant == "a_v220_homogeneous_frozen_geometry":
        command.extend(
            [
                "--pose_verification_reference_geometry_mode",
                "guarded_frozen_homogeneous_v5",
                "--pose_verification_frozen_min_match_support",
                "24",
                "--pose_verification_frozen_min_live_ratio",
                "0.50",
                "--pose_verification_frozen_min_reference_count",
                "2",
                "--pose_verification_frozen_min_total_support",
                "48",
            ]
        )
    if spec.variant == "a_v221_frozen_global_support_guard":
        command.extend(
            [
                "--pose_verification_reference_geometry_mode",
                "frozen_global_support_guard_v6",
                "--pose_verification_frozen_min_total_support",
                "24",
            ]
        )
    if spec.variant == "a_v222_pose_safe_v18":
        command.extend(
            [
                "--pose_direct_retry_mode",
                "pose_safe_v18",
            ]
        )
    if spec.variant == "a_v27_frozen_verification":
        command.extend(
            [
                "--pose_verification_reference_geometry_mode",
                "frozen_verification_only_v2",
            ]
        )
    if spec.variant in {
        "a_v28_stable_anchor",
        "a_v29_reference_guard",
        "a_v210_high_risk_reference_guard",
        "a_v211_trigger_aligned_reference_guard",
        "a_v212_temporal_safe_reference_guard",
        "a_v213_deterministic_pose_sampling",
        "a_v214_deterministic_opencv_registration",
        "a_v215_fixed_async_pose_budget",
        "a_v216_combined_reference_support",
    }:
        command.extend(
            [
                "--pose_verification_anchor_reference_mode",
                "stable_anchor_v1",
                "--pose_verification_anchor_candidate_scope",
                (
                    "combined_v1"
                    if spec.variant == "a_v216_combined_reference_support"
                    else "anchor_only_v1"
                ),
                "--pose_verification_reference_geometry_mode",
                "frozen_verification_only_v2",
                "--pose_verification_anchor_pre_error_scale",
                "4.0",
            ]
        )
    if spec.variant in {
        "a_v29_reference_guard",
        "a_v210_high_risk_reference_guard",
        "a_v211_trigger_aligned_reference_guard",
        "a_v212_temporal_safe_reference_guard",
        "a_v213_deterministic_pose_sampling",
        "a_v214_deterministic_opencv_registration",
        "a_v215_fixed_async_pose_budget",
        "a_v216_combined_reference_support",
    }:
        command.extend(
            [
                "--pose_verification_reference_policy",
                (
                    "conservative_high_risk_v2"
                    if spec.variant
                    in {
                        "a_v210_high_risk_reference_guard",
                        "a_v211_trigger_aligned_reference_guard",
                        "a_v212_temporal_safe_reference_guard",
                        "a_v213_deterministic_pose_sampling",
                        "a_v214_deterministic_opencv_registration",
                        "a_v215_fixed_async_pose_budget",
                        "a_v216_combined_reference_support",
                    }
                    else "conservative_quarantine_v1"
                ),
                "--pose_verification_reference_risk_threshold",
                (
                    "0.10"
                    if spec.variant
                    in {
                        "a_v211_trigger_aligned_reference_guard",
                        "a_v212_temporal_safe_reference_guard",
                        "a_v213_deterministic_pose_sampling",
                        "a_v214_deterministic_opencv_registration",
                        "a_v215_fixed_async_pose_budget",
                        "a_v216_combined_reference_support",
                    }
                    else "0.12"
                ),
                "--pose_verification_reference_cooldown_frames",
                "20",
            ]
        )
    if spec.variant == "a_v212_temporal_safe_reference_guard":
        command.extend(
            [
                "--pose_verification_max_temporal_score_ratio",
                "1.0",
            ]
        )
    if spec.variant in {
        "a_v213_deterministic_pose_sampling",
        "a_v214_deterministic_opencv_registration",
    }:
        command.extend(
            [
                "--pose_verification_registration_sampling_mode",
                "frame_deterministic_v1",
            ]
        )
    if spec.variant == "a_v214_deterministic_opencv_registration":
        command.extend(
            [
                "--pose_verification_registration_solver_mode",
                "deterministic_opencv_v2",
            ]
        )
    if spec.variant == "a_v215_fixed_async_pose_budget":
        command.extend(
            [
                "--pose_verification_async_pose_protection_mode",
                "fixed_joint_budget_v1",
            ]
        )
    return command


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def _git_dirty() -> bool:
    process = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.returncode != 0 or bool(process.stdout.strip())


def gpu_inventory() -> list[dict[str, str]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "nvidia-smi failed")
    rows: list[dict[str, str]] = []
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4:
            rows.append(
                {
                    "index": parts[0],
                    "uuid": parts[1],
                    "name": parts[2],
                    "driver_version": parts[3],
                }
            )
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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
    metadata = _read_json(spec.model_dir / "metadata.json")
    return bool(status and status.get("returncode") == 0 and metadata)


def _safe_remove_model(spec: ExperimentSpec, output_root: Path) -> None:
    if not spec.model_dir.exists():
        return
    model = spec.model_dir.resolve()
    root = Path(output_root).resolve()
    if model == root or root not in model.parents:
        raise ValueError(f"refusing to remove model outside output root: {model}")
    shutil.rmtree(model)


def _summary(
    spec: ExperimentSpec,
    *,
    gpu: str,
    returncode: int,
    wall_time_seconds: float = 0.0,
    skipped: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    metadata = _read_json(spec.model_dir / "metadata.json") or {}
    trace = _read_json(spec.model_dir / "pose_initialization_risk_trace.json") or {}
    trace_summary = trace.get("summary", {})
    if not isinstance(trace_summary, dict):
        trace_summary = {}
    return {
        "job_id": spec.job_id,
        "repeat": spec.repeat,
        "variant": spec.variant,
        "a_mode": VARIANTS[spec.variant],
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "test_hold": spec.scene.test_hold,
        "gpu": str(gpu),
        "returncode": int(returncode),
        "skipped": bool(skipped),
        "dry_run": bool(dry_run),
        "wall_time_seconds": float(wall_time_seconds),
        "num_keyframes": metadata.get("num keyframes", metadata.get("num_keyframes")),
        "a_eligible": trace_summary.get("eligible", 0),
        "a_candidates": trace_summary.get("verification_candidates", 0),
        "a_attempts": trace_summary.get("verification_attempts", 0),
        "a_accepts": trace_summary.get("verification_accepted", 0),
        "model_dir": str(spec.model_dir),
        "log_path": str(spec.run_dir / "train.log"),
    }


def _write_manifest(output_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    existing: list[dict[str, Any]] = []
    manifest_path = output_root / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            existing = [row for row in value if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        pass
    by_job_id: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in [*existing, *rows]:
        job_id = str(row.get("job_id", "") or "")
        if job_id:
            by_job_id[job_id] = dict(row)
        else:
            anonymous.append(dict(row))
    ordered = [*by_job_id.values(), *anonymous]
    _write_json_atomic(output_root / "manifest.json", ordered)
    if not ordered:
        return
    fields: list[str] = []
    for row in ordered:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = output_root / "manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(output_root / "manifest.csv")


def run_one(
    spec: ExperimentSpec,
    *,
    output_root: Path,
    python: Path,
    gpu: str,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    command = build_command(spec, python=python)
    if dry_run:
        print(f"[dry-run {spec.job_id} gpu={gpu}] {' '.join(command)}", flush=True)
        return _summary(spec, gpu=gpu, returncode=0, dry_run=True)
    if skip_existing and _is_complete(spec):
        print(f"[skip {spec.job_id}] complete", flush=True)
        return _summary(spec, gpu=gpu, returncode=0, skipped=True)

    spec.run_dir.mkdir(parents=True, exist_ok=True)
    _safe_remove_model(spec, output_root)
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "job_id": spec.job_id,
        "repeat": spec.repeat,
        "variant": spec.variant,
        "a_mode": VARIANTS[spec.variant],
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "source": str(spec.scene.source),
        "test_hold": spec.scene.test_hold,
        "gpu": str(gpu),
        "repo": str(ROOT),
        "repo_commit": _git_commit(),
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
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time = time.time() - started
    status = {
        **context,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_time_seconds": wall_time,
        "returncode": int(process.returncode),
        "metadata_exists": (spec.model_dir / "metadata.json").is_file(),
    }
    _write_json_atomic(spec.run_dir / "run_status.json", status)
    row = _summary(
        spec,
        gpu=gpu,
        returncode=process.returncode,
        wall_time_seconds=wall_time,
    )
    print(
        f"[{spec.job_id}] returncode={process.returncode} "
        f"attempts={row['a_attempts']} accepts={row['a_accepts']}",
        flush=True,
    )
    return row


def preflight(
    specs: Sequence[ExperimentSpec],
    python: Path,
    gpu: str,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if not python.is_file():
        raise FileNotFoundError(python)
    repo_dirty = _git_dirty()
    if repo_dirty and not allow_dirty:
        raise RuntimeError("experiment repository must be clean before launch")
    for spec in specs:
        if not spec.scene.source.is_dir():
            raise FileNotFoundError(spec.scene.source)
    inventory = gpu_inventory()
    matches = [row for row in inventory if row["index"] == str(gpu)]
    if len(matches) != 1:
        raise ValueError(f"GPU index {gpu} was not found exactly once")
    return {
        "repo": str(ROOT),
        "repo_commit": _git_commit(),
        "repo_dirty": bool(repo_dirty),
        "python": str(python),
        "gpu": matches[0],
        "job_count": len(specs),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=[])
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    parser.add_argument("--repeats", nargs="*", type=int, default=[])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gpu = validate_single_gpu(args.gpu)
    variants = list(args.variants) if args.variants else list(VARIANTS)
    scene_names = list(args.scenes) if args.scenes else list(SCENES)
    repeats = list(args.repeats) if args.repeats else [1, 2, 3, 4, 5]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or RESULTS_PARENT / f"run_{timestamp}"
    specs = build_specs(
        output_root=output_root,
        variants=variants,
        scene_names=scene_names,
        repeats=repeats,
    )
    context = preflight(
        specs,
        args.python,
        gpu,
        allow_dirty=bool(args.allow_dirty),
    )
    context.update(
        {
            "output_root": str(output_root),
            "variants": variants,
            "scenes": scene_names,
            "repeats": repeats,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    _write_json_atomic(output_root / "experiment_context.json", context)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_one(
            spec,
            output_root=output_root,
            python=args.python,
            gpu=gpu,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )
        rows.append(row)
        _write_manifest(output_root, rows)
        if int(row["returncode"]) != 0:
            print(f"stopping after failed job: {spec.job_id}", flush=True)
            return int(row["returncode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
