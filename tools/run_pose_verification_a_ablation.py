#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/home/zxd/miniconda3/envs/otf/bin/python")
RESULTS_PARENT = ROOT / "results" / "BRANCH_EXPERIMENTS_20260703"
PADDED_ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly_padded_datasets")


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
    model_dir: Path
    run_dir: Path


SCENES: dict[str, SceneSpec] = {
    "bonsai": SceneSpec("bonsai", "MipNeRF360", PADDED_ROOT / "MipNerf360/bonsai", 8),
    "counter": SceneSpec("counter", "MipNeRF360", PADDED_ROOT / "MipNerf360/counter", 8),
    "garden": SceneSpec("garden", "MipNeRF360", PADDED_ROOT / "MipNerf360/garden", 8),
    "forest1": SceneSpec("forest1", "StaticHikes", PADDED_ROOT / "StaticHikes/forest1", 10),
    "forest2": SceneSpec("forest2", "StaticHikes", PADDED_ROOT / "StaticHikes/forest2", 10),
    "university2": SceneSpec(
        "university2", "StaticHikes", PADDED_ROOT / "StaticHikes/university2", 10
    ),
    "desk": SceneSpec("desk", "TUM RGB-D", PADDED_ROOT / "TUM/rgbd_dataset_freiburg1_desk", 30),
    "xyz": SceneSpec("xyz", "TUM RGB-D", PADDED_ROOT / "TUM/rgbd_dataset_freiburg2_xyz", 30),
    "long_office": SceneSpec(
        "long_office",
        "TUM RGB-D",
        PADDED_ROOT / "TUM/rgbd_dataset_freiburg3_long_office_household",
        30,
    ),
}

VARIANTS: dict[str, str] = {
    "v31_control": "off",
    "a_observe": "observe_v1",
    "a_full": "verify_v1",
}

V31_PROFILE_ARGS = (
    "--risk_admission_mode",
    "on_the_fly_innovation_v1",
    "--paper_aligned_pose_render_assimilation_profile",
    "baseline_render_lock_intra_frame_v31",
)

A_SHARED_ARGS = (
    "--pose_risk_utility_admission_mode",
    "off",
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
    "--pose_verification_mad_scale",
    "2.5",
    "--pose_verification_min_support",
    "24",
    "--pose_verification_min_improvement",
    "0.02",
    "--pose_verification_max_p90_ratio",
    "1.01",
    "--pose_verification_min_support_ratio",
    "0.80",
)


def validate_gpus(values: list[str]) -> list[str]:
    gpus = [str(value).strip() for value in values if str(value).strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if len(gpus) > 3:
        raise ValueError("This ablation permits at most three GPUs")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU identifiers must be unique")
    return gpus


def build_specs(
    output_root: Path,
    variants: list[str],
    scene_names: list[str],
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for scene_name in scene_names:
        if scene_name not in SCENES:
            raise ValueError(f"Unknown scene: {scene_name}")
        for variant in variants:
            if variant not in VARIANTS:
                raise ValueError(f"Unknown variant: {variant}")
            run_dir = output_root / variant / scene_name
            specs.append(
                ExperimentSpec(
                    variant=variant,
                    scene=SCENES[scene_name],
                    model_dir=run_dir / "model",
                    run_dir=run_dir,
                )
            )
    return specs


def build_command(
    spec: ExperimentSpec,
    *,
    python: Path = DEFAULT_PYTHON,
) -> list[str]:
    return [
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
    ]


def distribute_specs(
    specs: list[ExperimentSpec],
    gpus: list[str],
) -> dict[str, list[ExperimentSpec]]:
    output = {gpu: [] for gpu in gpus}
    scene_groups: dict[str, list[ExperimentSpec]] = {}
    for spec in specs:
        scene_groups.setdefault(spec.scene.name, []).append(spec)
    for index, scene_specs in enumerate(scene_groups.values()):
        output[gpus[index % len(gpus)]].extend(scene_specs)
    return output


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def _read_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _status_path(spec: ExperimentSpec) -> Path:
    return spec.run_dir / "run_status.json"


def _is_complete(spec: ExperimentSpec) -> bool:
    if not _status_path(spec).exists() or not (spec.model_dir / "metadata.json").exists():
        return False
    try:
        status = json.loads(_status_path(spec).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(status.get("returncode", 1)) == 0


def _summary(spec: ExperimentSpec, returncode: int, *, skipped: bool = False) -> dict[str, Any]:
    metadata = _read_metadata(spec.model_dir)
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


def _remove_model(spec: ExperimentSpec, output_root: Path) -> None:
    if not spec.model_dir.exists():
        return
    model = spec.model_dir.resolve()
    root = output_root.resolve()
    if model == root or root not in model.parents:
        raise ValueError(f"Refusing to remove model outside output root: {model}")
    shutil.rmtree(model)


def run_one(
    spec: ExperimentSpec,
    *,
    output_root: Path,
    python: Path,
    gpu: str,
    skip_existing: bool,
) -> dict[str, Any]:
    command = build_command(spec, python=python)
    label = f"{spec.variant}/{spec.scene.name}/gpu{gpu}"
    if skip_existing and _is_complete(spec):
        print(f"[{label}] existing complete run", flush=True)
        return _summary(spec, 0, skipped=True)
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    _remove_model(spec, output_root)
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "variant": spec.variant,
        "a_mode": VARIANTS[spec.variant],
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "test_hold": spec.scene.test_hold,
        "source": str(spec.scene.source),
        "repo": str(ROOT),
        "repo_commit": _git_commit(),
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
    print(f"[{label}] start", flush=True)
    with (spec.run_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
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
    _status_path(spec).write_text(
        json.dumps(status, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    row = _summary(spec, process.returncode)
    print(
        f"[{label}] rc={row['returncode']} PSNR={row['PSNR']} "
        f"SSIM={row['SSIM']} LPIPS={row['LPIPS']} time={row['time']}",
        flush=True,
    )
    return row


def _run_worker(
    gpu: str,
    specs: list[ExperimentSpec],
    *,
    output_root: Path,
    python: Path,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    return [
        run_one(
            spec,
            output_root=output_root,
            python=python,
            gpu=gpu,
            skip_existing=skip_existing,
        )
        for spec in specs
    ]


def _write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
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


def _preflight(specs: list[ExperimentSpec], python: Path) -> None:
    if not python.exists():
        raise FileNotFoundError(python)
    if not (ROOT / "train.py").exists():
        raise FileNotFoundError(ROOT / "train.py")
    for spec in specs:
        if not spec.scene.source.is_dir():
            raise FileNotFoundError(spec.scene.source)


def _prewarm_caches(
    specs: list[ExperimentSpec],
    *,
    python: Path,
    gpu: str,
    output_root: Path,
) -> None:
    sources = list(dict.fromkeys(str(spec.scene.source) for spec in specs))
    command = [
        str(python),
        "tools/prewarm_model_caches.py",
        *sources,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "cache_prewarm.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=[])
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    parser.add_argument("--gpus", nargs="*", default=[])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_cache_prewarm", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environment_gpus = [
        part.strip()
        for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if part.strip()
    ]
    gpus = validate_gpus(list(args.gpus) or environment_gpus)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or RESULTS_PARENT / f"pose_verification_a_full9_{timestamp}"
    variants = list(args.variants) if args.variants else list(VARIANTS)
    scene_names = list(args.scenes) if args.scenes else list(SCENES)
    specs = build_specs(output_root, variants, scene_names)
    _preflight(specs, args.python)
    if args.dry_run:
        for spec in specs:
            print(" ".join(build_command(spec, python=args.python)))
        print(f"dry-run jobs={len(specs)} gpus={','.join(gpus)} output={output_root}")
        return 0

    if not args.skip_cache_prewarm:
        print(f"prewarming shared model caches on gpu{gpus[0]}", flush=True)
        _prewarm_caches(
            specs,
            python=args.python,
            gpu=gpus[0],
            output_root=output_root,
        )

    groups = distribute_specs(specs, gpus)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                _run_worker,
                gpu,
                worker_specs,
                output_root=output_root,
                python=args.python,
                skip_existing=args.skip_existing,
            ): gpu
            for gpu, worker_specs in groups.items()
            if worker_specs
        }
        for future in as_completed(futures):
            rows.extend(future.result())

    scene_order = {name: index for index, name in enumerate(scene_names)}
    variant_order = {name: index for index, name in enumerate(variants)}
    rows.sort(key=lambda row: (scene_order[row["scene"]], variant_order[row["variant"]]))
    _write_manifest(output_root, rows)
    failed = [row for row in rows if int(row["returncode"]) != 0]
    print(f"output_root={output_root} completed={len(rows) - len(failed)}/{len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
