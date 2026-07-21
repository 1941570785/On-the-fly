#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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
    RESULTS_PARENT,
    SCENES,
    V31_PROFILE_ARGS,
    SceneSpec,
)
from tools.run_v31_a_official_pose_benchmark import git_commit


DEFAULT_BUDGETS = (0, 2, 4, 8, 12, 16, 20)
DEFAULT_SEEDS = (0,)
ACTIVE_SCENES = tuple(SCENES)
INACTIVE_SCENES: tuple[str, ...] = ()
PHASES = ("sweep", "validation", "smoke")
A_MODULE_ARGS = (
    "--pose_initialization_risk_mode",
    "verify_v1",
    *A_SHARED_ARGS,
)


@dataclass(frozen=True)
class RoundSpec:
    phase: str
    budget: int
    seed: int
    scene: SceneSpec
    run_dir: Path
    model_dir: Path

    @property
    def job_id(self) -> str:
        return f"{self.phase}:{self.scene.name}:seed{self.seed}:k{self.budget}"


def validate_gpus(gpus: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(gpu).strip() for gpu in gpus if str(gpu).strip())
    if not values or len(values) > 6 or len(set(values)) != len(values):
        raise ValueError("provide one to six unique GPU IDs")
    if any("," in value for value in values):
        raise ValueError("each GPU ID must identify exactly one physical GPU")
    return values


def validate_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(budget) for budget in budgets)
    if not values or len(set(values)) != len(values):
        raise ValueError("provide unique round budgets")
    if any(value < 0 or (value > 0 and value % 2 != 0) for value in values):
        raise ValueError("round budgets must be zero or positive even integers")
    return values


def validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("provide unique experiment seeds")
    if any(value < 0 for value in values):
        raise ValueError("experiment seeds must be non-negative")
    return values


def budget_fraction(budget: int) -> float:
    if int(budget) <= 0:
        return 0.0
    return (int(budget) - 0.5) / 30.0


def budget_args(budget: int) -> list[str]:
    value = validate_budgets((budget,))[0]
    if value == 0:
        return [
            "--paper_aligned_v31_component_ablation",
            "disable_extra_optimization",
        ]
    return [
        "--paper_aligned_pose_render_extra_optimization_fraction",
        format(budget_fraction(value), ".17g"),
        "--paper_aligned_pose_render_extra_optimization_max_extra",
        str(value),
    ]


def build_specs(
    *,
    output_root: Path,
    phase: str,
    scene_names: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
) -> list[RoundSpec]:
    phase_name = str(phase)
    if phase_name not in PHASES:
        raise ValueError(f"unknown phase: {phase_name}")
    budget_values = validate_budgets(budgets)
    seed_values = validate_seeds(seeds)
    names = tuple(str(name) for name in scene_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("provide unique scene names")
    unknown = set(names) - set(SCENES)
    if unknown:
        raise ValueError(f"unknown scenes: {sorted(unknown)}")

    specs: list[RoundSpec] = []
    for scene_name in names:
        scene = SCENES[scene_name]
        for seed in seed_values:
            for budget in budget_values:
                run_dir = (
                    Path(output_root)
                    / phase_name
                    / f"k{budget:02d}"
                    / f"seed{seed}"
                    / scene_name
                )
                specs.append(
                    RoundSpec(
                        phase=phase_name,
                        budget=budget,
                        seed=seed,
                        scene=scene,
                        run_dir=run_dir,
                        model_dir=run_dir / "model",
                    )
                )
    return specs


def build_command(
    spec: RoundSpec,
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
        *A_MODULE_ARGS,
        "--experiment_seed",
        str(spec.seed),
        *budget_args(spec.budget),
    ]


def build_worker_queues(
    specs: Sequence[RoundSpec], gpus: Sequence[str]
) -> dict[str, list[RoundSpec]]:
    gpu_values = validate_gpus(gpus)
    queues = {gpu: [] for gpu in gpu_values}
    blocks: dict[tuple[str, int], list[RoundSpec]] = {}
    for spec in specs:
        blocks.setdefault((spec.scene.name, spec.seed), []).append(spec)

    scene_order = {name: index for index, name in enumerate(SCENES)}
    ordered_blocks = sorted(
        blocks.items(), key=lambda item: (scene_order[item[0][0]], item[0][1])
    )
    for block_index, (_, jobs) in enumerate(ordered_blocks):
        gpu = gpu_values[block_index % len(gpu_values)]
        ordered = sorted(jobs, key=lambda spec: spec.budget)
        shift = block_index % len(ordered)
        queues[gpu].extend(ordered[shift:] + ordered[:shift])
    return queues


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def is_complete(spec: RoundSpec) -> bool:
    status = _read_json_dict(spec.run_dir / "run_status.json")
    metadata = _read_json_dict(spec.model_dir / "metadata.json")
    return bool(status is not None and status.get("returncode") == 0 and metadata is not None)


def read_metadata(spec: RoundSpec) -> dict[str, Any]:
    return _read_json_dict(spec.model_dir / "metadata.json") or {}


def summarize(
    spec: RoundSpec,
    *,
    gpu: str,
    returncode: int,
    wall_time_seconds: float = 0.0,
    skipped: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    metadata = read_metadata(spec)
    extra = metadata.get("pose_render_extra_optimization", {})
    if not isinstance(extra, dict):
        extra = {}
    return {
        "job_id": spec.job_id,
        "phase": spec.phase,
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "test_hold": spec.scene.test_hold,
        "budget": spec.budget,
        "requested_fraction": budget_fraction(spec.budget),
        "seed": spec.seed,
        "gpu": str(gpu),
        "returncode": int(returncode),
        "skipped": bool(skipped),
        "dry_run": bool(dry_run),
        "PSNR": metadata.get("PSNR"),
        "SSIM": metadata.get("SSIM"),
        "LPIPS": metadata.get("LPIPS"),
        "time": metadata.get("time"),
        "wall_time_seconds": float(wall_time_seconds),
        "num_keyframes": metadata.get("num keyframes", metadata.get("num_keyframes")),
        "num_anchors": metadata.get("num anchors", metadata.get("num_anchors")),
        "extra_mode": extra.get("mode", "off" if spec.budget == 0 else ""),
        "extra_events": int(extra.get("events", 0) or 0),
        "extra_applied": int(extra.get("applied", 0) or 0),
        "extra_iterations": int(extra.get("extra_iterations_sum", 0) or 0),
        "extra_iterations_mean": float(extra.get("extra_iterations_mean", 0.0) or 0.0),
        "model_dir": str(spec.model_dir),
        "log_path": str(spec.run_dir / "train.log"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_manifest(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["phase"]),
            str(row["scene"]),
            int(row["seed"]),
            int(row["budget"]),
        ),
    )
    _write_json(output_dir / "manifest.json", ordered)
    if not ordered:
        return
    fields: list[str] = []
    for row in ordered:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = output_dir / "manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(output_dir / "manifest.csv")


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
    result: list[dict[str, str]] = []
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4:
            result.append(
                {
                    "index": parts[0],
                    "uuid": parts[1],
                    "name": parts[2],
                    "driver_version": parts[3],
                }
            )
    return result


def _safe_remove_model(spec: RoundSpec, output_root: Path) -> None:
    if not spec.model_dir.exists():
        return
    resolved_model = spec.model_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_model == resolved_root or resolved_root not in resolved_model.parents:
        raise ValueError(f"refusing to remove model outside output root: {resolved_model}")
    shutil.rmtree(resolved_model)


def run_one(
    spec: RoundSpec,
    *,
    output_root: Path,
    python: Path,
    gpu: str,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    command = build_command(spec, python=python)
    if dry_run:
        print(f"[dry-run gpu={gpu} {spec.job_id}] {' '.join(command)}", flush=True)
        return summarize(spec, gpu=gpu, returncode=0, dry_run=True)
    if skip_existing and is_complete(spec):
        status = _read_json_dict(spec.run_dir / "run_status.json") or {}
        print(f"[skip gpu={gpu} {spec.job_id}] complete", flush=True)
        return summarize(
            spec,
            gpu=str(status.get("gpu", gpu)),
            returncode=0,
            wall_time_seconds=float(status.get("wall_time_seconds", 0.0) or 0.0),
            skipped=True,
        )

    spec.run_dir.mkdir(parents=True, exist_ok=True)
    _safe_remove_model(spec, output_root)
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "job_id": spec.job_id,
        "phase": spec.phase,
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "source": str(spec.scene.source),
        "test_hold": spec.scene.test_hold,
        "budget": spec.budget,
        "requested_fraction": budget_fraction(spec.budget),
        "seed": spec.seed,
        "gpu": str(gpu),
        "repo": str(ROOT),
        "repo_commit": git_commit(ROOT),
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(spec.run_dir / "command.json", context)
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
    _write_json(spec.run_dir / "run_status.json", status)
    row = summarize(
        spec,
        gpu=gpu,
        returncode=process.returncode,
        wall_time_seconds=wall_time,
    )
    print(
        f"[gpu={gpu} {spec.job_id}] returncode={process.returncode} "
        f"PSNR={row['PSNR']} SSIM={row['SSIM']} LPIPS={row['LPIPS']}",
        flush=True,
    )
    return row


def run_specs(
    specs: Sequence[RoundSpec],
    *,
    output_root: Path,
    python: Path,
    gpus: Sequence[str],
    skip_existing: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if not specs:
        raise ValueError("at least one job is required")
    queues = build_worker_queues(specs, gpus)
    manifest_dir = output_root / specs[0].phase
    rows: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def run_queue(gpu: str, jobs: Sequence[RoundSpec]) -> None:
        for spec in jobs:
            row = run_one(
                spec,
                output_root=output_root,
                python=python,
                gpu=gpu,
                skip_existing=skip_existing,
                dry_run=dry_run,
            )
            with lock:
                rows[spec.job_id] = row
                write_manifest(manifest_dir, list(rows.values()))

    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [executor.submit(run_queue, gpu, jobs) for gpu, jobs in queues.items()]
        for future in futures:
            future.result()
    return list(rows.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired V31+A extra-optimization round ablation."
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--phase", choices=PHASES, default="sweep")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--budgets", nargs="*", type=int, default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    parser.add_argument("--selected-budget", type=int, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_layout(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    if args.phase == "sweep":
        scene_names = tuple(args.scenes) if args.scenes else ACTIVE_SCENES
        budgets = tuple(args.budgets) if args.budgets else DEFAULT_BUDGETS
        seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    elif args.phase == "validation":
        scene_names = tuple(args.scenes) if args.scenes else INACTIVE_SCENES
        if args.budgets:
            budgets = tuple(args.budgets)
        elif args.selected_budget is not None:
            budgets = (0, int(args.selected_budget))
        else:
            raise ValueError("validation requires --selected-budget or --budgets")
        seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    else:
        scene_names = tuple(args.scenes) if args.scenes else ("desk",)
        budgets = tuple(args.budgets) if args.budgets else (2,)
        seeds = tuple(args.seeds) if args.seeds else (0,)
    return scene_names, validate_budgets(budgets), validate_seeds(seeds)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gpus = validate_gpus(args.gpus)
    scene_names, budgets, seeds = resolve_layout(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else RESULTS_PARENT / f"extra_optimization_round_ablation_{timestamp}"
    )
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    specs = build_specs(
        output_root=output_root,
        phase=args.phase,
        scene_names=scene_names,
        budgets=budgets,
        seeds=seeds,
    )
    for spec in specs:
        if not spec.scene.source.is_dir():
            raise FileNotFoundError(spec.scene.source)

    inventory = gpu_inventory()
    selected = [item for item in inventory if item["index"] in set(gpus)]
    if len(selected) != len(gpus):
        raise ValueError(f"unknown GPU IDs: {sorted(set(gpus) - {item['index'] for item in selected})}")
    phase_dir = output_root / args.phase
    _write_json(
        phase_dir / "experiment_context.json",
        {
            "phase": args.phase,
            "repo": str(ROOT),
            "repo_commit": git_commit(ROOT),
            "python": str(args.python),
            "gpus": selected,
            "scenes": list(scene_names),
            "budgets": list(budgets),
            "seeds": list(seeds),
            "jobs": len(specs),
            "dry_run": bool(args.dry_run),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    rows = run_specs(
        specs,
        output_root=output_root,
        python=args.python,
        gpus=gpus,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    failures = [row for row in rows if int(row["returncode"]) != 0]
    print(
        f"phase={args.phase} jobs={len(rows)} failures={len(failures)} "
        f"output={phase_dir}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
