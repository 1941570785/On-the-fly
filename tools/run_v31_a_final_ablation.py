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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v31_a_official_pose_benchmark import (
    A_MODULE_ARGS,
    DEFAULT_PYTHON,
    RESULTS_PARENT,
    SCENES,
    V31_PROFILE_ARGS,
    SceneSpec,
    git_commit,
    read_metadata,
    validate_single_gpu,
)


VARIANTS = (
    "w_o_pose_risk_a",
    "w_o_response_sampling",
    "w_o_extra_optimization",
    "full",
)


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    scene: SceneSpec
    model_dir: Path
    run_dir: Path


def build_specs(
    output_root: Path,
    variants: list[str] | None,
    scene_names: list[str] | None,
) -> list[AblationSpec]:
    selected_variants = list(variants) if variants else list(VARIANTS)
    selected_scenes = list(scene_names) if scene_names else list(SCENES)
    unknown_variants = set(selected_variants) - set(VARIANTS)
    unknown_scenes = set(selected_scenes) - set(SCENES)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")
    if unknown_scenes:
        raise ValueError(f"Unknown scenes: {sorted(unknown_scenes)}")
    specs: list[AblationSpec] = []
    for scene_name in selected_scenes:
        for variant in selected_variants:
            run_dir = output_root / variant / scene_name
            specs.append(
                AblationSpec(
                    variant=variant,
                    scene=SCENES[scene_name],
                    model_dir=run_dir / "model",
                    run_dir=run_dir,
                )
            )
    return specs


def build_command(
    spec: AblationSpec,
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
    ]
    if spec.variant == "w_o_pose_risk_a":
        command.extend(
            [
                "--pose_initialization_risk_mode",
                "off",
                "--pose_risk_utility_admission_mode",
                "off",
            ]
        )
    else:
        command.extend(A_MODULE_ARGS)
        if spec.variant == "w_o_response_sampling":
            command.extend(
                [
                    "--paper_aligned_v31_component_ablation",
                    "disable_response_sampling",
                ]
            )
        elif spec.variant == "w_o_extra_optimization":
            command.extend(
                [
                    "--paper_aligned_v31_component_ablation",
                    "disable_extra_optimization",
                ]
            )
        elif spec.variant != "full":
            raise ValueError(f"Unknown variant: {spec.variant}")
    return command


def status_path(spec: AblationSpec) -> Path:
    return spec.run_dir / "run_status.json"


def is_complete(spec: AblationSpec) -> bool:
    if not status_path(spec).exists() or not (spec.model_dir / "metadata.json").exists():
        return False
    try:
        status = json.loads(status_path(spec).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(status.get("returncode", 1)) == 0


def summarize(
    spec: AblationSpec,
    returncode: int,
    *,
    skipped: bool = False,
) -> dict[str, Any]:
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
        "num_anchors": metadata.get("num anchors", metadata.get("num_anchors")),
        "model_dir": str(spec.model_dir),
        "log_path": str(spec.run_dir / "train.log"),
    }


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0]) if rows else []
    with (output_root / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    spec: AblationSpec,
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
    if spec.model_dir.exists():
        resolved_model = spec.model_dir.resolve()
        resolved_root = output_root.resolve()
        if resolved_model == resolved_root or resolved_root not in resolved_model.parents:
            raise ValueError(f"Refusing to remove model outside output root: {resolved_model}")
        shutil.rmtree(resolved_model)
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "variant": spec.variant,
        "dataset": spec.scene.dataset,
        "scene": spec.scene.name,
        "test_hold": spec.scene.test_hold,
        "source": str(spec.scene.source),
        "repo": str(ROOT),
        "repo_commit": git_commit(ROOT),
        "cuda_visible_devices": gpu,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (spec.run_dir / "command.json").write_text(
        json.dumps(context, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=[])
    parser.add_argument("--scenes", nargs="*", choices=tuple(SCENES), default=[])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gpu = validate_single_gpu(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or RESULTS_PARENT / f"v31_a_final_ablation_{timestamp}"
    specs = build_specs(output_root, args.variants, args.scenes)
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    for spec in specs:
        if not spec.scene.source.is_dir():
            raise FileNotFoundError(spec.scene.source)
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
