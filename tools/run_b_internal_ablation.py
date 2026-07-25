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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr_gs.config import resolve_config


B_SIGNAL_MODES = ("base", "r", "r_e", "r_d", "r_e_d")


@dataclass(frozen=True)
class Scene:
    dataset: str
    name: str
    relative_path: str
    test_hold: int


@dataclass(frozen=True)
class Job:
    signal_mode: str
    repeat: int
    seed: int


XYZ_SCENE = Scene(
    "TUM RGB-D",
    "xyz",
    "TUM/rgbd_dataset_freiburg2_xyz",
    30,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the five-way internal B-module ablation on TUM RGB-D xyz."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_gpus(gpus: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(gpu) for gpu in gpus)
    if not 1 <= len(values) <= 3:
        raise ValueError("B ablation requires between one and three GPUs")
    if len(set(values)) != len(values):
        raise ValueError("GPU identifiers must be unique")
    return values


def build_jobs(*, repeat: int, base_seed: int) -> list[Job]:
    if repeat < 1:
        raise ValueError("--repeat must be positive")
    jobs = []
    for repeat_index in range(repeat):
        seed = base_seed + repeat_index
        ordered_modes = (
            B_SIGNAL_MODES[repeat_index % len(B_SIGNAL_MODES) :]
            + B_SIGNAL_MODES[: repeat_index % len(B_SIGNAL_MODES)]
        )
        jobs.extend(
            Job(mode, repeat_index + 1, seed) for mode in ordered_modes
        )
    return jobs


def build_train_command(
    *,
    scene: Scene,
    source_path: Path,
    output_path: Path,
    signal_mode: str,
    seed: int,
    deterministic: bool,
) -> list[str]:
    if signal_mode not in B_SIGNAL_MODES:
        raise ValueError(f"unsupported B signal mode: {signal_mode}")
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "-s",
        str(source_path),
        "-m",
        str(output_path),
        "--method",
        "asr-gs",
        "--ablate-c",
        "--b-signal-mode",
        signal_mode,
        "--test_hold",
        str(scene.test_hold),
        "--test_frequency",
        "-1",
        "--viewer_mode",
        "none",
        "--enable_reboot",
        "--experiment-seed",
        str(seed),
    ]
    if deterministic:
        command.append("--deterministic")
    return command


def _git_output(arguments: list[str]) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def git_state() -> dict[str, object]:
    revision = _git_output(["rev-parse", "HEAD"]).decode().strip()
    status = _git_output(["status", "--porcelain=v1", "-z"])
    fingerprint = hashlib.sha256()
    fingerprint.update(status)
    fingerprint.update(_git_output(["diff", "--binary", "HEAD"]))
    fingerprint.update(_git_output(["diff", "--binary", "--cached", "HEAD"]))
    untracked = _git_output(
        ["ls-files", "--others", "--exclude-standard", "-z"]
    )
    for encoded_path in sorted(filter(None, untracked.split(b"\0"))):
        path = ROOT / encoded_path.decode(errors="surrogateescape")
        fingerprint.update(encoded_path)
        if path.is_file():
            fingerprint.update(path.read_bytes())
    return {
        "revision": revision,
        "dirty": bool(status),
        "working_tree_sha256": fingerprint.hexdigest(),
    }


def prepare_output_directory(
    output_path: Path,
    output_root: Path,
    *,
    force: bool,
) -> None:
    root = output_root.resolve()
    target = output_path.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"unsafe ablation output path: {target}")
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def _read_result(
    *,
    output_path: Path,
    job: Job,
    expected_fingerprint: str,
) -> dict[str, object]:
    with (output_path / "metadata.json").open(
        "r", encoding="utf-8"
    ) as source:
        metadata = json.load(source)
    method_data = metadata.get("asr_gs", {})
    fingerprint = str(method_data.get("fingerprint", ""))
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"{output_path}: expected config {expected_fingerprint}, "
            f"found {fingerprint or 'none'}"
        )
    sampling = method_data.get("response_sampling", {})
    seconds = float(metadata["time"])
    return {
        "signal_mode": job.signal_mode,
        "repeat": job.repeat,
        "seed": job.seed,
        "PSNR": float(metadata["PSNR"]),
        "SSIM": float(metadata["SSIM"]),
        "LPIPS": float(metadata["LPIPS"]),
        "time_seconds": seconds,
        "config_fingerprint": fingerprint,
        "b_events": int(sampling.get("events", 0)),
        "b_applied": int(sampling.get("applied", 0)),
        "b_clipped_pixels": int(sampling.get("clipped_pixels", 0)),
        "b_mass_before": float(sampling.get("mass_before", 0.0)),
        "b_mass_after": float(sampling.get("mass_after", 0.0)),
        "output_path": str(output_path),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for mode in B_SIGNAL_MODES:
        selected = [row for row in rows if row["signal_mode"] == mode]
        if not selected:
            continue
        count = len(selected)
        output.append(
            {
                "signal_mode": mode,
                "runs": count,
                "PSNR_mean": sum(float(row["PSNR"]) for row in selected)
                / count,
                "SSIM_mean": sum(float(row["SSIM"]) for row in selected)
                / count,
                "LPIPS_mean": sum(float(row["LPIPS"]) for row in selected)
                / count,
                "time_seconds_mean": sum(
                    float(row["time_seconds"]) for row in selected
                )
                / count,
                "b_applied_mean": sum(
                    int(row["b_applied"]) for row in selected
                )
                / count,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    gpus = validate_gpus(args.gpus)
    jobs_to_run = build_jobs(repeat=args.repeat, base_seed=args.seed)
    source_path = args.data_root / XYZ_SCENE.relative_path
    if not (source_path / "images").is_dir():
        raise FileNotFoundError(f"missing scene images: {source_path}")

    configs = {
        mode: resolve_config(
            "asr-gs",
            {"c"},
            sampling_mode=mode,
        )
        for mode in B_SIGNAL_MODES
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "B internal signal ablation",
        "fixed_components": {
            "A": "enabled with final ASR-GS configuration",
            "C": "disabled",
            "sampling_budget": "unchanged",
        },
        "B_signal_modes": {
            mode: asdict(configs[mode].sampling)
            for mode in B_SIGNAL_MODES
        },
        "config_fingerprints": {
            mode: configs[mode].fingerprint for mode in B_SIGNAL_MODES
        },
        "scene": asdict(XYZ_SCENE),
        "repeat": args.repeat,
        "base_seed": args.seed,
        "gpus": list(gpus),
        "git": git_state(),
    }
    with (args.output_root / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(manifest, output, indent=2)

    jobs: queue.Queue[Job | None] = queue.Queue()
    for job in jobs_to_run:
        jobs.put(job)
    for _ in gpus:
        jobs.put(None)

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            job = jobs.get()
            if job is None:
                jobs.task_done()
                return
            output_path = (
                args.output_root
                / job.signal_mode
                / f"repeat_{job.repeat}"
                / "TUM_RGB-D"
                / XYZ_SCENE.name
            )
            try:
                metadata_path = output_path / "metadata.json"
                if args.force or not metadata_path.is_file():
                    prepare_output_directory(
                        output_path,
                        args.output_root,
                        force=True,
                    )
                    command = build_train_command(
                        scene=XYZ_SCENE,
                        source_path=source_path,
                        output_path=output_path,
                        signal_mode=job.signal_mode,
                        seed=job.seed,
                        deterministic=args.deterministic,
                    )
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = gpu
                    with (output_path / "train.log").open(
                        "w", encoding="utf-8"
                    ) as log:
                        subprocess.run(
                            command,
                            cwd=ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                row = _read_result(
                    output_path=output_path,
                    job=job,
                    expected_fingerprint=configs[
                        job.signal_mode
                    ].fingerprint,
                )
                with lock:
                    rows.append(row)
                    print(
                        f"[GPU {gpu}] {job.signal_mode} "
                        f"repeat {job.repeat}: "
                        f"PSNR={row['PSNR']:.3f}, "
                        f"SSIM={row['SSIM']:.4f}, "
                        f"LPIPS={row['LPIPS']:.4f}",
                        flush=True,
                    )
            except Exception as error:
                with lock:
                    errors.append(
                        f"{job.signal_mode} repeat {job.repeat}: {error}"
                    )
            finally:
                jobs.task_done()

    workers = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in gpus
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    rows.sort(
        key=lambda row: (
            B_SIGNAL_MODES.index(str(row["signal_mode"])),
            int(row["repeat"]),
        )
    )
    _write_csv(args.output_root / "run_metrics.csv", rows)
    _write_csv(
        args.output_root / "aggregate_metrics.csv",
        aggregate_rows(rows),
    )
    if errors:
        raise RuntimeError("B ablation failures:\n" + "\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
