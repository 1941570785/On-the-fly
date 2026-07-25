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


@dataclass(frozen=True)
class Scene:
    dataset: str
    name: str
    relative_path: str
    test_hold: int


SCENES = (
    Scene("Mip-NeRF360", "bonsai", "MipNerf360/bonsai", 8),
    Scene("Mip-NeRF360", "counter", "MipNerf360/counter", 8),
    Scene("Mip-NeRF360", "garden", "MipNerf360/garden", 8),
    Scene("StaticHikes", "forest1", "StaticHikes/forest1", 10),
    Scene("StaticHikes", "forest2", "StaticHikes/forest2", 10),
    Scene("StaticHikes", "university2", "StaticHikes/university2", 10),
    Scene("TUM RGB-D", "desk", "TUM/rgbd_dataset_freiburg1_desk", 30),
    Scene("TUM RGB-D", "xyz", "TUM/rgbd_dataset_freiburg2_xyz", 30),
    Scene(
        "TUM RGB-D",
        "long_office",
        "TUM/rgbd_dataset_freiburg3_long_office_household",
        30,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed ASR-GS nine-scene evaluation protocol."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument(
        "--method",
        choices=("asr-gs", "baseline"),
        default="asr-gs",
    )
    parser.add_argument(
        "--ablate",
        action="append",
        choices=("a", "b", "c"),
        default=[],
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_gpus(gpus: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(gpu) for gpu in gpus)
    if not 1 <= len(values) <= 3:
        raise ValueError("ASR-GS benchmark requires between one and three GPUs")
    if len(set(values)) != len(values):
        raise ValueError("GPU identifiers must be unique")
    return values


def variant_name(method: str, ablations: Iterable[str]) -> str:
    disabled = sorted(set(ablations))
    suffix = "".join(f"_wo_{name}" for name in disabled)
    return f"{method}{suffix}"


def build_train_command(
    *,
    scene: Scene,
    source_path: Path,
    output_path: Path,
    method: str,
    ablations: Iterable[str],
    seed: int,
    deterministic: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "-s",
        str(source_path),
        "-m",
        str(output_path),
        "--method",
        method,
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
    for module in sorted(set(ablations)):
        command.append(f"--ablate-{module}")
    if deterministic:
        command.append("--deterministic")
    return command


def _git_output(arguments: list[str], root: Path) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def git_state(root: Path = ROOT) -> dict[str, object]:
    revision = _git_output(["rev-parse", "HEAD"], root).decode().strip()
    status = _git_output(["status", "--porcelain=v1", "-z"], root)
    fingerprint = hashlib.sha256()
    fingerprint.update(status)
    fingerprint.update(_git_output(["diff", "--binary", "HEAD"], root))
    fingerprint.update(
        _git_output(["diff", "--binary", "--cached", "HEAD"], root)
    )
    untracked = _git_output(
        ["ls-files", "--others", "--exclude-standard", "-z"],
        root,
    )
    for encoded_path in sorted(filter(None, untracked.split(b"\0"))):
        path = root / encoded_path.decode(errors="surrogateescape")
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
        raise ValueError(f"unsafe benchmark output path: {target}")
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def _read_result(
    *,
    scene: Scene,
    output_path: Path,
    repeat: int,
    seed: int,
    expected_fingerprint: str,
) -> dict[str, object]:
    metadata_path = output_path / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    method_data = metadata.get("asr_gs", {})
    fingerprint = str(method_data.get("fingerprint", ""))
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"{output_path}: expected config {expected_fingerprint}, "
            f"found {fingerprint or 'none'}"
        )
    pose = method_data.get("pose_reliability", {})
    sampling = method_data.get("response_sampling", {})
    refinement = method_data.get("transactional_refinement", {})
    seconds = float(metadata["time"])
    return {
        "dataset": scene.dataset,
        "scene": scene.name,
        "repeat": repeat,
        "seed": seed,
        "PSNR": float(metadata["PSNR"]),
        "SSIM": float(metadata["SSIM"]),
        "LPIPS": float(metadata["LPIPS"]),
        "time_seconds": seconds,
        "time_hms": (
            f"{int(seconds // 3600)}:"
            f"{int(seconds % 3600 // 60):02d}:"
            f"{int(seconds % 60):02d}"
        ),
        "config_fingerprint": fingerprint,
        "a_events": int(pose.get("events", 0)),
        "a_registered": int(pose.get("registered", 0)),
        "a_review_attempted": int(pose.get("review_attempted", 0)),
        "a_review_accepted": int(pose.get("review_accepted", 0)),
        "b_events": int(sampling.get("events", 0)),
        "b_applied": int(sampling.get("applied", 0)),
        "c_events": int(refinement.get("events", 0)),
        "c_requested": int(refinement.get("requested", 0)),
        "c_attempted": int(refinement.get("attempted", 0)),
        "c_committed": int(refinement.get("committed", 0)),
        "c_rolled_back": int(refinement.get("rolled_back", 0)),
        "c_realized_iterations": int(
            refinement.get("realized_iterations", 0)
        ),
        "output_path": str(output_path),
    }


def aggregate_dataset_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["repeat"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (dataset, repeat), selected in sorted(grouped.items()):
        count = len(selected)
        output.append(
            {
                "dataset": dataset,
                "repeat": repeat,
                "scenes": count,
                "PSNR": sum(float(row["PSNR"]) for row in selected) / count,
                "SSIM": sum(float(row["SSIM"]) for row in selected) / count,
                "LPIPS": sum(float(row["LPIPS"]) for row in selected) / count,
                "time_seconds": sum(
                    float(row["time_seconds"]) for row in selected
                )
                / count,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    gpus = validate_gpus(args.gpus)
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.method == "baseline" and args.ablate:
        raise ValueError("--ablate is only valid with --method asr-gs")

    config = resolve_config(args.method, args.ablate)
    variant = variant_name(args.method, args.ablate)
    missing = [
        str(args.data_root / scene.relative_path)
        for scene in SCENES
        if not (args.data_root / scene.relative_path / "images").is_dir()
    ]
    if missing:
        raise FileNotFoundError("Missing scene directories:\n" + "\n".join(missing))

    args.output_root.mkdir(parents=True, exist_ok=True)
    repository_state = git_state()
    manifest = {
        "git_revision": repository_state["revision"],
        "git_dirty": repository_state["dirty"],
        "working_tree_sha256": repository_state["working_tree_sha256"],
        "method": args.method,
        "ablations": sorted(set(args.ablate)),
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
        "gpus": list(gpus),
        "repeat": args.repeat,
        "base_seed": args.seed,
        "scenes": [asdict(scene) for scene in SCENES],
    }
    with (args.output_root / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(manifest, output, indent=2)

    jobs: queue.Queue[tuple[Scene, int, int] | None] = queue.Queue()
    for repeat in range(args.repeat):
        seed = args.seed + repeat
        for scene in SCENES:
            jobs.put((scene, repeat, seed))
    for _ in gpus:
        jobs.put(None)

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            scene, repeat, seed = job
            output_path = (
                args.output_root
                / variant
                / f"repeat_{repeat + 1}"
                / scene.dataset.replace(" ", "_")
                / scene.name
            )
            metadata_path = output_path / "metadata.json"
            try:
                if args.force or not metadata_path.is_file():
                    prepare_output_directory(
                        output_path,
                        args.output_root,
                        force=True,
                    )
                    command = build_train_command(
                        scene=scene,
                        source_path=args.data_root / scene.relative_path,
                        output_path=output_path,
                        method=args.method,
                        ablations=args.ablate,
                        seed=seed,
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
                    scene=scene,
                    output_path=output_path,
                    repeat=repeat + 1,
                    seed=seed,
                    expected_fingerprint=config.fingerprint,
                )
                with lock:
                    rows.append(row)
                    print(
                        f"[GPU {gpu}] {scene.name}: "
                        f"PSNR={row['PSNR']:.3f}, "
                        f"SSIM={row['SSIM']:.4f}, "
                        f"LPIPS={row['LPIPS']:.4f}"
                    )
            except Exception as error:
                with lock:
                    errors.append(f"{scene.name} repeat {repeat + 1}: {error}")
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

    rows.sort(key=lambda row: (int(row["repeat"]), str(row["dataset"]), str(row["scene"])))
    _write_csv(args.output_root / "scene_metrics.csv", rows)
    _write_csv(
        args.output_root / "dataset_macro_metrics.csv",
        aggregate_dataset_rows(rows),
    )
    if errors:
        raise RuntimeError("Benchmark failures:\n" + "\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
