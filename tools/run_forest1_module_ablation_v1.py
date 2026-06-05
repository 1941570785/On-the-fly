#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE = {
    "PSNR": 18.340284754832584,
    "SSIM": 0.5195986876885096,
    "LPIPS": 0.3996109717835983,
    "R°": 1.7395496102507535,
    "t": 0.23227432370185852,
    "num anchors": 4,
    "num keyframes": 343,
}

SUMMARY_FIELDS = [
    "name",
    "returncode",
    "max_frames",
    "num anchors",
    "num keyframes",
    "PSNR",
    "SSIM",
    "LPIPS",
    "R°",
    "t",
    "psnr_delta_to_baseline",
    "ssim_delta_to_baseline",
    "lpips_delta_to_baseline",
    "r_delta_to_baseline",
    "t_delta_to_baseline",
    "stage_metric_dir",
    "interval_metric_dir",
    "model_dir",
    "log_path",
]


@dataclass(frozen=True)
class AblationSpec:
    name: str
    model_dir: Path
    extra_args: list[str]


def build_ablation_specs(
    output_root: Path,
    max_frames: int = 300,
    direct_density_upper_per_100: float | None = None,
    direct_density_hard_upper_per_100: float | None = None,
) -> list[AblationSpec]:
    specs: list[tuple[str, list[str]]] = [
        (
            "A0_baseline_passthrough",
            [
                "--risk_admission_mode", "paper_aligned_baseline_passthrough",
            ],
        ),
        (
            "A1_risk_only_no_pool",
            [
                "--risk_admission_mode", "on_the_fly_innovation_v1",
                "--paper_aligned_semantic_recovery_max_attempts", "0",
                "--paper_aligned_semantic_recovery_attempts_per_tick", "0",
                "--paper_aligned_recovery_commit_bridge", "off",
                "--paper_aligned_defer_recovery_support_bridge", "off",
                "--paper_aligned_recovery_commit_control", "off",
                "--paper_aligned_direct_density_control", "off",
            ],
        ),
        (
            "A2_risk_plus_pool_no_commit",
            [
                "--risk_admission_mode", "on_the_fly_innovation_v1",
                "--paper_aligned_recovery_commit_bridge", "off",
                "--paper_aligned_defer_recovery_support_bridge", "off",
                "--paper_aligned_recovery_commit_control", "off",
                "--paper_aligned_direct_density_control", "off",
            ],
        ),
        (
            "A3_true_source_commit_no_control",
            [
                "--risk_admission_mode", "on_the_fly_innovation_v1",
                "--paper_aligned_recovery_commit_bridge", "true_source_commit",
                "--paper_aligned_defer_recovery_support_bridge", "v1",
                "--paper_aligned_recovery_commit_control", "off",
                "--paper_aligned_direct_density_control", "off",
            ],
        ),
        (
            "A4_commit_control_no_density",
            [
                "--risk_admission_mode", "on_the_fly_innovation_v1",
                "--paper_aligned_recovery_commit_bridge", "true_source_commit",
                "--paper_aligned_defer_recovery_support_bridge", "v1",
                "--paper_aligned_recovery_commit_control", "recovery_commit_early_seed_v7",
                "--paper_aligned_direct_density_control", "off",
            ],
        ),
        (
            "A5_optional_commit_control_plus_density",
            [
                "--risk_admission_mode", "on_the_fly_innovation_v1",
                "--paper_aligned_recovery_commit_bridge", "true_source_commit",
                "--paper_aligned_defer_recovery_support_bridge", "v1",
                "--paper_aligned_recovery_commit_control", "recovery_commit_early_seed_v7",
                "--paper_aligned_direct_density_control", "target_band_v2_2_2_1",
            ],
        ),
    ]
    if direct_density_upper_per_100 is not None or direct_density_hard_upper_per_100 is not None:
        for name, extra_args in specs:
            if name != "A5_optional_commit_control_plus_density":
                continue
            if direct_density_upper_per_100 is not None:
                extra_args.extend(
                    [
                        "--paper_aligned_direct_density_upper_per_100",
                        str(float(direct_density_upper_per_100)),
                    ]
                )
            if direct_density_hard_upper_per_100 is not None:
                extra_args.extend(
                    [
                        "--paper_aligned_direct_density_hard_upper_per_100",
                        str(float(direct_density_hard_upper_per_100)),
                    ]
                )
            break
    return [AblationSpec(name, output_root / name / f"short{max_frames}", args) for name, args in specs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/StaticHikes/forest1")
    parser.add_argument("--output_root", default="results/StaticHikes/forest1/NATURAL_ORDER_MODULE_ABLATION_20260605")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--test_hold", type=int, default=10)
    parser.add_argument("--test_frequency", type=int, default=20)
    parser.add_argument("--baseline_json", default="results/StaticHikes/forest1/compatible/metadata.json")
    parser.add_argument("--direct_density_upper_per_100", type=float, default=None)
    parser.add_argument("--direct_density_hard_upper_per_100", type=float, default=None)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    specs = build_ablation_specs(
        output_root,
        max_frames=args.max_frames,
        direct_density_upper_per_100=args.direct_density_upper_per_100,
        direct_density_hard_upper_per_100=args.direct_density_hard_upper_per_100,
    )
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in specs if spec.name in wanted]

    baseline = _load_json(Path(args.baseline_json)) or dict(BASELINE)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "baseline_reference.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_one(spec, args, baseline)
        summary_rows.append(row)
        write_summary(output_root / "module_ablation_summary.csv", summary_rows)
        (output_root / "module_ablation_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(int(row.get("returncode", 1)) == 0 for row in summary_rows) else 1


def run_one(spec: AblationSpec, args: argparse.Namespace, baseline: dict[str, Any]) -> dict[str, Any]:
    model_dir = spec.model_dir
    trace_path = model_dir / "semantic_trace.json"
    log_path = model_dir / "train.log"
    metadata_path = model_dir / "metadata.json"
    if model_dir.exists() and not args.skip_existing:
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and metadata_path.exists():
        return summarize(spec, args, baseline, returncode=0)

    cmd = [
        sys.executable,
        "train.py",
        "-s", str(args.source),
        "-m", str(model_dir),
        "--test_hold", str(args.test_hold),
        "--test_frequency", str(args.test_frequency),
        "--max_frames", str(args.max_frames),
        "--paper_aligned_contract_trace_path", str(trace_path),
        *spec.extra_args,
    ]
    (model_dir / "command.json").write_text(json.dumps(cmd, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    stage_dir = model_dir / "stage_metrics"
    if trace_path.exists():
        stage_cmd = [
            sys.executable,
            "tools/build_stage_metric_evaluation_v1.py",
            "--trace_json", str(trace_path),
            "--output_dir", str(stage_dir),
            "--run_quality_json", str(metadata_path),
        ]
        frame_metrics = model_dir / "frame_metrics.csv"
        if frame_metrics.exists():
            stage_cmd.extend(["--frame_metrics_csv", str(frame_metrics)])
        subprocess.run(stage_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_metrics = model_dir / "frame_metrics.csv"
    if frame_metrics.exists():
        interval_dir = model_dir / "interval_metrics"
        interval_cmd = [
            sys.executable,
            "tools/build_interval_metric_evaluation_v1.py",
            "--frame_metrics_csv", str(frame_metrics),
            "--baseline_json", str(args.baseline_json),
            "--output_dir", str(interval_dir),
            "--interval_size", "100",
        ]
        subprocess.run(interval_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return summarize(spec, args, baseline, returncode=proc.returncode)


def summarize(spec: AblationSpec, args: argparse.Namespace, baseline: dict[str, Any], returncode: int) -> dict[str, Any]:
    model_dir = spec.model_dir
    metadata = _load_json(model_dir / "metadata.json")
    frame_metric_means = _load_frame_metric_means(model_dir / "frame_metrics.csv")
    row: dict[str, Any] = {
        "name": spec.name,
        "returncode": int(returncode),
        "max_frames": int(args.max_frames),
        "stage_metric_dir": str(model_dir / "stage_metrics") if (model_dir / "stage_metrics").exists() else "",
        "interval_metric_dir": str(model_dir / "interval_metrics") if (model_dir / "interval_metrics").exists() else "",
        "model_dir": str(model_dir),
        "log_path": str(model_dir / "train.log"),
    }
    for key in ["num anchors", "num keyframes", "PSNR", "SSIM", "LPIPS", "R°", "t"]:
        row[key] = metadata.get(key, "")
    if row["R°"] == "":
        row["R°"] = frame_metric_means.get("R°", "")
    if row["t"] == "":
        row["t"] = frame_metric_means.get("t", "")
    row["psnr_delta_to_baseline"] = _delta(row.get("PSNR"), baseline.get("PSNR"))
    row["ssim_delta_to_baseline"] = _delta(row.get("SSIM"), baseline.get("SSIM"))
    row["lpips_delta_to_baseline"] = _delta(row.get("LPIPS"), baseline.get("LPIPS"))
    row["r_delta_to_baseline"] = _delta(row.get("R°"), baseline.get("R°", baseline.get("R_deg")))
    row["t_delta_to_baseline"] = _delta(row.get("t"), baseline.get("t"))
    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_frame_metric_means(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    totals = {"t": 0.0, "R°": 0.0}
    counts = {"t": 0, "R°": 0}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for output_key, csv_key in (("t", "abs_trans_error"), ("R°", "abs_rot_error_deg")):
                value = row.get(csv_key, "")
                if value == "":
                    continue
                try:
                    parsed = float(value)
                except ValueError:
                    continue
                totals[output_key] += parsed
                counts[output_key] += 1
    return {key: totals[key] / counts[key] for key in totals if counts[key] > 0}


def _delta(value: Any, baseline: Any) -> float | str:
    try:
        return float(value) - float(baseline)
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
