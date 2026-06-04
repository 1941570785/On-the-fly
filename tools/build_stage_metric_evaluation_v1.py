#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aligned_policy.stage_metrics import (  # noqa: E402
    build_stage_metric_evaluation,
    load_lifecycle_csv,
    write_csv,
    write_json,
)


STAGE_FIELDS = [
    "stage_name",
    "frame_count",
    "final_keyframe_count",
    "pose_attempt_count",
    "pose_success_count",
    "recovery_attempt_count",
    "true_source_materialized_count",
    "psnr_mean",
    "ssim_mean",
    "lpips_mean",
    "absolute_relative_translation_error_mean",
    "absolute_relative_rotation_error_mean",
    "valid_2d3d_median",
    "pnp_inliers_median",
    "miniba_inliers_median",
    "risk_score_mean",
    "visibility_score_mean",
    "quality_proxy_score_mean",
    "lifecycle_state_counts",
    "risk_bucket_counts",
]

FRAME_FIELDS = [
    "stage_name",
    "frame_id",
    "current_frame_id",
    "action",
    "lifecycle_state",
    "risk_bucket",
    "recoverability_bucket",
    "final_keyframe_count",
    "pose_attempt_count",
    "pose_success_count",
    "recovery_attempt_count",
    "true_source_materialized_count",
    "valid_2d3d",
    "pnp_inliers",
    "miniba_inliers",
    "psnr",
    "ssim",
    "lpips",
    "absolute_relative_translation_error",
    "absolute_relative_rotation_error",
    "risk_score",
    "visibility_score",
    "quality_proxy_score",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lifecycle_csv")
    parser.add_argument("--run_quality_json")
    args = parser.parse_args(argv)

    trace_path = Path(args.trace_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    lifecycle_rows = load_lifecycle_csv(args.lifecycle_csv)
    run_quality = _load_optional_json(args.run_quality_json)

    evaluation = build_stage_metric_evaluation(
        trace,
        lifecycle_rows=lifecycle_rows,
        run_quality=run_quality,
    )

    write_json(output_dir / "stage_metric_summary.json", evaluation)
    write_csv(output_dir / "stage_metric_table.csv", evaluation["stage_rows"], STAGE_FIELDS)
    write_csv(output_dir / "stage_metric_frame_table.csv", evaluation["frame_rows"], FRAME_FIELDS)
    (output_dir / "stage_metric_report.md").write_text(
        _render_report(evaluation),
        encoding="utf-8",
    )
    return 0


def _load_optional_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _render_report(evaluation: dict[str, Any]) -> str:
    summary = evaluation.get("overall_summary", {}) or {}
    lines = [
        "# Stage Metric Evaluation",
        "",
        "## Contract",
        "",
        f"- Online quality metric fields: {evaluation.get('metric_contract', {}).get('online_quality_metric_fields', [])}",
        f"- Offline stage metric fields: {evaluation.get('metric_contract', {}).get('offline_stage_metric_fields', [])}",
        f"- Online decision quality fields seen: {evaluation.get('online_decision_metric_fields_seen', [])}",
        "",
        "## Overall",
        "",
        f"- Frames: {summary.get('frame_count', 0)}",
        f"- Final keyframes: {summary.get('final_keyframe_count', 0)}",
        f"- Keyframe gap p90: {summary.get('keyframe_gap_p90', 0.0)}",
        f"- Recovery attempts: {summary.get('recovery_attempt_count', 0)}",
        f"- True source materialized: {summary.get('true_source_materialized_count', 0)}",
        "",
        "## Stage Rows",
        "",
        "| stage | frames | final kfs | PSNR | SSIM | LPIPS | t err | R err | pnp med | miniba med |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in evaluation.get("stage_rows", []):
        lines.append(
            "| {stage} | {frames} | {kfs} | {psnr} | {ssim} | {lpips} | {t_err} | {r_err} | {pnp} | {miniba} |".format(
                stage=row.get("stage_name", ""),
                frames=row.get("frame_count", 0),
                kfs=row.get("final_keyframe_count", 0),
                psnr=_fmt(row.get("psnr_mean")),
                ssim=_fmt(row.get("ssim_mean")),
                lpips=_fmt(row.get("lpips_mean")),
                t_err=_fmt(row.get("absolute_relative_translation_error_mean")),
                r_err=_fmt(row.get("absolute_relative_rotation_error_mean")),
                pnp=_fmt(row.get("pnp_inliers_median")),
                miniba=_fmt(row.get("miniba_inliers_median")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
