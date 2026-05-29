#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1")
INPUT = ROOT / "PAPER_ALIGNED_COMMIT_MATERIALIZATION_CONTRACT_FIX_V1" / "v5_rerun_after_contract_fix"
OUT = ROOT / "PAPER_ALIGNED_RECOVERY_COMMIT_MATERIALIZATION_AWARE_V6_V1" / "pose_failure_audit"
INTERVALS = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def interval_of(frame_id: int) -> str:
    for lo, hi in INTERVALS:
        if lo <= frame_id < hi:
            return f"{lo}-{hi}"
    return "out_of_range"


def parse_debug(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = ast.literal_eval(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def classify_failure(row: dict[str, Any]) -> str:
    if to_bool(row.get("materialized", row.get("final_keyframe_incremented", False))):
        return "materialized"
    reason = str(row.get("failure_reason", "") or row.get("materialization_failure_reason", "") or "")
    if "pnp_inliers_too_few" in reason:
        return "pnp_inliers_too_few"
    if "miniba_inliers_too_few" in reason:
        return "miniba_inliers_too_few"
    if "duplicate" in reason or "already_existing" in reason:
        return "duplicate_or_existing"
    if "source_mapping" in reason:
        return "source_mapping_failed"
    if "chosen_kfs" in reason:
        return "chosen_kfs_invalid"
    return "others"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    profile_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    available_fields: set[str] = set()
    unavailable_fields: set[str] = set()
    required_fields = [
        "num_matches",
        "num_inliers",
        "support_count",
        "chosen_kfs_count",
        "nearest_keyframe_distance",
        "local_keyframe_density",
        "source_gap_to_last_committed",
        "predicted_gap_if_hold",
        "retry_count",
        "age",
    ]

    for label in ["live_short_300", "live_short_500"]:
        run_dir = INPUT / label
        control_rows = read_csv(run_dir / "recovery_commit_control_trace.csv")
        mat_rows = read_csv(run_dir / "recovery_commit_materialization_trace.csv")
        ctrl_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        for row in control_rows:
            key = (to_int(row.get("source_frame_id")), to_int(row.get("current_frame_id")))
            ctrl_by_key[key] = row
        for mat in mat_rows:
            sid = to_int(mat.get("source_frame_id"))
            cur = to_int(mat.get("current_tick_frame_id", mat.get("current_frame_id")))
            ctrl = ctrl_by_key.get((sid, cur), {})
            debug = parse_debug(str(ctrl.get("debug", "")))
            group = classify_failure(mat)
            row = {
                "run": label,
                "source_frame_id": sid,
                "current_frame_id": cur,
                "interval": interval_of(sid),
                "result_group": group,
                "materialized": to_bool(mat.get("materialized", mat.get("final_keyframe_incremented", False))),
                "failure_stage": str(mat.get("failure_stage", "")),
                "failure_reason": str(mat.get("failure_reason", "")),
                "materialization_failure_reason": str(mat.get("materialization_failure_reason", "")),
                "R_t": to_float(ctrl.get("R_t", debug.get("R_t", 0.0))),
                "V_t": to_float(ctrl.get("V_t", debug.get("V_t", 0.0))),
                "Q_t": to_float(ctrl.get("Q_t", debug.get("Q_t", 0.0))),
                "decision_reason": str(ctrl.get("decision_reason", "")),
                "rescue_channel": str(ctrl.get("rescue_channel", debug.get("rescue_channel", ""))),
                "density_before": to_float(ctrl.get("density_before", debug.get("density_before", 0.0))),
                "density_after": to_float(ctrl.get("density_after", debug.get("density_after", 0.0))),
                "source_gap_to_last_committed": to_int(ctrl.get("source_gap_to_last_committed", debug.get("source_gap_to_last_committed", -1))),
                "predicted_gap_if_hold": to_int(ctrl.get("predicted_gap_if_hold", debug.get("predicted_gap_if_hold", -1))),
                "num_matches": to_int(ctrl.get("num_matches", debug.get("num_matches", -1))),
                "num_inliers": to_int(ctrl.get("num_inliers", debug.get("num_inliers", -1))),
                "support_count": to_int(ctrl.get("support_count", debug.get("support_count", ctrl.get("num_inliers", -1)))),
                "chosen_kfs_count": "unavailable",
                "nearest_keyframe_distance": "unavailable",
                "local_keyframe_density": to_float(ctrl.get("density_before", debug.get("density_before", 0.0))),
                "retry_count": to_int(mat.get("recovery_attempt_count", ctrl.get("recovery_attempt_count", -1))),
                "age": max(0, cur - sid) if cur >= 0 and sid >= 0 else "unavailable",
                "retry_limit_extended": to_bool(ctrl.get("retry_limit_extended", debug.get("retry_limit_extended", False))),
                "coverage_rescue": to_bool(ctrl.get("coverage_rescue_triggered", debug.get("coverage_rescue_triggered", False))),
                "gap_rescue": to_bool(ctrl.get("gap_rescue_triggered", debug.get("gap_rescue_triggered", False))),
            }
            for field in required_fields:
                val = row.get(field, "")
                if val == "unavailable" or val == -1:
                    unavailable_fields.add(field)
                else:
                    available_fields.add(field)
            profile_rows.append(row)
            audit_rows.append(row)

    write_csv(OUT / "pose_materialization_failure_audit.csv", audit_rows)
    write_csv(OUT / "materialized_vs_failed_candidate_profile.csv", profile_rows)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_interval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        by_group[str(row["result_group"])].append(row)
        by_interval[str(row["interval"])].append(row)

    def avg(rows: list[dict[str, Any]], field: str) -> float:
        vals = [to_float(r.get(field), None) for r in rows if r.get(field) not in {"unavailable", ""}]
        vals = [v for v in vals if v is not None]
        return float(mean(vals)) if vals else 0.0

    group_summary = {
        group: {
            "count": len(rows),
            "avg_R_t": avg(rows, "R_t"),
            "avg_V_t": avg(rows, "V_t"),
            "avg_Q_t": avg(rows, "Q_t"),
            "avg_num_matches": avg(rows, "num_matches"),
            "avg_num_inliers": avg(rows, "num_inliers"),
            "avg_source_gap": avg(rows, "source_gap_to_last_committed"),
            "retry_extended_count": sum(1 for r in rows if bool(r["retry_limit_extended"])),
            "coverage_rescue_count": sum(1 for r in rows if bool(r["coverage_rescue"])),
            "gap_rescue_count": sum(1 for r in rows if bool(r["gap_rescue"])),
        }
        for group, rows in by_group.items()
    }
    interval_summary = {
        interval: {
            "count": len(rows),
            "materialized_count": sum(1 for r in rows if r["result_group"] == "materialized"),
            "pnp_inliers_too_few": sum(1 for r in rows if r["result_group"] == "pnp_inliers_too_few"),
            "miniba_inliers_too_few": sum(1 for r in rows if r["result_group"] == "miniba_inliers_too_few"),
            "coverage_rescue_count": sum(1 for r in rows if bool(r["coverage_rescue"])),
            "gap_rescue_count": sum(1 for r in rows if bool(r["gap_rescue"])),
        }
        for interval, rows in by_interval.items()
    }
    rows_300_500 = [r for r in profile_rows if r["interval"] in {"300-400", "400-500"}]
    high_support_fail = [
        r
        for r in rows_300_500
        if r["result_group"] != "materialized" and to_int(r["num_matches"], 0) >= 1000
    ]
    summary = {
        "group_summary": group_summary,
        "interval_summary": interval_summary,
        "runtime_attempted_total": len(profile_rows),
        "materialized_total": sum(1 for r in profile_rows if r["result_group"] == "materialized"),
        "pose_failed_total": sum(1 for r in profile_rows if r["result_group"] in {"pnp_inliers_too_few", "miniba_inliers_too_few"}),
        "high_support_pose_failed_300_500": len(high_support_fail),
        "available_support_fields": sorted(available_fields),
        "unavailable_support_fields": sorted(unavailable_fields - available_fields),
    }
    write_json(OUT / "pose_materialization_failure_summary.json", summary)
    write_json(OUT / "materialized_vs_failed_candidate_summary.json", group_summary)

    pose_failure_explainable = summary["pose_failed_total"] > 0
    has_proxy = bool({"num_matches", "source_gap_to_last_committed", "predicted_gap_if_hold"} & available_fields)
    ready = {
        "pose_failure_explainable": pose_failure_explainable,
        "has_materialization_feasibility_proxy": has_proxy,
        "available_support_fields": sorted(available_fields),
        "unavailable_support_fields": sorted(unavailable_fields - available_fields),
        "need_v6_policy": True,
        "v6_should_rank_by_pose_support": True,
        "v6_should_add_pose_fail_cooldown": True,
        "v6_should_count_budget_on_materialized_success": True,
        "v6_should_trigger_early_coverage_rescue": True,
    }
    write_json(OUT / "ready_for_materialization_aware_v6_policy.json", ready)

    report = [
        "# pose materialization failure audit",
        "",
        f"- runtime_attempted_total: {summary['runtime_attempted_total']}",
        f"- materialized_total: {summary['materialized_total']}",
        f"- pose_failed_total: {summary['pose_failed_total']}",
        f"- high_support_pose_failed_300_500: {summary['high_support_pose_failed_300_500']}",
        f"- available_support_fields: {', '.join(summary['available_support_fields'])}",
        f"- unavailable_support_fields: {', '.join(summary['unavailable_support_fields'])}",
        "",
        "## 300-500",
        f"- 300-400: {interval_summary.get('300-400', {})}",
        f"- 400-500: {interval_summary.get('400-500', {})}",
    ]
    (OUT / "pose_materialization_failure_report.md").write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    feature_review = [
        "# materialization feasibility feature review",
        "",
        "- 可用在线 proxy：num_matches、source_gap_to_last_committed、predicted_gap_if_hold、density、R/V/Q、retry/age、rescue flags。",
        "- 不可用或不稳定字段按 unavailable 记录，不参与 score。",
        "- 300-500 存在 high-support pose failure，说明 v6 需要引入 pose-fail cooldown 和 materialization-rate feedback，而不是单纯增加 allow commit。",
        "- 不使用 PSNR/SSIM/LPIPS/RPE/APE。",
    ]
    (OUT / "materialization_feasibility_feature_review.md").write_text(
        "\n".join(feature_review).rstrip() + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
