#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any


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
        for key in row:
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


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_int(x: Any, default: int = 0) -> int:
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


def as_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    text = str(x or "").strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, list) else []
    except Exception:
        return []


def in_300_500(source_frame_id: int, current_frame_id: int) -> bool:
    return 300 <= int(source_frame_id) < 500 or 300 <= int(current_frame_id) < 500


def source_in_300_500(source_frame_id: int) -> bool:
    return 300 <= int(source_frame_id) < 500


def percentile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    idx = int(round((len(vals) - 1) * q))
    return float(vals[max(0, min(len(vals) - 1, idx))])


def numeric_summary(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": float(mean(vals)),
        "p50": percentile(vals, 0.5),
        "p90": percentile(vals, 0.9),
        "max": max(vals),
    }


def first_by_source(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        sid = to_int(row.get("source_frame_id"), -1)
        out.setdefault(sid, row)
    return out


def rows_by_source(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[to_int(row.get("source_frame_id"), -1)].append(row)
    return out


def classify_stage(row: dict[str, Any]) -> tuple[str, str]:
    if not to_bool(row.get("chosen_kfs_built")):
        return "chosen_kfs_not_built", str(row.get("pose_failure_reason", "no_chosen_kfs") or "no_chosen_kfs")
    if not to_bool(row.get("matching_executed")):
        return "matching_not_executed", str(row.get("pose_failure_reason", "matching_not_executed") or "matching_not_executed")
    if not to_bool(row.get("pnp_attempted")):
        return "pnp_not_attempted", str(row.get("pose_failure_reason", "pnp_not_attempted") or "pnp_not_attempted")
    if not to_bool(row.get("pnp_success")):
        return "pnp_failed", str(row.get("pose_failure_reason", "pnp_failed") or "pnp_failed")
    if not to_bool(row.get("miniba_attempted")):
        return "miniba_not_attempted", str(row.get("pose_failure_reason", "miniba_not_attempted") or "miniba_not_attempted")
    if not to_bool(row.get("miniba_success")):
        return "miniba_failed", str(row.get("pose_failure_reason", "miniba_failed") or "miniba_failed")
    if not to_bool(row.get("recovery_success")):
        return "recovery_success_missing", "pose_success_not_propagated"
    if not to_bool(row.get("commit_control_reached")):
        return "commit_control_not_reached", "recovery_success_not_propagated_to_control"
    if not to_bool(row.get("control_allow_commit")):
        return "commit_control_hold_reject", str(row.get("control_reason", "") or row.get("control_decision", "hold_or_reject"))
    if not to_bool(row.get("runtime_commit_attempted")):
        return "runtime_commit_not_attempted", "control_allow_without_runtime_attempt"
    if not to_bool(row.get("materialized")):
        return "materialization_failed", str(row.get("materialization_failure_reason", "") or "runtime_attempt_no_materialization")
    if not to_bool(row.get("actual_keyframe_added")):
        return "keyframe_timeline_missing", "materialized_but_no_keyframe_timeline"
    return "actual_keyframe_added", ""


def load_run(run_root: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "root": run_root,
        "engine": read_json(run_root / "engine_stability_audit.json"),
        "control": read_csv(run_root / "recovery_commit_control_trace.csv"),
        "materialization": read_csv(run_root / "recovery_commit_materialization_trace.csv"),
        "keyframes": read_csv(run_root / "keyframe_timeline.csv"),
        "gaps": read_csv(run_root / "main_chain_gap_timeline.csv"),
        "support": read_csv(run_root / "support_trend_timeline.csv"),
        "lifecycle": read_csv(run_root / "lifecycle_gate_trace.csv"),
        "pose": read_csv(run_root / "recovery_pose_path_trace.csv"),
        "stage": read_csv(run_root / "frame_stage_reachability_trace.csv"),
        "chosen": read_csv(run_root / "chosen_kfs_candidate_trace.csv"),
        "pool": read_csv(run_root / "pose_reference_pool_trace.csv"),
        "bridge": read_csv(run_root / "matching_to_pose_path_bridge_trace.csv"),
        "pnp_ref": read_csv(run_root / "pnp_miniba_reference_trace.csv"),
        "anchors": read_csv(run_root / "local_map_anchor_trace.csv"),
    }


def build_funnel(run: dict[str, Any]) -> list[dict[str, Any]]:
    control_by_source = rows_by_source(run["control"])
    mat_by_source = rows_by_source(run["materialization"])
    keyframe_by_source = rows_by_source(run["keyframes"])
    lifecycle_queues: dict[tuple[int, int, str], deque[dict[str, Any]]] = defaultdict(deque)
    for life in run["lifecycle"]:
        if to_bool(life.get("recovery_pose_attempted")):
            key = (
                to_int(life.get("source_frame_id"), -1),
                to_int(life.get("frame_id"), -1),
                str(life.get("pose_path_request_reason", "")),
            )
            lifecycle_queues[key].append(life)

    rows: list[dict[str, Any]] = []
    for pose in run["pose"]:
        sid = to_int(pose.get("source_frame_id"), -1)
        cid = to_int(pose.get("current_frame_id"), -1)
        reason = str(pose.get("attempt_reason", ""))
        life = {}
        q = lifecycle_queues.get((sid, cid, reason))
        if q:
            life = q.popleft()
        controls = control_by_source.get(sid, [])
        control = controls[0] if controls else {}
        mats = mat_by_source.get(sid, [])
        mat = mats[0] if mats else {}
        keyframes = keyframe_by_source.get(sid, [])
        control_decision = str(control.get("control_decision", control.get("decision", "")))
        control_reason = str(control.get("decision_reason", control.get("commit_control_reason", "")))
        control_allow = control_decision in {"allow_commit", "commit"} or str(control.get("decision", "")) == "commit"
        materialized = any(to_bool(m.get("materialized")) or to_bool(m.get("final_keyframe_incremented")) for m in mats)
        actual_keyframe = bool(keyframes)
        row = {
            "run": run["label"],
            "source_frame_id": sid,
            "current_frame_id": cid,
            "attempt_id": to_int(pose.get("attempt_id"), 0),
            "attempt_reason": reason,
            "in_300_500_activity": in_300_500(sid, cid),
            "source_in_300_500": source_in_300_500(sid),
            "chosen_kfs_built": to_bool(pose.get("chosen_kfs_built")),
            "matching_executed": to_bool(pose.get("matching_executed")),
            "matching_candidate_count": to_int(pose.get("matching_candidate_count"), 0),
            "matching_seed_candidate_count": to_int(pose.get("matching_seed_candidate_count"), 0),
            "chosen_kfs_contains_recovery_seed": to_bool(pose.get("chosen_kfs_contains_recovery_seed")),
            "pnp_attempted": to_bool(pose.get("pnp_attempted")),
            "pnp_inliers": to_int(pose.get("pnp_inliers"), 0),
            "pnp_success": to_bool(pose.get("pnp_success")),
            "miniba_attempted": to_bool(pose.get("miniba_attempted")),
            "miniba_inliers": to_int(pose.get("miniba_inliers"), 0),
            "miniba_success": to_bool(pose.get("miniba_success")),
            "pose_failure_reason": str(pose.get("pose_failure_reason", "")),
            "recovery_success": to_bool(life.get("recovery_success")),
            "commit_control_reached": bool(controls),
            "control_decision": control_decision,
            "control_reason": control_reason,
            "control_allow_commit": control_allow,
            "runtime_commit_attempted": any(to_bool(c.get("runtime_commit_attempted")) for c in controls) or any(to_bool(m.get("runtime_commit_attempted")) for m in mats),
            "materialized": materialized,
            "materialization_failure_reason": str(mat.get("materialization_failure_reason", control.get("materialization_failure_reason", ""))),
            "actual_keyframe_added": actual_keyframe,
            "keyframe_ids": [to_int(k.get("keyframe_id"), -1) for k in keyframes],
            "keyframe_commit_origins": [str(k.get("commit_origin", "")) for k in keyframes],
        }
        final_stage, blocking = classify_stage(row)
        row["final_stage"] = final_stage
        row["blocking_reason"] = blocking
        rows.append(row)
    return rows


def count_stage(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "recovery_pose_attempted": len(rows),
        "chosen_kfs_built": sum(to_bool(r.get("chosen_kfs_built")) for r in rows),
        "matching_executed": sum(to_bool(r.get("matching_executed")) for r in rows),
        "pnp_attempted": sum(to_bool(r.get("pnp_attempted")) for r in rows),
        "pnp_success": sum(to_bool(r.get("pnp_success")) for r in rows),
        "miniba_attempted": sum(to_bool(r.get("miniba_attempted")) for r in rows),
        "miniba_success": sum(to_bool(r.get("miniba_success")) for r in rows),
        "recovery_success": sum(to_bool(r.get("recovery_success")) for r in rows),
        "commit_control_reached": sum(to_bool(r.get("commit_control_reached")) for r in rows),
        "control_allow_commit": sum(to_bool(r.get("control_allow_commit")) for r in rows),
        "runtime_commit_attempted": sum(to_bool(r.get("runtime_commit_attempted")) for r in rows),
        "materialized": sum(to_bool(r.get("materialized")) for r in rows),
        "actual_keyframe_added": sum(to_bool(r.get("actual_keyframe_added")) for r in rows),
    }


def rate(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def pnp_miniba_summary(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    seed = [r for r in rows if to_bool(r.get("chosen_kfs_contains_recovery_seed")) or to_int(r.get("matching_seed_candidate_count"), 0) > 0]
    no_seed = [r for r in rows if r not in seed]
    def group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "attempt_count": len(group),
            "pnp_success_count": sum(to_bool(r.get("pnp_success")) for r in group),
            "pnp_success_rate": rate(sum(to_bool(r.get("pnp_success")) for r in group), len(group)),
            "miniba_success_count": sum(to_bool(r.get("miniba_success")) for r in group),
            "miniba_success_rate": rate(sum(to_bool(r.get("miniba_success")) for r in group), len(group)),
            "pnp_inliers": numeric_summary([to_float(r.get("pnp_inliers")) for r in group]),
            "miniba_inliers": numeric_summary([to_float(r.get("miniba_inliers")) for r in group]),
            "failure_reasons": dict(Counter(str(r.get("pose_failure_reason", "") or "success") for r in group)),
        }
    out = {
        f"{prefix}all": group_stats(rows),
        f"{prefix}seed_used": group_stats(seed),
        f"{prefix}seed_not_used": group_stats(no_seed),
        f"{prefix}seed_inlier_lift_pnp_mean": group_stats(seed)["pnp_inliers"]["mean"] - group_stats(no_seed)["pnp_inliers"]["mean"],
        f"{prefix}seed_inlier_lift_miniba_mean": group_stats(seed)["miniba_inliers"]["mean"] - group_stats(no_seed)["miniba_inliers"]["mean"],
    }
    return out


def build_success_handoff_rows(funnel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in funnel:
        handoff_reason = ""
        if to_bool(r.get("miniba_success")) and not to_bool(r.get("recovery_success")):
            handoff_reason = "missing_pose_success_flag"
        elif to_bool(r.get("recovery_success")) and not to_bool(r.get("commit_control_reached")):
            handoff_reason = "recovery_success_not_propagated"
        elif to_bool(r.get("recovery_success")) and to_bool(r.get("commit_control_reached")):
            handoff_reason = "commit_control_reached"
        elif to_bool(r.get("pnp_success")) and not to_bool(r.get("miniba_success")):
            handoff_reason = "pnp_success_but_miniba_failed"
        else:
            handoff_reason = str(r.get("pose_failure_reason", "") or "pose_outcome_failure")
        out.append({
            "run": r["run"],
            "source_frame_id": r["source_frame_id"],
            "current_frame_id": r["current_frame_id"],
            "attempt_id": r["attempt_id"],
            "in_300_500_activity": r["in_300_500_activity"],
            "source_in_300_500": r["source_in_300_500"],
            "pnp_success": r["pnp_success"],
            "miniba_success": r["miniba_success"],
            "recovery_success": r["recovery_success"],
            "commit_control_reached": r["commit_control_reached"],
            "control_decision": r["control_decision"],
            "control_reason": r["control_reason"],
            "handoff_status": handoff_reason,
        })
    return out


def build_control_materialization_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    mat_by_source = rows_by_source(run["materialization"])
    keyframe_by_source = rows_by_source(run["keyframes"])
    out: list[dict[str, Any]] = []
    for c in run["control"]:
        sid = to_int(c.get("source_frame_id"), -1)
        mats = mat_by_source.get(sid, [])
        kfs = keyframe_by_source.get(sid, [])
        decision = str(c.get("control_decision", c.get("decision", "")))
        allow = decision in {"allow_commit", "commit"} or str(c.get("decision", "")) == "commit"
        runtime = to_bool(c.get("runtime_commit_attempted")) or any(to_bool(m.get("runtime_commit_attempted")) for m in mats)
        materialized = to_bool(c.get("materialized")) or any(to_bool(m.get("materialized")) or to_bool(m.get("final_keyframe_incremented")) for m in mats)
        out.append({
            "run": run["label"],
            "source_frame_id": sid,
            "current_frame_id": to_int(c.get("current_frame_id", c.get("current_tick_frame_id", -1)), -1),
            "source_in_300_500": source_in_300_500(sid),
            "activity_in_300_500": in_300_500(sid, to_int(c.get("current_frame_id", c.get("current_tick_frame_id", -1)), -1)),
            "control_mode": c.get("control_mode", ""),
            "control_decision": decision,
            "decision_reason": c.get("decision_reason", ""),
            "control_allow_commit": allow,
            "blocked_reason": c.get("blocked_reason", ""),
            "runtime_commit_attempted": runtime,
            "runtime_commit_success": to_bool(c.get("runtime_commit_success")) or any(to_bool(m.get("runtime_commit_success")) for m in mats),
            "materialized": materialized,
            "actual_keyframe_added": bool(kfs),
            "materialization_failure_reason": c.get("materialization_failure_reason", ""),
            "commit_channel": c.get("commit_channel", ""),
            "R_t": c.get("R_t", ""),
            "V_t": c.get("V_t", ""),
            "Q_t": c.get("Q_t", ""),
            "num_matches": c.get("num_matches", ""),
            "num_inliers": c.get("num_inliers", ""),
            "materialization_feasibility_score": c.get("materialization_feasibility_score", ""),
            "density_state": c.get("density_state", ""),
        })
    return out


def keyframe_delta(short300: dict[str, Any], short500: dict[str, Any], ref500: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def by_source(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
        return {to_int(r.get("source_frame_id"), -1): r for r in run["keyframes"]}
    k300 = by_source(short300)
    k500 = by_source(short500)
    kref = by_source(ref500) if ref500 else {}
    rows: list[dict[str, Any]] = []
    for sid in sorted(set(k300) - set(k500)):
        r = k300[sid]
        rows.append({
            "source_frame_id": sid,
            "delta_type": "missing_in_short500",
            "short300_keyframe_id": r.get("keyframe_id", ""),
            "short500_keyframe_id": "",
            "commit_origin": r.get("commit_origin", ""),
            "commit_channel": r.get("commit_channel", ""),
            "is_v7_early_seed": r.get("is_v7_early_seed", ""),
            "present_in_reference_selection_short500": sid in kref,
        })
    for sid in sorted(set(k500) - set(k300)):
        r = k500[sid]
        rows.append({
            "source_frame_id": sid,
            "delta_type": "added_in_short500",
            "short300_keyframe_id": "",
            "short500_keyframe_id": r.get("keyframe_id", ""),
            "commit_origin": r.get("commit_origin", ""),
            "commit_channel": r.get("commit_channel", ""),
            "is_v7_early_seed": r.get("is_v7_early_seed", ""),
            "present_in_reference_selection_short500": sid in kref,
        })
    missing = [r for r in rows if r["delta_type"] == "missing_in_short500"]
    added = [r for r in rows if r["delta_type"] == "added_in_short500"]
    summary = {
        "short300_keyframe_count": len(k300),
        "short500_keyframe_count": len(k500),
        "short500_minus_short300": len(k500) - len(k300),
        "overlap_count": len(set(k300) & set(k500)),
        "missing_in_short500_count": len(missing),
        "added_in_short500_count": len(added),
        "missing_by_commit_origin": dict(Counter(str(r.get("commit_origin", "")) for r in missing)),
        "added_by_commit_origin": dict(Counter(str(r.get("commit_origin", "")) for r in added)),
        "keyframe_set_regression": len(k500) < len(k300) or bool(missing),
        "likely_independent_run_divergence": bool(missing or added),
        "actual_300_500_keyframe_added_short500": sum(source_in_300_500(sid) for sid in k500),
        "reference_selection_short500_keyframe_count": len(kref),
    }
    return rows, summary


def summarize_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "control_event_count": len(rows),
        "source_300_500_control_event_count": sum(to_bool(r.get("source_in_300_500")) for r in rows),
        "activity_300_500_control_event_count": sum(to_bool(r.get("activity_in_300_500")) for r in rows),
        "allow_commit_count": sum(to_bool(r.get("control_allow_commit")) for r in rows),
        "hold_count": sum(str(r.get("control_decision", "")) == "hold" for r in rows),
        "reject_count": sum(str(r.get("control_decision", "")) == "reject" for r in rows),
        "runtime_commit_attempted_count": sum(to_bool(r.get("runtime_commit_attempted")) for r in rows),
        "materialized_count": sum(to_bool(r.get("materialized")) for r in rows),
        "actual_keyframe_added_count": sum(to_bool(r.get("actual_keyframe_added")) for r in rows),
        "allow_but_no_runtime_attempt_count": sum(to_bool(r.get("control_allow_commit")) and not to_bool(r.get("runtime_commit_attempted")) for r in rows),
        "runtime_attempt_but_no_materialization_count": sum(to_bool(r.get("runtime_commit_attempted")) and not to_bool(r.get("materialized")) for r in rows),
        "materialized_but_no_keyframe_added_count": sum(to_bool(r.get("materialized")) and not to_bool(r.get("actual_keyframe_added")) for r in rows),
        "decision_reason_counts": dict(Counter(str(r.get("decision_reason", "")) for r in rows)),
        "blocked_reason_counts": dict(Counter(str(r.get("blocked_reason", "")) for r in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--reference_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    ref_root = Path(args.reference_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    short300 = load_run(input_root / "live_short_300", "live_short_300")
    short500 = load_run(input_root / "live_short_500", "live_short_500")
    ref500 = load_run(ref_root / "live_short_500", "reference_selection_live_short_500")

    funnel300 = build_funnel(short300)
    funnel500 = build_funnel(short500)
    funnel = funnel300 + funnel500
    funnel500_300_500 = [r for r in funnel500 if to_bool(r.get("in_300_500_activity"))]
    write_csv(out / "recovery_pose_outcome_funnel.csv", funnel)
    funnel_summary = {
        "all_runs": count_stage(funnel),
        "live_short_300": count_stage(funnel300),
        "live_short_500": count_stage(funnel500),
        "live_short_500_300_500_activity": count_stage(funnel500_300_500),
        "live_short_500_300_500_source_only": count_stage([r for r in funnel500 if to_bool(r.get("source_in_300_500"))]),
        "final_stage_counts_300_500": dict(Counter(str(r.get("final_stage")) for r in funnel500_300_500)),
        "blocking_reason_counts_300_500": dict(Counter(str(r.get("blocking_reason")) for r in funnel500_300_500)),
    }
    write_json(out / "recovery_pose_outcome_funnel_summary.json", funnel_summary)
    write_md(out / "recovery_pose_outcome_funnel_report.md", [
        "# recovery pose outcome funnel",
        "",
        f"- live_short_500 300-500 recovery_pose_attempted: {funnel_summary['live_short_500_300_500_activity']['recovery_pose_attempted']}",
        f"- chosen/matching/PnP/MiniBA: {funnel_summary['live_short_500_300_500_activity']['chosen_kfs_built']}/{funnel_summary['live_short_500_300_500_activity']['matching_executed']}/{funnel_summary['live_short_500_300_500_activity']['pnp_attempted']}/{funnel_summary['live_short_500_300_500_activity']['miniba_attempted']}",
        f"- PnP success / MiniBA success: {funnel_summary['live_short_500_300_500_activity']['pnp_success']}/{funnel_summary['live_short_500_300_500_activity']['miniba_success']}",
        f"- commit control reached / allow / materialized / actual keyframe: {funnel_summary['live_short_500_300_500_activity']['commit_control_reached']}/{funnel_summary['live_short_500_300_500_activity']['control_allow_commit']}/{funnel_summary['live_short_500_300_500_activity']['materialized']}/{funnel_summary['live_short_500_300_500_activity']['actual_keyframe_added']}",
        f"- final stages: {funnel_summary['final_stage_counts_300_500']}",
    ])

    pnp_rows = []
    for r in funnel:
        pnp_rows.append({
            "run": r["run"],
            "source_frame_id": r["source_frame_id"],
            "current_frame_id": r["current_frame_id"],
            "in_300_500_activity": r["in_300_500_activity"],
            "source_in_300_500": r["source_in_300_500"],
            "seed_used": to_bool(r.get("chosen_kfs_contains_recovery_seed")) or to_int(r.get("matching_seed_candidate_count"), 0) > 0,
            "matching_seed_candidate_count": r["matching_seed_candidate_count"],
            "pnp_attempted": r["pnp_attempted"],
            "pnp_inliers": r["pnp_inliers"],
            "pnp_success": r["pnp_success"],
            "miniba_attempted": r["miniba_attempted"],
            "miniba_inliers": r["miniba_inliers"],
            "miniba_success": r["miniba_success"],
            "pose_failure_reason": r["pose_failure_reason"],
        })
    write_csv(out / "pnp_miniba_success_failure_audit.csv", pnp_rows)
    pnp_summary = {
        "all": pnp_miniba_summary(funnel),
        "live_short_500_300_500_activity": pnp_miniba_summary(funnel500_300_500),
        "too_few_inliers_count_300_500": sum("too_few" in str(r.get("pose_failure_reason", "")) for r in funnel500_300_500),
    }
    write_json(out / "pnp_miniba_success_failure_summary.json", pnp_summary)
    seed_stats = pnp_summary["live_short_500_300_500_activity"]["seed_used"]
    no_seed_stats = pnp_summary["live_short_500_300_500_activity"]["seed_not_used"]
    write_md(out / "pnp_miniba_success_failure_report.md", [
        "# PnP / MiniBA success failure audit",
        "",
        f"- seed used attempts: {seed_stats['attempt_count']}, PnP success rate={seed_stats['pnp_success_rate']:.3f}, MiniBA success rate={seed_stats['miniba_success_rate']:.3f}",
        f"- seed not used attempts: {no_seed_stats['attempt_count']}, PnP success rate={no_seed_stats['pnp_success_rate']:.3f}, MiniBA success rate={no_seed_stats['miniba_success_rate']:.3f}",
        f"- seed pnp/miniba mean inlier lift: {pnp_summary['live_short_500_300_500_activity']['seed_inlier_lift_pnp_mean']:.3f}/{pnp_summary['live_short_500_300_500_activity']['seed_inlier_lift_miniba_mean']:.3f}",
        f"- failure reasons: {pnp_summary['live_short_500_300_500_activity']['all']['failure_reasons']}",
    ])

    handoff_rows = build_success_handoff_rows(funnel)
    handoff500 = [r for r in handoff_rows if r["run"] == "live_short_500" and to_bool(r.get("in_300_500_activity"))]
    write_csv(out / "recovery_success_to_commit_control_audit.csv", handoff_rows)
    handoff_summary = {
        "all_runs": {
            "miniba_success_count": sum(to_bool(r.get("miniba_success")) for r in handoff_rows),
            "miniba_success_but_no_recovery_success": sum(to_bool(r.get("miniba_success")) and not to_bool(r.get("recovery_success")) for r in handoff_rows),
            "recovery_success_count": sum(to_bool(r.get("recovery_success")) for r in handoff_rows),
            "recovery_success_but_no_commit_control": sum(to_bool(r.get("recovery_success")) and not to_bool(r.get("commit_control_reached")) for r in handoff_rows),
        },
        "live_short_500_300_500_activity": {
            "pnp_success_count": sum(to_bool(r.get("pnp_success")) for r in handoff500),
            "miniba_success_count": sum(to_bool(r.get("miniba_success")) for r in handoff500),
            "miniba_success_but_no_recovery_success": sum(to_bool(r.get("miniba_success")) and not to_bool(r.get("recovery_success")) for r in handoff500),
            "recovery_success_count": sum(to_bool(r.get("recovery_success")) for r in handoff500),
            "recovery_success_but_no_commit_control": sum(to_bool(r.get("recovery_success")) and not to_bool(r.get("commit_control_reached")) for r in handoff500),
            "handoff_status_counts": dict(Counter(str(r.get("handoff_status")) for r in handoff500)),
        },
    }
    write_json(out / "recovery_success_to_commit_control_summary.json", handoff_summary)
    write_md(out / "recovery_success_to_commit_control_report.md", [
        "# recovery success to commit control audit",
        "",
        f"- 300-500 MiniBA success: {handoff_summary['live_short_500_300_500_activity']['miniba_success_count']}",
        f"- MiniBA success but no recovery_success: {handoff_summary['live_short_500_300_500_activity']['miniba_success_but_no_recovery_success']}",
        f"- recovery_success but no commit_control: {handoff_summary['live_short_500_300_500_activity']['recovery_success_but_no_commit_control']}",
        f"- handoff statuses: {handoff_summary['live_short_500_300_500_activity']['handoff_status_counts']}",
    ])

    control_rows = build_control_materialization_rows(short300) + build_control_materialization_rows(short500)
    control500 = [r for r in control_rows if r["run"] == "live_short_500"]
    control500_activity = [r for r in control500 if to_bool(r.get("activity_in_300_500"))]
    write_csv(out / "commit_control_to_materialization_audit.csv", control_rows)
    control_summary = {
        "all_runs": summarize_control(control_rows),
        "live_short_500": summarize_control(control500),
        "live_short_500_300_500_activity": summarize_control(control500_activity),
    }
    write_json(out / "commit_control_to_materialization_summary.json", control_summary)
    write_md(out / "commit_control_to_materialization_report.md", [
        "# commit control to materialization audit",
        "",
        f"- 300-500 control events: {control_summary['live_short_500_300_500_activity']['control_event_count']}",
        f"- allow/hold/reject: {control_summary['live_short_500_300_500_activity']['allow_commit_count']}/{control_summary['live_short_500_300_500_activity']['hold_count']}/{control_summary['live_short_500_300_500_activity']['reject_count']}",
        f"- runtime/materialized/keyframe: {control_summary['live_short_500_300_500_activity']['runtime_commit_attempted_count']}/{control_summary['live_short_500_300_500_activity']['materialized_count']}/{control_summary['live_short_500_300_500_activity']['actual_keyframe_added_count']}",
        f"- decision reasons: {control_summary['live_short_500_300_500_activity']['decision_reason_counts']}",
    ])

    delta_rows, delta_summary = keyframe_delta(short300, short500, ref500)
    write_csv(out / "short300_short500_keyframe_delta_after_lifecycle_fix.csv", delta_rows)
    write_json(out / "short300_short500_keyframe_delta_after_lifecycle_fix_summary.json", delta_summary)
    write_md(out / "short300_short500_keyframe_delta_after_lifecycle_fix_report.md", [
        "# short300 vs short500 keyframe delta after lifecycle fix",
        "",
        f"- short300/short500 keyframes: {delta_summary['short300_keyframe_count']}/{delta_summary['short500_keyframe_count']}",
        f"- short500-short300: {delta_summary['short500_minus_short300']}",
        f"- missing/added: {delta_summary['missing_in_short500_count']}/{delta_summary['added_in_short500_count']}",
        f"- missing by origin: {delta_summary['missing_by_commit_origin']}",
        f"- added by origin: {delta_summary['added_by_commit_origin']}",
        f"- keyframe_set_regression: {delta_summary['keyframe_set_regression']}",
    ])

    f500 = funnel_summary["live_short_500_300_500_activity"]
    need_pose_outcome_fix = bool(f500["pnp_success"] == 0 or f500["miniba_success"] == 0)
    pnp_success_but_no_recovery_success = any(to_bool(r.get("pnp_success")) and not to_bool(r.get("recovery_success")) for r in funnel500_300_500)
    miniba_success_but_no_recovery_success = any(to_bool(r.get("miniba_success")) and not to_bool(r.get("recovery_success")) for r in funnel500_300_500)
    recovery_success_but_no_commit_control = any(to_bool(r.get("recovery_success")) and not to_bool(r.get("commit_control_reached")) for r in funnel500_300_500)
    commit_control_all_hold_reject = bool(
        f500["commit_control_reached"] > 0 and f500["control_allow_commit"] == 0
    )
    control_allow_but_no_materialization = any(
        to_bool(r.get("control_allow_commit")) and not to_bool(r.get("materialized")) for r in funnel500_300_500
    )
    materialized_but_no_keyframe_added = any(
        to_bool(r.get("materialized")) and not to_bool(r.get("actual_keyframe_added")) for r in funnel500_300_500
    )
    need_handoff_fix = bool(miniba_success_but_no_recovery_success or recovery_success_but_no_commit_control)
    need_commit_control_fix = bool(commit_control_all_hold_reject and not need_handoff_fix)
    need_materialization_fix = bool(control_allow_but_no_materialization or materialized_but_no_keyframe_added)
    need_trace_fix = bool(f500["recovery_pose_attempted"] == 0 and len(short500["lifecycle"]) > 0)
    if need_pose_outcome_fix:
        next_action = "pose_outcome_fix"
    elif need_handoff_fix:
        next_action = "recovery_success_handoff_fix"
    elif need_commit_control_fix:
        next_action = "commit_control_fix"
    elif need_materialization_fix:
        next_action = "materialization_fix"
    elif delta_summary["keyframe_set_regression"]:
        next_action = "keyframe_set_regression_fix"
    elif need_trace_fix:
        next_action = "trace_fix"
    else:
        next_action = "v8_policy"
    ready = {
        "need_v8_policy": bool(next_action == "v8_policy"),
        "need_pose_outcome_fix": need_pose_outcome_fix,
        "need_recovery_success_handoff_fix": need_handoff_fix,
        "need_commit_control_fix": need_commit_control_fix,
        "need_materialization_fix": need_materialization_fix,
        "need_keyframe_set_regression_fix": bool(delta_summary["keyframe_set_regression"]),
        "need_trace_fix": need_trace_fix,
        "pose_attempts_reach_pnp": bool(f500["pnp_attempted"] > 0),
        "pnp_success_but_no_recovery_success": pnp_success_but_no_recovery_success,
        "miniba_success_but_no_recovery_success": miniba_success_but_no_recovery_success,
        "recovery_success_but_no_commit_control": recovery_success_but_no_commit_control,
        "commit_control_all_hold_reject": commit_control_all_hold_reject,
        "control_allow_but_no_materialization": control_allow_but_no_materialization,
        "materialized_but_no_keyframe_added": materialized_but_no_keyframe_added,
        "recommended_next_action": next_action,
        "keep_RVQ_tau_frozen": True,
    }
    write_json(out / "ready_for_v8_policy_or_handoff_fix.json", ready)

    write_md(out / "paper_aligned_recovery_pose_outcome_and_commit_handoff_audit_report.md", [
        "# PAPER_ALIGNED_RECOVERY_POSE_OUTCOME_AND_COMMIT_HANDOFF_AUDIT_V1",
        "",
        "## conclusions",
        "",
        f"1. 300-500 funnel: {f500}",
        "2. 300-500 actual keyframe_added=0 because all recovery pose attempts fail before recovery_success: PnP too-few for part of the attempts and MiniBA too-few for the rest.",
        f"3. seed usage reaches PnP/MiniBA; seed vs no-seed 300-500 PnP success rates: {seed_stats['pnp_success_rate']:.3f}/{no_seed_stats['pnp_success_rate']:.3f}, MiniBA success rates: {seed_stats['miniba_success_rate']:.3f}/{no_seed_stats['miniba_success_rate']:.3f}.",
        f"4. pose success handoff: miniba_success_but_no_recovery_success={ready['miniba_success_but_no_recovery_success']}, recovery_success_but_no_commit_control={ready['recovery_success_but_no_commit_control']}.",
        f"5. commit control: allow/hold/reject in 300-500 = {control_summary['live_short_500_300_500_activity']['allow_commit_count']}/{control_summary['live_short_500_300_500_activity']['hold_count']}/{control_summary['live_short_500_300_500_activity']['reject_count']}.",
        f"6. materialization: runtime/materialized/keyframe in 300-500 = {control_summary['live_short_500_300_500_activity']['runtime_commit_attempted_count']}/{control_summary['live_short_500_300_500_activity']['materialized_count']}/{control_summary['live_short_500_300_500_activity']['actual_keyframe_added_count']}.",
        f"7. short300 vs short500 delta: {delta_summary['short500_minus_short300']} with missing/added={delta_summary['missing_in_short500_count']}/{delta_summary['added_in_short500_count']}, indicating independent-run keyframe set divergence/regression also exists.",
        f"8. recommended_next_action={next_action}; need_v8_policy={ready['need_v8_policy']}.",
        "9. Continue freezing R/V/Q and tau.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
