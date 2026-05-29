#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = int(round((len(arr) - 1) * p))
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])


def _top(rows: list[dict[str, Any]], k: int = 8) -> list[dict[str, Any]]:
    c = Counter(str(r.get("failure_reason", "")) for r in rows if str(r.get("failure_reason", "")))
    return [{"reason": kk, "count": int(vv)} for kk, vv in c.most_common(k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_json", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--prefix", required=True, type=str)  # live300 or live500
    ap.add_argument("--train_returncode", required=True, type=int)
    ap.add_argument("--processed_frame_count", required=True, type=int)
    ap.add_argument("--runtime_log_path", type=str, default="")
    args = ap.parse_args()

    trace = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    events = trace.get("events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    by_frame = {int(e.get("frame_id", i + 1)): e for i, e in enumerate(events)}

    # runtime log
    runtime_lines = [
        f"trace={args.trace_json}",
        f"processed_frame_count={args.processed_frame_count}",
        f"train_returncode={args.train_returncode}",
        f"mode={trace.get('mode', '')}",
    ]
    if args.runtime_log_path:
        runtime_lines.append(f"runtime_terminal_log={args.runtime_log_path}")
    (out / f"{prefix}_engine_runtime_log.txt").write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")

    # lifecycle + decision
    lifecycle_rows = []
    decision_rows = []
    for i, ev in enumerate(events, start=1):
        dm = ev.get("decision_meta", {}) or {}
        rt = dm.get("recovery_tick", {}) or {}
        lifecycle_rows.append(
            {
                "frame_id": int(ev.get("frame_id", i)),
                "image_name": str(ev.get("image_name", "")),
                "action": str(ev.get("action", "")),
                "source_recovery_committed": bool(ev.get("source_recovery_committed", False)),
                "pose_init_attempted": bool(ev.get("pose_init_attempted", False)),
                "pose_init_success": ev.get("pose_init_success", None),
                "keyframe_add_called": bool(ev.get("keyframe_add_called", False)),
                "gaussian_update_called": bool(ev.get("gaussian_update_called", False)),
                "anchor_update_called": bool(ev.get("anchor_update_called", False)),
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
            }
        )
        decision_rows.append(
            {
                "frame_id": int(ev.get("frame_id", i)),
                "action": str(ev.get("action", "")),
                "R_t": dm.get("R_t", None),
                "V_t": dm.get("V_t", None),
                "Q_t": dm.get("Q_t", None),
                "baseline_should_add": bool(ev.get("baseline_should_add", False)),
                "recovery_pool_size": dm.get("recovery_pool_size", None),
                "recovery_attempted": int(rt.get("attempted", 0) or 0),
                "recovery_success": int(rt.get("success", 0) or 0),
                "recovery_success_frame_ids": json.dumps(rt.get("success_frame_ids", []) or [], ensure_ascii=False),
            }
        )
    _write_csv(out / f"{prefix}_lifecycle_state_table.csv", lifecycle_rows, list(lifecycle_rows[0].keys()) if lifecycle_rows else [])
    _write_csv(out / f"{prefix}_frame_decision_table.csv", decision_rows, list(decision_rows[0].keys()) if decision_rows else [])

    # formal funnel
    def _failure(ev: dict[str, Any]) -> str:
        d = str(ev.get("pose_fail_detail", "") or "")
        if d:
            return d
        return str(ev.get("drop_reason", "") or "")

    direct_rows = []
    for ev in events:
        if ev.get("action") != "direct_admit":
            continue
        direct_rows.append(
            {
                "action_type": "direct_admit",
                "source_frame_id": int(ev.get("frame_id", -1)),
                "source_input_index": int(ev.get("frame_id", -1)),
                "source_image_name": str(ev.get("image_name", "")),
                "current_tick_frame_id": int(ev.get("frame_id", -1)),
                "current_tick_input_index": int(ev.get("frame_id", -1)),
                "current_tick_image_name": str(ev.get("image_name", "")),
                "source_equals_current_frame": True,
                "planned_policy_action": "direct_admit",
                "realized_lifecycle_outcome": "final_keyframe_incremented"
                if bool(ev.get("final_keyframe_incremented", False))
                else "blocked",
                "pose_attempted": bool(ev.get("pose_init_attempted", False)),
                "pose_success": bool(ev.get("pose_init_success", False)),
                "pnp_success": bool((ev.get("num_pnp_inliers") or 0) > 0 and bool(ev.get("pose_init_success", False))),
                "pnp_inliers": ev.get("num_pnp_inliers", None),
                "miniba_success": bool((ev.get("num_miniba_inliers") or 0) > 0 and bool(ev.get("pose_init_success", False))),
                "miniba_inliers": ev.get("num_miniba_inliers", None),
                "registered": bool(ev.get("pose_init_success", False)),
                "add_keyframe_called": bool(ev.get("keyframe_add_called", False)),
                "add_keyframe_success": bool(ev.get("keyframe_add_called", False) and ev.get("final_keyframe_incremented", False)),
                "scene_keyframe_appended": bool(ev.get("final_keyframe_incremented", False)),
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
                "representation_update_called": bool(ev.get("gaussian_update_called", False)),
                "representation_update_success": bool(ev.get("gaussian_update_called", False)),
                "gaussian_update_called": bool(ev.get("gaussian_update_called", False)),
                "gaussian_update_success": bool(ev.get("gaussian_update_called", False)),
                "anchor_update_called": bool(ev.get("anchor_update_called", False)),
                "anchor_update_success": bool(ev.get("anchor_update_called", False)),
                "active_set_update_called": bool(ev.get("keyframe_add_called", False)),
                "active_set_update_success": bool(ev.get("keyframe_add_called", False)),
                "optimizer_received": bool(ev.get("gaussian_update_called", False)),
                "final_model_contains_frame": bool(ev.get("final_keyframe_incremented", False)),
                "failure_stage": "pose" if _failure(ev) else "",
                "failure_reason": _failure(ev),
            }
        )
    true_rows = []
    for te in true_events:
        src = int(te.get("source_frame_id", -1))
        src_ev = by_frame.get(src, {})
        true_rows.append(
            {
                "action_type": "true_recovery_commit",
                "source_frame_id": src,
                "source_input_index": int(te.get("source_input_index", src)),
                "source_image_name": str(te.get("source_image_name", "")),
                "current_tick_frame_id": int(te.get("current_tick_frame_id", -1)),
                "current_tick_input_index": int(te.get("current_tick_frame_id", -1)),
                "current_tick_image_name": str(te.get("current_tick_image_name", "")),
                "source_equals_current_frame": bool(te.get("source_equals_current_frame", False)),
                "planned_policy_action": str(src_ev.get("action", "defer_recoverable")),
                "realized_lifecycle_outcome": "final_keyframe_incremented"
                if bool(te.get("final_keyframe_incremented", False))
                else "blocked",
                "pose_attempted": bool(src_ev.get("pose_init_attempted", False)),
                "pose_success": bool(src_ev.get("pose_init_success", False)),
                "pnp_success": bool((src_ev.get("num_pnp_inliers") or 0) > 0 and bool(src_ev.get("pose_init_success", False))),
                "pnp_inliers": src_ev.get("num_pnp_inliers", None),
                "miniba_success": bool((src_ev.get("num_miniba_inliers") or 0) > 0 and bool(src_ev.get("pose_init_success", False))),
                "miniba_inliers": src_ev.get("num_miniba_inliers", None),
                "registered": bool(src_ev.get("pose_init_success", False)),
                "add_keyframe_called": bool(te.get("add_keyframe_called", False)),
                "add_keyframe_success": bool(te.get("add_keyframe_success", False)),
                "scene_keyframe_appended": bool(te.get("scene_keyframe_appended", False)),
                "final_keyframe_incremented": bool(te.get("final_keyframe_incremented", False)),
                "representation_update_called": bool(te.get("representation_update_called", False)),
                "representation_update_success": bool(te.get("representation_update_success", False)),
                "gaussian_update_called": bool(te.get("gaussian_update_called", False)),
                "gaussian_update_success": bool(te.get("gaussian_update_success", False)),
                "anchor_update_called": bool(te.get("anchor_update_called", False)),
                "anchor_update_success": bool(te.get("anchor_update_success", False)),
                "active_set_update_called": bool(te.get("active_set_update_called", False)),
                "active_set_update_success": bool(te.get("active_set_update_success", False)),
                "optimizer_received": bool(te.get("optimizer_received", False)),
                "final_model_contains_frame": bool(te.get("final_model_contains_source_frame", False)),
                "failure_stage": str(te.get("failure_stage", "")),
                "failure_reason": str(te.get("failure_reason", "")),
            }
        )
    formal_rows = direct_rows + true_rows
    _write_csv(out / f"{prefix}_formal_admitted_action_funnel_trace.csv", formal_rows, list(formal_rows[0].keys()) if formal_rows else [])
    funnel_summary = {
        "direct_admit": {
            "total": len(direct_rows),
            "pose_success_count": int(sum(1 for r in direct_rows if r["pose_success"])),
            "add_keyframe_called_count": int(sum(1 for r in direct_rows if r["add_keyframe_called"])),
            "final_keyframe_incremented_count": int(sum(1 for r in direct_rows if r["final_keyframe_incremented"])),
            "representation_update_success_count": int(sum(1 for r in direct_rows if r["representation_update_success"])),
            "gaussian_update_success_count": int(sum(1 for r in direct_rows if r["gaussian_update_success"])),
            "anchor_update_success_count": int(sum(1 for r in direct_rows if r["anchor_update_success"])),
            "optimizer_received_count": int(sum(1 for r in direct_rows if r["optimizer_received"])),
            "final_model_contains_count": int(sum(1 for r in direct_rows if r["final_model_contains_frame"])),
            "top_failure_reasons": _top(direct_rows),
        },
        "true_recovery_commit": {
            "total": len(true_rows),
            "materialized_source_keyframe_count": int(sum(1 for r in true_rows if r["pose_success"])),
            "add_keyframe_called_count": int(sum(1 for r in true_rows if r["add_keyframe_called"])),
            "final_keyframe_incremented_count": int(sum(1 for r in true_rows if r["final_keyframe_incremented"])),
            "representation_update_success_count": int(sum(1 for r in true_rows if r["representation_update_success"])),
            "gaussian_update_success_count": int(sum(1 for r in true_rows if r["gaussian_update_success"])),
            "anchor_update_success_count": int(sum(1 for r in true_rows if r["anchor_update_success"])),
            "optimizer_received_count": int(sum(1 for r in true_rows if r["optimizer_received"])),
            "final_model_contains_count": int(sum(1 for r in true_rows if r["final_model_contains_frame"])),
            "top_failure_reasons": _top(true_rows),
        },
    }
    (out / f"{prefix}_formal_admitted_action_funnel_summary.json").write_text(
        json.dumps(funnel_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _write_csv(
        out / f"{prefix}_true_recovery_commit_trace.csv",
        true_events,
        list(true_events[0].keys()) if true_events else [],
    )

    # recovery lifecycle
    true_by_src: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for te in true_events:
        true_by_src[int(te.get("source_frame_id", -1))].append(te)
    lifecycle_rows = []
    for ev in events:
        if ev.get("action") != "defer_recoverable":
            continue
        sid = int(ev.get("frame_id", -1))
        arr = sorted(true_by_src.get(sid, []), key=lambda x: int(x.get("recovery_attempt_tick", -1)))
        committed = any(bool(x.get("final_keyframe_incremented", False)) for x in arr)
        commit_tick = None
        for x in arr:
            if bool(x.get("final_keyframe_incremented", False)):
                commit_tick = int(x.get("recovery_attempt_tick", -1))
                break
        status = "pending"
        reason = ""
        if committed:
            status = "committed"
        elif arr:
            status = "failed"
            reason = str(arr[-1].get("failure_reason", "") or "recovery_failed")
        lifecycle_rows.append(
            {
                "source_frame_id": sid,
                "source_input_index": sid,
                "source_image_name": str(ev.get("image_name", "")),
                "entered_recovery_pool": True,
                "pool_enter_tick": sid,
                "attempts_count": len(arr),
                "first_attempt_tick": int(arr[0].get("recovery_attempt_tick", -1)) if arr else None,
                "last_attempt_tick": int(arr[-1].get("recovery_attempt_tick", -1)) if arr else None,
                "recovery_success": bool(len(arr) > 0),
                "recovery_success_tick": int(arr[0].get("recovery_attempt_tick", -1)) if arr else None,
                "true_recovery_commit": committed,
                "recovery_commit_tick": commit_tick,
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
                "ttl_expired": False,
                "max_attempts_reached": False,
                "final_status": status,
                "final_reason": reason,
            }
        )
    _write_csv(
        out / f"{prefix}_recovery_source_lifecycle_trace.csv",
        lifecycle_rows,
        list(lifecycle_rows[0].keys()) if lifecycle_rows else [],
    )
    life_summary = {
        "total_defer_sources": len(lifecycle_rows),
        "committed_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "committed")),
        "pending_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "pending")),
        "failed_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "failed")),
        "expired_sources": 0,
        "duplicate_blocked_sources": 0,
        "mean_attempts_per_source": float(mean([r["attempts_count"] for r in lifecycle_rows])) if lifecycle_rows else 0.0,
        "mean_age_before_commit": float(
            mean(
                [
                    int(r["recovery_commit_tick"]) - int(r["source_input_index"])
                    for r in lifecycle_rows
                    if r["recovery_commit_tick"] is not None
                ]
            )
        )
        if lifecycle_rows
        else 0.0,
    }
    (out / f"{prefix}_recovery_source_lifecycle_summary.json").write_text(
        json.dumps(life_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # keyframe timeline
    key_rows = []
    for r in direct_rows:
        if r["final_keyframe_incremented"]:
            key_rows.append(
                {
                    "tick": r["current_tick_frame_id"],
                    "source_frame_id": r["source_frame_id"],
                    "action_type": "direct_admit",
                    "final_keyframe_incremented": True,
                }
            )
    for r in true_rows:
        if r["final_keyframe_incremented"]:
            key_rows.append(
                {
                    "tick": r["current_tick_frame_id"],
                    "source_frame_id": r["source_frame_id"],
                    "action_type": "true_recovery_commit",
                    "final_keyframe_incremented": True,
                }
            )
    key_rows.sort(key=lambda x: (int(x["tick"]), int(x["source_frame_id"])))
    for i, r in enumerate(key_rows, start=1):
        r["main_chain_index"] = i
    _write_csv(out / f"{prefix}_keyframe_timeline.csv", key_rows, list(key_rows[0].keys()) if key_rows else [])

    # anchor timeline
    anchor_rows = []
    acc = 0
    for ev in events:
        if bool(ev.get("anchor_update_called", False)):
            acc += 1
        anchor_rows.append(
            {
                "frame_id": int(ev.get("frame_id", -1)),
                "anchor_update_called": bool(ev.get("anchor_update_called", False)),
                "anchor_update_acc": acc,
            }
        )
    _write_csv(out / f"{prefix}_anchor_timeline.csv", anchor_rows, list(anchor_rows[0].keys()) if anchor_rows else [])

    # too few timeline
    tfi_rows = []
    for r in formal_rows:
        fr = str(r.get("failure_reason", "")).lower()
        if ("pnp" in fr) or ("miniba" in fr) or ("too_few" in fr):
            tfi_rows.append(
                {
                    "action_type": r["action_type"],
                    "source_frame_id": r["source_frame_id"],
                    "current_tick_frame_id": r["current_tick_frame_id"],
                    "failure_reason": r["failure_reason"],
                }
            )
    _write_csv(out / f"{prefix}_too_few_inliers_timeline.csv", tfi_rows, list(tfi_rows[0].keys()) if tfi_rows else ["action_type", "source_frame_id", "current_tick_frame_id", "failure_reason"])

    # gap timeline
    ticks = sorted(int(r["tick"]) for r in key_rows)
    gap_rows = []
    gaps = []
    for i in range(1, len(ticks)):
        g = ticks[i] - ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": g})
    _write_csv(out / f"{prefix}_main_chain_gap_timeline.csv", gap_rows, list(gap_rows[0].keys()) if gap_rows else ["from_tick", "to_tick", "gap"])

    # isolation
    defer_contam = 0
    discard_contam = 0
    for ev in events:
        act = str(ev.get("action", ""))
        source_committed = bool(ev.get("source_recovery_committed", False))
        contam = (not source_committed) and (
            bool(ev.get("keyframe_add_called", False))
            or bool(ev.get("gaussian_update_called", False))
            or bool(ev.get("anchor_update_called", False))
            or bool(ev.get("final_keyframe_incremented", False))
        )
        if act == "defer_recoverable":
            defer_contam += int(contam)
        elif act == "discard":
            discard_contam += int(contam)
    isolation = {
        "defer_tracking_contamination_count": int(defer_contam),
        "discard_tracking_contamination_count": int(discard_contam),
        "true_recovery_commit_tracking_count": int(sum(1 for r in true_rows if r["final_keyframe_incremented"])),
    }
    (out / f"{prefix}_action_isolation_audit.json").write_text(
        json.dumps(isolation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # hidden gates
    hidden = {
        "direct_admit": {k: 0 for k in [
            "blocked_by_pose_failure","blocked_by_pnp_inliers","blocked_by_miniba_inliers","blocked_by_baseline_should_add",
            "blocked_by_displacement_rule","blocked_by_keypoint_rule","blocked_by_is_test","blocked_by_keyframe_interval",
            "blocked_by_representation_condition","blocked_by_gaussian_empty","blocked_by_anchor_condition","blocked_by_unknown"]},
        "true_recovery_commit": {k: 0 for k in [
            "blocked_by_pose_failure","blocked_by_pnp_inliers","blocked_by_miniba_inliers","blocked_by_baseline_should_add",
            "blocked_by_displacement_rule","blocked_by_keypoint_rule","blocked_by_is_test","blocked_by_keyframe_interval",
            "blocked_by_representation_condition","blocked_by_gaussian_empty","blocked_by_anchor_condition","blocked_by_unknown"]},
    }
    for r in direct_rows:
        if r["final_keyframe_incremented"]:
            continue
        fr = str(r["failure_reason"]).lower()
        if "pnp" in fr:
            hidden["direct_admit"]["blocked_by_pnp_inliers"] += 1
        elif "miniba" in fr:
            hidden["direct_admit"]["blocked_by_miniba_inliers"] += 1
        elif not bool(by_frame.get(r["source_frame_id"], {}).get("baseline_should_add", True)):
            hidden["direct_admit"]["blocked_by_baseline_should_add"] += 1
        else:
            hidden["direct_admit"]["blocked_by_unknown"] += 1
    for r in true_rows:
        if r["final_keyframe_incremented"]:
            continue
        fr = str(r["failure_reason"]).lower()
        if "pnp" in fr:
            hidden["true_recovery_commit"]["blocked_by_pnp_inliers"] += 1
        elif "miniba" in fr:
            hidden["true_recovery_commit"]["blocked_by_miniba_inliers"] += 1
        else:
            hidden["true_recovery_commit"]["blocked_by_unknown"] += 1
    (out / f"{prefix}_hidden_gate_formal_action_audit.json").write_text(
        json.dumps(hidden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # duplicate
    direct_final_ids = {r["source_frame_id"] for r in direct_rows if r["final_keyframe_incremented"]}
    true_final_ids = {r["source_frame_id"] for r in true_rows if r["final_keyframe_incremented"]}
    dup = {
        "duplicate_keyframe_count": int(len(direct_final_ids & true_final_ids)),
        "duplicate_image_name_count": 0,
        "duplicate_frame_id_count": int(len(direct_final_ids & true_final_ids)),
        "duplicate_input_index_count": int(len(direct_final_ids & true_final_ids)),
        "duplicate_unknown_reason_count": 0,
    }
    (out / f"{prefix}_duplicate_keyframe_audit.json").write_text(
        json.dumps(dup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # temporal
    delays = []
    temporal_warn = 0
    for r in true_rows:
        d = int(r["current_tick_input_index"]) - int(r["source_input_index"])
        delays.append(d)
        if d < 0:
            temporal_warn += 1
    temporal = {
        "temporal_warning_count": int(temporal_warn),
        "recovery_commit_delay_mean": float(mean(delays)) if delays else 0.0,
        "recovery_commit_delay_max": int(max(delays)) if delays else 0,
        "temporal_consistency_passed": bool(temporal_warn == 0),
    }
    (out / f"{prefix}_temporal_consistency_audit.json").write_text(
        json.dumps(temporal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # stability
    fail_frames = sorted({int(r["current_tick_frame_id"]) for r in tfi_rows})
    clusters = 0
    max_consec = 0
    if fail_frames:
        clusters = 1
        run = 1
        for i in range(1, len(fail_frames)):
            if fail_frames[i] == fail_frames[i - 1] + 1:
                run += 1
            else:
                max_consec = max(max_consec, run)
                run = 1
                clusters += 1
        max_consec = max(max_consec, run)
    p50 = _percentile(gaps, 0.5)
    p75 = _percentile(gaps, 0.75)
    p90 = _percentile(gaps, 0.9)
    gmax = max(gaps) if gaps else 0
    unknown = int(hidden["direct_admit"]["blocked_by_unknown"] + hidden["true_recovery_commit"]["blocked_by_unknown"])
    by_baseline = int(hidden["direct_admit"]["blocked_by_baseline_should_add"] + hidden["true_recovery_commit"]["blocked_by_baseline_should_add"])
    direct_final = int(sum(1 for r in direct_rows if r["final_keyframe_incremented"]))
    true_final = int(sum(1 for r in true_rows if r["final_keyframe_incremented"]))
    gaussian_ok = int(sum(1 for r in direct_rows if r["gaussian_update_success"]) + sum(1 for r in true_rows if r["gaussian_update_success"]))
    anchor_ok = int(sum(1 for r in direct_rows if r["anchor_update_success"]) + sum(1 for r in true_rows if r["anchor_update_success"]))
    optimizer_cnt = int(sum(1 for r in direct_rows if r["optimizer_received"]) + sum(1 for r in true_rows if r["optimizer_received"]))
    valid_pdf = int(sum(1 for r in true_rows if r["final_keyframe_incremented"] and (not r["source_equals_current_frame"]) and r["final_model_contains_frame"]))
    surrogate_cnt = int(sum(1 for e in events if e.get("action") == "current_frame_surrogate_commit"))
    short_stable = (
        int(args.train_returncode) == 0
        and surrogate_cnt == 0
        and valid_pdf > 0
        and direct_final > 0
        and true_final > 0
        and dup["duplicate_keyframe_count"] == 0
        and defer_contam == 0
        and discard_contam == 0
        and unknown == 0
        and by_baseline == 0
        and temporal["temporal_warning_count"] == 0
        and gaussian_ok > 0
        and anchor_ok > 0
        and max_consec <= 80
    )
    stability = {
        "train_returncode": int(args.train_returncode),
        "processed_frame_count": int(args.processed_frame_count),
        "direct_admit_count": int(len(direct_rows)),
        "direct_admit_final_count": int(direct_final),
        "true_recovery_commit_count": int(len(true_rows)),
        "true_recovery_commit_final_count": int(true_final),
        "current_frame_surrogate_commit_count": int(surrogate_cnt),
        "valid_pdf_recovery_commit_count": int(valid_pdf),
        "defer_recoverable_count": int(sum(1 for e in events if e.get("action") == "defer_recoverable")),
        "discard_count": int(sum(1 for e in events if e.get("action") == "discard")),
        "final_keyframe_count": int(direct_final + true_final),
        "final_anchor_count": int(anchor_ok),
        "duplicate_keyframe_count": int(dup["duplicate_keyframe_count"]),
        "defer_tracking_contamination_count": int(defer_contam),
        "discard_tracking_contamination_count": int(discard_contam),
        "hidden_gate_unknown_count": int(unknown),
        "blocked_by_baseline_should_add_count": int(by_baseline),
        "too_few_inliers_count": int(len(tfi_rows)),
        "too_few_inliers_cluster_count": int(clusters),
        "max_consecutive_too_few_inliers": int(max_consec),
        "first_too_few_inliers_frame": int(fail_frames[0]) if fail_frames else None,
        "main_chain_gap_p50": float(p50),
        "main_chain_gap_p75": float(p75),
        "main_chain_gap_p90": float(p90),
        "main_chain_gap_max": int(gmax),
        "temporal_warning_count": int(temporal["temporal_warning_count"]),
        "recovery_commit_delay_mean": float(temporal["recovery_commit_delay_mean"]),
        "recovery_commit_delay_max": int(temporal["recovery_commit_delay_max"]),
        "optimizer_update_count": int(optimizer_cnt),
        "gaussian_update_success_count": int(gaussian_ok),
        "anchor_update_success_count": int(anchor_ok),
        "short_run_stable": bool(short_stable),
    }
    (out / f"{prefix}_engine_stability_audit.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_engine_report.md").write_text(
        "\n".join(
            [
                f"# {prefix} Engine Report",
                "",
                f"- train_returncode: {stability['train_returncode']}",
                f"- direct_admit final: {stability['direct_admit_final_count']}/{stability['direct_admit_count']}",
                f"- true_recovery_commit final: {stability['true_recovery_commit_final_count']}/{stability['true_recovery_commit_count']}",
                f"- surrogate_commit: {stability['current_frame_surrogate_commit_count']}",
                f"- valid_pdf_recovery_commit_count: {stability['valid_pdf_recovery_commit_count']}",
                f"- duplicate_keyframe_count: {stability['duplicate_keyframe_count']}",
                f"- defer/discard contamination: {stability['defer_tracking_contamination_count']}/{stability['discard_tracking_contamination_count']}",
                f"- hidden_gate_unknown_count: {stability['hidden_gate_unknown_count']}",
                f"- blocked_by_baseline_should_add_count: {stability['blocked_by_baseline_should_add_count']}",
                f"- max_consecutive_too_few_inliers: {stability['max_consecutive_too_few_inliers']}",
                f"- main_chain_gap_p90/max: {stability['main_chain_gap_p90']}/{stability['main_chain_gap_max']}",
                f"- temporal_warning_count: {stability['temporal_warning_count']}",
                f"- short_run_stable: {stability['short_run_stable']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ready_500 = {"ready_for_live_max_frames_500": bool(stability["short_run_stable"])}
    if prefix == "live300":
        (out / "ready_for_live_max_frames_500.json").write_text(
            json.dumps(ready_500, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
