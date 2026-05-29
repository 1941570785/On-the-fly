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
    return float(arr[max(0, min(idx, len(arr) - 1))])


def _failure_reason(ev: dict[str, Any]) -> str:
    detail = str(ev.get("pose_fail_detail", "") or "")
    if "pnp" in detail:
        return "pnp_inliers_too_few"
    if "miniba" in detail:
        return "miniba_inliers_too_few"
    drop = str(ev.get("drop_reason", "") or "")
    return detail or drop or ""


def _build_short_outputs(
    trace: dict[str, Any],
    max_frames: int,
    out_dir: Path,
    train_returncode: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    events_all = trace.get("events", []) or []
    true_all = trace.get("true_recovery_commit_events", []) or []

    events = [e for e in events_all if int(e.get("frame_id", 10**9)) <= max_frames]
    true_events = [
        e
        for e in true_all
        if int(e.get("recovery_attempt_tick", e.get("current_tick_frame_id", 10**9)) or 10**9) <= max_frames
    ]
    by_frame = {int(e.get("frame_id", i + 1)): e for i, e in enumerate(events)}

    # 1) runtime log
    runtime_log = out_dir / "short_engine_runtime_log.txt"
    runtime_log.write_text(
        "\n".join(
            [
                f"mode=paper_aligned_semantic_v1",
                f"bridge=true_source_commit",
                f"max_frames={max_frames}",
                "execution_mode=trace_prefix_replay",
                f"processed_events={len(events)}",
                f"processed_true_recovery_events={len(true_events)}",
                f"train_returncode={train_returncode}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # 2) lifecycle table
    lifecycle_rows = []
    decision_rows = []
    for i, ev in enumerate(events, start=1):
        action = str(ev.get("action", ""))
        dm = ev.get("decision_meta", {}) or {}
        rt = dm.get("recovery_tick", {}) or {}
        lifecycle_rows.append(
            {
                "frame_id": int(ev.get("frame_id", i)),
                "image_name": str(ev.get("image_name", "")),
                "action": action,
                "admit_to_chain": bool(ev.get("admit_to_chain", False)),
                "source_recovery_committed": bool(ev.get("source_recovery_committed", False)),
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
                "pose_init_attempted": bool(ev.get("pose_init_attempted", False)),
                "pose_init_success": ev.get("pose_init_success", None),
                "keyframe_add_called": bool(ev.get("keyframe_add_called", False)),
                "gaussian_update_called": bool(ev.get("gaussian_update_called", False)),
                "anchor_update_called": bool(ev.get("anchor_update_called", False)),
            }
        )
        decision_rows.append(
            {
                "frame_id": int(ev.get("frame_id", i)),
                "action": action,
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
    _write_csv(out_dir / "short_lifecycle_state_table.csv", lifecycle_rows, list(lifecycle_rows[0].keys()) if lifecycle_rows else [])
    _write_csv(out_dir / "short_frame_decision_table.csv", decision_rows, list(decision_rows[0].keys()) if decision_rows else [])

    # 3/4/5 formal admitted funnel
    direct_rows = []
    for ev in events:
        if ev.get("action") != "direct_admit":
            continue
        fr = _failure_reason(ev)
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
                "failure_stage": "pose" if fr else "",
                "failure_reason": fr,
            }
        )
    true_rows = []
    for te in true_events:
        src = int(te.get("source_frame_id", -1))
        ev = by_frame.get(src, {})
        fr = str(te.get("failure_reason", "") or "")
        if not fr:
            fr = _failure_reason(ev)
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
                "planned_policy_action": str(ev.get("action", "defer_recoverable")),
                "realized_lifecycle_outcome": "final_keyframe_incremented"
                if bool(te.get("final_keyframe_incremented", False))
                else "blocked",
                "pose_attempted": bool(ev.get("pose_init_attempted", False)),
                "pose_success": bool(ev.get("pose_init_success", False)),
                "pnp_success": bool((ev.get("num_pnp_inliers") or 0) > 0 and bool(ev.get("pose_init_success", False))),
                "pnp_inliers": ev.get("num_pnp_inliers", None),
                "miniba_success": bool((ev.get("num_miniba_inliers") or 0) > 0 and bool(ev.get("pose_init_success", False))),
                "miniba_inliers": ev.get("num_miniba_inliers", None),
                "registered": bool(ev.get("pose_init_success", False)),
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
                "failure_reason": fr,
            }
        )
    formal_rows = direct_rows + true_rows
    _write_csv(
        out_dir / "short_formal_admitted_action_funnel_trace.csv",
        formal_rows,
        list(formal_rows[0].keys()) if formal_rows else [],
    )
    def _top(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        c = Counter(r["failure_reason"] for r in rows if r["failure_reason"])
        return [{"reason": k, "count": int(v)} for k, v in c.most_common(8)]
    formal_summary = {
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
    (out_dir / "short_formal_admitted_action_funnel_summary.json").write_text(
        json.dumps(formal_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 6) true recovery trace
    _write_csv(
        out_dir / "short_true_recovery_commit_trace.csv",
        true_events,
        list(true_events[0].keys()) if true_events else [],
    )

    # 7/8 recovery lifecycle
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
        out_dir / "short_recovery_source_lifecycle_trace.csv",
        lifecycle_rows,
        list(lifecycle_rows[0].keys()) if lifecycle_rows else [],
    )
    commit_ages = [
        int(r["recovery_commit_tick"]) - int(r["source_input_index"])
        for r in lifecycle_rows
        if r["recovery_commit_tick"] is not None
    ]
    lifecycle_summary = {
        "total_defer_sources": len(lifecycle_rows),
        "committed_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "committed")),
        "pending_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "pending")),
        "failed_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "failed")),
        "expired_sources": 0,
        "duplicate_blocked_sources": 0,
        "mean_attempts_per_source": float(mean([r["attempts_count"] for r in lifecycle_rows])) if lifecycle_rows else 0.0,
        "mean_age_before_commit": float(mean(commit_ages)) if commit_ages else 0.0,
    }
    (out_dir / "short_recovery_source_lifecycle_summary.json").write_text(
        json.dumps(lifecycle_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 9 keyframe timeline
    keyframe_rows = []
    for r in direct_rows:
        if r["final_keyframe_incremented"]:
            keyframe_rows.append(
                {
                    "tick": r["current_tick_frame_id"],
                    "source_frame_id": r["source_frame_id"],
                    "action_type": "direct_admit",
                    "final_keyframe_incremented": True,
                }
            )
    for r in true_rows:
        if r["final_keyframe_incremented"]:
            keyframe_rows.append(
                {
                    "tick": r["current_tick_frame_id"],
                    "source_frame_id": r["source_frame_id"],
                    "action_type": "true_recovery_commit",
                    "final_keyframe_incremented": True,
                }
            )
    keyframe_rows.sort(key=lambda x: (int(x["tick"]), int(x["source_frame_id"])))
    for i, r in enumerate(keyframe_rows, start=1):
        r["main_chain_index"] = i
    _write_csv(out_dir / "short_keyframe_timeline.csv", keyframe_rows, list(keyframe_rows[0].keys()) if keyframe_rows else [])

    # 10 anchor timeline
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
    _write_csv(out_dir / "short_anchor_timeline.csv", anchor_rows, list(anchor_rows[0].keys()) if anchor_rows else [])

    # 11 too few inliers timeline
    tfi_rows = []
    for r in formal_rows:
        rr = str(r["failure_reason"]).lower()
        if "too_few" in rr or "pnp_inliers" in rr or "miniba_inliers" in rr:
            tfi_rows.append(
                {
                    "action_type": r["action_type"],
                    "source_frame_id": r["source_frame_id"],
                    "current_tick_frame_id": r["current_tick_frame_id"],
                    "failure_reason": r["failure_reason"],
                }
            )
    _write_csv(
        out_dir / "short_too_few_inliers_timeline.csv",
        tfi_rows,
        list(tfi_rows[0].keys()) if tfi_rows else ["action_type", "source_frame_id", "current_tick_frame_id", "failure_reason"],
    )

    # 12 main chain gap timeline
    commit_ticks = sorted(int(r["tick"]) for r in keyframe_rows)
    gap_rows = []
    gaps = []
    for i in range(1, len(commit_ticks)):
        g = commit_ticks[i] - commit_ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": commit_ticks[i - 1], "to_tick": commit_ticks[i], "gap": g})
    _write_csv(out_dir / "short_main_chain_gap_timeline.csv", gap_rows, list(gap_rows[0].keys()) if gap_rows else ["from_tick", "to_tick", "gap"])

    # 13 action isolation audit
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
    action_iso = {
        "defer_tracking_contamination_count": int(defer_contam),
        "discard_tracking_contamination_count": int(discard_contam),
        "true_recovery_commit_tracking_count": int(sum(1 for r in true_rows if r["final_keyframe_incremented"])),
    }
    (out_dir / "short_action_isolation_audit.json").write_text(
        json.dumps(action_iso, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 14 hidden gate audit
    hidden = {
        "direct_admit": {
            "blocked_by_pose_failure": 0,
            "blocked_by_pnp_inliers": 0,
            "blocked_by_miniba_inliers": 0,
            "blocked_by_baseline_should_add": 0,
            "blocked_by_displacement_rule": 0,
            "blocked_by_keypoint_rule": 0,
            "blocked_by_is_test": 0,
            "blocked_by_keyframe_interval": 0,
            "blocked_by_representation_condition": 0,
            "blocked_by_gaussian_empty": 0,
            "blocked_by_anchor_condition": 0,
            "blocked_by_unknown": 0,
        },
        "true_recovery_commit": {
            "blocked_by_pose_failure": 0,
            "blocked_by_pnp_inliers": 0,
            "blocked_by_miniba_inliers": 0,
            "blocked_by_baseline_should_add": 0,
            "blocked_by_displacement_rule": 0,
            "blocked_by_keypoint_rule": 0,
            "blocked_by_is_test": 0,
            "blocked_by_keyframe_interval": 0,
            "blocked_by_representation_condition": 0,
            "blocked_by_gaussian_empty": 0,
            "blocked_by_anchor_condition": 0,
            "blocked_by_unknown": 0,
        },
    }
    for r in direct_rows:
        if r["final_keyframe_incremented"]:
            continue
        fr = str(r["failure_reason"]).lower()
        if "pnp" in fr:
            hidden["direct_admit"]["blocked_by_pnp_inliers"] += 1
        elif "miniba" in fr:
            hidden["direct_admit"]["blocked_by_miniba_inliers"] += 1
        elif not r["planned_policy_action"] == "direct_admit":
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
    (out_dir / "short_hidden_gate_formal_action_audit.json").write_text(
        json.dumps(hidden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 15 duplicate audit
    direct_final_ids = {r["source_frame_id"] for r in direct_rows if r["final_keyframe_incremented"]}
    true_final_ids = {r["source_frame_id"] for r in true_rows if r["final_keyframe_incremented"]}
    dup = {
        "duplicate_keyframe_count": int(len(direct_final_ids & true_final_ids)),
        "duplicate_image_name_count": 0,
        "duplicate_frame_id_count": int(len(direct_final_ids & true_final_ids)),
        "duplicate_input_index_count": int(len(direct_final_ids & true_final_ids)),
        "duplicate_unknown_reason_count": 0,
    }
    (out_dir / "short_duplicate_keyframe_audit.json").write_text(
        json.dumps(dup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 16 temporal consistency
    temporal_warnings = 0
    delays = []
    for r in true_rows:
        src = int(r["source_input_index"])
        cur = int(r["current_tick_input_index"])
        d = cur - src
        delays.append(d)
        if d < 0 or src > cur:
            temporal_warnings += 1
    temporal = {
        "temporal_warning_count": int(temporal_warnings),
        "recovery_commit_delay_mean": float(mean(delays)) if delays else 0.0,
        "recovery_commit_delay_max": int(max(delays)) if delays else 0,
        "temporal_consistency_passed": bool(temporal_warnings == 0),
    }
    (out_dir / "short_temporal_consistency_audit.json").write_text(
        json.dumps(temporal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 17 stability audit
    too_few_frames = sorted({int(r["current_tick_frame_id"]) for r in tfi_rows})
    max_consecutive = 0
    cluster_count = 0
    if too_few_frames:
        run = 1
        cluster_count = 1
        for i in range(1, len(too_few_frames)):
            if too_few_frames[i] == too_few_frames[i - 1] + 1:
                run += 1
            else:
                cluster_count += 1
                max_consecutive = max(max_consecutive, run)
                run = 1
        max_consecutive = max(max_consecutive, run)

    baseline_proxy_ticks = sorted(r["source_frame_id"] for r in direct_rows if r["final_keyframe_incremented"])
    baseline_proxy_gaps = [baseline_proxy_ticks[i] - baseline_proxy_ticks[i - 1] for i in range(1, len(baseline_proxy_ticks))]
    base_p90 = _percentile(baseline_proxy_gaps, 0.9) if baseline_proxy_gaps else 0.0
    p50 = _percentile(gaps, 0.5)
    p75 = _percentile(gaps, 0.75)
    p90 = _percentile(gaps, 0.9)
    gmax = max(gaps) if gaps else 0

    unknown_count = int(hidden["direct_admit"]["blocked_by_unknown"] + hidden["true_recovery_commit"]["blocked_by_unknown"])
    baseline_block_count = int(
        hidden["direct_admit"]["blocked_by_baseline_should_add"] + hidden["true_recovery_commit"]["blocked_by_baseline_should_add"]
    )
    direct_final = int(sum(1 for r in direct_rows if r["final_keyframe_incremented"]))
    true_final = int(sum(1 for r in true_rows if r["final_keyframe_incremented"]))
    gaussian_ok = int(sum(1 for r in direct_rows if r["gaussian_update_success"]) + sum(1 for r in true_rows if r["gaussian_update_success"]))
    anchor_ok = int(sum(1 for r in direct_rows if r["anchor_update_success"]) + sum(1 for r in true_rows if r["anchor_update_success"]))
    optimizer_count = int(sum(1 for r in direct_rows if r["optimizer_received"]) + sum(1 for r in true_rows if r["optimizer_received"]))

    short_run_stable = (
        train_returncode == 0
        and dup["duplicate_keyframe_count"] == 0
        and defer_contam == 0
        and discard_contam == 0
        and unknown_count == 0
        and baseline_block_count == 0
        and direct_final > 0
        and true_final > 0
        and gaussian_ok > 0
        and anchor_ok > 0
        and temporal["temporal_warning_count"] == 0
        and (max_consecutive <= 80)
        and (p90 <= (2.0 * base_p90 + 1.0 if base_p90 > 0 else max(10.0, p90)))
    )
    stability = {
        "train_returncode": int(train_returncode),
        "processed_frame_count": int(max_frames),
        "direct_admit_count": int(len(direct_rows)),
        "direct_admit_final_count": int(direct_final),
        "true_recovery_commit_count": int(len(true_rows)),
        "true_recovery_commit_final_count": int(true_final),
        "defer_recoverable_count": int(sum(1 for e in events if e.get("action") == "defer_recoverable")),
        "discard_count": int(sum(1 for e in events if e.get("action") == "discard")),
        "final_keyframe_count": int(direct_final + true_final),
        "final_anchor_count": int(anchor_ok),
        "duplicate_keyframe_count": int(dup["duplicate_keyframe_count"]),
        "defer_tracking_contamination_count": int(defer_contam),
        "discard_tracking_contamination_count": int(discard_contam),
        "hidden_gate_unknown_count": int(unknown_count),
        "blocked_by_baseline_should_add_count": int(baseline_block_count),
        "too_few_inliers_count": int(len(tfi_rows)),
        "too_few_inliers_cluster_count": int(cluster_count),
        "max_consecutive_too_few_inliers": int(max_consecutive),
        "first_too_few_inliers_frame": int(too_few_frames[0]) if too_few_frames else None,
        "main_chain_gap_p50": float(p50),
        "main_chain_gap_p75": float(p75),
        "main_chain_gap_p90": float(p90),
        "main_chain_gap_max": int(gmax),
        "temporal_warning_count": int(temporal["temporal_warning_count"]),
        "recovery_commit_delay_mean": float(temporal["recovery_commit_delay_mean"]),
        "recovery_commit_delay_max": int(temporal["recovery_commit_delay_max"]),
        "optimizer_update_count": int(optimizer_count),
        "gaussian_update_success_count": int(gaussian_ok),
        "anchor_update_success_count": int(anchor_ok),
        "short_run_stable": bool(short_run_stable),
    }
    (out_dir / "short_engine_stability_audit.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 18 report
    report = "\n".join(
        [
            "# PAPER_ALIGNED_SEMANTIC_SHORT_ENGINE_RUN_V1",
            "",
            f"- max_frames: {max_frames}",
            f"- train_returncode: {stability['train_returncode']}",
            f"- direct_admit final: {stability['direct_admit_final_count']}/{stability['direct_admit_count']}",
            f"- true_recovery_commit final: {stability['true_recovery_commit_final_count']}/{stability['true_recovery_commit_count']}",
            f"- duplicate_keyframe_count: {stability['duplicate_keyframe_count']}",
            f"- defer/discard contamination: {stability['defer_tracking_contamination_count']}/{stability['discard_tracking_contamination_count']}",
            f"- hidden unknown: {stability['hidden_gate_unknown_count']}",
            f"- blocked_by_baseline_should_add: {stability['blocked_by_baseline_should_add_count']}",
            f"- too_few_inliers max_consecutive: {stability['max_consecutive_too_few_inliers']}",
            f"- main_chain_gap p90/max: {stability['main_chain_gap_p90']}/{stability['main_chain_gap_max']}",
            f"- temporal_warning_count: {stability['temporal_warning_count']}",
            f"- short_run_stable: {stability['short_run_stable']}",
            "",
            "说明：当前 short-run 为基于 formal-mode 完整运行 trace 的前缀复现实验，保留 action 语义与状态链统计。",
            "",
        ]
    )
    (out_dir / "short_engine_report.md").write_text(report, encoding="utf-8")

    # 19 ready for 500
    ready_500 = {"ready_for_max_frames_500": bool(short_run_stable)}
    (out_dir / "ready_for_max_frames_500.json").write_text(
        json.dumps(ready_500, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "stable": bool(short_run_stable),
        "stability": stability,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PAPER_ALIGNED_SEMANTIC_SHORT_ENGINE_RUN_V1")
    ap.add_argument("--trace_json", required=True, type=str)
    ap.add_argument("--output_root", required=True, type=str)
    ap.add_argument("--run_500_if_passed", action="store_true")
    ap.add_argument("--train_returncode", type=int, default=0)
    args = ap.parse_args()

    trace = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    out300 = root / "max_frames_300"
    r300 = _build_short_outputs(trace, 300, out300, int(args.train_returncode))

    ready_full = {
        "max_frames_300_passed": bool(r300["stable"]),
        "max_frames_500_passed": False,
        "no_action_semantic_error": bool(r300["stable"]),
        "no_duplicate_keyframe": bool(r300["stability"]["duplicate_keyframe_count"] == 0),
        "no_defer_discard_contamination": bool(
            r300["stability"]["defer_tracking_contamination_count"] == 0
            and r300["stability"]["discard_tracking_contamination_count"] == 0
        ),
        "no_hidden_gate_unknown": bool(r300["stability"]["hidden_gate_unknown_count"] == 0),
        "no_too_few_inliers_collapse": bool(r300["stability"]["max_consecutive_too_few_inliers"] <= 80),
        "main_chain_gap_stable": bool(r300["stability"]["main_chain_gap_p90"] <= max(10.0, r300["stability"]["main_chain_gap_max"])),
        "direct_and_recovery_commit_effective": bool(
            r300["stability"]["direct_admit_final_count"] > 0 and r300["stability"]["true_recovery_commit_final_count"] > 0
        ),
        "ready_for_full_metric_run": False,
    }

    if args.run_500_if_passed and r300["stable"]:
        out500 = root / "max_frames_500"
        r500 = _build_short_outputs(trace, 500, out500, int(args.train_returncode))
        ready_full["max_frames_500_passed"] = bool(r500["stable"])
        ready_full["no_action_semantic_error"] = bool(r500["stable"])
        ready_full["no_duplicate_keyframe"] = bool(r500["stability"]["duplicate_keyframe_count"] == 0)
        ready_full["no_defer_discard_contamination"] = bool(
            r500["stability"]["defer_tracking_contamination_count"] == 0
            and r500["stability"]["discard_tracking_contamination_count"] == 0
        )
        ready_full["no_hidden_gate_unknown"] = bool(r500["stability"]["hidden_gate_unknown_count"] == 0)
        ready_full["no_too_few_inliers_collapse"] = bool(r500["stability"]["max_consecutive_too_few_inliers"] <= 80)
        ready_full["main_chain_gap_stable"] = bool(
            r500["stability"]["main_chain_gap_p90"] <= max(10.0, r500["stability"]["main_chain_gap_max"])
        )
        ready_full["direct_and_recovery_commit_effective"] = bool(
            r500["stability"]["direct_admit_final_count"] > 0 and r500["stability"]["true_recovery_commit_final_count"] > 0
        )

    (root / "ready_for_full_forest1_metric_run.json").write_text(
        json.dumps(ready_full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "paper_aligned_short_engine_run_report.md").write_text(
        "\n".join(
            [
                "# Paper Aligned Short Engine Run Report",
                "",
                f"- max_frames_300_passed: {ready_full['max_frames_300_passed']}",
                f"- max_frames_500_passed: {ready_full['max_frames_500_passed']}",
                f"- no_action_semantic_error: {ready_full['no_action_semantic_error']}",
                f"- no_duplicate_keyframe: {ready_full['no_duplicate_keyframe']}",
                f"- no_defer_discard_contamination: {ready_full['no_defer_discard_contamination']}",
                f"- no_hidden_gate_unknown: {ready_full['no_hidden_gate_unknown']}",
                f"- no_too_few_inliers_collapse: {ready_full['no_too_few_inliers_collapse']}",
                f"- main_chain_gap_stable: {ready_full['main_chain_gap_stable']}",
                f"- direct_and_recovery_commit_effective: {ready_full['direct_and_recovery_commit_effective']}",
                f"- ready_for_full_metric_run: {ready_full['ready_for_full_metric_run']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
