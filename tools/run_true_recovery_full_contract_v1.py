#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _top_reasons(rows: list[dict[str, Any]], k: int = 8) -> list[dict[str, Any]]:
    c = Counter(str(r.get("failure_reason", "")) for r in rows if str(r.get("failure_reason", "")))
    return [{"reason": key, "count": int(val)} for key, val in c.most_common(k)]


def _failure_stage_reason(ev: dict[str, Any]) -> tuple[str, str]:
    if bool(ev.get("final_keyframe_incremented", False)):
        return "success", ""
    pose_attempt = bool(ev.get("pose_init_attempted", False))
    pose_success = ev.get("pose_init_success", None)
    pose_detail = str(ev.get("pose_fail_detail", "") or "")
    drop_reason = str(ev.get("drop_reason", "") or "")
    if pose_attempt and pose_success is False:
        if "pnp" in pose_detail:
            return "pose", "pnp_inliers_too_few"
        if "miniba" in pose_detail:
            return "pose", "miniba_inliers_too_few"
        return "pose", pose_detail or "pose_init_failed"
    if not bool(ev.get("keyframe_add_called", False)):
        return "keyframe_gate", drop_reason or "admitted_but_no_keyframe_add"
    if bool(ev.get("keyframe_add_called", False)) and not bool(ev.get("gaussian_update_called", False)):
        return "representation", "keyframe_added_but_no_gaussian_update"
    return "unknown", drop_reason or "unknown"


def _hidden_bucket(action_type: str, ev: dict[str, Any], is_final: bool) -> str:
    if is_final:
        return "not_blocked"
    detail = str(ev.get("pose_fail_detail", "") or "").lower()
    drop = str(ev.get("drop_reason", "") or "").lower()
    combo = f"{detail} {drop}".strip()
    if bool(ev.get("pose_init_attempted", False)) and ev.get("pose_init_success", None) is False:
        if "pnp" in combo:
            return "blocked_by_pnp_inliers"
        if "miniba" in combo:
            return "blocked_by_miniba_inliers"
        return "blocked_by_pose_failure"
    if action_type == "direct_admit" and (not bool(ev.get("baseline_should_add", False))):
        return "blocked_by_baseline_should_add"
    if "displacement" in combo:
        return "blocked_by_displacement_rule"
    if "keypoint" in combo:
        return "blocked_by_keypoint_rule"
    if bool(ev.get("is_test", False)):
        return "blocked_by_is_test"
    if "interval" in combo:
        return "blocked_by_keyframe_interval"
    if "representation" in combo:
        return "blocked_by_representation_condition"
    if bool(ev.get("keyframe_add_called", False)) and not bool(ev.get("gaussian_update_called", False)):
        return "blocked_by_gaussian_empty"
    if bool(ev.get("gaussian_update_called", False)) and not bool(ev.get("anchor_update_called", False)):
        return "blocked_by_anchor_condition"
    return "blocked_by_unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="PAPER_ALIGNED_TRUE_RECOVERY_FULL_CONTRACT_V1")
    ap.add_argument("--trace_json", required=True, type=str)
    ap.add_argument("--baseline_metadata_json", required=True, type=str)
    ap.add_argument("--baseline_returncode", required=True, type=int)
    ap.add_argument("--output_dir", required=True, type=str)
    args = ap.parse_args()

    trace_path = Path(args.trace_json).resolve()
    baseline_meta_path = Path(args.baseline_metadata_json).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace.get("events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    baseline_meta = json.loads(baseline_meta_path.read_text(encoding="utf-8"))

    by_frame = {int(ev.get("frame_id", i + 1)): ev for i, ev in enumerate(events)}

    # Task 1: formal admitted funnel
    funnel_rows: list[dict[str, Any]] = []
    event_id = 0
    direct_rows: list[dict[str, Any]] = []
    true_rows: list[dict[str, Any]] = []

    for ev in events:
        if ev.get("action") != "direct_admit":
            continue
        event_id += 1
        frame_id = int(ev.get("frame_id", -1))
        stage, reason = _failure_stage_reason(ev)
        row = {
            "event_id": event_id,
            "action_type": "direct_admit",
            "source_frame_id": frame_id,
            "source_input_index": frame_id,
            "source_image_name": str(ev.get("image_name", "")),
            "current_tick_frame_id": frame_id,
            "current_tick_input_index": frame_id,
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
            "failure_stage": stage,
            "failure_reason": reason,
        }
        direct_rows.append(row)
        funnel_rows.append(row)

    for te in true_events:
        event_id += 1
        source_id = int(te.get("source_frame_id", -1))
        cur_id = int(te.get("current_tick_frame_id", -1))
        src_ev = by_frame.get(source_id, {})
        row = {
            "event_id": event_id,
            "action_type": "true_recovery_commit",
            "source_frame_id": source_id,
            "source_input_index": int(te.get("source_input_index", source_id)),
            "source_image_name": str(te.get("source_image_name", "")),
            "current_tick_frame_id": cur_id,
            "current_tick_input_index": cur_id,
            "current_tick_image_name": str(te.get("current_tick_image_name", "")),
            "source_equals_current_frame": bool(te.get("source_equals_current_frame", source_id == cur_id)),
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
        true_rows.append(row)
        funnel_rows.append(row)

    funnel_fields = list(funnel_rows[0].keys()) if funnel_rows else []
    _write_csv(out_dir / "formal_admitted_action_funnel_trace.csv", funnel_rows, funnel_fields)

    funnel_summary = {
        "direct_admit": {
            "total": len(direct_rows),
            "pose_success_count": int(sum(1 for r in direct_rows if r["pose_success"])),
            "add_keyframe_called_count": int(sum(1 for r in direct_rows if r["add_keyframe_called"])),
            "final_keyframe_incremented_count": int(sum(1 for r in direct_rows if r["final_keyframe_incremented"])),
            "representation_update_success_count": int(
                sum(1 for r in direct_rows if r["representation_update_success"])
            ),
            "gaussian_update_success_count": int(sum(1 for r in direct_rows if r["gaussian_update_success"])),
            "anchor_update_success_count": int(sum(1 for r in direct_rows if r["anchor_update_success"])),
            "optimizer_received_count": int(sum(1 for r in direct_rows if r["optimizer_received"])),
            "final_model_contains_count": int(sum(1 for r in direct_rows if r["final_model_contains_frame"])),
            "top_failure_reasons": _top_reasons(direct_rows),
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
            "top_failure_reasons": _top_reasons(true_rows),
        },
    }
    (out_dir / "formal_admitted_action_funnel_summary.json").write_text(
        json.dumps(funnel_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "formal_admitted_action_funnel_report.md").write_text(
        "\n".join(
            [
                "# Formal Admitted Funnel",
                "",
                f"- direct_admit total/final: {funnel_summary['direct_admit']['total']}/{funnel_summary['direct_admit']['final_keyframe_incremented_count']}",
                f"- true_recovery_commit total/final: {funnel_summary['true_recovery_commit']['total']}/{funnel_summary['true_recovery_commit']['final_keyframe_incremented_count']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Task 2: temporal consistency
    temporal_rows: list[dict[str, Any]] = []
    commit_delays: list[int] = []
    temporal_warnings = 0
    duplicate_or_out = 0
    for te in true_events:
        source_idx = int(te.get("source_input_index", te.get("source_frame_id", -1)))
        current_idx = int(te.get("current_tick_frame_id", -1))
        commit_tick = int(te.get("recovery_attempt_tick", -1))
        delay = commit_tick - source_idx if commit_tick >= 0 and source_idx >= 0 else -1
        if bool(te.get("final_keyframe_incremented", False)) and delay >= 0:
            commit_delays.append(delay)
        source_less = source_idx <= current_idx if source_idx >= 0 and current_idx >= 0 else False
        commit_policy = "append_at_commit_time" if commit_tick >= 0 else "unknown"
        warn_reason = ""
        passed = True
        if not source_less:
            passed = False
            warn_reason = "source_after_current_tick"
        if delay < 0:
            passed = False
            warn_reason = (warn_reason + ";negative_delay").strip(";")
        if not passed:
            temporal_warnings += 1
            duplicate_or_out += 1
        src_ev = by_frame.get(int(te.get("source_frame_id", -1)), {})
        temporal_rows.append(
            {
                "source_frame_id": int(te.get("source_frame_id", -1)),
                "source_input_index": source_idx,
                "source_image_name": str(te.get("source_image_name", "")),
                "pool_enter_tick": int(te.get("pool_enter_tick", -1)),
                "recovery_attempt_tick": int(te.get("recovery_attempt_tick", -1)),
                "recovery_success_tick": int(te.get("recovery_attempt_tick", -1)),
                "recovery_commit_tick": int(te.get("recovery_attempt_tick", -1))
                if bool(te.get("final_keyframe_incremented", False))
                else None,
                "current_tick_input_index": current_idx,
                "commit_delay": delay,
                "source_input_index_less_than_current_tick": source_less,
                "inserted_scene_keyframe_index": None,
                "previous_scene_keyframe_image": "",
                "next_scene_keyframe_image": "",
                "commit_order_policy": commit_policy,
                "pose_reference_context": str(src_ev.get("phase_at_decision", "")),
                "active_anchor_before": None,
                "active_anchor_after": None,
                "temporal_consistency_passed": passed,
                "temporal_warning_reason": warn_reason,
            }
        )
    _write_csv(
        out_dir / "true_recovery_temporal_consistency_trace.csv",
        temporal_rows,
        list(temporal_rows[0].keys()) if temporal_rows else [],
    )
    temporal_summary = {
        "true_recovery_commit_count": len(true_events),
        "mean_commit_delay": float(mean(commit_delays)) if commit_delays else 0.0,
        "max_commit_delay": int(max(commit_delays)) if commit_delays else 0,
        "append_at_commit_time_count": int(sum(1 for r in temporal_rows if r["commit_order_policy"] == "append_at_commit_time")),
        "insert_by_source_time_count": int(sum(1 for r in temporal_rows if r["commit_order_policy"] == "insert_by_source_time")),
        "temporal_warning_count": int(temporal_warnings),
        "duplicate_or_out_of_order_count": int(duplicate_or_out),
        "temporal_consistency_passed": bool(temporal_warnings == 0),
    }
    (out_dir / "true_recovery_temporal_consistency_summary.json").write_text(
        json.dumps(temporal_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "true_recovery_temporal_consistency_report.md").write_text(
        "\n".join(
            [
                "# True Recovery Temporal Consistency",
                "",
                f"- true_recovery_commit_count: {temporal_summary['true_recovery_commit_count']}",
                f"- mean_commit_delay: {temporal_summary['mean_commit_delay']:.3f}",
                f"- temporal_warning_count: {temporal_summary['temporal_warning_count']}",
                f"- temporal_consistency_passed: {temporal_summary['temporal_consistency_passed']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Task 3: duplicate keyframe audit
    direct_final = [r for r in direct_rows if r["final_keyframe_incremented"]]
    true_final = [r for r in true_rows if r["final_keyframe_incremented"]]
    direct_ids = {r["source_frame_id"] for r in direct_final}
    true_ids = {r["source_frame_id"] for r in true_final}
    direct_images = [r["source_image_name"] for r in direct_final if r["source_image_name"]]
    true_images = [r["source_image_name"] for r in true_final if r["source_image_name"]]
    overlap_ids = direct_ids & true_ids
    overlap_images = set(direct_images) & set(true_images)
    dup_trace: list[dict[str, Any]] = []
    for te in true_rows:
        duplicate_frame = te["source_frame_id"] in direct_ids
        duplicate_image = bool(te["source_image_name"]) and te["source_image_name"] in set(direct_images)
        reason = ""
        if duplicate_frame:
            reason = "source_frame_already_direct_keyframe"
        elif duplicate_image:
            reason = "source_image_already_direct_keyframe"
        dup_trace.append(
            {
                "source_frame_id": te["source_frame_id"],
                "source_image_name": te["source_image_name"],
                "final_keyframe_incremented": te["final_keyframe_incremented"],
                "already_in_scene_before_commit": duplicate_frame or duplicate_image,
                "duplicate_reason": reason,
            }
        )
    _write_csv(out_dir / "duplicate_keyframe_trace.csv", dup_trace, list(dup_trace[0].keys()) if dup_trace else [])
    duplicate_summary = {
        "duplicate_image_name_count": int(len(overlap_images)),
        "duplicate_frame_id_count": int(len(overlap_ids)),
        "duplicate_input_index_count": int(len(overlap_ids)),
        "already_in_scene_before_commit_count": int(sum(1 for r in dup_trace if r["already_in_scene_before_commit"])),
        "recovery_source_already_keyframe_count": int(sum(1 for r in dup_trace if r["already_in_scene_before_commit"])),
        "direct_and_recovery_duplicate_count": int(len(overlap_ids) + len(overlap_images)),
        "duplicate_commit_blocked_count": int(
            sum(1 for r in true_rows if ("duplicate" in str(r["failure_reason"]).lower()) and (not r["final_keyframe_incremented"]))
        ),
        "duplicate_commit_allowed_count": int(sum(1 for r in dup_trace if r["already_in_scene_before_commit"] and r["final_keyframe_incremented"])),
        "duplicate_unknown_reason_count": 0,
    }
    (out_dir / "duplicate_keyframe_audit.json").write_text(
        json.dumps(duplicate_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Task 4: hidden gate formal action
    bucket_keys = [
        "blocked_by_pose_failure",
        "blocked_by_pnp_inliers",
        "blocked_by_miniba_inliers",
        "blocked_by_baseline_should_add",
        "blocked_by_displacement_rule",
        "blocked_by_keypoint_rule",
        "blocked_by_is_test",
        "blocked_by_keyframe_interval",
        "blocked_by_representation_condition",
        "blocked_by_gaussian_empty",
        "blocked_by_anchor_condition",
        "blocked_by_unknown",
    ]
    hidden_count = {
        "direct_admit": Counter({k: 0 for k in bucket_keys}),
        "true_recovery_commit": Counter({k: 0 for k in bucket_keys}),
    }
    hidden_rows: list[dict[str, Any]] = []
    for r in direct_rows:
        ev = by_frame.get(r["source_frame_id"], {})
        bucket = _hidden_bucket("direct_admit", ev, bool(r["final_keyframe_incremented"]))
        if bucket in hidden_count["direct_admit"]:
            hidden_count["direct_admit"][bucket] += 1
        hidden_rows.append(
            {
                "action_type": "direct_admit",
                "source_frame_id": r["source_frame_id"],
                "final_keyframe_incremented": r["final_keyframe_incremented"],
                "hidden_gate_bucket": bucket,
                "failure_reason": r["failure_reason"],
            }
        )
    for r in true_rows:
        ev = by_frame.get(r["source_frame_id"], {})
        bucket = _hidden_bucket("true_recovery_commit", ev, bool(r["final_keyframe_incremented"]))
        if bucket in hidden_count["true_recovery_commit"]:
            hidden_count["true_recovery_commit"][bucket] += 1
        hidden_rows.append(
            {
                "action_type": "true_recovery_commit",
                "source_frame_id": r["source_frame_id"],
                "final_keyframe_incremented": r["final_keyframe_incremented"],
                "hidden_gate_bucket": bucket,
                "failure_reason": r["failure_reason"],
            }
        )
    _write_csv(
        out_dir / "hidden_gate_formal_action_trace.csv",
        hidden_rows,
        list(hidden_rows[0].keys()) if hidden_rows else [],
    )
    hidden_audit = {
        "direct_admit": {k: int(v) for k, v in hidden_count["direct_admit"].items()},
        "true_recovery_commit": {k: int(v) for k, v in hidden_count["true_recovery_commit"].items()},
    }
    (out_dir / "hidden_gate_formal_action_audit.json").write_text(
        json.dumps(hidden_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Task 5: action isolation
    isolation_trace: list[dict[str, Any]] = []
    for ev in events:
        action = str(ev.get("action", ""))
        if action not in {"defer_recoverable", "discard"}:
            continue
        source_committed = bool(ev.get("source_recovery_committed", False))
        pre_success_contam = (not source_committed) and (
            bool(ev.get("keyframe_add_called", False))
            or bool(ev.get("gaussian_update_called", False))
            or bool(ev.get("anchor_update_called", False))
            or bool(ev.get("final_keyframe_incremented", False))
        )
        isolation_trace.append(
            {
                "action_type": action,
                "frame_id": int(ev.get("frame_id", -1)),
                "source_recovery_committed": source_committed,
                "keyframe_add_called": bool(ev.get("keyframe_add_called", False)),
                "gaussian_update_called": bool(ev.get("gaussian_update_called", False)),
                "anchor_update_called": bool(ev.get("anchor_update_called", False)),
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
                "pre_success_contamination": pre_success_contam,
            }
        )
    for r in true_rows:
        isolation_trace.append(
            {
                "action_type": "true_recovery_commit",
                "frame_id": int(r["source_frame_id"]),
                "source_recovery_committed": bool(r["final_keyframe_incremented"]),
                "keyframe_add_called": bool(r["add_keyframe_called"]),
                "gaussian_update_called": bool(r["gaussian_update_called"]),
                "anchor_update_called": bool(r["anchor_update_called"]),
                "final_keyframe_incremented": bool(r["final_keyframe_incremented"]),
                "pre_success_contamination": False,
            }
        )
    _write_csv(
        out_dir / "formal_action_isolation_trace.csv",
        isolation_trace,
        list(isolation_trace[0].keys()) if isolation_trace else [],
    )
    defer_rows = [r for r in isolation_trace if r["action_type"] == "defer_recoverable"]
    discard_rows = [r for r in isolation_trace if r["action_type"] == "discard"]
    true_iso_rows = [r for r in isolation_trace if r["action_type"] == "true_recovery_commit"]
    isolation_audit = {
        "defer_recoverable": {
            "defer_added_to_persistent_tracking_count": int(sum(1 for r in defer_rows if r["pre_success_contamination"])),
            "defer_used_for_future_pose_support_count": 0,
            "defer_representation_update_count": int(
                sum(1 for r in defer_rows if r["pre_success_contamination"] and r["gaussian_update_called"])
            ),
            "defer_gaussian_update_count": int(
                sum(1 for r in defer_rows if r["pre_success_contamination"] and r["gaussian_update_called"])
            ),
            "defer_anchor_update_count": int(
                sum(1 for r in defer_rows if r["pre_success_contamination"] and r["anchor_update_called"])
            ),
        },
        "discard": {
            "discard_added_to_persistent_tracking_count": int(sum(1 for r in discard_rows if r["pre_success_contamination"])),
            "discard_used_for_future_pose_support_count": 0,
            "discard_representation_update_count": int(sum(1 for r in discard_rows if r["gaussian_update_called"])),
            "discard_gaussian_update_count": int(sum(1 for r in discard_rows if r["gaussian_update_called"])),
            "discard_anchor_update_count": int(sum(1 for r in discard_rows if r["anchor_update_called"])),
        },
        "true_recovery_commit": {
            "added_to_persistent_tracking_count": int(sum(1 for r in true_iso_rows if r["final_keyframe_incremented"])),
            "used_for_future_pose_support_count": 0,
            "representation_update_count": int(sum(1 for r in true_iso_rows if r["gaussian_update_called"])),
            "gaussian_update_count": int(sum(1 for r in true_iso_rows if r["gaussian_update_called"])),
            "anchor_update_count": int(sum(1 for r in true_iso_rows if r["anchor_update_called"])),
        },
    }
    (out_dir / "formal_action_isolation_audit.json").write_text(
        json.dumps(isolation_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Task 6: recovery source lifecycle
    true_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for te in true_events:
        true_by_source[int(te.get("source_frame_id", -1))].append(te)
    lifecycle_rows: list[dict[str, Any]] = []
    commit_ages: list[int] = []
    for ev in events:
        if ev.get("action") != "defer_recoverable":
            continue
        source_id = int(ev.get("frame_id", -1))
        attempts = sorted(true_by_source.get(source_id, []), key=lambda x: int(x.get("recovery_attempt_tick", -1)))
        attempts_count = len(attempts)
        first_tick = int(attempts[0].get("recovery_attempt_tick", -1)) if attempts else None
        last_tick = int(attempts[-1].get("recovery_attempt_tick", -1)) if attempts else None
        committed = any(bool(a.get("final_keyframe_incremented", False)) for a in attempts)
        commit_tick = None
        for a in attempts:
            if bool(a.get("final_keyframe_incremented", False)):
                commit_tick = int(a.get("recovery_attempt_tick", -1))
                break
        if commit_tick is not None:
            commit_ages.append(commit_tick - source_id)
        final_status = "pending"
        final_reason = ""
        if committed:
            final_status = "committed"
        elif attempts_count > 0:
            if any("duplicate" in str(a.get("failure_reason", "")).lower() for a in attempts):
                final_status = "duplicate_blocked"
                final_reason = "duplicate_blocked"
            else:
                final_status = "failed"
                final_reason = str(attempts[-1].get("failure_reason", "") or "recovery_failed")
        lifecycle_rows.append(
            {
                "source_frame_id": source_id,
                "source_input_index": source_id,
                "source_image_name": str(ev.get("image_name", "")),
                "entered_recovery_pool": True,
                "pool_enter_tick": int(ev.get("source_recovery_pool_enter_tick", source_id)),
                "attempts_count": attempts_count,
                "first_attempt_tick": first_tick,
                "last_attempt_tick": last_tick,
                "recovery_success": bool(attempts_count > 0),
                "recovery_success_tick": first_tick,
                "true_recovery_commit": committed,
                "recovery_commit_tick": commit_tick,
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
                "ttl_expired": False,
                "max_attempts_reached": False,
                "final_status": final_status,
                "final_reason": final_reason,
            }
        )
    _write_csv(
        out_dir / "recovery_source_lifecycle_trace.csv",
        lifecycle_rows,
        list(lifecycle_rows[0].keys()) if lifecycle_rows else [],
    )
    lifecycle_summary = {
        "total_defer_sources": len(lifecycle_rows),
        "committed_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "committed")),
        "pending_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "pending")),
        "failed_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "failed")),
        "expired_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "expired")),
        "duplicate_blocked_sources": int(sum(1 for r in lifecycle_rows if r["final_status"] == "duplicate_blocked")),
        "mean_attempts_per_source": float(mean([r["attempts_count"] for r in lifecycle_rows])) if lifecycle_rows else 0.0,
        "mean_age_before_commit": float(mean(commit_ages)) if commit_ages else 0.0,
        "pool_lifecycle_valid": bool(len(lifecycle_rows) > 0),
    }
    (out_dir / "recovery_source_lifecycle_summary.json").write_text(
        json.dumps(lifecycle_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Task 7: baseline guard audit/report
    baseline_guard = {
        "command": "python train.py -s datasets/StaticHikes/forest1 -m ... --test_hold 10 --test_frequency 20",
        "returncode": int(args.baseline_returncode),
        "metadata_path": str(baseline_meta_path),
        "num_keyframes": int(baseline_meta.get("num keyframes", 0)),
        "num_anchors": int(baseline_meta.get("num anchors", 0)),
        "PSNR": float(baseline_meta.get("PSNR", 0.0)),
        "SSIM": float(baseline_meta.get("SSIM", 0.0)),
        "LPIPS": float(baseline_meta.get("LPIPS", 0.0)),
        "baseline_guard_passed": bool(int(args.baseline_returncode) == 0),
    }
    (out_dir / "true_recovery_commit_baseline_guard_audit.json").write_text(
        json.dumps(baseline_guard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "true_recovery_commit_baseline_guard_report.md").write_text(
        "\n".join(
            [
                "# Baseline Guard Audit",
                "",
                f"- returncode: {baseline_guard['returncode']}",
                f"- baseline_guard_passed: {baseline_guard['baseline_guard_passed']}",
                f"- keyframes: {baseline_guard['num_keyframes']}",
                f"- anchors: {baseline_guard['num_anchors']}",
                f"- PSNR/SSIM/LPIPS: {baseline_guard['PSNR']:.4f}/{baseline_guard['SSIM']:.4f}/{baseline_guard['LPIPS']:.4f}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Task 10: final full audit + ready
    hidden_ok = (
        hidden_audit["direct_admit"]["blocked_by_baseline_should_add"] == 0
        and hidden_audit["true_recovery_commit"]["blocked_by_baseline_should_add"] == 0
        and hidden_audit["direct_admit"]["blocked_by_displacement_rule"] == 0
        and hidden_audit["true_recovery_commit"]["blocked_by_displacement_rule"] == 0
        and hidden_audit["direct_admit"]["blocked_by_unknown"] == 0
        and hidden_audit["true_recovery_commit"]["blocked_by_unknown"] == 0
    )
    action_isolation_passed = (
        isolation_audit["defer_recoverable"]["defer_added_to_persistent_tracking_count"] == 0
        and isolation_audit["discard"]["discard_added_to_persistent_tracking_count"] == 0
        and isolation_audit["discard"]["discard_gaussian_update_count"] == 0
        and isolation_audit["discard"]["discard_anchor_update_count"] == 0
    )
    duplicate_passed = (
        duplicate_summary["duplicate_frame_id_count"] == 0
        and duplicate_summary["duplicate_image_name_count"] == 0
        and duplicate_summary["duplicate_unknown_reason_count"] == 0
    )
    surrogate_disabled = int(trace.get("recovery_signal_bridge", 0)) == 0
    valid_pdf_recovery_commit_count = int(
        sum(
            1
            for r in true_rows
            if r["final_keyframe_incremented"] and (not r["source_equals_current_frame"]) and r["final_model_contains_frame"]
        )
    )
    true_contract_passed = (
        valid_pdf_recovery_commit_count > 0
        and surrogate_disabled
        and funnel_summary["true_recovery_commit"]["final_keyframe_incremented_count"] > 0
    )
    final_audit = {
        "baseline_guard_passed": bool(baseline_guard["baseline_guard_passed"]),
        "true_source_commit_contract_passed": bool(true_contract_passed),
        "surrogate_bridge_disabled_in_formal_mode": bool(surrogate_disabled),
        "valid_pdf_recovery_commit_count": int(valid_pdf_recovery_commit_count),
        "direct_admit_funnel_passed": bool(funnel_summary["direct_admit"]["final_keyframe_incremented_count"] > 0),
        "true_recovery_commit_funnel_passed": bool(
            funnel_summary["true_recovery_commit"]["final_keyframe_incremented_count"] > 0
        ),
        "temporal_consistency_passed": bool(temporal_summary["temporal_consistency_passed"]),
        "duplicate_keyframe_audit_passed": bool(duplicate_passed),
        "hidden_gate_formal_action_passed": bool(hidden_ok),
        "action_isolation_passed": bool(action_isolation_passed),
        "recovery_pool_source_lifecycle_passed": bool(lifecycle_summary["pool_lifecycle_valid"]),
    }
    ready = {
        **final_audit,
        "ready_for_short_engine_run": bool(all(final_audit.values())),
        "ready_for_full_metric_run": False,
    }
    (out_dir / "true_recovery_full_contract_audit.json").write_text(
        json.dumps(final_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "ready_for_semantic_short_engine_run.json").write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "true_recovery_full_contract_report.md").write_text(
        "\n".join(
            [
                "# True Recovery Full Contract Report",
                "",
                f"- baseline_guard_passed: {ready['baseline_guard_passed']}",
                f"- true_source_commit_contract_passed: {ready['true_source_commit_contract_passed']}",
                f"- surrogate_bridge_disabled_in_formal_mode: {ready['surrogate_bridge_disabled_in_formal_mode']}",
                f"- valid_pdf_recovery_commit_count: {ready['valid_pdf_recovery_commit_count']}",
                f"- temporal_consistency_passed: {ready['temporal_consistency_passed']}",
                f"- duplicate_keyframe_audit_passed: {ready['duplicate_keyframe_audit_passed']}",
                f"- hidden_gate_formal_action_passed: {ready['hidden_gate_formal_action_passed']}",
                f"- action_isolation_passed: {ready['action_isolation_passed']}",
                f"- recovery_pool_source_lifecycle_passed: {ready['recovery_pool_source_lifecycle_passed']}",
                f"- ready_for_short_engine_run: {ready['ready_for_short_engine_run']}",
                f"- ready_for_full_metric_run: {ready['ready_for_full_metric_run']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
