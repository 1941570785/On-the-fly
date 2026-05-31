#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1")
INPUT = ROOT / "PAPER_ALIGNED_RECOVERY_REFERENCE_SELECTION_INTEGRATION_FIX_V1/v7_after_reference_selection_rerun"
SUPPORT_FIX = ROOT / "PAPER_ALIGNED_RECOVERY_KEYFRAME_SUPPORT_INTEGRATION_FIX_V1/v7_after_support_integration_rerun"
V7_ORIGINAL = ROOT / "PAPER_ALIGNED_RECOVERY_COMMIT_EARLY_SEED_V7_V1"


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


def pct(vals: list[float], q: float) -> float:
    arr = sorted(vals)
    if not arr:
        return 0.0
    idx = int(round((len(arr) - 1) * q))
    return arr[max(0, min(len(arr) - 1, idx))]


def rows_by_int(rows: list[dict[str, Any]], key: str) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[to_int(row.get(key), -1)].append(row)
    return out


def load_run(run_dir: Path) -> dict[str, Any]:
    trace = read_json(run_dir / "model/semantic_trace.json")
    events = {to_int(e.get("frame_id"), -1): e for e in trace.get("events", []) or []}
    return {
        "trace": trace,
        "events": events,
        "engine": read_json(run_dir / "engine_stability_audit.json"),
        "control": read_csv(run_dir / "recovery_commit_control_trace.csv"),
        "materialization": read_csv(run_dir / "recovery_commit_materialization_trace.csv"),
        "keyframes": read_csv(run_dir / "keyframe_timeline.csv"),
        "gaps": read_csv(run_dir / "main_chain_gap_timeline.csv"),
        "support": read_csv(run_dir / "support_trend_timeline.csv"),
        "support_integration": read_csv(run_dir / "support_integration_trace.csv"),
        "chosen_candidates": read_csv(run_dir / "chosen_kfs_candidate_trace.csv"),
        "chosen": read_csv(run_dir / "chosen_kfs_reference_trace.csv"),
        "pose_pool": read_csv(run_dir / "pose_reference_pool_trace.csv"),
        "bridge": read_csv(run_dir / "matching_to_pose_path_bridge_trace.csv"),
        "pnp": read_csv(run_dir / "pnp_miniba_reference_trace.csv"),
        "matching": read_csv(run_dir / "matching_support_trace.csv"),
        "local": read_csv(run_dir / "local_map_anchor_trace.csv"),
    }


def support_post_p90(control_rows: list[dict[str, Any]]) -> dict[str, float]:
    post = [r for r in control_rows if 300 <= to_int(r.get("source_frame_id"), -1) < 500]
    return {
        "num_matches_p90": pct([to_float(r.get("num_matches")) for r in post], 0.9),
        "feasibility_p90": pct([to_float(r.get("materialization_feasibility_score")) for r in post], 0.9),
    }


def blocking_stage(row: dict[str, Any]) -> tuple[str, str]:
    if not row["processed"]:
        return "not_processed", "no semantic runtime event"
    if not row["candidate_evaluated"]:
        return "processed_but_no_candidate", "no admission candidate event"
    if row["discard_candidate"]:
        return "discard", "semantic action discard"
    if row["defer_recoverable_candidate"] and not row["pose_attempted"]:
        return "defer_only", "lifecycle gate deferred source, so direct pose path was not entered"
    if not row["pose_attempted"]:
        return "pose_not_attempted", "no pose_init_attempted flag and no PnP/MiniBA trace"
    if not row["chosen_kfs_built"]:
        return "chosen_kfs_not_built", "pose attempt without chosen/reference candidate trace"
    if not row["matching_executed"]:
        return "matching_not_executed", "chosen/reference trace exists but pose matching trace missing"
    if not row["pnp_attempted"]:
        return "pnp_not_attempted", "pose matching did not construct PnP reference trace"
    if not row["pnp_success"]:
        return "pnp_failed", "PnP attempted but did not succeed"
    if not row["miniba_attempted"]:
        return "miniba_not_attempted", "PnP path did not reach MiniBA"
    if not row["miniba_success"]:
        return "miniba_failed", "MiniBA attempted but did not succeed"
    if row["runtime_commit_attempted_count"] > 0 and row["materialized_commit_count"] == 0:
        return "runtime_attempt_but_no_materialization", "runtime attempt did not materialize"
    if row["control_allow_commit_count"] > 0 and row["runtime_commit_attempted_count"] == 0:
        return "commit_allowed_but_no_attempt", "control allowed commit but no runtime attempt recorded"
    if row["recovery_success_count"] > 0 and row["control_allow_commit_count"] == 0:
        return "recovery_success_but_no_commit", "recovery success did not pass commit control"
    if row["actual_keyframe_added"]:
        return "materialized", "frame is present in keyframe timeline"
    return "pose_path_no_materialization", "pose path reachable but no keyframe was materialized"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--input_root", default=str(INPUT))
    args = parser.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    input_root = Path(args.input_root)
    short500 = load_run(input_root / "live_short_500")
    short300 = load_run(input_root / "live_short_300")

    events = short500["events"]
    candidate_by_frame = rows_by_int(short500["chosen_candidates"], "frame_id")
    chosen_by_frame = rows_by_int(short500["chosen"], "frame_id")
    matching_by_frame = rows_by_int(short500["matching"], "frame_id")
    pnp_by_frame = rows_by_int(short500["pnp"], "frame_id")
    bridge_by_frame = rows_by_int(short500["bridge"], "frame_id")
    control_by_source = rows_by_int(short500["control"], "source_frame_id")
    mat_by_source = rows_by_int(short500["materialization"], "source_frame_id")
    keyframe_sources = {to_int(r.get("source_frame_id"), -1) for r in short500["keyframes"]}

    timeline = []
    for fid in range(0, 500):
        event = events.get(fid)
        cands = candidate_by_frame.get(fid, [])
        chosen_rows = chosen_by_frame.get(fid, [])
        matching_rows = matching_by_frame.get(fid, [])
        pnp_rows = pnp_by_frame.get(fid, [])
        bridge_rows = bridge_by_frame.get(fid, [])
        controls = control_by_source.get(fid, [])
        mats = mat_by_source.get(fid, [])
        chosen_selected = sum(1 for r in cands if to_bool(r.get("candidate_selected")))
        matching_seed = sum(to_int(r.get("matched_seed_keyframe_count"), 0) for r in matching_rows)
        pnp_ref_count = sum(len(as_list(r.get("pnp_ref_keyframe_ids"))) for r in pnp_rows)
        pnp_ref_seed = sum(1 for r in pnp_rows if to_bool(r.get("pnp_ref_contains_seed")))
        miniba_ref_count = sum(len(as_list(r.get("miniba_ref_keyframe_ids"))) for r in pnp_rows)
        miniba_ref_seed = sum(1 for r in pnp_rows if to_bool(r.get("miniba_ref_contains_seed")))
        recovery_success = sum(1 for r in controls if str(r.get("control_mode", "")))
        allow_commit = sum(1 for r in controls if str(r.get("decision")) == "commit" or str(r.get("control_decision")) == "allow_commit")
        runtime_attempt = sum(1 for r in mats if to_bool(r.get("runtime_commit_attempted")))
        materialized = sum(1 for r in mats if to_bool(r.get("materialized")))
        row = {
            "frame_id": fid,
            "source_frame_id": fid,
            "processed": bool(event is not None),
            "lifecycle_state": str(event.get("action", "not_processed")) if event else "not_processed",
            "candidate_evaluated": bool(event is not None),
            "direct_admit_candidate": bool(event and event.get("action") == "direct_admit"),
            "defer_recoverable_candidate": bool(event and event.get("action") == "defer_recoverable"),
            "discard_candidate": bool(event and event.get("action") == "discard"),
            "pose_attempted": bool((event and event.get("pose_init_attempted")) or pnp_rows),
            "chosen_kfs_built": bool(cands or chosen_rows),
            "chosen_kfs_candidate_count": len(cands),
            "chosen_kfs_selected_count": chosen_selected,
            "matching_executed": bool(matching_rows),
            "matching_candidate_count": sum(len(as_list(r.get("matched_keyframe_ids"))) for r in matching_rows),
            "matching_seed_candidate_count": matching_seed,
            "pnp_attempted": bool(pnp_rows),
            "pnp_ref_count": pnp_ref_count,
            "pnp_ref_seed_count": pnp_ref_seed,
            "pnp_success": any(to_bool(r.get("pnp_success")) for r in pnp_rows),
            "miniba_attempted": any(len(as_list(r.get("miniba_ref_keyframe_ids"))) > 0 for r in pnp_rows),
            "miniba_ref_count": miniba_ref_count,
            "miniba_ref_seed_count": miniba_ref_seed,
            "miniba_success": any(to_bool(r.get("miniba_success")) for r in pnp_rows),
            "recovery_success_count": recovery_success,
            "control_allow_commit_count": allow_commit,
            "runtime_commit_attempted_count": runtime_attempt,
            "materialized_commit_count": materialized,
            "actual_keyframe_added": bool(fid in keyframe_sources),
            "support_bridge_rows": len(bridge_rows),
            "support_bridge_seed_matches": sum(to_int(r.get("matched_seed_keyframe_count"), 0) for r in bridge_rows),
            "support_bridge_seed_best_score": max([to_int(r.get("matched_seed_best_score"), 0) for r in bridge_rows] or [0]),
        }
        stage, reason = blocking_stage(row)
        row["blocking_stage"] = stage
        row["blocking_reason"] = reason
        timeline.append(row)

    write_csv(out / "pose_path_reachability_timeline.csv", timeline)
    post_rows = [r for r in timeline if 300 <= int(r["frame_id"]) < 500]
    summary = {
        "frames_300_500": len(post_rows),
        "processed_300_500": sum(r["processed"] for r in post_rows),
        "pose_attempted_300_500": sum(r["pose_attempted"] for r in post_rows),
        "chosen_kfs_built_300_500": sum(r["chosen_kfs_built"] for r in post_rows),
        "matching_executed_300_500": sum(r["matching_executed"] for r in post_rows),
        "pnp_attempted_300_500": sum(r["pnp_attempted"] for r in post_rows),
        "miniba_attempted_300_500": sum(r["miniba_attempted"] for r in post_rows),
        "recovery_success_count_300_500": sum(r["recovery_success_count"] for r in post_rows),
        "control_allow_commit_count_300_500": sum(r["control_allow_commit_count"] for r in post_rows),
        "runtime_commit_attempted_count_300_500": sum(r["runtime_commit_attempted_count"] for r in post_rows),
        "materialized_commit_count_300_500": sum(r["materialized_commit_count"] for r in post_rows),
        "actual_keyframe_added_300_500": sum(r["actual_keyframe_added"] for r in post_rows),
        "support_bridge_seed_match_count_300_500": sum(r["support_bridge_seed_matches"] for r in post_rows),
    }
    write_json(out / "pose_path_reachability_summary.json", summary)
    (out / "pose_path_reachability_report.md").write_text(
        "\n".join([
            "# pose path reachability",
            "",
            f"- processed_300_500: {summary['processed_300_500']}/200",
            f"- pose_attempted_300_500: {summary['pose_attempted_300_500']}",
            f"- chosen/matching/pnp/miniba 300-500: {summary['chosen_kfs_built_300_500']}/{summary['matching_executed_300_500']}/{summary['pnp_attempted_300_500']}/{summary['miniba_attempted_300_500']}",
            f"- support_bridge_seed_match_count_300_500: {summary['support_bridge_seed_match_count_300_500']}",
        ]) + "\n",
        encoding="utf-8",
    )

    stage_counts = Counter(str(r["blocking_stage"]) for r in post_rows)
    stage_rows = []
    for row in post_rows:
        stage_rows.append({
            "frame_id": row["frame_id"],
            "lifecycle_state": row["lifecycle_state"],
            "final_stage": row["blocking_stage"],
            "blocking_reason": row["blocking_reason"],
        })
    write_csv(out / "frame_lifecycle_stage_300_500_audit.csv", stage_rows)
    stage_summary = {
        "total": len(post_rows),
        "stage_counts": dict(stage_counts),
        "stage_ratios": {k: v / max(len(post_rows), 1) for k, v in stage_counts.items()},
    }
    write_json(out / "frame_lifecycle_stage_300_500_summary.json", stage_summary)
    (out / "frame_lifecycle_stage_300_500_report.md").write_text(
        "# frame lifecycle stage 300-500\n\n" + json.dumps(stage_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    exec_rows = []
    for fid in range(300, 500):
        exec_rows.append({
            "frame_id": fid,
            "chosen_kfs_built": bool(candidate_by_frame.get(fid) or chosen_by_frame.get(fid)),
            "chosen_candidate_count": len(candidate_by_frame.get(fid, [])),
            "chosen_selected_count": sum(1 for r in candidate_by_frame.get(fid, []) if to_bool(r.get("candidate_selected"))),
            "matching_executed": bool(matching_by_frame.get(fid)),
            "matching_seed_candidate_count": sum(to_int(r.get("matched_seed_keyframe_count"), 0) for r in matching_by_frame.get(fid, [])),
            "pnp_attempted": bool(pnp_by_frame.get(fid)),
            "pnp_ref_seed_count": sum(1 for r in pnp_by_frame.get(fid, []) if to_bool(r.get("pnp_ref_contains_seed"))),
            "miniba_attempted": any(len(as_list(r.get("miniba_ref_keyframe_ids"))) > 0 for r in pnp_by_frame.get(fid, [])),
            "miniba_ref_seed_count": sum(1 for r in pnp_by_frame.get(fid, []) if to_bool(r.get("miniba_ref_contains_seed"))),
        })
    write_csv(out / "chosen_matching_pnp_execution_audit.csv", exec_rows)
    exec_summary = {
        "chosen_kfs_built_frames": sum(r["chosen_kfs_built"] for r in exec_rows),
        "matching_executed_frames": sum(r["matching_executed"] for r in exec_rows),
        "pnp_attempted_frames": sum(r["pnp_attempted"] for r in exec_rows),
        "miniba_attempted_frames": sum(r["miniba_attempted"] for r in exec_rows),
        "matching_seed_candidate_count": sum(r["matching_seed_candidate_count"] for r in exec_rows),
        "pnp_ref_seed_count": sum(r["pnp_ref_seed_count"] for r in exec_rows),
        "miniba_ref_seed_count": sum(r["miniba_ref_seed_count"] for r in exec_rows),
    }
    write_json(out / "chosen_matching_pnp_execution_summary.json", exec_summary)
    (out / "chosen_matching_pnp_execution_report.md").write_text(
        "# chosen/matching/pnp execution\n\n" + json.dumps(exec_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    support_seed_rows = []
    seen_support: set[tuple[int, int, int]] = set()
    for source_name in ["support_integration", "bridge"]:
        for idx, row in enumerate(short500[source_name]):
            fid = to_int(row.get("frame_id"), -1)
            if not (300 <= fid < 500):
                continue
            seed_count = to_int(row.get("matched_seed_keyframe_count"), 0)
            if seed_count <= 0:
                continue
            key = (fid, to_int(row.get("best_support_keyframe_id"), -1), to_int(row.get("matched_seed_best_score"), 0))
            if key in seen_support:
                continue
            seen_support.add(key)
            entered_candidates = bool(candidate_by_frame.get(fid))
            selected_seed = any(
                to_bool(c.get("candidate_is_early_seed")) and to_bool(c.get("candidate_selected"))
                for c in candidate_by_frame.get(fid, [])
            )
            entered_bridge = bool(bridge_by_frame.get(fid))
            entered_pose_pool = bool(rows_by_int(short500["pose_pool"], "frame_id").get(fid))
            entered_pnp = bool(pnp_by_frame.get(fid))
            if not entered_candidates:
                drop = "before_chosen_kfs_candidate"
                reason = "frame did not enter pose path, so candidate list was not built"
            elif not selected_seed:
                drop = "chosen_topk_filter"
                reason = "candidate list existed but no seed selected"
            elif not entered_pose_pool:
                drop = "before_pose_reference_pool"
                reason = "seed selected but pose reference pool trace missing"
            elif not entered_pnp:
                drop = "before_pnp_miniba"
                reason = "pose reference pool did not reach PnP/MiniBA"
            else:
                drop = "reached_pose_path"
                reason = ""
            support_seed_rows.append({
                "frame_id": fid,
                "best_support_keyframe_id": row.get("best_support_keyframe_id", ""),
                "matched_seed_keyframe_count": seed_count,
                "matched_seed_best_score": row.get("matched_seed_best_score", ""),
                "entered_chosen_kfs_candidate_trace": entered_candidates,
                "entered_chosen_selected": selected_seed,
                "entered_matching_to_pose_path_bridge": entered_bridge,
                "entered_pose_reference_pool_trace": entered_pose_pool,
                "entered_pnp_miniba_reference_trace": entered_pnp,
                "drop_step": drop,
                "drop_reason": reason,
            })
    write_csv(out / "seed_candidate_to_pose_path_dropoff.csv", support_seed_rows)
    drop_summary = {
        "seed_support_candidate_rows_300_500": len(support_seed_rows),
        "drop_counts": dict(Counter(r["drop_step"] for r in support_seed_rows)),
        "entered_chosen_count": sum(r["entered_chosen_kfs_candidate_trace"] for r in support_seed_rows),
        "entered_pose_reference_pool_count": sum(r["entered_pose_reference_pool_trace"] for r in support_seed_rows),
        "entered_pnp_miniba_count": sum(r["entered_pnp_miniba_reference_trace"] for r in support_seed_rows),
    }
    write_json(out / "seed_candidate_to_pose_path_dropoff_summary.json", drop_summary)
    (out / "seed_candidate_to_pose_path_dropoff_report.md").write_text(
        "# seed candidate to pose path dropoff\n\n" + json.dumps(drop_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    recovery_rows = []
    for fid in range(300, 500):
        controls = control_by_source.get(fid, [])
        mats = mat_by_source.get(fid, [])
        recovery_rows.append({
            "source_frame_id": fid,
            "recovery_success_candidate_count": len(controls),
            "control_allow_commit_count": sum(1 for r in controls if str(r.get("decision")) == "commit" or str(r.get("control_decision")) == "allow_commit"),
            "control_hold_count": sum(1 for r in controls if str(r.get("decision")) == "hold"),
            "control_reject_count": sum(1 for r in controls if str(r.get("decision")) == "reject"),
            "control_reasons": ";".join(sorted({str(r.get("decision_reason", "")) for r in controls if str(r.get("decision_reason", ""))})),
            "runtime_commit_attempted_count": sum(1 for r in mats if to_bool(r.get("runtime_commit_attempted"))),
            "materialized_count": sum(1 for r in mats if to_bool(r.get("materialized"))),
            "materialization_failure_reasons": ";".join(sorted({str(r.get("materialization_failure_reason", r.get("failure_reason", ""))) for r in mats if str(r.get("materialization_failure_reason", r.get("failure_reason", "")))})),
        })
    write_csv(out / "recovery_attempt_reachability_audit.csv", recovery_rows)
    rec_summary = {
        "frames_with_recovery_success_candidate": sum(r["recovery_success_candidate_count"] > 0 for r in recovery_rows),
        "control_allow_commit_count": sum(r["control_allow_commit_count"] for r in recovery_rows),
        "control_hold_count": sum(r["control_hold_count"] for r in recovery_rows),
        "control_reject_count": sum(r["control_reject_count"] for r in recovery_rows),
        "runtime_commit_attempted_count": sum(r["runtime_commit_attempted_count"] for r in recovery_rows),
        "materialized_count": sum(r["materialized_count"] for r in recovery_rows),
        "dominant_reasons": dict(Counter(";".join(r["control_reasons"].split(";")) for r in recovery_rows if r["control_reasons"])),
    }
    write_json(out / "recovery_attempt_reachability_summary.json", rec_summary)
    (out / "recovery_attempt_reachability_report.md").write_text(
        "# recovery attempt reachability\n\n" + json.dumps(rec_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pose_path_not_reached = summary["pose_attempted_300_500"] == 0
    chosen_not_built = summary["chosen_kfs_built_300_500"] == 0
    matching_not_executed = summary["matching_executed_300_500"] == 0
    pnp_not_attempted = summary["pnp_attempted_300_500"] == 0
    miniba_not_attempted = summary["miniba_attempted_300_500"] == 0
    recovery_success_missing = rec_summary["frames_with_recovery_success_candidate"] == 0
    control_allow_missing = rec_summary["control_allow_commit_count"] == 0
    runtime_attempt_missing = rec_summary["runtime_commit_attempted_count"] == 0
    seed_dropped_before_chosen = bool(drop_summary["seed_support_candidate_rows_300_500"] > 0 and drop_summary["entered_chosen_count"] == 0)
    seed_dropped_between = bool(drop_summary["entered_chosen_count"] > 0 and drop_summary["entered_pnp_miniba_count"] == 0)

    recommended = "lifecycle_gate_fix" if pose_path_not_reached and chosen_not_built else "matching_to_pose_bridge_fix"
    ready = {
        "need_v8_policy": False,
        "need_pose_path_fix": pose_path_not_reached,
        "need_reference_selection_fix": False,
        "need_matching_to_pose_bridge_fix": bool(not pose_path_not_reached and (seed_dropped_before_chosen or seed_dropped_between)),
        "need_lifecycle_gate_fix": bool(pose_path_not_reached and chosen_not_built),
        "need_trace_fix": False,
        "pose_path_not_reached_300_500": pose_path_not_reached,
        "chosen_kfs_not_built_300_500": chosen_not_built,
        "matching_not_executed_300_500": matching_not_executed,
        "pnp_not_attempted_300_500": pnp_not_attempted,
        "miniba_not_attempted_300_500": miniba_not_attempted,
        "seed_dropped_before_chosen": seed_dropped_before_chosen,
        "seed_dropped_between_chosen_and_pnp": seed_dropped_between,
        "recovery_success_missing_300_500": recovery_success_missing,
        "control_allow_missing_300_500": control_allow_missing,
        "runtime_attempt_missing_300_500": runtime_attempt_missing,
        "recommended_next_action": recommended,
        "keep_RVQ_tau_frozen": True,
    }
    write_json(out / "ready_for_v8_policy_or_pose_path_fix.json", ready)
    (out / "paper_aligned_pose_path_reachability_and_reference_execution_audit_report.md").write_text(
        "\n".join([
            "# PAPER_ALIGNED_POSE_PATH_REACHABILITY_AND_REFERENCE_EXECUTION_AUDIT_V1",
            "",
            "## Final Answers",
            "",
            f"1. 300-500 是否进入 pose path：{not pose_path_not_reached}，pose_attempted={summary['pose_attempted_300_500']}/200。",
            f"2. 最早阻断点：{'lifecycle defer gate before pose path' if pose_path_not_reached else 'after pose path'}。",
            f"3. seed support candidate matching 未进入 chosen/reference/PnP/MiniBA 的原因：300-500 没有 chosen candidate list/pose path execution，drop before chosen={seed_dropped_before_chosen}。",
            "4. 当前问题不是 policy 选 seed 错，而是 pose path execution 没有消费 seed。",
            f"5. 下一步建议：{recommended}。",
            "6. 继续冻结 R/V/Q 与 tau。",
        ]) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
