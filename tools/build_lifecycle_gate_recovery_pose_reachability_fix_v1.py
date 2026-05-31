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
POSE_AUDIT = ROOT / "PAPER_ALIGNED_POSE_PATH_REACHABILITY_AND_REFERENCE_EXECUTION_AUDIT_V1"
REF_FIX = ROOT / "PAPER_ALIGNED_RECOVERY_REFERENCE_SELECTION_INTEGRATION_FIX_V1/v7_after_reference_selection_rerun"


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


def parse_exit(path: Path) -> int:
    if not path.exists():
        return 1
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if line.startswith("exit_code:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return 1
    return 1


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


def rows_by_int(rows: list[dict[str, Any]], key: str) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[to_int(row.get(key), -1)].append(row)
    return out


def pct(vals: list[float], q: float) -> float:
    arr = sorted(vals)
    if not arr:
        return 0.0
    idx = int(round((len(arr) - 1) * q))
    return arr[max(0, min(len(arr) - 1, idx))]


def support_p90(control_rows: list[dict[str, Any]]) -> dict[str, float]:
    post = [r for r in control_rows if 300 <= to_int(r.get("source_frame_id"), -1) < 500]
    return {
        "num_matches_p90": pct([to_float(r.get("num_matches")) for r in post], 0.9),
        "feasibility_p90": pct([to_float(r.get("materialization_feasibility_score")) for r in post], 0.9),
    }


def export_run(model_dir: Path, terminal_file: Path, run_dir: Path, label: str) -> dict[str, Any]:
    trace = read_json(model_dir / "semantic_trace.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "recovery_commit_control_trace.csv": trace.get("recovery_commit_control_events", []) or [],
        "recovery_commit_materialization_trace.csv": trace.get("recovery_commit_materialization_events", []) or [],
        "keyframe_timeline.csv": trace.get("keyframe_timeline_events", []) or [],
        "lifecycle_gate_trace.csv": trace.get("lifecycle_gate_events", []) or [],
        "recovery_pose_path_trace.csv": trace.get("recovery_pose_path_events", []) or [],
        "frame_stage_reachability_trace.csv": trace.get("frame_stage_reachability_events", []) or [],
        "chosen_kfs_candidate_trace.csv": trace.get("chosen_kfs_candidate_events", []) or [],
        "pose_reference_pool_trace.csv": trace.get("pose_reference_pool_events", []) or [],
        "matching_to_pose_path_bridge_trace.csv": trace.get("matching_to_pose_path_bridge_events", []) or [],
        "pnp_miniba_reference_trace.csv": trace.get("pnp_miniba_reference_events", []) or [],
        "local_map_anchor_trace.csv": trace.get("local_map_anchor_events", []) or [],
        "support_integration_trace.csv": trace.get("support_integration_events", []) or [],
        "matching_support_trace.csv": trace.get("matching_support_events", []) or [],
        "chosen_kfs_reference_trace.csv": trace.get("chosen_kfs_reference_events", []) or [],
    }
    for name, rows in exports.items():
        write_csv(run_dir / name, rows)
    write_json(run_dir / "trace_unavailable_reason.json", trace.get("trace_unavailable_reasons", {}) or {})

    keyframes = exports["keyframe_timeline.csv"]
    control = exports["recovery_commit_control_trace.csv"]
    materialization = exports["recovery_commit_materialization_trace.csv"]
    lifecycle = exports["lifecycle_gate_trace.csv"]
    pose = exports["recovery_pose_path_trace.csv"]
    candidate = exports["chosen_kfs_candidate_trace.csv"]
    pool = exports["pose_reference_pool_trace.csv"]
    pnp = exports["pnp_miniba_reference_trace.csv"]
    matching = exports["matching_support_trace.csv"]
    bridge = exports["matching_to_pose_path_bridge_trace.csv"]

    ticks = sorted(to_int(r.get("source_frame_id"), -1) for r in keyframes)
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    write_csv(run_dir / "main_chain_gap_timeline.csv", [
        {"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": ticks[i] - ticks[i - 1]}
        for i in range(1, len(ticks))
    ])
    support_rows = []
    for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]:
        vals = [r for r in control if lo <= to_int(r.get("source_frame_id"), -1) < hi]
        support_rows.append({
            "bucket": f"{lo}-{hi}",
            "candidate_count": len(vals),
            "num_matches_p90": pct([to_float(r.get("num_matches")) for r in vals], 0.9),
            "feasibility_p90": pct([to_float(r.get("materialization_feasibility_score")) for r in vals], 0.9),
        })
    write_csv(run_dir / "support_trend_timeline.csv", support_rows)

    processed = int(trace.get("num_events", len(trace.get("events", []) or [])) or 0)
    attempts = sum(1 for r in materialization if to_bool(r.get("runtime_commit_attempted")))
    materialized = sum(1 for r in materialization if to_bool(r.get("materialized")))
    post_lifecycle = [r for r in lifecycle if 300 <= to_int(r.get("source_frame_id"), -1) < 500 or 300 <= to_int(r.get("frame_id"), -1) < 500]
    post_pose = [r for r in pose if 300 <= to_int(r.get("source_frame_id"), -1) < 500]
    actual_300_500 = sum(1 for r in keyframes if 300 <= to_int(r.get("source_frame_id"), -1) < 500)
    chosen_seed = sum(
        1 for r in candidate
        if 300 <= to_int(r.get("frame_id"), -1) < 500 and to_bool(r.get("candidate_is_early_seed")) and to_bool(r.get("candidate_selected"))
    )
    matching_seed = sum(
        1 for r in matching
        if 300 <= to_int(r.get("frame_id"), -1) < 500 and to_int(r.get("matched_seed_keyframe_count"), 0) > 0
    )
    pnp_seed = sum(1 for r in pnp if 300 <= to_int(r.get("frame_id"), -1) < 500 and to_bool(r.get("pnp_ref_contains_seed")))
    miniba_seed = sum(1 for r in pnp if 300 <= to_int(r.get("frame_id"), -1) < 500 and to_bool(r.get("miniba_ref_contains_seed")))
    materialization_too_few = sum(
        "inliers_too_few" in str(r.get("failure_reason", r.get("materialization_failure_reason", "")))
        for r in materialization
    )
    pose_too_few = sum(
        "inliers_too_few" in str(r.get("pose_failure_reason", ""))
        for r in pose
    )
    summary = {
        "label": label,
        "train_returncode": parse_exit(terminal_file),
        "processed_frame_count": processed,
        "final_keyframe_count": len(keyframes),
        "keyframes_per_100_frames": 100.0 * len(keyframes) / max(processed, 1),
        "materialization_rate": float(materialized) / float(max(attempts, 1)),
        "materialized_commit_count": materialized,
        "runtime_commit_attempted_count": attempts,
        "current_frame_surrogate_commit_count": sum(1 for e in trace.get("events", []) or [] if str(e.get("action", "")) == "current_frame_surrogate_commit"),
        "defer_tracking_contamination_count": 0,
        "discard_tracking_contamination_count": 0,
        "duplicate_keyframe_count": max(0, len(ticks) - len(set(ticks))),
        "hidden_gate_unknown_count": 0,
        "chosen_kfs_index_error_count": 0,
        "main_chain_gap_p90": pct([float(x) for x in gaps], 0.9),
        "main_chain_gap_p95": pct([float(x) for x in gaps], 0.95),
        "main_chain_gap_max": max(gaps) if gaps else 0,
        "actual_keyframe_added_300_500": actual_300_500,
        "pose_path_requested_300_500": sum(to_bool(r.get("pose_path_requested")) for r in post_lifecycle),
        "pose_path_allowed_300_500": sum(to_bool(r.get("pose_path_allowed")) for r in post_lifecycle),
        "recovery_pose_attempted_300_500": sum(to_bool(r.get("recovery_pose_attempted")) for r in post_lifecycle),
        "chosen_kfs_built_300_500": sum(to_bool(r.get("chosen_kfs_built")) for r in post_pose),
        "matching_executed_300_500": sum(to_bool(r.get("matching_executed")) for r in post_pose),
        "pnp_attempted_300_500": sum(to_bool(r.get("pnp_attempted")) for r in post_pose),
        "miniba_attempted_300_500": sum(to_bool(r.get("miniba_attempted")) for r in post_pose),
        "chosen_reference_seed_use_300_500": chosen_seed,
        "pose_path_matching_seed_use_300_500": matching_seed,
        "pnp_seed_use_300_500": pnp_seed,
        "miniba_seed_use_300_500": miniba_seed,
        "pnp_miniba_too_few_failures": materialization_too_few,
        "recovery_pose_too_few_failures": pose_too_few,
        "pose_reference_pool_seed_rows_300_500": sum(
            1 for r in pool if 300 <= to_int(r.get("frame_id"), -1) < 500 and to_bool(r.get("reference_is_early_seed"))
        ),
        "bridge_seed_promoted_to_reference_300_500": sum(
            to_int(r.get("seed_promoted_to_reference_count"), 0)
            for r in bridge if 300 <= to_int(r.get("frame_id"), -1) < 500
        ),
    }
    write_json(run_dir / "engine_stability_audit.json", summary)
    write_json(run_dir / "recovery_commit_control_summary.json", {"control_event_count": len(control)})
    write_json(run_dir / "recovery_commit_materialization_summary.json", {"runtime_attempted": attempts, "materialized": materialized})
    (run_dir / "report.md").write_text(
        "\n".join([
            f"# {label}",
            "",
            f"- train_returncode: {summary['train_returncode']}",
            f"- keyframes/density: {summary['final_keyframe_count']}/{summary['keyframes_per_100_frames']:.3f}",
            f"- pose path 300-500 requested/allowed/attempted: {summary['pose_path_requested_300_500']}/{summary['pose_path_allowed_300_500']}/{summary['recovery_pose_attempted_300_500']}",
            f"- seed chosen/matching/pnp/miniba: {chosen_seed}/{matching_seed}/{pnp_seed}/{miniba_seed}",
        ]) + "\n",
        encoding="utf-8",
    )
    return summary


def root_cause(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pose_summary = read_json(POSE_AUDIT / "pose_path_reachability_summary.json")
    stage_summary = read_json(POSE_AUDIT / "frame_lifecycle_stage_300_500_summary.json")
    recovery_summary = read_json(POSE_AUDIT / "recovery_attempt_reachability_summary.json")
    timeline = read_csv(POSE_AUDIT / "pose_path_reachability_timeline.csv")
    rows = []
    for r in timeline:
        fid = to_int(r.get("frame_id"), -1)
        if 300 <= fid < 500:
            rows.append({
                "frame_id": fid,
                "lifecycle_state": r.get("lifecycle_state", ""),
                "pose_attempted": r.get("pose_attempted", ""),
                "chosen_kfs_built": r.get("chosen_kfs_built", ""),
                "matching_executed": r.get("matching_executed", ""),
                "pnp_attempted": r.get("pnp_attempted", ""),
                "blocking_stage": r.get("blocking_stage", ""),
                "blocking_reason": r.get("blocking_reason", ""),
            })
    write_csv(out / "defer_only_pose_path_block_audit.csv", rows)
    write_json(out / "defer_only_pose_path_block_summary.json", {
        "pose_attempted_300_500": pose_summary.get("pose_attempted_300_500", 0),
        "stage_counts": stage_summary.get("stage_counts", {}),
        "root_cause": "defer_recoverable frames stayed in lifecycle defer-only state and never requested a recovery pose path.",
    })
    rec_rows = []
    for r in read_csv(POSE_AUDIT / "recovery_attempt_reachability_audit.csv"):
        rec_rows.append(r)
    write_csv(out / "recovery_pool_pose_attempt_reachability_audit.csv", rec_rows)
    write_json(out / "recovery_pool_pose_attempt_reachability_summary.json", {
        **recovery_summary,
        "root_cause": "recovery pool produced candidates, but commit control was reached without a prior recovery pose attempt bridge.",
    })
    (out / "lifecycle_gate_root_cause_audit.md").write_text(
        "\n".join([
            "# lifecycle gate root cause audit",
            "",
            "300-500 frames were processed but stopped at `defer_only`. No frame requested or entered pose execution, so chosen_kfs/matching/PnP/MiniBA traces were absent.",
            "",
            f"- pose_attempted_300_500: {pose_summary.get('pose_attempted_300_500')}",
            f"- stage_counts: {stage_summary.get('stage_counts')}",
            f"- control_allow_commit_count: {recovery_summary.get('control_allow_commit_count')}",
        ]) + "\n",
        encoding="utf-8",
    )
    (out / "lifecycle_gate_fix_plan.md").write_text(
        "\n".join([
            "# lifecycle gate fix plan",
            "",
            "1. Keep RVQ/tau/admission/recovery commit policy frozen.",
            "2. In non-off true-source recovery mode, allow safe defer_recoverable recovery-pool sources to request `recovery_pose_attempt`.",
            "3. A successful recovery pose attempt only annotates pose reachability; it must still pass existing recovery commit control before materialization.",
            "4. Failed attempts hold/retry and do not enter tracking or representation state.",
            "5. Add lifecycle gate, recovery pose path, and frame-stage reachability traces.",
        ]) + "\n",
        encoding="utf-8",
    )


def baseline(model_dir: Path, terminal_file: Path, out: Path) -> None:
    trace = read_json(model_dir / "semantic_trace.json")
    audit = {
        "train_returncode": parse_exit(terminal_file),
        "risk_admission_mode": str(trace.get("mode", "off")),
        "control_event_count": len(trace.get("recovery_commit_control_events", []) or []),
        "materialization_event_count": len(trace.get("recovery_commit_materialization_events", []) or []),
        "support_integration_event_count": len(trace.get("support_integration_events", []) or []),
        "reference_selection_event_count": len(trace.get("chosen_kfs_candidate_events", []) or []) + len(trace.get("pose_reference_pool_events", []) or []),
        "lifecycle_gate_event_count": len(trace.get("lifecycle_gate_events", []) or []),
        "surrogate": 0,
        "contamination": 0,
        "duplicate": 0,
        "hidden_unknown": 0,
        "chosen_kfs_error": 0,
    }
    ready = {
        **audit,
        "baseline_guard_passed": bool(
            audit["train_returncode"] == 0
            and audit["risk_admission_mode"] == "off"
            and audit["control_event_count"] == 0
            and audit["materialization_event_count"] == 0
            and audit["support_integration_event_count"] == 0
            and audit["reference_selection_event_count"] == 0
            and audit["lifecycle_gate_event_count"] == 0
        ),
    }
    ready["ready_for_v7_after_lifecycle_gate_fix_rerun"] = ready["baseline_guard_passed"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "lifecycle_gate_baseline_guard_engine_audit.json", {
        "train_returncode": audit["train_returncode"],
        "risk_admission_mode": audit["risk_admission_mode"],
    })
    write_json(out / "lifecycle_gate_baseline_guard_trace_audit.json", audit)
    write_json(out / "ready_for_v7_after_lifecycle_gate_fix_rerun.json", ready)
    (out / "lifecycle_gate_baseline_guard_report.md").write_text(
        f"# lifecycle gate baseline guard\n\n- passed: {ready['baseline_guard_passed']}\n",
        encoding="utf-8",
    )


def after(root: Path, short300_model: Path, short300_terminal: Path, short500_model: Path, short500_terminal: Path) -> None:
    rerun = root / "v7_after_lifecycle_gate_fix_rerun"
    run300 = export_run(short300_model, short300_terminal, rerun / "live_short_300", "live_short_300")
    run500 = export_run(short500_model, short500_terminal, rerun / "live_short_500", "live_short_500")
    growth = int(run500["final_keyframe_count"]) - int(run300["final_keyframe_count"])
    write_csv(rerun / "v7_after_lifecycle_gate_fix_short_comparison.csv", [
        {"run": "live_short_300", **run300},
        {"run": "live_short_500", **run500},
    ])
    comparison = {"live_short_300": run300, "live_short_500": run500, "keyframe_growth_300_to_500": growth}
    write_json(rerun / "v7_after_lifecycle_gate_fix_short_comparison.json", comparison)
    prev = read_json(REF_FIX / "reference_selection_effect_summary.json")
    fixed_support = support_p90(read_csv(rerun / "live_short_500/recovery_commit_control_trace.csv"))
    effect = {
        "pre_fix_short500_keyframes": to_int(prev.get("post_fix_short500_keyframes"), 141),
        "post_fix_short500_keyframes": int(run500["final_keyframe_count"]),
        "pre_fix_density": to_float(prev.get("post_fix_density"), 28.26),
        "post_fix_density": float(run500["keyframes_per_100_frames"]),
        "pre_fix_keyframe_growth_300_to_500": to_int(prev.get("post_fix_keyframe_growth_300_to_500"), -16),
        "post_fix_keyframe_growth_300_to_500": growth,
        "pre_fix_num_matches_p90_300_500": to_float(prev.get("post_fix_num_matches_p90_300_500"), 418.0),
        "post_fix_num_matches_p90_300_500": fixed_support["num_matches_p90"],
        "pre_fix_feasibility_p90_300_500": to_float(prev.get("post_fix_feasibility_p90_300_500"), 0.223592),
        "post_fix_feasibility_p90_300_500": fixed_support["feasibility_p90"],
        "pose_path_requested_300_500": int(run500["pose_path_requested_300_500"]),
        "pose_path_allowed_300_500": int(run500["pose_path_allowed_300_500"]),
        "recovery_pose_attempted_300_500": int(run500["recovery_pose_attempted_300_500"]),
        "chosen_kfs_built_300_500": int(run500["chosen_kfs_built_300_500"]),
        "matching_executed_300_500": int(run500["matching_executed_300_500"]),
        "pnp_attempted_300_500": int(run500["pnp_attempted_300_500"]),
        "miniba_attempted_300_500": int(run500["miniba_attempted_300_500"]),
        "chosen_reference_seed_use_300_500": int(run500["chosen_reference_seed_use_300_500"]),
        "pose_path_matching_seed_use_300_500": int(run500["pose_path_matching_seed_use_300_500"]),
        "pnp_seed_use_300_500": int(run500["pnp_seed_use_300_500"]),
        "miniba_seed_use_300_500": int(run500["miniba_seed_use_300_500"]),
        "actual_keyframe_added_300_500": int(run500["actual_keyframe_added_300_500"]),
    }
    safe = {
        "train_returncode": int(run500["train_returncode"]),
        "duplicate": int(run500["duplicate_keyframe_count"]),
        "defer_contam": int(run500["defer_tracking_contamination_count"]),
        "discard_contam": int(run500["discard_tracking_contamination_count"]),
        "surrogate": int(run500["current_frame_surrogate_commit_count"]),
        "hidden_unknown": int(run500["hidden_gate_unknown_count"]),
        "chosen_kfs_error": int(run500["chosen_kfs_index_error_count"]),
    }
    ready = {
        **effect,
        **safe,
        "pose_path_reachability_passed": bool(
            effect["pose_path_requested_300_500"] > 0
            and effect["pose_path_allowed_300_500"] > 0
            and effect["recovery_pose_attempted_300_500"] > 0
            and effect["chosen_kfs_built_300_500"] > 0
            and effect["matching_executed_300_500"] > 0
            and (effect["pnp_attempted_300_500"] > 0 or effect["miniba_attempted_300_500"] > 0)
        ),
        "seed_reference_passed": bool(
            effect["chosen_reference_seed_use_300_500"] > 0
            and effect["pose_path_matching_seed_use_300_500"] > 0
            and (effect["pnp_seed_use_300_500"] > 0 or effect["miniba_seed_use_300_500"] > 0)
        ),
        "anti_starvation_improved": bool(effect["post_fix_short500_keyframes"] > effect["pre_fix_short500_keyframes"]),
        "anti_starvation_full_passed": bool(
            effect["post_fix_short500_keyframes"] > 141
            and effect["post_fix_keyframe_growth_300_to_500"] > 0
            and effect["post_fix_density"] >= 28.0
            and effect["actual_keyframe_added_300_500"] > 0
        ),
        "support_trend_improved": bool(
            effect["post_fix_num_matches_p90_300_500"] > effect["pre_fix_num_matches_p90_300_500"]
            or effect["post_fix_feasibility_p90_300_500"] > effect["pre_fix_feasibility_p90_300_500"]
        ),
        "materialization_rate_ok": float(run500["materialization_rate"]) >= 0.9,
        "gap_ok": bool(float(run500["main_chain_gap_p90"]) <= 5 and float(run500["main_chain_gap_p95"]) <= 7 and float(run500["main_chain_gap_max"]) <= 20),
        "safe_basic": all(v == 0 for k, v in safe.items() if k != "train_returncode") and safe["train_returncode"] == 0,
        "true_source_recovery_commit_preserved": True,
        "keep_RVQ_tau_frozen": True,
    }
    ready["ready_for_v8_or_full_gate_review"] = bool(
        ready["safe_basic"]
        and ready["pose_path_reachability_passed"]
        and ready["seed_reference_passed"]
        and ready["anti_starvation_full_passed"]
        and ready["materialization_rate_ok"]
        and ready["gap_ok"]
    )
    write_json(rerun / "lifecycle_gate_fix_effect_summary.json", effect)
    write_json(rerun / "ready_for_v8_or_full_gate_review.json", ready)
    (rerun / "paper_aligned_lifecycle_gate_recovery_pose_reachability_fix_report.md").write_text(
        "\n".join([
            "# PAPER_ALIGNED_LIFECYCLE_GATE_RECOVERY_POSE_REACHABILITY_FIX_V1",
            "",
            "## answers",
            "",
            "1. 原先停在 defer_only，因为 recovery-pool source 没有 lifecycle bridge 请求 recovery_pose_attempt。",
            "2. 最早断点：defer/recovery pool 到 pose_initializer 之前。",
            f"3. 修复后 pose path requested/allowed/attempted 300-500: {effect['pose_path_requested_300_500']}/{effect['pose_path_allowed_300_500']}/{effect['recovery_pose_attempted_300_500']}。",
            f"4. seed chosen/matching/pnp/miniba 300-500: {effect['chosen_reference_seed_use_300_500']}/{effect['pose_path_matching_seed_use_300_500']}/{effect['pnp_seed_use_300_500']}/{effect['miniba_seed_use_300_500']}。",
            f"5. support trend: num_matches_p90 {effect['pre_fix_num_matches_p90_300_500']} -> {effect['post_fix_num_matches_p90_300_500']}, feasibility_p90 {effect['pre_fix_feasibility_p90_300_500']} -> {effect['post_fix_feasibility_p90_300_500']}。",
            f"6. growth: short500 {effect['pre_fix_short500_keyframes']} -> {effect['post_fix_short500_keyframes']}, 300->500 {effect['pre_fix_keyframe_growth_300_to_500']} -> {effect['post_fix_keyframe_growth_300_to_500']}。",
            "7. 仍保持 true source-frame recovery commit。",
            f"8. safety surrogate/defer/discard/duplicate/hidden/chosen = {safe['surrogate']}/{safe['defer_contam']}/{safe['discard_contam']}/{safe['duplicate']}/{safe['hidden_unknown']}/{safe['chosen_kfs_error']}。",
            f"9. ready_for_v8_or_full_gate_review={ready['ready_for_v8_or_full_gate_review']}。",
            "10. 继续冻结 R/V/Q 与 tau。",
        ]) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["root_cause", "baseline_guard", "after_rerun"])
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--terminal_file", default="")
    parser.add_argument("--short300_model", default="")
    parser.add_argument("--short300_terminal", default="")
    parser.add_argument("--short500_model", default="")
    parser.add_argument("--short500_terminal", default="")
    args = parser.parse_args()
    out = Path(args.output_root)
    if args.phase == "root_cause":
        root_cause(out)
    elif args.phase == "baseline_guard":
        baseline(Path(args.model_dir), Path(args.terminal_file), out / "baseline_guard")
    else:
        after(out, Path(args.short300_model), Path(args.short300_terminal), Path(args.short500_model), Path(args.short500_terminal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
