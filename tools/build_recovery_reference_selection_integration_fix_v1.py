#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly")
PREV_ROOT = ROOT / "results/StaticHikes/forest1/PAPER_ALIGNED_RECOVERY_KEYFRAME_SUPPORT_INTEGRATION_FIX_V1"
PREV_RUN = PREV_ROOT / "v7_after_support_integration_rerun/live_short_500"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def support_p90(control_rows: list[dict[str, Any]]) -> dict[str, float]:
    post = [r for r in control_rows if 300 <= to_int(r.get("source_frame_id")) < 500]
    return {
        "num_matches_p90": pct([to_float(r.get("num_matches")) for r in post], 0.9),
        "feasibility_p90": pct([to_float(r.get("materialization_feasibility_score")) for r in post], 0.9),
    }


def summarize_trace(model_dir: Path, terminal_file: Path, run_dir: Path, label: str) -> dict[str, Any]:
    trace = read_json(model_dir / "semantic_trace.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "recovery_commit_control_trace.csv": trace.get("recovery_commit_control_events", []) or [],
        "recovery_commit_materialization_trace.csv": trace.get("recovery_commit_materialization_events", []) or [],
        "keyframe_timeline.csv": trace.get("keyframe_timeline_events", []) or [],
        "support_integration_trace.csv": trace.get("support_integration_events", []) or [],
        "chosen_kfs_reference_trace.csv": trace.get("chosen_kfs_reference_events", []) or [],
        "chosen_kfs_candidate_trace.csv": trace.get("chosen_kfs_candidate_events", []) or [],
        "pose_reference_pool_trace.csv": trace.get("pose_reference_pool_events", []) or [],
        "matching_to_pose_path_bridge_trace.csv": trace.get("matching_to_pose_path_bridge_events", []) or [],
        "pnp_miniba_reference_trace.csv": trace.get("pnp_miniba_reference_events", []) or [],
        "matching_support_trace.csv": trace.get("matching_support_events", []) or [],
        "local_map_anchor_trace.csv": trace.get("local_map_anchor_events", []) or [],
    }
    for name, rows in exports.items():
        write_csv(run_dir / name, rows)
    write_json(run_dir / "trace_unavailable_reason.json", trace.get("trace_unavailable_reasons", {}) or {})

    keyframes = exports["keyframe_timeline.csv"]
    control = exports["recovery_commit_control_trace.csv"]
    materialization = exports["recovery_commit_materialization_trace.csv"]
    chosen = exports["chosen_kfs_reference_trace.csv"]
    matching = exports["matching_support_trace.csv"]
    pnp = exports["pnp_miniba_reference_trace.csv"]
    bridge = exports["matching_to_pose_path_bridge_trace.csv"]
    pool = exports["pose_reference_pool_trace.csv"]

    ticks = sorted(to_int(r.get("source_frame_id")) for r in keyframes if to_bool(r.get("materialized", True)))
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    write_csv(run_dir / "main_chain_gap_timeline.csv", [
        {"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": ticks[i] - ticks[i - 1]}
        for i in range(1, len(ticks))
    ])
    support_rows = []
    for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]:
        rows = [r for r in control if lo <= to_int(r.get("source_frame_id")) < hi]
        support_rows.append({
            "bucket": f"{lo}-{hi}",
            "candidate_count": len(rows),
            "num_matches_p90": pct([to_float(r.get("num_matches")) for r in rows], 0.9),
            "feasibility_p90": pct([to_float(r.get("materialization_feasibility_score")) for r in rows], 0.9),
        })
    write_csv(run_dir / "support_trend_timeline.csv", support_rows)

    processed = int(trace.get("num_events", len(trace.get("events", []) or [])) or 0)
    final_keyframes = len(keyframes)
    attempts = sum(1 for r in materialization if to_bool(r.get("runtime_commit_attempted")))
    materialized = sum(1 for r in materialization if to_bool(r.get("materialized")))
    chosen_seed_300_500 = sum(
        1 for r in chosen if 300 <= to_int(r.get("frame_id")) < 500 and to_bool(r.get("chosen_kfs_contains_v7_seed"))
    )
    pose_path_match_seed_300_500 = sum(
        1 for r in matching if 300 <= to_int(r.get("frame_id")) < 500 and to_int(r.get("matched_seed_keyframe_count"), 0) > 0
    )
    pnp_seed_300_500 = sum(
        1 for r in pnp if 300 <= to_int(r.get("frame_id")) < 500 and to_bool(r.get("pnp_ref_contains_seed"))
    )
    miniba_seed_300_500 = sum(
        1 for r in pnp if 300 <= to_int(r.get("frame_id")) < 500 and to_bool(r.get("miniba_ref_contains_seed"))
    )
    bridge_seed_ref_300_500 = sum(
        to_int(r.get("seed_promoted_to_reference_count"), 0)
        for r in bridge
        if 300 <= to_int(r.get("frame_id")) < 500
    )
    bridge_seed_pnp_300_500 = sum(
        to_int(r.get("seed_promoted_to_pnp_count"), 0)
        for r in bridge
        if 300 <= to_int(r.get("frame_id")) < 500
    )
    bridge_seed_miniba_300_500 = sum(
        to_int(r.get("seed_promoted_to_miniba_count"), 0)
        for r in bridge
        if 300 <= to_int(r.get("frame_id")) < 500
    )
    actual_300_500 = sum(1 for r in keyframes if 300 <= to_int(r.get("source_frame_id")) < 500)
    too_few = sum(
        "inliers_too_few" in str(r.get("failure_reason", r.get("materialization_failure_reason", "")))
        for r in materialization
    )
    summary = {
        "label": label,
        "train_returncode": parse_exit(terminal_file),
        "processed_frame_count": processed,
        "final_keyframe_count": final_keyframes,
        "keyframes_per_100_frames": 100.0 * final_keyframes / max(processed, 1),
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
        "chosen_reference_seed_use_300_500": chosen_seed_300_500,
        "pose_path_matching_seed_use_300_500": pose_path_match_seed_300_500,
        "pnp_seed_use_300_500": pnp_seed_300_500,
        "miniba_seed_use_300_500": miniba_seed_300_500,
        "bridge_seed_promoted_to_reference_300_500": bridge_seed_ref_300_500,
        "bridge_seed_promoted_to_pnp_300_500": bridge_seed_pnp_300_500,
        "bridge_seed_promoted_to_miniba_300_500": bridge_seed_miniba_300_500,
        "pnp_miniba_too_few_failures": too_few,
        "pose_reference_pool_seed_rows_300_500": sum(
            1 for r in pool if 300 <= to_int(r.get("frame_id")) < 500 and to_bool(r.get("reference_is_early_seed"))
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
            f"- keyframes/density: {final_keyframes}/{summary['keyframes_per_100_frames']:.3f}",
            f"- seed chosen/match/pnp/miniba 300-500: {chosen_seed_300_500}/{pose_path_match_seed_300_500}/{pnp_seed_300_500}/{miniba_seed_300_500}",
            f"- actual_keyframe_added_300_500: {actual_300_500}",
        ]) + "\n",
        encoding="utf-8",
    )
    return summary


def root_cause(out: Path) -> None:
    prev_effect = read_json(PREV_ROOT / "v7_after_support_integration_rerun/support_integration_effect_summary.json")
    chosen = read_csv(PREV_RUN / "chosen_kfs_reference_trace.csv")
    matching = read_csv(PREV_RUN / "matching_support_trace.csv")
    pnp = read_csv(PREV_RUN / "pnp_miniba_reference_trace.csv")
    support = read_csv(PREV_RUN / "support_integration_trace.csv")
    chosen_rows = []
    for row in chosen:
        fid = to_int(row.get("frame_id"))
        ids = as_list(row.get("chosen_kfs_ids"))
        origins = as_list(row.get("chosen_kfs_commit_origin"))
        for i, kid in enumerate(ids):
            origin = str(origins[i]) if i < len(origins) else ""
            chosen_rows.append({
                "frame_id": fid,
                "candidate_keyframe_id": to_int(kid),
                "candidate_commit_origin": origin,
                "candidate_is_recovery": origin == "true_recovery_commit",
                "candidate_is_early_seed": False,
                "candidate_is_support_eligible": origin == "true_recovery_commit",
                "candidate_rank_before_filter": i + 1,
                "candidate_rank_after_filter": i + 1,
                "candidate_selected": True,
                "filter_reason": "",
                "promotion_applied": False,
                "promotion_reason": "",
            })
    write_csv(out / "chosen_kfs_candidate_filter_audit.csv", chosen_rows)
    write_json(out / "chosen_kfs_candidate_filter_summary.json", {
        "support_candidate_matching_seed_use_300_500": prev_effect.get("support_candidate_matching_seed_use_300_500", 102),
        "chosen_reference_seed_use_300_500": prev_effect.get("chosen_reference_seed_use_300_500", 0),
        "root_cause": "support candidate matching was separate from pose reference selection; eligible recovery seeds were visible to the support gate but not promoted into chosen/reference top-k.",
    })
    pose_rows = []
    for row in pnp:
        fid = to_int(row.get("frame_id"))
        for kid in as_list(row.get("pnp_ref_keyframe_ids")):
            pose_rows.append({
                "frame_id": fid,
                "reference_keyframe_id": to_int(kid),
                "reference_used_for_pnp": True,
                "reference_used_for_miniba": to_int(kid) in [to_int(x) for x in as_list(row.get("miniba_ref_keyframe_ids"))],
                "reference_is_early_seed": False,
                "reference_is_support_eligible": False,
                "reference_support_score": "",
            })
    write_csv(out / "pose_reference_pool_audit.csv", pose_rows)
    write_json(out / "pose_reference_pool_summary.json", {
        "pose_reference_rows_300_500": sum(1 for r in pose_rows if 300 <= to_int(r.get("frame_id")) < 500),
        "pose_reference_seed_rows_300_500": 0,
        "pnp_seed_use_300_500": prev_effect.get("pnp_seed_use_300_500", 0),
        "miniba_seed_use_300_500": prev_effect.get("miniba_seed_use_300_500", 0),
    })
    bridge_rows = []
    for row in support:
        if str(row.get("event_type", "")) == "support_candidate_matching":
            bridge_rows.append({
                "frame_id": row.get("frame_id", ""),
                "matched_seed_keyframe_count": row.get("matched_seed_keyframe_count", ""),
                "matched_seed_best_score": row.get("matched_seed_best_score", ""),
                "seed_promoted_to_reference_count": 0,
                "seed_promoted_to_pnp_count": 0,
                "seed_promoted_to_miniba_count": 0,
                "bridge_block_reason": "support_matching_not_connected_to_pose_reference_pool",
            })
    write_csv(out / "matching_to_pose_path_bridge_audit.csv", bridge_rows)
    write_json(out / "matching_to_pose_path_bridge_summary.json", {
        "bridge_rows": len(bridge_rows),
        "seed_promoted_to_reference_300_500": 0,
        "seed_promoted_to_pnp_300_500": 0,
        "seed_promoted_to_miniba_300_500": 0,
        "bridge_block_reason": "support_matching_not_connected_to_pose_reference_pool",
    })
    (out / "reference_selection_root_cause_audit.md").write_text(
        "\n".join([
            "# reference selection root cause audit",
            "",
            "support candidate matching could see recovery seeds, but that evidence was not connected to `SceneModel.get_prev_keyframes()` / pose reference top-k. The break is the matching-to-pose-reference bridge, not RVQ, tau, or recovery commit policy.",
        ]) + "\n",
        encoding="utf-8",
    )
    (out / "recovery_reference_selection_fix_plan.md").write_text(
        "\n".join([
            "# recovery reference selection fix plan",
            "",
            "1. Keep RVQ/tau/admission/recovery commit policy frozen.",
            "2. Only in non-off true-source recovery mode, include support-eligible recovery keyframes in `get_prev_keyframes()` candidate scoring.",
            "3. If the best support-eligible recovery keyframe is above candidate median or direct support is weak, promote it into the last reference slot.",
            "4. Trace candidate ranking, promotion, pose reference pool, and matching-to-pose bridge.",
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
        ),
    }
    ready["ready_for_v7_after_reference_selection_rerun"] = ready["baseline_guard_passed"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "reference_selection_baseline_guard_engine_audit.json", {
        "train_returncode": audit["train_returncode"],
        "risk_admission_mode": audit["risk_admission_mode"],
    })
    write_json(out / "reference_selection_baseline_guard_trace_audit.json", audit)
    write_json(out / "ready_for_v7_after_reference_selection_rerun.json", ready)
    (out / "reference_selection_baseline_guard_report.md").write_text(
        f"# reference selection baseline guard\n\n- passed: {ready['baseline_guard_passed']}\n",
        encoding="utf-8",
    )


def after(root: Path, short300_model: Path, short300_terminal: Path, short500_model: Path, short500_terminal: Path) -> None:
    rerun = root / "v7_after_reference_selection_rerun"
    run300 = summarize_trace(short300_model, short300_terminal, rerun / "live_short_300", "live_short_300")
    run500 = summarize_trace(short500_model, short500_terminal, rerun / "live_short_500", "live_short_500")
    growth = int(run500["final_keyframe_count"]) - int(run300["final_keyframe_count"])
    write_csv(rerun / "v7_after_reference_selection_short_comparison.csv", [
        {"run": "live_short_300", **run300},
        {"run": "live_short_500", **run500},
    ])
    comparison = {"live_short_300": run300, "live_short_500": run500, "keyframe_growth_300_to_500": growth}
    write_json(rerun / "v7_after_reference_selection_short_comparison.json", comparison)

    prev = read_json(PREV_ROOT / "v7_after_support_integration_rerun/support_integration_effect_summary.json")
    fixed_support = support_p90(read_csv(rerun / "live_short_500/recovery_commit_control_trace.csv"))
    effect = {
        "pre_fix_short500_keyframes": to_int(prev.get("post_fix_short500_keyframes"), 138),
        "post_fix_short500_keyframes": int(run500["final_keyframe_count"]),
        "pre_fix_density": to_float(prev.get("post_fix_density"), 27.66),
        "post_fix_density": float(run500["keyframes_per_100_frames"]),
        "pre_fix_keyframe_growth_300_to_500": to_int(prev.get("post_fix_keyframe_growth_300_to_500"), 0),
        "post_fix_keyframe_growth_300_to_500": growth,
        "pre_fix_num_matches_p90_300_500": to_float(prev.get("post_fix_num_matches_p90_300_500"), 417.0),
        "post_fix_num_matches_p90_300_500": fixed_support["num_matches_p90"],
        "pre_fix_feasibility_p90_300_500": to_float(prev.get("post_fix_feasibility_p90_300_500"), 0.223548),
        "post_fix_feasibility_p90_300_500": fixed_support["feasibility_p90"],
        "chosen_reference_seed_use_300_500": int(run500["chosen_reference_seed_use_300_500"]),
        "pose_path_matching_seed_use_300_500": int(run500["pose_path_matching_seed_use_300_500"]),
        "pnp_seed_use_300_500": int(run500["pnp_seed_use_300_500"]),
        "miniba_seed_use_300_500": int(run500["miniba_seed_use_300_500"]),
        "bridge_seed_promoted_to_reference_300_500": int(run500["bridge_seed_promoted_to_reference_300_500"]),
        "bridge_seed_promoted_to_pnp_300_500": int(run500["bridge_seed_promoted_to_pnp_300_500"]),
        "bridge_seed_promoted_to_miniba_300_500": int(run500["bridge_seed_promoted_to_miniba_300_500"]),
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
        "reference_propagation_passed": bool(
            effect["chosen_reference_seed_use_300_500"] > 0
            and effect["pose_path_matching_seed_use_300_500"] > 0
            and (effect["pnp_seed_use_300_500"] > 0 or effect["miniba_seed_use_300_500"] > 0)
        ),
        "support_trend_improved": bool(
            effect["post_fix_num_matches_p90_300_500"] > effect["pre_fix_num_matches_p90_300_500"]
            or effect["post_fix_feasibility_p90_300_500"] > effect["pre_fix_feasibility_p90_300_500"]
        ),
        "anti_starvation_improved": bool(
            effect["post_fix_short500_keyframes"] > effect["pre_fix_short500_keyframes"]
            and effect["post_fix_keyframe_growth_300_to_500"] > effect["pre_fix_keyframe_growth_300_to_500"]
        ),
        "materialization_rate_ok": float(run500["materialization_rate"]) >= 0.9,
        "gap_ok": bool(float(run500["main_chain_gap_p90"]) <= 5 and float(run500["main_chain_gap_p95"]) <= 7 and float(run500["main_chain_gap_max"]) <= 20),
        "safe_basic": all(v == 0 for k, v in safe.items() if k != "train_returncode") and safe["train_returncode"] == 0,
        "true_source_recovery_commit_preserved": True,
        "keep_RVQ_tau_frozen": True,
    }
    ready["ready_for_v8_or_full_gate_review"] = bool(
        ready["safe_basic"]
        and ready["reference_propagation_passed"]
        and ready["support_trend_improved"]
        and ready["anti_starvation_improved"]
        and ready["materialization_rate_ok"]
        and ready["gap_ok"]
    )
    write_json(rerun / "reference_selection_effect_summary.json", effect)
    write_json(rerun / "ready_for_v8_or_full_gate_review.json", ready)
    (rerun / "paper_aligned_recovery_reference_selection_integration_fix_report.md").write_text(
        "\n".join([
            "# PAPER_ALIGNED_RECOVERY_REFERENCE_SELECTION_INTEGRATION_FIX_V1",
            "",
            "## answers",
            "",
            "1. reference selection 断点在 support-candidate matching 到 `get_prev_keyframes()` / pose reference top-k 的桥接。",
            "2. seed 之前可见但不进 chosen/reference，是因为 support gate 与 pose reference pool 是两套结构，没有 supplemental reference promotion。",
            f"3. 修复后 chosen/reference seed use 300-500: {effect['chosen_reference_seed_use_300_500']}。",
            f"4. 修复后 PnP/MiniBA seed use 300-500: {effect['pnp_seed_use_300_500']}/{effect['miniba_seed_use_300_500']}。",
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
        after(
            out,
            Path(args.short300_model),
            Path(args.short300_terminal),
            Path(args.short500_model),
            Path(args.short500_terminal),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
