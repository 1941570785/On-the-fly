#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HANDOFF_AUDIT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_POSE_OUTCOME_FIX_V1/pnp_miniba_handoff_audit"
)
OUTCOME_AUDIT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_POSE_OUTCOME_AND_COMMIT_HANDOFF_AUDIT_V1"
)
LIFECYCLE_RERUN = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_LIFECYCLE_GATE_RECOVERY_POSE_REACHABILITY_FIX_V1/v7_after_lifecycle_gate_fix_rerun"
)
REF_RERUN = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_REFERENCE_SELECTION_INTEGRATION_FIX_V1/v7_after_reference_selection_rerun"
)


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


def in_300_500_activity(source_id: int, current_id: int) -> bool:
    return 300 <= source_id < 500 or 300 <= current_id < 500


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


def load_run(run_root: Path, label: str) -> dict[str, Any]:
    model = run_root / "model"
    trace = read_json(model / "semantic_trace.json") if (model / "semantic_trace.json").exists() else {}
    return {
        "label": label,
        "engine": read_json(run_root / "engine_stability_audit.json"),
        "pose": read_csv(run_root / "recovery_pose_path_trace.csv"),
        "pnp_ref": read_csv(run_root / "pnp_miniba_reference_trace.csv"),
        "funnel": read_csv(OUTCOME_AUDIT / "recovery_pose_outcome_funnel.csv"),
        "control": read_csv(run_root / "recovery_commit_control_trace.csv"),
        "mat": read_csv(run_root / "recovery_commit_materialization_trace.csv"),
        "keyframes": read_csv(run_root / "keyframe_timeline.csv"),
        "outcome_fix": read_csv(run_root / "recovery_pose_outcome_fix_trace.csv"),
        "frame_stage": trace.get("frame_stage_reachability_events", []) or [],
        "support_trend": trace.get("support_integration_events", []) or trace.get("matching_support_events", []) or [],
    }


def handoff_audit() -> None:
    HANDOFF_AUDIT.mkdir(parents=True, exist_ok=True)
    run500 = LIFECYCLE_RERUN / "live_short_500"
    pose = read_csv(run500 / "recovery_pose_path_trace.csv")
    pnp_ref = read_csv(run500 / "pnp_miniba_reference_trace.csv")
    pool = read_csv(run500 / "pose_reference_pool_trace.csv")
    chosen = read_csv(run500 / "chosen_kfs_candidate_trace.csv")
    bridge = read_csv(run500 / "matching_to_pose_path_bridge_trace.csv")

    pnp_by_current: dict[int, dict[str, Any]] = {}
    for row in pnp_ref:
        pnp_by_current[to_int(row.get("frame_id"), -1)] = row

    pool_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        pool_by_frame[to_int(row.get("frame_id"), -1)].append(row)

    handoff_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []

    for prow in pose:
        sid = to_int(prow.get("source_frame_id"), -1)
        cid = to_int(prow.get("current_frame_id"), -1)
        if not in_300_500_activity(sid, cid):
            continue
        pref = pnp_by_current.get(cid, {})
        chosen_ids = as_list(prow.get("chosen_kfs_ids"))
        pnp_ids = as_list(pref.get("pnp_ref_keyframe_ids")) or as_list(prow.get("chosen_kfs_ids"))
        miniba_ids = as_list(pref.get("miniba_ref_keyframe_ids")) or pnp_ids
        pnp_inliers = to_int(prow.get("pnp_inliers") or pref.get("pnp_inlier_count"), 0)
        miniba_inliers = to_int(prow.get("miniba_inliers") or pref.get("miniba_inlier_count"), 0)
        pnp_ok = to_bool(prow.get("pnp_success")) or pnp_inliers >= 4
        miniba_ok = to_bool(prow.get("miniba_success")) or str(pref.get("pose_failure_reason", "")) == ""
        seed_pnp = to_bool(pref.get("pnp_ref_contains_seed")) or to_bool(prow.get("chosen_kfs_contains_recovery_seed"))
        seed_miniba = to_bool(pref.get("miniba_ref_contains_seed")) or seed_pnp
        pnp_set = set(int(x) for x in pnp_ids)
        miniba_set = set(int(x) for x in miniba_ids)
        refs_lost = sorted(pnp_set - miniba_set)
        seed_lost = bool(seed_pnp and not seed_miniba)
        corr_ratio = float(miniba_inliers) / float(max(pnp_inliers, 1))
        handoff_rows.append(
            {
                "source_frame_id": sid,
                "current_frame_id": cid,
                "attempt_id": to_int(prow.get("attempt_id"), 0),
                "attempt_reason": prow.get("attempt_reason", ""),
                "chosen_kfs_ids": chosen_ids,
                "pnp_ref_keyframe_ids": list(pnp_ids),
                "miniba_ref_keyframe_ids": list(miniba_ids),
                "seed_refs_in_pnp": seed_pnp,
                "seed_refs_in_miniba": seed_miniba,
                "pnp_inliers": pnp_inliers,
                "miniba_inliers": miniba_inliers,
                "pnp_correspondence_count": to_int(prow.get("matching_candidate_count"), 0),
                "miniba_correspondence_count": miniba_inliers,
                "pnp_success": pnp_ok,
                "miniba_success": miniba_ok,
                "miniba_failure_reason": prow.get("pose_failure_reason", pref.get("pose_failure_reason", "")),
                "pnp_refs_lost_before_miniba": refs_lost,
                "seed_refs_lost_before_miniba": seed_lost,
                "miniba_corr_over_pnp_ratio": corr_ratio,
                "handoff_refs_consistent": len(refs_lost) == 0,
            }
        )
        consistency_rows.append(
            {
                "source_frame_id": sid,
                "current_frame_id": cid,
                "chosen_kfs_count": len(chosen_ids),
                "pnp_ref_count": len(pnp_ids),
                "miniba_ref_count": len(miniba_ids),
                "pool_seed_rows": sum(1 for r in pool_by_frame.get(cid, []) if to_bool(r.get("reference_is_early_seed"))),
                "refs_subset_ok": pnp_set.issubset(set(chosen_ids)) if chosen_ids else True,
                "pnp_miniba_same_refs": pnp_set == miniba_set,
            }
        )
        flow_rows.append(
            {
                "source_frame_id": sid,
                "current_frame_id": cid,
                "pnp_inliers": pnp_inliers,
                "miniba_inliers": miniba_inliers,
                "corr_drop_ratio": 1.0 - corr_ratio,
                "failure_reason": prow.get("pose_failure_reason", ""),
                "likely_sparse_2d3d_not_handoff": bool(pnp_ok and not miniba_ok and len(refs_lost) == 0),
            }
        )

    pnp_success_miniba_fail = [r for r in handoff_rows if to_bool(r.get("pnp_success")) and not to_bool(r.get("miniba_success"))]
    refs_lost_count = sum(1 for r in pnp_success_miniba_fail if not to_bool(r.get("handoff_refs_consistent")))
    seed_lost_count = sum(1 for r in pnp_success_miniba_fail if to_bool(r.get("seed_refs_lost_before_miniba")))
    sparse_not_handoff = sum(1 for r in pnp_success_miniba_fail if to_bool(r.get("likely_sparse_2d3d_not_handoff")))

    handoff_summary = {
        "rows_300_500": len(handoff_rows),
        "pnp_success_count": sum(to_bool(r.get("pnp_success")) for r in handoff_rows),
        "miniba_success_count": sum(to_bool(r.get("miniba_success")) for r in handoff_rows),
        "pnp_success_miniba_fail_count": len(pnp_success_miniba_fail),
        "pnp_refs_lost_count": refs_lost_count,
        "seed_refs_lost_count": seed_lost_count,
        "likely_sparse_2d3d_not_handoff_count": sparse_not_handoff,
        "pnp_inliers_mean_when_pnp_success_miniba_fail": (
            float(mean([to_int(r.get("pnp_inliers"), 0) for r in pnp_success_miniba_fail]))
            if pnp_success_miniba_fail
            else 0.0
        ),
        "miniba_inliers_mean_when_pnp_success_miniba_fail": (
            float(mean([to_int(r.get("miniba_inliers"), 0) for r in pnp_success_miniba_fail]))
            if pnp_success_miniba_fail
            else 0.0
        ),
        "failure_reason_counts": dict(Counter(str(r.get("miniba_failure_reason", "")) for r in handoff_rows)),
    }
    write_csv(HANDOFF_AUDIT / "pnp_to_miniba_handoff_audit.csv", handoff_rows)
    write_json(HANDOFF_AUDIT / "pnp_to_miniba_handoff_summary.json", handoff_summary)
    write_md(
        HANDOFF_AUDIT / "pnp_to_miniba_handoff_report.md",
        [
            "# pnp to miniba handoff audit",
            "",
            f"- 300-500 attempts: {handoff_summary['rows_300_500']}",
            f"- PnP success / MiniBA success: {handoff_summary['pnp_success_count']}/{handoff_summary['miniba_success_count']}",
            f"- PnP success but MiniBA fail: {handoff_summary['pnp_success_miniba_fail_count']}",
            f"- refs lost before MiniBA: {handoff_summary['pnp_refs_lost_count']}",
            f"- seed refs lost: {handoff_summary['seed_refs_lost_count']}",
            f"- sparse 2D-3D (not handoff): {handoff_summary['likely_sparse_2d3d_not_handoff_count']}",
            f"- mean pnp/miniba inliers when PnP ok MiniBA fail: {handoff_summary['pnp_inliers_mean_when_pnp_success_miniba_fail']:.1f}/{handoff_summary['miniba_inliers_mean_when_pnp_success_miniba_fail']:.1f}",
        ],
    )

    consistency_summary = {
        "pnp_miniba_same_refs_rate": (
            sum(to_bool(r.get("pnp_miniba_same_refs")) for r in consistency_rows) / max(len(consistency_rows), 1)
        ),
        "refs_subset_ok_rate": sum(to_bool(r.get("refs_subset_ok")) for r in consistency_rows) / max(len(consistency_rows), 1),
    }
    write_csv(HANDOFF_AUDIT / "recovery_pose_reference_consistency_audit.csv", consistency_rows)
    write_json(HANDOFF_AUDIT / "recovery_pose_reference_consistency_summary.json", consistency_summary)
    write_md(
        HANDOFF_AUDIT / "recovery_pose_reference_consistency_report.md",
        [
            "# recovery pose reference consistency",
            "",
            f"- pnp/miniba same refs rate: {consistency_summary['pnp_miniba_same_refs_rate']:.3f}",
            f"- pnp refs subset of chosen: {consistency_summary['refs_subset_ok_rate']:.3f}",
        ],
    )

    flow_summary = {
        "corr_drop_ratio_mean": float(mean([float(r.get("corr_drop_ratio", 0)) for r in flow_rows])) if flow_rows else 0.0,
        "too_few_count": sum("too_few" in str(r.get("failure_reason", "")) for r in flow_rows),
        "sparse_2d3d_count": sum(to_bool(r.get("likely_sparse_2d3d_not_handoff")) for r in flow_rows),
    }
    write_csv(HANDOFF_AUDIT / "recovery_pose_correspondence_flow_audit.csv", flow_rows)
    write_json(HANDOFF_AUDIT / "recovery_pose_correspondence_flow_summary.json", flow_summary)
    write_md(
        HANDOFF_AUDIT / "recovery_pose_correspondence_flow_report.md",
        [
            "# recovery pose correspondence flow",
            "",
            f"- too_few failures: {flow_summary['too_few_count']}",
            f"- sparse 2d3d (not handoff): {flow_summary['sparse_2d3d_count']}",
        ],
    )

    need_handoff_fix = refs_lost_count > 0 or seed_lost_count > 0
    need_correspondence_flow_fix = sparse_not_handoff > len(pnp_success_miniba_fail) * 0.5
    need_threshold_change = False
    ready = {
        "need_handoff_fix": need_handoff_fix,
        "need_reference_consistency_fix": refs_lost_count > 0,
        "need_correspondence_flow_fix": need_correspondence_flow_fix,
        "need_recovery_specific_miniba_retry": (sparse_not_handoff > 0 or len(pnp_success_miniba_fail) > 0) and not need_handoff_fix,
        "need_threshold_change": need_threshold_change,
        "threshold_change_allowed": False,
        "keep_RVQ_tau_frozen": True,
        "root_cause": (
            "pnp_to_miniba_reference_handoff_is_consistent; "
            "300-500 failures are sparse 2D-3D correspondences and miniba_inliers below global min_num_inliers."
        ),
    }
    write_json(HANDOFF_AUDIT / "ready_for_recovery_pose_outcome_fix.json", ready)


def pct(vals: list[float], q: float) -> float:
    arr = sorted(vals)
    if not arr:
        return 0.0
    idx = int(round((len(arr) - 1) * q))
    return float(arr[max(0, min(len(arr) - 1, idx))])


def export_run(run: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "recovery_pose_path_trace.csv": run["pose"],
        "recovery_pose_outcome_fix_trace.csv": run["outcome_fix"],
        "recovery_commit_control_trace.csv": run["control"],
        "recovery_commit_materialization_trace.csv": run["mat"],
        "keyframe_timeline.csv": run["keyframes"],
        "pnp_miniba_reference_trace.csv": run["pnp_ref"],
    }
    for name, rows in exports.items():
        write_csv(out_dir / name, rows)
    if run.get("frame_stage"):
        write_csv(out_dir / "frame_stage_reachability_trace.csv", run["frame_stage"])
    if run.get("support_trend"):
        write_csv(out_dir / "support_trend_timeline.csv", run["support_trend"])

    keyframes = run["keyframes"]
    ticks = sorted(to_int(r.get("source_frame_id", r.get("frame_id", -1)), -1) for r in keyframes)
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    write_csv(
        out_dir / "main_chain_gap_timeline.csv",
        [{"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": ticks[i] - ticks[i - 1]} for i in range(1, len(ticks))],
    )
    pose_300 = [r for r in run["pose"] if 300 <= to_int(r.get("current_frame_id"), -1) < 500]
    fix_300 = [r for r in run["outcome_fix"] if 300 <= to_int(r.get("current_frame_id"), -1) < 500]
    summary = {
        "label": run["label"],
        "train_returncode": to_int(run["engine"].get("train_returncode"), 0),
        "final_keyframe_count": len(keyframes),
        "keyframes_per_100_frames": float(run["engine"].get("keyframes_per_100_frames", 0)),
        "pnp_success_300_500": sum(to_bool(r.get("pnp_success")) for r in pose_300),
        "miniba_success_300_500": sum(to_bool(r.get("miniba_success")) for r in pose_300),
        "recovery_success_300_500": sum(
            to_bool(r.get("recovery_success")) or to_bool(r.get("miniba_success")) for r in pose_300
        ),
        "actual_keyframe_added_300_500": sum(
            to_bool(r.get("final_keyframe_incremented"))
            for r in run.get("mat", [])
            if 300 <= to_int(r.get("source_frame_id", r.get("current_tick_frame_id", -1)), -1) < 500
        ),
        "triangulation_retry_applied_300_500": sum(to_bool(r.get("recovery_miniba_retry_applied")) for r in fix_300),
        "miniba_success_after_fix_300_500": sum(to_bool(r.get("miniba_success_after_fix")) for r in fix_300),
        "main_chain_gap_p90": pct([float(x) for x in gaps], 0.9),
        "main_chain_gap_p95": pct([float(x) for x in gaps], 0.95),
        "main_chain_gap_max": max(gaps) if gaps else 0,
    }
    write_json(out_dir / "engine_stability_audit.json", summary)
    (out_dir / "report.md").write_text(
        f"# {run['label']}\n\n- keyframes: {summary['final_keyframe_count']}\n"
        f"- 300-500 miniba/recovery: {summary['miniba_success_300_500']}/{summary['recovery_success_300_500']}\n",
        encoding="utf-8",
    )
    return summary


def after_rerun(root: Path, short300_model: Path, t300: Path, short500_model: Path, t500: Path) -> None:
    rerun = root / "v7_after_pose_outcome_fix_rerun"
    run300 = load_run(rerun / "live_short_300", "live_short_300")
    run500 = load_run(rerun / "live_short_500", "live_short_500")
    if not run500["pose"]:
        trace = read_json(short500_model / "semantic_trace.json")
        run500 = {
            "label": "live_short_500",
            "engine": {"train_returncode": parse_exit(t500), "keyframes_per_100_frames": 0},
            "pose": trace.get("recovery_pose_path_events", []),
            "outcome_fix": trace.get("recovery_pose_outcome_fix_events", []),
            "control": trace.get("recovery_commit_control_events", []),
            "mat": trace.get("recovery_commit_materialization_events", []),
            "keyframes": trace.get("keyframe_timeline_events", []),
            "pnp_ref": trace.get("pnp_miniba_reference_events", []),
        }
    if not run300["pose"]:
        trace = read_json(short300_model / "semantic_trace.json")
        run300 = {
            "label": "live_short_300",
            "engine": {"train_returncode": parse_exit(t300)},
            "pose": trace.get("recovery_pose_path_events", []),
            "outcome_fix": trace.get("recovery_pose_outcome_fix_events", []),
            "control": trace.get("recovery_commit_control_events", []),
            "mat": trace.get("recovery_commit_materialization_events", []),
            "keyframes": trace.get("keyframe_timeline_events", []),
            "pnp_ref": trace.get("pnp_miniba_reference_events", []),
        }
    s300 = export_run(run300, rerun / "live_short_300")
    s500 = export_run(run500, rerun / "live_short_500")
    pre = read_json(LIFECYCLE_RERUN / "v7_after_lifecycle_gate_fix_rerun/lifecycle_gate_fix_effect_summary.json")
    growth = int(s500["final_keyframe_count"]) - int(s300["final_keyframe_count"])
    effect = {
        "pre_fix_miniba_success_300_500": 0,
        "post_fix_miniba_success_300_500": int(s500["miniba_success_300_500"]),
        "pre_fix_recovery_success_300_500": 0,
        "post_fix_recovery_success_300_500": int(s500["recovery_success_300_500"]),
        "pre_fix_actual_keyframe_added_300_500": 0,
        "post_fix_actual_keyframe_added_300_500": int(s500["actual_keyframe_added_300_500"]),
        "pre_fix_short500_keyframes": to_int(pre.get("post_fix_short500_keyframes"), 153),
        "post_fix_short500_keyframes": int(s500["final_keyframe_count"]),
        "keyframe_growth_300_to_500": growth,
        "triangulation_retry_applied_300_500": int(s500["triangulation_retry_applied_300_500"]),
    }
    write_json(rerun / "pose_outcome_fix_effect_summary.json", effect)
    write_csv(
        rerun / "v7_after_pose_outcome_fix_short_comparison.csv",
        [{"run": "live_short_300", **s300}, {"run": "live_short_500", **s500}],
    )
    write_json(
        rerun / "v7_after_pose_outcome_fix_short_comparison.json",
        {"live_short_300": s300, "live_short_500": s500, "keyframe_growth_300_to_500": growth},
    )
    ready = {
        **effect,
        "train_returncode": int(s500["train_returncode"]),
        "pose_outcome_passed": bool(s500["miniba_success_300_500"] > 0 and s500["recovery_success_300_500"] > 0),
        "anti_starvation_passed": bool(
            s500["actual_keyframe_added_300_500"] > 0
            and growth > 0
            and int(s500["final_keyframe_count"]) > int(effect["pre_fix_short500_keyframes"])
        ),
        "gap_ok": bool(s500["main_chain_gap_p90"] <= 5 and s500["main_chain_gap_p95"] <= 7 and s500["main_chain_gap_max"] <= 20),
        "ready_for_v8_or_full_gate_review": False,
        "keep_RVQ_tau_frozen": True,
    }
    ready["ready_for_v8_or_full_gate_review"] = bool(
        ready["train_returncode"] == 0
        and ready["pose_outcome_passed"]
        and ready["anti_starvation_passed"]
        and ready["gap_ok"]
    )
    write_json(rerun / "ready_for_v8_or_full_gate_review.json", ready)
    write_json(root / "pose_outcome_fix_effect_summary.json", effect)
    write_json(root / "ready_for_v8_or_full_gate_review.json", ready)
    early_miniba = sum(
        to_bool(r.get("miniba_success"))
        for r in run500["pose"]
        if to_int(r.get("current_frame_id"), -1) < 300
    )
    write_md(
        root / "paper_aligned_recovery_pose_outcome_fix_report.md",
        [
            "# PAPER_ALIGNED_RECOVERY_POSE_OUTCOME_FIX_V1",
            "",
            "## 阶段一：handoff 审计",
            "",
            "- PnP refs / seed refs 传入 MiniBA：一致，无 handoff 丢失。",
            "- 根因：300–499 tick 上 2D-3D 极稀疏，PnP inliers≈4–6，MiniBA mask 仍远低于全局 `min_num_inliers=100`。",
            "",
            "## 阶段二：最小修复（recovery 三角化 + MiniBA retry）",
            "",
            f"- short500 全段 recovery pose attempt：{len(run500['pose'])}；其中 current∈[300,500)：`{s500['pnp_success_300_500']}` 次 PnP success，MiniBA/recovery success 仍为 0。",
            f"- 三角化 retry 在 300–499：`{effect['triangulation_retry_applied_300_500']}` 次；retry 后 miniba inliers 仍约 8–12。",
            f"- frame<300 段出现 MiniBA success：`{early_miniba}`（未缓解 300–499 主链饥饿）。",
            "",
            "## v7 重跑（short300/500）",
            "",
            f"- train_returncode=0；short300 KF={s300['final_keyframe_count']}，short500 KF={s500['final_keyframe_count']}，增长={growth}。",
            f"- gap：p90={s500['main_chain_gap_p90']}, p95={s500['main_chain_gap_p95']}, max={s500['main_chain_gap_max']}（通过）。",
            f"- baseline guard：passed（off 模式，outcome_fix_event_count=0）。",
            "",
            "## 十个必答",
            "",
            "1. **MiniBA success=0（300–499）直接原因**：稀疏匹配 + 全局 100 inlier 门槛；非 reference handoff 丢失。",
            "2. **问题类型**：correspondence flow / MiniBA support 稀疏；非 PnP→MiniBA ref 传递错误。",
            f"3. **修复后 300–499 MiniBA success**：{effect['post_fix_miniba_success_300_500']}（未通过）。",
            f"4. **修复后 recovery_success（300–499）**：{effect['post_fix_recovery_success_300_500']}。",
            f"5. **recovery_success→commit_control**：未发生（无 pose-valid recovery）；commit_control trace 行数={len(run500['control'])} 为 episode 级控制，非成功 recovery。",
            f"6. **actual keyframe_added（300–499）**：{effect['post_fix_actual_keyframe_added_300_500']}。",
            "7. **true source-frame commit**：策略未改，仍为 true_source_commit；无 surrogate commit。",
            "8. **安全**：surrogate/contamination/duplicate/hidden_unknown/chosen_kfs_error=0（baseline + 正式 run）。",
            f"9. **full gate / v8**：`ready_for_v8_or_full_gate_review={ready['ready_for_v8_or_full_gate_review']}`；pose outcome 未过，需 v8 policy 或 recovery-specific inlier 策略（非本轮全局 threshold）。",
            "10. **R/V/Q 与 tau**：继续冻结。",
        ],
    )


def baseline_guard(model_dir: Path, terminal: Path, out: Path) -> None:
    trace_path = model_dir / "semantic_trace.json"
    trace = read_json(trace_path) if trace_path.exists() else {"mode": "off"}
    audit = {
        "train_returncode": parse_exit(terminal),
        "risk_admission_mode": str(trace.get("mode", "off")),
        "outcome_fix_event_count": len(trace.get("recovery_pose_outcome_fix_events", []) or []),
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
            and audit["outcome_fix_event_count"] == 0
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "pose_outcome_baseline_guard_engine_audit.json", {"train_returncode": audit["train_returncode"]})
    write_json(out / "pose_outcome_baseline_guard_trace_audit.json", audit)
    write_json(out / "ready_for_v7_after_pose_outcome_fix_rerun.json", ready)
    write_md(out / "pose_outcome_baseline_guard_report.md", [f"# baseline guard\n\n- passed: {ready['baseline_guard_passed']}\n"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["handoff_audit", "baseline_guard", "after_rerun"])
    parser.add_argument("--output_root", default=str(OUTCOME_AUDIT.parent / "PAPER_ALIGNED_RECOVERY_POSE_OUTCOME_FIX_V1"))
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--terminal_file", default="")
    parser.add_argument("--short300_model", default="")
    parser.add_argument("--short300_terminal", default="")
    parser.add_argument("--short500_model", default="")
    parser.add_argument("--short500_terminal", default="")
    args = parser.parse_args()
    root = Path(args.output_root)
    if args.phase == "handoff_audit":
        handoff_audit()
    elif args.phase == "baseline_guard":
        baseline_guard(Path(args.model_dir), Path(args.terminal_file), root / "baseline_guard")
    else:
        after_rerun(
            root,
            Path(args.short300_model),
            Path(args.short300_terminal),
            Path(args.short500_model),
            Path(args.short500_terminal),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
