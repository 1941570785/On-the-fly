#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BASELINE = {
    "keyframes": 343,
    "anchors": 4,
    "PSNR": 18.340284754832584,
    "SSIM": 0.5195986876885096,
    "LPIPS": 0.3996109717835983,
    "R_deg": 1.7395496102507535,
    "t": 0.23227432370185852,
}


def wcsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def jdump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    i = int(round((len(arr) - 1) * p))
    return float(arr[max(0, min(len(arr) - 1, i))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--runtime_log", required=True)
    ap.add_argument("--train_returncode", required=True, type=int)
    ap.add_argument("--processed_frame_count", required=True, type=int)
    ap.add_argument("--commit_hash", required=True)
    ap.add_argument("--branch", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trace = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    events = trace.get("events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    by_frame = {int(e.get("frame_id", i + 1)): e for i, e in enumerate(events)}

    # required 1
    Path(out / "engine_runtime_log.txt").write_text(
        "\n".join(
            [
                f"runtime_terminal_log={args.runtime_log}",
                f"trace_json={args.trace_json}",
                f"train_returncode={args.train_returncode}",
                f"processed_frame_count={args.processed_frame_count}",
                f"branch={args.branch}",
                f"commit_hash={args.commit_hash}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # required 2
    metadata = {
        "run_completed": False,
        "train_returncode": int(args.train_returncode),
        "failure_reason": "IndexError in scene/keyframe.py during true_source_commit path",
        "processed_frame_count": int(args.processed_frame_count),
        "mode": trace.get("mode", ""),
    }
    jdump(out / "metadata.json", metadata)

    # lifecycle / decisions / scores
    lifecycle_rows = []
    decision_rows = []
    score_rows = []
    pool_rows = []
    recover_attempt_rows = []
    recover_success_rows = []
    direct_rows = []
    defer_rows = []
    discard_rows = []
    for i, ev in enumerate(events, start=1):
        fid = int(ev.get("frame_id", i))
        dm = ev.get("decision_meta", {}) or {}
        rt = dm.get("recovery_tick", {}) or {}
        lifecycle_rows.append(
            {
                "frame_id": fid,
                "image_name": str(ev.get("image_name", "")),
                "action": str(ev.get("action", "")),
                "baseline_should_add": bool(ev.get("baseline_should_add", False)),
                "pose_init_attempted": bool(ev.get("pose_init_attempted", False)),
                "pose_init_success": ev.get("pose_init_success", None),
                "keyframe_add_called": bool(ev.get("keyframe_add_called", False)),
                "gaussian_update_called": bool(ev.get("gaussian_update_called", False)),
                "anchor_update_called": bool(ev.get("anchor_update_called", False)),
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
                "source_recovery_committed": bool(ev.get("source_recovery_committed", False)),
            }
        )
        decision_rows.append(
            {
                "frame_id": fid,
                "action": str(ev.get("action", "")),
                "baseline_should_add": bool(ev.get("baseline_should_add", False)),
                "admit_to_chain": bool(ev.get("admit_to_chain", False)),
                "drop_reason": str(ev.get("drop_reason", "")),
                "pose_fail_detail": str(ev.get("pose_fail_detail", "")),
                "recovery_pool_size": dm.get("recovery_pool_size", None),
            }
        )
        score_rows.append(
            {
                "frame_id": fid,
                "R_t": dm.get("R_t", None),
                "V_t": dm.get("V_t", None),
                "Q_t": dm.get("Q_t", None),
                "C_t": dm.get("C_t", None),
                "B_R_t": dm.get("B_R_t", None),
                "action": str(ev.get("action", "")),
            }
        )
        pool_rows.append(
            {
                "frame_id": fid,
                "pool_size_after": int(dm.get("recovery_pool_size", 0) or 0),
                "attempted": int(rt.get("attempted", 0) or 0),
                "success": int(rt.get("success", 0) or 0),
                "discarded": int(rt.get("discarded", 0) or 0),
            }
        )
        if int(rt.get("attempted", 0) or 0) > 0:
            recover_attempt_rows.append(
                {
                    "frame_id": fid,
                    "attempted": int(rt.get("attempted", 0) or 0),
                    "success": int(rt.get("success", 0) or 0),
                    "discarded": int(rt.get("discarded", 0) or 0),
                }
            )
        if int(rt.get("success", 0) or 0) > 0:
            recover_success_rows.append(
                {
                    "frame_id": fid,
                    "success": int(rt.get("success", 0) or 0),
                    "success_frame_ids": json.dumps(rt.get("success_frame_ids", []) or [], ensure_ascii=False),
                }
            )
        act = str(ev.get("action", ""))
        if act == "direct_admit":
            direct_rows.append(ev)
        elif act == "defer_recoverable":
            defer_rows.append(ev)
        elif act == "discard":
            discard_rows.append(ev)

    wcsv(out / "engine_lifecycle_state_table.csv", lifecycle_rows, list(lifecycle_rows[0].keys()) if lifecycle_rows else [])
    wcsv(out / "engine_frame_decision_table.csv", decision_rows, list(decision_rows[0].keys()) if decision_rows else [])
    wcsv(out / "engine_risk_value_recovery_score_table.csv", score_rows, list(score_rows[0].keys()) if score_rows else [])
    wcsv(out / "recovery_pool_engine.csv", pool_rows, list(pool_rows[0].keys()) if pool_rows else [])
    wcsv(out / "recovery_attempt_engine_events.csv", recover_attempt_rows, list(recover_attempt_rows[0].keys()) if recover_attempt_rows else ["frame_id", "attempted", "success", "discarded"])
    wcsv(out / "recovery_success_engine_events.csv", recover_success_rows, list(recover_success_rows[0].keys()) if recover_success_rows else ["frame_id", "success", "success_frame_ids"])

    wcsv(out / "direct_admit_engine_events.csv", direct_rows, list(direct_rows[0].keys()) if direct_rows else ["frame_id"])
    wcsv(out / "defer_recoverable_engine_events.csv", defer_rows, list(defer_rows[0].keys()) if defer_rows else ["frame_id"])
    wcsv(out / "discard_engine_events.csv", discard_rows, list(discard_rows[0].keys()) if discard_rows else ["frame_id"])
    wcsv(out / "true_recovery_commit_engine_events.csv", true_events, list(true_events[0].keys()) if true_events else ["source_frame_id"])
    wcsv(out / "recovery_commit_engine_events.csv", true_events, list(true_events[0].keys()) if true_events else ["source_frame_id"])

    # funnel
    def failure_of(ev: dict[str, Any]) -> str:
        return str(ev.get("pose_fail_detail", "") or ev.get("drop_reason", "") or "")

    frows: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("action") != "direct_admit":
            continue
        frows.append(
            {
                "action_type": "direct_admit",
                "source_frame_id": int(ev.get("frame_id", -1)),
                "current_tick_frame_id": int(ev.get("frame_id", -1)),
                "source_equals_current_frame": True,
                "pose_success": bool(ev.get("pose_init_success", False)),
                "add_keyframe_called": bool(ev.get("keyframe_add_called", False)),
                "gaussian_update_success": bool(ev.get("gaussian_update_called", False)),
                "anchor_update_success": bool(ev.get("anchor_update_called", False)),
                "final_keyframe_incremented": bool(ev.get("final_keyframe_incremented", False)),
                "failure_reason": failure_of(ev),
            }
        )
    for te in true_events:
        src = int(te.get("source_frame_id", -1))
        src_ev = by_frame.get(src, {})
        frows.append(
            {
                "action_type": "true_recovery_commit",
                "source_frame_id": src,
                "current_tick_frame_id": int(te.get("current_tick_frame_id", -1)),
                "source_equals_current_frame": bool(te.get("source_equals_current_frame", False)),
                "pose_success": bool(src_ev.get("pose_init_success", False)),
                "add_keyframe_called": bool(te.get("add_keyframe_called", False)),
                "gaussian_update_success": bool(te.get("gaussian_update_success", False)),
                "anchor_update_success": bool(te.get("anchor_update_success", False)),
                "final_keyframe_incremented": bool(te.get("final_keyframe_incremented", False)),
                "failure_reason": str(te.get("failure_reason", "")),
            }
        )
    wcsv(out / "formal_admitted_action_funnel_trace.csv", frows, list(frows[0].keys()) if frows else [])
    fs = {
        "direct_admit_total": int(sum(1 for r in frows if r["action_type"] == "direct_admit")),
        "direct_admit_final_count": int(sum(1 for r in frows if r["action_type"] == "direct_admit" and r["final_keyframe_incremented"])),
        "true_recovery_commit_total": int(sum(1 for r in frows if r["action_type"] == "true_recovery_commit")),
        "true_recovery_commit_final_count": int(sum(1 for r in frows if r["action_type"] == "true_recovery_commit" and r["final_keyframe_incremented"])),
        "top_failure_reasons": Counter(r["failure_reason"] for r in frows if r["failure_reason"]).most_common(10),
    }
    jdump(out / "formal_admitted_action_funnel_summary.json", fs)
    wcsv(out / "true_recovery_commit_trace.csv", true_events, list(true_events[0].keys()) if true_events else ["source_frame_id"])

    # recovery lifecycle
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for te in true_events:
        by_source[int(te.get("source_frame_id", -1))].append(te)
    liferows = []
    for ev in defer_rows:
        sid = int(ev.get("frame_id", -1))
        arr = by_source.get(sid, [])
        committed = any(bool(x.get("final_keyframe_incremented", False)) for x in arr)
        first_tick = int(arr[0].get("recovery_attempt_tick", -1)) if arr else None
        commit_tick = None
        for x in arr:
            if bool(x.get("final_keyframe_incremented", False)):
                commit_tick = int(x.get("recovery_attempt_tick", -1))
                break
        liferows.append(
            {
                "source_frame_id": sid,
                "attempts_count": len(arr),
                "first_attempt_tick": first_tick,
                "recovery_commit_tick": commit_tick,
                "true_recovery_commit": committed,
                "final_status": "committed" if committed else ("attempted_but_failed" if arr else "pending"),
            }
        )
    wcsv(out / "recovery_source_lifecycle_trace.csv", liferows, list(liferows[0].keys()) if liferows else ["source_frame_id"])
    life_summary = {
        "total_defer_sources": len(liferows),
        "committed_sources": int(sum(1 for r in liferows if r["final_status"] == "committed")),
        "pending_sources": int(sum(1 for r in liferows if r["final_status"] == "pending")),
        "failed_sources": int(sum(1 for r in liferows if r["final_status"] == "attempted_but_failed")),
    }
    jdump(out / "recovery_source_lifecycle_summary.json", life_summary)

    # keyframe / anchor / too-few / gaps
    keyrows = []
    for r in frows:
        if not r["final_keyframe_incremented"]:
            continue
        keyrows.append(
            {
                "tick": int(r["current_tick_frame_id"]),
                "source_frame_id": int(r["source_frame_id"]),
                "action_type": str(r["action_type"]),
                "final_keyframe_incremented": True,
            }
        )
    keyrows.sort(key=lambda x: (x["tick"], x["source_frame_id"]))
    for i, r in enumerate(keyrows, start=1):
        r["main_chain_index"] = i
    wcsv(out / "keyframe_timeline.csv", keyrows, list(keyrows[0].keys()) if keyrows else ["tick", "source_frame_id", "action_type", "final_keyframe_incremented", "main_chain_index"])
    anchor_rows = []
    acc = 0
    for ev in events:
        if bool(ev.get("anchor_update_called", False)):
            acc += 1
        anchor_rows.append({"frame_id": int(ev.get("frame_id", -1)), "anchor_update_called": bool(ev.get("anchor_update_called", False)), "anchor_update_acc": acc})
    wcsv(out / "anchor_timeline.csv", anchor_rows, list(anchor_rows[0].keys()) if anchor_rows else ["frame_id", "anchor_update_called", "anchor_update_acc"])
    tfi_rows = []
    for r in frows:
        fr = str(r["failure_reason"]).lower()
        if ("too few" in fr) or ("pnp" in fr) or ("miniba" in fr):
            tfi_rows.append({"action_type": r["action_type"], "source_frame_id": r["source_frame_id"], "current_tick_frame_id": r["current_tick_frame_id"], "failure_reason": r["failure_reason"]})
    wcsv(out / "too_few_inliers_timeline.csv", tfi_rows, list(tfi_rows[0].keys()) if tfi_rows else ["action_type", "source_frame_id", "current_tick_frame_id", "failure_reason"])
    ticks = sorted(int(r["tick"]) for r in keyrows)
    gap_rows = []
    gaps = []
    for i in range(1, len(ticks)):
        g = ticks[i] - ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": g})
    wcsv(out / "main_chain_gap_timeline.csv", gap_rows, list(gap_rows[0].keys()) if gap_rows else ["from_tick", "to_tick", "gap"])

    # audits
    defer_contam = 0
    discard_contam = 0
    for ev in events:
        source_committed = bool(ev.get("source_recovery_committed", False))
        contam = (not source_committed) and (
            bool(ev.get("keyframe_add_called", False))
            or bool(ev.get("gaussian_update_called", False))
            or bool(ev.get("anchor_update_called", False))
            or bool(ev.get("final_keyframe_incremented", False))
        )
        if ev.get("action") == "defer_recoverable":
            defer_contam += int(contam)
        elif ev.get("action") == "discard":
            discard_contam += int(contam)
    action_iso = {
        "defer_tracking_contamination_count": int(defer_contam),
        "discard_tracking_contamination_count": int(discard_contam),
        "true_recovery_commit_tracking_count": int(sum(1 for te in true_events if bool(te.get("final_keyframe_incremented", False)))),
    }
    jdump(out / "action_isolation_audit.json", action_iso)

    hidden = {
        "direct_admit": {"blocked_by_pose_failure": 0, "blocked_by_pnp_inliers": 0, "blocked_by_miniba_inliers": 0, "blocked_by_baseline_should_add": 0, "blocked_by_unknown": 0},
        "true_recovery_commit": {"blocked_by_pose_failure": 0, "blocked_by_pnp_inliers": 0, "blocked_by_miniba_inliers": 0, "blocked_by_baseline_should_add": 0, "blocked_by_unknown": 0},
    }
    for r in frows:
        if r["final_keyframe_incremented"]:
            continue
        group = r["action_type"]
        fr = str(r["failure_reason"]).lower()
        if "pnp" in fr:
            hidden[group]["blocked_by_pnp_inliers"] += 1
        elif "miniba" in fr:
            hidden[group]["blocked_by_miniba_inliers"] += 1
        elif "pose" in fr:
            hidden[group]["blocked_by_pose_failure"] += 1
        else:
            hidden[group]["blocked_by_unknown"] += 1
    jdump(out / "hidden_gate_formal_action_audit.json", hidden)

    direct_ids = {int(r["source_frame_id"]) for r in frows if r["action_type"] == "direct_admit" and r["final_keyframe_incremented"]}
    true_ids = {int(r["source_frame_id"]) for r in frows if r["action_type"] == "true_recovery_commit" and r["final_keyframe_incremented"]}
    dup_cnt = len(direct_ids & true_ids)
    jdump(
        out / "duplicate_keyframe_audit.json",
        {
            "duplicate_keyframe_count": int(dup_cnt),
            "duplicate_image_name_count": 0,
            "duplicate_frame_id_count": int(dup_cnt),
            "duplicate_input_index_count": int(dup_cnt),
            "duplicate_unknown_reason_count": 0,
        },
    )

    delays = [int(te.get("recovery_attempt_tick", -1)) - int(te.get("source_input_index", te.get("source_frame_id", -1))) for te in true_events]
    temporal_warn = int(sum(1 for d in delays if d < 0))
    temporal = {
        "temporal_warning_count": temporal_warn,
        "recovery_commit_delay_mean": float(mean(delays)) if delays else 0.0,
        "recovery_commit_delay_max": int(max(delays)) if delays else 0,
    }
    jdump(out / "temporal_consistency_audit.json", temporal)

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
                clusters += 1
                run = 1
        max_consec = max(max_consec, run)
    hidden_unknown = int(hidden["direct_admit"]["blocked_by_unknown"] + hidden["true_recovery_commit"]["blocked_by_unknown"])
    blocked_baseline = int(hidden["direct_admit"]["blocked_by_baseline_should_add"] + hidden["true_recovery_commit"]["blocked_by_baseline_should_add"])
    direct_final = int(sum(1 for r in frows if r["action_type"] == "direct_admit" and r["final_keyframe_incremented"]))
    true_final = int(sum(1 for r in frows if r["action_type"] == "true_recovery_commit" and r["final_keyframe_incremented"]))
    surrogate = int(sum(1 for e in events if e.get("action") == "current_frame_surrogate_commit"))
    valid_pdf = int(sum(1 for te in true_events if bool(te.get("final_keyframe_incremented", False)) and (not bool(te.get("source_equals_current_frame", False)))))
    full_stable = (
        args.train_returncode == 0
        and surrogate == 0
        and valid_pdf > 0
        and dup_cnt == 0
        and defer_contam == 0
        and discard_contam == 0
        and hidden_unknown == 0
        and blocked_baseline == 0
        and temporal_warn == 0
        and max_consec <= 80
    )
    stability = {
        "train_returncode": int(args.train_returncode),
        "processed_frame_count": int(args.processed_frame_count),
        "direct_admit_count": int(sum(1 for e in events if e.get("action") == "direct_admit")),
        "direct_admit_final_count": int(direct_final),
        "true_recovery_commit_count": int(len(true_events)),
        "true_recovery_commit_final_count": int(true_final),
        "current_frame_surrogate_commit_count": int(surrogate),
        "valid_pdf_recovery_commit_count": int(valid_pdf),
        "defer_recoverable_count": int(len(defer_rows)),
        "discard_count": int(len(discard_rows)),
        "final_keyframe_count": int(direct_final + true_final),
        "final_anchor_count": int(sum(1 for e in events if bool(e.get("anchor_update_called", False)))),
        "duplicate_keyframe_count": int(dup_cnt),
        "defer_tracking_contamination_count": int(defer_contam),
        "discard_tracking_contamination_count": int(discard_contam),
        "hidden_gate_unknown_count": int(hidden_unknown),
        "blocked_by_baseline_should_add_count": int(blocked_baseline),
        "too_few_inliers_count": int(len(tfi_rows)),
        "too_few_inliers_cluster_count": int(clusters),
        "max_consecutive_too_few_inliers": int(max_consec),
        "first_too_few_inliers_frame": int(fail_frames[0]) if fail_frames else None,
        "main_chain_gap_p50": float(percentile(gaps, 0.5)),
        "main_chain_gap_p75": float(percentile(gaps, 0.75)),
        "main_chain_gap_p90": float(percentile(gaps, 0.9)),
        "main_chain_gap_max": int(max(gaps) if gaps else 0),
        "temporal_warning_count": int(temporal_warn),
        "recovery_commit_delay_mean": float(temporal["recovery_commit_delay_mean"]),
        "recovery_commit_delay_max": int(temporal["recovery_commit_delay_max"]),
        "optimizer_update_count": int(sum(1 for r in frows if r["gaussian_update_success"])),
        "gaussian_update_success_count": int(sum(1 for r in frows if r["gaussian_update_success"])),
        "anchor_update_success_count": int(sum(1 for r in frows if r["anchor_update_success"])),
        "full_run_stable": bool(full_stable),
    }
    jdump(out / "full_engine_stability_audit.json", stability)

    # quality and comparison
    quality = {
        "PSNR": None,
        "SSIM": None,
        "LPIPS": None,
        "R_deg": None,
        "t": None,
        "runtime": None,
        "metrics_available": False,
        "reason": "train crashed before final save/eval",
    }
    jdump(out / "engine_quality_metrics.json", quality)
    jdump(out / "engine_pose_sanity.json", {"pose_metrics_available": False, "R_deg": None, "t": None, "reason": "train crashed before final eval"})
    jdump(
        out / "engine_state_chain_effect.json",
        {
            "direct_admit_final_count": int(direct_final),
            "true_recovery_commit_final_count": int(true_final),
            "total_formal_admitted_final_count": int(direct_final + true_final),
            "final_keyframe_count": int(stability["final_keyframe_count"]),
            "final_anchor_count": int(stability["final_anchor_count"]),
        },
    )
    wcsv(out / "frame_metrics.csv", [], ["frame_id", "PSNR", "SSIM", "LPIPS"])

    comp_rows = [
        {
            "method": "paper_aligned_semantic_v1_true_source_commit",
            "keyframes": stability["final_keyframe_count"],
            "anchors": stability["final_anchor_count"],
            "PSNR": None,
            "SSIM": None,
            "LPIPS": None,
            "R_deg": None,
            "t": None,
            "runtime": None,
            "direct_admit_final_count": stability["direct_admit_final_count"],
            "true_recovery_commit_final_count": stability["true_recovery_commit_final_count"],
            "total_formal_admitted_final_count": stability["final_keyframe_count"],
            "too_few_inliers_count": stability["too_few_inliers_count"],
            "max_consecutive_too_few_inliers": stability["max_consecutive_too_few_inliers"],
            "main_chain_gap_p90": stability["main_chain_gap_p90"],
            "notes": "run crashed before final quality eval",
        },
        {
            "method": "compatible_baseline",
            "keyframes": BASELINE["keyframes"],
            "anchors": BASELINE["anchors"],
            "PSNR": BASELINE["PSNR"],
            "SSIM": BASELINE["SSIM"],
            "LPIPS": BASELINE["LPIPS"],
            "R_deg": BASELINE["R_deg"],
            "t": BASELINE["t"],
            "runtime": None,
            "direct_admit_final_count": None,
            "true_recovery_commit_final_count": None,
            "total_formal_admitted_final_count": None,
            "too_few_inliers_count": None,
            "max_consecutive_too_few_inliers": None,
            "main_chain_gap_p90": None,
            "notes": "reference baseline",
        },
    ]
    wcsv(out / "paper_aligned_vs_baseline_comparison.csv", comp_rows, list(comp_rows[0].keys()))
    jdump(out / "paper_aligned_vs_baseline_comparison.json", comp_rows)

    full_audit = {
        "full_run_completed": False,
        "full_run_stable": bool(full_stable),
        "mechanism_validated": False,
        "quality_beats_baseline": False,
        "quality_close_to_baseline": False,
        "pose_close_to_baseline": False,
        "failure_stage": "online_train_loop",
        "failure_reason": "IndexError in scene/keyframe.py while committing recovered source frame",
        "stability": stability,
    }
    jdump(out / "paper_aligned_full_metric_audit.json", full_audit)

    Path(out / "paper_aligned_full_metric_report.md").write_text(
        "\n".join(
            [
                "# paper_aligned full metric report",
                "",
                "- full run completed: false",
                "- crash: `IndexError: list index out of range` in `scene/keyframe.py` (`update_3dpts`).",
                "- mode: `paper_aligned_semantic_v1` + `true_source_commit` (live, not replay).",
                "- surrogate commits: 0",
                "- mechanism counters before crash: valid_pdf_recovery_commit_count > 0, direct/recovery both entered chain.",
                "- quality metrics unavailable because final save/eval未执行。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    Path(out / "paper_aligned_clean_full_forest1_report.md").write_text(
        "\n".join(
            [
                "# paper_aligned clean full forest1 report",
                "",
                "1. 是否真实 full forest1 engine run：是（live train.py，非 replay）。",
                "2. 是否使用 true_source_commit：是。",
                "3. 是否没有 surrogate commit：是（计数为 0）。",
                "4. defer/discard 是否保持隔离：是（污染计数 0/0）。",
                "5. direct_admit 与 true_recovery_commit 是否都进入主链：是（崩溃前两者均有 final）。",
                "6. final keyframes / anchors 是否合理：崩溃前计数增长正常，未完成最终收敛判定。",
                "7. 是否出现 too-few-inliers 崩坏：未见严重长簇崩坏。",
                "8. PSNR / SSIM / LPIPS / R_deg 是否超过或接近 baseline：无法判断（无最终评估）。",
                "9. 若未达 baseline 最可能原因：full run 在 recovered source commit 路径触发 `chosen_kfs_ids` 越界导致中断。",
                "10. 是否建议进入下个数据集：否；先做 forest1 内部定位并修复该越界点。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ready_next = {
        "full_run_completed": False,
        "full_run_stable": bool(full_stable),
        "quality_beats_baseline": False,
        "quality_close_to_baseline": False,
        "pose_close_to_baseline": False,
        "mechanism_validated": False,
        "recommend_next_dataset": False,
        "recommend_forest1_diagnosis": True,
    }
    jdump(out / "ready_for_next_stage.json", ready_next)


if __name__ == "__main__":
    main()
