#!/usr/bin/env python3
"""GAP_SOURCE_ATTRIBUTION_AUDIT_V1 — read-only gap attribution for v2.2.2.1 short runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

OUT_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_1_GAP_FIX_V1"
)
LIFECYCLE_CSV = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/lifecycle/all_input_frame_lifecycle.csv"
)

ATTRIBUTION_CATEGORIES = (
    "candidate_missing",
    "candidate_held_by_finalization",
    "pose_failed_after_candidate",
    "safety_blocked",
    "mixed_or_unknown",
)


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_csv_rows(p: Path) -> list[dict[str, str]]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def pct(vals: list[int], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(round((len(s) - 1) * q))
    return float(s[max(0, min(len(s) - 1, i))])


def load_lifecycle_by_frame(max_frame: int) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if not LIFECYCLE_CSV.exists():
        return out
    with LIFECYCLE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fid = to_int(row.get("frame_id"))
            if fid <= max_frame:
                out[fid] = row
    return out


def keyframe_ticks(run_dir: Path, trace: dict[str, Any], max_frame: int) -> list[int]:
    ticks: list[int] = []
    for row in read_csv_rows(run_dir / "keyframe_timeline.csv"):
        if to_bool(row.get("materialized", "true")):
            cf = to_int(row.get("current_frame_id"))
            if cf <= max_frame:
                ticks.append(cf)
    if not ticks:
        ticks = sorted(
            int(e["frame_id"])
            for e in trace.get("events", []) or []
            if to_bool(e.get("final_keyframe_incremented")) and int(e["frame_id"]) <= max_frame
        )
    return sorted(set(ticks))


def index_by_frame(rows: list[dict[str, Any]], key: str = "frame_id") -> dict[int, list[dict[str, Any]]]:
    idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if key not in r:
            continue
        idx[int(r[key])].append(r)
    return idx


def index_ddc(ddc_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {to_int(r["frame_id"]): r for r in ddc_rows if r.get("frame_id")}


def classify_gap(stats: dict[str, Any]) -> tuple[str, str]:
    """Return (category, rationale)."""
    da = stats["direct_admit_candidate_count"]
    rec = stats["recovery_candidate_count"]
    held = (
        stats["density_hold_count"]
        + stats["redundancy_hold_count"]
        + stats["novelty_hold_count"]
        + stats["gap_budget_hold_count"]
    )
    pose_fail = stats["pose_failed_count"]
    safety = stats["safety_blocked_count"]
    fin = stats["finalized_keyframe_count"]
    missing_flag = stats["candidate_missing_flag"]
    held_flag = stats["candidate_held_flag"]
    pose_flag = stats["pose_failed_flag"]
    safety_flag = stats["safety_blocked_flag"]

    if missing_flag and da == 0 and rec == 0:
        return "candidate_missing", "no_direct_admit_and_no_recovery_candidate_in_gap_interior"
    if held_flag and held >= max(pose_fail, safety, 1) and fin == 0:
        return "candidate_held_by_finalization", "direct_admit_seen_but_finalization_holds_dominate"
    if pose_flag and pose_fail >= held and pose_fail >= safety and fin == 0:
        return "pose_failed_after_candidate", "pose_pnp_miniba_failures_after_admit"
    if safety_flag and safety >= max(held, pose_fail) and fin == 0:
        return "safety_blocked", "discard_or_commit_control_or_pose_path_block"
    if da == 0 and rec > 0 and fin == 0:
        return "pose_failed_after_candidate", "recovery_only_gap_without_interior_kf"
    if da > 0 and held > 0:
        return "candidate_held_by_finalization", "mixed_hold_signals"
    if da > 0 and pose_fail > 0:
        return "pose_failed_after_candidate", "mixed_pose_fail"
    if da == 0:
        return "candidate_missing", "no_direct_admit_in_gap"
    return "mixed_or_unknown", "insufficient_dominant_signal"


def analyze_run(run_name: str, run_dir: Path, max_frame: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = read_json(run_dir / "model" / "semantic_trace.json")
    if not trace:
        trace = read_json(run_dir / "semantic_trace.json")

    ticks = keyframe_ticks(run_dir, trace, max_frame)
    gaps_lengths = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]

    events = trace.get("events", []) or []
    events_by_frame = {int(e["frame_id"]): e for e in events}

    ddc = index_ddc(read_csv_rows(run_dir / "direct_density_control_v2_2_2_1_trace.csv"))
    if not ddc:
        ddc = index_ddc(
            [
                {k: str(v) for k, v in e.items()}
                for e in trace.get("direct_density_control_v2_2_2_1_events", []) or []
            ]
        )

    pnp_by = index_by_frame(trace.get("pnp_miniba_reference_events", []) or [])
    recovery_pose_by = index_by_frame(
        trace.get("recovery_pose_path_events", []) or [], key="current_frame_id"
    )
    recovery_pose_src = index_by_frame(
        trace.get("recovery_pose_path_events", []) or [], key="source_frame_id"
    )
    lifecycle_by = index_by_frame(trace.get("lifecycle_gate_events", []) or [])
    lifecycle_src = index_by_frame(
        trace.get("lifecycle_gate_events", []) or [], key="source_frame_id"
    )
    true_recovery = trace.get("true_recovery_commit_events", []) or []
    lifecycle_csv = load_lifecycle_by_frame(max_frame)

    hard_rescue_events: list[dict[str, Any]] = []
    for fid, row in sorted(ddc.items()):
        triggered = to_bool(row.get("hard_gap_rescue_triggered")) or to_bool(
            row.get("hard_gap_rescue_finalized")
        )
        decision = str(row.get("direct_finalization_decision", ""))
        if triggered or "hard_gap_rescue" in decision:
            hard_rescue_events.append(
                {
                    "frame_id": fid,
                    "decision": decision,
                    "finalized": to_bool(row.get("direct_keyframe_finalized")),
                    "reason": str(row.get("direct_finalization_reason", "")),
                }
            )

    gap_rows: list[dict[str, Any]] = []

    for i in range(1, len(ticks)):
        prev_kf = ticks[i - 1]
        next_kf = ticks[i]
        gap_len = next_kf - prev_kf
        interior = list(range(prev_kf + 1, next_kf))

        direct_admit_frames: list[int] = []
        recovery_frames: list[int] = []
        defer_frames: list[int] = []
        discard_frames: list[int] = []
        finalized_frames: list[int] = []
        pose_attempted = 0
        pnp_success = 0
        pnp_fail = 0
        miniba_success = 0
        miniba_fail = 0
        too_few_inliers = 0
        density_hold = 0
        redundancy_hold = 0
        novelty_hold = 0
        gap_budget_hold = 0
        safety_blocked = 0

        for f in interior:
            ev = events_by_frame.get(f)
            if ev:
                action = str(ev.get("action", ""))
                if action == "direct_admit":
                    direct_admit_frames.append(f)
                if action == "defer_recoverable":
                    defer_frames.append(f)
                    recovery_frames.append(f)
                if action == "discard":
                    discard_frames.append(f)
                    safety_blocked += 1
                if to_bool(ev.get("pose_init_attempted")):
                    pose_attempted += 1
                if ev.get("pose_init_success") is False:
                    detail = str(ev.get("pose_fail_detail", "") or ev.get("drop_reason", ""))
                    if "inlier" in detail.lower():
                        too_few_inliers += 1
                if to_bool(ev.get("final_keyframe_incremented")):
                    finalized_frames.append(f)

            for rp in recovery_pose_by.get(f, []) + recovery_pose_src.get(f, []):
                recovery_frames.append(f)
            for lg in lifecycle_by.get(f, []) + lifecycle_src.get(f, []):
                if to_bool(lg.get("recovery_pose_attempted")):
                    recovery_frames.append(f)

            drow = ddc.get(f)
            if drow:
                dec = str(drow.get("direct_finalization_decision", ""))
                if dec == "hold_density_high":
                    density_hold += 1
                elif dec == "hold_redundant":
                    redundancy_hold += 1
                elif "hold_gap_rescue_budget" in dec:
                    gap_budget_hold += 1
                elif dec == "hold_density_high" or "high_novelty" in dec:
                    novelty_hold += 1
                if "high_novelty" in dec and "hold" in dec:
                    novelty_hold += 1
                if to_bool(drow.get("hold_gap_rescue_budget_exhausted")):
                    gap_budget_hold += 1
                if to_bool(drow.get("direct_keyframe_finalized")):
                    finalized_frames.append(f)

            for pr in pnp_by.get(f, []):
                if to_bool(pr.get("pnp_success")):
                    pnp_success += 1
                else:
                    pnp_fail += 1
                if to_bool(pr.get("miniba_success")):
                    miniba_success += 1
                else:
                    miniba_fail += 1
                reason = str(pr.get("pose_failure_reason", ""))
                if "inlier" in reason.lower():
                    too_few_inliers += 1

            lc = lifecycle_csv.get(f, {})
            if lc and str(lc.get("state_admission", "")) == "discard":
                safety_blocked += 1

        direct_admit_frames = sorted(set(direct_admit_frames))
        recovery_frames = sorted(set(recovery_frames))
        defer_frames = sorted(set(defer_frames))
        finalized_frames = sorted(set(finalized_frames))

        pose_failed_count = 0
        for f in interior:
            ev = events_by_frame.get(f)
            if ev and ev.get("pose_init_success") is False:
                pose_failed_count += 1
            elif f in pnp_by and not to_bool(pnp_by[f][-1].get("pnp_success")):
                pose_failed_count += 1

        held_total = density_hold + redundancy_hold + novelty_hold + gap_budget_hold
        candidate_missing_flag = len(direct_admit_frames) == 0 and len(recovery_frames) == 0
        candidate_held_flag = len(direct_admit_frames) > 0 and held_total > 0
        pose_failed_flag = pose_failed_count > 0 or too_few_inliers > 0
        safety_blocked_flag = safety_blocked > 0 or len(discard_frames) > 0

        stats = {
            "direct_admit_candidate_count": len(direct_admit_frames),
            "recovery_candidate_count": len(recovery_frames),
            "defer_recoverable_count": len(defer_frames),
            "finalized_keyframe_count": len(finalized_frames),
            "pose_attempted_count": pose_attempted,
            "pnp_success_count": pnp_success,
            "pnp_fail_count": pnp_fail,
            "miniba_success_count": miniba_success,
            "miniba_fail_count": miniba_fail,
            "too_few_inliers_count": too_few_inliers,
            "density_hold_count": density_hold,
            "redundancy_hold_count": redundancy_hold,
            "novelty_hold_count": novelty_hold,
            "gap_budget_hold_count": gap_budget_hold,
            "safety_blocked_count": safety_blocked,
            "pose_failed_count": pose_failed_count,
            "candidate_missing_flag": candidate_missing_flag,
            "candidate_held_flag": candidate_held_flag,
            "pose_failed_flag": pose_failed_flag,
            "safety_blocked_flag": safety_blocked_flag,
        }
        category, rationale = classify_gap(stats)
        post500 = bool(next_kf > 500 or prev_kf > 500 or any(f > 500 for f in interior))

        gap_rows.append(
            {
                "run": run_name,
                "gap_index": i,
                "gap_start_frame": prev_kf,
                "gap_end_frame": next_kf,
                "gap_length": gap_len,
                "previous_keyframe": prev_kf,
                "next_keyframe": next_kf,
                "gap_interior_input_frame_count": len(interior),
                "post500_gap": post500,
                **stats,
                "candidate_missing_mark": candidate_missing_flag,
                "candidate_held_mark": candidate_held_flag,
                "pose_failed_mark": pose_failed_flag,
                "safety_blocked_mark": safety_blocked_flag,
                "gap_attribution": category,
                "gap_attribution_rationale": rationale,
            }
        )

    # hard rescue in gaps
    hard_in_gaps: list[dict[str, Any]] = []
    for hr in hard_rescue_events:
        fid = hr["frame_id"]
        for i in range(1, len(ticks)):
            if ticks[i - 1] < fid <= ticks[i]:
                hard_in_gaps.append({**hr, "gap_index": i, "gap_start": ticks[i - 1], "gap_end": ticks[i]})
                break

    attr_counts = Counter(r["gap_attribution"] for r in gap_rows)
    post500_gaps = [r for r in gap_rows if r["post500_gap"]]
    post500_attr = Counter(r["gap_attribution"] for r in post500_gaps)

    summary = {
        "run": run_name,
        "max_frame": max_frame,
        "keyframe_count": len(ticks),
        "gap_count": len(gap_rows),
        "gap_p50": pct(gaps_lengths, 0.5),
        "gap_p90": pct(gaps_lengths, 0.9),
        "gap_p95": pct(gaps_lengths, 0.95),
        "gap_max": max(gaps_lengths) if gaps_lengths else 0,
        "attribution_counts": dict(attr_counts),
        "attribution_fraction": {
            k: round(v / max(len(gap_rows), 1), 4) for k, v in attr_counts.items()
        },
        "post500_gap_count": len(post500_gaps),
        "post500_gap_fraction": round(len(post500_gaps) / max(len(gap_rows), 1), 4),
        "post500_attribution_counts": dict(post500_attr),
        "post500_attribution_fraction": {
            k: round(v / max(len(post500_gaps), 1), 4) for k, v in post500_attr.items()
        },
        "hard_rescue_events": hard_rescue_events,
        "hard_rescue_in_gap": hard_in_gaps,
        "hard_rescue_materialized_count": sum(1 for h in hard_rescue_events if h["finalized"]),
        "large_gaps_ge11": [
            {
                "gap_start": r["gap_start_frame"],
                "gap_end": r["gap_end_frame"],
                "gap_length": r["gap_length"],
                "attribution": r["gap_attribution"],
            }
            for r in gap_rows
            if r["gap_length"] >= 11
        ],
    }
    return gap_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def gate_recommendation(
    s800: dict[str, Any], s1000: dict[str, Any], rows800: list[dict[str, Any]], rows1000: list[dict[str, Any]]
) -> dict[str, Any]:
    def dominant(fr: dict[str, float]) -> str:
        if not fr:
            return "mixed_or_unknown"
        return max(fr.items(), key=lambda x: x[1])[0]

    def slice_stats(rows: list[dict[str, Any]], *, post500: bool | None = None, min_len: int = 0) -> dict[str, Any]:
        sel = rows
        if post500 is True:
            sel = [r for r in sel if to_bool(r.get("post500_gap"))]
        if min_len > 0:
            sel = [r for r in sel if to_int(r.get("gap_length")) >= min_len]
        c = Counter(r["gap_attribution"] for r in sel)
        n = len(sel)
        fr = {k: round(v / max(n, 1), 4) for k, v in c.items()}
        return {"count": n, "attribution_counts": dict(c), "attribution_fraction": fr, "dominant": dominant(fr)}

    d800 = dominant(s800.get("attribution_fraction", {}))
    d1000 = dominant(s1000.get("attribution_fraction", {}))
    post800 = slice_stats(rows800, post500=True)
    post1000 = slice_stats(rows1000, post500=True)
    big800 = slice_stats(rows800, min_len=11)
    big1000 = slice_stats(rows1000, min_len=11)

    # Failing-tail focus: post-500 + large gaps drive gate failure.
    post_missing_dominant = (
        post800["attribution_fraction"].get("candidate_missing", 0.0) >= 0.6
        and post1000["attribution_fraction"].get("candidate_missing", 0.0) >= 0.6
    )
    big_missing_dominant = (
        big800["dominant"] == "candidate_missing"
        and big1000["dominant"] == "candidate_missing"
    )
    held_overall = d800 == "candidate_held_by_finalization" or d1000 == "candidate_held_by_finalization"

    if post_missing_dominant and big_missing_dominant:
        recommendation = (
            "停止继续调 density controller。未达标 gap（post-500 与 gap>=11）主要由 "
            "candidate_missing 主导：gap 内几乎没有 direct_admit / recovery 候选，"
            "周期性 11-gap 来自 admission/anchor 节奏而非 finalization hold。"
            "建议转入候选版本 full diagnostic（lifecycle + admission + anchor 节奏），"
            "不要启动 v2.2.2.2 或继续 finalization 微调。"
        )
        allow_finalization_fix = False
        stop_density_controller = True
    elif held_overall:
        recommendation = (
            "整体 gap 以 candidate_held_by_finalization 为主（多为短 gap），但失败尾部 "
            "（post-500 / 大 gap）仍以 candidate_missing 为主。finalization 微调仅可能改善"
            "短 gap hold，难以修复 11-gap 周期；不建议继续 density-only 路线。"
        )
        allow_finalization_fix = True
        stop_density_controller = True
    else:
        recommendation = (
            "归因混合。建议先候选与 pose 分层诊断，再决定是否 finalization 实验。"
        )
        allow_finalization_fix = False
        stop_density_controller = False

    return {
        "overall_dominant_short800": d800,
        "overall_dominant_short1000": d1000,
        "post500_stats_short800": post800,
        "post500_stats_short1000": post1000,
        "large_gap_ge11_short800": big800,
        "large_gap_ge11_short1000": big1000,
        "post500_candidate_missing_dominant": post_missing_dominant,
        "large_gap_candidate_missing_dominant": big_missing_dominant,
        "stop_density_controller_tuning": stop_density_controller,
        "allow_finalization_layer_minimal_fix": allow_finalization_fix,
        "recommendation": recommendation,
        "constraints": {
            "no_RVQ_tau_change": True,
            "no_direct_defer_discard_semantic_change": True,
            "no_online_psnr_ssim_lpips_rpe_ape": True,
            "no_v2222_implementation": True,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    root = args.output_root

    rows800, s800 = analyze_run(
        "short800", root / "direct_density_v2_2_2_1_short800", 800
    )
    rows1000, s1000 = analyze_run(
        "short1000", root / "direct_density_v2_2_2_1_short1000", 1000
    )

    write_csv(root / "gap_source_attribution_short800.csv", rows800)
    write_csv(root / "gap_source_attribution_short1000.csv", rows1000)

    summary = {
        "audit": "GAP_SOURCE_ATTRIBUTION_AUDIT_V1",
        "output_root": str(root),
        "short800": s800,
        "short1000": s1000,
        "gate": gate_recommendation(s800, s1000, rows800, rows1000),
    }
    (root / "gap_source_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
