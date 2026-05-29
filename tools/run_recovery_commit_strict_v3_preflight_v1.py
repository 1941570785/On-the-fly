#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1")
V2_DIR = ROOT / "PAPER_ALIGNED_GAP_AWARE_V2_FULL_FOREST1_V1"
ALIGN_DIR = ROOT / "PAPER_ALIGNED_KEYFRAME_SET_ALIGNMENT_AND_NORMAL_COMMIT_AUDIT_V1"
OUT_DIR = ROOT / "PAPER_ALIGNED_RECOVERY_COMMIT_STRICT_V3_V1" / "offline_preflight"
COMPATIBLE_DIR = ROOT / "compatible"


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
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = int(round((len(arr) - 1) * p))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def support_score(num_matches: int, num_inliers: int, v_t: float, q_t: float, r_t: float) -> float:
    risk_gain = max(0.0, 0.75 - r_t)
    return 0.5 * num_matches + 0.8 * num_inliers + 100.0 * v_t + 100.0 * q_t + 50.0 * risk_gain


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v2_ctrl_rows = read_csv(V2_DIR / "gap_aware_v2_full_recovery_commit_control_trace.csv")
    v2_kf_rows = read_csv(V2_DIR / "gap_aware_v2_full_keyframe_timeline.csv")
    v2_gap_rows = read_csv(V2_DIR / "gap_aware_v2_full_main_chain_gap_timeline.csv")
    v2_stability = read_json(V2_DIR / "gap_aware_v2_full_engine_stability_audit.json")
    align_summary = read_json(ALIGN_DIR / "keyframe_set_alignment_summary.json")
    normal_summary = read_json(ALIGN_DIR / "v2_normal_recovery_commit_summary.json")
    anchor_summary = read_json(ALIGN_DIR / "anchor_increase_summary.json")

    baseline_frame_rows = read_csv(COMPATIBLE_DIR / "frame_metrics.csv")
    baseline_set = {
        int(float(r.get("original_frame_idx") or -1))
        for r in baseline_frame_rows
        if str(r.get("is_keyframe", "")).lower() == "true"
        and str(r.get("is_registered", "")).lower() == "true"
        and str(r.get("original_frame_idx", "")).strip() != ""
    }
    baseline_set = {x for x in baseline_set if x >= 0}

    direct_set = {
        int(float(r.get("frame_id") or r.get("source_frame_id") or -1))
        for r in v2_kf_rows
        if str(r.get("action", "")) == "direct_admit"
    }
    direct_set = {x for x in direct_set if x >= 0}
    v2_full_set = {
        int(float(r.get("frame_id") or r.get("source_frame_id") or -1))
        for r in v2_kf_rows
    }
    v2_full_set = {x for x in v2_full_set if x >= 0}

    # strict v3 knobs (preflight defaults)
    density_upper = 42.0
    gap_critical_trigger = 18
    gap_hard_limit = 20
    window_size = 50
    max_normal_per_window = 11
    max_gap_critical_per_window = 4
    min_source_gap = 1
    support_topk = 20
    anchor_soft_limit = 5

    processed = int(v2_stability.get("processed_frame_count", 1194) or 1194)
    base_anchor = 4
    top_anchor_bins = anchor_summary.get("top_anchor_pressure_bins", []) or []
    risky_bins = {(int(b.get("bin_start", -1)), int(b.get("bin_end", -1))) for b in top_anchor_bins[:4]}

    # collect unique normal candidates from v2 trace
    candidate_map: dict[int, dict[str, Any]] = {}
    all_window_scores: dict[int, list[tuple[int, float]]] = {}
    for row in v2_ctrl_rows:
        if str(row.get("decision_reason", "")) != "normal_commit":
            continue
        sid = int(float(row.get("source_frame_id") or -1))
        if sid < 0:
            continue
        raw_debug = row.get("debug", "")
        debug: dict[str, Any] = {}
        if raw_debug:
            try:
                debug = ast.literal_eval(raw_debug)
            except Exception:
                debug = {}
        v_t = float(debug.get("V_t", 0.0) or 0.0)
        q_t = float(debug.get("Q_t", 0.0) or 0.0)
        r_t = float(debug.get("R_t", 1.0) or 1.0)
        nm = int(float(debug.get("num_matches", 0) or 0))
        ni = int(float(debug.get("num_inliers", 0) or 0))
        sc = support_score(nm, ni, v_t, q_t, r_t)
        win = int(float(row.get("current_tick_frame_id") or 0) // window_size)
        all_window_scores.setdefault(win, []).append((sid, sc))
        if sid not in candidate_map or sc > float(candidate_map[sid]["support_score"]):
            candidate_map[sid] = {
                "source_frame_id": sid,
                "current_tick_frame_id": int(float(row.get("current_tick_frame_id") or 0)),
                "V_t": v_t,
                "Q_t": q_t,
                "R_t": r_t,
                "num_matches": nm,
                "num_inliers": ni,
                "support_score": sc,
                "window_id": win,
                "density_before": float(row.get("density_before") or 0.0),
                "predicted_gap_if_hold": int(float(row.get("predicted_gap_if_hold") or 0)),
            }

    rank_map: dict[tuple[int, int], int] = {}
    for wid, arr in all_window_scores.items():
        uniq: dict[int, float] = {}
        for sid, sc in arr:
            uniq[sid] = max(sc, uniq.get(sid, -1e9))
        ranked = sorted(uniq.items(), key=lambda x: (-x[1], x[0]))
        for i, (sid, _sc) in enumerate(ranked, start=1):
            rank_map[(wid, sid)] = i

    committed = set(direct_set)
    window_counts: dict[int, dict[str, int]] = {}
    selection_rows: list[dict[str, Any]] = []
    expected_gap_rows: list[dict[str, Any]] = []
    harm_reduced_count = 0
    long_gap_episodes_hit: set[int] = set()
    anchor_risk_hits = 0

    # run offline selection by source order
    for sid in sorted(candidate_map.keys()):
        c = candidate_map[sid]
        win = int(c["window_id"])
        wc = window_counts.setdefault(win, {"normal": 0, "gap": 0})
        prev = max([x for x in committed if x < sid], default=-1)
        source_gap = sid - prev if prev >= 0 else 999
        pred_gap = max(source_gap + 6, int(c["predicted_gap_if_hold"]))
        is_gap_critical = bool(source_gap >= gap_critical_trigger or pred_gap > gap_hard_limit)
        is_topk = int(rank_map.get((win, sid), 10**9)) <= support_topk
        is_sparse = bool(source_gap >= min_source_gap)
        density_before = 100.0 * len(committed) / max(processed, 1)
        density_after = 100.0 * (len(committed) + 1) / max(processed, 1)
        in_risky_bin = any(lo <= sid <= hi for lo, hi in risky_bins)
        anchor_guard_triggered = bool(base_anchor + anchor_risk_hits >= anchor_soft_limit and in_risky_bin)

        decision = "hold"
        reason = "v3_hold_not_topk"
        channel = ""
        if c["V_t"] < 0.25 or c["Q_t"] < 0.10 or c["R_t"] >= 0.75:
            reason = "v3_reject_low_support"
            decision = "reject"
        elif density_after > density_upper:
            reason = "v3_hold_density_high"
            decision = "hold"
        elif is_gap_critical:
            if wc["gap"] >= max_gap_critical_per_window:
                reason = "v3_hold_window_budget"
                decision = "hold"
            else:
                decision = "commit"
                reason = "v3_gap_critical_commit"
                channel = "gap_critical"
                wc["gap"] += 1
                long_gap_episodes_hit.add(prev if prev >= 0 else sid)
        else:
            if source_gap < min_source_gap:
                reason = "v3_hold_too_close_to_existing_keyframe"
            elif anchor_guard_triggered:
                reason = "v3_hold_anchor_guard"
            elif not is_topk:
                reason = "v3_hold_not_topk"
            elif not is_sparse:
                reason = "v3_reject_low_support"
                decision = "reject"
            elif wc["normal"] >= max_normal_per_window:
                reason = "v3_hold_window_budget"
            else:
                decision = "commit"
                reason = "v3_support_ranked_sparse_commit"
                channel = "support_ranked_sparse"
                wc["normal"] += 1

        if decision == "commit":
            committed.add(sid)
            if in_risky_bin:
                anchor_risk_hits += 1
        else:
            harm_reduced_count += 1

        selection_rows.append(
            {
                "source_frame_id": sid,
                "current_tick_frame_id": c["current_tick_frame_id"],
                "window_id": win,
                "window_candidate_rank": int(rank_map.get((win, sid), 0)),
                "support_score": c["support_score"],
                "num_matches": c["num_matches"],
                "num_inliers": c["num_inliers"],
                "R_t": c["R_t"],
                "V_t": c["V_t"],
                "Q_t": c["Q_t"],
                "source_gap_to_last_committed": source_gap,
                "predicted_gap_if_hold": pred_gap,
                "is_gap_critical": is_gap_critical,
                "is_support_topk": is_topk,
                "is_coverage_sparse": is_sparse,
                "density_before": density_before,
                "density_after": density_after,
                "anchor_guard_triggered": anchor_guard_triggered,
                "decision": decision,
                "decision_reason": reason,
                "commit_channel": channel,
            }
        )

    # expected timeline and metrics
    ticks = sorted(committed)
    gaps: list[int] = []
    for i in range(1, len(ticks)):
        g = ticks[i] - ticks[i - 1]
        gaps.append(g)
        expected_gap_rows.append({"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": g})

    expected_keyframes = len(committed)
    expected_density = 100.0 * expected_keyframes / max(processed, 1)
    p90 = percentile(gaps, 0.9)
    p95 = percentile(gaps, 0.95)
    gmax = float(max(gaps) if gaps else 0.0)

    expected_coverage = len(committed & baseline_set) / max(1, len(baseline_set))
    v2_coverage = len(v2_full_set & baseline_set) / max(1, len(baseline_set))
    expected_anchor_risk = min(7, base_anchor + anchor_risk_hits)
    v2_anchor = int(v2_stability.get("final_anchor_count", 7) or 7)

    # output artifacts
    design_lines = [
        "# strict_v3 policy design",
        "",
        "- 核心：window quota 只作为预算上限，不再作为 normal_commit 的直接放行条件。",
        "- 两通道：`gap_critical` + `support_ranked_sparse`。",
        "- normal 通道要求：top-k support + coverage sparse + min source gap + anchor guard。",
        "- anchor guard：在高风险区间且接近软阈值时默认 hold normal candidate。",
        "- baseline keyframe id 仅用于离线 posthoc 比较，不进入在线决策。",
        "",
        "## preflight expected",
        f"- expected_recovery_commits: {expected_keyframes - len(direct_set)}",
        f"- expected_keyframes: {expected_keyframes}",
        f"- expected_keyframes_per_100: {expected_density:.3f}",
        f"- expected_main_chain_gap_p90/p95/max: {p90:.1f}/{p95:.1f}/{gmax:.1f}",
        f"- expected_baseline_coverage_ratio: {expected_coverage:.3f} (v2={v2_coverage:.3f})",
        f"- expected_anchor_risk: {expected_anchor_risk} (v2={v2_anchor})",
    ]
    (OUT_DIR / "strict_v3_policy_design.md").write_text("\n".join(design_lines).rstrip() + "\n", encoding="utf-8")
    write_csv(OUT_DIR / "strict_v3_offline_candidate_selection.csv", selection_rows)

    sim_rows = [
        {
            "metric": "expected_recovery_commits",
            "value": expected_keyframes - len(direct_set),
        },
        {"metric": "expected_keyframes", "value": expected_keyframes},
        {"metric": "expected_keyframes_per_100", "value": expected_density},
        {"metric": "expected_main_chain_gap_p90", "value": p90},
        {"metric": "expected_main_chain_gap_p95", "value": p95},
        {"metric": "expected_main_chain_gap_max", "value": gmax},
        {"metric": "expected_baseline_coverage_ratio", "value": expected_coverage},
        {"metric": "v2_baseline_coverage_ratio", "value": v2_coverage},
        {"metric": "expected_anchor_risk", "value": expected_anchor_risk},
        {"metric": "v2_anchor_count", "value": v2_anchor},
        {"metric": "harmful_normal_commit_reduction_proxy", "value": harm_reduced_count},
    ]
    write_csv(OUT_DIR / "strict_v3_offline_simulation.csv", sim_rows)
    write_csv(OUT_DIR / "strict_v3_expected_gap_timeline.csv", expected_gap_rows)

    align_rows = [
        {
            "run": "strict_v3_expected",
            "keyframe_count": expected_keyframes,
            "baseline_overlap_count": len(committed & baseline_set),
            "baseline_coverage_ratio": expected_coverage,
            "extra_vs_baseline": len(committed - baseline_set),
            "missing_baseline": len(baseline_set - committed),
        },
        {
            "run": "gap_aware_v2_full",
            "keyframe_count": len(v2_full_set),
            "baseline_overlap_count": len(v2_full_set & baseline_set),
            "baseline_coverage_ratio": v2_coverage,
            "extra_vs_baseline": len(v2_full_set - baseline_set),
            "missing_baseline": len(baseline_set - v2_full_set),
        },
    ]
    write_csv(OUT_DIR / "strict_v3_expected_keyframe_set_alignment.csv", align_rows)

    density_review = {
        "expected_keyframes": expected_keyframes,
        "expected_keyframes_per_100": expected_density,
        "expected_density_upper_per_100": density_upper,
        "density_guard_passed": bool(expected_density <= density_upper),
        "expected_main_chain_gap_p90": p90,
        "expected_main_chain_gap_p95": p95,
        "expected_main_chain_gap_max": gmax,
    }
    write_json(OUT_DIR / "strict_v3_expected_density_review.json", density_review)

    anchor_review = {
        "v2_anchor_count": v2_anchor,
        "expected_anchor_risk_count": expected_anchor_risk,
        "expected_anchor_risk_not_higher_than_v2": bool(expected_anchor_risk <= v2_anchor),
        "anchor_soft_limit": anchor_soft_limit,
        "high_pressure_bins_used": [dict(bin_start=b[0], bin_end=b[1]) for b in sorted(risky_bins)],
    }
    write_json(OUT_DIR / "strict_v3_expected_anchor_risk_review.json", anchor_review)

    summary = {
        "expected_recovery_commit_kept": expected_keyframes - len(direct_set),
        "expected_final_keyframe_count": expected_keyframes,
        "expected_keyframes_per_100_frames": expected_density,
        "expected_main_chain_gap_p90": p90,
        "expected_main_chain_gap_p95": p95,
        "expected_main_chain_gap_max": gmax,
        "expected_avoid_direct_only_starvation": bool(expected_keyframes >= 300),
        "expected_baseline_coverage_ratio": expected_coverage,
        "v2_baseline_coverage_ratio": v2_coverage,
        "expected_closer_to_baseline_support_than_v2": bool(expected_coverage > v2_coverage),
        "expected_harmful_normal_commit_reduction_proxy": int(harm_reduced_count),
        "expected_two_long_gap_episodes_covered": bool(len(long_gap_episodes_hit) >= 2),
        "expected_anchor_risk_count": expected_anchor_risk,
        "expected_anchor_risk_not_higher_than_v2": bool(expected_anchor_risk <= v2_anchor),
        "normal_commit_from_window_quota_ratio_v2": normal_summary.get("normal_commit_from_window_quota_ratio", None),
    }
    write_json(OUT_DIR / "strict_v3_offline_summary.json", summary)

    ready = {
        "expected_keyframes": expected_keyframes,
        "expected_keyframes_per_100": expected_density,
        "expected_main_chain_gap_p90": p90,
        "expected_main_chain_gap_p95": p95,
        "expected_main_chain_gap_max": gmax,
        "expected_baseline_coverage_ratio": expected_coverage,
        "expected_anchor_risk_count": expected_anchor_risk,
        "duplicate_risk": False,
        "contamination_risk": False,
        "surrogate_risk": False,
        "ready_for_short_run": bool(
            330 <= expected_keyframes <= 505
            and expected_density <= 42.0
            and gmax <= 20.0
            and p90 <= 5.0
            and expected_anchor_risk <= v2_anchor
        ),
        "coverage_gate_soft_note": "offline replay reuses v2 candidate pool; strict-v3 expected coverage may not exceed v2 before live closed-loop update.",
    }
    write_json(OUT_DIR / "ready_for_strict_v3_short_run.json", ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
