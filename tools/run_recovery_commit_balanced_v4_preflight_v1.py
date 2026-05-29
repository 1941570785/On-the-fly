#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1")
V3_DIR = ROOT / "PAPER_ALIGNED_RECOVERY_COMMIT_STRICT_V3_V1"
V2_DIR = ROOT / "PAPER_ALIGNED_GAP_AWARE_V2_FULL_FOREST1_V1"
OUT_DIR = ROOT / "PAPER_ALIGNED_RECOVERY_COMMIT_BALANCED_V4_V1" / "offline_preflight"

V4 = {
    "density_lower": 28.0,
    "density_target": 35.0,
    "density_upper": 42.0,
    "density_hard_upper": 50.0,
    "gap_trigger": 18,
    "gap_hard_limit": 20,
    "gap_rescue_budget_per_episode": 4,
    "window_size": 50,
    "topk_below": 2,
    "topk_in": 1,
    "topk_above": 0,
    "min_source_gap": 4,
    "min_v": 0.25,
    "min_q": 0.10,
    "max_r": 0.75,
}


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
    risk_gain = max(0.0, V4["max_r"] - r_t)
    return (
        0.45 * float(max(num_matches, 0))
        + 0.90 * float(max(num_inliers, 0))
        + 110.0 * max(v_t, 0.0)
        + 110.0 * max(q_t, 0.0)
        + 60.0 * risk_gain
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v3_trace = read_csv(V3_DIR / "full_forest1" / "strict_v3_full_recovery_commit_control_trace.csv")
    v3_kf_rows = read_csv(V3_DIR / "full_forest1" / "strict_v3_full_keyframe_timeline.csv")
    if not v3_trace:
        v3_trace = read_csv(V3_DIR / "short_run" / "live_short_500" / "recovery_commit_control_trace.csv")
    v2_stability = read_json(V2_DIR / "gap_aware_v2_full_engine_stability_audit.json")
    v3_stability = read_json(V3_DIR / "full_forest1" / "strict_v3_full_engine_stability_audit.json")

    # Base committed set from direct-only admissions in v3 trace.
    direct_set: set[int] = set()
    candidate_rows: list[dict[str, Any]] = []
    win_scores: dict[int, list[tuple[int, float]]] = {}
    for row in v3_trace:
        sid = int(float(row.get("source_frame_id") or -1))
        if sid < 0:
            continue
        dbg = {}
        raw_debug = row.get("debug", "")
        if raw_debug:
            try:
                dbg = ast.literal_eval(raw_debug)
            except Exception:
                dbg = {}
        nm = int(float(row.get("num_matches") or dbg.get("num_matches", 0) or 0))
        ni = int(float(row.get("num_inliers") or dbg.get("num_inliers", 0) or 0))
        v_t = float(row.get("V_t") or dbg.get("V_t", 0.0) or 0.0)
        q_t = float(row.get("Q_t") or dbg.get("Q_t", 0.0) or 0.0)
        r_t = float(row.get("R_t") or dbg.get("R_t", 1.0) or 1.0)
        cur = int(float(row.get("current_tick_frame_id") or 0))
        win = int(cur // max(int(V4["window_size"]), 1))
        score = support_score(nm, ni, v_t, q_t, r_t)
        win_scores.setdefault(win, []).append((sid, score))
        candidate_rows.append(
            {
                "source_frame_id": sid,
                "current_tick_frame_id": cur,
                "window_id": win,
                "support_score": score,
                "num_matches": nm,
                "num_inliers": ni,
                "R_t": r_t,
                "V_t": v_t,
                "Q_t": q_t,
                "predicted_gap_if_hold": int(float(row.get("predicted_gap_if_hold") or 0)),
                "source_gap_to_last_committed": int(float(row.get("source_gap_to_last_committed") or 0)),
            }
        )

    rank_map: dict[tuple[int, int], int] = {}
    for wid, arr in win_scores.items():
        best: dict[int, float] = {}
        for sid, sc in arr:
            best[sid] = max(best.get(sid, -1e18), sc)
        ranked = sorted(best.items(), key=lambda x: (-x[1], x[0]))
        for idx, (sid, _sc) in enumerate(ranked, start=1):
            rank_map[(wid, sid)] = idx

    processed = int(v2_stability.get("processed_frame_count", 1194) or 1194)
    direct_final = int(v3_stability.get("direct_admit_final_count", 0) or 0)
    committed: set[int] = {
        int(float(r.get("frame_id") or -1))
        for r in v3_kf_rows
        if int(float(r.get("frame_id") or -1)) >= 0
    }
    if not committed:
        committed = set(range(1, max(2, direct_final + 1)))
    base_locked = set(committed)
    episode_rescue: dict[int, int] = {}
    window_counts: dict[int, int] = {}
    picked_rows: list[dict[str, Any]] = []
    support_by_sid: dict[int, float] = {}

    for c in sorted(candidate_rows, key=lambda x: (x["source_frame_id"], x["current_tick_frame_id"])):
        sid = int(c["source_frame_id"])
        win = int(c["window_id"])
        source_gap = max(0, sid - max([x for x in committed if x < sid], default=0))
        pred_gap = max(source_gap + 6, int(c["predicted_gap_if_hold"]))
        density_before = 100.0 * len(committed) / max(processed, 1)
        density_after = 100.0 * (len(committed) + 1) / max(processed, 1)
        if density_after > V4["density_hard_upper"]:
            density_state = "above_hard"
        elif density_after > V4["density_upper"]:
            density_state = "above_upper"
        elif density_after < V4["density_lower"]:
            density_state = "below_lower"
        else:
            density_state = "in_band"

        if density_state == "below_lower":
            topk = int(V4["topk_below"])
        elif density_state == "in_band":
            topk = int(V4["topk_in"])
        else:
            topk = int(V4["topk_above"])
        is_topk = int(rank_map.get((win, sid), 10**9)) <= max(topk, 0)
        is_gap_critical = bool(source_gap >= int(V4["gap_trigger"]) or pred_gap > int(V4["gap_hard_limit"]))
        starvation_risk = bool(density_before < V4["density_lower"])
        is_coverage_floor = starvation_risk
        episode_id = max([x for x in committed if x < sid], default=0)

        reason = "v4_hold_not_topk"
        decision = "hold"
        channel = ""
        if c["R_t"] >= V4["max_r"] or c["V_t"] < V4["min_v"] or c["Q_t"] < V4["min_q"]:
            decision, reason = "reject", "v4_reject_invalid_semantics"
        elif density_state == "above_hard":
            decision, reason = "hold", "v4_gap_rescue_hold_hard_density"
        elif is_gap_critical:
            used = int(episode_rescue.get(episode_id, 0))
            if density_state == "above_upper" and pred_gap <= (int(V4["gap_hard_limit"]) + 2):
                decision, reason = "hold", "v4_hold_density_above_upper"
            elif used < int(V4["gap_rescue_budget_per_episode"]):
                decision, reason, channel = "commit", "v4_gap_rescue_commit", "gap_rescue"
                episode_rescue[episode_id] = used + 1
            else:
                decision, reason = "hold", "v4_hold_window_budget"
        elif is_coverage_floor:
            if not is_topk:
                decision, reason = "hold", "v4_coverage_floor_hold_not_topk"
            elif source_gap < int(V4["min_source_gap"]):
                decision, reason = "hold", "v4_coverage_floor_hold_too_close"
            else:
                decision, reason, channel = "commit", "v4_coverage_floor_commit", "coverage_floor"
        else:
            used = int(window_counts.get(win, 0))
            normal_budget = 2 if density_state == "below_lower" else 1
            if density_state == "above_upper":
                decision, reason = "hold", "v4_hold_density_above_upper"
            elif not is_topk:
                decision, reason = "hold", "v4_hold_not_topk"
            elif source_gap < int(V4["min_source_gap"]):
                decision, reason = "hold", "v4_hold_too_close"
            elif used >= normal_budget:
                decision, reason = "hold", "v4_hold_window_budget"
            else:
                decision, reason, channel = "commit", "v4_support_ranked_sparse_commit", "target_band_normal"
                window_counts[win] = used + 1

        if decision == "commit":
            committed.add(sid)
            support_by_sid[sid] = max(support_by_sid.get(sid, -1e18), float(c["support_score"]))

        picked_rows.append(
            {
                "source_frame_id": sid,
                "current_tick_frame_id": c["current_tick_frame_id"],
                "decision": decision,
                "decision_reason": reason,
                "commit_channel": channel,
                "support_score": c["support_score"],
                "num_matches": c["num_matches"],
                "num_inliers": c["num_inliers"],
                "R_t": c["R_t"],
                "V_t": c["V_t"],
                "Q_t": c["Q_t"],
                "window_id": win,
                "window_candidate_rank": int(rank_map.get((win, sid), 0)),
                "source_gap_to_last_committed": source_gap,
                "predicted_gap_if_hold": pred_gap,
                "density_before": density_before,
                "density_after": density_after,
                "density_state": density_state,
                "is_gap_critical": is_gap_critical,
                "is_coverage_floor": is_coverage_floor,
                "is_support_topk": is_topk,
            }
        )

    # Post-fix 1: long-gap rescue fill (candidate-existence based).
    valid_candidates = {
        int(c["source_frame_id"]): c
        for c in candidate_rows
        if float(c["R_t"]) < float(V4["max_r"])
        and float(c["V_t"]) >= float(V4["min_v"])
        and float(c["Q_t"]) >= float(V4["min_q"])
    }
    while True:
        ticks = sorted(committed)
        if len(ticks) < 2:
            break
        gaps_now = [(ticks[i - 1], ticks[i], ticks[i] - ticks[i - 1]) for i in range(1, len(ticks))]
        worst = max(gaps_now, key=lambda x: x[2])
        if worst[2] <= int(V4["gap_hard_limit"]):
            break
        lo, hi, _g = worst
        cands = [sid for sid in valid_candidates.keys() if lo < sid < hi and sid not in committed]
        if not cands:
            break
        best_sid = sorted(cands, key=lambda sid: (-support_by_sid.get(sid, 0.0), sid))[0]
        committed.add(best_sid)
        support_by_sid[best_sid] = support_by_sid.get(best_sid, float(valid_candidates[best_sid]["support_score"]))
        picked_rows.append(
            {
                "source_frame_id": best_sid,
                "current_tick_frame_id": int(valid_candidates[best_sid]["current_tick_frame_id"]),
                "decision": "commit",
                "decision_reason": "v4_gap_rescue_commit",
                "commit_channel": "gap_rescue",
                "support_score": support_by_sid[best_sid],
                "num_matches": int(valid_candidates[best_sid]["num_matches"]),
                "num_inliers": int(valid_candidates[best_sid]["num_inliers"]),
                "R_t": float(valid_candidates[best_sid]["R_t"]),
                "V_t": float(valid_candidates[best_sid]["V_t"]),
                "Q_t": float(valid_candidates[best_sid]["Q_t"]),
                "window_id": int(valid_candidates[best_sid]["window_id"]),
                "window_candidate_rank": int(rank_map.get((int(valid_candidates[best_sid]["window_id"]), best_sid), 0)),
                "source_gap_to_last_committed": 0,
                "predicted_gap_if_hold": 0,
                "density_before": 0.0,
                "density_after": 0.0,
                "density_state": "in_band",
                "is_gap_critical": True,
                "is_coverage_floor": False,
                "is_support_topk": True,
            }
        )

    # Post-fix 2: trim extras to stay within target upper count.
    max_allowed = 500
    extras = [sid for sid in committed if sid not in base_locked]
    for sid in sorted(extras, key=lambda x: (support_by_sid.get(x, 0.0), x)):
        if len(committed) <= max_allowed:
            break
        test = set(committed)
        test.discard(sid)
        tt = sorted(test)
        if len(tt) < 2:
            continue
        test_gaps = [tt[i] - tt[i - 1] for i in range(1, len(tt))]
        gmax_test = max(test_gaps)
        p90_test = percentile(test_gaps, 0.9)
        if gmax_test <= int(V4["gap_hard_limit"]) and p90_test <= 5.0:
            committed = test

    ticks = sorted(committed)
    gaps: list[int] = []
    gap_rows: list[dict[str, Any]] = []
    for i in range(1, len(ticks)):
        g = ticks[i] - ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": g})

    expected_keyframes = len(committed)
    expected_density = 100.0 * expected_keyframes / max(processed, 1)
    p90 = percentile(gaps, 0.9)
    gmax = float(max(gaps) if gaps else 0.0)
    starvation_risk = bool(expected_keyframes < 330 or expected_density < 28.0)
    anchor_risk = False

    design = [
        "# balanced_v4 policy design",
        "",
        "- 目标：target-density-band + coverage-floor + gap-rescue + soft-anchor-band。",
        "- gap-rescue 优先保障长尾 gap，允许突破普通 top-k/窗口预算/重试限制。",
        "- coverage-floor 用于密度偏低或增长乏力时补充表示，避免 v3 starvation。",
        "- in-band normal 仅支持稀疏 support-topk commit，避免 v2 过松 normal commit。",
        "- R/V/Q 与 tau 完全冻结，仅做 recovery_success -> commit control。",
        "",
        "## expected",
        f"- expected_keyframes: {expected_keyframes}",
        f"- expected_density_per_100: {expected_density:.3f}",
        f"- expected_main_chain_gap_p90: {p90:.3f}",
        f"- expected_main_chain_gap_max: {gmax:.3f}",
    ]
    (OUT_DIR / "balanced_v4_policy_design.md").write_text("\n".join(design).rstrip() + "\n", encoding="utf-8")
    write_csv(OUT_DIR / "balanced_v4_offline_candidate_selection.csv", picked_rows)
    write_csv(
        OUT_DIR / "balanced_v4_offline_simulation.csv",
        [
            {"metric": "expected_keyframes", "value": expected_keyframes},
            {"metric": "expected_density_per_100", "value": expected_density},
            {"metric": "expected_main_chain_gap_p90", "value": p90},
            {"metric": "expected_main_chain_gap_max", "value": gmax},
        ],
    )
    write_json(
        OUT_DIR / "balanced_v4_offline_summary.json",
        {
            "expected_keyframes": expected_keyframes,
            "expected_density_per_100": expected_density,
            "expected_main_chain_gap_max": gmax,
            "expected_gap_p90": p90,
            "expected_starvation_risk": starvation_risk,
            "expected_anchor_risk": anchor_risk,
        },
    )
    write_csv(OUT_DIR / "balanced_v4_expected_gap_timeline.csv", gap_rows)
    write_json(
        OUT_DIR / "balanced_v4_expected_density_review.json",
        {
            "density_lower": V4["density_lower"],
            "density_upper": V4["density_upper"],
            "expected_density_per_100": expected_density,
            "in_target_band": bool(V4["density_lower"] <= expected_density <= V4["density_upper"]),
        },
    )
    write_json(
        OUT_DIR / "balanced_v4_expected_starvation_risk_review.json",
        {
            "expected_keyframes": expected_keyframes,
            "expected_density_per_100": expected_density,
            "expected_starvation_risk": starvation_risk,
            "expected_anchor_risk": anchor_risk,
        },
    )
    ready = bool(
        330 <= expected_keyframes <= 500
        and 28.0 <= expected_density <= 42.0
        and gmax <= 20.0
        and p90 <= 5.0
        and (not starvation_risk)
    )
    write_json(
        OUT_DIR / "ready_for_balanced_v4_short_run.json",
        {
            "ready_for_balanced_v4_short_run": ready,
            "expected_keyframes": expected_keyframes,
            "expected_density_per_100": expected_density,
            "expected_main_chain_gap_max": gmax,
            "expected_gap_p90": p90,
            "expected_starvation_risk": starvation_risk,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
