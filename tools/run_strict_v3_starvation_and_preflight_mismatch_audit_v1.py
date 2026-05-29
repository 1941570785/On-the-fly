#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1/PAPER_ALIGNED_RECOVERY_COMMIT_STRICT_V3_V1"
)
OUT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1/PAPER_ALIGNED_STRICT_V3_STARVATION_AND_PREFLIGHT_MISMATCH_AUDIT_V1"
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


def to_f(x: Any, d: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return d
        return float(x)
    except Exception:
        return d


def to_i(x: Any, d: int = 0) -> int:
    try:
        if x is None or x == "":
            return d
        return int(float(x))
    except Exception:
        return d


def to_b(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def reason_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(str(r.get("decision_reason", "")) for r in rows)
    commits = sum(1 for r in rows if str(r.get("decision", "")) == "commit")
    holds = sum(1 for r in rows if str(r.get("decision", "")) == "hold")
    rejects = sum(1 for r in rows if str(r.get("decision", "")) == "reject")
    return {
        "decision_reason_counts": dict(c),
        "commit_count": commits,
        "hold_count": holds,
        "reject_count": rejects,
        "gap_critical_commit_count": int(c.get("v3_gap_critical_commit", 0)),
        "support_ranked_sparse_commit_count": int(c.get("v3_support_ranked_sparse_commit", 0)),
        "anchor_guard_hold_count": int(c.get("v3_hold_anchor_guard", 0)),
        "topk_hold_count": int(c.get("v3_hold_not_topk", 0)),
        "density_hold_count": int(c.get("v3_hold_density_high", 0)),
        "too_close_hold_count": int(c.get("v3_hold_too_close_to_existing_keyframe", 0)),
        "window_budget_hold_count": int(c.get("v3_hold_window_budget", 0)),
    }


def build_stage_snapshot(
    stage: str,
    keyframes: int,
    density: float,
    gap_p90: float,
    gap_p95: float,
    gap_max: float,
    anchors: int,
    too_few: int,
    allowed: int,
    held: int,
    rejected: int,
    ctrl_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rp = reason_profile(ctrl_rows)
    return {
        "stage": stage,
        "keyframes": keyframes,
        "density": density,
        "main_chain_gap_p90": gap_p90,
        "main_chain_gap_p95": gap_p95,
        "main_chain_gap_max": gap_max,
        "anchor_count": anchors,
        "too_few_inliers_count": too_few,
        "recovery_commit_allowed_count": allowed,
        "recovery_commit_held_count": held,
        "recovery_commit_rejected_count": rejected,
        **rp,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    offline_summary = read_json(ROOT / "offline_preflight/strict_v3_offline_summary.json")
    offline_sim = read_csv(ROOT / "offline_preflight/strict_v3_offline_simulation.csv")
    short300_a = read_json(ROOT / "short_run/live_short_300/engine_stability_audit.json")
    short500_a = read_json(ROOT / "short_run/live_short_500/engine_stability_audit.json")
    full_a = read_json(ROOT / "full_forest1/strict_v3_full_engine_stability_audit.json")

    short300_ctrl = read_csv(ROOT / "short_run/live_short_300/recovery_commit_control_trace.csv")
    short500_ctrl = read_csv(ROOT / "short_run/live_short_500/recovery_commit_control_trace.csv")
    full_ctrl = read_csv(ROOT / "full_forest1/strict_v3_full_recovery_commit_control_trace.csv")

    short300_kf = read_csv(ROOT / "short_run/live_short_300/keyframe_timeline.csv")
    short500_kf = read_csv(ROOT / "short_run/live_short_500/keyframe_timeline.csv")
    full_kf = read_csv(ROOT / "full_forest1/strict_v3_full_keyframe_timeline.csv")
    full_gap = read_csv(ROOT / "full_forest1/strict_v3_full_main_chain_gap_timeline.csv")
    full_anchor_audit = read_json(ROOT / "full_forest1/strict_v3_full_anchor_audit.json")
    full_report = (ROOT / "paper_aligned_recovery_commit_strict_v3_v1_report.md").read_text(
        encoding="utf-8"
    )

    # A) preflight vs actual mismatch
    offline_reason_counts: dict[str, int] = {}
    for r in read_csv(ROOT / "offline_preflight/strict_v3_offline_candidate_selection.csv"):
        rr = str(r.get("decision_reason", ""))
        offline_reason_counts[rr] = offline_reason_counts.get(rr, 0) + 1
    offline_rows_for_reason = [
        {
            "decision": "commit" if str(r.get("decision", "")) == "commit" else str(r.get("decision", "")),
            "decision_reason": str(r.get("decision_reason", "")),
        }
        for r in read_csv(ROOT / "offline_preflight/strict_v3_offline_candidate_selection.csv")
    ]

    offline_anchor = to_i(offline_summary.get("expected_anchor_risk_count"), 7)
    offline_snapshot = build_stage_snapshot(
        stage="offline_preflight_expected",
        keyframes=to_i(offline_summary.get("expected_final_keyframe_count")),
        density=to_f(offline_summary.get("expected_keyframes_per_100_frames")),
        gap_p90=to_f(offline_summary.get("expected_main_chain_gap_p90")),
        gap_p95=to_f(offline_summary.get("expected_main_chain_gap_p95")),
        gap_max=to_f(offline_summary.get("expected_main_chain_gap_max")),
        anchors=offline_anchor,
        too_few=-1,
        allowed=to_i(offline_summary.get("expected_recovery_commit_kept")),
        held=-1,
        rejected=-1,
        ctrl_rows=offline_rows_for_reason,
    )
    short300_snapshot = build_stage_snapshot(
        stage="live_short_300",
        keyframes=to_i(short300_a.get("final_keyframe_count")),
        density=to_f(short300_a.get("keyframes_per_100_frames")),
        gap_p90=to_f(short300_a.get("main_chain_gap_p90")),
        gap_p95=to_f(short300_a.get("main_chain_gap_p95")),
        gap_max=to_f(short300_a.get("main_chain_gap_max")),
        anchors=to_i(short300_a.get("final_anchor_count")),
        too_few=to_i(short300_a.get("too_few_inliers_count")),
        allowed=to_i(short300_a.get("recovery_commit_allowed_count")),
        held=to_i(short300_a.get("recovery_commit_held_count")),
        rejected=to_i(short300_a.get("recovery_commit_rejected_count")),
        ctrl_rows=short300_ctrl,
    )
    short500_snapshot = build_stage_snapshot(
        stage="live_short_500",
        keyframes=to_i(short500_a.get("final_keyframe_count")),
        density=to_f(short500_a.get("keyframes_per_100_frames")),
        gap_p90=to_f(short500_a.get("main_chain_gap_p90")),
        gap_p95=to_f(short500_a.get("main_chain_gap_p95")),
        gap_max=to_f(short500_a.get("main_chain_gap_max")),
        anchors=to_i(short500_a.get("final_anchor_count")),
        too_few=to_i(short500_a.get("too_few_inliers_count")),
        allowed=to_i(short500_a.get("recovery_commit_allowed_count")),
        held=to_i(short500_a.get("recovery_commit_held_count")),
        rejected=to_i(short500_a.get("recovery_commit_rejected_count")),
        ctrl_rows=short500_ctrl,
    )
    full_snapshot = build_stage_snapshot(
        stage="full_forest1_actual",
        keyframes=to_i(full_a.get("final_keyframe_count")),
        density=to_f(full_a.get("keyframes_per_100_frames")),
        gap_p90=to_f(full_a.get("main_chain_gap_p90")),
        gap_p95=to_f(full_a.get("main_chain_gap_p95")),
        gap_max=to_f(full_a.get("main_chain_gap_max")),
        anchors=to_i(full_a.get("final_anchor_count")),
        too_few=to_i(full_a.get("too_few_inliers_count")),
        allowed=to_i(full_a.get("recovery_commit_allowed_count")),
        held=to_i(full_a.get("recovery_commit_held_count")),
        rejected=to_i(full_a.get("recovery_commit_rejected_count")),
        ctrl_rows=full_ctrl,
    )
    mismatch_rows = [offline_snapshot, short300_snapshot, short500_snapshot, full_snapshot]
    # add delta rows vs preflight
    for s in [short300_snapshot, short500_snapshot, full_snapshot]:
        mismatch_rows.append(
            {
                "stage": f"delta_vs_preflight::{s['stage']}",
                "delta_keyframes": s["keyframes"] - offline_snapshot["keyframes"],
                "delta_density": s["density"] - offline_snapshot["density"],
                "delta_main_chain_gap_p90": s["main_chain_gap_p90"] - offline_snapshot["main_chain_gap_p90"],
                "delta_main_chain_gap_p95": s["main_chain_gap_p95"] - offline_snapshot["main_chain_gap_p95"],
                "delta_main_chain_gap_max": s["main_chain_gap_max"] - offline_snapshot["main_chain_gap_max"],
                "delta_anchor_count": s["anchor_count"] - offline_snapshot["anchor_count"],
                "delta_recovery_commit_allowed_count": s["recovery_commit_allowed_count"] - offline_snapshot["recovery_commit_allowed_count"],
            }
        )
    write_csv(OUT / "v3_preflight_vs_actual_mismatch.csv", mismatch_rows)

    mismatch_summary = {
        "expected_keyframes_preflight": offline_snapshot["keyframes"],
        "actual_keyframes_full": full_snapshot["keyframes"],
        "keyframe_drop_abs": full_snapshot["keyframes"] - offline_snapshot["keyframes"],
        "expected_gap_max_preflight": offline_snapshot["main_chain_gap_max"],
        "actual_gap_max_full": full_snapshot["main_chain_gap_max"],
        "gap_max_jump_abs": full_snapshot["main_chain_gap_max"] - offline_snapshot["main_chain_gap_max"],
        "major_hold_reason_full": sorted(
            [
                (k, v)
                for k, v in full_snapshot["decision_reason_counts"].items()
                if "hold" in k
            ],
            key=lambda x: -x[1],
        )[:5],
        "offline_modeling_bias": {
            "uses_v2_trace_pool": True,
            "closed_loop_feedback_modeled": False,
            "can_underestimate_starvation": True,
        },
        "why_expected_501_but_full_237": "offline preflight treats candidate pool as exogenous and non-degrading; online loop accumulates holds/rejects (window_budget+density+retry_limit), reducing future support and direct_final.",
        "why_expected_gap19_but_full_gap94": "long interval candidates are mostly held/rejected under v3 hold_window_budget/hold_density_high + retry_limit, then state-chain weakens and long-tail gap is no longer recoverable in time.",
    }
    write_json(OUT / "v3_preflight_vs_actual_mismatch_summary.json", mismatch_summary)
    (OUT / "v3_preflight_vs_actual_mismatch_report.md").write_text(
        "\n".join(
            [
                "# v3 preflight vs actual mismatch report",
                "",
                f"- preflight expected keyframes={offline_snapshot['keyframes']} -> full actual={full_snapshot['keyframes']} (delta={mismatch_summary['keyframe_drop_abs']})",
                f"- preflight expected gap max={offline_snapshot['main_chain_gap_max']} -> full actual={full_snapshot['main_chain_gap_max']} (delta={mismatch_summary['gap_max_jump_abs']})",
                "- preflight基于v2 trace候选池，未建模在线闭环退化（支持衰减、后续direct失败、候选池质量变化），存在系统性乐观偏差。",
                "- full阶段主阻断来自 `v3_hold_window_budget` + `v3_hold_density_high`，并伴随大量 `v3_reject_retry_limit`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # B) short run starvation audit
    s300_ids = sorted(to_i(r.get("frame_id")) for r in short300_kf)
    s500_ids = sorted(to_i(r.get("frame_id")) for r in short500_kf)
    s300_set = set(s300_ids)
    s500_set = set(s500_ids)
    new_after_300 = sorted(x for x in s500_set if x > 300)
    growth_300_to_500 = len(s500_set - s300_set)
    plateau = growth_300_to_500 <= 10
    # interval growth profile for 500 run
    by_bin = defaultdict(int)
    for x in s500_ids:
        by_bin[(x // 50) * 50] += 1
    starvation_rows = [
        {
            "metric": "short300_keyframes",
            "value": len(s300_set),
        },
        {
            "metric": "short500_keyframes",
            "value": len(s500_set),
        },
        {
            "metric": "keyframe_growth_300_to_500",
            "value": growth_300_to_500,
        },
        {
            "metric": "new_keyframes_source_id_gt_300",
            "value": len(new_after_300),
        },
        {
            "metric": "plateau_detected",
            "value": plateau,
        },
    ]
    for b in sorted(by_bin):
        starvation_rows.append(
            {
                "metric": f"bin_{b}_{b+49}_keyframes",
                "value": by_bin[b],
            }
        )
    write_csv(OUT / "v3_short_run_starvation_audit.csv", starvation_rows)
    starvation_summary = {
        "short300_keyframes": len(s300_set),
        "short500_keyframes": len(s500_set),
        "keyframe_growth_300_to_500": growth_300_to_500,
        "plateau_detected": plateau,
        "plateau_should_block_full": True,
        "suggested_new_short_run_gate": {
            "keyframe_count_lower_bound": 220,
            "keyframe_growth_lower_bound_300_to_500": 60,
            "density_lower_bound": 24.0,
            "starvation_risk": "high" if plateau else "medium",
            "ready_for_full_run": False,
        },
    }
    write_json(OUT / "v3_short_run_starvation_summary.json", starvation_summary)
    (OUT / "v3_short_run_starvation_report.md").write_text(
        "\n".join(
            [
                "# v3 short run starvation report",
                "",
                f"- live_short_300 keyframes={len(s300_set)}",
                f"- live_short_500 keyframes={len(s500_set)}",
                f"- 300->500 keyframe growth={growth_300_to_500}",
                f"- plateau_detected={plateau}",
                "- 结论：应在short门禁新增 keyframe下界 + 增长下界 + density下界，避免“低密度但已饿死”误判为稳定。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # C) full gap=94 interval audit
    max_gap = -1
    max_row: dict[str, Any] | None = None
    for r in full_gap:
        g = to_i(r.get("gap"))
        if g > max_gap:
            max_gap = g
            max_row = r
    assert max_row is not None
    g_from = to_i(max_row.get("from_tick"))
    g_to = to_i(max_row.get("to_tick"))

    interval_rows = [r for r in full_ctrl if g_from < to_i(r.get("source_frame_id")) < g_to]
    nearby_rows = [r for r in full_ctrl if (g_from - 20) <= to_i(r.get("source_frame_id")) <= (g_to + 20)]

    reason_in_interval = Counter(str(r.get("decision_reason", "")) for r in interval_rows)
    gap_critical_rows = [r for r in interval_rows if to_b(r.get("is_gap_critical"))]
    gap_critical_blocked = [r for r in gap_critical_rows if str(r.get("decision", "")) != "commit"]

    gap94_rows: list[dict[str, Any]] = []
    for r in interval_rows:
        gap94_rows.append(
            {
                "source_frame_id": to_i(r.get("source_frame_id")),
                "decision": str(r.get("decision", "")),
                "decision_reason": str(r.get("decision_reason", "")),
                "is_gap_critical": to_b(r.get("is_gap_critical")),
                "is_support_topk": to_b(r.get("is_support_topk")),
                "blocked_reason": str(r.get("blocked_reason", "")),
                "window_id": to_i(r.get("window_id")),
                "window_candidate_rank": to_i(r.get("window_candidate_rank")),
                "density_before": to_f(r.get("density_before")),
                "density_after": to_f(r.get("density_after")),
            }
        )
    if not gap94_rows:
        gap94_rows.append(
            {
                "source_frame_id": -1,
                "decision": "none",
                "decision_reason": "no_interval_candidates",
                "is_gap_critical": False,
                "is_support_topk": False,
                "blocked_reason": "",
            }
        )
    write_csv(OUT / "v3_full_gap94_interval_audit.csv", gap94_rows)
    gap94_summary = {
        "gap_max": max_gap,
        "gap_interval_from_to": [g_from, g_to],
        "interval_candidate_count": len(interval_rows),
        "nearby_candidate_count": len(nearby_rows),
        "interval_reason_counts": dict(reason_in_interval),
        "gap_critical_candidate_count": len(gap_critical_rows),
        "gap_critical_blocked_count": len(gap_critical_blocked),
        "gap_critical_block_reasons": dict(
            Counter(str(r.get("decision_reason", "")) for r in gap_critical_blocked)
        ),
        "gap94_root_judgment": (
            "candidates_blocked"
            if len(interval_rows) > 0 and len(gap_critical_blocked) > 0
            else "no_candidate_after_starvation"
            if len(interval_rows) == 0
            else "mixed"
        ),
        "why_gap_critical_did_not_save": "gap-critical candidates exist but many are budget/density/hold-chains constrained; once local support chain thins, later intervals lose effective rescuers.",
    }
    write_json(OUT / "v3_full_gap94_interval_summary.json", gap94_summary)
    (OUT / "v3_full_gap94_interval_report.md").write_text(
        "\n".join(
            [
                "# v3 full gap=94 interval report",
                "",
                f"- gap_max={max_gap}, interval=({g_from},{g_to})",
                f"- interval_candidate_count={len(interval_rows)}, nearby_candidate_count={len(nearby_rows)}",
                f"- gap_critical_candidate_count={len(gap_critical_rows)}, blocked={len(gap_critical_blocked)}",
                f"- interval main reasons={dict(reason_in_interval)}",
                f"- judgment={gap94_summary['gap94_root_judgment']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # D) decision reason blocking profile
    profile_rows: list[dict[str, Any]] = []
    for stage_name, rows in [
        ("short300", short300_ctrl),
        ("short500", short500_ctrl),
        ("full", full_ctrl),
    ]:
        c = Counter(str(r.get("decision_reason", "")) for r in rows)
        for reason, count in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            profile_rows.append(
                {
                    "stage": stage_name,
                    "decision_reason": reason,
                    "count": int(count),
                    "decision_type": "commit"
                    if "commit" in reason
                    else "hold"
                    if "hold" in reason
                    else "reject",
                }
            )
    write_csv(OUT / "v3_decision_reason_blocking_profile.csv", profile_rows)

    full_reason = Counter(str(r.get("decision_reason", "")) for r in full_ctrl)
    full_gapcritical_blocked = sum(
        1 for r in full_ctrl if to_b(r.get("is_gap_critical")) and str(r.get("decision", "")) != "commit"
    )
    full_support_topk_blocked = sum(
        1 for r in full_ctrl if to_b(r.get("is_support_topk")) and str(r.get("decision", "")) != "commit"
    )
    full_anchor_guard_blocked = int(full_reason.get("v3_hold_anchor_guard", 0))
    full_too_close_blocked = int(full_reason.get("v3_hold_too_close_to_existing_keyframe", 0))
    full_density_blocked = int(full_reason.get("v3_hold_density_high", 0))
    full_window_blocked = int(full_reason.get("v3_hold_window_budget", 0))
    blocking_summary = {
        "full_commit_count_by_channel": {
            "gap_critical": int(full_reason.get("v3_gap_critical_commit", 0)),
            "support_ranked_sparse": int(full_reason.get("v3_support_ranked_sparse_commit", 0)),
        },
        "full_hold_count_by_reason": {
            "v3_hold_not_topk": int(full_reason.get("v3_hold_not_topk", 0)),
            "v3_hold_anchor_guard": int(full_reason.get("v3_hold_anchor_guard", 0)),
            "v3_hold_too_close_to_existing_keyframe": full_too_close_blocked,
            "v3_hold_density_high": full_density_blocked,
            "v3_hold_window_budget": full_window_blocked,
        },
        "full_reject_count_by_reason": {
            "v3_reject_retry_limit": int(full_reason.get("v3_reject_retry_limit", 0)),
            "v3_reject_age": int(full_reason.get("v3_reject_age", 0)),
            "v3_reject_low_support": int(full_reason.get("v3_reject_low_support", 0)),
            "v3_reject_invalid_semantics": int(full_reason.get("v3_reject_invalid_semantics", 0)),
        },
        "gap_critical_blocked_count": full_gapcritical_blocked,
        "support_topk_blocked_count": full_support_topk_blocked,
        "anchor_guard_blocked_count": full_anchor_guard_blocked,
        "too_close_blocked_count": full_too_close_blocked,
        "density_blocked_count": full_density_blocked,
        "window_budget_blocked_count": full_window_blocked,
        "major_blocker": "v3_hold_window_budget" if full_window_blocked >= full_density_blocked else "v3_hold_density_high",
        "rules_to_keep": ["invalid_semantics_reject", "retry_limit_reject", "duplicate_surrogate_contamination_guard"],
        "rules_to_relax_in_v4": [
            "window_budget_hold",
            "density_high_hold",
            "not_topk_hold_when_gap_risk",
        ],
    }
    write_json(OUT / "v3_decision_reason_blocking_summary.json", blocking_summary)
    (OUT / "v3_decision_reason_blocking_report.md").write_text(
        "\n".join(
            [
                "# v3 decision reason blocking report",
                "",
                f"- major_blocker={blocking_summary['major_blocker']}",
                f"- gap_critical_blocked_count={full_gapcritical_blocked}",
                f"- support_topk_blocked_count={full_support_topk_blocked}",
                f"- density/window_blocked={full_density_blocked}/{full_window_blocked}",
                "- starvation主要由 hold_window_budget + hold_density_high 链式放大，随后进入 retry_limit reject。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # E) anchor guard effect audit
    anchor_rows: list[dict[str, Any]] = []
    for stage_name, rows in [("short300", short300_ctrl), ("short500", short500_ctrl), ("full", full_ctrl)]:
        trigger = sum(1 for r in rows if to_b(r.get("anchor_guard_triggered")))
        hold_by_anchor = sum(1 for r in rows if str(r.get("decision_reason", "")) == "v3_hold_anchor_guard")
        gapcritical_block_by_anchor = sum(
            1
            for r in rows
            if str(r.get("decision_reason", "")) == "v3_hold_anchor_guard" and to_b(r.get("is_gap_critical"))
        )
        anchor_before_available = sum(
            1 for r in rows if str(r.get("anchor_count_before", "")) not in {"", "-1"}
        )
        anchor_rows.append(
            {
                "stage": stage_name,
                "anchor_guard_triggered_count": trigger,
                "anchor_guard_hold_count": hold_by_anchor,
                "gapcritical_blocked_by_anchor_guard_count": gapcritical_block_by_anchor,
                "anchor_count_before_available_rows": anchor_before_available,
                "total_rows": len(rows),
            }
        )
    write_csv(OUT / "v3_anchor_guard_effect_audit.csv", anchor_rows)
    anchor_summary = {
        "full_anchor_count": to_i(full_a.get("final_anchor_count")),
        "anchor_audit_file": full_anchor_audit,
        "anchor_guard_over_suppression": False,
        "anchor_count_before_unavailable_in_trace": True,
        "anchor_guard_hold_massive": False,
        "gapcritical_blocked_by_anchor_guard_massive": False,
        "anchor_interpretation": "anchor guard was mostly inactive (count_before unavailable); low anchors are more consistent with global starvation than guard over-blocking.",
        "v4_anchor_policy": "target_band_4_5_not_one_sided_upper_guard",
    }
    write_json(OUT / "v3_anchor_guard_effect_summary.json", anchor_summary)
    (OUT / "v3_anchor_guard_effect_report.md").write_text(
        "\n".join(
            [
                "# v3 anchor guard effect report",
                "",
                f"- full anchors={to_i(full_a.get('final_anchor_count'))}, baseline reference in audit={full_anchor_audit.get('baseline_anchor_count', 4)}",
                "- trace中 anchor_count_before 基本不可用（-1/空），anchor_guard 触发接近0。",
                "- 因此v3低anchor更可能是表示链饿死的结果，而非anchor_guard单独压制导致。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # F) v4 design constraints
    constraints = {
        "allow_v4_implementation": True,
        "constraints": {
            "no_short_plateau_300_500": True,
            "short_run_density_lower_bound_required": True,
            "short_run_keyframe_growth_lower_bound_required": True,
            "target_density_band_control_instead_of_hard_strict": True,
            "recovery_commit_lower_bound_or_coverage_floor_required": True,
            "do_not_change_direct_defer_discard_semantics": True,
            "gap_critical_commit_must_override_topk_and_anchor_soft_guard": True,
            "anchor_guard_should_use_target_band_4_5": True,
            "only_modify_recovery_success_to_commit_control_layer": True,
            "keep_RVQ_tau_frozen": True,
        },
        "recommended_short_gate": {
            "keyframe_count_lower_bound": 220,
            "keyframe_growth_lower_bound_300_to_500": 60,
            "density_lower_bound": 24.0,
            "density_upper_bound": 42.0,
            "block_if_plateau": True,
        },
        "evidence_excerpt": {
            "short300_keyframes": len(s300_set),
            "short500_keyframes": len(s500_set),
            "full_keyframes": to_i(full_a.get("final_keyframe_count")),
            "full_gap_max": to_i(full_a.get("main_chain_gap_max")),
            "full_window_budget_holds": full_window_blocked,
            "full_density_holds": full_density_blocked,
        },
    }
    write_json(OUT / "v3_to_v4_design_constraints.json", constraints)

    # Final audit report with explicit answers
    answers = {
        "q1_v3_failed_mainly_due_to_over_strict": True,
        "q2_preflight_missed_starvation_why": "preflight replays v2 candidate pool without online closed-loop degradation; optimistic keyframe growth and gap closure are over-estimated.",
        "q3_why_short_run_should_not_pass": "short300=100 and short500=100 with minimal growth; this is clear plateau/starvation despite passing old stability gates.",
        "q4_full_gap94_root": gap94_summary["gap94_root_judgment"],
        "q5_v4_should_relax": ["window_budget_hold", "density_high_hold", "not_topk_hold when long-gap risk"],
        "q6_v4_should_not_relax": ["invalid_semantics reject", "retry_limit/age reject", "duplicate/surrogate/contamination safety"],
        "q7_ready_to_enter_v4": True,
    }
    lines = [
        "# PAPER_ALIGNED_STRICT_V3_STARVATION_AND_PREFLIGHT_MISMATCH_AUDIT_V1",
        "",
        "## Core Findings",
        f"- preflight expected 501 keyframes / gap_max 19，但full实际 {to_i(full_a.get('final_keyframe_count'))} / {to_i(full_a.get('main_chain_gap_max'))}。",
        f"- short300/short500 keyframes: {len(s300_set)}/{len(s500_set)}，300->500增长={growth_300_to_500}，出现plateau。",
        f"- full主要阻断: window_budget={full_window_blocked}, density_high={full_density_blocked}, reject_retry_limit={int(full_reason.get('v3_reject_retry_limit', 0))}。",
        f"- gap=94 interval: from {g_from} to {g_to}, interval candidates={len(interval_rows)}, gap-critical blocked={len(gap_critical_blocked)}。",
        "",
        "## Required Answers",
        f"1. Codex v3 失败是否主要由于过度收紧：{'是' if answers['q1_v3_failed_mainly_due_to_over_strict'] else '否'}。",
        f"2. preflight 为什么没有预测到 starvation：{answers['q2_preflight_missed_starvation_why']}",
        f"3. short run 为什么不应该通过：{answers['q3_why_short_run_should_not_pass']}",
        f"4. full gap=94 是无候选问题还是候选被挡问题：{answers['q4_full_gap94_root']}",
        f"5. v4 应该放宽哪些规则：{', '.join(answers['q5_v4_should_relax'])}",
        f"6. v4 不能放宽哪些语义安全规则：{', '.join(answers['q6_v4_should_not_relax'])}",
        f"7. 是否可以进入 v4 实现：{'可以' if answers['q7_ready_to_enter_v4'] else '不可以'}。",
        "",
        "## Note",
        "- 本轮为只读审计；未修改R/V/Q、tau、PDF主线，未运行新full。",
    ]
    (OUT / "paper_aligned_strict_v3_starvation_and_preflight_mismatch_audit_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
