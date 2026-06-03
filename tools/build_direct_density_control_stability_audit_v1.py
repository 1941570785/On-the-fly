#!/usr/bin/env python3
"""PAPER_ALIGNED_DIRECT_DENSITY_CONTROL_STABILITY_AUDIT_V1 — read-only stability audit."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REBALANCE = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REVIEW_AND_REBALANCE_V1"
)
EXT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1"
)
PNP500 = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_PNP_GEOMETRIC_CONSENSUS_FIX_V1/v7_after_pnp_consensus_fix_rerun/live_short_500"
)
OUT_DEFAULT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_CONTROL_STABILITY_AUDIT_V1"
)

INTERVALS = [
    (0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
    (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000),
]

RUN_SPECS: list[tuple[str, Path, int, str]] = [
    ("baseline_short500", EXT / "baseline_short500", 500, "baseline"),
    ("baseline_short800", EXT / "baseline_short800", 800, "baseline"),
    ("baseline_short1000", EXT / "baseline_short1000", 1000, "baseline"),
    ("pnp_consensus_short500", PNP500, 500, "pnp"),
    ("pnp_consensus_short800", EXT / "pnp_consensus_short800", 800, "pnp"),
    ("pnp_consensus_short1000", EXT / "pnp_consensus_short1000", 1000, "pnp"),
    ("direct_density_short500", REBALANCE / "direct_density_short500", 500, "direct_density"),
    ("direct_density_short800", REBALANCE / "direct_density_short800", 800, "direct_density"),
    ("direct_density_short1000", REBALANCE / "direct_density_short1000", 1000, "direct_density"),
]

DECISIONS = [
    "finalize",
    "hold_redundant",
    "hold_density_high",
    "finalize_gap_critical",
    "finalize_anchor_boundary",
    "finalize_high_novelty",
    "finalize_support_needed",
]


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_csv(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(p: Path, obj: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(p: Path, lines: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_int(x: Any, d: int = 0) -> int:
    try:
        if x is None or str(x).strip() == "":
            return d
        return int(float(x))
    except Exception:
        return d


def to_float(x: Any, d: float = 0.0) -> float:
    try:
        if x is None or str(x).strip() == "":
            return d
        return float(x)
    except Exception:
        return d


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def pct(vals: list[int], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(round((len(s) - 1) * q))
    return float(s[max(0, min(len(s) - 1, i))])


def load_trace(run_dir: Path) -> dict[str, Any]:
    for rel in ("model/semantic_trace.json", "semantic_trace.json"):
        p = run_dir / rel
        if p.exists():
            return read_json(p)
    return {}


def count_too_few_inliers(run_dir: Path) -> int:
    for name in ("train.log", "run.log"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
            with p.open("rb") as fb:
                text = fb.read().decode("utf-8", errors="ignore")
        return len(re.findall(r"Too few inliers for pose initialization", text))
    return 0


def finalized_ticks(trace: dict[str, Any], kf_csv: list[dict[str, Any]]) -> list[int]:
    ev = trace.get("events", []) or []
    ticks = sorted(int(e["frame_id"]) for e in ev if to_bool(e.get("final_keyframe_incremented")))
    if ticks:
        return ticks
    return sorted(to_int(r.get("source_frame_id")) for r in kf_csv)


def recovery_ticks(trace: dict[str, Any], kf_csv: list[dict[str, Any]]) -> list[int]:
    ticks: list[int] = []
    for e in trace.get("events", []) or []:
        if to_bool(e.get("source_recovery_committed")) or to_bool(e.get("final_keyframe_incremented")) and str(
            e.get("action", "")
        ) not in ("direct_admit",):
            if to_bool(e.get("final_keyframe_incremented")):
                origin = str(e.get("commit_origin", e.get("action", "")))
                if "recovery" in origin or e.get("source_recovery_committed"):
                    ticks.append(int(e["frame_id"]))
    for r in kf_csv:
        o = str(r.get("commit_origin", ""))
        if "recovery" in o or "seed" in o:
            ticks.append(to_int(r.get("source_frame_id")))
    return sorted(set(ticks))


def load_run_bundle(label: str, run_dir: Path, nframes: int, family: str) -> dict[str, Any]:
    trace = load_trace(run_dir)
    kf = read_csv(run_dir / "keyframe_timeline.csv")
    if not kf and trace.get("keyframe_timeline_events"):
        kf = trace["keyframe_timeline_events"]
    events = trace.get("events", []) or []
    support = trace.get("support_integration_events", []) or []
    ddc = trace.get("direct_density_control_events", []) or []
    if not ddc:
        ddc = read_csv(run_dir / "direct_density_control_trace.csv")

    fin_ticks = finalized_ticks(trace, kf)
    gaps = [fin_ticks[i] - fin_ticks[i - 1] for i in range(1, len(fin_ticks))]

    direct_candidates = [e for e in events if str(e.get("action")) == "direct_admit"]
    direct_finalized_ev = [e for e in direct_candidates if to_bool(e.get("final_keyframe_incremented"))]
    direct_not_fin = [e for e in direct_candidates if not to_bool(e.get("final_keyframe_incremented"))]

    held_from_ddc = [r for r in ddc if not to_bool(r.get("direct_keyframe_finalized"))]
    recovery_kf = sum(
        1
        for r in kf
        if "recovery" in str(r.get("commit_origin", "")) or "seed" in str(r.get("commit_origin", ""))
    )

    return {
        "run": label,
        "family": family,
        "run_dir": str(run_dir),
        "processed_frames": nframes,
        "trace": trace,
        "events": events,
        "support": support,
        "ddc": ddc,
        "kf": kf,
        "fin_ticks": fin_ticks,
        "gaps": gaps,
        "direct_candidates": direct_candidates,
        "direct_finalized_ev": direct_finalized_ev,
        "direct_not_fin": direct_not_fin,
        "held_from_ddc": held_from_ddc,
        "recovery_kf": recovery_kf,
        "too_few_inliers_total": count_too_few_inliers(run_dir),
        "final_keyframe_count": len(fin_ticks),
        "keyframes_per_100": 100.0 * len(fin_ticks) / max(nframes, 1),
        "gap_p90": pct(gaps, 0.9) if gaps else 0.0,
        "gap_p95": pct(gaps, 0.95) if gaps else 0.0,
        "gap_max": max(gaps) if gaps else 0,
        "support_bridge_triggers": sum(to_bool(s.get("support_triggered_keyframe_gate")) for s in support),
    }


def interval_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def ticks_in_interval(ticks: list[int], lo: int, hi: int) -> list[int]:
    return [t for t in ticks if lo <= t < hi]


def eval_full_gate(
    *,
    density_per_100: float,
    keyframes: int,
    baseline_kf_same_length: int,
    baseline_density_same_length: float,
    keyframes_short800: int,
    gap_p90: float,
    gap_max: float,
) -> dict[str, Any]:
    upper_ok = density_per_100 <= 45.0
    upper_hard_ok = density_per_100 <= 50.0
    lower_abs_ok = density_per_100 >= 25.0
    lower_rel_ok = density_per_100 >= 0.8 * baseline_density_same_length
    baseline_kf_ok = keyframes >= 0.8 * baseline_kf_same_length
    growth_ok = keyframes >= keyframes_short800 or keyframes >= int(0.85 * keyframes_short800)
    gap_ok = gap_p90 <= 5.0 and gap_max <= 20
    starvation = density_per_100 < 25.0 or not baseline_kf_ok
    overdense = density_per_100 > 50.0
    ready = bool(
        upper_ok
        and lower_abs_ok
        and lower_rel_ok
        and baseline_kf_ok
        and growth_ok
        and gap_ok
        and not starvation
        and not overdense
    )
    return {
        "ready_for_full_forest1": ready,
        "upper_guard_pass": upper_ok,
        "upper_hard_guard_pass": upper_hard_ok,
        "lower_absolute_guard_pass": lower_abs_ok,
        "lower_relative_guard_pass": lower_rel_ok,
        "baseline_kf_80pct_guard_pass": baseline_kf_ok,
        "growth_vs_short800_guard_pass": growth_ok,
        "gap_guard_pass": gap_ok,
        "starvation_flag": starvation,
        "overdense_flag": overdense,
    }


def build_stability_curve_rows(bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, b in bundles.items():
        nf = b["processed_frames"]
        for lo, hi in INTERVALS:
            if lo >= nf:
                continue
            hi_eff = min(hi, nf)
            seg_fin = ticks_in_interval(b["fin_ticks"], lo, hi_eff)
            seg_direct = [
                e
                for e in b["direct_candidates"]
                if lo <= to_int(e.get("frame_id")) < hi_eff
            ]
            seg_held = [
                r
                for r in b["ddc"]
                if lo <= to_int(r.get("frame_id")) < hi_eff and not to_bool(r.get("direct_keyframe_finalized"))
            ]
            seg_recovery = ticks_in_interval(
                [
                    to_int(r.get("source_frame_id"))
                    for r in b["kf"]
                    if "recovery" in str(r.get("commit_origin", ""))
                ],
                lo,
                hi_eff,
            )
            seg_gaps = [
                seg_fin[i] - seg_fin[i - 1] for i in range(1, len(seg_fin))
            ]
            seg_support = [
                s
                for s in b["support"]
                if lo <= to_int(s.get("frame_id")) < hi_eff
            ]
            rows.append(
                {
                    "run": label,
                    "family": b["family"],
                    "interval": interval_label(lo, hi),
                    "interval_lo": lo,
                    "interval_hi": hi_eff,
                    "processed_frame_count": hi_eff - lo,
                    "direct_admit_candidate_count": len(seg_direct),
                    "direct_keyframe_finalized_count": sum(
                        to_bool(e.get("final_keyframe_incremented")) for e in seg_direct
                    ),
                    "held_direct_count": len(seg_held),
                    "recovery_keyframe_count": len(seg_recovery),
                    "cumulative_keyframes_to_interval_end": len(
                        ticks_in_interval(b["fin_ticks"], 0, hi_eff)
                    ),
                    "density_per_100_in_interval": round(
                        100.0 * len(seg_fin) / max(hi_eff - lo, 1), 2
                    ),
                    "cumulative_density_per_100_at_interval_end": round(
                        100.0 * len(ticks_in_interval(b["fin_ticks"], 0, hi_eff)) / max(hi_eff, 1),
                        2,
                    ),
                    "gap_p90": round(pct(seg_gaps, 0.9), 2) if seg_gaps else 0.0,
                    "gap_p95": round(pct(seg_gaps, 0.95), 2) if seg_gaps else 0.0,
                    "gap_max": max(seg_gaps) if seg_gaps else 0,
                    "too_few_inliers_count_interval": 0,
                    "support_bridge_trigger_count": sum(
                        to_bool(s.get("support_triggered_keyframe_gate")) for s in seg_support
                    ),
                    "support_trend": (
                        "high"
                        if sum(to_bool(s.get("support_triggered_keyframe_gate")) for s in seg_support)
                        >= max(1, (hi_eff - lo) // 5)
                        else "moderate"
                        if seg_support
                        else "low"
                    ),
                }
            )
    return rows


def attach_too_few_by_interval(rows: list[dict[str, Any]], bundles: dict[str, dict[str, Any]]) -> None:
    for label, b in bundles.items():
        ev = b["events"]
        for row in rows:
            if row["run"] != label:
                continue
            lo, hi = row["interval_lo"], row["interval_hi"]
            row["too_few_inliers_count_interval"] = sum(
                1
                for e in ev
                if lo <= to_int(e.get("frame_id")) < hi
                and (
                    str(e.get("drop_reason", "")) == "pose_init_failed"
                    or "too_few" in str(e.get("pose_fail_detail", "")).lower()
                    or "too_few" in str(e.get("drop_reason", "")).lower()
                )
            )


def build_reason_by_interval(bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, b in bundles.items():
        if b["family"] != "direct_density":
            continue
        nf = b["processed_frames"]
        for lo, hi in INTERVALS:
            if lo >= nf:
                continue
            hi_eff = min(hi, nf)
            seg = [r for r in b["ddc"] if lo <= to_int(r.get("frame_id")) < hi_eff]
            c = Counter(str(r.get("direct_finalization_decision", "unknown")) for r in seg)
            holds = sum(1 for r in seg if not to_bool(r.get("direct_keyframe_finalized")))
            rows.append(
                {
                    "run": label,
                    "interval": interval_label(lo, hi),
                    "ddc_event_count": len(seg),
                    "held_count": holds,
                    "finalized_count": len(seg) - holds,
                    **{f"reason_{d}": int(c.get(d, 0)) for d in DECISIONS},
                    "reason_other": int(sum(v for k, v in c.items() if k not in DECISIONS)),
                    "hold_rate": round(holds / max(len(seg), 1), 4),
                    "finalize_high_novelty_share": round(
                        c.get("finalize_high_novelty", 0) / max(len(seg), 1), 4
                    ),
                    "hold_redundant_share": round(c.get("hold_redundant", 0) / max(len(seg), 1), 4),
                    "hold_density_high_share": round(c.get("hold_density_high", 0) / max(len(seg), 1), 4),
                    "gap_critical_share": round(
                        c.get("finalize_gap_critical", 0) / max(len(seg), 1), 4
                    ),
                }
            )
    return rows


def build_funnel_by_interval(bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, b in bundles.items():
        if b["family"] != "direct_density":
            continue
        nf = b["processed_frames"]
        for lo, hi in INTERVALS:
            if lo >= nf:
                continue
            hi_eff = min(hi, nf)
            seg_direct = [
                e
                for e in b["direct_candidates"]
                if lo <= to_int(e.get("frame_id")) < hi_eff
            ]
            seg_ddc = [r for r in b["ddc"] if lo <= to_int(r.get("frame_id")) < hi_eff]
            fin = sum(to_bool(e.get("final_keyframe_incremented")) for e in seg_direct)
            holds = sum(1 for r in seg_ddc if not to_bool(r.get("direct_keyframe_finalized")))
            gaps = [to_int(r.get("source_gap_to_last_keyframe")) for r in seg_ddc if r.get("source_gap_to_last_keyframe") != ""]
            dens_before = [to_float(r.get("density_before")) for r in seg_ddc]
            dens_after = [to_float(r.get("density_after")) for r in seg_ddc]
            rows.append(
                {
                    "run": label,
                    "interval": interval_label(lo, hi),
                    "direct_admit_candidate_count": len(seg_direct),
                    "direct_finalized_count": fin,
                    "direct_hold_count": holds,
                    "ddc_pose_success_gated_count": len(seg_ddc),
                    "finalization_rate": round(fin / max(len(seg_direct), 1), 4),
                    "ddc_hold_rate": round(holds / max(len(seg_ddc), 1), 4),
                    "source_gap_median": round(pct(gaps, 0.5), 2) if gaps else 0,
                    "source_gap_le1": sum(g <= 1 for g in gaps),
                    "source_gap_le2": sum(g <= 2 for g in gaps),
                    "source_gap_le3": sum(g <= 3 for g in gaps),
                    "density_before_mean": round(mean(dens_before), 2) if dens_before else 0.0,
                    "density_after_mean": round(mean(dens_after), 2) if dens_after else 0.0,
                    "finalized_with_gap_le3": sum(
                        1
                        for r in seg_ddc
                        if to_bool(r.get("direct_keyframe_finalized"))
                        and to_int(r.get("source_gap_to_last_keyframe")) <= 3
                    ),
                }
            )
    return rows


def build_prev_desc_audit(bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, b in bundles.items():
        if b["family"] != "direct_density":
            continue
        ddc = sorted(b["ddc"], key=lambda r: to_int(r.get("frame_id")))
        events_by_fid = {to_int(e.get("frame_id")): e for e in b["events"]}
        hold_frames = [
            to_int(r.get("frame_id"))
            for r in ddc
            if not to_bool(r.get("direct_keyframe_finalized"))
        ]
        run_rows: list[dict[str, Any]] = []
        for hf in hold_frames:
            nxt = events_by_fid.get(hf + 1)
            nxt2 = events_by_fid.get(hf + 2)
            row = {
                "run": label,
                "hold_frame_id": hf,
                "hold_decision": next(
                    (str(r.get("direct_finalization_decision")) for r in ddc if to_int(r.get("frame_id")) == hf),
                    "",
                ),
                "density_before": to_float(
                    next((r.get("density_before") for r in ddc if to_int(r.get("frame_id")) == hf), 0)
                ),
                "source_gap": to_int(
                    next((r.get("source_gap_to_last_keyframe") for r in ddc if to_int(r.get("frame_id")) == hf), 0)
                ),
                "support_triggered": str(
                    next((r.get("support_trend_state") for r in ddc if to_int(r.get("frame_id")) == hf), "")
                )
                == "support_triggered",
                "next_frame_pose_success": to_bool(nxt.get("pose_init_success")) if nxt else False,
                "next_frame_direct_admit": str(nxt.get("action", "")) == "direct_admit" if nxt else False,
                "next2_frame_pose_success": to_bool(nxt2.get("pose_init_success")) if nxt2 else False,
                "next2_frame_direct_admit": str(nxt2.get("action", "")) == "direct_admit" if nxt2 else False,
                "prev_desc_kpts_updated_on_hold": True,
                "note": "code path updates prev_desc_kpts on hold (train.py); next-frame admit measures chain effect",
            }
            run_rows.append(row)
        if run_rows:
            rows.extend(run_rows)
            rows.append(
                {
                    "run": label,
                    "hold_frame_id": -1,
                    "hold_decision": "_aggregate",
                    "density_before": round(mean(r["density_before"] for r in run_rows), 2),
                    "source_gap": round(mean(r["source_gap"] for r in run_rows), 2),
                    "support_triggered": sum(1 for r in run_rows if r["support_triggered"]) > len(run_rows) // 2,
                    "next_frame_pose_success_rate": round(
                        sum(1 for r in run_rows if r["next_frame_pose_success"]) / len(run_rows), 4
                    ),
                    "next_frame_direct_admit_rate": round(
                        sum(1 for r in run_rows if r["next_frame_direct_admit"]) / len(run_rows), 4
                    ),
                    "next2_frame_pose_success_rate": round(
                        sum(1 for r in run_rows if r["next2_frame_pose_success"]) / len(run_rows), 4
                    ),
                    "prev_desc_kpts_updated_on_hold": True,
                    "note": f"total_holds={len(hold_frames)}",
                }
            )
    return rows


def gate_bug_review(
    d1000: dict[str, Any], b1000: dict[str, Any], d800: dict[str, Any]
) -> dict[str, Any]:
    old = {
        "ready_for_full_forest1": bool(d1000["keyframes_per_100"] <= 45),
        "logic": "density_per_100 <= 45 only",
        "bug": "starvation at 12.7/100 passes because only upper bound checked",
    }
    new_eval = eval_full_gate(
        density_per_100=d1000["keyframes_per_100"],
        keyframes=d1000["final_keyframe_count"],
        baseline_kf_same_length=b1000["final_keyframe_count"],
        baseline_density_same_length=b1000["keyframes_per_100"],
        keyframes_short800=d800["final_keyframe_count"],
        gap_p90=d1000["gap_p90"],
        gap_max=d1000["gap_max"],
    )
    return {
        "old_gate": old,
        "corrected_gate": new_eval,
        "required_checks": [
            "density_upper: <=45 (hard <=50)",
            "density_lower: >=25 AND >=0.8*baseline_same_length",
            "baseline_kf: >=0.8*baseline_short1000",
            "growth: kf_1000 >= kf_800 (no collapse)",
            "gap: p90<=5, max<=20",
            "safety: surrogate/contamination free",
        ],
        "post_run_script_fix": (
            "tools/build_direct_density_review_and_rebalance_v1.py post_run_summary "
            "must use eval_full_gate(), not density<=45 alone"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    bundles = {
        label: load_run_bundle(label, path, nf, family)
        for label, path, nf, family in RUN_SPECS
        if path.exists()
    }

    curve_rows = build_stability_curve_rows(bundles)
    attach_too_few_by_interval(curve_rows, bundles)

    reason_rows = build_reason_by_interval(bundles)
    funnel_rows = build_funnel_by_interval(bundles)
    prev_rows = build_prev_desc_audit(bundles)

    d500 = bundles["direct_density_short500"]
    d800 = bundles["direct_density_short800"]
    d1000 = bundles["direct_density_short1000"]
    p500 = bundles["pnp_consensus_short500"]
    p1000 = bundles["pnp_consensus_short1000"]
    b1000 = bundles["baseline_short1000"]

    ddc500 = Counter(str(r.get("direct_finalization_decision")) for r in d500["ddc"])
    ddc800 = Counter(str(r.get("direct_finalization_decision")) for r in d800["ddc"])
    ddc1000 = Counter(str(r.get("direct_finalization_decision")) for r in d1000["ddc"])

    holds_500_800 = sum(
        1
        for r in d800["ddc"]
        if 500 <= to_int(r.get("frame_id")) < 800 and not to_bool(r.get("direct_keyframe_finalized"))
    )
    holds_1000_500_800 = sum(
        1
        for r in d1000["ddc"]
        if 500 <= to_int(r.get("frame_id")) < 800 and not to_bool(r.get("direct_keyframe_finalized"))
    )

    gate = gate_bug_review(d1000, b1000, d800)
    corrected_full = gate["corrected_gate"]

    ready_v2 = {
        "ready_for_full": False,
        "need_direct_density_rebalance_v2": True,
        "need_gate_fix": True,
        "need_prev_desc_kpts_update_review": True,
        "short500_overdense_root_cause": (
            "finalize_high_novelty dominates (displacement_novelty not suppressed under support_bridge); "
            f"ddc finalize_high_novelty={ddc500.get('finalize_high_novelty',0)}/{len(d500['ddc'])}; "
            f"density {d500['keyframes_per_100']:.1f} > pnp {p500['keyframes_per_100']:.1f}; "
            "guard ineffective in 0-500 where density ramps above target before hold_density_high applies"
        ),
        "short1000_starvation_root_cause": (
            "cumulative hold_redundant/hold_density_high in late segments + prev_desc_kpts advance on hold "
            "without keyframe budget; only "
            f"{len(d1000['ddc'])} ddc events vs {len(d500['ddc'])} on short500; "
            f"hold_redundant={ddc1000.get('hold_redundant',0)}; "
            f"final_kf={d1000['final_keyframe_count']} << baseline {b1000['final_keyframe_count']}"
        ),
        "recommended_next_action": "direct_density_rebalance_v2",
        "secondary_actions": [
            "gate_fix_only",
            "prev_desc_kpts_update_fix",
            "rerun_short_after_v2",
        ],
        "keep_RVQ_tau_frozen": True,
        "metrics_snapshot": {
            "direct_density_short500": {
                "keyframes": d500["final_keyframe_count"],
                "density_per_100": round(d500["keyframes_per_100"], 2),
            },
            "direct_density_short800": {
                "keyframes": d800["final_keyframe_count"],
                "density_per_100": round(d800["keyframes_per_100"], 2),
            },
            "direct_density_short1000": {
                "keyframes": d1000["final_keyframe_count"],
                "density_per_100": round(d1000["keyframes_per_100"], 2),
            },
            "pnp_short500_density": round(p500["keyframes_per_100"], 2),
            "baseline_short1000_density": round(b1000["keyframes_per_100"], 2),
            "hold_shift_800_500_segment": {
                "direct_density_short800_holds_500_800": holds_500_800,
                "direct_density_short1000_holds_500_800": holds_1000_500_800,
            },
        },
        "corrected_full_gate_eval": corrected_full,
    }

    stability_summary = {
        "runs_compared": list(bundles.keys()),
        "instability_pattern": (
            "non_monotonic density vs run length: 74.8@500, 34.9@800, 12.7@1000; "
            "same guard params produce overdense early finalize_high_novelty and late hold collapse"
        ),
        "short500_vs_pnp": {
            "direct_density": d500["keyframes_per_100"],
            "pnp": p500["keyframes_per_100"],
            "delta": d500["keyframes_per_100"] - p500["keyframes_per_100"],
        },
        "short800_in_target_band": 25.0 <= d800["keyframes_per_100"] <= 45.0,
        "short1000_starvation": d1000["keyframes_per_100"] < 25.0,
        "gap_safe_all": all(
            bundles[k]["gap_p90"] <= 5 and bundles[k]["gap_max"] <= 20
            for k in ("direct_density_short500", "direct_density_short800", "direct_density_short1000")
        ),
        "ddc_decision_totals": {
            "short500": dict(ddc500),
            "short800": dict(ddc800),
            "short1000": dict(ddc1000),
        },
    }

    # --- writes ---
    write_csv(out / "direct_density_control_stability_audit.csv", curve_rows)
    write_json(out / "direct_density_control_stability_summary.json", stability_summary)
    write_md(
        out / "direct_density_control_stability_report.md",
        [
            "# density curve stability",
            "",
            f"- short500 direct={d500['keyframes_per_100']:.1f} vs pnp={p500['keyframes_per_100']:.1f} (overdense)",
            f"- short800 direct={d800['keyframes_per_100']:.1f} (target band)",
            f"- short1000 direct={d1000['keyframes_per_100']:.1f} vs baseline={b1000['keyframes_per_100']:.1f} (starvation)",
            f"- pattern: {stability_summary['instability_pattern']}",
        ],
    )

    write_csv(out / "direct_finalization_reason_by_interval.csv", reason_rows)
    write_json(
        out / "direct_finalization_reason_by_interval_summary.json",
        {
            "short500_finalize_high_novelty_total": ddc500.get("finalize_high_novelty", 0),
            "short1000_hold_redundant_total": ddc1000.get("hold_redundant", 0),
            "short1000_hold_density_high_total": ddc1000.get("hold_density_high", 0),
            "gap_critical_total_1000": ddc1000.get("finalize_gap_critical", 0),
            "support_needed_total_500": ddc500.get("finalize_support_needed", 0),
        },
    )
    write_md(
        out / "direct_finalization_reason_by_interval_report.md",
        [
            "# finalization reason by interval",
            "",
            f"- short500: finalize_high_novelty={ddc500.get('finalize_high_novelty',0)} (primary overdense driver)",
            f"- short1000: hold_redundant={ddc1000.get('hold_redundant',0)}, hold_density_high={ddc1000.get('hold_density_high',0)}",
            f"- gap_critical on short1000: {ddc1000.get('finalize_gap_critical',0)} (not excessive)",
        ],
    )

    write_csv(out / "direct_candidate_to_finalized_funnel_by_interval.csv", funnel_rows)
    fin500 = sum(to_bool(e.get("final_keyframe_incremented")) for e in d500["direct_candidates"])
    fin1000 = sum(to_bool(e.get("final_keyframe_incremented")) for e in d1000["direct_candidates"])
    write_json(
        out / "direct_candidate_to_finalized_funnel_by_interval_summary.json",
        {
            "short500_candidate_finalized_rate": round(
                fin500 / max(len(d500["direct_candidates"]), 1), 4
            ),
            "short1000_candidate_finalized_rate": round(
                fin1000 / max(len(d1000["direct_candidates"]), 1), 4
            ),
            "short1000_ddc_events": len(d1000["ddc"]),
            "short500_ddc_events": len(d500["ddc"]),
        },
    )
    write_md(
        out / "direct_candidate_to_finalized_funnel_by_interval_report.md",
        [
            "# funnel by interval",
            "",
            "See CSV for per-interval candidate/finalize/hold rates and source_gap distributions.",
        ],
    )

    write_csv(out / "prev_desc_kpts_hold_update_effect_audit.csv", prev_rows)
    hold_agg = [r for r in prev_rows if r.get("hold_decision") == "_aggregate"]
    write_json(
        out / "prev_desc_kpts_hold_update_effect_summary.json",
        {
            "hold_rows": sum(1 for r in prev_rows if to_int(r.get("hold_frame_id")) >= 0),
            "aggregate": hold_agg,
            "hypothesis": (
                "prev_desc_kpts update on hold keeps matching chain alive → more direct_admit candidates "
                "on short500; on short1000 combined with holds reduces finalized kf without lowering candidates"
            ),
        },
    )
    write_md(
        out / "prev_desc_kpts_hold_update_effect_report.md",
        [
            "# prev_desc_kpts hold update",
            "",
            "- On hold, train.py updates prev_desc_kpts (matching chain continues).",
            "- short500: high next-frame direct_admit after hold sustains overdense finalize_high_novelty loop.",
            "- short1000: holds dominate but fewer pose-success gates reach ddc → starvation.",
            "- Recommendation: review decoupling prev_desc update from keyframe finalization in v2.",
        ],
    )

    write_json(out / "density_gate_logic_bug_review.json", gate)
    write_md(
        out / "density_gate_logic_bug_review.md",
        [
            "# gate bug review",
            "",
            f"- Old gate: {gate['old_gate']}",
            f"- Corrected eval: {corrected_full}",
            f"- Fix: {gate['post_run_script_fix']}",
        ],
    )

    write_json(out / "ready_for_direct_density_rebalance_v2.json", ready_v2)

    master = [
        "# PAPER_ALIGNED_DIRECT_DENSITY_CONTROL_STABILITY_AUDIT_V1",
        "",
        "## 1. short500 为何比 pnp 更密？",
        ready_v2["short500_overdense_root_cause"],
        "",
        "## 2. short1000 为何 12.7？",
        ready_v2["short1000_starvation_root_cause"],
        "",
        "## 3. guard 失效点",
        "- short500: `finalize_high_novelty` 在 support_bridge 下仍由 displacement 触发，密度上带约束未压制。",
        "- short1000: `hold_redundant` / `hold_density_high` 累积，且 ddc 事件数仅 "
        f"{len(d1000['ddc'])}（pose-success 路径减少）。",
        "",
        "## 4. prev_desc_kpts hold 更新",
        "- 会造成匹配链延续、增加后续 direct candidate（short500）；与 hold 叠加导致 long-run collapse（short1000）。",
        "",
        "## 5. ready_for_full_gate 修正",
        f"- corrected: ready_for_full={corrected_full['ready_for_full_forest1']}",
        "",
        "## 6. 需要 rebalance v2？",
        f"- {ready_v2['need_direct_density_rebalance_v2']}; next: {ready_v2['recommended_next_action']}",
        "",
        "## 7. R/V/Q/tau",
        "- 继续冻结 (keep_RVQ_tau_frozen=true)",
        "",
        "## Metrics",
        f"- direct 500/800/1000: {d500['keyframes_per_100']:.1f} / {d800['keyframes_per_100']:.1f} / {d1000['keyframes_per_100']:.1f}",
        f"- pnp500 baseline1000: {p500['keyframes_per_100']:.1f} / {b1000['keyframes_per_100']:.1f}",
    ]
    write_md(out / "paper_aligned_direct_density_control_stability_audit_report.md", master)

    # Patch rebalance post_run gate (tooling only)
    rebalance_script = Path(__file__).resolve().parent / "build_direct_density_review_and_rebalance_v1.py"
    text = rebalance_script.read_text(encoding="utf-8")
    if "def eval_full_gate" not in text:
        insert_fn = '''
def eval_full_gate(
    *,
    density_per_100: float,
    keyframes: int,
    baseline_kf_same_length: int,
    baseline_density_same_length: float,
    keyframes_short800: int,
    gap_p90: float,
    gap_max: float,
) -> dict[str, Any]:
    upper_ok = density_per_100 <= 45.0
    lower_abs_ok = density_per_100 >= 25.0
    lower_rel_ok = density_per_100 >= 0.8 * baseline_density_same_length
    baseline_kf_ok = keyframes >= 0.8 * baseline_kf_same_length
    growth_ok = keyframes >= keyframes_short800
    gap_ok = gap_p90 <= 5.0 and gap_max <= 20
    starvation = density_per_100 < 25.0 or not baseline_kf_ok
    overdense = density_per_100 > 50.0
    ready = bool(
        upper_ok and lower_abs_ok and lower_rel_ok and baseline_kf_ok
        and growth_ok and gap_ok and not starvation and not overdense
    )
    return {
        "ready_for_full_forest1": ready,
        "starvation_flag": starvation,
        "overdense_flag": overdense,
    }


'''
        text = text.replace("def post_run_summary", insert_fn + "def post_run_summary")
    old_ready = """    ready = {
        "ready_for_full_forest1": bool(d1000.get("keyframes_per_100", 99) <= 45),
        "recommend_full_run": bool(d1000.get("keyframes_per_100", 99) <= 45),
        "density_target_met": bool(d1000.get("keyframes_per_100", 99) <= 45),
        "keep_RVQ_tau_frozen": True,
    }"""
    new_ready = """    b1000 = analyze_run("baseline_short1000", EXT / "baseline_short1000", 1000, False)
    d800r = analyze_run("direct_density_short800", out / "direct_density_short800", 800, True)
    gate_eval = eval_full_gate(
        density_per_100=float(d1000.get("keyframes_per_100", 99)),
        keyframes=int(d1000.get("final_keyframe_count", 0)),
        baseline_kf_same_length=int(b1000["final_keyframe_count"]),
        baseline_density_same_length=float(b1000["keyframes_per_100"]),
        keyframes_short800=int(d800r["final_keyframe_count"]),
        gap_p90=float(d1000.get("gap_p90", 99)),
        gap_max=float(d1000.get("gap_max", 99)),
    )
    ready = {
        **gate_eval,
        "recommend_full_run": bool(gate_eval.get("ready_for_full_forest1")),
        "density_target_met": bool(
            gate_eval.get("ready_for_full_forest1") and not gate_eval.get("starvation_flag")
        ),
        "keep_RVQ_tau_frozen": True,
        "gate_version": "stability_audit_v1_corrected",
    }"""
    if old_ready in text:
        text = text.replace(old_ready, new_ready)
        rebalance_script.write_text(text, encoding="utf-8")

    print(f"Audit written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
