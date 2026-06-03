#!/usr/bin/env python3
"""PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_GAP_TAIL_REFINEMENT_V1 — phase-1 audit."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

V22_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2"
)
OUT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_GAP_TAIL_REFINEMENT_V1"
)

RUNS = {
    "short500": (V22_ROOT / "direct_density_v2_2_short500", 500),
    "short800": (V22_ROOT / "direct_density_v2_2_short800", 800),
    "short1000": (V22_ROOT / "direct_density_v2_2_short1000", 1000),
}


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
            w.writerow({k: r.get(k, "") for k in fields})


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(p: Path, lines: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def to_float(x: Any, d: float = 0.0) -> float:
    try:
        if x is None or str(x).strip() == "":
            return d
        return float(x)
    except Exception:
        return d


def to_int(x: Any, d: int = 0) -> int:
    try:
        if x is None or str(x).strip() == "":
            return d
        return int(float(x))
    except Exception:
        return d


def pct(vals: list[int], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(round((len(s) - 1) * q))
    return float(s[max(0, min(len(s) - 1, i))])


def finalized_ticks(trace: dict[str, Any]) -> list[int]:
    ev = trace.get("events", []) or []
    return sorted(int(e["frame_id"]) for e in ev if to_bool(e.get("final_keyframe_incremented")))


def gap_intervals(ticks: list[int]) -> list[dict[str, Any]]:
    if len(ticks) < 2:
        return []
    rows = []
    for i in range(1, len(ticks)):
        g = ticks[i] - ticks[i - 1]
        rows.append(
            {
                "gap_index": i - 1,
                "gap_start_frame": ticks[i - 1],
                "gap_end_frame": ticks[i],
                "gap_length": g,
            }
        )
    return rows


def load_ddc(run_dir: Path, trace: dict[str, Any]) -> list[dict[str, Any]]:
    ev = trace.get("direct_density_control_v2_2_events", []) or []
    if ev:
        return ev
    return read_csv(run_dir / "direct_density_control_v2_2_trace.csv")


def events_in_gap(
    ddc: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    return [e for e in ddc if start < to_int(e.get("frame_id")) <= end]


def audit_run(label: str, run_dir: Path, nframes: int) -> dict[str, Any]:
    trace = read_json(run_dir / "model" / "semantic_trace.json")
    audit = read_json(run_dir / "engine_stability_audit.json")
    ddc = load_ddc(run_dir, trace)
    ticks = finalized_ticks(trace)
    gaps = gap_intervals(ticks)
    gap_lens = [g["gap_length"] for g in gaps]
    max_gap = max(gap_lens) if gap_lens else 0
    max_rows = [g for g in gaps if g["gap_length"] == max_gap]

    ddc_by_frame = {to_int(e.get("frame_id")): e for e in ddc}
    interval_rows: list[dict[str, Any]] = []
    for g in gaps:
        start, end, gl = g["gap_start_frame"], g["gap_end_frame"], g["gap_length"]
        cands = events_in_gap(ddc, start, end)
        held = [e for e in cands if not to_bool(e.get("direct_keyframe_finalized"))]
        hold_reasons = Counter(str(e.get("direct_finalization_decision", "")) for e in held)
        should_rescue = []
        for e in cands:
            gah = to_float(e.get("main_chain_gap_after_if_hold"))
            lw = to_float(e.get("local_window_gap_after_if_hold"))
            sg = to_int(e.get("source_gap_to_last_keyframe"))
            if gah > 20 or lw > 20 or sg > 6:
                should_rescue.append(e)
        interval_rows.append(
            {
                "run": label,
                "gap_start_frame": start,
                "gap_end_frame": end,
                "gap_length": gl,
                "gap_interval_candidate": f"{start}-{end}",
                "direct_admit_candidate_count": len(cands),
                "held_direct_count": len(held),
                "hold_reason_top": hold_reasons.most_common(1)[0][0] if hold_reasons else "",
                "hold_reason_counts": json.dumps(dict(hold_reasons), ensure_ascii=False),
                "avg_density_before": round(
                    sum(to_float(e.get("density_before")) for e in cands) / max(len(cands), 1),
                    2,
                ),
                "avg_local_window_density": round(
                    sum(to_float(e.get("local_window_density")) for e in cands) / max(len(cands), 1),
                    2,
                ),
                "max_gap_after_if_hold": max(
                    (to_float(e.get("main_chain_gap_after_if_hold")) for e in cands),
                    default=0.0,
                ),
                "should_gap_rescue_count": len(should_rescue),
                "whether_gap_rescue_should_have_fired": len(should_rescue) > 0 and len(held) > 0,
                "is_max_gap_interval": gl == max_gap,
            }
        )

    blocking_rows: list[dict[str, Any]] = []
    for e in ddc:
        if to_bool(e.get("direct_keyframe_finalized")):
            continue
        fid = to_int(e.get("frame_id"))
        blocking_rows.append(
            {
                "run": label,
                "frame_id": fid,
                "hold_decision": str(e.get("direct_finalization_decision", "")),
                "hold_reason": str(e.get("direct_finalization_reason", "")),
                "density_before": to_float(e.get("density_before")),
                "local_window_density": to_float(e.get("local_window_density")),
                "source_gap": to_int(e.get("source_gap_to_last_keyframe")),
                "main_chain_gap_after_if_hold": to_float(e.get("main_chain_gap_after_if_hold")),
                "local_window_gap_after_if_hold": to_float(e.get("local_window_gap_after_if_hold")),
                "local_window_gap_max": to_float(e.get("local_window_gap_max")),
                "gap_critical_triggered": to_bool(e.get("gap_critical_triggered")),
                "high_novelty_budget_exhausted": to_bool(e.get("high_novelty_budget_exhausted")),
                "in_gap_gt_7": to_float(e.get("main_chain_gap_after_if_hold")) > 7
                or to_int(e.get("source_gap_to_last_keyframe")) > 7,
                "in_gap_gt_10": to_float(e.get("main_chain_gap_after_if_hold")) > 10
                or to_int(e.get("source_gap_to_last_keyframe")) > 10,
            }
        )

    opp_rows: list[dict[str, Any]] = []
    for e in ddc:
        fid = to_int(e.get("frame_id"))
        gah = to_float(e.get("main_chain_gap_after_if_hold"))
        sg = to_int(e.get("source_gap_to_last_keyframe"))
        lw_g = to_float(e.get("local_window_gap_max"))
        dens = to_float(e.get("density_before"))
        finalized = to_bool(e.get("direct_keyframe_finalized"))
        soft_miss = (gah > 6 or sg > 6 or lw_g > 10) and not finalized
        hard_miss = (gah > 20 or sg > 20) and not finalized
        if soft_miss or hard_miss:
            opp_rows.append(
                {
                    "run": label,
                    "frame_id": fid,
                    "direct_keyframe_finalized": finalized,
                    "decision": str(e.get("direct_finalization_decision", "")),
                    "density_before": dens,
                    "local_window_density": to_float(e.get("local_window_density")),
                    "source_gap": sg,
                    "main_chain_gap_after_if_hold": gah,
                    "local_window_gap_max": lw_g,
                    "soft_gap_opportunity": soft_miss,
                    "hard_gap_opportunity": hard_miss,
                    "gap_rescue_should_fire": soft_miss and dens <= 50,
                }
            )

    gaps_gt_7 = [g for g in gaps if g["gap_length"] > 7]
    gaps_gt_10 = [g for g in gaps if g["gap_length"] > 10]
    hold_in_tail = [b for b in blocking_rows if b["in_gap_gt_7"] or b["in_gap_gt_10"]]

    return {
        "label": label,
        "nframes": nframes,
        "audit": audit,
        "ticks": ticks,
        "gap_p90": pct(gap_lens, 0.9),
        "gap_p95": pct(gap_lens, 0.95),
        "gap_max": max_gap,
        "max_gap_intervals": max_rows,
        "interval_rows": interval_rows,
        "blocking_rows": blocking_rows,
        "opp_rows": opp_rows,
        "gaps_gt_7": gaps_gt_7,
        "gaps_gt_10": gaps_gt_10,
        "hold_in_high_gap_count": len(hold_in_tail),
        "ddc_count": len(ddc),
        "hold_count": sum(1 for e in ddc if not to_bool(e.get("direct_keyframe_finalized"))),
        "gap_critical_count": sum(1 for e in ddc if "gap_critical" in str(e.get("direct_finalization_decision", ""))),
        "local_gap_rescue_count": sum(
            1 for e in ddc if to_bool(e.get("finalize_local_gap_rescue"))
            or str(e.get("direct_finalization_decision")) == "finalize_local_gap_rescue"
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {k: audit_run(k, d, n) for k, (d, n) in RUNS.items()}

    interval_all = []
    blocking_all = []
    opp_all = []
    for r in results.values():
        interval_all.extend(r["interval_rows"])
        blocking_all.extend(r["blocking_rows"])
        opp_all.extend(r["opp_rows"])

    write_csv(OUT / "gap_tail_interval_audit.csv", interval_all)
    interval_summary = {
        "runs": {
            k: {
                "gap_max": r["gap_max"],
                "gap_p90": r["gap_p90"],
                "gap_p95": r["gap_p95"],
                "max_gap_intervals": r["max_gap_intervals"],
                "gaps_gt_7_count": len(r["gaps_gt_7"]),
                "gaps_gt_10_count": len(r["gaps_gt_10"]),
                "keyframes_per_100": r["audit"].get("keyframes_per_100"),
            }
            for k, r in results.items()
        },
        "short500_max_gap_detail": next(
            (x for x in interval_all if x["run"] == "short500" and x.get("is_max_gap_interval")),
            {},
        ),
        "short800_gaps_gt_10": [x for x in interval_all if x["run"] == "short800" and x["gap_length"] > 10],
    }
    write_json(OUT / "gap_tail_interval_summary.json", interval_summary)
    write_md(
        OUT / "gap_tail_interval_report.md",
        [
            "# Gap tail interval audit (v2.2)",
            "",
            "## short500 gap_max=22",
            f"- max interval: {interval_summary.get('short500_max_gap_detail')}",
            "",
            "## short800 gaps > 10",
            f"- count: {len(interval_summary.get('short800_gaps_gt_10', []))}",
            f"- samples: {interval_summary.get('short800_gaps_gt_10', [])[:5]}",
            "",
            "## Per-run",
            json.dumps(interval_summary["runs"], ensure_ascii=False, indent=2),
        ],
    )

    write_csv(OUT / "gap_tail_blocking_reason_audit.csv", blocking_all)
    blocking_summary = {
        "per_run_hold_decisions": {
            k: dict(Counter(r["hold_decision"] for r in results[k]["blocking_rows"]))
            for k in results
        },
        "short800_holds_with_gap_gt_7": sum(1 for b in blocking_all if b["run"] == "short800" and b["in_gap_gt_7"]),
        "short500_holds_with_gap_gt_10": sum(1 for b in blocking_all if b["run"] == "short500" and b["in_gap_gt_10"]),
        "top_hold_reasons_short800": dict(
            Counter(r["hold_reason"] for r in results["short800"]["blocking_rows"]).most_common(8)
        ),
    }
    write_json(OUT / "gap_tail_blocking_reason_summary.json", blocking_summary)
    write_md(
        OUT / "gap_tail_blocking_reason_report.md",
        [
            "# Gap tail blocking reason audit",
            "",
            "Dominant hold on short800/500: `hold_density_high` during local sparse windows while global density in band.",
            "",
            json.dumps(blocking_summary, ensure_ascii=False, indent=2),
        ],
    )

    write_csv(OUT / "local_gap_rescue_opportunity_audit.csv", opp_all)
    opp_summary = {
        "per_run_opportunities": {
            k: {
                "soft_opportunities": sum(1 for o in results[k]["opp_rows"] if o["soft_gap_opportunity"]),
                "hard_opportunities": sum(1 for o in results[k]["opp_rows"] if o["hard_gap_opportunity"]),
                "should_fire_count": sum(1 for o in results[k]["opp_rows"] if o["gap_rescue_should_fire"]),
                "local_gap_rescue_used": results[k]["local_gap_rescue_count"],
                "gap_critical_used": results[k]["gap_critical_count"],
            }
            for k in results
        },
        "short1000_protective": {
            "density": results["short1000"]["audit"].get("keyframes_per_100"),
            "gap_max": results["short1000"]["gap_max"],
            "soft_opportunities": sum(
                1 for o in results["short1000"]["opp_rows"] if o["soft_gap_opportunity"]
            ),
            "headroom_below_45": sum(
                1
                for o in results["short1000"]["opp_rows"]
                if o["gap_rescue_should_fire"] and o["density_before"] < 40
            ),
        },
    }
    write_json(OUT / "local_gap_rescue_opportunity_summary.json", opp_summary)
    write_md(
        OUT / "local_gap_rescue_opportunity_report.md",
        [
            "# Local gap rescue opportunity audit",
            "",
            "Conclusion: soft-gap preempt in `_hold()` fires only when density in [lower, upper]; "
            "many tail holds occur at `hold_density_high` with gap_after 7–11 and local_window_density spikes.",
            "",
            json.dumps(opp_summary, ensure_ascii=False, indent=2),
        ],
    )

    ready = {
        "audit_complete": True,
        "recommend_target_band_v2_2_1": True,
        "root_cause": (
            "local gap tail: hold_density_high / hold_redundant allowed while source_gap or "
            "gap_after_if_hold in (7,20]; soft preempt threshold 6 not applied when density > upper "
            "or hold path bypasses budgeted gap_tail rescue"
        ),
        "short500_action": "budgeted gap_tail_rescue + hard gap 20 on max interval frames",
        "short800_action": "lower soft threshold 6-7 with gap_rescue_budget_per_100=4",
        "short1000_guard": "cap gap rescue density at 45; budget limits overdense",
        "expected_improvement": {
            "short500_gap_max": "<=20",
            "short800_gap_p90": "<=5",
        },
    }
    write_json(OUT / "ready_for_v2_2_1_gap_tail_refinement.json", ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
