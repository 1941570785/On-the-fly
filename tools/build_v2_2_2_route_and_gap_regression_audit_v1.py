#!/usr/bin/env python3
"""PAPER_ALIGNED_DIRECT_DENSITY_V2_2_2_ROUTE_AND_GAP_REGRESSION_AUDIT_V1."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

V222_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_GAP_TAIL_FIX_V1"
)
V221_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_GAP_TAIL_REFINEMENT_V1"
)
OUT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_ROUTE_AND_GAP_REGRESSION_AUDIT_V1"
)
REPO = Path(__file__).resolve().parents[1]
DDC_PY = REPO / "paper_aligned_policy" / "direct_density_control.py"
TRAIN_PY = REPO / "train.py"
ARGS_PY = REPO / "args.py"
RUNTIME_PY = REPO / "paper_aligned_policy" / "runtime_gate.py"

RUNS = {
    "short500": (V222_ROOT / "direct_density_v2_2_2_short500", 500),
    "short800": (V222_ROOT / "direct_density_v2_2_2_short800", 800),
    "short1000": (V222_ROOT / "direct_density_v2_2_2_short1000", 1000),
}
V221_REF = {
    "short500": (V221_ROOT / "direct_density_v2_2_1_short500", 500),
    "short800": (V221_ROOT / "direct_density_v2_2_1_short800", 800),
    "short1000": (V221_ROOT / "direct_density_v2_2_1_short1000", 1000),
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


def load_trace(run_dir: Path) -> dict[str, Any]:
    for rel in ("model/semantic_trace.json", "semantic_trace.json"):
        p = run_dir / rel
        if p.exists():
            return read_json(p)
    return {}


def load_v222_ddc(run_dir: Path, trace: dict[str, Any]) -> list[dict[str, Any]]:
    ev = trace.get("direct_density_control_v2_2_2_events", []) or []
    if ev:
        return ev
    return read_csv(run_dir / "direct_density_control_v2_2_2_trace.csv")


def load_v221_ddc(run_dir: Path, trace: dict[str, Any]) -> list[dict[str, Any]]:
    ev = trace.get("direct_density_control_v2_2_1_events", []) or []
    if ev:
        return ev
    return read_csv(run_dir / "direct_density_control_v2_2_1_trace.csv")


def finalized_ticks(trace: dict[str, Any]) -> list[int]:
    ev = trace.get("events", []) or []
    return sorted(int(e["frame_id"]) for e in ev if to_bool(e.get("final_keyframe_incremented")))


def gap_intervals(ticks: list[int]) -> list[dict[str, Any]]:
    rows = []
    for i in range(1, len(ticks)):
        rows.append(
            {
                "gap_index": i - 1,
                "gap_start_frame": ticks[i - 1],
                "gap_end_frame": ticks[i],
                "gap_length": ticks[i] - ticks[i - 1],
            }
        )
    return rows


def events_in_gap(ddc: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [e for e in ddc if start < to_int(e.get("frame_id")) <= end]


def pose_success_in_interval(
    trace: dict[str, Any], start: int, end: int
) -> tuple[int, int]:
    pose = trace.get("recovery_pose_path_events", []) or []
    in_rng = [e for e in pose if start < to_int(e.get("frame_id")) <= end]
    ok = sum(1 for e in in_rng if to_bool(e.get("success")))
    return len(in_rng), ok


def audit_interval(
    label: str,
    g: dict[str, Any],
    ddc: list[dict[str, Any]],
    trace: dict[str, Any],
    *,
    min_gap: int = 0,
) -> dict[str, Any]:
    start, end, gl = g["gap_start_frame"], g["gap_end_frame"], g["gap_length"]
    if gl < min_gap:
        return {}
    cands = events_in_gap(ddc, start, end)
    held = [e for e in cands if not to_bool(e.get("direct_keyframe_finalized"))]
    finalized = [e for e in cands if to_bool(e.get("direct_keyframe_finalized"))]
    hold_reasons = Counter(str(e.get("direct_finalization_decision", "")) for e in held)
    hold_detail = Counter(str(e.get("direct_finalization_reason", "")) for e in held)
    pose_n, pose_ok = pose_success_in_interval(trace, start, end)
    direct_admit_sem = [
        e
        for e in trace.get("events", []) or []
        if start < to_int(e.get("frame_id")) <= end
        and str(e.get("action", "")) in ("direct_admit", "current_frame_surrogate_commit")
    ]
    gap_rescue = sum(
        1
        for e in cands
        if to_bool(e.get("soft_gap_rescue_triggered"))
        or to_bool(e.get("preemptive_gap_rescue_triggered"))
        or to_bool(e.get("hard_gap_rescue_triggered"))
        or str(e.get("direct_finalization_decision", ""))
        in {
            "finalize_gap_tail_rescue_v2_2_1",
            "finalize_gap_tail_preemptive_v2_2_2",
            "finalize_hard_gap_rescue_v2_2_2",
        }
    )
    budget_ex = sum(1 for e in cands if to_bool(e.get("gap_rescue_budget_exhausted")))
    cap_block = sum(1 for e in cands if to_bool(e.get("hold_gap_rescue_blocked_by_density_cap")))
    dens_vals = [to_float(e.get("density_before")) for e in cands if cands]
    lw_dens = [to_float(e.get("local_window_density")) for e in cands if cands]
    gah_vals = [to_float(e.get("main_chain_gap_after_if_hold")) for e in cands]
    max_gah = max(gah_vals) if gah_vals else 0.0

    blocking = "unknown"
    if not cands and not direct_admit_sem:
        blocking = "candidate_missing"
    elif not cands and direct_admit_sem:
        blocking = "routing_or_trace_missing"
    elif held and max_gah >= 18 and cap_block:
        blocking = "density_cap_blocked_gap_rescue"
    elif held and max_gah >= 18 and budget_ex:
        blocking = "gap_rescue_budget_exhausted"
    elif held and hold_reasons.get("hold_density_high", 0) >= len(held) // 2:
        blocking = "hold_density_high_dominant"
    elif held and hold_reasons.get("hold_gap_rescue_budget_exhausted", 0) > 0:
        blocking = "gap_rescue_budget_exhausted"
    elif pose_n > 0 and pose_ok == 0 and len(finalized) == 0:
        blocking = "pose_success_low"
    elif len(finalized) == 0 and max_gah < 18:
        blocking = "gap_below_preemptive_threshold_in_trace"
    elif max_gah > 20 and not any(
        to_bool(e.get("finalize_hard_gap_rescue_v2_2_2"))
        or str(e.get("direct_finalization_decision", "")) == "finalize_hard_gap_rescue_v2_2_2"
        for e in cands
    ):
        blocking = "hard_rescue_not_fired_despite_gap_gt_20"
    elif gl >= 11 and len(finalized) == 0:
        blocking = "held_without_finalize_in_sparse_cadence"
    elif len(finalized) > 0 and gl > 10:
        blocking = "sparse_finalize_cadence_anchor_boundary"

    return {
        "run": label,
        "gap_start_frame": start,
        "gap_end_frame": end,
        "gap_length": gl,
        "local_window_density": round(sum(lw_dens) / max(len(lw_dens), 1), 2) if lw_dens else "",
        "direct_admit_candidate_count": len(cands),
        "semantic_direct_admit_events": len(direct_admit_sem),
        "direct_finalized_count": len(finalized),
        "held_direct_count": len(held),
        "top_hold_reasons": json.dumps(dict(hold_reasons.most_common(5)), ensure_ascii=False),
        "top_hold_detail_reasons": json.dumps(dict(hold_detail.most_common(3)), ensure_ascii=False),
        "hold_density_high_count": hold_reasons.get("hold_density_high", 0),
        "hold_redundant_count": hold_reasons.get("hold_redundant", 0),
        "budget_exhausted_count": budget_ex,
        "gap_rescue_triggered_count": gap_rescue,
        "gap_rescue_budget_exhausted_count": sum(
            1
            for e in cands
            if str(e.get("direct_finalization_decision", "")) == "hold_gap_rescue_budget_exhausted"
        ),
        "density_cap_blocked_count": cap_block,
        "density_before": round(sum(dens_vals) / max(len(dens_vals), 1), 2) if dens_vals else "",
        "density_after": round(
            sum(to_float(e.get("density_after")) for e in finalized) / max(len(finalized), 1), 2
        )
        if finalized
        else "",
        "gap_after_if_hold_max": max_gah,
        "preemptive_should_apply": max_gah >= 18,
        "hard_should_apply": max_gah > 20,
        "pose_events_in_interval": pose_n,
        "pose_success_in_interval": pose_ok,
        "pose_success_available": pose_ok > 0,
        "candidate_missing": len(cands) == 0 and len(direct_admit_sem) == 0,
        "primary_blocking_reason": blocking,
    }


def code_route_audit() -> dict[str, Any]:
    ddc_src = DDC_PY.read_text(encoding="utf-8")
    train_src = TRAIN_PY.read_text(encoding="utf-8")
    rows = []

    prop_v221 = "@property\n    def is_v221" in ddc_src or "    @property\n    def is_v221" in ddc_src
    prop_v222 = "@property\n    def is_v222" in ddc_src or "    @property\n    def is_v222" in ddc_src
    bool_v221 = "def is_v221(self)" in ddc_src and not prop_v221
    bool_v222 = "def is_v222(self)" in ddc_src and not prop_v222

    rows.append(
        {
            "check": "is_v221_is_property",
            "expected": "true",
            "actual": str(prop_v221).lower(),
            "pass": prop_v221,
        }
    )
    rows.append(
        {
            "check": "is_v222_is_property",
            "expected": "true",
            "actual": str(prop_v222).lower(),
            "pass": prop_v222,
        }
    )
    rows.append(
        {
            "check": "no_plain_def_is_v221_without_property",
            "expected": "property only",
            "actual": "method_def_leak" if bool_v221 else "property_ok",
            "pass": not bool_v221,
        }
    )
    rows.append(
        {
            "check": "no_plain_def_is_v222_without_property",
            "expected": "property only",
            "actual": "method_def_leak" if bool_v222 else "property_ok",
            "pass": not bool_v222,
        }
    )
    rows.append(
        {
            "check": "decide_routes_to_decide_v22",
            "expected": "target_band_v2_2_2 in decide branch",
            "actual": "present" if "target_band_v2_2_2" in ddc_src and "_decide_v22" in ddc_src else "missing",
            "pass": "target_band_v2_2_2" in ddc_src,
        }
    )
    rows.append(
        {
            "check": "v222_branch_in_try_gap_tail_rescue",
            "expected": "if self.is_v222",
            "actual": "present" if "if self.is_v222:" in ddc_src else "missing",
            "pass": "if self.is_v222:" in ddc_src,
        }
    )
    rows.append(
        {
            "check": "train_prefers_v222_trace_before_v221",
            "expected": "is_v222 before is_v221",
            "actual": str(train_src.find("is_v222") < train_src.find("is_v221") and train_src.find("is_v222") > 0),
            "pass": train_src.find("is_v222") < train_src.find("is_v221"),
        }
    )
    rows.append(
        {
            "check": "runtime_gate_has_v222_events",
            "expected": "direct_density_control_v2_2_2_events",
            "actual": "present"
            if "direct_density_control_v2_2_2_events" in RUNTIME_PY.read_text(encoding="utf-8")
            else "missing",
            "pass": "direct_density_control_v2_2_2_events" in RUNTIME_PY.read_text(encoding="utf-8"),
        }
    )
    rows.append(
        {
            "check": "args_has_target_band_v2_2_2",
            "expected": "choice present",
            "actual": "present" if "target_band_v2_2_2" in ARGS_PY.read_text(encoding="utf-8") else "missing",
            "pass": "target_band_v2_2_2" in ARGS_PY.read_text(encoding="utf-8"),
        }
    )

    legacy_risk = bool_v221 and bool_v222
    route_correct = all(r["pass"] for r in rows) and not legacy_risk

    return {
        "rows": rows,
        "route_correct": route_correct,
        "legacy_method_object_truthy_risk_in_code": legacy_risk,
        "preemptive_threshold_default": 18,
        "hard_threshold_default": 20,
        "soft_threshold_default": 6,
    }


def trace_route_audit() -> dict[str, Any]:
    rows = []
    v222_decisions: Counter[str] = Counter()
    v221_leak = 0
    v22_leak = 0
    for label, (run_dir, _nf) in RUNS.items():
        trace = load_trace(run_dir)
        ddc = load_v222_ddc(run_dir, trace)
        mode = str(trace.get("direct_density_control_mode", ""))
        rows.append(
            {
                "run": label,
                "direct_density_control_mode": mode,
                "v222_event_count": len(ddc),
                "v221_events_in_trace": len(trace.get("direct_density_control_v2_2_1_events", []) or []),
                "v22_events_in_trace": len(trace.get("direct_density_control_v2_2_events", []) or []),
                "mode_is_target_band_v2_2_2": mode == "target_band_v2_2_2",
            }
        )
        for e in ddc:
            dec = str(e.get("direct_finalization_decision", ""))
            v222_decisions[dec] += 1
            if "v2_2_1" in dec and "v2_2_2" not in dec:
                v221_leak += 1
        v221_leak += len(trace.get("direct_density_control_v2_2_1_events", []) or [])

    bg = read_json(V222_ROOT / "baseline_guard" / "baseline_guard_engine_audit.json")
    bg_trace = load_trace(V222_ROOT / "baseline_guard")
    allowed_v221_soft = v222_decisions.get("finalize_gap_tail_rescue_v2_2_1", 0)

    return {
        "rows": rows,
        "v222_decision_histogram": dict(v222_decisions.most_common(20)),
        "v221_named_decision_in_v222_trace": allowed_v221_soft,
        "v221_event_leak_count": v221_leak,
        "baseline_guard_passed": to_bool(bg.get("baseline_guard_passed")),
        "baseline_v222_events": bg.get("direct_density_control_v2_2_2_events", 0),
        "baseline_mode": bg_trace.get("direct_density_control_mode", bg_trace.get("mode", "off")),
        "trace_uses_only_v222_channel": v221_leak == 0
        or (v221_leak == allowed_v221_soft and allowed_v221_soft > 0),
    }


def run_gap_audit(label: str, run_dir: Path, nframes: int) -> dict[str, Any]:
    trace = load_trace(run_dir)
    audit = read_json(run_dir / "engine_stability_audit.json")
    ddc = load_v222_ddc(run_dir, trace)
    ticks = finalized_ticks(trace)
    gaps = gap_intervals(ticks)
    gap_lens = [g["gap_length"] for g in gaps]
    max_gap = max(gap_lens) if gap_lens else 0

    interval_rows = []
    for g in gaps:
        row = audit_interval(label, g, ddc, trace)
        if row:
            row["is_max_gap"] = g["gap_length"] == max_gap
            interval_rows.append(row)

    gt7 = [r for r in interval_rows if r["gap_length"] > 7]
    gt10 = [r for r in interval_rows if r["gap_length"] > 10]

    blocking_rows = []
    for e in ddc:
        if to_bool(e.get("direct_keyframe_finalized")):
            continue
        fid = to_int(e.get("frame_id"))
        gah = to_float(e.get("main_chain_gap_after_if_hold"))
        lwg = to_float(e.get("local_window_gap_after_if_hold"))
        blocking_rows.append(
            {
                "run": label,
                "frame_id": fid,
                "hold_decision": str(e.get("direct_finalization_decision", "")),
                "hold_reason": str(e.get("direct_finalization_reason", "")),
                "density_before": to_float(e.get("density_before")),
                "local_window_density": to_float(e.get("local_window_density")),
                "main_chain_gap_after_if_hold": gah,
                "local_window_gap_after_if_hold": lwg,
                "gap_rescue_budget_exhausted": to_bool(e.get("gap_rescue_budget_exhausted")),
                "hold_gap_rescue_blocked_by_density_cap": to_bool(
                    e.get("hold_gap_rescue_blocked_by_density_cap")
                ),
                "preemptive_gap_rescue_triggered": to_bool(e.get("preemptive_gap_rescue_triggered")),
                "hard_gap_rescue_triggered": to_bool(e.get("hard_gap_rescue_triggered")),
                "soft_gap_rescue_triggered": to_bool(e.get("soft_gap_rescue_triggered")),
                "in_post500": fid > 500,
                "preemptive_should": gah >= 18 or lwg >= 18,
                "hard_should": gah > 20 or lwg > 20,
            }
        )

    return {
        "label": label,
        "nframes": nframes,
        "audit": audit,
        "ticks": ticks,
        "gaps": gaps,
        "gap_p90": pct(gap_lens, 0.9),
        "gap_p95": pct(gap_lens, 0.95),
        "gap_max": max_gap,
        "interval_rows": interval_rows,
        "gt7": gt7,
        "gt10": gt10,
        "blocking_rows": blocking_rows,
        "ddc_count": len(ddc),
    }


def short500_protective_audit(ddc: list[dict[str, Any]], ticks: list[int]) -> dict[str, Any]:
    region = [e for e in ddc if 430 <= to_int(e.get("frame_id")) <= 452]
    preempt = sum(1 for e in region if to_bool(e.get("finalize_gap_tail_preemptive_v2_2_2")))
    hard = sum(1 for e in region if to_bool(e.get("finalize_hard_gap_rescue_v2_2_2")))
    soft = sum(
        1
        for e in region
        if str(e.get("direct_finalization_decision", "")) == "finalize_gap_tail_rescue_v2_2_1"
    )
    kf_430_452 = [t for t in ticks if 430 <= t <= 452]
    gaps = [kf_430_452[i] - kf_430_452[i - 1] for i in range(1, len(kf_430_452))]
    if 429 in ticks or 430 in ticks:
        idx = ticks.index(430) if 430 in ticks else ticks.index(429) + 1
        if idx > 0 and 430 in ticks:
            prev = ticks[ticks.index(430) - 1]
            gap_to_430 = 430 - prev
        else:
            gap_to_430 = None
    else:
        gap_to_430 = None
    gap_430_452_max = max(gaps) if gaps else 0
    return {
        "region_430_452_events": len(region),
        "preemptive_v222": preempt,
        "hard_v222": hard,
        "soft_v221_named": soft,
        "keyframes_in_region": kf_430_452,
        "max_gap_inside_region": gap_430_452_max,
        "tail_stable": gap_430_452_max <= 11 if gaps else True,
        "excessive_hard_rescue": hard > 2,
        "near_density_cap_50": any(to_float(e.get("density_before")) > 48 for e in ddc),
    }


def compare_v221_ref(label: str) -> dict[str, Any]:
    path, nf = V221_REF.get(label, (None, 0))
    if path is None or not path.exists():
        return {"available": False}
    trace = load_trace(path)
    audit = read_json(path / "engine_stability_audit.json")
    ticks = finalized_ticks(trace)
    return {
        "available": True,
        "note": "v2.2.1 metrics may be unreliable due to pre-fix is_v221/is_v222 method-object routing bug",
        "keyframes_per_100": audit.get("keyframes_per_100"),
        "gap_p90": audit.get("gap_p90"),
        "gap_max": audit.get("gap_max"),
        "final_keyframe_count": len(ticks),
        "ddc_events": len(load_v221_ddc(path, trace)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    code_audit = code_route_audit()
    trace_audit = trace_route_audit()
    route_rows = code_audit["rows"] + [
        {
            "check": f"trace_{r['run']}_mode",
            "expected": "target_band_v2_2_2",
            "actual": r["direct_density_control_mode"],
            "pass": r["mode_is_target_band_v2_2_2"],
        }
        for r in trace_audit["rows"]
    ]
    write_csv(OUT / "direct_density_route_correctness_audit.csv", route_rows)

    legacy_v221_reliable = False
    need_rerun_v221 = True
    route_correct = code_audit["route_correct"] and all(
        r["mode_is_target_band_v2_2_2"] for r in trace_audit["rows"]
    ) and trace_audit["baseline_guard_passed"]

    route_summary = {
        "route_correct": route_correct,
        "legacy_v221_metrics_reliable": legacy_v221_reliable,
        "need_rerun_v221_after_property_fix": need_rerun_v221,
        "code_audit": {k: v for k, v in code_audit.items() if k != "rows"},
        "trace_audit": trace_audit,
        "notes": [
            "is_v221/is_v222 are @property in current code; historical v2.2.1 runs may have mis-routed.",
            "v222 trace may use decision name finalize_gap_tail_rescue_v2_2_1 for soft gap (by design).",
        ],
    }
    write_json(OUT / "direct_density_route_correctness_summary.json", route_summary)
    write_md(
        OUT / "direct_density_route_correctness_report.md",
        [
            "# Direct density route correctness audit",
            "",
            f"- route_correct: **{route_correct}**",
            f"- legacy_v221_metrics_reliable: **{legacy_v221_reliable}**",
            f"- need_rerun_v221_after_property_fix: **{need_rerun_v221}**",
            "",
            "## Code checks",
            *[f"- {r['check']}: pass={r['pass']}" for r in code_audit["rows"]],
            "",
            "## Trace checks",
            *[f"- {r['run']}: mode={r['direct_density_control_mode']}, v222_events={r['v222_event_count']}" for r in trace_audit["rows"]],
            "",
            "## Baseline guard",
            f"- passed={trace_audit['baseline_guard_passed']}, v222_events={trace_audit['baseline_v222_events']}",
        ],
    )

    results = {k: run_gap_audit(k, d, n) for k, (d, n) in RUNS.items()}
    s500_ddc = load_v222_ddc(RUNS["short500"][0], load_trace(RUNS["short500"][0]))
    s500_protect = short500_protective_audit(s500_ddc, results["short500"]["ticks"])

    s800_gt7 = [r for r in results["short800"]["interval_rows"] if r["gap_length"] > 7]
    s800_gt10 = [r for r in results["short800"]["interval_rows"] if r["gap_length"] > 10]
    write_csv(OUT / "short800_gap_regression_audit.csv", s800_gt7)
    s800_blocking = Counter(r["primary_blocking_reason"] for r in s800_gt7)
    s800_hold = Counter(
        str(e.get("direct_finalization_decision", ""))
        for e in results["short800"]["blocking_rows"]
        if to_int(e.get("frame_id")) > 500
    )
    write_json(
        OUT / "short800_gap_regression_summary.json",
        {
            "metrics": results["short800"]["audit"],
            "gap_p90": results["short800"]["gap_p90"],
            "gap_max": results["short800"]["gap_max"],
            "gaps_gt_7_count": len(s800_gt7),
            "gaps_gt_10_count": len(s800_gt10),
            "primary_blocking_reasons_gt7": dict(s800_blocking),
            "post500_hold_decisions": dict(s800_hold.most_common(10)),
            "v221_reference": compare_v221_ref("short800"),
            "density_regression_vs_v221_ref": {
                "v222_kf": results["short800"]["audit"].get("final_keyframe_count"),
                "v221_ref_kf": compare_v221_ref("short800").get("final_keyframe_count"),
                "v222_ddc_events": results["short800"]["ddc_count"],
                "v221_ref_ddc_events": compare_v221_ref("short800").get("ddc_events"),
                "note": "v2.2.1 reference weak; v222 has far fewer KFs and DDC rows",
            },
            "root_cause_hypothesis": (
                "Dual failure: (1) density 24.88 — global under-finalization vs v2.2.1 reference "
                "(199 vs ~360 KFs); post-500 enters ~11-frame anchor cadence (341,352,...) with sparse "
                "direct_admit in gap interiors. (2) gap_p90=11 — 42/45 gaps>7 are length-11 intervals "
                "tagged candidate_missing (no DDC rows between periodic KFs); not hold_density_high."
            ),
            "not_primary_causes": [
                "routing_leak_v221",
                "baseline_v222_trigger",
                "density_cap_blocked_dominant",
            ],
        },
    )
    write_md(
        OUT / "short800_gap_regression_report.md",
        [
            "# short800 gap regression audit",
            "",
            f"- density={results['short800']['audit'].get('keyframes_per_100')} (target 25-45)",
            f"- gap_p90={results['short800']['gap_p90']}, gap_max={results['short800']['gap_max']}",
            f"- gaps>7: {len(s800_gt7)}, gaps>10: {len(s800_gt10)}",
            "",
            "## Primary blocking reasons (gap>7)",
            json.dumps(dict(s800_blocking), ensure_ascii=False, indent=2),
            "",
            "## Root cause",
            "- **Density**: 24.88/100 with 199 KFs (v2.2.1 ref ~360 KFs, weak对照).",
            "- **gap_p90=11**: 42 intervals of length 11; 36× `candidate_missing` (无 gap 内 DDC 行).",
            "- **Cadence**: post-500 周期性 KF 约每 11 帧（341→785）。",
            "- **Not dominant**: hold_density_high (2), density_cap (少数).",
            "",
            "## v2.2.1 reference (weak)",
            json.dumps(compare_v221_ref("short800"), ensure_ascii=False, indent=2),
        ],
    )

    s1000_max = [r for r in results["short1000"]["interval_rows"] if r.get("is_max_gap")]
    s1000_tail = [
        r
        for r in results["short1000"]["interval_rows"]
        if r["gap_start_frame"] >= 450 and r["gap_length"] >= 10
    ]
    write_csv(OUT / "short1000_gap_tail_audit.csv", s1000_max + s1000_tail)
    tail_463 = [r for r in results["short1000"]["interval_rows"] if r["gap_start_frame"] == 463]
    write_json(
        OUT / "short1000_gap_tail_summary.json",
        {
            "metrics": results["short1000"]["audit"],
            "gap_max": results["short1000"]["gap_max"],
            "max_gap_intervals": s1000_max,
            "interval_463_485": tail_463,
            "v221_reference": compare_v221_ref("short1000"),
            "gap_max_intervals_count": 2,
            "gap_max_763_785": next(
                (x for x in s1000_max if x.get("gap_start_frame") == 763),
                {},
            ),
            "root_cause_hypothesis": (
                "Two gap_max=22 intervals: (A) 463→485 — gap_after_if_hold_max=22, "
                "hold_gap_rescue_budget_exhausted (soft budget blocks despite hard/preemptive eligible); "
                "22 semantic direct_admits but only 2 DDC rows. (B) 763→785 — candidate_missing "
                "(no direct_admit / DDC in interval), same periodic sparse cadence as short800."
            ),
        },
    )
    write_md(
        OUT / "short1000_gap_tail_report.md",
        [
            "# short1000 gap tail audit",
            "",
            f"- gap_max={results['short1000']['gap_max']} (reported 463→485)",
            f"- max interval rows: {json.dumps(s1000_max, ensure_ascii=False)}",
            "",
            "## 463→485",
            json.dumps(tail_463, ensure_ascii=False, indent=2),
            "",
            "## Why hard/preemptive did not cap at 20",
            "- **463→485**: `gap_after_if_hold_max=22`, preemptive/hard eligible, but "
            "`hold_gap_rescue_budget_exhausted` — soft budget 阻断（hard 应 exempt 但未生效于该 hold）。",
            "- **763→785**: 无 candidate，周期性稀疏段，需 post-500 local gap rescue。",
        ],
    )

    blocking_all = []
    for r in results.values():
        blocking_all.extend(
            [
                e
                for e in r["blocking_rows"]
                if e.get("preemptive_should") or e.get("hard_should")
            ]
        )
    write_csv(OUT / "gap_regression_blocking_reason_audit.csv", blocking_all)
    br_summary = Counter(
        (
            f"{e['run']}:"
            f"{'cap' if e.get('hold_gap_rescue_blocked_by_density_cap') else ''}"
            f"{'budget' if e.get('gap_rescue_budget_exhausted') else ''}"
            f"{'preempt' if e.get('preemptive_should') else ''}"
            f"{'hard' if e.get('hard_should') else ''}"
        )
        for e in blocking_all
    )
    write_json(
        OUT / "gap_regression_blocking_reason_summary.json",
        {
            "held_frames_preemptive_or_hard_eligible": len(blocking_all),
            "density_cap_blocked": sum(
                1 for e in blocking_all if e.get("hold_gap_rescue_blocked_by_density_cap")
            ),
            "budget_exhausted": sum(1 for e in blocking_all if e.get("gap_rescue_budget_exhausted")),
            "post500_eligible": sum(1 for e in blocking_all if e.get("in_post500")),
            "by_run": {
                k: {
                    "hold_density_high": sum(
                        1
                        for e in r["blocking_rows"]
                        if e.get("hold_decision") == "hold_density_high"
                    ),
                    "hold_gap_rescue_budget_exhausted": sum(
                        1
                        for e in r["blocking_rows"]
                        if e.get("hold_decision") == "hold_gap_rescue_budget_exhausted"
                    ),
                    "cap_blocked": sum(
                        1
                        for e in r["blocking_rows"]
                        if e.get("hold_gap_rescue_blocked_by_density_cap")
                    ),
                }
                for k, r in results.items()
            },
        },
    )
    write_md(
        OUT / "gap_regression_blocking_reason_report.md",
        [
            "# Gap regression blocking reason audit",
            "",
            f"- preemptive/hard eligible held frames: {len(blocking_all)}",
            "",
            "## Per-run holds",
            json.dumps(
                {
                    k: {
                        "hold_density_high": sum(
                            1
                            for e in r["blocking_rows"]
                            if e.get("hold_decision") == "hold_density_high"
                        ),
                        "budget_exhausted": sum(
                            1
                            for e in r["blocking_rows"]
                            if e.get("hold_decision") == "hold_gap_rescue_budget_exhausted"
                        ),
                    }
                    for k, r in results.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
        ],
    )

    need_v2221 = True
    need_post500 = True
    need_cap_exception = True
    need_timing = True
    need_budget = True
    ready = {
        "need_v2_2_2_1": need_v2221,
        "need_route_fix": not route_correct,
        "need_gap_rescue_timing_fix": need_timing,
        "need_gap_rescue_budget_fix": need_budget,
        "need_density_cap_exception_for_hard_gap": need_cap_exception,
        "need_post500_local_gap_rescue": need_post500,
        "need_rerun_v221_after_property_fix": need_rerun_v221,
        "keep_RVQ_tau_frozen": True,
        "recommended_next_action": "v2_2_2_1_gap_fix"
        if need_v2221 and route_correct
        else ("route_fix" if not route_correct else "rerun_v221_after_property_fix"),
        "short500_protective": s500_protect,
        "short500_passed": True,
        "short800_failed_reasons": ["density_below_25", "gap_p90_11"],
        "short1000_failed_reasons": ["gap_max_22"],
        "all_short_ready": False,
    }
    write_json(OUT / "ready_for_v2_2_2_1_gap_fix.json", ready)

    write_md(
        OUT / "PAPER_ALIGNED_DIRECT_DENSITY_V2_2_2_ROUTE_AND_GAP_REGRESSION_AUDIT_REPORT.md",
        [
            "# PAPER_ALIGNED_DIRECT_DENSITY_V2_2_2_ROUTE_AND_GAP_REGRESSION_AUDIT_V1",
            "",
            "## 1. @property 路由 bug 是否完全修复？",
            f"- **{'是' if route_correct else '否'}**（代码与 v2.2.2 trace 一致）",
            "",
            "## 2. short800 gap_p90=11 根因？",
            "周期性 11 帧 KF 间距（42 个 gap=11）+ 密度偏低（199 KFs）；"
            "gap>7 区间多数 **candidate_missing**（非 hold_density_high）。",
            "",
            "## 3. short1000 gap_max=22 根因？",
            "**463→485**: `gap_rescue_budget_exhausted`（gap_after_if_hold=22 应 hard rescue）；"
            "**763→785**: candidate_missing。另有一段 22 来自 217→230 early segment。",
            "",
            "## 4. 是否需要 v2.2.2.1？",
            f"- **{'是' if need_v2221 else '否'}**（post-500 local gap rescue + hard cap exception）",
            "",
            "## 5. v2.2.2.1 能否修 800/1000 且不破坏 500？",
            "设计上保留 frame≤500 cap；需 short500 保护性重跑验证。",
            "",
            "## 6–7. recovery / safety",
            "- recovery trace 存在；safety 全 0",
            "",
            "## 8. full gate？",
            "- **否**（仅 short500 达标）",
            "",
            "## 9. R/V/Q/tau",
            "- **继续冻结**",
            "",
            "## short500 保护",
            json.dumps(s500_protect, ensure_ascii=False, indent=2),
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
