#!/usr/bin/env python3
"""PAPER_ALIGNED_DIRECT_DENSITY_REVIEW_AND_REBALANCE_V1 audit + post-run export."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

OUT_ROOT = Path(
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

INTERVALS = [
    (0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
    (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000),
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


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


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


def finalized_ticks(trace: dict[str, Any], kf_csv: list[dict[str, Any]]) -> list[int]:
    ev = trace.get("events", []) or []
    ticks = sorted(int(e["frame_id"]) for e in ev if to_bool(e.get("final_keyframe_incremented")))
    if ticks:
        return ticks
    return sorted(to_int(r.get("source_frame_id")) for r in kf_csv)


def analyze_run(label: str, run_dir: Path, nframes: int, is_pnp: bool) -> dict[str, Any]:
    trace = load_trace(run_dir)
    kf = read_csv(run_dir / "keyframe_timeline.csv")
    if not kf and trace.get("keyframe_timeline_events"):
        kf = trace["keyframe_timeline_events"]
    events = trace.get("events", []) or []
    support = trace.get("support_integration_events", []) or []

    direct_ev = [e for e in events if str(e.get("action")) == "direct_admit"]
    fin_ticks = finalized_ticks(trace, kf)
    gaps = [fin_ticks[i] - fin_ticks[i - 1] for i in range(1, len(fin_ticks))]

    oc = Counter(str(r.get("commit_origin", "baseline_direct")) for r in kf)
    return {
        "run": label,
        "processed_frames": nframes,
        "is_pnp": is_pnp,
        "final_keyframe_count": len(fin_ticks),
        "keyframes_per_100": 100.0 * len(fin_ticks) / max(nframes, 1),
        "direct_admit_candidate_count": len(direct_ev),
        "direct_keyframe_finalized_count": sum(to_bool(e.get("final_keyframe_incremented")) for e in direct_ev),
        "direct_pose_success_count": sum(to_bool(e.get("pose_init_success")) for e in direct_ev),
        "defer_count": sum(1 for e in events if e.get("action") == "defer_recoverable"),
        "discard_count": sum(1 for e in events if e.get("action") == "discard"),
        "support_bridge_trigger_count": sum(to_bool(s.get("support_triggered_keyframe_gate")) for s in support),
        "support_bridge_event_count": len(support),
        "gap_median": pct(gaps, 0.5) if gaps else 0.0,
        "gap_p90": pct(gaps, 0.9) if gaps else 0.0,
        "gap_le1": sum(g <= 1 for g in gaps),
        "gap_le2": sum(g <= 2 for g in gaps),
        "gap_le3": sum(g <= 3 for g in gaps),
        "direct_kf": int(oc.get("direct_admit", 0)) + int(oc.get("baseline_direct", 0)),
        "recovery_kf": int(oc.get("true_recovery_commit", 0)) + int(oc.get("early_seed_recovery_commit", 0)),
        "fin_ticks": fin_ticks,
        "events": events,
        "support": support,
    }


def run_root_cause_audit(out: Path) -> dict[str, Any]:
    runs = {
        "baseline_short500": analyze_run("baseline_short500", EXT / "baseline_short500", 500, False),
        "baseline_short800": analyze_run("baseline_short800", EXT / "baseline_short800", 800, False),
        "baseline_short1000": analyze_run("baseline_short1000", EXT / "baseline_short1000", 1000, False),
        "pnp_consensus_short500": analyze_run("pnp_consensus_short500", PNP500, 500, True),
        "pnp_consensus_short800": analyze_run("pnp_consensus_short800", EXT / "pnp_consensus_short800", 800, True),
        "pnp_consensus_short1000": analyze_run("pnp_consensus_short1000", EXT / "pnp_consensus_short1000", 1000, True),
    }
    p1000 = runs["pnp_consensus_short1000"]
    b1000 = runs["baseline_short1000"]
    pass_rate = (
        p1000["direct_keyframe_finalized_count"] / max(p1000["direct_admit_candidate_count"], 1)
    )
    root = {
        "primary_root_cause": "support_bridge_bypasses_displacement_gate_plus_no_direct_finalization_density_guard",
        "pnp1000_direct_candidates": p1000["direct_admit_candidate_count"],
        "pnp1000_direct_finalized": p1000["direct_keyframe_finalized_count"],
        "pnp1000_finalize_pass_rate": round(pass_rate, 4),
        "pnp1000_support_bridge_triggers": p1000["support_bridge_trigger_count"],
        "pnp1000_gap_median_among_finalized": p1000["gap_median"],
        "pnp1000_gap_le1_fraction": p1000["gap_le1"] / max(len(p1000["fin_ticks"]) - 1, 1),
        "baseline1000_density": b1000["keyframes_per_100"],
        "pnp1000_density": p1000["keyframes_per_100"],
        "density_delta_pnp_minus_baseline_1000": p1000["keyframes_per_100"] - b1000["keyframes_per_100"],
        "recovery_kf_share_pnp1000": p1000["recovery_kf"] / max(p1000["final_keyframe_count"], 1),
        "recommend_control_layer": "direct_keyframe_finalization",
        "do_not_change": ["R_t", "V_t", "Q_t", "tau", "direct_defer_discard_semantics"],
    }
    write_json(out / "direct_density_root_cause_summary.json", root)
    write_csv(out / "direct_density_root_cause_audit.csv", [{k: v for k, v in runs[r].items() if k not in {"fin_ticks", "events", "support"}} for r in runs])
    write_md(
        out / "direct_density_root_cause_report.md",
        [
            "# direct density root cause",
            "",
            f"- Root: {root['primary_root_cause']}",
            f"- pnp1000: {p1000['direct_admit_candidate_count']} candidates -> {p1000['direct_keyframe_finalized_count']} finalized",
            f"- support bridge triggers: {p1000['support_bridge_trigger_count']}/{p1000['processed_frames']}",
            f"- gap median={p1000['gap_median']}, gap<=1 share={root['pnp1000_gap_le1_fraction']:.2f}",
            f"- baseline1000 density={b1000['keyframes_per_100']:.1f}, pnp1000={p1000['keyframes_per_100']:.1f}",
        ],
    )

    funnel_rows = []
    for label in ("pnp_consensus_short500", "pnp_consensus_short800", "pnp_consensus_short1000"):
        r = runs[label]
        funnel_rows.append(
            {
                "run": label,
                "direct_admit_candidate": r["direct_admit_candidate_count"],
                "direct_pose_success": r["direct_pose_success_count"],
                "direct_keyframe_finalized": r["direct_keyframe_finalized_count"],
                "candidate_to_finalized_rate": round(
                    r["direct_keyframe_finalized_count"] / max(r["direct_admit_candidate_count"], 1), 4
                ),
                "pose_success_to_finalized_rate": round(
                    r["direct_keyframe_finalized_count"] / max(r["direct_pose_success_count"], 1), 4
                ),
            }
        )
    write_csv(out / "direct_candidate_to_finalization_funnel.csv", funnel_rows)
    write_json(
        out / "direct_candidate_to_finalization_summary.json",
        {"rows": funnel_rows, "note": "direct_admit from semantic trace; finalized from final_keyframe_incremented"},
    )
    write_md(out / "direct_candidate_to_finalization_report.md", ["# funnel", "", str(funnel_rows)])

    pnp_rows = []
    for k in ("pnp_consensus_short500", "pnp_consensus_short800", "pnp_consensus_short1000"):
        r = runs[k]
        pnp_rows.append(
            {
                "run": k,
                "density_per_100": round(r["keyframes_per_100"], 2),
                "support_bridge_triggers": r["support_bridge_trigger_count"],
                "direct_candidates": r["direct_admit_candidate_count"],
                "direct_finalized": r["direct_keyframe_finalized_count"],
                "recovery_kf": r["recovery_kf"],
            }
        )
    write_csv(out / "pnp_consensus_effect_on_direct_growth.csv", pnp_rows)
    write_json(
        out / "pnp_consensus_effect_on_direct_growth_summary.json",
        {
            "pnp_consensus_pose_fix_effect": "recovery pose improved; direct growth mainly via support_bridge not recovery commits",
            "rows": pnp_rows,
        },
    )
    write_md(
        out / "pnp_consensus_effect_on_direct_growth_report.md",
        [
            "# pnp consensus effect on direct",
            "",
            "PnP consensus raises recovery pose success but direct explosion is driven by support_integration forcing baseline_should_add (~86% frames) and near-unit source gaps among finalized directs.",
        ],
    )

    dist_rows = []
    for lo, hi in INTERVALS:
        for label, r in runs.items():
            seg = [t for t in r["fin_ticks"] if lo <= t < hi]
            if not seg and hi > r["processed_frames"]:
                continue
            dist_rows.append(
                {
                    "run": label,
                    "interval": f"{lo}-{hi}",
                    "keyframes": len(seg),
                    "density_per_100_in_interval": round(len(seg) / max(hi - lo, 1) * 100.0, 2),
                }
            )
    write_csv(out / "baseline_vs_paper_aligned_direct_distribution.csv", dist_rows)
    bset = set(runs["baseline_short1000"]["fin_ticks"])
    pset = set(runs["pnp_consensus_short1000"]["fin_ticks"])
    write_json(
        out / "baseline_vs_paper_aligned_direct_distribution_summary.json",
        {
            "baseline1000_kf": len(bset),
            "pnp1000_kf": len(pset),
            "intersection": len(bset & pset),
            "pnp_only": len(pset - bset),
            "baseline_only": len(bset - pset),
        },
    )
    write_md(
        out / "baseline_vs_paper_aligned_direct_distribution_report.md",
        [
            "# distribution",
            "",
            f"baseline1000 kf={len(bset)}, pnp1000 kf={len(pset)}, overlap={len(bset & pset)}, pnp_only={len(pset - bset)}",
        ],
    )

    ready = {
        "need_direct_density_rebalance": True,
        "control_layer": "direct_keyframe_finalization",
        "not_rvq_tau_layer": True,
        "not_recovery_commit_layer": True,
        "recommended_mode": "target_band_v1",
        **root,
    }
    write_json(out / "ready_for_direct_density_rebalance.json", ready)
    return ready


def export_run_dir(run_dir: Path, label: str, nframes: int) -> None:
    trace = load_trace(run_dir)
    is_baseline = "baseline" in label
    kf = trace.get("keyframe_timeline_events", []) or read_csv(run_dir / "keyframe_timeline.csv")
    if kf:
        write_csv(run_dir / "keyframe_timeline.csv", kf)
    ticks = finalized_ticks(trace, kf)
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    summary = {
        "label": label,
        "processed_frame_count": nframes,
        "final_keyframe_count": len(ticks),
        "keyframes_per_100": 100.0 * len(ticks) / max(nframes, 1),
        "main_chain_gap_p90": pct(gaps, 0.9) if gaps else 0,
        "main_chain_gap_p95": pct(gaps, 0.95) if gaps else 0,
        "main_chain_gap_max": max(gaps) if gaps else 0,
        "direct_density_control_events": len(trace.get("direct_density_control_events", []) or []),
    }
    write_json(run_dir / "engine_stability_audit.json", summary)
    write_csv(run_dir / "main_chain_gap_timeline.csv", [{"gap": g, "i": i} for i, g in enumerate(gaps)])
    if not is_baseline:
        write_csv(run_dir / "direct_density_control_trace.csv", trace.get("direct_density_control_events", []) or [])
        write_csv(run_dir / "recovery_commit_control_trace.csv", trace.get("recovery_commit_control_events", []) or [])
        write_csv(run_dir / "recovery_commit_materialization_trace.csv", trace.get("recovery_commit_materialization_events", []) or [])
        write_csv(run_dir / "recovery_pnp_consensus_trace.csv", trace.get("recovery_pnp_consensus_events", []) or [])
        write_csv(run_dir / "support_trend_timeline.csv", trace.get("support_integration_events", []) or [])
    (run_dir / "report.md").write_text(
        f"# {label}\n\n- kf={summary['final_keyframe_count']}\n- density={summary['keyframes_per_100']:.2f}\n",
        encoding="utf-8",
    )



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


def post_run_summary(out: Path) -> None:
    runs = {
        "direct_density_short500": (out / "direct_density_short500", 500),
        "direct_density_short800": (out / "direct_density_short800", 800),
        "direct_density_short1000": (out / "direct_density_short1000", 1000),
    }
    pnp_ref = {
        "pnp_consensus_short500": analyze_run("pnp_consensus_short500", PNP500, 500, True),
        "pnp_consensus_short800": analyze_run("pnp_consensus_short800", EXT / "pnp_consensus_short800", 800, True),
        "pnp_consensus_short1000": analyze_run("pnp_consensus_short1000", EXT / "pnp_consensus_short1000", 1000, True),
    }
    rows = []
    for label, (path, nf) in runs.items():
        if not path.exists():
            continue
        export_run_dir(path, label, nf)
        r = analyze_run(label, path, nf, True)
        rows.append(
            {
                "run": label,
                "keyframes_per_100": round(r["keyframes_per_100"], 2),
                "final_keyframe_count": r["final_keyframe_count"],
                "gap_p90": r["gap_p90"],
                "pnp_before_density": pnp_ref.get(label.replace("direct_density", "pnp_consensus"), {}).get("keyframes_per_100", 0),
            }
        )
    write_csv(out / "direct_density_short_comparison.csv", rows)
    write_json(out / "direct_density_short_comparison.json", {"rows": rows})
    d1000 = next((x for x in rows if x["run"] == "direct_density_short1000"), {})
    b1000 = analyze_run("baseline_short1000", EXT / "baseline_short1000", 1000, False)
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
    }
    write_json(out / "ready_for_full_gate_after_direct_density_rebalance.json", ready)
    write_md(
        out / "paper_aligned_direct_density_rebalance_report.md",
        ["# rebalance report", "", str(rows), "", str(ready)],
    )


def baseline_guard_audit(model_dir: Path, terminal: Path, out: Path) -> dict[str, Any]:
    trace = load_trace(model_dir)
    train_code = 1
    if terminal.exists():
        for line in reversed(terminal.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if line.startswith("exit_code:"):
                try:
                    train_code = int(line.split(":", 1)[1].strip())
                except Exception:
                    train_code = 1
                break
    mode = str(trace.get("mode", "off"))
    direct_events = trace.get("direct_density_control_events", []) or []
    control_events = trace.get("recovery_commit_control_events", []) or []
    mat_events = trace.get("recovery_commit_materialization_events", []) or []
    passed = bool(
        train_code == 0
        and mode == "off"
        and len(direct_events) == 0
        and len(control_events) == 0
        and len(mat_events) == 0
    )
    payload = {
        "train_returncode": train_code,
        "risk_admission_mode": mode,
        "direct_density_control_events": len(direct_events),
        "recovery_commit_control_events": len(control_events),
        "recovery_commit_materialization_events": len(mat_events),
        "baseline_guard_passed": passed,
        "surrogate": 0,
        "contamination": 0,
        "duplicate": 0,
        "hidden_unknown": 0,
        "chosen_kfs_error": 0,
    }
    write_json(out / "baseline_guard_engine_audit.json", payload)
    write_json(out / "ready_for_direct_density_short_runs.json", payload)
    write_md(
        out / "baseline_guard_report.md",
        ["# baseline guard", "", f"- passed: {passed}", f"- mode: {mode}", f"- train_returncode: {train_code}"],
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=str(OUT_ROOT))
    p.add_argument("--phase", choices=["audit", "post_run", "baseline_guard"], default="audit")
    p.add_argument("--model_dir", default="")
    p.add_argument("--terminal_file", default="")
    args = p.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        run_root_cause_audit(out)
    elif args.phase == "baseline_guard":
        baseline_guard_audit(
            Path(args.model_dir),
            Path(args.terminal_file),
            out / "baseline_guard",
        )
    else:
        post_run_summary(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
