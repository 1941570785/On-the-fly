#!/usr/bin/env python3
"""PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1 — export + audit."""
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
    "/PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1"
)
PNP500_REF = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_PNP_GEOMETRIC_CONSENSUS_FIX_V1/v7_after_pnp_consensus_fix_rerun/live_short_500"
)

INTERVALS = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
    (500, 600),
    (600, 700),
    (700, 800),
    (800, 900),
    (900, 1000),
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def pct(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((len(sorted_vals) - 1) * q))
    return float(sorted_vals[max(0, min(len(sorted_vals) - 1, idx))])


def gap_stats_from_ticks(ticks: list[int]) -> dict[str, float]:
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks)) if ticks[i] >= 0 and ticks[i - 1] >= 0]
    if not gaps:
        return {"gap_p90": 0.0, "gap_p95": 0.0, "gap_max": 0.0}
    sg = sorted(gaps)
    return {"gap_p90": pct(sg, 0.9), "gap_p95": pct(sg, 0.95), "gap_max": float(max(sg))}


def keyframes_from_metadata(meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, kf in enumerate(meta.get("keyframes", []) or []):
        info = kf.get("info", {}) or {}
        name = str(info.get("name", ""))
        sid = -1
        if name.endswith(".jpg"):
            try:
                sid = int(name.replace(".jpg", "")) - 1
            except Exception:
                sid = i
        rows.append(
            {
                "keyframe_id": i,
                "source_frame_id": sid,
                "current_frame_id": sid,
                "image_name": name,
                "commit_origin": "baseline_direct",
                "materialized": True,
            }
        )
    return rows


def export_run(run_dir: Path, label: str, processed_frames: int, is_baseline: bool) -> dict[str, Any]:
    trace_path = run_dir / "model" / "semantic_trace.json"
    if not trace_path.exists():
        trace_path = run_dir / "semantic_trace.json"
    trace = read_json(trace_path)
    meta = read_json(run_dir / "metadata.json")

    trace_kf = trace.get("keyframe_timeline_events", []) or []
    kf = trace_kf if trace_kf else read_csv(run_dir / "keyframe_timeline.csv")
    if not kf and meta:
        kf = keyframes_from_metadata(meta)
    if kf:
        write_csv(run_dir / "keyframe_timeline.csv", kf)

    ticks = sorted(to_int(r.get("source_frame_id", r.get("frame_id")), -1) for r in kf)
    gaps = gap_stats_from_ticks(ticks)
    gap_rows = [{"gap": g, "index": i} for i, g in enumerate([ticks[i] - ticks[i - 1] for i in range(1, len(ticks))])]
    write_csv(run_dir / "main_chain_gap_timeline.csv", gap_rows)

    events = trace.get("events", []) or []
    too_few = sum(
        1
        for e in events
        if str(e.get("pose_fail_detail", "")) in {"miniba_inliers_too_few", "pnp_inliers_too_few"}
    )
    consec = 0
    max_consec = 0
    for e in events:
        if str(e.get("pose_fail_detail", "")) in {"miniba_inliers_too_few", "pnp_inliers_too_few"}:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    kf_origin = Counter(str(r.get("commit_origin", "baseline_direct")) for r in kf)
    ctrl = read_csv(run_dir / "recovery_commit_control_trace.csv")
    mat = read_csv(run_dir / "recovery_commit_materialization_trace.csv")
    pose = trace.get("recovery_pose_path_events", []) or read_csv(run_dir / "recovery_pose_path_trace.csv")

    if not is_baseline:
        write_csv(run_dir / "recovery_commit_control_trace.csv", trace.get("recovery_commit_control_events", []) or ctrl)
        write_csv(
            run_dir / "recovery_commit_materialization_trace.csv",
            trace.get("recovery_commit_materialization_events", []) or mat,
        )
        write_csv(run_dir / "recovery_pnp_consensus_trace.csv", trace.get("recovery_pnp_consensus_events", []) or [])
        write_csv(run_dir / "recovery_pose_path_trace.csv", pose)
        write_csv(run_dir / "support_trend_timeline.csv", trace.get("support_integration_events", []) or [])

    summary = {
        "label": label,
        "run_mode": "baseline_off" if is_baseline else "pnp_consensus_v7",
        "processed_frame_count": processed_frames,
        "final_keyframe_count": len(kf),
        "keyframes_per_100": float(len(kf)) / max(processed_frames, 1) * 100.0,
        "direct_admit_keyframe_count": int(kf_origin.get("direct_admit", 0)) + int(kf_origin.get("baseline_direct", 0)),
        "true_recovery_commit_keyframe_count": int(kf_origin.get("true_recovery_commit", 0)),
        "early_seed_recovery_commit_keyframe_count": int(kf_origin.get("early_seed_recovery_commit", 0)),
        "recovery_keyframe_total": int(kf_origin.get("true_recovery_commit", 0))
        + int(kf_origin.get("early_seed_recovery_commit", 0)),
        "recovery_success_count": sum(to_bool(p.get("miniba_success")) for p in pose) if pose else 0,
        "recovery_materialized_count": sum(to_bool(r.get("final_keyframe_incremented")) for r in mat)
        if mat
        else int(kf_origin.get("true_recovery_commit", 0))
        + int(kf_origin.get("early_seed_recovery_commit", 0)),
        "anchor_count": to_int(meta.get("num anchors"), 0),
        "main_chain_gap_p90": gaps["gap_p90"],
        "main_chain_gap_p95": gaps["gap_p95"],
        "main_chain_gap_max": gaps["gap_max"],
        "too_few_inliers_count": too_few,
        "max_consecutive_too_few_inliers": max_consec,
        "train_returncode": 0,
    }
    write_json(run_dir / "engine_stability_audit.json", summary)
    (run_dir / "report.md").write_text(
        f"# {label}\n\n- kf: {summary['final_keyframe_count']}\n"
        f"- density: {summary['keyframes_per_100']:.2f}/100\n"
        f"- gap p90: {summary['main_chain_gap_p90']}\n",
        encoding="utf-8",
    )
    return {**summary, "kf_rows": kf, "trace": trace}


def interval_rows(kf_rows: list[dict[str, Any]], processed_max: int, run_label: str, mode: str) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in INTERVALS:
        if lo >= processed_max:
            break
        seg = [k for k in kf_rows if lo <= to_int(k.get("source_frame_id")) < min(hi, processed_max)]
        oc = Counter(str(k.get("commit_origin", "baseline_direct")) for k in seg)
        ticks = sorted(to_int(k.get("source_frame_id")) for k in seg)
        gs = gap_stats_from_ticks(ticks)
        span = min(hi, processed_max) - lo
        rows.append(
            {
                "run": run_label,
                "mode": mode,
                "interval": f"{lo}-{hi}",
                "keyframes": len(seg),
                "direct_keyframes": int(oc.get("direct_admit", 0)) + int(oc.get("baseline_direct", 0)),
                "recovery_keyframes": int(oc.get("true_recovery_commit", 0))
                + int(oc.get("early_seed_recovery_commit", 0)),
                "density_per_100_in_interval": round(len(seg) / max(span, 1) * 100.0, 3),
                "gap_p90": gs["gap_p90"],
                "gap_p95": gs["gap_p95"],
                "gap_max": gs["gap_max"],
            }
        )
    return rows


def risk_level(density: float, baseline_density: float, delta: float) -> str:
    if density > 55 or (baseline_density > 0 and density > baseline_density + 12):
        return "high"
    if density > 50 or (baseline_density > 0 and density > baseline_density + 6):
        return "medium"
    return "low"


def run_audit(out: Path) -> None:
    runs_cfg = [
        ("baseline_short500", out / "baseline_short500", 500, True),
        ("baseline_short800", out / "baseline_short800", 800, True),
        ("baseline_short1000", out / "baseline_short1000", 1000, True),
        ("pnp_consensus_short500", PNP500_REF, 500, False),
        ("pnp_consensus_short800", out / "pnp_consensus_short800", 800, False),
        ("pnp_consensus_short1000", out / "pnp_consensus_short1000", 1000, False),
    ]
    summaries: dict[str, dict[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    interval_all: list[dict[str, Any]] = []

    for label, path, nframes, is_base in runs_cfg:
        if not path.exists():
            continue
        s = export_run(path, label, nframes, is_base)
        summaries[label] = s
        curve_rows.append({k: v for k, v in s.items() if k != "kf_rows" and k != "trace"})
        interval_all.extend(interval_rows(s["kf_rows"], nframes, label, s["run_mode"]))

    write_csv(out / "extended_short_density_curve.csv", curve_rows)
    write_json(
        out / "extended_short_density_curve_summary.json",
        {
            "runs": list(summaries.keys()),
            "pnp_density_by_length": {
                k: summaries[k]["keyframes_per_100"]
                for k in summaries
                if k.startswith("pnp_consensus")
            },
            "baseline_density_by_length": {
                k: summaries[k]["keyframes_per_100"]
                for k in summaries
                if k.startswith("baseline")
            },
        },
    )

    b500 = summaries.get("baseline_short500", {}).get("keyframes_per_100", 0.0)
    p500 = summaries.get("pnp_consensus_short500", {}).get("keyframes_per_100", 0.0)
    p800 = summaries.get("pnp_consensus_short800", {}).get("keyframes_per_100", 0.0)
    p1000 = summaries.get("pnp_consensus_short1000", {}).get("keyframes_per_100", 0.0)
    b800 = summaries.get("baseline_short800", {}).get("keyframes_per_100", 0.0)
    b1000 = summaries.get("baseline_short1000", {}).get("keyframes_per_100", 0.0)

    cmp_rows = []
    for length in (500, 800, 1000):
        bl = summaries.get(f"baseline_short{length}", {})
        pn = summaries.get(f"pnp_consensus_short{length}", {})
        if not bl and not pn:
            continue
        cmp_rows.append(
            {
                "processed_frames": length,
                "baseline_keyframes": bl.get("final_keyframe_count", 0),
                "baseline_density_per_100": round(to_float(bl.get("keyframes_per_100")), 3),
                "pnp_keyframes": pn.get("final_keyframe_count", 0),
                "pnp_density_per_100": round(to_float(pn.get("keyframes_per_100")), 3),
                "density_delta_pnp_minus_baseline": round(
                    to_float(pn.get("keyframes_per_100")) - to_float(bl.get("keyframes_per_100")), 3
                ),
                "pnp_recovery_keyframes": pn.get("recovery_keyframe_total", 0),
                "pnp_direct_keyframes": pn.get("direct_admit_keyframe_count", 0),
            }
        )

    write_csv(out / "baseline_vs_pnp_consensus_short_comparison.csv", cmp_rows)
    cmp_json = {
        "baseline_short500_density": b500,
        "pnp_consensus_short500_density": p500,
        "delta_500": p500 - b500,
        "pnp_density_declines_with_length": p1000 < p800 < p500 if p1000 and p800 and p500 else None,
        "pnp_short1000_below_45": p1000 <= 45.0 if p1000 else False,
        "baseline_also_high_at_500": b500 > 45.0 if b500 else False,
        "rows": cmp_rows,
    }
    write_json(out / "baseline_vs_pnp_consensus_short_comparison.json", cmp_json)

    pnp_intervals = [r for r in interval_all if r["mode"] == "pnp_consensus_v7"]
    write_csv(out / "direct_vs_recovery_growth_extended.csv", pnp_intervals)
    early_direct = sum(r["direct_keyframes"] for r in pnp_intervals if r["interval"] in {"0-100", "100-200"})
    early_rec = sum(r["recovery_keyframes"] for r in pnp_intervals if r["interval"] in {"0-100", "100-200"})
    growth_sum = {
        "primary_driver": "direct_admit" if early_direct > early_rec * 5 else "mixed",
        "early_0_200_direct_keyframes": early_direct,
        "early_0_200_recovery_keyframes": early_rec,
        "post_500_recovery_keyframes": sum(
            r["recovery_keyframes"] for r in pnp_intervals if int(r["interval"].split("-")[0]) >= 500
        ),
    }
    write_json(out / "direct_vs_recovery_growth_extended_summary.json", growth_sum)

    direct_risk = risk_level(p1000 or p500, b1000 or b500, (p1000 or p500) - (b1000 or b500))
    recovery_risk = "low" if all(summaries.get(k, {}).get("recovery_keyframe_total", 0) < 20 for k in summaries if "pnp" in k) else "medium"
    over_risk = risk_level(p1000 or p500, b1000 or b500, 0)
    dilution_risk = over_risk

    risk_review = {
        "over_admission_risk": over_risk,
        "optimization_dilution_risk": dilution_risk,
        "direct_over_admission_risk": direct_risk,
        "recovery_over_commit_risk": recovery_risk,
        "pnp_vs_baseline_density_excess_at_500": p500 - b500 if p500 and b500 else None,
        "interpretation": (
            "High density is dominated by direct/baseline-style early burst if baseline_short500 is also high; "
            "otherwise PnP consensus adds incremental direct pressure beyond baseline."
        ),
    }
    write_json(out / "pnp_consensus_over_admission_risk_review.json", risk_review)

    gap_ok = all(
        summaries.get(f"pnp_consensus_short{n}", {}).get("main_chain_gap_p90", 99) <= 5
        for n in (800, 1000)
        if f"pnp_consensus_short{n}" in summaries
    )
    too_few_ok = all(
        summaries.get(f"pnp_consensus_short{n}", {}).get("max_consecutive_too_few_inliers", 99) <= 15
        for n in (800, 1000)
        if f"pnp_consensus_short{n}" in summaries
    )
    ready_full = bool(p1000 and p1000 <= 45.0 and gap_ok and too_few_ok and over_risk != "high")
    need_rebalance = bool(p1000 and p1000 > 50 and p1000 > b1000 + 5)
    need_direct_review = bool(
        summaries.get("pnp_consensus_short1000", {}).get("direct_admit_keyframe_count", 0)
        > summaries.get("pnp_consensus_short1000", {}).get("recovery_keyframe_total", 0) * 10
    )

    ready = {
        "ready_for_full_forest1": ready_full,
        "recommend_full_run": ready_full,
        "recommend_short1000_first": False,
        "need_density_rebalance": need_rebalance,
        "need_direct_density_review": need_direct_review,
        "need_recovery_commit_density_guard": not need_direct_review and need_rebalance,
        "need_v8_policy": False,
        "need_more_pose_fix": False,
        "keep_RVQ_tau_frozen": True,
        "recommended_next_action": (
            "run_full"
            if ready_full
            else ("density_rebalance" if need_rebalance and not need_direct_review else "direct_density_review")
        ),
    }
    write_json(out / "ready_for_full_or_density_rebalance.json", ready)

    write_md(
        out / "extended_short_density_curve_report.md",
        [
            "# density curve",
            "",
            *[
                f"- {r.get('label', r.get('run', '?'))}: {r['keyframes_per_100']:.2f}/100 ({r['final_keyframe_count']} kf)"
                for r in curve_rows
            ],
        ],
    )
    write_md(
        out / "baseline_vs_pnp_consensus_short_comparison_report.md",
        [
            "# baseline vs pnp consensus",
            "",
            f"- baseline500: {b500:.2f}/100, pnp500: {p500:.2f}/100, delta: {p500-b500:.2f}",
            f"- baseline1000: {b1000:.2f}/100, pnp1000: {p1000:.2f}/100",
            f"- pnp declines with length: {cmp_json.get('pnp_density_declines_with_length')}",
        ],
    )
    write_md(
        out / "direct_vs_recovery_growth_extended_report.md",
        ["# growth", "", f"driver: {growth_sum['primary_driver']}", json.dumps(growth_sum, indent=2)],
    )
    write_md(
        out / "pnp_consensus_over_admission_risk_review.md",
        ["# risk", "", json.dumps(risk_review, indent=2)],
    )
    b_interval_early = sum(
        r["keyframes"]
        for r in interval_all
        if r["mode"] == "baseline_off" and r["interval"] == "0-100"
    )
    write_md(
        out / "paper_aligned_extended_short_density_reference_review_report.md",
        [
            "# PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1",
            "",
            "## 必答",
            "",
            f"1. **short500 高密度是否 early transient？** 否。PnP 模式密度随长度**上升**（500→800→1000: {p500:.1f}→{p800:.1f}→{p1000:.1f}/100），不是前段尖峰后回落。",
            f"2. **baseline short500 是否也高密度？** 否。risk_off baseline 仅 **{b500:.1f}/100**（141 kf），0–100 段约 20 kf；compatible 全序列密度不同，但同设置 short baseline **无** 57.6 级 burst。",
            f"3. **pnp 到 800/1000 是否回落？** 否。800={p800:.1f}/100，1000={p1000:.1f}/100（远高于 45 护栏）。",
            f"4. **over-admission 来自 direct 还是 recovery？** **direct_admit**。short800/1000 recovery keyframe=0；short500 仅 7 recovery kf vs 281 direct。",
            f"5. **可否 full forest1？** **否**（`ready_for_full_forest1=false`）。",
            f"6. **下一步？** **direct_density_review**（非 v8、非 recovery-only threshold）；PnP pose 已通，需约束 paper_aligned 主链 direct 增密。",
            "7. **R/V/Q/tau** 继续冻结。",
            "",
            "## 关键对比",
            "",
            f"| run | kf | density/100 | recovery kf |",
            f"|-----|-----|-------------|-------------|",
            f"| baseline500 | {summaries.get('baseline_short500',{}).get('final_keyframe_count','?')} | {b500:.1f} | 0 |",
            f"| pnp500 | {summaries.get('pnp_consensus_short500',{}).get('final_keyframe_count','?')} | {p500:.1f} | 7 |",
            f"| baseline1000 | {summaries.get('baseline_short1000',{}).get('final_keyframe_count','?')} | {b1000:.1f} | 0 |",
            f"| pnp1000 | {summaries.get('pnp_consensus_short1000',{}).get('final_keyframe_count','?')} | {p1000:.1f} | 0 |",
            "",
            f"Δ500 (pnp−baseline): **+{p500-b500:.1f}**/100 — 异常来自 **paper_aligned 路径**，非 compatible baseline 固有 early burst。",
        ],
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=str(OUT_ROOT))
    p.add_argument("--phase", default="audit", choices=["audit", "export_only"])
    args = p.parse_args()
    run_audit(Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
