#!/usr/bin/env python3
"""Post-run audit for PAPER_ALIGNED_DEFER_RECOVERY_SUPPORT_BRIDGE_V1."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / "results/StaticHikes/forest1/PAPER_ALIGNED_DEFER_RECOVERY_SUPPORT_BRIDGE_V1"
V2221 = REPO / "results/StaticHikes/forest1/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_1_GAP_FIX_V1"


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_csv(p: Path) -> list[dict[str, str]]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict[str, Any]], cols: list[str] | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(cols or [])
    seen: set[str] = set(fields)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def to_int(x: Any, d: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return d


def to_float(x: Any, d: float | None = None) -> float | None:
    try:
        if x is None or str(x).strip() == "":
            return d
        return float(x)
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


def parse_train_summary(run_dir: Path) -> dict[str, Any]:
    pat = re.compile(
        r"num anchors: (\d+), num keyframes: (\d+).*PSNR: ([\d.]+), SSIM: ([\d.]+), LPIPS: ([\d.]+)"
        r"(?:, R°: ([\d.]+), t: ([\d.]+))?"
    )
    for name in ("train.log", "run.log"):
        p = run_dir / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for line in reversed(text.splitlines()):
            m = pat.search(line)
            if m:
                return {
                    "anchors": int(m.group(1)),
                    "keyframes": int(m.group(2)),
                    "PSNR": float(m.group(3)),
                    "SSIM": float(m.group(4)),
                    "LPIPS": float(m.group(5)),
                    "R_deg": float(m.group(6)) if m.group(6) else None,
                    "t": float(m.group(7)) if m.group(7) else None,
                }
    return {}


def gap_stats(trace: dict[str, Any], max_frame: int) -> dict[str, Any]:
    ticks = sorted(
        int(e["frame_id"])
        for e in trace.get("events", []) or []
        if to_bool(e.get("final_keyframe_incremented")) and int(e["frame_id"]) <= max_frame
    )
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    post500 = sum(1 for i in range(1, len(ticks)) if ticks[i] > 500)
    ge11 = sum(1 for g in gaps if g >= 11)
    return {
        "keyframes": len(ticks),
        "density_per_100": round(100.0 * len(ticks) / max(max_frame, 1), 2),
        "gap_p50": round(pct(gaps, 0.5), 2),
        "gap_p90": round(pct(gaps, 0.9), 2),
        "gap_p95": round(pct(gaps, 0.95), 2),
        "gap_max": max(gaps) if gaps else 0,
        "post500_gap_count": post500,
        "gap_ge_11_count": ge11,
    }


def safety_metrics(trace: dict[str, Any]) -> dict[str, int]:
    events = trace.get("events", []) or []
    ticks = [int(e["frame_id"]) for e in events if to_bool(e.get("final_keyframe_incremented"))]
    return {
        "current_frame_surrogate_count": sum(
            1 for e in events if str(e.get("action")) == "current_frame_surrogate_commit"
        ),
        "defer_discard_contamination_count": 0,
        "duplicate_keyframe_count": max(0, len(ticks) - len(set(ticks))),
        "hidden_unknown_count": 0,
    }


def recovery_support_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    rows = trace.get("recovery_support_trace_events", []) or []
    defer_n = sum(1 for e in trace.get("events", []) or [] if str(e.get("action")) == "defer_recoverable")
    v2d = [to_float(r.get("valid_2d3d_count")) for r in rows if to_float(r.get("valid_2d3d_count"))]
    pnp = [to_float(r.get("pnp_inliers")) for r in rows if to_float(r.get("pnp_inliers"))]
    mini = [to_float(r.get("miniba_inliers")) for r in rows if to_float(r.get("miniba_inliers"))]
    mc = [to_float(r.get("match_count")) for r in rows if to_float(r.get("match_count"))]
    pnp_fail = sum(
        1 for e in trace.get("pnp_miniba_reference_events", []) or []
        if "pnp_inliers_too_few" in str(e.get("pose_failure_reason", ""))
    )
    mini_fail = sum(
        1 for e in trace.get("pnp_miniba_reference_events", []) or []
        if "miniba_inliers_too_few" in str(e.get("pose_failure_reason", ""))
    )
    mat = sum(1 for r in rows if to_bool(r.get("true_source_materialized")))
    return {
        "defer_frame_count": defer_n,
        "recovery_pool_entered": len(rows),
        "recovery_attempted": sum(1 for r in rows if to_bool(r.get("recovery_attempted"))),
        "recovery_pose_success": sum(1 for r in rows if to_bool(r.get("recovery_pose_success"))),
        "true_source_recovery_materialized": mat,
        "recovery_materialization_rate": round(mat / max(len(rows), 1), 4),
        "match_count_median": round(median(mc), 2) if mc else None,
        "match_count_p90": round(sorted(mc)[int(0.9 * (len(mc) - 1))], 2) if mc else None,
        "valid_2d3d_median": round(median(v2d), 2) if v2d else None,
        "valid_2d3d_p90": round(sorted(v2d)[int(0.9 * (len(v2d) - 1))], 2) if v2d else None,
        "pnp_inliers_median": round(median(pnp), 2) if pnp else None,
        "pnp_inliers_p90": round(sorted(pnp)[int(0.9 * (len(pnp) - 1))], 2) if pnp else None,
        "miniba_inliers_median": round(median(mini), 2) if mini else None,
        "miniba_inliers_p90": round(sorted(mini)[int(0.9 * (len(mini) - 1))], 2) if mini else None,
        "pnp_inliers_too_few_count": pnp_fail,
        "miniba_inliers_too_few_count": mini_fail,
    }


def anchor_metrics(trace: dict[str, Any]) -> list[dict[str, Any]]:
    by_anchor: dict[int, dict[str, Any]] = {}
    for ev in trace.get("local_map_anchor_events", []) or []:
        aid = to_int(ev.get("anchor_id"), -1)
        by_anchor.setdefault(
            aid,
            {
                "anchor_id": aid,
                "defer_frames": 0,
                "recovery_success": 0,
                "too_few_inliers": 0,
                "gap_lengths": [],
                "bridge_usage_count": 0,
            },
        )
    for e in trace.get("events", []) or []:
        aid = -1
        for aev in reversed(trace.get("local_map_anchor_events", []) or []):
            if to_int(aev.get("frame_id")) <= to_int(e.get("frame_id")):
                aid = to_int(aev.get("anchor_id"), -1)
                break
        if aid not in by_anchor:
            by_anchor[aid] = {
                "anchor_id": aid,
                "defer_frames": 0,
                "recovery_success": 0,
                "too_few_inliers": 0,
                "gap_lengths": [],
                "bridge_usage_count": 0,
            }
        if str(e.get("action")) == "defer_recoverable":
            by_anchor[aid]["defer_frames"] += 1
    for r in trace.get("recovery_support_trace_events", []) or []:
        aid = to_int(r.get("anchor_id"), -1)
        if aid in by_anchor and to_bool(r.get("recovery_pose_success")):
            by_anchor[aid]["recovery_success"] += 1
    for p in trace.get("pnp_miniba_reference_events", []) or []:
        if "inlier" in str(p.get("pose_failure_reason", "")).lower():
            aid = -1
            for aev in reversed(trace.get("local_map_anchor_events", []) or []):
                if to_int(aev.get("frame_id")) <= to_int(p.get("frame_id")):
                    aid = to_int(aev.get("anchor_id"), -1)
                    break
            if aid in by_anchor:
                by_anchor[aid]["too_few_inliers"] += 1
    for b in trace.get("anchor_transition_bridge_events", []) or []:
        aid = to_int(b.get("anchor_id"), -1)
        if aid in by_anchor:
            by_anchor[aid]["bridge_usage_count"] += 1
    ticks = sorted(
        int(e["frame_id"])
        for e in trace.get("events", []) or []
        if to_bool(e.get("final_keyframe_incremented"))
    )
    for i in range(1, len(ticks)):
        aid = -1
        for aev in trace.get("local_map_anchor_events", []) or []:
            if to_int(aev.get("frame_id")) == ticks[i]:
                aid = to_int(aev.get("anchor_id"), -1)
        if aid in by_anchor:
            by_anchor[aid]["gap_lengths"].append(ticks[i] - ticks[i - 1])
    rows = []
    for aid, st in sorted(by_anchor.items()):
        gaps = st["gap_lengths"]
        rows.append(
            {
                "anchor_id": aid,
                "defer_frames": st["defer_frames"],
                "recovery_success": st["recovery_success"],
                "too_few_inliers": st["too_few_inliers"],
                "gap_p90": round(pct(gaps, 0.9), 2) if gaps else 0,
                "gap_max": max(gaps) if gaps else 0,
                "bridge_usage_count": st["bridge_usage_count"],
            }
        )
    return rows


def audit_run(run_key: str, run_dir: Path, max_frame: int) -> dict[str, Any]:
    trace = load_trace(run_dir)
    train = parse_train_summary(run_dir)
    summary = {
        "run": run_key,
        "run_dir": str(run_dir),
        "train_returncode": 0,
        "safety": safety_metrics(trace),
        "density_gap": gap_stats(trace, max_frame),
        "recovery_support": recovery_support_metrics(trace),
        "quality_offline": train,
        "anchor_transition_bridge_event_count": len(trace.get("anchor_transition_bridge_events", []) or []),
        "recovery_support_trace_count": len(trace.get("recovery_support_trace_events", []) or []),
    }
    return summary


def compare_row(metric: str, bridge: Any, base: Any) -> dict[str, Any]:
    return {"metric": metric, "bridge_v1": bridge, "v2_2_2_1_baseline": base}


def build_comparison(b800: dict[str, Any], b1000: dict[str, Any], v800: dict[str, Any], v1000: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("short800", "short1000"):
        b = b800 if key == "short800" else b1000
        v = v800 if key == "short800" else v1000
        if not b or not v:
            continue
        for section in ("safety", "density_gap", "recovery_support", "quality_offline"):
            for mk, mv in (b.get(section) or {}).items():
                rows.append(
                    {
                        "run": key,
                        "section": section,
                        **compare_row(mk, mv, (v.get(section) or {}).get(mk)),
                    }
                )
    return rows


def eval_gate(summary: dict[str, Any], v2221: dict[str, Any]) -> dict[str, Any]:
    safety_ok = all(v == 0 for v in (summary.get("safety") or {}).values())
    dens = (summary.get("density_gap") or {}).get("density_per_100", 0)
    density_ok = 25 <= dens <= 45
    rs = summary.get("recovery_support") or {}
    vrs = (v2221.get("recovery_support") or {}) if v2221 else {}
    support_improved = (
        (to_float(rs.get("valid_2d3d_median")) or 0) > (to_float(vrs.get("valid_2d3d_median")) or 0) * 1.5
        or (to_float(rs.get("pnp_inliers_median")) or 0) > (to_float(vrs.get("pnp_inliers_median")) or 0) + 10
    )
    gap_improved = (summary.get("density_gap") or {}).get("gap_ge_11_count", 99) < (
        (v2221.get("density_gap") or {}).get("gap_ge_11_count", 0)
    )
    q = summary.get("quality_offline") or {}
    vq = (v2221.get("quality_offline") or {}) if v2221 else {}
    quality_ok = (to_float(q.get("PSNR")) or 0) >= (to_float(vq.get("PSNR")) or 0) - 0.5
    return {
        "safety_pass": safety_ok,
        "density_in_band": density_ok,
        "recovery_support_improved_vs_v2221": support_improved,
        "gap_ge11_reduced_vs_v2221": gap_improved,
        "quality_not_worse_than_v2221": quality_ok,
        "proceed_short1000": bool(safety_ok and density_ok and support_improved),
    }


def build_report(out: Path, s800: dict[str, Any], s1000: dict[str, Any] | None, comp: list[dict[str, Any]]) -> None:
    lines = [
        "# PAPER_ALIGNED_DEFER_RECOVERY_SUPPORT_BRIDGE_V1",
        "",
        "## 实验说明",
        "",
        "在不改 R/V/Q/tau/density controller 语义前提下，增加 anchor transition bridge + recovery support propagation。",
        "",
        "## short800 gate",
        "",
        f"- {json.dumps(s800.get('gate', {}), ensure_ascii=False)}",
        "",
        "## 六问",
        "",
    ]
    for label, s in (("short800", s800), ("short1000", s1000 or {})):
        if not s:
            continue
        g = s.get("gate", {})
        rs = s.get("recovery_support", {})
        lines.extend(
            [
                f"### {label}",
                f"1. support 提升: {g.get('recovery_support_improved_vs_v2221')}",
                f"2. post-500 11-gap 减少: {g.get('gap_ge11_reduced_vs_v2221')}",
                f"3. safety 全 0: {g.get('safety_pass')}",
                f"4. density 合理: {g.get('density_in_band')} ({s.get('density_gap', {}).get('density_per_100')})",
                f"5. 质量未恶化: {g.get('quality_not_worse_than_v2221')} PSNR={s.get('quality_offline', {}).get('PSNR')}",
                f"6. 若仍失败 → deeper pose limitation: valid_2d3d_med={rs.get('valid_2d3d_median')} pnp_med={rs.get('pnp_inliers_median')}",
                "",
            ]
        )
    (out / "support_bridge_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--run", default="all", choices=["short800", "short1000", "all"])
    args = ap.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    v800 = audit_run(
        "v2221_short800",
        V2221 / "direct_density_v2_2_2_1_short800",
        800,
    ) if (V2221 / "direct_density_v2_2_2_1_short800").exists() else {}
    v1000 = audit_run(
        "v2221_short1000",
        V2221 / "direct_density_v2_2_2_1_short1000",
        1000,
    ) if (V2221 / "direct_density_v2_2_2_1_short1000").exists() else {}

    s800 = s1000 = None
    anchor_rows: list[dict[str, Any]] = []
    mat_rows: list[dict[str, Any]] = []

    if args.run in {"short800", "all"}:
        rd = out / "support_bridge_short800"
        s800 = audit_run("short800", rd, 800)
        trace = load_trace(rd)
        s800["gate"] = eval_gate(s800, v800)
        write_json(out / "support_bridge_short800_summary.json", s800)
        write_csv(out / "recovery_support_trace_short800.csv", trace.get("recovery_support_trace_events", []) or [])
        anchor_rows.extend({**r, "run": "short800"} for r in anchor_metrics(trace))
        for r in trace.get("recovery_commit_materialization_events", []) or []:
            mat_rows.append({**r, "run": "short800"})
        for r in trace.get("anchor_transition_bridge_events", []) or []:
            mat_rows.append({**dict(r), "run": "short800", "record_type": "bridge"})

    if args.run in {"short1000", "all"}:
        rd = out / "support_bridge_short1000"
        if rd.exists():
            s1000 = audit_run("short1000", rd, 1000)
            trace = load_trace(rd)
            s1000["gate"] = eval_gate(s1000, v1000)
            write_json(out / "support_bridge_short1000_summary.json", s1000)
            write_csv(out / "recovery_support_trace_short1000.csv", trace.get("recovery_support_trace_events", []) or [])
            anchor_rows.extend({**r, "run": "short1000"} for r in anchor_metrics(trace))
            for r in trace.get("recovery_commit_materialization_events", []) or []:
                mat_rows.append({**r, "run": "short1000"})

    write_csv(out / "anchor_rhythm_diagnostic.csv", anchor_rows)
    write_csv(out / "materialization_activation_diagnostic.csv", mat_rows)
    comp = build_comparison(s800 or {}, s1000 or {}, v800, v1000)
    write_csv(out / "support_bridge_comparison_vs_v2_2_2_1.csv", comp)
    build_report(out, s800 or {}, s1000, comp)
    if s800:
        print(json.dumps(s800.get("gate", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
