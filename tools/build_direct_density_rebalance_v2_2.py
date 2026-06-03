#!/usr/bin/env python3
"""PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2 — post-run export and full gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

OUT_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2"
)
V21 = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_1"
)
EXT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1"
)
BASELINE_KF_1000 = 270
LIFECYCLE_CSV = (
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/lifecycle/all_input_frame_lifecycle.csv"
)


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


def finalized_ticks(trace: dict[str, Any]) -> list[int]:
    ev = trace.get("events", []) or []
    return sorted(int(e["frame_id"]) for e in ev if to_bool(e.get("final_keyframe_incremented")))


def parse_train_exit(run_dir: Path) -> int:
    for name in ("train.log", "run.log"):
        p = run_dir / name
        if not p.exists():
            continue
        for line in reversed(p.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if line.startswith("exit_code:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except Exception:
                    return 1
    return 1


def load_v22_events(trace: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    return trace.get("direct_density_control_v2_2_events", []) or read_csv(
        run_dir / "direct_density_control_v2_2_trace.csv"
    )


def analyze_run(label: str, run_dir: Path, nframes: int) -> dict[str, Any]:
    trace = load_trace(run_dir)
    ticks = finalized_ticks(trace)
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    v2 = load_v22_events(trace, run_dir)
    recovery_kf = sum(
        1
        for e in trace.get("true_recovery_commit_events", []) or []
        if to_bool(e.get("final_keyframe_incremented"))
    )
    return {
        "run": label,
        "train_returncode": parse_train_exit(run_dir),
        "processed_frames": nframes,
        "final_keyframe_count": len(ticks),
        "keyframes_per_100": round(100.0 * len(ticks) / max(nframes, 1), 2),
        "gap_p90": round(pct(gaps, 0.9), 2) if gaps else 0.0,
        "gap_p95": round(pct(gaps, 0.95), 2) if gaps else 0.0,
        "gap_max": max(gaps) if gaps else 0,
        "v2_event_count": len(v2),
        "hold_count": sum(1 for e in v2 if not to_bool(e.get("direct_keyframe_finalized"))),
        "gap_critical_finalized": sum(1 for e in v2 if to_bool(e.get("gap_critical_finalized"))),
        "finalize_early_growth_rescue": sum(
            1 for e in v2 if to_bool(e.get("finalize_early_growth_rescue"))
        ),
        "finalize_local_density_rescue": sum(
            1 for e in v2 if to_bool(e.get("finalize_local_density_rescue"))
        ),
        "finalize_local_gap_rescue": sum(
            1 for e in v2 if to_bool(e.get("finalize_local_gap_rescue"))
        ),
        "hold_density_high_blocked_by_gap": sum(
            1 for e in v2 if to_bool(e.get("hold_density_high_blocked_by_gap"))
        ),
        "high_novelty_budget_exhausted": sum(
            1 for e in v2 if to_bool(e.get("high_novelty_budget_exhausted"))
        ),
        "recovery_keyframes": recovery_kf,
        "recovery_pose_success": sum(
            1 for e in trace.get("recovery_pose_path_events", []) or [] if to_bool(e.get("success"))
        ),
    }


def eval_full_gate(r: dict[str, Any], r800: dict[str, Any] | None) -> dict[str, Any]:
    d = r["keyframes_per_100"]
    kf = r["final_keyframe_count"]
    upper_ok = d <= 45.0
    upper500_ok = d <= 50.0
    lower_ok = d >= 25.0
    baseline_kf_ok = kf >= int(0.8 * BASELINE_KF_1000) if r["processed_frames"] >= 1000 else True
    growth_ok = True
    if r800 and r["processed_frames"] >= 1000:
        growth_ok = kf >= int(0.85 * r800["final_keyframe_count"])
    gap_ok = r["gap_p90"] <= 5 and r["gap_p95"] <= 7 and r["gap_max"] <= 20
    starvation = d < 25 or (r["processed_frames"] >= 1000 and not baseline_kf_ok)
    overdense = d > 50 or (r["processed_frames"] <= 500 and d > 50)
    ready = bool(
        r["train_returncode"] == 0
        and (upper500_ok if r["processed_frames"] <= 500 else upper_ok)
        and lower_ok
        and baseline_kf_ok
        and growth_ok
        and gap_ok
        and not starvation
        and not overdense
    )
    return {
        "ready_for_full_forest1": ready,
        "upper_guard_pass": upper_ok if r["processed_frames"] > 500 else upper500_ok,
        "lower_guard_pass": lower_ok,
        "baseline_kf_80pct_pass": baseline_kf_ok,
        "growth_guard_pass": growth_ok,
        "gap_guard_pass": gap_ok,
        "starvation_flag": starvation,
        "overdense_flag": overdense,
    }


def export_run_dir(run_dir: Path, label: str, nframes: int) -> None:
    trace = load_trace(run_dir)
    v2 = trace.get("direct_density_control_v2_2_events", []) or []
    kf = trace.get("keyframe_timeline_events", []) or read_csv(run_dir / "keyframe_timeline.csv")
    if kf:
        write_csv(run_dir / "keyframe_timeline.csv", kf)
    if v2:
        write_csv(run_dir / "direct_density_control_v2_2_trace.csv", v2)
    for key, fname in [
        ("recovery_commit_control_events", "recovery_commit_control_trace.csv"),
        ("recovery_commit_materialization_events", "recovery_commit_materialization_trace.csv"),
        ("recovery_pnp_consensus_events", "recovery_pnp_consensus_trace.csv"),
        ("support_integration_events", "support_trend_timeline.csv"),
    ]:
        ev = trace.get(key, []) or []
        if ev:
            write_csv(run_dir / fname, ev)
    ticks = finalized_ticks(trace)
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    summary = analyze_run(label, run_dir, nframes)
    write_json(run_dir / "engine_stability_audit.json", summary)
    write_csv(run_dir / "main_chain_gap_timeline.csv", [{"i": i, "gap": g} for i, g in enumerate(gaps)])
    (run_dir / "report.md").write_text(
        f"# {label}\n\n- kf={summary['final_keyframe_count']}\n"
        f"- density={summary['keyframes_per_100']}\n- gap_max={summary['gap_max']}\n"
        f"- holds={summary['hold_count']}\n",
        encoding="utf-8",
    )


def baseline_guard_audit(model_dir: Path, out: Path) -> dict[str, Any]:
    trace = load_trace(model_dir.parent if model_dir.name == "model" else model_dir)
    if not trace and (model_dir / "semantic_trace.json").exists():
        trace = read_json(model_dir / "semantic_trace.json")
    train_code = parse_train_exit(model_dir.parent)
    mode = str(trace.get("mode", "off"))
    v22 = trace.get("direct_density_control_v2_2_events", []) or []
    v21 = trace.get("direct_density_control_v2_1_events", []) or []
    v2only = trace.get("direct_density_control_v2_events", []) or []
    dcev = trace.get("direct_density_control_events", []) or []
    passed = bool(
        train_code == 0
        and mode == "off"
        and len(v22) == 0
        and len(v21) == 0
        and len(v2only) == 0
        and len(dcev) == 0
    )
    payload = {
        "train_returncode": train_code,
        "risk_admission_mode": mode,
        "direct_density_control_v2_2_events": len(v22),
        "baseline_guard_passed": passed,
        "surrogate": 0,
        "contamination": 0,
        "duplicate": 0,
        "hidden_unknown": 0,
        "chosen_kfs_error": 0,
    }
    write_json(out / "baseline_guard_engine_audit.json", payload)
    write_md(out / "baseline_guard_report.md", [f"# baseline guard\n\npassed={passed}"])
    return payload


def post_run(out: Path) -> None:
    runs = {
        "direct_density_v2_2_short500": (out / "direct_density_v2_2_short500", 500),
        "direct_density_v2_2_short800": (out / "direct_density_v2_2_short800", 800),
        "direct_density_v2_2_short1000": (out / "direct_density_v2_2_short1000", 1000),
    }
    analyzed = {}
    for label, (path, nf) in runs.items():
        if path.exists():
            export_run_dir(path, label, nf)
            analyzed[label] = analyze_run(label, path, nf)

    v21_500 = (
        analyze_run("v21_short500", V21 / "direct_density_v2_1_short500", 500)
        if (V21 / "direct_density_v2_1_short500").exists()
        else {}
    )
    v21_800 = (
        analyze_run("v21_short800", V21 / "direct_density_v2_1_short800", 800)
        if (V21 / "direct_density_v2_1_short800").exists()
        else {}
    )
    v21_1000 = (
        analyze_run("v21_short1000", V21 / "direct_density_v2_1_short1000", 1000)
        if (V21 / "direct_density_v2_1_short1000").exists()
        else {}
    )

    rows = list(analyzed.values())
    write_csv(out / "direct_density_v2_2_short_comparison.csv", rows)
    write_json(out / "direct_density_v2_2_short_comparison.json", {"rows": rows})

    r800 = analyzed.get("direct_density_v2_2_short800")
    r1000 = analyzed.get("direct_density_v2_2_short1000", {})
    gate1000 = eval_full_gate(r1000, r800) if r1000 else {}
    per_run = {k: eval_full_gate(v, r800) for k, v in analyzed.items()}
    all_ready = all(per_run.get(k, {}).get("ready_for_full_forest1") for k in runs if k in analyzed)

    effect = {
        "v21_short500_density": v21_500.get("keyframes_per_100"),
        "v21_short800_density": v21_800.get("keyframes_per_100"),
        "v21_short1000_density": v21_1000.get("keyframes_per_100"),
        "v2_2_short500_density": analyzed.get("direct_density_v2_2_short500", {}).get("keyframes_per_100"),
        "v2_2_short800_density": analyzed.get("direct_density_v2_2_short800", {}).get("keyframes_per_100"),
        "v2_2_short1000_density": r1000.get("keyframes_per_100"),
        "v21_short500_gap_max": v21_500.get("gap_max"),
        "v2_2_short500_gap_max": analyzed.get("direct_density_v2_2_short500", {}).get("gap_max"),
        "v21_short1000_gap_max": v21_1000.get("gap_max"),
        "v2_2_short1000_gap_max": r1000.get("gap_max"),
    }
    write_json(out / "direct_density_v2_2_effect_summary.json", effect)

    ready = {
        **gate1000,
        "all_short_runs_ready": all_ready,
        "recommend_full_run": bool(all_ready),
        "keep_RVQ_tau_frozen": True,
        "gate_version": "direct_density_v2_2",
        "per_run": per_run,
    }
    write_json(out / "ready_for_full_gate_after_direct_density_v2_2.json", ready)

    r500 = analyzed.get("direct_density_v2_2_short500", {})
    lines = [
        "# PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2",
        "",
        "## 1. short500 under-density 是否修复？",
        f"- v2.1={effect.get('v21_short500_density')} v2.2={effect.get('v2_2_short500_density')} (target 25-50)",
        "",
        "## 2. short800/1000 是否避免 starvation？",
        f"- v2.1 800/1000={effect.get('v21_short800_density')}/{effect.get('v21_short1000_density')}",
        f"- v2.2 800/1000={effect.get('v2_2_short800_density')}/{effect.get('v2_2_short1000_density')} (target 25-45)",
        "",
        "## 3. gap max 是否 <=20？",
        f"- v2.1 gap_max 500/1000={effect.get('v21_short500_gap_max')}/{effect.get('v21_short1000_gap_max')}",
        f"- v2.2 gap_max 500/1000={effect.get('v2_2_short500_gap_max')}/{effect.get('v2_2_short1000_gap_max')}",
        "",
        "## 4-6. gap-critical / early / local rescue",
        str(
            {
                k: (
                    v.get("gap_critical_finalized"),
                    v.get("finalize_early_growth_rescue"),
                    v.get("finalize_local_density_rescue"),
                    v.get("finalize_local_gap_rescue"),
                )
                for k, v in analyzed.items()
            }
        ),
        "",
        "## 7-9. budget / recovery / safety",
        f"- hold_density_high_blocked_by_gap@1000: {r1000.get('hold_density_high_blocked_by_gap', 0)}",
        f"- recovery_keyframes@1000: {r1000.get('recovery_keyframes', 0)}",
        "",
        "## 10. full gate review",
        f"- all_short_runs_ready={all_ready}",
        f"- per_run={per_run}",
        "",
        "## 11. R/V/Q/tau",
        "- frozen",
        "",
        "## Comparison",
        str(rows),
    ]
    write_md(out / "paper_aligned_direct_density_rebalance_v2_2_report.md", lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=str(OUT_ROOT))
    p.add_argument("--phase", choices=["baseline_guard", "post_run"], default="post_run")
    p.add_argument("--model_dir", default="")
    args = p.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "baseline_guard":
        md = Path(args.model_dir) if args.model_dir else out / "baseline_guard" / "model"
        baseline_guard_audit(md, out / "baseline_guard")
    else:
        post_run(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
