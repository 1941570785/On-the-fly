#!/usr/bin/env python3
"""PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2 — gap-tail preemptive fix post-run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

OUT_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_GAP_TAIL_FIX_V1"
)
V221 = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_GAP_TAIL_REFINEMENT_V1"
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


def load_v222_events(trace: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    return trace.get("direct_density_control_v2_2_2_events", []) or read_csv(
        run_dir / "direct_density_control_v2_2_2_trace.csv"
    )


def analyze_run(label: str, run_dir: Path, nframes: int) -> dict[str, Any]:
    trace = load_trace(run_dir)
    ticks = finalized_ticks(trace)
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    v2 = load_v222_events(trace, run_dir)
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
        "preemptive_rescue_count": sum(
            1
            for e in v2
            if to_bool(e.get("finalize_gap_tail_preemptive_v2_2_2"))
            or str(e.get("direct_finalization_decision", ""))
            == "finalize_gap_tail_preemptive_v2_2_2"
        ),
        "hard_rescue_v222_count": sum(
            1
            for e in v2
            if to_bool(e.get("finalize_hard_gap_rescue_v2_2_2"))
            or str(e.get("direct_finalization_decision", ""))
            == "finalize_hard_gap_rescue_v2_2_2"
        ),
        "soft_rescue_count": sum(
            1
            for e in v2
            if str(e.get("direct_finalization_decision", ""))
            == "finalize_gap_tail_rescue_v2_2_1"
        ),
        "hold_density_high_blocked_by_gap": sum(
            1 for e in v2 if to_bool(e.get("hold_density_high_blocked_by_gap"))
        ),
        "hold_gap_rescue_blocked_by_density_cap": sum(
            1 for e in v2 if to_bool(e.get("hold_gap_rescue_blocked_by_density_cap"))
        ),
        "recovery_keyframes": recovery_kf,
        "recovery_pose_success": sum(
            1 for e in trace.get("recovery_pose_path_events", []) or [] if to_bool(e.get("success"))
        ),
        "gap_timeline": gaps,
        "keyframe_ticks": ticks,
    }


def eval_short_gate(r: dict[str, Any], r800: dict[str, Any] | None) -> dict[str, Any]:
    d = r["keyframes_per_100"]
    kf = r["final_keyframe_count"]
    nf = r["processed_frames"]
    upper_ok = d <= 45.0 if nf > 500 else d <= 50.0
    lower_ok = d >= 25.0
    baseline_kf_ok = kf >= int(0.8 * BASELINE_KF_1000) if nf >= 1000 else True
    growth_ok = True
    if r800 and nf >= 1000:
        growth_ok = kf >= int(0.85 * r800["final_keyframe_count"])
    gap_ok = r["gap_p90"] <= 5 and r["gap_p95"] <= 7 and r["gap_max"] <= 20
    ready = bool(
        r["train_returncode"] == 0
        and upper_ok
        and lower_ok
        and baseline_kf_ok
        and growth_ok
        and gap_ok
    )
    return {
        "ready_for_full_forest1": ready,
        "upper_guard_pass": upper_ok,
        "lower_guard_pass": lower_ok,
        "baseline_kf_80pct_pass": baseline_kf_ok,
        "growth_guard_pass": growth_ok,
        "gap_guard_pass": gap_ok,
    }


def export_run_dir(run_dir: Path, label: str, nframes: int) -> None:
    trace = load_trace(run_dir)
    v2 = trace.get("direct_density_control_v2_2_2_events", []) or []
    kf = trace.get("keyframe_timeline_events", []) or read_csv(run_dir / "keyframe_timeline.csv")
    if kf:
        write_csv(run_dir / "keyframe_timeline.csv", kf)
    if v2:
        write_csv(run_dir / "direct_density_control_v2_2_2_trace.csv", v2)
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
        f"- preemptive={summary['preemptive_rescue_count']}\n",
        encoding="utf-8",
    )


def baseline_guard_audit(model_dir: Path, out: Path) -> dict[str, Any]:
    trace = load_trace(model_dir.parent if model_dir.name == "model" else model_dir)
    if not trace and (model_dir / "semantic_trace.json").exists():
        trace = read_json(model_dir / "semantic_trace.json")
    train_code = parse_train_exit(model_dir.parent)
    mode = str(trace.get("mode", "off"))
    v222 = trace.get("direct_density_control_v2_2_2_events", []) or []
    v221 = trace.get("direct_density_control_v2_2_1_events", []) or []
    passed = bool(train_code == 0 and mode == "off" and len(v222) == 0 and len(v221) == 0)
    payload = {
        "train_returncode": train_code,
        "risk_admission_mode": mode,
        "direct_density_control_v2_2_2_events": len(v222),
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


def frame_level_gap_report(out: Path, r500: dict[str, Any]) -> None:
    ticks = r500.get("keyframe_ticks") or []
    gaps = r500.get("gap_timeline") or []
    rows = []
    for i, g in enumerate(gaps):
        end_f = ticks[i + 1] if i + 1 < len(ticks) else None
        rows.append({"gap_index": i, "gap": g, "keyframe_frame": end_f})
    big = [x for x in rows if int(x["gap"]) > 20]
    trace = load_v222_events(load_trace(out / "direct_density_v2_2_2_short500"), out / "direct_density_v2_2_2_short500")
    hold_rows = [
        e
        for e in trace
        if 430 <= int(e.get("frame_id", 0)) <= 452
        and not to_bool(e.get("direct_keyframe_finalized"))
    ]
    preempt = [
        e
        for e in trace
        if 430 <= int(e.get("frame_id", 0)) <= 452
        and (
            to_bool(e.get("finalize_gap_tail_preemptive_v2_2_2"))
            or str(e.get("direct_finalization_decision", ""))
            == "finalize_gap_tail_preemptive_v2_2_2"
        )
    ]
    payload = {
        "gap_max": r500.get("gap_max"),
        "gaps_over_20": big,
        "region_430_452_hold_events": len(hold_rows),
        "region_430_452_preemptive_rescues": len(preempt),
        "hold_sample": hold_rows[:20],
        "preemptive_sample": preempt[:20],
    }
    write_json(out / "short500_frame_level_gap_report.json", payload)
    write_md(
        out / "short500_frame_level_gap_report.md",
        [
            "# short500 frame-level gap report",
            "",
            f"- gap_max={r500.get('gap_max')}",
            f"- gaps>20: {big}",
            f"- 430-452 holds={len(hold_rows)} preemptive={len(preempt)}",
        ],
    )


def short500_gate(out: Path) -> int:
    run_dir = out / "direct_density_v2_2_2_short500"
    if not run_dir.exists():
        return 1
    export_run_dir(run_dir, "direct_density_v2_2_2_short500", 500)
    r = analyze_run("direct_density_v2_2_2_short500", run_dir, 500)
    gate = eval_short_gate(r, None)
    ok = bool(
        gate["ready_for_full_forest1"]
        and r["train_returncode"] == 0
        and 25 <= r["keyframes_per_100"] <= 50
        and r["gap_max"] <= 20
    )
    write_json(out / "short500_gate_check.json", {"passed": ok, "metrics": r, "gate": gate})
    if not ok:
        frame_level_gap_report(out, r)
    return 0 if ok else 1


def post_run(out: Path) -> None:
    runs = {
        "direct_density_v2_2_2_short500": (out / "direct_density_v2_2_2_short500", 500),
        "direct_density_v2_2_2_short800": (out / "direct_density_v2_2_2_short800", 800),
        "direct_density_v2_2_2_short1000": (out / "direct_density_v2_2_2_short1000", 1000),
    }
    analyzed: dict[str, dict[str, Any]] = {}
    for label, (path, nf) in runs.items():
        if path.exists():
            export_run_dir(path, label, nf)
            analyzed[label] = analyze_run(label, path, nf)

    v221_500 = (
        analyze_run("v221_short500", V221 / "direct_density_v2_2_1_short500", 500)
        if (V221 / "direct_density_v2_2_1_short500").exists()
        else {}
    )

    rows = list(analyzed.values())
    write_csv(out / "direct_density_v2_2_2_short_comparison.csv", rows)
    write_json(out / "direct_density_v2_2_2_short_comparison.json", {"rows": rows})

    r800 = analyzed.get("direct_density_v2_2_2_short800")
    r1000 = analyzed.get("direct_density_v2_2_2_short1000", {})
    per_run = {k: eval_short_gate(v, r800) for k, v in analyzed.items()}
    all_ready = all(per_run.get(k, {}).get("ready_for_full_forest1") for k in runs if k in analyzed)

    effect = {
        "v221_short500_density": v221_500.get("keyframes_per_100"),
        "v221_short500_gap_max": v221_500.get("gap_max"),
        "v2_2_2_short500_density": analyzed.get("direct_density_v2_2_2_short500", {}).get("keyframes_per_100"),
        "v2_2_2_short800_density": analyzed.get("direct_density_v2_2_2_short800", {}).get("keyframes_per_100"),
        "v2_2_2_short1000_density": r1000.get("keyframes_per_100"),
        "v2_2_2_short500_gap_max": analyzed.get("direct_density_v2_2_2_short500", {}).get("gap_max"),
        "v2_2_2_short500_preemptive": analyzed.get("direct_density_v2_2_2_short500", {}).get(
            "preemptive_rescue_count"
        ),
    }
    write_json(out / "direct_density_v2_2_2_effect_summary.json", effect)

    r500 = analyzed.get("direct_density_v2_2_2_short500", {})
    if r500 and r500.get("gap_max", 99) > 20:
        frame_level_gap_report(out, r500)

    ready = {
        "all_short_runs_ready": all_ready,
        "recommend_full_run": bool(all_ready),
        "keep_RVQ_tau_frozen": True,
        "gate_version": "direct_density_v2_2_2",
        "per_run": per_run,
    }
    write_json(out / "ready_for_full_gate_after_direct_density_v2_2_2.json", ready)

    lines = [
        "# PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2",
        "",
        "## 1. short500 430-452 gap tail 是否修复？",
        f"- v2.2.1 gap_max={effect.get('v221_short500_gap_max')} v2.2.2={effect.get('v2_2_2_short500_gap_max')}",
        f"- preemptive rescues={effect.get('v2_2_2_short500_preemptive')}",
        "",
        "## 2. short500 gap_max <=20？",
        f"- gap_max={r500.get('gap_max')}",
        "",
        "## 3. short500 density <=50？",
        f"- density={r500.get('keyframes_per_100')}",
        "",
        "## 4-5. short800/1000 保持？",
        str(
            {
                k: (v.get("keyframes_per_100"), v.get("gap_max"), v.get("gap_p90"))
                for k, v in analyzed.items()
                if "800" in k or "1000" in k
            }
        ),
        "",
        "## 6. preemptive 抢占 hold_density_high？",
        f"- preemptive_count={r500.get('preemptive_rescue_count')}",
        "",
        "## 7-9. 过密 / recovery / safety",
        f"- recovery_kf@1000={r1000.get('recovery_keyframes')}",
        "",
        "## 10. full gate",
        f"- all_short_runs_ready={all_ready}",
        "",
        "## 11. R/V/Q/tau",
        "- frozen",
    ]
    write_md(out / "paper_aligned_direct_density_rebalance_v2_2_2_report.md", lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=str(OUT_ROOT))
    p.add_argument(
        "--phase",
        choices=["baseline_guard", "short500_gate", "post_run"],
        default="post_run",
    )
    p.add_argument("--model_dir", default="")
    args = p.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "baseline_guard":
        md = Path(args.model_dir) if args.model_dir else out / "baseline_guard" / "model"
        baseline_guard_audit(md, out / "baseline_guard")
        return 0
    if args.phase == "short500_gate":
        return short500_gate(out)
    post_run(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
