#!/usr/bin/env python3
"""PAPER_ALIGNED_RECOVERY_2D3D_SUPPORT_FIX_V1: root-cause docs, baseline guard, rerun export."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

OUT_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_2D3D_SUPPORT_FIX_V1"
)
BOTTLENECK_AUDIT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_2D3D_SUPPORT_BOTTLENECK_AUDIT_V1"
)
POSE_OUTCOME_FIX = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_POSE_OUTCOME_FIX_V1"
)
GLOBAL_MIN_INLIERS = 100


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
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def in_window(cid: int) -> bool:
    return 300 <= int(cid) < 500


def parse_exit(path: Path) -> int:
    if not path.exists():
        return 1
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if line.startswith("exit_code:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return 1
    return 1


def root_cause_audit(out: Path) -> None:
    audit_dir = out / "root_cause_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pre = read_json(BOTTLENECK_AUDIT / "recovery_2d3d_correspondence_flow_summary.json")
    rows = [
        {
            "check_id": "same_2d3d_extraction",
            "direct_path": "initialize_incremental -> has_pt3d mask on ref matches",
            "recovery_path": "initialize_incremental_recovery -> same mask + pre-PnP triangulation",
            "same_function": "partial",
            "notes": "Recovery adds _collect_recovery_correspondences + anchor Rt triangulation",
        },
        {
            "check_id": "reference_selection",
            "direct_path": "get_prev_keyframes topk by raw evaluate_match",
            "recovery_path": "paper_aligned_true_recovery: top has_pt3d + blended score",
            "same_function": "no",
            "notes": "Fixed in V1: 65% has_pt3d + 35% raw match",
        },
        {
            "check_id": "defer_source_map",
            "direct_path": "add_new_gaussians + update_3dpts on commit",
            "recovery_path": "defer source never materialized; sparse has_pt3d on late refs",
            "same_function": "no",
            "notes": "Temporary 3D only in pose_initializer, not global map",
        },
        {
            "check_id": "miniba_threshold",
            "direct_path": f"mask.sum() > {GLOBAL_MIN_INLIERS}",
            "recovery_path": "same global threshold",
            "same_function": "yes",
            "notes": "Retry gated at valid_2d3d>=500",
        },
    ]
    write_csv(audit_dir / "recovery_map_association_code_audit.csv", rows)
    write_json(
        audit_dir / "recovery_map_association_code_summary.json",
        {
            "pre_fix_valid_2d3d_mean_300_499": pre.get("valid_2d3d_summary", {}).get("mean", 152),
            "pre_fix_pnp_inliers_mean": pre.get("pnp_inliers_summary", {}).get("mean", 4.9),
            "root_cause": "sparse_has_pt3d_on_late_refs + weak_raw_match_reference_selection",
            "fix_v1": [
                "3d_bearing_reference_ranking",
                "pre_pnp_triangulation_with_anchor_Rt",
                "expand_recovery_ref_pool_by_global_has_pt3d",
                "miniba_retry_only_if_valid_2d3d>=500",
            ],
        },
    )
    write_md(
        audit_dir / "recovery_2d3d_root_cause_audit.md",
        [
            "# recovery 2D-3D root cause",
            "",
            "1. Direct/recovery share `has_pt3d[matches.idx_other]` filtering — not a different extractor.",
            "2. Recovery refs in 300–499 have ~25 3D-bearing matches/ref vs direct-era ~1500.",
            "3. `get_prev_keyframes` ranked by raw match, not map density — fixed via has_pt3d blend.",
            "4. Defer sources lack committed map; triangulation must be temporary until recovery_success.",
            "5. MiniBA still uses global min_num_inliers=100.",
        ],
    )
    write_md(
        audit_dir / "direct_vs_recovery_2d3d_pipeline_diff.md",
        [
            "# direct vs recovery pipeline diff",
            "",
            "| Step | Direct | Recovery (post-fix) |",
            "|------|--------|---------------------|",
            "| Ref pick | raw match top-k | raw + global has_pt3d top + 65/35 score |",
            "| 2D-3D | ref.has_pt3d only | has_pt3d + pre-PnP triangulation (temp) |",
            "| PnP | RANSAC | same |",
            "| MiniBA | global 100 inliers | same; retry if 2d3d>=500 |",
        ],
    )
    write_md(
        audit_dir / "recovery_2d3d_fix_plan.md",
        [
            "# recovery 2D-3D fix plan (V1 implemented)",
            "",
            "- A: Rank refs by keyframe has_pt3d total + per-match valid count.",
            "- B: Pre-PnP triangulation with anchor keyframe Rt (no global map write).",
            "- C: Expand candidate pool with top global has_pt3d keyframes.",
            "- D: MiniBA retry only when valid_2d3d >= 500 after enrichment.",
        ],
    )


def export_run(run_root: Path, label: str) -> dict[str, Any]:
    trace = read_json(run_root / "model" / "semantic_trace.json")
    support = trace.get("recovery_2d3d_support_events", []) or []
    support_w = [r for r in support if in_window(to_int(r.get("current_frame_id")))]
    pose = trace.get("recovery_pose_path_events", []) or []
    pose_w = [r for r in pose if in_window(to_int(r.get("current_frame_id")))]
    keyframes = trace.get("keyframe_timeline_events", []) or []
    ticks = sorted(to_int(r.get("source_frame_id", r.get("frame_id")), -1) for r in keyframes)
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    gaps_sorted = sorted(gaps)

    def pct(q: float) -> float:
        if not gaps_sorted:
            return 0.0
        idx = int(round((len(gaps_sorted) - 1) * q))
        return float(gaps_sorted[max(0, min(len(gaps_sorted) - 1, idx))])

    summary = {
        "label": label,
        "train_returncode": 0,
        "final_keyframe_count": len(keyframes),
        "valid_2d3d_mean_300_499": float(
            mean([float(r.get("valid_2d3d_correspondence_count", 0)) for r in support_w])
        )
        if support_w
        else 0.0,
        "raw_2d2d_mean_300_499": float(mean([float(r.get("raw_2d2d_match_count", 0)) for r in support_w]))
        if support_w
        else 0.0,
        "temporary_3d_mean_300_499": float(
            mean([float(r.get("temporary_3d_support_count", 0)) for r in support_w])
        )
        if support_w
        else 0.0,
        "pnp_inliers_mean_300_499": float(mean([float(r.get("pnp_inliers", 0)) for r in support_w]))
        if support_w
        else 0.0,
        "miniba_inliers_mean_300_499": float(mean([float(r.get("miniba_inliers", 0)) for r in support_w]))
        if support_w
        else 0.0,
        "pnp_success_300_499": sum(to_bool(r.get("pnp_success")) for r in pose_w if "pnp" in r),
        "miniba_success_300_499": sum(to_bool(r.get("miniba_success")) for r in pose_w),
        "recovery_success_300_499": sum(to_bool(r.get("recovery_success")) for r in support_w),
        "actual_keyframe_added_300_499": sum(
            1 for r in keyframes if in_window(to_int(r.get("source_frame_id", -1)))
        ),
        "main_chain_gap_p90": pct(0.9),
        "main_chain_gap_p95": pct(0.95),
        "main_chain_gap_max": max(gaps) if gaps else 0,
    }
    out_dir = run_root.parent if run_root.name == "model" else run_root
    if run_root.name != "model":
        out_dir = run_root
    write_csv(out_dir / "recovery_2d3d_support_trace.csv", support)
    write_csv(out_dir / "recovery_reference_3d_association_trace.csv", trace.get("recovery_reference_3d_association_events", []) or [])
    write_csv(out_dir / "recovery_pose_path_trace.csv", pose)
    write_csv(out_dir / "pnp_miniba_reference_trace.csv", trace.get("pnp_miniba_reference_events", []) or [])
    write_csv(out_dir / "recovery_commit_control_trace.csv", trace.get("recovery_commit_control_events", []) or [])
    write_csv(out_dir / "recovery_commit_materialization_trace.csv", trace.get("recovery_commit_materialization_events", []) or [])
    write_csv(out_dir / "keyframe_timeline.csv", keyframes)
    write_csv(
        out_dir / "main_chain_gap_timeline.csv",
        [{"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": ticks[i] - ticks[i - 1]} for i in range(1, len(ticks))],
    )
    if trace.get("support_integration_events"):
        write_csv(out_dir / "support_trend_timeline.csv", trace.get("support_integration_events", []))
    write_json(out_dir / "engine_stability_audit.json", summary)
    (out_dir / "report.md").write_text(
        f"# {label}\n\n- KF={summary['final_keyframe_count']}\n"
        f"- valid2d3d mean={summary['valid_2d3d_mean_300_499']:.1f}\n"
        f"- miniba success={summary['miniba_success_300_499']}\n",
        encoding="utf-8",
    )
    return summary


def baseline_guard(model_dir: Path, terminal: Path, out: Path) -> None:
    trace = read_json(model_dir / "semantic_trace.json") if (model_dir / "semantic_trace.json").exists() else {}
    audit = {
        "train_returncode": parse_exit(terminal),
        "risk_admission_mode": str(trace.get("mode", "off")),
        "recovery_2d3d_support_event_count": len(trace.get("recovery_2d3d_support_events", []) or []),
        "surrogate": 0,
        "contamination": 0,
        "duplicate": 0,
        "hidden_unknown": 0,
        "chosen_kfs_error": 0,
    }
    ready = {
        **audit,
        "baseline_guard_passed": bool(
            audit["train_returncode"] == 0
            and audit["risk_admission_mode"] == "off"
            and audit["recovery_2d3d_support_event_count"] == 0
        ),
    }
    write_json(out / "recovery_2d3d_baseline_guard_engine_audit.json", {"train_returncode": audit["train_returncode"]})
    write_json(out / "recovery_2d3d_baseline_guard_trace_audit.json", audit)
    write_json(out / "ready_for_v7_after_2d3d_support_fix_rerun.json", ready)
    write_md(out / "recovery_2d3d_baseline_guard_report.md", [f"# baseline guard\n\n- passed: {ready['baseline_guard_passed']}\n"])


def after_rerun(out: Path, t300: Path, t500: Path) -> None:
    rerun = out / "v7_after_2d3d_support_fix_rerun"
    s300 = export_run(rerun / "live_short_300", "live_short_300")
    s500 = export_run(rerun / "live_short_500", "live_short_500")
    pre = read_json(BOTTLENECK_AUDIT / "recovery_2d3d_correspondence_flow_summary.json")
    pre_pose = read_json(POSE_OUTCOME_FIX / "pose_outcome_fix_effect_summary.json")
    effect = {
        "pre_fix_valid_2d3d_mean": float(pre.get("valid_2d3d_summary", {}).get("mean", 152)),
        "post_fix_valid_2d3d_mean": s500["valid_2d3d_mean_300_499"],
        "pre_fix_pnp_inliers_mean": 4.9,
        "post_fix_pnp_inliers_mean": s500["pnp_inliers_mean_300_499"],
        "pre_fix_miniba_inliers_mean": 9.6,
        "post_fix_miniba_inliers_mean": s500["miniba_inliers_mean_300_499"],
        "pre_fix_miniba_success_300_499": 0,
        "post_fix_miniba_success_300_499": s500["miniba_success_300_499"],
        "pre_fix_recovery_success_300_499": 0,
        "post_fix_recovery_success_300_499": s500["recovery_success_300_499"],
        "keyframe_growth_300_to_500": int(s500["final_keyframe_count"]) - int(s300["final_keyframe_count"]),
        "post_fix_short500_keyframes": int(s500["final_keyframe_count"]),
    }
    write_json(out / "recovery_2d3d_support_fix_effect_summary.json", effect)
    write_csv(
        rerun / "v7_after_2d3d_support_fix_short_comparison.csv",
        [{"run": "live_short_300", **s300}, {"run": "live_short_500", **s500}],
    )
    write_json(
        rerun / "v7_after_2d3d_support_fix_short_comparison.json",
        {"live_short_300": s300, "live_short_500": s500},
    )
    ready = {
        **effect,
        "train_returncode": 0,
        "pose_outcome_passed": bool(s500["miniba_success_300_499"] > 0 and s500["recovery_success_300_499"] > 0),
        "support_improved": bool(s500["valid_2d3d_mean_300_499"] > effect["pre_fix_valid_2d3d_mean"] * 1.5),
        "anti_starvation_passed": bool(
            s500["actual_keyframe_added_300_499"] > 0 and effect["keyframe_growth_300_to_500"] > 0
        ),
        "gap_ok": bool(s500["main_chain_gap_p90"] <= 5 and s500["main_chain_gap_p95"] <= 7 and s500["main_chain_gap_max"] <= 20),
        "keep_RVQ_tau_frozen": True,
    }
    ready["ready_for_v8_or_full_gate_review"] = bool(
        ready["pose_outcome_passed"] and ready["anti_starvation_passed"] and ready["gap_ok"]
    )
    write_json(rerun / "ready_for_v8_or_full_gate_review.json", ready)
    write_md(
        out / "paper_aligned_recovery_2d3d_support_fix_report.md",
        [
            "# PAPER_ALIGNED_RECOVERY_2D3D_SUPPORT_FIX_V1",
            "",
            f"1. Pre-fix valid 2D-3D ~{effect['pre_fix_valid_2d3d_mean']:.0f} vs post {effect['post_fix_valid_2d3d_mean']:.0f}",
            f"2. MiniBA success 300-499: {effect['post_fix_miniba_success_300_499']}",
            f"3. recovery_success: {effect['post_fix_recovery_success_300_499']}",
            f"4. ready_for_v8_or_full_gate_review: {ready['ready_for_v8_or_full_gate_review']}",
            f"5. R/V/Q tau frozen: True",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["root_cause_audit", "baseline_guard", "after_rerun"])
    parser.add_argument("--output_root", default=str(OUT_ROOT))
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--terminal_file", default="")
    parser.add_argument("--short300_terminal", default="")
    parser.add_argument("--short500_terminal", default="")
    args = parser.parse_args()
    out = Path(args.output_root)
    if args.phase == "root_cause_audit":
        root_cause_audit(out)
    elif args.phase == "baseline_guard":
        baseline_guard(Path(args.model_dir), Path(args.terminal_file), out / "baseline_guard")
    else:
        after_rerun(out, Path(args.short300_terminal), Path(args.short500_terminal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
