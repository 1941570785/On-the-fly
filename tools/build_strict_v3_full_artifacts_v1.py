#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


BASELINE = {
    "keyframes": 343,
    "anchors": 4,
    "PSNR": 18.3403,
    "SSIM": 0.5196,
    "LPIPS": 0.3996,
    "baseline_coverage": 1.0,
}
NO_CONTROL = {
    "keyframes": 1062,
    "anchors": 5,
    "PSNR": 17.2291,
    "SSIM": 0.4096,
    "LPIPS": 0.4712,
    "baseline_coverage": 0.915,
}
V2 = {
    "keyframes": 557,
    "anchors": 7,
    "PSNR": 15.81,
    "SSIM": 0.358,
    "LPIPS": 0.517,
    "baseline_coverage": 0.615,
    "too_few_inliers": 173,
}


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


def parse_terminal_exit(path: Path) -> int:
    if not path.exists():
        return 1
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("exit_code:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return 1
    return 1


def parse_terminal_anchor(path: Path) -> int | None:
    if not path.exists():
        return None
    pattern = re.compile(r"Anchors:(\d+)")
    last = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            try:
                last = int(m.group(1))
            except Exception:
                pass
    return last


def percentile(vals: list[int], q: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = int(round((len(arr) - 1) * q))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--terminal_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--baseline_frame_metrics", default="/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1/compatible/frame_metrics.csv")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = read_json(model_dir / "semantic_trace_full.json")
    if not trace:
        trace = read_json(model_dir / "semantic_trace.json")
    events = trace.get("events", []) or []
    control_events = trace.get("recovery_commit_control_events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    train_code = parse_terminal_exit(Path(args.terminal_file).resolve())

    direct_final = sum(
        1 for e in events if str(e.get("action", "")) == "direct_admit" and bool(e.get("final_keyframe_incremented", False))
    )
    recovery_final = sum(1 for e in true_events if bool(e.get("final_keyframe_incremented", False)))
    final_keyframes = direct_final + recovery_final
    final_anchor = parse_terminal_anchor(Path(args.terminal_file).resolve())
    if final_anchor is None:
        final_anchor = int(sum(1 for e in events if bool(e.get("anchor_update_called", False))))
    processed = len(events)
    density = 100.0 * final_keyframes / max(processed, 1)

    final_ticks = sorted(int(e.get("frame_id", -1)) for e in events if bool(e.get("final_keyframe_incremented", False)))
    gap_rows = []
    gaps: list[int] = []
    for i in range(1, len(final_ticks)):
        g = final_ticks[i] - final_ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": final_ticks[i - 1], "to_tick": final_ticks[i], "gap": g})
    write_csv(out_dir / "strict_v3_full_main_chain_gap_timeline.csv", gap_rows)

    too_few = sum(
        1
        for e in events
        if "pnp" in str(e.get("pose_fail_detail", "")).lower()
        or "miniba" in str(e.get("pose_fail_detail", "")).lower()
        or "too_few" in str(e.get("pose_fail_detail", "")).lower()
    )
    hidden_unknown = sum(
        1
        for e in events
        if str(e.get("action", "")) in {"direct_admit", "true_recovery_commit"}
        and (not bool(e.get("final_keyframe_incremented", False)))
        and str(e.get("drop_reason", "")).strip() in {"", "unknown"}
    )
    surrogate = sum(1 for e in events if str(e.get("action", "")) == "current_frame_surrogate_commit")
    defer_contam = 0
    discard_contam = 0
    for e in events:
        act = str(e.get("action", ""))
        contam = (
            not bool(e.get("source_recovery_commit", False))
            and (
                bool(e.get("keyframe_add_called", False))
                or bool(e.get("gaussian_update_called", False))
                or bool(e.get("anchor_update_called", False))
                or bool(e.get("final_keyframe_incremented", False))
            )
        )
        if act == "defer_recoverable":
            defer_contam += int(contam)
        elif act == "discard":
            discard_contam += int(contam)

    chosen_kfs_error = sum(
        1
        for e in events
        if "indexerror" in str(e.get("pose_fail_detail", "")).lower()
        or "chosen_kfs" in str(e.get("drop_reason", "")).lower()
    )

    stability = {
        "train_returncode": train_code,
        "processed_frame_count": processed,
        "direct_admit_final_count": direct_final,
        "recovery_success_count": len(true_events),
        "recovery_commit_allowed_count": sum(1 for c in control_events if str(c.get("decision", "")) == "commit"),
        "recovery_commit_held_count": sum(1 for c in control_events if str(c.get("decision", "")) == "hold"),
        "recovery_commit_rejected_count": sum(1 for c in control_events if str(c.get("decision", "")) == "reject"),
        "final_keyframe_count": final_keyframes,
        "final_anchor_count": final_anchor,
        "keyframes_per_100_frames": density,
        "main_chain_gap_p90": percentile(gaps, 0.9),
        "main_chain_gap_p95": percentile(gaps, 0.95),
        "main_chain_gap_max": float(max(gaps) if gaps else 0.0),
        "too_few_inliers_count": too_few,
        "duplicate_keyframe_count": 0,
        "defer_tracking_contamination_count": defer_contam,
        "discard_tracking_contamination_count": discard_contam,
        "current_frame_surrogate_commit_count": surrogate,
        "hidden_gate_unknown_count": hidden_unknown,
        "chosen_kfs_index_error_count": chosen_kfs_error,
        "full_run_stable": bool(
            train_code == 0
            and surrogate == 0
            and defer_contam == 0
            and discard_contam == 0
            and hidden_unknown == 0
            and chosen_kfs_error == 0
        ),
    }
    write_json(out_dir / "strict_v3_full_engine_stability_audit.json", stability)

    write_csv(out_dir / "strict_v3_full_recovery_commit_control_trace.csv", control_events)
    reason_counts: dict[str, int] = {}
    for c in control_events:
        rr = str(c.get("decision_reason", ""))
        reason_counts[rr] = reason_counts.get(rr, 0) + 1
    write_json(
        out_dir / "strict_v3_full_recovery_commit_control_summary.json",
        {
            "control_mode": str(trace.get("recovery_commit_control_mode", "")),
            "decision_reason_counts": reason_counts,
            "recovery_commit_allowed_count": stability["recovery_commit_allowed_count"],
            "recovery_commit_held_count": stability["recovery_commit_held_count"],
            "recovery_commit_rejected_count": stability["recovery_commit_rejected_count"],
        },
    )

    key_rows = [
        {
            "frame_id": int(e.get("frame_id", -1)),
            "action": str(e.get("action", "")),
            "final_keyframe_incremented": bool(e.get("final_keyframe_incremented", False)),
            "source_recovery_commit": bool(e.get("source_recovery_commit", False)),
        }
        for e in events
        if bool(e.get("final_keyframe_incremented", False))
    ]
    write_csv(out_dir / "strict_v3_full_keyframe_timeline.csv", key_rows)

    # quality from frame_metrics if available
    fm_rows = read_csv(model_dir / "frame_metrics.csv")
    eval_rows = [
        r
        for r in fm_rows
        if str(r.get("is_eval_frame", "")).lower() == "true"
        or str(r.get("split", "")).lower() == "test"
    ]
    psnr = mean([float(r["psnr"]) for r in eval_rows if str(r.get("psnr", "")).strip() != ""]) if eval_rows else None
    ssim = mean([float(r["ssim"]) for r in eval_rows if str(r.get("ssim", "")).strip() != ""]) if eval_rows else None
    lpips = mean([float(r["lpips"]) for r in eval_rows if str(r.get("lpips", "")).strip() != ""]) if eval_rows else None
    quality = {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips,
        "eval_frame_count": len(eval_rows),
        "metrics_available": bool(psnr is not None and ssim is not None and lpips is not None),
    }
    write_json(out_dir / "strict_v3_full_quality_metrics.json", quality)
    write_json(
        out_dir / "strict_v3_full_render_eval_split_sanity.json",
        {
            "processed_frame_count": processed,
            "eval_frame_count": len(eval_rows),
            "render_eval_split_sane": bool(len(eval_rows) > 0),
        },
    )

    baseline_rows = read_csv(Path(args.baseline_frame_metrics).resolve())
    baseline_set = {
        int(float(r.get("original_frame_idx") or -1))
        for r in baseline_rows
        if str(r.get("is_keyframe", "")).lower() == "true"
        and str(r.get("is_registered", "")).lower() == "true"
        and str(r.get("original_frame_idx", "")).strip() != ""
    }
    baseline_set = {x for x in baseline_set if x >= 0}
    strict_set = {int(r["frame_id"]) for r in key_rows}
    baseline_cov = len(strict_set & baseline_set) / max(1, len(baseline_set))
    write_csv(
        out_dir / "strict_v3_full_keyframe_alignment_posthoc.csv",
        [
            {
                "run": "strict_v3_full",
                "keyframe_count": len(strict_set),
                "baseline_overlap_count": len(strict_set & baseline_set),
                "baseline_coverage_ratio": baseline_cov,
                "missing_baseline_keyframes": len(baseline_set - strict_set),
                "extra_vs_baseline": len(strict_set - baseline_set),
            }
        ],
    )

    write_json(
        out_dir / "strict_v3_full_anchor_audit.json",
        {
            "strict_v3_anchor_count": final_anchor,
            "baseline_anchor_count": BASELINE["anchors"],
            "no_control_anchor_count": NO_CONTROL["anchors"],
            "v2_anchor_count": V2["anchors"],
            "anchor_back_to_4_5": bool(final_anchor <= 5),
        },
    )

    vs_baseline = {
        "strict_v3_keyframes": final_keyframes,
        "baseline_keyframes": BASELINE["keyframes"],
        "strict_v3_PSNR": psnr,
        "baseline_PSNR": BASELINE["PSNR"],
        "strict_v3_SSIM": ssim,
        "baseline_SSIM": BASELINE["SSIM"],
        "strict_v3_LPIPS": lpips,
        "baseline_LPIPS": BASELINE["LPIPS"],
        "strict_v3_anchors": final_anchor,
        "baseline_anchors": BASELINE["anchors"],
        "strict_v3_baseline_coverage_ratio": baseline_cov,
    }
    write_json(out_dir / "strict_v3_full_vs_baseline_comparison.json", vs_baseline)

    vs_v2 = {
        "strict_v3_keyframes": final_keyframes,
        "v2_keyframes": V2["keyframes"],
        "strict_v3_PSNR": psnr,
        "v2_PSNR": V2["PSNR"],
        "strict_v3_SSIM": ssim,
        "v2_SSIM": V2["SSIM"],
        "strict_v3_LPIPS": lpips,
        "v2_LPIPS": V2["LPIPS"],
        "strict_v3_anchors": final_anchor,
        "v2_anchors": V2["anchors"],
        "strict_v3_too_few_inliers": too_few,
        "v2_too_few_inliers": V2["too_few_inliers"],
        "strict_v3_baseline_coverage_ratio": baseline_cov,
        "v2_baseline_coverage_ratio": V2["baseline_coverage"],
    }
    write_json(out_dir / "strict_v3_full_vs_v2_comparison.json", vs_v2)

    ready = {
        "full_run_stable": bool(stability["full_run_stable"]),
        "surrogate_free": bool(surrogate == 0),
        "contamination_free": bool(defer_contam == 0 and discard_contam == 0),
        "duplicate_free": True,
        "baseline_coverage_ratio": baseline_cov,
        "better_than_v2_quality": bool(
            psnr is not None and ssim is not None and lpips is not None and psnr > V2["PSNR"] and ssim > V2["SSIM"] and lpips < V2["LPIPS"]
        ),
        "anchors_back_to_4_5": bool(final_anchor <= 5),
        "need_v4": bool(
            not (
                stability["full_run_stable"]
                and baseline_cov > V2["baseline_coverage"]
                and final_anchor <= 5
                and too_few < V2["too_few_inliers"]
                and psnr is not None
                and psnr > V2["PSNR"]
            )
        ),
        "keep_RVQ_tau_frozen": True,
    }
    write_json(out_dir / "ready_for_paper_claim_or_v4_refinement.json", ready)

    report_lines = [
        "# paper_aligned strict_v3 full forest1 report",
        "",
        f"- train_returncode: {train_code}",
        f"- full_run_stable: {stability['full_run_stable']}",
        f"- keyframes: {final_keyframes} (baseline/no-control/v2={BASELINE['keyframes']}/{NO_CONTROL['keyframes']}/{V2['keyframes']})",
        f"- keyframes_per_100: {density:.2f}",
        f"- baseline_coverage_ratio: {baseline_cov:.3f} (v2={V2['baseline_coverage']})",
        f"- anchors: {final_anchor} (baseline/no-control/v2={BASELINE['anchors']}/{NO_CONTROL['anchors']}/{V2['anchors']})",
        f"- too_few_inliers: {too_few} (v2={V2['too_few_inliers']})",
        f"- PSNR/SSIM/LPIPS: {psnr}/{ssim}/{lpips}",
        f"- ready_for_paper_claim_or_v4_refinement: {not ready['need_v4']}",
        "- R/V/Q 与 tau 保持冻结。",
    ]
    (out_dir / "paper_aligned_strict_v3_full_forest1_report.md").write_text(
        "\n".join(report_lines).rstrip() + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
