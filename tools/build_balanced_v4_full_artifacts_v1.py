#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


BASELINE = {"keyframes": 343, "anchors": 4}
V2 = {"keyframes": 557, "anchors": 7, "baseline_coverage": 0.615, "too_few_inliers": 173}
V3 = {"keyframes": 237, "anchors": 3}


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

    direct_final = sum(1 for e in events if str(e.get("action", "")) == "direct_admit" and bool(e.get("final_keyframe_incremented", False)))
    recovery_final = sum(1 for e in true_events if bool(e.get("final_keyframe_incremented", False)))
    final_keyframes = direct_final + recovery_final
    processed = len(events)
    density = 100.0 * final_keyframes / max(processed, 1)
    final_anchor = parse_terminal_anchor(Path(args.terminal_file).resolve())
    if final_anchor is None:
        final_anchor = int(sum(1 for e in events if bool(e.get("anchor_update_called", False))))

    ticks = sorted(int(e.get("frame_id", -1)) for e in events if bool(e.get("final_keyframe_incremented", False)))
    gaps: list[int] = []
    gap_rows: list[dict[str, Any]] = []
    for i in range(1, len(ticks)):
        g = ticks[i] - ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": g})
    write_csv(out_dir / "balanced_v4_full_main_chain_gap_timeline.csv", gap_rows)

    too_few = sum(
        1
        for e in events
        if "pnp" in str(e.get("pose_fail_detail", "")).lower()
        or "miniba" in str(e.get("pose_fail_detail", "")).lower()
        or "too_few" in str(e.get("pose_fail_detail", "")).lower()
    )
    surrogate = sum(1 for e in events if str(e.get("action", "")) == "current_frame_surrogate_commit")
    defer_contam = 0
    discard_contam = 0
    hidden_unknown = 0
    chosen_kfs_error = 0
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
        if act in {"direct_admit", "true_recovery_commit"} and (not bool(e.get("final_keyframe_incremented", False))):
            if str(e.get("drop_reason", "")).strip() in {"", "unknown"}:
                hidden_unknown += 1
        if "indexerror" in str(e.get("pose_fail_detail", "")).lower() or "chosen_kfs" in str(e.get("drop_reason", "")).lower():
            chosen_kfs_error += 1

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
    write_json(out_dir / "balanced_v4_full_engine_stability_audit.json", stability)
    write_csv(out_dir / "balanced_v4_full_recovery_commit_control_trace.csv", control_events)

    reason_counts: dict[str, int] = {}
    for c in control_events:
        rr = str(c.get("decision_reason", ""))
        reason_counts[rr] = reason_counts.get(rr, 0) + 1
    write_json(
        out_dir / "balanced_v4_full_recovery_commit_control_summary.json",
        {"control_mode": str(trace.get("recovery_commit_control_mode", "")), "decision_reason_counts": reason_counts},
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
    write_csv(out_dir / "balanced_v4_full_keyframe_timeline.csv", key_rows)

    fm_rows = read_csv(model_dir / "frame_metrics.csv")
    eval_rows = [
        r
        for r in fm_rows
        if str(r.get("is_eval_frame", "")).lower() == "true" or str(r.get("split", "")).lower() == "test"
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
    write_json(out_dir / "balanced_v4_full_quality_metrics.json", quality)
    write_json(
        out_dir / "balanced_v4_full_render_eval_split_sanity.json",
        {"processed_frame_count": processed, "eval_frame_count": len(eval_rows), "render_eval_split_sane": bool(len(eval_rows) > 0)},
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
    v4_set = {int(r["frame_id"]) for r in key_rows}
    baseline_cov = len(v4_set & baseline_set) / max(1, len(baseline_set))
    write_csv(
        out_dir / "balanced_v4_full_keyframe_alignment_posthoc.csv",
        [
            {
                "run": "balanced_v4_full",
                "keyframe_count": len(v4_set),
                "baseline_overlap_count": len(v4_set & baseline_set),
                "baseline_coverage_ratio": baseline_cov,
                "missing_baseline_keyframes": len(baseline_set - v4_set),
                "extra_vs_baseline": len(v4_set - baseline_set),
            }
        ],
    )

    write_json(
        out_dir / "balanced_v4_full_anchor_audit.json",
        {
            "balanced_v4_anchor_count": final_anchor,
            "baseline_anchor_count": BASELINE["anchors"],
            "v2_anchor_count": V2["anchors"],
            "v3_anchor_count": V3["anchors"],
            "anchor_in_target_or_nearby": bool(4 <= final_anchor <= 6),
        },
    )
    write_json(
        out_dir / "balanced_v4_full_vs_baseline_comparison.json",
        {
            "balanced_v4_keyframes": final_keyframes,
            "baseline_keyframes": BASELINE["keyframes"],
            "balanced_v4_anchors": final_anchor,
            "baseline_anchors": BASELINE["anchors"],
            "balanced_v4_baseline_coverage_ratio": baseline_cov,
            "balanced_v4_PSNR": psnr,
            "balanced_v4_SSIM": ssim,
            "balanced_v4_LPIPS": lpips,
        },
    )
    write_json(
        out_dir / "balanced_v4_full_vs_v2_v3_comparison.json",
        {
            "balanced_v4_keyframes": final_keyframes,
            "v2_keyframes": V2["keyframes"],
            "v3_keyframes": V3["keyframes"],
            "balanced_v4_anchors": final_anchor,
            "v2_anchors": V2["anchors"],
            "v3_anchors": V3["anchors"],
            "balanced_v4_too_few_inliers": too_few,
            "v2_too_few_inliers": V2["too_few_inliers"],
            "balanced_v4_baseline_coverage_ratio": baseline_cov,
            "v2_baseline_coverage_ratio": V2["baseline_coverage"],
            "balanced_v4_PSNR": psnr,
            "balanced_v4_SSIM": ssim,
            "balanced_v4_LPIPS": lpips,
        },
    )

    ready_for_claim = bool(
        stability["full_run_stable"]
        and 330 <= final_keyframes <= 500
        and density <= 45.0
        and 4 <= final_anchor <= 6
        and float(stability["main_chain_gap_max"]) <= 20.0
        and baseline_cov > V2["baseline_coverage"]
        and too_few < V2["too_few_inliers"]
        and bool(quality["metrics_available"])
    )
    write_json(
        out_dir / "ready_for_paper_claim_or_v5_refinement.json",
        {
            "ready_for_paper_claim": ready_for_claim,
            "need_v5": not ready_for_claim,
            "keep_RVQ_tau_frozen": True,
            "quality_metrics_available": bool(quality["metrics_available"]),
        },
    )
    (out_dir / "paper_aligned_balanced_v4_full_forest1_report.md").write_text(
        "\n".join(
            [
                "# paper_aligned balanced_v4 full forest1 report",
                "",
                f"- train_returncode: {train_code}",
                f"- full_run_stable: {stability['full_run_stable']}",
                f"- keyframes/density: {final_keyframes}/{density:.3f}",
                f"- anchors: {final_anchor}",
                f"- gap p90/p95/max: {stability['main_chain_gap_p90']}/{stability['main_chain_gap_p95']}/{stability['main_chain_gap_max']}",
                f"- baseline coverage: {baseline_cov:.3f}",
                f"- too_few_inliers: {too_few}",
                f"- PSNR/SSIM/LPIPS: {psnr}/{ssim}/{lpips}",
                f"- ready_for_paper_claim: {ready_for_claim}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
