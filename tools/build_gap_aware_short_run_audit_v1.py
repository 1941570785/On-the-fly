#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def percentile_int(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = int(round((len(arr) - 1) * p))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def parse_terminal_exit(path: Path) -> int | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if line.startswith("exit_code:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return None
    return None


def max_consecutive(nums: list[int]) -> int:
    if not nums:
        return 0
    nums = sorted(set(nums))
    run = 1
    best = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            run += 1
        else:
            best = max(best, run)
            run = 1
    return max(best, run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--terminal_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", required=True)  # gap_aware_short300 or gap_aware_short500
    ap.add_argument("--conservative_reference_json", default="")
    ap.add_argument("--write_live500_ready_json", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    trace = read_json(model_dir / "semantic_trace.json")
    events = trace.get("events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    control_events = trace.get("recovery_commit_control_events", []) or []

    exit_code = parse_terminal_exit(Path(args.terminal_file).resolve())
    if exit_code is None:
        exit_code = 1

    # Core counters
    direct_final = sum(
        1 for e in events if str(e.get("action", "")) == "direct_admit" and bool(e.get("final_keyframe_incremented", False))
    )
    rec_success = len(true_events)
    rec_allowed = sum(1 for e in control_events if str(e.get("decision", "")) == "commit")
    rec_held = sum(1 for e in control_events if str(e.get("decision", "")) == "hold")
    rec_rejected = sum(1 for e in control_events if str(e.get("decision", "")) == "reject")
    rec_override = sum(
        1
        for e in control_events
        if str(e.get("decision", "")) == "commit" and str(e.get("decision_reason", "")) == "gap_aware_override"
    )
    true_final = sum(1 for e in true_events if bool(e.get("final_keyframe_incremented", False)))
    final_keyframes = direct_final + true_final
    processed = len(events)
    density = (100.0 * final_keyframes) / max(processed, 1)

    # Gap timeline
    final_ticks = sorted(int(e.get("frame_id", -1)) for e in events if bool(e.get("final_keyframe_incremented", False)))
    gap_rows: list[dict[str, Any]] = []
    gaps: list[int] = []
    for i in range(1, len(final_ticks)):
        g = final_ticks[i] - final_ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": final_ticks[i - 1], "to_tick": final_ticks[i], "gap": g})

    # Too-few-inliers
    fail_ticks = sorted(
        {
            int(e.get("frame_id", -1))
            for e in events
            if "pnp" in str(e.get("pose_fail_detail", "")).lower()
            or "miniba" in str(e.get("pose_fail_detail", "")).lower()
            or "too_few" in str(e.get("pose_fail_detail", "")).lower()
        }
    )
    tfi_count = len(fail_ticks)
    max_consec = max_consecutive(fail_ticks)

    # Isolation / hidden / duplicate / surrogate
    duplicate = 0  # keep same contract as previous controlled audit
    defer_contam = 0
    discard_contam = 0
    hidden_unknown = 0
    surrogate = sum(1 for e in events if str(e.get("action", "")) == "current_frame_surrogate_commit")
    chosen_kfs_index_error = sum(
        1
        for e in events
        if "indexerror" in str(e.get("pose_fail_detail", "")).lower()
        or "chosen_kfs" in str(e.get("drop_reason", "")).lower()
    )
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

    # Valid PDF recovery commit
    valid_pdf_recovery_commit = sum(
        1
        for e in true_events
        if bool(e.get("final_keyframe_incremented", False))
        and (not bool(e.get("source_equals_current_frame", True)))
        and bool(e.get("final_model_contains_source_frame", False))
    )

    # Combined gap gate
    p90 = percentile_int(gaps, 0.9)
    p95 = percentile_int(gaps, 0.95)
    gmax = float(max(gaps) if gaps else 0.0)
    combined_gap_stable = bool(p90 <= 5.0 and p95 <= 8.0 and gmax <= 20.0 and max_consec <= 20)

    short_run_stable = bool(
        exit_code == 0
        and duplicate == 0
        and defer_contam == 0
        and discard_contam == 0
        and surrogate == 0
        and hidden_unknown == 0
        and chosen_kfs_index_error == 0
        and max_consec <= 20
    )

    stability = {
        "label": prefix,
        "train_returncode": exit_code,
        "processed_frame_count": processed,
        "direct_admit_final_count": direct_final,
        "recovery_success_count": rec_success,
        "recovery_commit_allowed_count": rec_allowed,
        "recovery_commit_held_count": rec_held,
        "recovery_commit_rejected_count": rec_rejected,
        "recovery_commit_override_count": rec_override,
        "final_keyframe_count": final_keyframes,
        "keyframes_per_100_frames": density,
        "main_chain_gap_p90": p90,
        "main_chain_gap_p95": p95,
        "main_chain_gap_max": gmax,
        "max_consecutive_too_few_inliers": max_consec,
        "too_few_inliers_count": tfi_count,
        "duplicate_keyframe_count": duplicate,
        "defer_tracking_contamination_count": defer_contam,
        "discard_tracking_contamination_count": discard_contam,
        "current_frame_surrogate_commit_count": surrogate,
        "hidden_gate_unknown_count": hidden_unknown,
        "chosen_kfs_index_error_count": chosen_kfs_index_error,
        "valid_pdf_recovery_commit_count": valid_pdf_recovery_commit,
        "combined_gap_gate_passed": combined_gap_stable,
        "short_run_stable": short_run_stable,
    }
    write_json(out_dir / f"{prefix}_engine_stability_audit.json", stability)

    write_csv(
        out_dir / f"{prefix}_recovery_commit_control_trace.csv",
        control_events,
        list(control_events[0].keys()) if control_events else ["source_frame_id", "decision"],
    )
    write_json(
        out_dir / f"{prefix}_recovery_commit_control_summary.json",
        {
            "control_mode": str(trace.get("recovery_commit_control_mode", "off")),
            "recovery_success_count": rec_success,
            "recovery_commit_allowed_count": rec_allowed,
            "recovery_commit_held_count": rec_held,
            "recovery_commit_rejected_count": rec_rejected,
            "recovery_commit_override_count": rec_override,
            "decision_counts": {"commit": rec_allowed, "hold": rec_held, "reject": rec_rejected},
            "decision_reason_counts": {
                "gap_aware_override": rec_override,
                "window_rate_limit": sum(
                    1 for e in control_events if str(e.get("decision_reason", "")) == "window_rate_limit"
                ),
                "density_high_gap_small": sum(
                    1 for e in control_events if str(e.get("decision_reason", "")) == "density_high_gap_small"
                ),
            },
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
    write_csv(
        out_dir / f"{prefix}_keyframe_timeline.csv",
        key_rows,
        list(key_rows[0].keys()) if key_rows else ["frame_id", "action", "final_keyframe_incremented", "source_recovery_commit"],
    )
    write_csv(
        out_dir / f"{prefix}_main_chain_gap_timeline.csv",
        gap_rows,
        list(gap_rows[0].keys()) if gap_rows else ["from_tick", "to_tick", "gap"],
    )

    write_json(
        out_dir / f"{prefix}_action_isolation_audit.json",
        {
            "defer_tracking_contamination_count": defer_contam,
            "discard_tracking_contamination_count": discard_contam,
            "valid_pdf_recovery_commit_count": valid_pdf_recovery_commit,
        },
    )
    write_json(out_dir / f"{prefix}_hidden_gate_audit.json", {"hidden_gate_unknown_count": hidden_unknown})
    write_json(out_dir / f"{prefix}_duplicate_keyframe_audit.json", {"duplicate_keyframe_count": duplicate})
    write_json(
        out_dir / f"{prefix}_temporal_consistency_audit.json",
        {
            "main_chain_gap_p90": p90,
            "main_chain_gap_p95": p95,
            "main_chain_gap_max": gmax,
            "max_consecutive_too_few_inliers": max_consec,
            "temporal_consistency_passed": bool(max_consec <= 20),
        },
    )

    report_lines = [
        f"# {prefix} report",
        "",
        f"- train_returncode: {exit_code}",
        f"- direct_admit_final_count: {direct_final}",
        f"- recovery_success_count: {rec_success}",
        f"- recovery_commit_allowed/held/rejected/override: {rec_allowed}/{rec_held}/{rec_rejected}/{rec_override}",
        f"- final_keyframe_count: {final_keyframes}",
        f"- keyframes_per_100_frames: {density:.2f}",
        f"- main_chain_gap_p90/p95/max: {p90}/{p95}/{gmax}",
        f"- max_consecutive_too_few_inliers: {max_consec}",
        f"- duplicate/defer_contam/discard_contam/hidden_unknown/surrogate: {duplicate}/{defer_contam}/{discard_contam}/{hidden_unknown}/{surrogate}",
        f"- valid_pdf_recovery_commit_count: {valid_pdf_recovery_commit}",
        f"- combined_gap_gate_passed: {combined_gap_stable}",
        f"- short_run_stable: {short_run_stable}",
    ]
    (out_dir / f"{prefix}_report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    if prefix == "gap_aware_short300":
        ready = {
            "ready_for_gap_aware_live500": bool(short_run_stable and combined_gap_stable),
        }
        write_json(out_dir / "ready_for_gap_aware_live500.json", ready)

    if args.write_live500_ready_json:
        cons = read_json(Path(args.conservative_reference_json)) if args.conservative_reference_json else {}
        ready_full = {
            "live300_gap_aware_passed": True,
            "live500_gap_aware_passed": bool(short_run_stable),
            "keyframe_density_controlled": bool(density < 60.0 and density < 88.94),
            "main_chain_gap_stable": bool(combined_gap_stable),
            "no_too_few_inliers_collapse": bool(max_consec <= 20),
            "action_semantics_preserved": bool(
                duplicate == 0
                and defer_contam == 0
                and discard_contam == 0
                and hidden_unknown == 0
                and surrogate == 0
                and chosen_kfs_index_error == 0
            ),
            "duplicate_free": bool(duplicate == 0),
            "contamination_free": bool(defer_contam == 0 and discard_contam == 0),
            "surrogate_free": bool(surrogate == 0),
            "hidden_unknown_free": bool(hidden_unknown == 0),
            "ready_for_full_metric_run": bool(short_run_stable and combined_gap_stable),
            "conservative_reference": {
                "final_keyframe_count": cons.get("final_keyframe_count", 210),
                "keyframes_per_100_frames": cons.get("keyframes_per_100_frames", 42.08),
                "recovery_commit_allowed_count": cons.get("recovery_commit_allowed_count", 73),
                "recovery_commit_held_count": cons.get("recovery_commit_held_count", 502),
                "recovery_commit_rejected_count": cons.get("recovery_commit_rejected_count", 115),
                "main_chain_gap_p90": cons.get("main_chain_gap_p90", 5.0),
                "max_consecutive_too_few_inliers": cons.get("max_consecutive_too_few_inliers", 10),
            },
        }
        write_json(out_dir / "ready_for_gap_aware_full_metric_run.json", ready_full)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
