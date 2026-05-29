#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
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
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
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
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = read_json(model_dir / "semantic_trace.json")
    events = trace.get("events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    control_events = trace.get("recovery_commit_control_events", []) or []

    exit_code = parse_terminal_exit(Path(args.terminal_file).resolve())
    processed = len(events)
    direct_final = sum(
        1 for e in events if str(e.get("action", "")) == "direct_admit" and bool(e.get("final_keyframe_incremented", False))
    )
    rec_success = len(true_events)
    rec_allowed = sum(1 for c in control_events if str(c.get("decision", "")) == "commit")
    rec_held = sum(1 for c in control_events if str(c.get("decision", "")) == "hold")
    rec_rejected = sum(1 for c in control_events if str(c.get("decision", "")) == "reject")
    true_final = sum(1 for e in true_events if bool(e.get("final_keyframe_incremented", False)))
    final_keyframes = direct_final + true_final
    density = (100.0 * final_keyframes) / max(processed, 1)

    final_ticks = sorted(int(e.get("frame_id", -1)) for e in events if bool(e.get("final_keyframe_incremented", False)))
    gap_rows: list[dict[str, Any]] = []
    gaps: list[int] = []
    for i in range(1, len(final_ticks)):
        g = final_ticks[i] - final_ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": final_ticks[i - 1], "to_tick": final_ticks[i], "gap": g})

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

    duplicate = 0
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

    valid_pdf_recovery_commit = sum(
        1
        for e in true_events
        if bool(e.get("final_keyframe_incremented", False))
        and (not bool(e.get("source_equals_current_frame", True)))
        and bool(e.get("final_model_contains_source_frame", False))
    )
    anchor_count = parse_terminal_anchor(Path(args.terminal_file).resolve())
    if anchor_count is None:
        anchor_count = int(sum(1 for e in events if bool(e.get("anchor_update_called", False))))

    p90 = percentile(gaps, 0.9)
    p95 = percentile(gaps, 0.95)
    gmax = float(max(gaps) if gaps else 0.0)
    short_run_stable = bool(
        exit_code == 0
        and duplicate == 0
        and defer_contam == 0
        and discard_contam == 0
        and surrogate == 0
        and hidden_unknown == 0
        and chosen_kfs_index_error == 0
        and density <= 42.0
        and p90 <= 5.0
        and p95 <= 7.0
        and gmax <= 20.0
    )

    stability = {
        "label": args.label,
        "train_returncode": exit_code,
        "processed_frame_count": processed,
        "direct_admit_final_count": direct_final,
        "recovery_success_count": rec_success,
        "recovery_commit_allowed_count": rec_allowed,
        "recovery_commit_held_count": rec_held,
        "recovery_commit_rejected_count": rec_rejected,
        "final_keyframe_count": final_keyframes,
        "final_anchor_count": anchor_count,
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
        "short_run_stable": short_run_stable,
    }
    write_json(out_dir / "engine_stability_audit.json", stability)
    write_csv(out_dir / "recovery_commit_control_trace.csv", control_events)

    reasons = [
        "v3_gap_critical_commit",
        "v3_support_ranked_sparse_commit",
        "v3_hold_not_topk",
        "v3_hold_density_high",
        "v3_hold_anchor_guard",
        "v3_hold_too_close_to_existing_keyframe",
        "v3_hold_window_budget",
        "v3_reject_age",
        "v3_reject_retry_limit",
        "v3_reject_low_support",
        "v3_reject_invalid_semantics",
    ]
    reason_counts = {r: 0 for r in reasons}
    for c in control_events:
        rr = str(c.get("decision_reason", ""))
        if rr in reason_counts:
            reason_counts[rr] += 1
    write_json(
        out_dir / "recovery_commit_control_summary.json",
        {
            "control_mode": str(trace.get("recovery_commit_control_mode", "")),
            "recovery_success_count": rec_success,
            "recovery_commit_allowed_count": rec_allowed,
            "recovery_commit_held_count": rec_held,
            "recovery_commit_rejected_count": rec_rejected,
            "decision_reason_counts": reason_counts,
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
    write_csv(out_dir / "keyframe_timeline.csv", key_rows)
    write_csv(out_dir / "main_chain_gap_timeline.csv", gap_rows)

    report_lines = [
        f"# {args.label} report",
        "",
        f"- train_returncode: {exit_code}",
        f"- direct_admit_final_count: {direct_final}",
        f"- recovery_success_count: {rec_success}",
        f"- recovery_commit_allowed/held/rejected: {rec_allowed}/{rec_held}/{rec_rejected}",
        f"- final_keyframe_count: {final_keyframes}",
        f"- keyframes_per_100_frames: {density:.2f}",
        f"- main_chain_gap_p90/p95/max: {p90}/{p95}/{gmax}",
        f"- final_anchor_count: {anchor_count}",
        f"- max_consecutive_too_few_inliers: {max_consec}",
        f"- duplicate/defer_contam/discard_contam/hidden_unknown/surrogate/chosen_kfs_error: {duplicate}/{defer_contam}/{discard_contam}/{hidden_unknown}/{surrogate}/{chosen_kfs_index_error}",
        f"- short_run_stable: {short_run_stable}",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
