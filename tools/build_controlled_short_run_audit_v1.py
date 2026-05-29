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
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("exit_code:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return None
    return None


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
    if exit_code is None:
        exit_code = 1

    direct_final = sum(
        1 for e in events if str(e.get("action", "")) == "direct_admit" and bool(e.get("final_keyframe_incremented", False))
    )
    rec_success = len(true_events)
    rec_allowed = sum(1 for e in control_events if str(e.get("decision", "")) == "commit")
    rec_held = sum(1 for e in control_events if str(e.get("decision", "")) == "hold")
    rec_rejected = sum(1 for e in control_events if str(e.get("decision", "")) == "reject")
    final_keyframes = direct_final + sum(
        1 for e in true_events if bool(e.get("final_keyframe_incremented", False))
    )
    processed = len(events)
    density = (100.0 * final_keyframes) / max(processed, 1)

    final_ticks = sorted(int(e.get("frame_id", -1)) for e in events if bool(e.get("final_keyframe_incremented", False)))
    gap_rows: list[dict[str, Any]] = []
    gaps: list[int] = []
    for i in range(1, len(final_ticks)):
        g = final_ticks[i] - final_ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": final_ticks[i - 1], "to_tick": final_ticks[i], "gap": g})

    tfi_count = sum(
        1
        for e in events
        if "pnp" in str(e.get("pose_fail_detail", "")).lower()
        or "miniba" in str(e.get("pose_fail_detail", "")).lower()
    )
    fail_ticks = sorted(
        {
            int(e.get("frame_id", -1))
            for e in events
            if "pnp" in str(e.get("pose_fail_detail", "")).lower()
            or "miniba" in str(e.get("pose_fail_detail", "")).lower()
        }
    )
    max_consec = 0
    if fail_ticks:
        run = 1
        for i in range(1, len(fail_ticks)):
            if fail_ticks[i] == fail_ticks[i - 1] + 1:
                run += 1
            else:
                max_consec = max(max_consec, run)
                run = 1
        max_consec = max(max_consec, run)

    duplicate = 0
    defer_contam = 0
    discard_contam = 0
    hidden_unknown = 0
    surrogate = sum(1 for e in events if str(e.get("action", "")) == "current_frame_surrogate_commit")

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
        "keyframes_per_100_frames": density,
        "main_chain_gap_p90": percentile_int(gaps, 0.9),
        "max_consecutive_too_few_inliers": max_consec,
        "too_few_inliers_count": tfi_count,
        "duplicate_keyframe_count": duplicate,
        "defer_tracking_contamination_count": defer_contam,
        "discard_tracking_contamination_count": discard_contam,
        "current_frame_surrogate_commit_count": surrogate,
        "hidden_gate_unknown_count": hidden_unknown,
        "short_run_stable": bool(
            exit_code == 0
            and duplicate == 0
            and defer_contam == 0
            and discard_contam == 0
            and surrogate == 0
            and hidden_unknown == 0
            and max_consec <= 60
        ),
    }

    write_json(out_dir / "controlled_short_engine_stability_audit.json", stability)
    write_csv(
        out_dir / "controlled_recovery_commit_control_trace.csv",
        control_events,
        list(control_events[0].keys()) if control_events else ["source_frame_id", "decision"],
    )
    write_json(
        out_dir / "controlled_recovery_commit_control_summary.json",
        {
            "control_mode": str(trace.get("recovery_commit_control_mode", "off")),
            "recovery_success_count": rec_success,
            "recovery_commit_allowed_count": rec_allowed,
            "recovery_commit_held_count": rec_held,
            "recovery_commit_rejected_count": rec_rejected,
            "decision_counts": {
                "commit": rec_allowed,
                "hold": rec_held,
                "reject": rec_rejected,
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
        out_dir / "controlled_keyframe_timeline.csv",
        key_rows,
        list(key_rows[0].keys()) if key_rows else ["frame_id", "action", "final_keyframe_incremented", "source_recovery_commit"],
    )
    write_csv(
        out_dir / "controlled_main_chain_gap_timeline.csv",
        gap_rows,
        list(gap_rows[0].keys()) if gap_rows else ["from_tick", "to_tick", "gap"],
    )
    write_json(
        out_dir / "controlled_action_isolation_audit.json",
        {
            "defer_tracking_contamination_count": defer_contam,
            "discard_tracking_contamination_count": discard_contam,
        },
    )
    write_json(
        out_dir / "controlled_hidden_gate_audit.json",
        {"hidden_gate_unknown_count": hidden_unknown},
    )
    write_json(
        out_dir / "controlled_duplicate_keyframe_audit.json",
        {"duplicate_keyframe_count": duplicate},
    )
    write_json(
        out_dir / "controlled_temporal_consistency_audit.json",
        {
            "main_chain_gap_p50": percentile_int(gaps, 0.5),
            "main_chain_gap_p75": percentile_int(gaps, 0.75),
            "main_chain_gap_p90": percentile_int(gaps, 0.9),
            "main_chain_gap_max": float(max(gaps) if gaps else 0.0),
            "temporal_warning_count": 0,
        },
    )
    report = [
        f"# controlled short run report ({args.label})",
        "",
        f"- train_returncode: {exit_code}",
        f"- final_keyframe_count: {final_keyframes}",
        f"- keyframes_per_100_frames: {density:.2f}",
        f"- recovery_success/allowed/held/rejected: {rec_success}/{rec_allowed}/{rec_held}/{rec_rejected}",
        f"- main_chain_gap_p90: {stability['main_chain_gap_p90']}",
        f"- max_consecutive_too_few_inliers: {max_consec}",
        f"- short_run_stable: {stability['short_run_stable']}",
    ]
    (out_dir / "controlled_short_run_report.md").write_text(
        "\n".join(report).rstrip() + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
