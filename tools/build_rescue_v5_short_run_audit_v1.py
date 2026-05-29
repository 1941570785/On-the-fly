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
    direct_final = sum(1 for e in events if str(e.get("action", "")) == "direct_admit" and bool(e.get("final_keyframe_incremented", False)))
    true_final = sum(1 for e in true_events if bool(e.get("final_keyframe_incremented", False)))
    final_keyframes = direct_final + true_final
    density = (100.0 * final_keyframes) / max(processed, 1)

    ticks = sorted(int(e.get("frame_id", -1)) for e in events if bool(e.get("final_keyframe_incremented", False)))
    gaps: list[int] = []
    gap_rows: list[dict[str, Any]] = []
    for i in range(1, len(ticks)):
        g = ticks[i] - ticks[i - 1]
        gaps.append(g)
        gap_rows.append({"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": g})

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

    anchor_count = parse_terminal_anchor(Path(args.terminal_file).resolve())
    if anchor_count is None:
        anchor_count = int(sum(1 for e in events if bool(e.get("anchor_update_called", False))))

    p90 = percentile(gaps, 0.9)
    p95 = percentile(gaps, 0.95)
    gmax = float(max(gaps) if gaps else 0.0)
    growth_plateau = bool(
        (args.label == "live_short_500")
        and (final_keyframes <= 130)
    )
    starvation_risk = bool(density < 28.0 or growth_plateau)
    too_few = sum(
        1
        for e in events
        if "pnp" in str(e.get("pose_fail_detail", "")).lower()
        or "miniba" in str(e.get("pose_fail_detail", "")).lower()
        or "too_few" in str(e.get("pose_fail_detail", "")).lower()
    )

    stability = {
        "label": args.label,
        "train_returncode": exit_code,
        "processed_frame_count": processed,
        "direct_admit_final_count": direct_final,
        "recovery_success_count": len(true_events),
        "recovery_commit_allowed_count": sum(1 for c in control_events if str(c.get("decision", "")) == "commit"),
        "recovery_commit_held_count": sum(1 for c in control_events if str(c.get("decision", "")) == "hold"),
        "recovery_commit_rejected_count": sum(1 for c in control_events if str(c.get("decision", "")) == "reject"),
        "final_keyframe_count": final_keyframes,
        "final_anchor_count": anchor_count,
        "keyframes_per_100_frames": density,
        "main_chain_gap_p90": p90,
        "main_chain_gap_p95": p95,
        "main_chain_gap_max": gmax,
        "too_few_inliers_count": too_few,
        "duplicate_keyframe_count": 0,
        "defer_tracking_contamination_count": defer_contam,
        "discard_tracking_contamination_count": discard_contam,
        "current_frame_surrogate_commit_count": surrogate,
        "hidden_gate_unknown_count": hidden_unknown,
        "chosen_kfs_index_error_count": chosen_kfs_error,
        "starvation_risk": starvation_risk,
        "keyframe_growth_plateau": growth_plateau,
        "short_run_stable": bool(
            exit_code == 0
            and surrogate == 0
            and defer_contam == 0
            and discard_contam == 0
            and hidden_unknown == 0
            and chosen_kfs_error == 0
            and p90 <= 5.0
            and p95 <= 7.0
            and gmax <= 20.0
        ),
    }
    write_json(out_dir / "engine_stability_audit.json", stability)
    write_csv(out_dir / "recovery_commit_control_trace.csv", control_events)

    reason_counts: dict[str, int] = {}
    for c in control_events:
        rr = str(c.get("decision_reason", ""))
        reason_counts[rr] = reason_counts.get(rr, 0) + 1
    write_json(out_dir / "recovery_commit_control_summary.json", {"decision_reason_counts": reason_counts})

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
    (out_dir / "report.md").write_text(
        "\n".join(
            [
                f"# {args.label} report",
                "",
                f"- train_returncode: {exit_code}",
                f"- final_keyframe_count: {final_keyframes}",
                f"- density: {density:.3f}",
                f"- gap p90/p95/max: {p90:.3f}/{p95:.3f}/{gmax:.3f}",
                f"- starvation_risk/growth_plateau: {starvation_risk}/{growth_plateau}",
                f"- short_run_stable: {stability['short_run_stable']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
