#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--terminal_file", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = model_dir / "semantic_trace.json"
    trace = read_json(trace_path)
    events = trace.get("events", []) or []
    control_events = trace.get("recovery_commit_control_events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    train_code = parse_terminal_exit(Path(args.terminal_file).resolve())
    mode = str(trace.get("mode", "off"))
    control_mode = str(trace.get("recovery_commit_control_mode", "off"))

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

    engine = {
        "train_returncode": train_code,
        "risk_admission_mode": mode,
        "recovery_commit_control_mode": control_mode,
        "processed_frame_count": len(events),
        "recovery_commit_control_triggered_count": len(control_events),
        "true_recovery_commit_events_count": len(true_events),
        "current_frame_surrogate_commit_count": surrogate,
        "defer_tracking_contamination_count": defer_contam,
        "discard_tracking_contamination_count": discard_contam,
        "hidden_gate_unknown_count": hidden_unknown,
        "chosen_kfs_index_error_count": chosen_kfs_error,
    }
    write_json(output_dir / "balanced_v4_baseline_guard_engine_audit.json", engine)
    write_json(
        output_dir / "balanced_v4_baseline_guard_trace_audit.json",
        {
            "trace_path": str(trace_path),
            "mode": mode,
            "recovery_commit_control_mode": control_mode,
            "num_events": len(events),
            "num_control_events": len(control_events),
            "num_true_recovery_events": len(true_events),
        },
    )
    passed = bool(
        train_code == 0
        and mode == "off"
        and len(control_events) == 0
        and len(true_events) == 0
        and surrogate == 0
        and defer_contam == 0
        and discard_contam == 0
        and hidden_unknown == 0
        and chosen_kfs_error == 0
    )
    (output_dir / "balanced_v4_baseline_guard_report.md").write_text(
        "\n".join(
            [
                "# balanced_v4 baseline guard report",
                "",
                f"- train_returncode: {train_code}",
                f"- risk_admission_mode: {mode}",
                f"- recovery_commit_control_mode: {control_mode}",
                f"- control_events/true_recovery_events: {len(control_events)}/{len(true_events)}",
                f"- surrogate/defer_contam/discard_contam/hidden_unknown/chosen_kfs_error: {surrogate}/{defer_contam}/{discard_contam}/{hidden_unknown}/{chosen_kfs_error}",
                f"- baseline_guard_passed: {passed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        output_dir / "ready_for_balanced_v4_short_run_after_guard.json",
        {"baseline_guard_passed": passed, "train_returncode": train_code},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
