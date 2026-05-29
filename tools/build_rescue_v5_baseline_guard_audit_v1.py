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
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = read_json(model_dir / "semantic_trace.json")
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
    write_json(
        out_dir / "rescue_v5_baseline_guard_engine_audit.json",
        {
            "train_returncode": train_code,
            "risk_admission_mode": mode,
            "recovery_commit_control_mode": control_mode,
            "baseline_guard_passed": passed,
        },
    )
    write_json(
        out_dir / "ready_for_rescue_v5_short_run_after_guard.json",
        {"baseline_guard_passed": passed, "train_returncode": train_code},
    )
    (out_dir / "rescue_v5_baseline_guard_report.md").write_text(
        "\n".join(
            [
                "# rescue_v5 baseline guard report",
                "",
                f"- train_returncode: {train_code}",
                f"- mode/control_mode: {mode}/{control_mode}",
                f"- surrogate/defer_contam/discard_contam/hidden_unknown/chosen_kfs_error: {surrogate}/{defer_contam}/{discard_contam}/{hidden_unknown}/{chosen_kfs_error}",
                f"- baseline_guard_passed: {passed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
