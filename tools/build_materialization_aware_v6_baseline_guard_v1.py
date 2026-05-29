#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_terminal_exit(path: Path) -> int:
    if not path.exists():
        return 1
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if line.startswith("exit_code:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return 1
    return 1


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--terminal_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = read_json(model_dir / "semantic_trace.json")
    events = trace.get("events", []) or []
    control_events = trace.get("recovery_commit_control_events", []) or []
    materialization_events = trace.get("recovery_commit_materialization_events", []) or []
    train_code = parse_terminal_exit(Path(args.terminal_file).resolve())
    mode = str(trace.get("mode", "off"))
    passed = bool(train_code == 0 and mode == "off" and len(control_events) == 0 and len(materialization_events) == 0)
    payload = {
        "train_returncode": train_code,
        "risk_admission_mode": mode,
        "events": len(events),
        "control_events": len(control_events),
        "materialization_events": len(materialization_events),
        "baseline_guard_passed": passed,
    }
    write_json(out_dir / "materialization_aware_v6_baseline_guard_audit.json", payload)
    write_json(out_dir / "ready_for_materialization_aware_v6_short_run_after_guard.json", payload)
    (out_dir / "materialization_aware_v6_baseline_guard_report.md").write_text(
        "\n".join(
            [
                "# materialization-aware v6 baseline guard",
                "",
                f"- train_returncode: {train_code}",
                f"- risk_admission_mode: {mode}",
                f"- control/materialization events: {len(control_events)}/{len(materialization_events)}",
                f"- baseline_guard_passed: {passed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
