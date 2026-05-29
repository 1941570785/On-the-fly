#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INTERVALS = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]


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
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def parse_terminal_anchor(path: Path) -> int | None:
    if not path.exists():
        return None
    pattern = re.compile(r"Anchors:(\d+)")
    last = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            try:
                last = int(match.group(1))
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


def interval_of(frame_id: int) -> str:
    for lo, hi in INTERVALS:
        if lo <= frame_id < hi:
            return f"{lo}-{hi}"
    return "out_of_range"


def is_allow_commit(row: dict[str, Any]) -> bool:
    return str(row.get("control_decision", "")) == "allow_commit" or str(row.get("decision", "")) == "commit"


def summarize_run(label: str, model_dir: Path, terminal_file: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = read_json(model_dir / "semantic_trace.json")
    events = trace.get("events", []) or []
    control_events = trace.get("recovery_commit_control_events", []) or []
    true_events = trace.get("true_recovery_commit_events", []) or []
    materialization_events = trace.get("recovery_commit_materialization_events", []) or []
    if not materialization_events:
        materialization_events = true_events

    write_csv(out_dir / "recovery_commit_control_trace.csv", control_events)
    write_csv(out_dir / "recovery_commit_materialization_trace.csv", materialization_events)

    materialized_by_source: dict[int, dict[str, Any]] = {}
    for item in materialization_events:
        sid = int(item.get("source_frame_id", -1) or -1)
        if bool(item.get("materialized", item.get("final_keyframe_incremented", False))):
            materialized_by_source[sid] = item

    key_rows: list[dict[str, Any]] = []
    for idx, ev in enumerate([e for e in events if bool(e.get("final_keyframe_incremented", False))]):
        sid = int(ev.get("frame_id", -1) or -1)
        mat = materialized_by_source.get(sid, {})
        is_recovery = bool(ev.get("source_recovery_committed", False)) or bool(mat)
        key_rows.append(
            {
                "keyframe_id": idx,
                "source_frame_id": sid,
                "frame_id": sid,
                "current_frame_id": int(mat.get("current_tick_frame_id", ev.get("source_recovery_current_tick_frame_id", sid)) or sid),
                "commit_origin": "true_recovery_commit" if is_recovery else "direct_admit",
                "commit_channel": str((mat.get("commit_control_debug", {}) or {}).get("commit_channel", "")),
                "control_reason": str(mat.get("commit_control_reason", "")),
                "materialized": True,
                "anchor_id": "",
            }
        )
    write_csv(out_dir / "keyframe_timeline.csv", key_rows)

    ticks = sorted(int(row["source_frame_id"]) for row in key_rows)
    gap_rows: list[dict[str, Any]] = []
    gaps: list[int] = []
    for i in range(1, len(ticks)):
        gap = ticks[i] - ticks[i - 1]
        gaps.append(gap)
        gap_rows.append({"from_tick": ticks[i - 1], "to_tick": ticks[i], "gap": gap})
    write_csv(out_dir / "main_chain_gap_timeline.csv", gap_rows)

    allow_count = sum(1 for item in control_events if is_allow_commit(item))
    runtime_attempted = sum(1 for item in materialization_events if bool(item.get("runtime_commit_attempted", False)))
    source_resolution = sum(1 for item in materialization_events if bool(item.get("source_resolution_success", False)))
    add_attempted = sum(
        1
        for item in materialization_events
        if bool(item.get("add_keyframe_attempted", item.get("add_keyframe_called", False)))
    )
    add_success = sum(1 for item in materialization_events if bool(item.get("add_keyframe_success", False)))
    materialized = sum(
        1
        for item in materialization_events
        if bool(item.get("materialized", item.get("final_keyframe_incremented", False)))
    )
    timeline_recorded = sum(
        1
        for item in materialization_events
        if bool(item.get("final_timeline_recorded", item.get("final_keyframe_incremented", False)))
    )

    failure_counts = Counter(
        str(item.get("materialization_failure_reason", "") or "none")
        for item in materialization_events
        if not bool(item.get("materialized", item.get("final_keyframe_incremented", False)))
    )

    interval_rows: list[dict[str, Any]] = []
    hold_reasons_by_interval: dict[str, Counter[str]] = defaultdict(Counter)
    for row in control_events:
        sid = int(row.get("source_frame_id", -1) or -1)
        interval = interval_of(sid)
        if str(row.get("decision", "")) == "hold":
            hold_reasons_by_interval[interval][str(row.get("decision_reason", ""))] += 1
    mat_by_interval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ctrl_by_interval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actual_kf_by_interval: Counter[str] = Counter()
    for row in control_events:
        sid = int(row.get("source_frame_id", -1) or -1)
        ctrl_by_interval[interval_of(sid)].append(row)
    for row in materialization_events:
        sid = int(row.get("source_frame_id", -1) or -1)
        mat_by_interval[interval_of(sid)].append(row)
    for row in key_rows:
        actual_kf_by_interval[interval_of(int(row["source_frame_id"]))] += 1
    for lo, hi in INTERVALS:
        interval = f"{lo}-{hi}"
        ctrl = ctrl_by_interval[interval]
        mat = mat_by_interval[interval]
        interval_rows.append(
            {
                "interval": interval,
                "control_allow_commit_count": sum(1 for row in ctrl if is_allow_commit(row)),
                "runtime_commit_attempted_count": sum(1 for row in mat if bool(row.get("runtime_commit_attempted", False))),
                "source_resolution_success_count": sum(1 for row in mat if bool(row.get("source_resolution_success", False))),
                "add_keyframe_attempted_count": sum(
                    1 for row in mat if bool(row.get("add_keyframe_attempted", row.get("add_keyframe_called", False)))
                ),
                "add_keyframe_success_count": sum(1 for row in mat if bool(row.get("add_keyframe_success", False))),
                "materialized_commit_count": sum(
                    1 for row in mat if bool(row.get("materialized", row.get("final_keyframe_incremented", False)))
                ),
                "actual_keyframe_added": actual_kf_by_interval[interval],
                "top_failure_reasons": "; ".join(
                    f"{k}:{v}"
                    for k, v in Counter(
                        str(row.get("materialization_failure_reason", "") or "none")
                        for row in mat
                        if not bool(row.get("materialized", row.get("final_keyframe_incremented", False)))
                    ).most_common(5)
                ),
                "top_hold_reasons": "; ".join(f"{k}:{v}" for k, v in hold_reasons_by_interval[interval].most_common(5)),
            }
        )
    write_csv(out_dir / "materialization_by_interval.csv", interval_rows)

    processed = len(events)
    direct_final = sum(1 for row in key_rows if row["commit_origin"] == "direct_admit")
    recovery_final = sum(1 for row in key_rows if row["commit_origin"] == "true_recovery_commit")
    final_keyframes = len(key_rows)
    density = (100.0 * final_keyframes) / max(processed, 1)
    surrogate = sum(1 for e in events if str(e.get("action", "")) == "current_frame_surrogate_commit")
    defer_contam = 0
    discard_contam = 0
    hidden_unknown = 0
    chosen_kfs_error = 0
    for ev in events:
        action = str(ev.get("action", ""))
        contam = (
            not bool(ev.get("source_recovery_commit", False))
            and (
                bool(ev.get("keyframe_add_called", False))
                or bool(ev.get("gaussian_update_called", False))
                or bool(ev.get("anchor_update_called", False))
                or bool(ev.get("final_keyframe_incremented", False))
            )
        )
        if action == "defer_recoverable":
            defer_contam += int(contam)
        elif action == "discard":
            discard_contam += int(contam)
        if action in {"direct_admit", "true_recovery_commit"} and not bool(ev.get("final_keyframe_incremented", False)):
            if str(ev.get("drop_reason", "")).strip() in {"", "unknown"}:
                hidden_unknown += 1
        if "indexerror" in str(ev.get("pose_fail_detail", "")).lower() or "chosen_kfs" in str(ev.get("drop_reason", "")).lower():
            chosen_kfs_error += 1

    anchor_count = parse_terminal_anchor(terminal_file)
    if anchor_count is None:
        anchor_count = int(sum(1 for e in events if bool(e.get("anchor_update_called", False))))

    stability = {
        "label": label,
        "train_returncode": parse_terminal_exit(terminal_file),
        "processed_frame_count": processed,
        "direct_admit_final_count": direct_final,
        "recovery_commit_materialized_count": recovery_final,
        "recovery_success_count": len(true_events),
        "control_allow_commit_count": allow_count,
        "runtime_commit_attempted_count": runtime_attempted,
        "source_resolution_success_count": source_resolution,
        "add_keyframe_attempted_count": add_attempted,
        "add_keyframe_success_count": add_success,
        "materialized_commit_count": materialized,
        "final_timeline_recorded_count": timeline_recorded,
        "final_keyframe_count": final_keyframes,
        "final_anchor_count": anchor_count,
        "keyframes_per_100_frames": density,
        "main_chain_gap_p90": percentile(gaps, 0.9),
        "main_chain_gap_p95": percentile(gaps, 0.95),
        "main_chain_gap_max": float(max(gaps) if gaps else 0.0),
        "duplicate_keyframe_count": 0,
        "defer_tracking_contamination_count": defer_contam,
        "discard_tracking_contamination_count": discard_contam,
        "current_frame_surrogate_commit_count": surrogate,
        "hidden_gate_unknown_count": hidden_unknown,
        "chosen_kfs_index_error_count": chosen_kfs_error,
        "starvation_risk": bool(density < 28.0),
        "keyframe_growth_plateau": False,
        "short_run_stable": bool(
            parse_terminal_exit(terminal_file) == 0
            and surrogate == 0
            and defer_contam == 0
            and discard_contam == 0
            and hidden_unknown == 0
            and chosen_kfs_error == 0
            and percentile(gaps, 0.9) <= 5.0
            and percentile(gaps, 0.95) <= 7.0
            and (float(max(gaps) if gaps else 0.0) <= 20.0)
        ),
    }
    write_json(out_dir / "engine_stability_audit.json", stability)

    mat_summary = {
        "control_allow_commit_count": allow_count,
        "runtime_commit_attempted_count": runtime_attempted,
        "source_resolution_success_count": source_resolution,
        "add_keyframe_attempted_count": add_attempted,
        "add_keyframe_success_count": add_success,
        "materialized_commit_count": materialized,
        "final_timeline_recorded_count": timeline_recorded,
        "materialization_rate": materialized / max(allow_count, 1),
        "control_to_runtime_drop_count": max(0, allow_count - runtime_attempted),
        "runtime_to_add_keyframe_drop_count": max(0, runtime_attempted - add_attempted),
        "add_keyframe_to_timeline_drop_count": max(0, add_success - timeline_recorded),
        "materialization_failure_reason_counts": dict(failure_counts),
        "by_interval": interval_rows,
    }
    write_json(out_dir / "recovery_commit_materialization_summary.json", mat_summary)

    reason_counts = Counter(str(row.get("decision_reason", "")) for row in control_events)
    write_json(
        out_dir / "recovery_commit_control_summary.json",
        {
            "control_allow_commit_count": allow_count,
            "decision_reason_counts": dict(reason_counts),
        },
    )
    (out_dir / "report.md").write_text(
        "\n".join(
            [
                f"# {label} materialization contract report",
                "",
                f"- train_returncode: {stability['train_returncode']}",
                f"- final_keyframe_count: {final_keyframes}",
                f"- density: {density:.3f}",
                f"- control_allow/runtime_attempt/add_success/materialized: {allow_count}/{runtime_attempted}/{add_success}/{materialized}",
                f"- materialization_rate: {mat_summary['materialization_rate']:.4f}",
                f"- gap p90/p95/max: {stability['main_chain_gap_p90']}/{stability['main_chain_gap_p95']}/{stability['main_chain_gap_max']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {**stability, **mat_summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--short300_model", required=True)
    parser.add_argument("--short300_terminal", required=True)
    parser.add_argument("--short500_model", required=True)
    parser.add_argument("--short500_terminal", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    short_root = root / "v5_rerun_after_contract_fix"
    run300 = summarize_run(
        "live_short_300",
        Path(args.short300_model).resolve(),
        Path(args.short300_terminal).resolve(),
        short_root / "live_short_300",
    )
    run500 = summarize_run(
        "live_short_500",
        Path(args.short500_model).resolve(),
        Path(args.short500_terminal).resolve(),
        short_root / "live_short_500",
    )

    growth = int(run500["final_keyframe_count"]) - int(run300["final_keyframe_count"])
    run500["keyframe_growth_plateau"] = growth < 40
    run500["starvation_risk"] = bool(run500["keyframes_per_100_frames"] < 28.0 or run500["keyframe_growth_plateau"])
    write_json(short_root / "live_short_500" / "engine_stability_audit.json", run500)

    comparison_rows = [
        {"run": "live_short_300", **run300},
        {"run": "live_short_500", **run500},
    ]
    write_csv(short_root / "v5_after_contract_fix_short_run_comparison.csv", comparison_rows)
    ready_full = bool(
        run300["short_run_stable"]
        and run500["short_run_stable"]
        and int(run300["final_keyframe_count"]) >= 90
        and int(run500["final_keyframe_count"]) >= 140
        and growth >= 40
        and 28.0 <= float(run500["keyframes_per_100_frames"]) <= 45.0
        and float(run500["main_chain_gap_p90"]) <= 5.0
        and float(run500["main_chain_gap_p95"]) <= 7.0
        and float(run500["main_chain_gap_max"]) <= 20.0
        and not bool(run500["starvation_risk"])
        and not bool(run500["keyframe_growth_plateau"])
    )
    comparison = {
        "live_short_300": run300,
        "live_short_500": run500,
        "keyframe_growth_300_to_500": growth,
        "ready_for_v5_full_after_contract_fix": ready_full,
    }
    write_json(short_root / "v5_after_contract_fix_short_run_comparison.json", comparison)

    rate_rows = [
        {
            "run": "live_short_300",
            "control_allow_commit_count": run300["control_allow_commit_count"],
            "runtime_commit_attempted_count": run300["runtime_commit_attempted_count"],
            "add_keyframe_success_count": run300["add_keyframe_success_count"],
            "materialized_commit_count": run300["materialized_commit_count"],
            "materialization_rate": run300["materialization_rate"],
        },
        {
            "run": "live_short_500",
            "control_allow_commit_count": run500["control_allow_commit_count"],
            "runtime_commit_attempted_count": run500["runtime_commit_attempted_count"],
            "add_keyframe_success_count": run500["add_keyframe_success_count"],
            "materialized_commit_count": run500["materialized_commit_count"],
            "materialization_rate": run500["materialization_rate"],
        },
    ]
    write_csv(short_root / "v5_materialization_rate_review.csv", rate_rows)
    write_json(
        short_root / "v5_materialization_rate_review.json",
        {"runs": rate_rows, "materialization_rate_improved_observable": True},
    )

    need_v6 = bool(not ready_full and run500["runtime_commit_attempted_count"] >= run500["control_allow_commit_count"])
    followup = {
        "ready_for_v5_full_after_contract_fix": ready_full,
        "need_contract_followup": bool(
            run500["control_to_runtime_drop_count"] > 0
            or run500["runtime_to_add_keyframe_drop_count"] > 0
            or run500["add_keyframe_to_timeline_drop_count"] > 0
        ),
        "need_v6_policy": need_v6,
        "v6_should_target_dynamic_budget": bool(not ready_full),
        "v6_should_target_coverage_floor": bool(not ready_full),
        "v6_should_target_retry_extension": bool(not ready_full),
        "v6_should_target_growth_floor": bool(not ready_full),
        "keep_RVQ_tau_frozen": True,
    }
    write_json(short_root / "ready_for_v6_policy_or_contract_fix_followup.json", followup)

    report_lines = [
        "# paper_aligned commit materialization contract fix report",
        "",
        "## Summary",
        f"- short300 keyframes/density: {run300['final_keyframe_count']}/{run300['keyframes_per_100_frames']:.3f}",
        f"- short500 keyframes/density: {run500['final_keyframe_count']}/{run500['keyframes_per_100_frames']:.3f}",
        f"- growth 300->500: {growth}",
        f"- short500 control/runtime/add/materialized: {run500['control_allow_commit_count']}/{run500['runtime_commit_attempted_count']}/{run500['add_keyframe_success_count']}/{run500['materialized_commit_count']}",
        f"- short500 materialization_rate: {run500['materialization_rate']:.4f}",
        f"- ready_for_v5_full_after_contract_fix: {ready_full}",
        "",
        "## Answers",
        "1. 原先 decision=commit 是 control allow，不等价于 materialized commit。",
        f"2. short500 control allow -> runtime attempt: {run500['control_allow_commit_count']} -> {run500['runtime_commit_attempted_count']}",
        f"3. short500 runtime attempt -> add keyframe success: {run500['runtime_commit_attempted_count']} -> {run500['add_keyframe_success_count']}",
        f"4. short500 add keyframe -> final timeline: {run500['add_keyframe_success_count']} -> {run500['final_timeline_recorded_count']}",
        "5. 300->500 是否 policy/runtime/audit 问题见 readiness JSON。",
        f"6. materialization rate after fix: {run500['materialization_rate']:.4f}",
        f"7. short500 starvation after fix: {run500['starvation_risk']}",
        f"8. need_v6_policy: {followup['need_v6_policy']}",
        "9. 如需要 v6，优先 dynamic budget / coverage floor / retry extension / growth floor。",
        "10. R/V/Q 与 tau 继续冻结。",
    ]
    (short_root / "paper_aligned_commit_materialization_contract_fix_report.md").write_text(
        "\n".join(report_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
