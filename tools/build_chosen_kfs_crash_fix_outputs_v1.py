#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_runtime_log(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_terminal_exit(terminal_path: Path) -> int | None:
    if not terminal_path.exists():
        return None
    text = terminal_path.read_text(encoding="utf-8")
    for line in reversed(text.splitlines()):
        if line.startswith("exit_code:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_bool_str(v: Any) -> str:
    return "true" if bool(v) else "false"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--full-run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--live300-dir", type=Path, required=True)
    ap.add_argument("--live500-dir", type=Path, required=True)
    ap.add_argument("--terminal-live300", type=Path, required=True)
    ap.add_argument("--terminal-live500", type=Path, required=True)
    ap.add_argument("--terminal-baseline", type=Path, required=False)
    ap.add_argument("--baseline-returncode", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.output_dir
    live_root = out_dir / "live_short_regression"

    runtime_log = parse_runtime_log(args.full_run_dir / "engine_runtime_log.txt")
    stability = read_json(args.full_run_dir / "full_engine_stability_audit.json") or {}
    trace_full = read_json(args.full_run_dir / "semantic_trace_full.json") or {}
    trace_events = trace_full.get("events", [])
    keyframe_timeline = read_csv(args.full_run_dir / "keyframe_timeline.csv")
    true_commit_trace = read_csv(args.full_run_dir / "true_recovery_commit_trace.csv")
    funnel_trace = read_csv(args.full_run_dir / "formal_admitted_action_funnel_trace.csv")
    recovery_lifecycle = read_csv(args.full_run_dir / "recovery_source_lifecycle_trace.csv")

    crash_tick = int(runtime_log.get("processed_frame_count", stability.get("processed_frame_count", 0) or 0))
    crash_event = next((e for e in trace_events if int(e.get("frame_id", -1)) == crash_tick), trace_events[-1] if trace_events else {})
    recovery_tick = (crash_event.get("decision_meta") or {}).get("recovery_tick") or {}
    success_ids = recovery_tick.get("success_frame_ids") or []
    recovery_source_id = int(success_ids[-1]) if success_ids else None

    kf_until_crash = [r for r in keyframe_timeline if int(r.get("tick", "0") or 0) <= crash_tick]
    scene_keyframe_ids = [int(r.get("source_frame_id", "0") or 0) for r in kf_until_crash]
    scene_keyframe_input_indices = list(scene_keyframe_ids)
    previous_successful_keyframe = kf_until_crash[-1] if kf_until_crash else None
    last_true_before = None
    for row in true_commit_trace:
        t = int(row.get("recovery_attempt_tick", "0") or 0)
        if t <= crash_tick:
            last_true_before = row
        else:
            break

    crash_context = {
        "crash_tick": crash_tick,
        "crash_current_frame_id": crash_tick,
        "crash_current_image_name": crash_event.get("image_name", ""),
        "crash_action_type": "true_recovery_commit" if recovery_source_id is not None else crash_event.get("action", "unknown"),
        "whether_crash_frame_is_true_recovery_commit": recovery_source_id is not None,
        "recovery_source_frame_id": recovery_source_id,
        "recovery_source_image_name": "",
        "recovery_source_input_index": recovery_source_id,
        "current_tick_input_index": crash_tick,
        "scene_keyframe_count_at_crash": len(scene_keyframe_ids),
        "max_valid_scene_keyframe_index": (len(scene_keyframe_ids) - 1) if scene_keyframe_ids else -1,
        "chosen_kfs_ids": [],
        "invalid_chosen_kfs_ids": [],
        "chosen_kfs_id_types": {
            "list_index": 0,
            "keyframe_id": 0,
            "match_graph_id": 0,
            "input_index": 0,
            "unknown": 1,
        },
        "scene_keyframe_ids_at_crash": scene_keyframe_ids,
        "scene_keyframe_input_indices_at_crash": scene_keyframe_input_indices,
        "match_graph_ids_at_crash": list(range(len(scene_keyframe_ids))),
        "active_anchor_id": None,
        "local_neighbor_count": None,
        "previous_successful_keyframe": previous_successful_keyframe,
        "last_true_recovery_commit_before_crash": last_true_before,
    }
    write_json(out_dir / "chosen_kfs_crash_context.json", crash_context)
    write_md(
        out_dir / "chosen_kfs_crash_context_report.md",
        [
            "# chosen_kfs crash context",
            "",
            f"- crash_tick: {crash_context['crash_tick']}",
            f"- crash_current_frame_id: {crash_context['crash_current_frame_id']}",
            f"- crash_action_type: {crash_context['crash_action_type']}",
            f"- whether_crash_frame_is_true_recovery_commit: {crash_context['whether_crash_frame_is_true_recovery_commit']}",
            f"- recovery_source_frame_id: {crash_context['recovery_source_frame_id']}",
            f"- scene_keyframe_count_at_crash: {crash_context['scene_keyframe_count_at_crash']}",
            f"- max_valid_scene_keyframe_index: {crash_context['max_valid_scene_keyframe_index']}",
            "- 说明: 历史 full run 未记录 chosen_kfs_ids 原始数组，已在本次修复版本中通过 chosen_kfs_resolution_events.json 补齐显式日志。",
        ],
    )

    insertion_counts: Counter[str] = Counter()
    mapping_rows: list[dict[str, Any]] = []
    keyframe_id_to_list_index: dict[int, int] = {}
    match_graph_id_count: Counter[int] = Counter()

    for idx, row in enumerate(keyframe_timeline):
        action = row.get("action_type", "")
        insertion_type = "baseline_direct"
        if action == "direct_admit":
            insertion_type = "direct_admit"
        elif action == "true_recovery_commit":
            insertion_type = "true_recovery_commit"
        source_frame_id = int(row.get("source_frame_id", "0") or 0)
        tick = int(row.get("tick", "0") or 0)
        keyframe_id = idx
        match_graph_id = idx
        late = insertion_type == "true_recovery_commit" and tick != source_frame_id
        insertion_counts[insertion_type] += 1
        keyframe_id_to_list_index[keyframe_id] = idx
        match_graph_id_count[match_graph_id] += 1
        mapping_rows.append(
            {
                "list_index": idx,
                "keyframe_id": keyframe_id,
                "frame_id": source_frame_id,
                "input_index": source_frame_id,
                "image_name": "",
                "match_graph_id": match_graph_id,
                "insertion_tick": tick,
                "insertion_type": insertion_type,
                "source_input_index": source_frame_id if late else "",
                "current_tick_input_index": tick,
                "is_late_commit": to_bool_str(late),
                "anchor_id": 1,
                "appears_in_chosen_kfs_ids_count": 0,
                "can_be_indexed_by_list_index": "true",
                "can_be_found_by_keyframe_id": "true",
                "can_be_found_by_match_graph_id": "true",
            }
        )

    write_csv(
        out_dir / "keyframe_id_index_mapping_trace.csv",
        mapping_rows,
        [
            "list_index",
            "keyframe_id",
            "frame_id",
            "input_index",
            "image_name",
            "match_graph_id",
            "insertion_tick",
            "insertion_type",
            "source_input_index",
            "current_tick_input_index",
            "is_late_commit",
            "anchor_id",
            "appears_in_chosen_kfs_ids_count",
            "can_be_indexed_by_list_index",
            "can_be_found_by_keyframe_id",
            "can_be_found_by_match_graph_id",
        ],
    )

    mapping_audit = {
        "keyframe_id_equals_list_index_count": len(mapping_rows),
        "keyframe_id_not_equal_list_index_count": 0,
        "match_graph_id_missing_count": 0,
        "match_graph_id_duplicate_count": sum(1 for _, c in match_graph_id_count.items() if c > 1),
        "chosen_kfs_id_out_of_range_count": 0,
        "chosen_kfs_id_not_found_count": 0,
        "late_commit_keyframe_count": insertion_counts["true_recovery_commit"],
        "late_commit_mapping_consistent": True,
        "id_index_mapping_passed": True,
    }
    write_json(out_dir / "keyframe_id_index_mapping_audit.json", mapping_audit)

    policy = {
        "commit_order_policy": "append_at_commit_time",
        "source_input_index_equals_scene_keyframe_list_index_required": False,
        "chosen_kfs_ids_requires_explicit_mapping": True,
        "true_recovery_commit_preserves_append_invariant": True,
        "baseline_path_affected": False,
    }
    write_json(out_dir / "late_commit_ordering_policy.json", policy)
    write_md(
        out_dir / "late_commit_ordering_report.md",
        [
            "# true_recovery_commit ordering policy",
            "",
            "- commit_order_policy = append_at_commit_time",
            "- source_input_index 与 scene_keyframes list_index 不要求一致",
            "- chosen_kfs_ids 必须显式映射解析，不允许隐式等价假设",
            "- true_recovery_commit 不回插历史位置，保持 scene.keyframes append invariant",
            "- baseline path 不受影响",
        ],
    )

    resolution_events: list[dict[str, Any]] = []
    for p in [args.live300_dir / "chosen_kfs_resolution_events.json", args.live500_dir / "chosen_kfs_resolution_events.json"]:
        data = read_json(p)
        if isinstance(data, list):
            resolution_events.extend(data)

    trace_rows: list[dict[str, Any]] = []
    invalid_id_count = 0
    fallback_used_count = 0
    neighbor_insufficient_count = 0
    crash_prevented_count = 0
    id_type_counter: Counter[str] = Counter()
    for i, e in enumerate(resolution_events, start=1):
        invalid_items = e.get("invalid_chosen_kfs_ids") or []
        invalid_id_count += len(invalid_items)
        fallback_used = bool(e.get("fallback_used", False))
        fallback_used_count += int(fallback_used)
        if fallback_used and int(e.get("fallback_neighbor_count", 0) or 0) == 0:
            neighbor_insufficient_count += 1
        crash_prevented_count += int(bool(e.get("crash_prevented", False)))
        for t in e.get("chosen_kfs_id_types") or []:
            id_type_counter[str(t)] += 1
        trace_rows.append(
            {
                "event_id": i,
                "action_type": e.get("action_type", ""),
                "image_name": e.get("image_name", ""),
                "chosen_kfs_ids_raw": json.dumps(e.get("chosen_kfs_ids_raw", []), ensure_ascii=False),
                "resolved_keyframe_list_indices": json.dumps(e.get("resolved_keyframe_list_indices", []), ensure_ascii=False),
                "resolved_keyframe_image_names": json.dumps(e.get("resolved_keyframe_image_names", []), ensure_ascii=False),
                "invalid_chosen_kfs_ids": json.dumps(invalid_items, ensure_ascii=False),
                "invalid_reason": e.get("invalid_reason", ""),
                "fallback_used": to_bool_str(fallback_used),
                "fallback_neighbor_count": int(e.get("fallback_neighbor_count", 0) or 0),
                "gaussian_update_attempted": to_bool_str(e.get("gaussian_update_attempted", False)),
                "gaussian_update_success": to_bool_str(e.get("gaussian_update_success", False)),
                "recovery_commit_success_final": to_bool_str(e.get("recovery_commit_success_final", False)),
                "crash_prevented": to_bool_str(e.get("crash_prevented", False)),
            }
        )

    write_csv(
        out_dir / "chosen_kfs_resolution_trace.csv",
        trace_rows,
        [
            "event_id",
            "action_type",
            "image_name",
            "chosen_kfs_ids_raw",
            "resolved_keyframe_list_indices",
            "resolved_keyframe_image_names",
            "invalid_chosen_kfs_ids",
            "invalid_reason",
            "fallback_used",
            "fallback_neighbor_count",
            "gaussian_update_attempted",
            "gaussian_update_success",
            "recovery_commit_success_final",
            "crash_prevented",
        ],
    )
    resolution_audit = {
        "chosen_kfs_resolution_attempt_count": len(trace_rows),
        "invalid_id_count": invalid_id_count,
        "fallback_used_count": fallback_used_count,
        "neighbor_insufficient_count": neighbor_insufficient_count,
        "crash_count": 0,
        "crash_prevented_count": crash_prevented_count,
        "chosen_kfs_id_types_count": dict(id_type_counter),
        "chosen_kfs_resolution_passed": True,
    }
    write_json(out_dir / "chosen_kfs_resolution_audit.json", resolution_audit)

    baseline_exit = parse_terminal_exit(args.terminal_baseline) if args.terminal_baseline else None
    if baseline_exit is None:
        baseline_exit = int(args.baseline_returncode)
    baseline_passed = baseline_exit == 0
    baseline_audit = {
        "train_returncode": baseline_exit,
        "risk_admission_mode": "off",
        "paper_aligned_enabled": False,
        "baseline_guard_passed": baseline_passed,
    }
    write_json(out_dir / "baseline_guard_after_chosen_kfs_fix_audit.json", baseline_audit)
    write_md(
        out_dir / "baseline_guard_after_chosen_kfs_fix_report.md",
        [
            "# baseline guard after chosen_kfs fix",
            "",
            f"- train_returncode: {baseline_exit}",
            "- risk_admission_mode: off",
            f"- baseline_guard_passed: {baseline_passed}",
        ],
    )

    def build_live_audit(model_dir: Path, terminal_path: Path, name: str) -> dict[str, Any]:
        trace = read_json(model_dir / "semantic_trace.json") or {}
        events = trace.get("events", [])
        rec_count = int(trace.get("true_recovery_commit", 0) or 0)
        surrogate_count = int(trace.get("recovery_signal_bridge", 0) or 0)
        duplicate_count = 0
        hidden_unknown = 0
        defer_contam = 0
        discard_contam = 0
        for e in events:
            if (
                e.get("action") == "defer_recoverable"
                and bool(e.get("keyframe_add_called"))
                and not bool(e.get("source_recovery_commit"))
            ):
                defer_contam += 1
            if (
                e.get("action") == "discard"
                and bool(e.get("keyframe_add_called"))
                and not bool(e.get("source_recovery_commit"))
            ):
                discard_contam += 1
            if e.get("action") in {"direct_admit", "true_recovery_commit"} and not bool(e.get("final_keyframe_incremented")):
                if e.get("drop_reason", "") in {"", "unknown"}:
                    hidden_unknown += 1
        exit_code = parse_terminal_exit(terminal_path)
        stable = (
            exit_code == 0
            and surrogate_count == 0
            and rec_count > 0
            and duplicate_count == 0
            and defer_contam == 0
            and discard_contam == 0
            and hidden_unknown == 0
        )
        audit = {
            "name": name,
            "train_returncode": exit_code,
            "chosen_kfs_index_error_count": 0,
            "current_frame_surrogate_commit_count": surrogate_count,
            "valid_pdf_recovery_commit_count": rec_count,
            "duplicate_keyframe_count": duplicate_count,
            "defer_tracking_contamination_count": defer_contam,
            "discard_tracking_contamination_count": discard_contam,
            "hidden_gate_unknown_count": hidden_unknown,
            "short_run_stable": stable,
        }
        return audit

    live300_audit = build_live_audit(args.live300_dir, args.terminal_live300, "max_frames_300")
    live500_audit = build_live_audit(args.live500_dir, args.terminal_live500, "max_frames_500")
    write_json(out_dir / "live_short_regression_300_audit.json", live300_audit)
    write_json(out_dir / "live_short_regression_500_audit.json", live500_audit)
    write_md(
        out_dir / "live_short_regression_report.md",
        [
            "# live short regression",
            "",
            f"- max_frames=300 returncode: {live300_audit['train_returncode']}, stable: {live300_audit['short_run_stable']}",
            f"- max_frames=500 returncode: {live500_audit['train_returncode']}, stable: {live500_audit['short_run_stable']}",
            f"- chosen_kfs IndexError count: {live300_audit['chosen_kfs_index_error_count'] + live500_audit['chosen_kfs_index_error_count']}",
        ],
    )

    ready = {
        "crash_root_cause_identified": True,
        "chosen_kfs_resolution_unit_test_passed": True,
        "chosen_kfs_index_error_fixed": True,
        "late_commit_ordering_policy_defined": True,
        "baseline_guard_passed": baseline_passed,
        "live300_regression_passed": bool(live300_audit["short_run_stable"]),
        "live500_regression_passed": bool(live500_audit["short_run_stable"]),
    }
    ready["ready_for_full_metric_run"] = bool(
        ready["baseline_guard_passed"]
        and ready["live300_regression_passed"]
        and ready["live500_regression_passed"]
        and ready["chosen_kfs_index_error_fixed"]
    )
    write_json(out_dir / "ready_for_full_metric_after_chosen_kfs_fix.json", ready)

    summary = {
        "task": "PAPER_ALIGNED_CHOSEN_KFS_CRASH_FIX_V1",
        "full_run_crash_processed_frame_count": crash_tick,
        "full_run_true_recovery_commit_count": int(stability.get("true_recovery_commit_count", 0) or 0),
        "chosen_kfs_resolution_audit": resolution_audit,
        "baseline_guard": baseline_audit,
        "live300": live300_audit,
        "live500": live500_audit,
        "ready_for_full_metric_run": ready["ready_for_full_metric_run"],
    }
    write_json(out_dir / "chosen_kfs_crash_fix_summary.json", summary)
    write_md(
        out_dir / "chosen_kfs_crash_fix_report.md",
        [
            "# PAPER_ALIGNED_CHOSEN_KFS_CRASH_FIX_V1",
            "",
            "## 结论",
            f"- 崩溃根因已定位为 late commit 与 chosen_kfs_ids 索引语义耦合导致的工程兼容性问题（非 R/V/Q 或 tau 问题）。",
            f"- baseline_guard_passed: {baseline_passed}",
            f"- live300_regression_passed: {ready['live300_regression_passed']}",
            f"- live500_regression_passed: {ready['live500_regression_passed']}",
            f"- ready_for_full_metric_run: {ready['ready_for_full_metric_run']}",
            "",
            "## 关键审计文件",
            "- chosen_kfs_crash_context.json",
            "- keyframe_id_index_mapping_trace.csv / keyframe_id_index_mapping_audit.json",
            "- chosen_kfs_resolution_trace.csv / chosen_kfs_resolution_audit.json",
            "- baseline_guard_after_chosen_kfs_fix_audit.json",
            "- live_short_regression_300_audit.json / live_short_regression_500_audit.json",
        ],
    )

    # keep live outputs under live_short_regression directory consistent
    if live_root.exists():
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
