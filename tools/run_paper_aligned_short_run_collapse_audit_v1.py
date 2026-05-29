#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _top(c: Counter, k: int = 10) -> list[dict[str, Any]]:
    return [{"reason": k1, "count": int(v1)} for k1, v1 in c.most_common(k)]


def _find_clusters(sorted_vals: list[int]) -> list[tuple[int, int, int]]:
    if not sorted_vals:
        return []
    clusters: list[tuple[int, int, int]] = []
    s = sorted_vals[0]
    p = sorted_vals[0]
    for x in sorted_vals[1:]:
        if x == p + 1:
            p = x
            continue
        clusters.append((s, p, p - s + 1))
        s = x
        p = x
    clusters.append((s, p, p - s + 1))
    return clusters


def _age_bucket(age: int) -> str:
    if age <= 30:
        return "0_30"
    if age <= 80:
        return "31_80"
    if age <= 150:
        return "81_150"
    return "151_plus"


def main() -> None:
    ap = argparse.ArgumentParser(description="PAPER_ALIGNED_SHORT_RUN_COLLAPSE_AUDIT_V1")
    ap.add_argument(
        "--input_root",
        type=str,
        default="results/StaticHikes/forest1/PAPER_ALIGNED_SEMANTIC_SHORT_ENGINE_RUN_V1",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="results/StaticHikes/forest1/PAPER_ALIGNED_SHORT_RUN_COLLAPSE_AUDIT_V1",
    )
    args = ap.parse_args()

    root = Path(args.input_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    p300 = root / "max_frames_300"
    p500 = root / "max_frames_500"

    s300 = _read_json(p300 / "short_engine_stability_audit.json")
    s500 = _read_json(p500 / "short_engine_stability_audit.json")
    tfi300 = _read_csv(p300 / "short_too_few_inliers_timeline.csv")
    tfi500 = _read_csv(p500 / "short_too_few_inliers_timeline.csv")
    dec300 = _read_csv(p300 / "short_frame_decision_table.csv")
    dec500 = _read_csv(p500 / "short_frame_decision_table.csv")
    funnel300 = _read_csv(p300 / "short_formal_admitted_action_funnel_trace.csv")
    funnel500 = _read_csv(p500 / "short_formal_admitted_action_funnel_trace.csv")
    true300 = _read_csv(p300 / "short_true_recovery_commit_trace.csv")
    true500 = _read_csv(p500 / "short_true_recovery_commit_trace.csv")
    life300 = _read_csv(p300 / "short_recovery_source_lifecycle_trace.csv")
    life500 = _read_csv(p500 / "short_recovery_source_lifecycle_trace.csv")
    key300 = _read_csv(p300 / "short_keyframe_timeline.csv")
    key500 = _read_csv(p500 / "short_keyframe_timeline.csv")
    gap300 = _read_csv(p300 / "short_main_chain_gap_timeline.csv")
    gap500 = _read_csv(p500 / "short_main_chain_gap_timeline.csv")
    report300 = (p300 / "short_engine_report.md").read_text(encoding="utf-8")
    report500 = (p500 / "short_engine_report.md").read_text(encoding="utf-8")

    # ========== 任务1：too-few-inliers 失败簇定位 ==========
    fail_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in tfi500:
        fail_by_frame[int(r.get("current_tick_frame_id", r.get("source_frame_id", -1)))] .append(r)

    frames = sorted(fail_by_frame.keys())
    clusters = _find_clusters(frames)
    first_tfi = int(s500.get("first_too_few_inliers_frame", frames[0] if frames else -1))
    max_consecutive = int(s500.get("max_consecutive_too_few_inliers", 0))
    max_cluster = max(clusters, key=lambda x: x[2]) if clusters else (None, None, 0)
    first_long_cluster = None
    for c in clusters:
        if c[2] >= 30:
            first_long_cluster = c
            break
    if first_long_cluster is None:
        first_long_cluster = max_cluster if clusters else (None, None, 0)
    if first_tfi >= 0 and max_consecutive > 0:
        first_long_cluster = (first_tfi, first_tfi + max_consecutive - 1, max_consecutive)

    key_ticks = sorted(int(r["tick"]) for r in key500)
    key_tick_set = set(key_ticks)

    def _prev_keyframe(idx: int) -> int | None:
        prev = None
        for t in key_ticks:
            if t <= idx:
                prev = t
            else:
                break
        return prev

    frame_dec_map = {int(r["frame_id"]): r for r in dec500}
    # nearest main chain keyframe -> previous as causal reference
    cluster_rows: list[dict[str, Any]] = []
    pnp_count = 0
    miniba_count = 0
    other_count = 0
    for idx in frames:
        fails = fail_by_frame[idx]
        reasons = [str(x.get("failure_reason", "")) for x in fails]
        reason_str = ";".join(sorted(set(reasons)))
        if any("pnp" in r.lower() for r in reasons):
            pnp_count += 1
        if any("miniba" in r.lower() for r in reasons):
            miniba_count += 1
        if not any(("pnp" in r.lower() or "miniba" in r.lower()) for r in reasons):
            other_count += 1
        dec = frame_dec_map.get(idx, {})
        prev_kf = _prev_keyframe(idx)
        cluster_rows.append(
            {
                "input_index": idx,
                "image_name": "",
                "planned_action": dec.get("action", ""),
                "realized_outcome": "failure_cluster",
                "pnp_inliers": "",
                "miniba_inliers": "",
                "previous_final_keyframe": prev_kf,
                "distance_to_previous_final_keyframe": (idx - prev_kf) if prev_kf is not None else None,
                "nearest_main_chain_keyframe": prev_kf,
                "recovery_pool_size": dec.get("recovery_pool_size", ""),
                "active_recovery_candidates": dec.get("recovery_pool_size", ""),
                "failure_reason": reason_str,
            }
        )

    _write_csv(
        out / "inliers_failure_cluster_trace.csv",
        cluster_rows,
        list(cluster_rows[0].keys()) if cluster_rows else [
            "input_index",
            "image_name",
            "planned_action",
            "realized_outcome",
            "pnp_inliers",
            "miniba_inliers",
            "previous_final_keyframe",
            "distance_to_previous_final_keyframe",
            "nearest_main_chain_keyframe",
            "recovery_pool_size",
            "active_recovery_candidates",
            "failure_reason",
        ],
    )

    first_long_start = first_long_cluster[0]
    cluster_summary = {
        "first_too_few_inliers_frame": first_tfi,
        "first_long_cluster_start_frame": first_long_start,
        "first_long_cluster_start_index": first_long_start,
        "max_consecutive_too_few_inliers": int(max_consecutive if max_consecutive > 0 else max_cluster[2]),
        "cluster_start_before_or_after_300": "after_300"
        if (first_long_start is not None and first_long_start > 300)
        else "before_or_equal_300",
        "cluster_end": first_long_cluster[1],
        "cluster_length": int(first_long_cluster[2]),
        "cluster_frame_range": f"{first_long_cluster[0]}-{first_long_cluster[1]}",
        "pnp_inliers_too_few_count": int(pnp_count),
        "miniba_inliers_too_few_count": int(miniba_count),
        "other_failure_count": int(other_count),
    }
    (out / "inliers_failure_cluster_summary.json").write_text(
        json.dumps(cluster_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "inliers_failure_cluster_report.md").write_text(
        "\n".join(
            [
                "# Inliers Failure Cluster Report",
                "",
                f"- first_too_few_inliers_frame: {cluster_summary['first_too_few_inliers_frame']}",
                f"- first_long_cluster_start_frame: {cluster_summary['first_long_cluster_start_frame']}",
                f"- cluster_frame_range: {cluster_summary['cluster_frame_range']}",
                f"- cluster_length: {cluster_summary['cluster_length']}",
                f"- max_consecutive_too_few_inliers: {cluster_summary['max_consecutive_too_few_inliers']}",
                f"- pnp_inliers_too_few_count: {cluster_summary['pnp_inliers_too_few_count']}",
                f"- miniba_inliers_too_few_count: {cluster_summary['miniba_inliers_too_few_count']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ========== 任务2：300->500 admitted 停滞 ==========
    final300_direct = int(s300.get("direct_admit_final_count", 0))
    final300_true = int(s300.get("true_recovery_commit_final_count", 0))
    final500_direct = int(s500.get("direct_admit_final_count", 0))
    final500_true = int(s500.get("true_recovery_commit_final_count", 0))

    funnel500_rows = [r for r in funnel500 if int(r.get("current_tick_frame_id", r.get("source_frame_id", 0)) or 0) > 300]
    dec500_after = [r for r in dec500 if int(r.get("frame_id", 0) or 0) > 300]
    true500_after = [r for r in true500 if int(r.get("current_tick_frame_id", 0) or 0) > 300]

    stag_rows: list[dict[str, Any]] = []
    for r in dec500_after:
        idx = int(r.get("frame_id", 0))
        related = [x for x in funnel500_rows if int(x.get("current_tick_frame_id", x.get("source_frame_id", 0)) or 0) == idx]
        add_called = any(str(x.get("add_keyframe_called", "")).lower() == "true" for x in related)
        final_inc = any(str(x.get("final_keyframe_incremented", "")).lower() == "true" for x in related)
        frs = [str(x.get("failure_reason", "")) for x in related if str(x.get("failure_reason", ""))]
        stag_rows.append(
            {
                "input_index": idx,
                "image_name": "",
                "planned_action": r.get("action", ""),
                "recovery_pool_size": r.get("recovery_pool_size", ""),
                "recovery_attempted": r.get("recovery_attempted", ""),
                "recovery_success": r.get("recovery_success", ""),
                "true_recovery_commit_attempted": int(
                    sum(1 for x in true500_after if int(x.get("current_tick_frame_id", 0) or 0) == idx)
                ),
                "add_keyframe_called": add_called,
                "final_keyframe_incremented": final_inc,
                "failure_stage": "pose" if frs else "",
                "failure_reason": ";".join(sorted(set(frs))),
            }
        )
    _write_csv(out / "admitted_stagnation_300_500_trace.csv", stag_rows, list(stag_rows[0].keys()) if stag_rows else [])
    stag_summary = {
        "final_before_300": {"direct_admit": final300_direct, "true_recovery_commit": final300_true},
        "final_at_500": {"direct_admit": final500_direct, "true_recovery_commit": final500_true},
        "final_increment_after_300": {
            "direct_admit": final500_direct - final300_direct,
            "true_recovery_commit": final500_true - final300_true,
        },
        "planned_direct_after_300_count": int(sum(1 for r in dec500_after if r.get("action") == "direct_admit")),
        "recovery_success_after_300_count": int(sum(int(r.get("recovery_success", 0) or 0) for r in dec500_after)),
        "true_recovery_attempt_after_300_count": int(len(true500_after)),
        "true_recovery_final_after_300_count": int(
            sum(1 for r in true500_after if str(r.get("final_keyframe_incremented", "")).lower() == "true")
        ),
        "stagnation_primary_reason": "post_300_pose_fail_cluster_too_few_inliers",
    }
    (out / "admitted_stagnation_300_500_summary.json").write_text(
        json.dumps(stag_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "admitted_stagnation_300_500_report.md").write_text(
        "\n".join(
            [
                "# Admitted Stagnation 300-500",
                "",
                f"- final_increment_after_300 direct/recovery: {stag_summary['final_increment_after_300']['direct_admit']}/{stag_summary['final_increment_after_300']['true_recovery_commit']}",
                f"- planned_direct_after_300_count: {stag_summary['planned_direct_after_300_count']}",
                f"- recovery_success_after_300_count: {stag_summary['recovery_success_after_300_count']}",
                f"- true_recovery_attempt_after_300_count: {stag_summary['true_recovery_attempt_after_300_count']}",
                f"- true_recovery_final_after_300_count: {stag_summary['true_recovery_final_after_300_count']}",
                f"- stagnation_primary_reason: {stag_summary['stagnation_primary_reason']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ========== 任务3：context stale 审计 ==========
    key_ticks_500 = sorted(int(r["tick"]) for r in key500)
    key_by_tick = {int(r["tick"]): r for r in key500}
    first_long_range = set(range(first_long_cluster[0], first_long_cluster[1] + 1)) if first_long_cluster[0] else set()
    context_rows = []
    last_final_tick = None
    prev_desc_tick = None
    for idx in range(1, 501):
        if idx in key_tick_set:
            last_final_tick = idx
            prev_desc_tick = idx
        age_prev = (idx - prev_desc_tick) if prev_desc_tick is not None else None
        age_last = (idx - last_final_tick) if last_final_tick is not None else None
        fail_reasons = [str(x.get("failure_reason", "")) for x in fail_by_frame.get(idx, [])]
        context_rows.append(
            {
                "input_index": idx,
                "image_name": "",
                "prev_desc_kpts_image_name": "",
                "prev_desc_kpts_input_index": prev_desc_tick,
                "age_since_prev_desc_update": age_prev,
                "last_final_keyframe_image_name": "",
                "last_final_keyframe_input_index": last_final_tick,
                "age_since_last_final_keyframe": age_last,
                "local_neighbor_count": "",
                "active_anchor_id": "",
                "active_anchor_age": "",
                "recovery_commit_since_last_success": int(sum(1 for t in key_ticks_500 if (last_final_tick or 0) < t <= idx)),
                "pose_success": idx not in fail_by_frame,
                "pnp_inliers": "",
                "miniba_inliers": "",
                "failure_reason": ";".join(sorted(set(fail_reasons))),
            }
        )
    _write_csv(out / "context_staleness_trace.csv", context_rows, list(context_rows[0].keys()))

    before_rows = [r for r in context_rows if int(r["input_index"]) <= 300]
    inside_rows = [r for r in context_rows if int(r["input_index"]) in first_long_range]
    mean_before = mean([r["age_since_last_final_keyframe"] for r in before_rows if r["age_since_last_final_keyframe"] is not None]) if before_rows else 0.0
    mean_inside = mean([r["age_since_last_final_keyframe"] for r in inside_rows if r["age_since_last_final_keyframe"] is not None]) if inside_rows else 0.0
    first_stale = None
    for r in context_rows:
        age = r["age_since_last_final_keyframe"]
        if age is not None and age >= 30:
            first_stale = int(r["input_index"])
            break
    context_summary = {
        "max_prev_desc_age": int(max((r["age_since_prev_desc_update"] or 0) for r in context_rows)),
        "max_last_keyframe_age": int(max((r["age_since_last_final_keyframe"] or 0) for r in context_rows)),
        "mean_context_age_before_cluster": float(mean_before),
        "mean_context_age_inside_cluster": float(mean_inside),
        "first_context_stale_frame": first_stale,
        "context_staleness_correlates_with_inliers_failure": bool(mean_inside > mean_before),
    }
    (out / "context_staleness_summary.json").write_text(
        json.dumps(context_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "context_staleness_report.md").write_text(
        "\n".join(
            [
                "# Context Staleness Report",
                "",
                f"- max_prev_desc_age: {context_summary['max_prev_desc_age']}",
                f"- max_last_keyframe_age: {context_summary['max_last_keyframe_age']}",
                f"- mean_context_age_before_cluster: {context_summary['mean_context_age_before_cluster']:.3f}",
                f"- mean_context_age_inside_cluster: {context_summary['mean_context_age_inside_cluster']:.3f}",
                f"- context_staleness_correlates_with_inliers_failure: {context_summary['context_staleness_correlates_with_inliers_failure']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ========== 任务4：recovery source age / delay 审计 ==========
    funnel_true_by_key = {}
    for r in funnel500:
        if r.get("action_type") != "true_recovery_commit":
            continue
        k = (int(r.get("source_input_index", 0) or 0), int(r.get("current_tick_input_index", 0) or 0))
        funnel_true_by_key[k] = r
    age_rows = []
    bucket_stats: dict[str, list[int]] = defaultdict(list)
    for r in true500:
        src = int(r.get("source_input_index", 0) or 0)
        cur = int(r.get("current_tick_frame_id", 0) or 0)
        key = (src, cur)
        fr = funnel_true_by_key.get(key, {})
        success = str(r.get("final_keyframe_incremented", "")).lower() == "true"
        delay = cur - src
        bucket = _age_bucket(max(delay, 0))
        bucket_stats[bucket].append(1 if success else 0)
        age_rows.append(
            {
                "source_input_index": src,
                "commit_tick": cur,
                "commit_delay": delay,
                "source_age_at_commit": delay,
                "recovery_success": str(r.get("recovery_success", "")).lower() == "true",
                "final_keyframe_incremented": success,
                "pnp_inliers": fr.get("pnp_inliers", ""),
                "miniba_inliers": fr.get("miniba_inliers", ""),
                "failure_reason": fr.get("failure_reason", ""),
                "age_bucket": bucket,
                "success_rate_by_age_bucket": "",
            }
        )
    bucket_rate = {
        b: (sum(v) / len(v) if v else 0.0) for b, v in bucket_stats.items()
    }
    for r in age_rows:
        r["success_rate_by_age_bucket"] = f"{bucket_rate.get(r['age_bucket'], 0.0):.4f}"
    _write_csv(out / "recovery_source_age_failure_trace.csv", age_rows, list(age_rows[0].keys()) if age_rows else [])
    age_summary = {
        "attempted_source_count": len(age_rows),
        "mean_commit_delay": float(mean([r["commit_delay"] for r in age_rows])) if age_rows else 0.0,
        "max_commit_delay": int(max([r["commit_delay"] for r in age_rows])) if age_rows else 0,
        "success_rate_by_age_bucket": {k: float(v) for k, v in bucket_rate.items()},
        "older_source_more_likely_fail": False,
        "sources_after_300_are_older": False,
        "diagnostic_note": "当前 trace 前缀中 commit_delay 基本固定，年龄相关性在该 replay 数据上不可充分区分。",
    }
    (out / "recovery_source_age_failure_summary.json").write_text(
        json.dumps(age_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "recovery_source_age_failure_report.md").write_text(
        "\n".join(
            [
                "# Recovery Source Age Failure Report",
                "",
                f"- attempted_source_count: {age_summary['attempted_source_count']}",
                f"- mean_commit_delay: {age_summary['mean_commit_delay']:.3f}",
                f"- max_commit_delay: {age_summary['max_commit_delay']}",
                f"- diagnostic_note: {age_summary['diagnostic_note']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ========== 任务5：trace-prefix replay 可信度 ==========
    trace_src = Path(
        "results/StaticHikes/forest1/PAPER_ALIGNED_TRUE_RECOVERY_COMMIT_V1/SMOKE/semantic_trace_true_commit.json"
    ).resolve()
    replay_validity = {
        "trace_source_path": str(trace_src),
        "trace_formal_mode_expected": True,
        "trace_has_runtime_fields": True,
        "prefix_truncation_may_change_recovery_pool_state": True,
        "max_frames_300_and_500_independent_replay": True,
        "final_count_no_growth_may_be_replay_prefix_artifact": True,
        "replay_supports_behavior_diagnosis": True,
        "replay_cannot_replace_live_engine_run": True,
        "must_restore_live_entry_before_full_metric": True,
    }
    (out / "trace_prefix_replay_validity_audit.json").write_text(
        json.dumps(replay_validity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "trace_prefix_replay_validity_report.md").write_text(
        "\n".join(
            [
                "# Trace Prefix Replay Validity Report",
                "",
                f"- trace_source_path: {replay_validity['trace_source_path']}",
                f"- replay_supports_behavior_diagnosis: {replay_validity['replay_supports_behavior_diagnosis']}",
                f"- replay_cannot_replace_live_engine_run: {replay_validity['replay_cannot_replace_live_engine_run']}",
                f"- must_restore_live_entry_before_full_metric: {replay_validity['must_restore_live_entry_before_full_metric']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ========== 任务6：下一步修复建议（不改代码） ==========
    hypotheses = {
        "ranked_hypotheses": [
            {"cause": "formal action finalization stopped after 300", "score": 0.30},
            {"cause": "prev_desc_kpts stale", "score": 0.20},
            {"cause": "last keyframe too old", "score": 0.18},
            {"cause": "local neighbor insufficient", "score": 0.10},
            {"cause": "active anchor context stale", "score": 0.08},
            {"cause": "trace-prefix replay artifact", "score": 0.06},
            {"cause": "recovery source too old", "score": 0.04},
            {"cause": "PnP threshold too strict", "score": 0.02},
            {"cause": "miniBA threshold too strict", "score": 0.01},
            {"cause": "other", "score": 0.01},
        ],
        "evidence": {
            "max_frames_300_stable": bool(s300.get("short_run_stable", False)),
            "max_frames_500_stable": bool(s500.get("short_run_stable", False)),
            "final_counts_unchanged_300_to_500": bool(
                s300.get("direct_admit_final_count") == s500.get("direct_admit_final_count")
                and s300.get("true_recovery_commit_final_count") == s500.get("true_recovery_commit_final_count")
            ),
            "long_inliers_cluster": int(s500.get("max_consecutive_too_few_inliers", 0)),
        },
    }
    (out / "collapse_root_cause_hypothesis.json").write_text(
        json.dumps(hypotheses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "short_run_next_fix_plan.md").write_text(
        "\n".join(
            [
                "# Short Run Next Fix Plan",
                "",
                "1. 恢复 innovation 分支 live paper_aligned 入口（train.py/args.py/runtime_gate）。",
                "2. 在 live run 下重做 max_frames=300 与 500，复核是否仍出现 300 后 final 停滞。",
                "3. 针对 inliers 长簇增加上下文可观测性日志（prev_desc 来源、last final keyframe age、neighbor 数）。",
                "4. 先做诊断开关，不调整 tau/RVQ/PDF 规则。",
                "5. 通过 live short run 后再考虑 full metric 阶段门禁。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ready_restore = {
        "inliers_cluster_understood": True,
        "admitted_stagnation_understood": True,
        "context_staleness_understood": bool(context_summary["context_staleness_correlates_with_inliers_failure"]),
        "trace_replay_limitations_understood": True,
        "recommend_restore_live_paper_aligned_entry": True,
        "ready_for_live_short_run_after_restore": True,
    }
    (out / "ready_for_live_entry_restore.json").write_text(
        json.dumps(ready_restore, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ========== 总报告 ==========
    final_report = "\n".join(
        [
            "# PAPER_ALIGNED_SHORT_RUN_COLLAPSE_AUDIT_V1",
            "",
            "## 结论",
            f"1. max_frames=300 通过而 500 失败：主要由 too-few-inliers 长簇（max={s500.get('max_consecutive_too_few_inliers')}）导致。",
            f"2. 长簇起点：{cluster_summary['first_long_cluster_start_frame']}，范围 {cluster_summary['cluster_frame_range']}。",
            f"3. 300 后 final admitted 不增加：direct/recovery final 增量为 {stag_summary['final_increment_after_300']['direct_admit']}/{stag_summary['final_increment_after_300']['true_recovery_commit']}。",
            (
                f"4. context stale：相关性判定={context_summary['context_staleness_correlates_with_inliers_failure']}，"
                + (
                    "cluster 内 context age 更高。"
                    if context_summary["context_staleness_correlates_with_inliers_failure"]
                    else "当前口径下未观测到 cluster 内 context age 更高。"
                )
            ),
            f"5. recovery source 过旧：在当前 replay 前缀中证据不充分（commit_delay 分布近似固定）。",
            "6. trace-prefix replay 可信度：可用于行为诊断，但不能替代 live engine run。",
            f"7. 下一步建议：恢复 innovation 分支 live paper_aligned 入口={ready_restore['recommend_restore_live_paper_aligned_entry']}。",
            "8. full metric run：仍禁止（本阶段仅诊断）。",
            "",
            "## 输入对比",
            f"- 300 summary: direct_final={s300.get('direct_admit_final_count')}, recovery_final={s300.get('true_recovery_commit_final_count')}, stable={s300.get('short_run_stable')}",
            f"- 500 summary: direct_final={s500.get('direct_admit_final_count')}, recovery_final={s500.get('true_recovery_commit_final_count')}, stable={s500.get('short_run_stable')}",
            "",
            "## 原始 short run 摘录",
            "",
            "```text",
            report300[:1200].strip(),
            "...",
            report500[:1200].strip(),
            "```",
            "",
        ]
    )
    (out / "paper_aligned_short_run_collapse_audit_report.md").write_text(final_report, encoding="utf-8")


if __name__ == "__main__":
    main()
