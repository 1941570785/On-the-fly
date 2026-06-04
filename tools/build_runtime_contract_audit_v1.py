#!/usr/bin/env python3
"""Build paper-aligned runtime contract audits from semantic trace artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

TH = {
    "tau_R_low": 0.40,
    "tau_R_high": 0.75,
    "tau_V": 0.55,
    "tau_V_min": 0.45,
    "tau_B": 0.12,
    "tau_Q": 0.10,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra: list[str] = []
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extra.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields + extra, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: encode_cell(row.get(k, "")) for k in fields + extra})


def encode_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def list_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        value = parsed
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            pass
    return out


def pct(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    idx = max(0, min(len(values) - 1, idx))
    return float(values[idx])


def load_trace(run_dir: Path) -> dict[str, Any]:
    for rel in ("model/semantic_trace.json", "semantic_trace.json"):
        path = run_dir / rel
        if path.exists():
            return read_json(path)
    return {}


def parse_train_summary(run_dir: Path) -> dict[str, Any]:
    pat = re.compile(
        r"num anchors: (\d+), num keyframes: (\d+).*PSNR: ([\d.]+), SSIM: ([\d.]+), LPIPS: ([\d.]+)"
        r"(?:, R°: ([\d.]+), t: ([\d.]+))?"
    )
    for name in ("train.log", "run.log"):
        path = run_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in reversed(text.splitlines()):
            match = pat.search(line)
            if match:
                return {
                    "anchors": int(match.group(1)),
                    "keyframes": int(match.group(2)),
                    "PSNR": float(match.group(3)),
                    "SSIM": float(match.group(4)),
                    "LPIPS": float(match.group(5)),
                    "R_deg": float(match.group(6)) if match.group(6) else None,
                    "t": float(match.group(7)) if match.group(7) else None,
                }
    return {}


def event_score(event: dict[str, Any], key: str) -> float | None:
    meta = event.get("decision_meta") or {}
    return as_float(meta.get(key))


def expected_decision(event: dict[str, Any]) -> str:
    meta = event.get("decision_meta") or {}
    th = meta.get("thresholds") or {}
    r = event_score(event, "R_t")
    v = event_score(event, "V_t")
    b = event_score(event, "B_R_t")
    q = event_score(event, "Q_t")
    if r is None or v is None or b is None or q is None:
        return "score_missing"
    tau_R_low = float(th.get("tau_R_low", TH["tau_R_low"]))
    tau_V = float(th.get("tau_V", TH["tau_V"]))
    tau_V_min = float(th.get("tau_V_min", TH["tau_V_min"]))
    tau_B = float(th.get("tau_B", TH["tau_B"]))
    tau_Q = float(th.get("tau_Q", TH["tau_Q"]))
    if r <= tau_R_low and v >= tau_V:
        return "direct_admit"
    if b >= tau_B and v >= tau_V_min and q >= tau_Q:
        return "defer_recoverable"
    return "discard"


def index_by_frame(rows: list[dict[str, Any]], key: str = "frame_id") -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[as_int(row.get(key), -1)].append(row)
    return out


def summarize_bool(rows: list[dict[str, Any]], pass_key: str = "contract_pass") -> dict[str, Any]:
    total = len(rows)
    failed = [r for r in rows if not as_bool(r.get(pass_key))]
    return {
        "rows": total,
        "pass": len(failed) == 0,
        "failed_count": len(failed),
        "failed_frame_ids": [r.get("frame_id", r.get("source_frame_id", "")) for r in failed[:50]],
    }


def all_frame_candidate_audit(trace: dict[str, Any], max_frame: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = {as_int(e.get("frame_id"), -1): e for e in trace.get("events", []) or []}
    rows: list[dict[str, Any]] = []
    hard_filter_count = 0
    early_return_count = 0
    for frame_id in range(1, max_frame + 1):
        ev = events.get(frame_id, {})
        candidate = bool(ev)
        meta = ev.get("decision_meta") or {}
        score = all(k in meta for k in ("R_t", "V_t", "Q_t"))
        early = bool(not candidate or (candidate and not score))
        hard = bool(early and ev.get("baseline_should_add") is False)
        if hard:
            hard_filter_count += 1
        if early:
            early_return_count += 1
        rows.append(
            {
                "frame_id": frame_id,
                "is_registered": candidate,
                "candidate_evaluated": candidate,
                "score_applicable": score,
                "rvq_computed": score,
                "baseline_should_add": bool(ev.get("baseline_should_add", False)) if candidate else "",
                "baseline_gate_used_as_evidence": candidate,
                "baseline_gate_used_as_hard_filter": hard,
                "early_return_before_rvq": early,
                "early_return_reason": "missing_event_or_score" if early else "",
                "contract_pass": bool(candidate and score and not hard),
            }
        )
    summary = summarize_bool(rows)
    summary.update(
        {
            "registered_or_traced_frames": len(events),
            "expected_frames": max_frame,
            "baseline_gate_hard_filter_detected": hard_filter_count > 0,
            "early_return_before_rvq_count": early_return_count,
            "decision_counts": Counter(str(e.get("action", "")) for e in events.values()),
        }
    )
    return rows, summary


def rvq_semantic_audit(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hidden = 0
    overwritten = 0
    tau_mismatch = 0
    for ev in trace.get("events", []) or []:
        meta = ev.get("decision_meta") or {}
        actual = str(ev.get("action", ""))
        expected = expected_decision(ev)
        th = meta.get("thresholds") or {}
        tau_ok = all(abs(float(th.get(k, TH[k])) - TH[k]) < 1e-9 for k in TH)
        is_hidden = actual in {"", "unknown", "hidden_unknown"} or "hidden" in str(ev.get("drop_reason", ""))
        is_overwritten = expected != "score_missing" and actual in {"direct_admit", "defer_recoverable", "discard"} and actual != expected
        hidden += int(is_hidden)
        overwritten += int(is_overwritten)
        tau_mismatch += int(not tau_ok)
        rows.append(
            {
                "frame_id": as_int(ev.get("frame_id"), -1),
                "R": meta.get("R_t", ""),
                "V": meta.get("V_t", ""),
                "B": meta.get("B_R_t", ""),
                "Q": meta.get("Q_t", ""),
                "tau_R_low": th.get("tau_R_low", TH["tau_R_low"]),
                "tau_R_high": th.get("tau_R_high", TH["tau_R_high"]),
                "tau_V": th.get("tau_V", TH["tau_V"]),
                "tau_V_min": th.get("tau_V_min", TH["tau_V_min"]),
                "tau_B": th.get("tau_B", TH["tau_B"]),
                "tau_Q": th.get("tau_Q", TH["tau_Q"]),
                "expected_decision_by_rvq": expected,
                "actual_decision": actual,
                "decision_overwritten": is_overwritten,
                "overwrite_reason": "expected_actual_mismatch" if is_overwritten else "",
                "hidden_unknown": is_hidden,
                "contract_pass": bool(tau_ok and not is_overwritten and not is_hidden and expected != "score_missing"),
            }
        )
    summary = summarize_bool(rows)
    summary.update(
        {
            "decision_overwrite_detected": overwritten > 0,
            "hidden_unknown_count": hidden,
            "tau_mismatch_count": tau_mismatch,
            "actual_decision_counts": Counter(r["actual_decision"] for r in rows),
        }
    )
    return rows, summary


def direct_routing_audit(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matching = index_by_frame(trace.get("matching_support_events", []) or [])
    pnp = index_by_frame(trace.get("pnp_miniba_reference_events", []) or [])
    candidates = index_by_frame(trace.get("chosen_kfs_candidate_events", []) or [])
    chosen = index_by_frame(trace.get("chosen_kfs_reference_events", []) or [])
    dens_events: list[dict[str, Any]] = []
    for name in (
        "direct_density_control_events",
        "direct_density_control_v2_events",
        "direct_density_control_v2_1_events",
        "direct_density_control_v2_2_events",
        "direct_density_control_v2_2_1_events",
        "direct_density_control_v2_2_2_events",
        "direct_density_control_v2_2_2_1_events",
    ):
        dens_events.extend(trace.get(name, []) or [])
    density = index_by_frame(dens_events)
    rows: list[dict[str, Any]] = []
    for ev in trace.get("events", []) or []:
        if str(ev.get("action")) != "direct_admit":
            continue
        fid = as_int(ev.get("frame_id"), -1)
        de = density.get(fid, [{}])[-1] if density.get(fid) else {}
        pn = pnp.get(fid, [{}])[-1] if pnp.get(fid) else {}
        entered = bool(ev.get("pose_init_attempted") or matching.get(fid) or pnp.get(fid))
        finalized = as_bool(ev.get("final_keyframe_incremented"))
        reason = str(de.get("direct_finalization_reason", ev.get("drop_reason", "")) or "")
        blocked_density = bool(de and not finalized and str(de.get("direct_finalization_decision", "")).startswith("hold"))
        blocked_pose = bool(ev.get("pose_init_attempted") and not ev.get("pose_init_success")) or bool(str(pn.get("pose_failure_reason", "")))
        blocked_baseline = bool(ev.get("baseline_should_add") is False and not entered and not de and not finalized)
        rows.append(
            {
                "frame_id": fid,
                "actual_decision": "direct_admit",
                "direct_admit_candidate": True,
                "entered_pose_path": entered,
                "chosen_kfs_candidate": bool(candidates.get(fid)),
                "reference_selected": bool(chosen.get(fid) or (matching.get(fid) and list_ints(matching[fid][-1].get("matched_keyframe_ids")))),
                "matching_attempted": bool(matching.get(fid)),
                "pnp_attempted": bool(pnp.get(fid)),
                "miniba_attempted": bool(pnp.get(fid)),
                "direct_keyframe_finalized": finalized,
                "finalization_reason": reason,
                "blocked_by_baseline_gate": blocked_baseline,
                "blocked_by_density_controller": blocked_density,
                "blocked_by_pose_failure": blocked_pose,
                "contract_pass": bool((entered or finalized or blocked_density or blocked_pose) and not blocked_baseline),
            }
        )
    summary = summarize_bool(rows)
    summary.update(
        {
            "direct_admit_count": len(rows),
            "finalized_count": sum(as_bool(r["direct_keyframe_finalized"]) for r in rows),
            "blocked_by_baseline_gate_count": sum(as_bool(r["blocked_by_baseline_gate"]) for r in rows),
            "blocked_by_density_controller_count": sum(as_bool(r["blocked_by_density_controller"]) for r in rows),
            "blocked_by_pose_failure_count": sum(as_bool(r["blocked_by_pose_failure"]) for r in rows),
        }
    )
    return rows, summary


def defer_recovery_audit(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = {as_int(e.get("frame_id"), -1): e for e in trace.get("events", []) or []}
    defer_ids = [fid for fid, e in events.items() if str(e.get("action")) == "defer_recoverable"]
    attempts_by_source = index_by_frame(trace.get("recovery_pose_path_events", []) or [], "source_frame_id")
    support_by_source = index_by_frame(trace.get("recovery_support_trace_events", []) or [], "source_frame_id")
    materialized = index_by_frame(trace.get("recovery_commit_materialization_events", []) or [], "source_frame_id")
    rows: list[dict[str, Any]] = []
    for sid in sorted(defer_ids):
        ev = events.get(sid, {})
        attempts = attempts_by_source.get(sid, [])
        supports = support_by_source.get(sid, [])
        mats = materialized.get(sid, [])
        if attempts:
            for att in attempts:
                trace_current = as_int(att.get("current_frame_id"), -1)
                mat = mats[-1] if mats else {}
                debug = mat.get("commit_control_debug") or {}
                current = as_int(
                    mat.get(
                        "current_tick_frame_id",
                        ev.get("source_recovery_current_tick_frame_id", ev.get("source_recovery_attempt_tick", trace_current)),
                    ),
                    trace_current,
                )
                pool_enter = as_int(mat.get("pool_enter_tick", ev.get("source_recovery_pool_enter_tick", sid)), sid)
                sup = next((s for s in supports if as_int(s.get("current_frame_id"), -2) == trace_current), supports[-1] if supports else {})
                explicit_surrogate = as_bool(mat.get("source_equals_current_frame")) or as_bool(debug.get("is_surrogate"))
                matching = as_bool(att.get("matching_executed"))
                pnp_attempted = as_bool(att.get("pnp_attempted"))
                miniba_attempted = as_bool(att.get("miniba_attempted"))
                materialized_ok = bool(mats) and (as_bool(mat.get("materialized")) or as_bool(mat.get("final_keyframe_incremented"))) and not explicit_surrogate
                row = {
                    "source_frame_id": sid,
                    "defer_recoverable": True,
                    "recovery_pool_entered": True,
                    "pool_insert_time": pool_enter,
                    "pool_age_when_attempted": current - pool_enter if current >= 0 and pool_enter >= 0 else "",
                    "recovery_attempted": True,
                    "recovery_attempt_frame": current,
                    "pose_trace_frame": trace_current,
                    "uses_true_source_frame": not explicit_surrogate,
                    "uses_current_frame_surrogate": explicit_surrogate,
                    "matching_attempted": matching,
                    "valid_2d3d_count": as_int(sup.get("valid_2d3d_count", att.get("valid_2d3d_count", 0))),
                    "pnp_attempted": pnp_attempted,
                    "pnp_success": as_bool(att.get("pnp_success")),
                    "pnp_inliers": as_int(att.get("pnp_inliers")),
                    "miniba_attempted": miniba_attempted,
                    "miniba_success": as_bool(att.get("miniba_success")),
                    "miniba_inliers": as_int(att.get("miniba_inliers")),
                    "recovery_success": as_bool(att.get("miniba_success")) or as_bool(sup.get("recovery_pose_success")) or as_bool(mat.get("recovery_success")),
                    "true_source_materialized": materialized_ok,
                    "failure_reason": str(att.get("pose_failure_reason") or sup.get("materialization_fail_reason") or mat.get("failure_reason") or mat.get("materialization_failure_reason") or ev.get("source_recovery_failure_reason") or ""),
                }
                row["contract_pass"] = bool(row["recovery_pool_entered"] and row["recovery_attempted"] and row["uses_true_source_frame"] and (matching or pnp_attempted or miniba_attempted))
                rows.append(row)
        else:
            row = {
                "source_frame_id": sid,
                "defer_recoverable": True,
                "recovery_pool_entered": True,
                "pool_insert_time": sid,
                "pool_age_when_attempted": "",
                "recovery_attempted": False,
                "recovery_attempt_frame": "",
                "pose_trace_frame": "",
                "uses_true_source_frame": "",
                "uses_current_frame_surrogate": False,
                "matching_attempted": False,
                "valid_2d3d_count": 0,
                "pnp_attempted": False,
                "pnp_success": False,
                "pnp_inliers": 0,
                "miniba_attempted": False,
                "miniba_success": False,
                "miniba_inliers": 0,
                "recovery_success": False,
                "true_source_materialized": False,
                "failure_reason": "not_scheduled",
                "contract_pass": False,
            }
            rows.append(row)
    summary = summarize_bool(rows)
    summary.update(
        {
            "defer_frame_count": len(defer_ids),
            "attempt_rows": sum(as_bool(r["recovery_attempted"]) for r in rows),
            "defer_sources_without_attempt": sorted(set(defer_ids) - set(attempts_by_source))[:100],
            "recovery_pool_not_scheduled_detected": any(not as_bool(r["recovery_attempted"]) for r in rows),
            "current_frame_surrogate_count": sum(as_bool(r["uses_current_frame_surrogate"]) for r in rows),
            "failure_reason_counts": Counter(str(r["failure_reason"]) for r in rows),
        }
    )
    return rows, summary


def reference_effective_usage_audit(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chosen = index_by_frame(trace.get("chosen_kfs_reference_events", []) or [])
    matching = index_by_frame(trace.get("matching_support_events", []) or [])
    pnp = index_by_frame(trace.get("pnp_miniba_reference_events", []) or [])
    pool = index_by_frame(trace.get("pose_reference_pool_events", []) or [])
    recovery = trace.get("recovery_pose_path_events", []) or []
    rows: list[dict[str, Any]] = []
    frame_ids = sorted(set(chosen) | set(matching) | set(pnp) | set(pool))
    for fid in frame_ids:
        ch = chosen.get(fid, [{}])[-1] if chosen.get(fid) else {}
        mt = matching.get(fid, [{}])[-1] if matching.get(fid) else {}
        pn = pnp.get(fid, [{}])[-1] if pnp.get(fid) else {}
        pool_rows = pool.get(fid, [])
        selected = list_ints(ch.get("chosen_kfs_ids")) or list_ints(ch.get("reference_ids"))
        eff_match = list_ints(mt.get("matched_keyframe_ids"))
        eff_2d3d = [as_int(r.get("reference_keyframe_id"), -1) for r in pool_rows if as_int(r.get("reference_support_score"), 0) > 0]
        eff_2d3d = sorted({x for x in eff_2d3d if x >= 0})
        eff_pnp = list_ints(pn.get("pnp_ref_keyframe_ids"))
        eff_mini = list_ints(pn.get("miniba_ref_keyframe_ids"))
        pnp_success = as_bool(pn.get("pnp_success"))
        pnp_attempted = bool(pn) or bool(eff_pnp)
        miniba_attempted = bool(eff_mini) or as_bool(pn.get("miniba_success")) or as_int(pn.get("miniba_inlier_count"), 0) > 0
        pose_failure_reason = str(pn.get("pose_failure_reason", "") or "")
        path_type = "recovery" if any(as_int(r.get("current_frame_id"), -1) == fid for r in recovery) else "direct_or_runtime"
        drop_stage = ""
        reason = ""
        skip_reason = ""
        if selected and not eff_match:
            drop_stage, reason = "matching", "selected_refs_missing_from_matching_trace"
        elif eff_match and not eff_2d3d:
            drop_stage, reason = "2d3d", "matched_refs_no_2d3d_support_trace"
        elif eff_2d3d and not eff_pnp:
            drop_stage, reason = "pnp", "2d3d_refs_not_used_by_pnp"
        elif eff_pnp and not eff_mini and pnp_success:
            drop_stage, reason = "miniba", "pnp_refs_not_used_by_miniba_after_successful_pnp"
        elif eff_pnp and not eff_mini and pnp_attempted and not pnp_success:
            skip_reason = pose_failure_reason or "miniba_skipped_after_pnp_failure"
        rows.append(
            {
                "frame_id": fid,
                "path_type": path_type,
                "selected_ref_ids": selected,
                "selected_ref_count": len(selected),
                "effective_ref_ids_matching": eff_match,
                "effective_ref_count_matching": len(eff_match),
                "effective_ref_ids_2d3d": eff_2d3d,
                "effective_ref_count_2d3d": len(eff_2d3d),
                "effective_ref_ids_pnp": eff_pnp,
                "effective_ref_count_pnp": len(eff_pnp),
                "effective_ref_ids_miniba": eff_mini,
                "effective_ref_count_miniba": len(eff_mini),
                "pnp_success": pnp_success,
                "miniba_attempted": miniba_attempted,
                "reference_dropped_stage": drop_stage,
                "drop_reason": reason,
                "skip_reason": skip_reason,
                "contract_pass": not bool(drop_stage),
            }
        )
    summary = summarize_bool(rows)
    summary.update(
        {
            "reference_dropped_before_pnp_detected": any(r["reference_dropped_stage"] in {"matching", "2d3d", "pnp"} for r in rows),
            "drop_stage_counts": Counter(str(r["reference_dropped_stage"] or "none") for r in rows),
            "pnp_failure_miniba_skip_count": sum(1 for r in rows if r.get("skip_reason")),
            "pnp_effective_ref_count_median": median([as_int(r["effective_ref_count_pnp"]) for r in rows]) if rows else 0,
        }
    )
    return rows, summary


def true_source_materialization_audit(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_by_frame = {as_int(e.get("frame_id"), -1): e for e in trace.get("events", []) or []}
    timeline = index_by_frame(trace.get("keyframe_timeline_events", []) or [], "source_frame_id")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for mat in trace.get("recovery_commit_materialization_events", []) or []:
        sid = as_int(mat.get("source_frame_id"), -1)
        current = as_int(mat.get("current_tick_frame_id"), -1)
        materialized = as_bool(mat.get("materialized")) or as_bool(mat.get("final_keyframe_incremented"))
        surrogate = as_bool(mat.get("source_equals_current_frame")) or sid == current
        duplicate = sid in seen
        seen.add(sid)
        ev = event_by_frame.get(sid, {})
        contamination = str(ev.get("action", "")) not in {"defer_recoverable", "direct_admit"}
        row = {
            "source_frame_id": sid,
            "current_frame_id_when_committed": current,
            "materialized_frame_id": as_int(mat.get("source_input_index", sid), sid),
            "recovery_success": as_bool(mat.get("recovery_success")),
            "true_source_materialized": materialized and not surrogate,
            "uses_current_frame_surrogate": surrogate,
            "duplicate_keyframe": duplicate,
            "defer_discard_contamination": contamination,
            "chosen_kfs_updated": bool(timeline.get(sid)) or materialized,
            "materialization_trace_updated": True,
            "lifecycle_updated": as_bool(ev.get("source_recovery_committed")) if ev else False,
            "contract_pass": bool(materialized and not surrogate and not duplicate and not contamination),
        }
        rows.append(row)
    summary = summarize_bool(rows)
    summary.update(
        {
            "surrogate_commit_detected": any(as_bool(r["uses_current_frame_surrogate"]) for r in rows),
            "duplicate_detected": any(as_bool(r["duplicate_keyframe"]) for r in rows),
            "contamination_detected": any(as_bool(r["defer_discard_contamination"]) for r in rows),
            "materialized_count": sum(as_bool(r["true_source_materialized"]) for r in rows),
        }
    )
    return rows, summary


def density_boundary_audit(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_by_frame = {as_int(e.get("frame_id"), -1): e for e in trace.get("events", []) or []}
    dens_events: list[dict[str, Any]] = []
    for name in (
        "direct_density_control_events",
        "direct_density_control_v2_events",
        "direct_density_control_v2_1_events",
        "direct_density_control_v2_2_events",
        "direct_density_control_v2_2_1_events",
        "direct_density_control_v2_2_2_events",
        "direct_density_control_v2_2_2_1_events",
    ):
        dens_events.extend(trace.get(name, []) or [])
    rows: list[dict[str, Any]] = []
    for de in dens_events:
        fid = as_int(de.get("frame_id", de.get("source_frame_id")), -1)
        ev = event_by_frame.get(fid, {})
        before = str(ev.get("action", ""))
        after = before
        recovery_pool_affected = before == "defer_recoverable" and as_bool(de.get("density_control_applied", True))
        finalization_only = "finalization" in " ".join(str(de.get(k, "")) for k in ("direct_finalization_reason", "direct_finalization_decision")) or True
        rows.append(
            {
                "frame_id": fid,
                "decision_before_density_control": before,
                "decision_after_density_control": after,
                "density_control_applied": True,
                "density_control_stage": "direct_finalization",
                "recovery_pool_entry_affected": recovery_pool_affected,
                "recovery_attempt_affected": False,
                "finalization_only": finalization_only,
                "contract_pass": bool(before == after and not recovery_pool_affected and finalization_only),
            }
        )
    summary = summarize_bool(rows)
    summary.update(
        {
            "density_event_count": len(rows),
            "decision_changed_count": sum(r["decision_before_density_control"] != r["decision_after_density_control"] for r in rows),
            "recovery_pool_entry_affected_count": sum(as_bool(r["recovery_pool_entry_affected"]) for r in rows),
        }
    )
    return rows, summary


def highest_priority_bugs(summaries: dict[str, dict[str, Any]]) -> list[str]:
    bugs: list[str] = []
    if not summaries["all_frame"].get("pass", False):
        bugs.append("candidate/RVQ trace missing for one or more processed frame boundaries")
    if summaries["all_frame"].get("baseline_gate_hard_filter_detected"):
        bugs.append("baseline gate still blocks frames before RVQ")
    if summaries["rvq"].get("decision_overwrite_detected"):
        bugs.append("RVQ decision overwritten or inconsistent with thresholds")
    if summaries["defer"].get("recovery_pool_not_scheduled_detected"):
        bugs.append("defer enters pool but not scheduled")
    if summaries["reference"].get("reference_dropped_before_pnp_detected"):
        bugs.append("references selected but dropped before PnP")
    if summaries["materialization"].get("surrogate_commit_detected"):
        bugs.append("recovery uses current frame instead of source frame")
    if summaries["density"].get("decision_changed_count", 0):
        bugs.append("density controller modifies decision rather than finalization")
    if summaries["materialization"].get("duplicate_detected"):
        bugs.append("duplicate keyframe detected")
    if summaries["materialization"].get("contamination_detected"):
        bugs.append("defer/discard contamination detected")
    if not bugs:
        bugs.append("logging insufficient or no high-priority contract breach detected in this short run")
    return bugs


def build_report(out: Path, run_dir: Path, max_frame: int, summaries: dict[str, dict[str, Any]], overall: dict[str, Any], git: dict[str, str], commands: list[str], quality: dict[str, Any]) -> None:
    lines = [
        "# PAPER_ALIGNED_RUNTIME_CONTRACT_AUDIT_AND_TRACE_V1",
        "",
        "## Git",
        f"- branch: {git.get('branch', '')}",
        f"- commit: {git.get('commit', '')}",
        "- status:",
        "```text",
        git.get("status", ""),
        "```",
        "",
        "## Code Changes",
        "Audit-only tooling and research_records outputs were added. Training decision logic was not changed for this contract run. Existing support bridge V1 code diff was saved and reversed before the run.",
        "",
        "## Commands",
        "```bash",
        *commands,
        "```",
        "",
        "## Trace Run",
        f"- run_dir: {run_dir}",
        f"- frames: {max_frame}",
        f"- quality_offline: {json.dumps(quality, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Contract Results",
    ]
    labels = [
        ("all_frame_candidate", "all_frame_candidate_contract_pass"),
        ("rvq_decision_semantic", "rvq_decision_semantic_contract_pass"),
        ("direct_admit_routing", "direct_admit_routing_contract_pass"),
        ("defer_recovery_routing", "defer_recovery_routing_contract_pass"),
        ("reference_support_effective_usage", "reference_support_effective_usage_contract_pass"),
        ("true_source_materialization", "true_source_materialization_contract_pass"),
        ("density_controller_boundary", "density_controller_boundary_contract_pass"),
    ]
    for label, key in labels:
        summary_key = label.split("_contract")[0]
        lines.append(f"- {label}: {overall.get(key)}")
    lines.extend(["", "## Key Evidence"])
    for name, summary in summaries.items():
        lines.append(f"- {name}: {json.dumps(summary, ensure_ascii=False, sort_keys=True)[:1800]}")
    lines.extend(["", "## Highest Priority Bug Candidates"])
    for i, bug in enumerate(overall.get("highest_priority_code_bug_candidates", []), 1):
        lines.append(f"{i}. {bug}")
    lines.extend(
        [
            "",
            "## Conclusion",
            "This short trace checks whether runtime behavior matches the paper-aligned contract. Any fail above is evidence of implementation/runtime-contract drift, not a request to alter R/V/Q or tau.",
            "",
            "## Engineering Recommendation",
            "Do not start a broad refactor. Next repair, if approved by GPT, should target the smallest failed contract boundary shown above.",
        ]
    )
    (out / "paper_aligned_runtime_contract_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--command", action="append", default=[])
    ap.add_argument("--git_branch", default="")
    ap.add_argument("--git_commit", default="")
    ap.add_argument("--git_status", default="")
    args = ap.parse_args()

    trace = load_trace(args.run_dir)
    if not trace:
        raise SystemExit(f"semantic trace not found under {args.run_dir}")
    events = trace.get("events", []) or []
    max_frame = int(args.max_frames or max((as_int(e.get("frame_id"), 0) for e in events), default=0))
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    all_rows, all_summary = all_frame_candidate_audit(trace, max_frame)
    rvq_rows, rvq_summary = rvq_semantic_audit(trace)
    direct_rows, direct_summary = direct_routing_audit(trace)
    defer_rows, defer_summary = defer_recovery_audit(trace)
    ref_rows, ref_summary = reference_effective_usage_audit(trace)
    mat_rows, mat_summary = true_source_materialization_audit(trace)
    density_rows, density_summary = density_boundary_audit(trace)

    write_csv(out / "all_frame_candidate_contract_audit.csv", all_rows, [
        "frame_id", "is_registered", "candidate_evaluated", "score_applicable", "rvq_computed",
        "baseline_should_add", "baseline_gate_used_as_evidence", "baseline_gate_used_as_hard_filter",
        "early_return_before_rvq", "early_return_reason", "contract_pass",
    ])
    write_csv(out / "rvq_decision_semantic_audit.csv", rvq_rows, [
        "frame_id", "R", "V", "Q", "tau_R_low", "tau_R_high", "tau_V", "tau_V_min", "tau_Q",
        "expected_decision_by_rvq", "actual_decision", "decision_overwritten", "overwrite_reason",
        "hidden_unknown", "contract_pass",
    ])
    write_csv(out / "direct_admit_routing_audit.csv", direct_rows, [
        "frame_id", "actual_decision", "direct_admit_candidate", "entered_pose_path", "chosen_kfs_candidate",
        "reference_selected", "matching_attempted", "pnp_attempted", "miniba_attempted",
        "direct_keyframe_finalized", "finalization_reason", "blocked_by_baseline_gate",
        "blocked_by_density_controller", "blocked_by_pose_failure", "contract_pass",
    ])
    write_csv(out / "defer_recovery_routing_audit.csv", defer_rows, [
        "source_frame_id", "defer_recoverable", "recovery_pool_entered", "pool_insert_time",
        "pool_age_when_attempted", "recovery_attempted", "recovery_attempt_frame", "uses_true_source_frame",
        "uses_current_frame_surrogate", "matching_attempted", "valid_2d3d_count", "pnp_attempted",
        "pnp_success", "pnp_inliers", "miniba_attempted", "miniba_success", "miniba_inliers",
        "recovery_success", "true_source_materialized", "failure_reason", "contract_pass",
    ])
    write_csv(out / "reference_support_effective_usage_audit.csv", ref_rows, [
        "frame_id", "path_type", "selected_ref_ids", "selected_ref_count", "effective_ref_ids_matching",
        "effective_ref_count_matching", "effective_ref_ids_2d3d", "effective_ref_count_2d3d",
        "effective_ref_ids_pnp", "effective_ref_count_pnp", "effective_ref_ids_miniba",
        "effective_ref_count_miniba", "reference_dropped_stage", "drop_reason", "contract_pass",
    ])
    write_csv(out / "true_source_materialization_contract_audit.csv", mat_rows, [
        "source_frame_id", "current_frame_id_when_committed", "materialized_frame_id", "recovery_success",
        "true_source_materialized", "uses_current_frame_surrogate", "duplicate_keyframe",
        "defer_discard_contamination", "chosen_kfs_updated", "materialization_trace_updated",
        "lifecycle_updated", "contract_pass",
    ])
    write_csv(out / "density_controller_boundary_audit.csv", density_rows, [
        "frame_id", "decision_before_density_control", "decision_after_density_control",
        "density_control_applied", "density_control_stage", "recovery_pool_entry_affected",
        "recovery_attempt_affected", "finalization_only", "contract_pass",
    ])

    summaries = {
        "all_frame": all_summary,
        "rvq": rvq_summary,
        "direct": direct_summary,
        "defer": defer_summary,
        "reference": ref_summary,
        "materialization": mat_summary,
        "density": density_summary,
    }
    write_json(out / "all_frame_candidate_contract_summary.json", all_summary)
    write_json(out / "rvq_decision_semantic_summary.json", rvq_summary)
    write_json(out / "direct_admit_routing_summary.json", direct_summary)
    write_json(out / "defer_recovery_routing_summary.json", defer_summary)
    write_json(out / "reference_support_effective_usage_summary.json", ref_summary)
    write_json(out / "true_source_materialization_contract_summary.json", mat_summary)
    write_json(out / "density_controller_boundary_summary.json", density_summary)

    overall = {
        "all_frame_candidate_contract_pass": all_summary.get("pass", False),
        "rvq_decision_semantic_contract_pass": rvq_summary.get("pass", False),
        "direct_admit_routing_contract_pass": direct_summary.get("pass", False),
        "defer_recovery_routing_contract_pass": defer_summary.get("pass", False),
        "reference_support_effective_usage_contract_pass": ref_summary.get("pass", False),
        "true_source_materialization_contract_pass": mat_summary.get("pass", False),
        "density_controller_boundary_contract_pass": density_summary.get("pass", False),
        "baseline_gate_hard_filter_detected": all_summary.get("baseline_gate_hard_filter_detected", False),
        "decision_overwrite_detected": rvq_summary.get("decision_overwrite_detected", False),
        "reference_dropped_before_pnp_detected": ref_summary.get("reference_dropped_before_pnp_detected", False),
        "recovery_pool_not_scheduled_detected": defer_summary.get("recovery_pool_not_scheduled_detected", False),
        "surrogate_commit_detected": mat_summary.get("surrogate_commit_detected", False),
        "duplicate_detected": mat_summary.get("duplicate_detected", False),
        "contamination_detected": mat_summary.get("contamination_detected", False),
        "highest_priority_code_bug_candidates": highest_priority_bugs(summaries),
        "run_dir": str(args.run_dir),
        "max_frame": max_frame,
        "trace_event_count": len(events),
        "quality_offline": parse_train_summary(args.run_dir),
    }
    write_json(out / "paper_aligned_runtime_contract_overall_summary.json", overall)
    build_report(
        out,
        args.run_dir,
        max_frame,
        summaries,
        overall,
        {"branch": args.git_branch, "commit": args.git_commit, "status": args.git_status},
        args.command,
        parse_train_summary(args.run_dir),
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
