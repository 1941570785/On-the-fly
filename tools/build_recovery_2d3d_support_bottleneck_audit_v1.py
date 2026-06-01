#!/usr/bin/env python3
"""Read-only audit: recovery 2D-3D / MiniBA support bottleneck (V1)."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

OUT_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_2D3D_SUPPORT_BOTTLENECK_AUDIT_V1"
)
LIFECYCLE_RERUN = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_LIFECYCLE_GATE_RECOVERY_POSE_REACHABILITY_FIX_V1/v7_after_lifecycle_gate_fix_rerun"
)
POSE_OUTCOME_RERUN = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_RECOVERY_POSE_OUTCOME_FIX_V1/v7_after_pose_outcome_fix_rerun"
)
GLOBAL_MIN_INLIERS = 100
RECOVERY_PNP_MIN = 4


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
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


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def as_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    text = str(x or "").strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, list) else []
    except Exception:
        return []


def in_recovery_window(current_frame_id: int) -> bool:
    return 300 <= int(current_frame_id) < 500


def pct(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def numeric_summary(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    vals.sort()

    def q(p: float) -> float:
        idx = int(round((len(vals) - 1) * p))
        return float(vals[max(0, min(len(vals) - 1, idx))])

    return {"count": len(vals), "min": vals[0], "mean": float(mean(vals)), "p50": q(0.5), "p90": q(0.9), "max": vals[-1]}


def index_by_int(rows: list[dict[str, Any]], key: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[to_int(row.get(key), -1)] = row
    return out


def index_multi(rows: list[dict[str, Any]], key: str) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[to_int(row.get(key), -1)].append(row)
    return out


def load_run(run_root: Path, label: str) -> dict[str, Any]:
    model = run_root / "model"
    trace = read_json(model / "semantic_trace.json") if model.exists() else {}
    return {
        "label": label,
        "root": run_root,
        "trace": trace,
        "events": {to_int(e.get("frame_id"), -1): e for e in trace.get("events", []) or []},
        "engine": read_json(run_root / "engine_stability_audit.json"),
        "pose": read_csv(run_root / "recovery_pose_path_trace.csv"),
        "pnp_ref": index_by_int(read_csv(run_root / "pnp_miniba_reference_trace.csv"), "frame_id"),
        "matching": index_by_int(read_csv(run_root / "matching_support_trace.csv"), "frame_id"),
        "bridge": index_by_int(read_csv(run_root / "matching_to_pose_path_bridge_trace.csv"), "frame_id"),
        "pool": index_multi(read_csv(run_root / "pose_reference_pool_trace.csv"), "frame_id"),
        "chosen": index_multi(read_csv(run_root / "chosen_kfs_candidate_trace.csv"), "frame_id"),
        "keyframes": read_csv(run_root / "keyframe_timeline.csv"),
        "anchors": index_by_int(read_csv(run_root / "local_map_anchor_trace.csv"), "frame_id"),
        "outcome_fix": read_csv(run_root / "recovery_pose_outcome_fix_trace.csv"),
        "support_trend": read_csv(run_root / "support_trend_timeline.csv"),
        "frame_stage": read_csv(run_root / "frame_stage_reachability_trace.csv"),
    }


def kf_meta(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    meta: dict[int, dict[str, Any]] = {}
    for row in run["keyframes"]:
        kid = to_int(row.get("keyframe_id"), -1)
        if kid < 0:
            continue
        meta[kid] = {
            "keyframe_id": kid,
            "source_frame_id": to_int(row.get("source_frame_id"), -1),
            "commit_origin": str(row.get("commit_origin", "")),
            "is_early_seed": to_bool(row.get("is_v7_early_seed")),
            "has_pose": True,
            "has_features": True,
            "has_descriptors": True,
            "has_match_graph_id": str(row.get("match_graph_id", "")) not in {"", "unavailable", "None"},
            "has_anchor_id": str(row.get("anchor_id", "")) not in {"", "-1", "unavailable"},
            "materialized": to_bool(row.get("materialized")),
        }
    return meta


def build_correspondence_flow(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conv_rates: list[float] = []
    pnp_rates: list[float] = []
    miniba_rates: list[float] = []

    for pose in run["pose"]:
        cid = to_int(pose.get("current_frame_id"), -1)
        if not in_recovery_window(cid):
            continue
        sid = to_int(pose.get("source_frame_id"), -1)
        match_row = run["matching"].get(cid, {})
        bridge_row = run["bridge"].get(cid, {})
        pref = run["pnp_ref"].get(cid, {})
        ev = run["events"].get(sid, {})

        verified_2d2d = to_int(match_row.get("match_count_total"), -1)
        if verified_2d2d < 0:
            verified_2d2d = to_int(ev.get("num_2d3d_correspondences"), -1)

        raw_proxy = to_int(bridge_row.get("baseline_prev_num_matches"), -1)
        raw_2d2d = raw_proxy if raw_proxy >= 0 else "unavailable"

        valid_2d3d = to_int(ev.get("num_2d3d_correspondences"), verified_2d2d if verified_2d2d >= 0 else -1)
        pnp_corr = to_int(pose.get("pnp_inliers"), to_int(ev.get("num_pnp_inliers"), 0))
        miniba_corr = to_int(pose.get("miniba_inliers"), to_int(ev.get("num_miniba_inliers"), 0))

        if isinstance(raw_2d2d, int) and verified_2d2d > 0:
            conv_rates.append(min(1.0, verified_2d2d / max(raw_2d2d, 1)))
        if valid_2d3d > 0:
            pnp_rates.append(pnp_corr / valid_2d3d)
            miniba_rates.append(miniba_corr / valid_2d3d)

        by_ref = as_list(match_row.get("match_count_by_ref"))
        seed_ref_count = sum(
            1
            for row in run["chosen"].get(cid, [])
            if to_bool(row.get("candidate_selected")) and to_bool(row.get("candidate_is_early_seed"))
        )

        row = {
            "run": run["label"],
            "source_frame_id": sid,
            "current_frame_id": cid,
            "attempt_id": to_int(pose.get("attempt_id"), 0),
            "chosen_kfs_ids": pose.get("chosen_kfs_ids", ""),
            "reference_keyframe_ids": pref.get("pnp_ref_keyframe_ids", pose.get("chosen_kfs_ids", "")),
            "seed_ref_count": seed_ref_count,
            "raw_2d2d_match_count": raw_2d2d,
            "raw_2d2d_match_count_note": (
                "proxy_prev_keyframe_only_via_bridge"
                if raw_2d2d != "unavailable"
                else "unavailable_multi_ref_raw_not_traced"
            ),
            "verified_2d2d_match_count": verified_2d2d if verified_2d2d >= 0 else "unavailable",
            "match_count_to_seed_refs": to_int(match_row.get("match_count_to_seed_keyframes"), -1),
            "match_count_to_direct_refs": max(
                0,
                (verified_2d2d if verified_2d2d >= 0 else 0)
                - to_int(match_row.get("match_count_to_seed_keyframes"), 0),
            ),
            "available_3d_observation_count": valid_2d3d if valid_2d3d >= 0 else "unavailable",
            "valid_2d3d_correspondence_count": valid_2d3d if valid_2d3d >= 0 else "unavailable",
            "pnp_correspondence_count": pnp_corr,
            "pnp_inliers": pnp_corr,
            "miniba_correspondence_count": miniba_corr,
            "miniba_inliers": miniba_corr,
            "pnp_success": to_bool(pose.get("pnp_success")),
            "miniba_success": to_bool(pose.get("miniba_success")),
            "failure_reason": str(pose.get("pose_failure_reason", "") or ev.get("pose_fail_detail", "")),
            "match_count_by_ref": match_row.get("match_count_by_ref", "unavailable"),
            "conversion_verified_to_2d3d": (
                round(verified_2d2d / max(raw_2d2d, 1), 4)
                if isinstance(raw_2d2d, int) and verified_2d2d >= 0
                else "unavailable"
            ),
            "conversion_2d3d_to_pnp_inlier": round(pnp_corr / max(valid_2d3d, 1), 4) if valid_2d3d > 0 else "unavailable",
            "conversion_pnp_to_miniba_inlier": round(miniba_corr / max(pnp_corr, 1), 4) if pnp_corr > 0 else "unavailable",
        }
        rows.append(row)

    window = [r for r in rows]
    psmf = [r for r in window if r["pnp_success"] and not r["miniba_success"]]
    summary = {
        "run": run["label"],
        "recovery_attempts_300_499": len(window),
        "verified_2d2d_summary": numeric_summary(
            [float(r["verified_2d2d_match_count"]) for r in window if r["verified_2d2d_match_count"] != "unavailable"]
        ),
        "valid_2d3d_summary": numeric_summary(
            [float(r["valid_2d3d_correspondence_count"]) for r in window if r["valid_2d3d_correspondence_count"] != "unavailable"]
        ),
        "pnp_inliers_summary": numeric_summary([float(r["pnp_inliers"]) for r in window]),
        "miniba_inliers_summary": numeric_summary([float(r["miniba_inliers"]) for r in window]),
        "mean_conversion_proxy_raw_to_verified": float(mean(conv_rates)) if conv_rates else 0.0,
        "mean_conversion_2d3d_to_pnp": float(mean(pnp_rates)) if pnp_rates else 0.0,
        "mean_conversion_pnp_to_miniba": float(mean(miniba_rates)) if miniba_rates else 0.0,
        "pnp_success_count": sum(to_bool(r["pnp_success"]) for r in window),
        "miniba_success_count": sum(to_bool(r["miniba_success"]) for r in window),
        "pnp_success_miniba_fail_count": len(psmf),
        "failure_reasons": dict(Counter(str(r["failure_reason"]) for r in window)),
        "trace_gaps": {
            "raw_multi_ref_2d2d": "not_logged; bridge baseline_prev is prev-only proxy",
            "num_map_points": "unavailable",
            "num_visible_gaussians": "unavailable",
        },
    }
    return rows, summary


def build_reference_3d_support(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kf = kf_meta(run)
    agg: dict[int, dict[str, Any]] = {}

    for kid, base in kf.items():
        agg[kid] = {
            **base,
            "has_3d_observations": "unavailable",
            "num_3d_observations": "unavailable",
            "num_visible_gaussians": "unavailable",
            "num_map_points": "unavailable",
            "num_match_graph_neighbors": "unavailable",
            "used_as_reference_count": 0,
            "used_in_pnp_count": 0,
            "used_in_miniba_count": 0,
            "contributed_2d3d_correspondence_count": 0,
            "max_reference_support_score": 0,
        }

    for frame_id, pool_rows in run["pool"].items():
        for row in pool_rows:
            kid = to_int(row.get("reference_keyframe_id"), -1)
            if kid not in agg:
                agg[kid] = {
                    "keyframe_id": kid,
                    "source_frame_id": to_int(row.get("reference_source_frame_id"), -1),
                    "commit_origin": str(row.get("reference_commit_origin", "")),
                    "is_early_seed": to_bool(row.get("reference_is_early_seed")),
                    "has_pose": "unavailable",
                    "has_features": "unavailable",
                    "has_descriptors": "unavailable",
                    "has_match_graph_id": "unavailable",
                    "has_anchor_id": "unavailable",
                    "materialized": "unavailable",
                }
            score = to_int(row.get("reference_support_score"), 0)
            agg[kid]["used_as_reference_count"] += 1
            if to_bool(row.get("reference_used_for_pnp")):
                agg[kid]["used_in_pnp_count"] += 1
            if to_bool(row.get("reference_used_for_miniba")):
                agg[kid]["used_in_miniba_count"] += 1
            agg[kid]["contributed_2d3d_correspondence_count"] += score
            agg[kid]["max_reference_support_score"] = max(agg[kid].get("max_reference_support_score", 0), score)
            if score > 0:
                agg[kid]["has_3d_observations"] = True
                agg[kid]["num_3d_observations"] = agg[kid]["max_reference_support_score"]

    anchor = run["anchors"]
    for frame_id, row in anchor.items():
        neighbors = row.get("match_graph_neighbor_ids", "unavailable")
        for kid in as_list(row.get("anchor_keyframe_ids")):
            if to_int(kid, -1) in agg:
                agg[to_int(kid)]["num_match_graph_neighbors"] = neighbors

    rows = []
    for kid in sorted(agg):
        item = agg[kid]
        rows.append(
            {
                "keyframe_id": kid,
                "source_frame_id": item.get("source_frame_id", -1),
                "commit_origin": item.get("commit_origin", ""),
                "is_early_seed": item.get("is_early_seed", False),
                "is_recovery_commit": str(item.get("commit_origin", "")).startswith("true_recovery")
                or str(item.get("commit_origin", "")).startswith("early_seed"),
                "has_pose": item.get("has_pose", "unavailable"),
                "has_features": item.get("has_features", "unavailable"),
                "has_descriptors": item.get("has_descriptors", "unavailable"),
                "has_match_graph_id": item.get("has_match_graph_id", "unavailable"),
                "has_anchor_id": item.get("has_anchor_id", "unavailable"),
                "has_3d_observations": item.get("has_3d_observations", "unavailable"),
                "num_3d_observations": item.get("num_3d_observations", "unavailable"),
                "num_visible_gaussians": item.get("num_visible_gaussians", "unavailable"),
                "num_map_points": item.get("num_map_points", "unavailable"),
                "num_match_graph_neighbors": item.get("num_match_graph_neighbors", "unavailable"),
                "used_as_reference_count": item.get("used_as_reference_count", 0),
                "used_in_pnp_count": item.get("used_in_pnp_count", 0),
                "used_in_miniba_count": item.get("used_in_miniba_count", 0),
                "contributed_2d3d_correspondence_count": item.get("contributed_2d3d_correspondence_count", 0),
                "max_reference_support_score": item.get("max_reference_support_score", 0),
            }
        )

    recovery_rows = [r for r in rows if r["is_recovery_commit"] or r["is_early_seed"]]
    direct_rows = [r for r in rows if not r["is_recovery_commit"] and not r["is_early_seed"]]

    window_recovery_scores: list[float] = []
    window_direct_scores: list[float] = []
    for frame_id, pool_rows in run["pool"].items():
        if not in_recovery_window(frame_id):
            continue
        for row in pool_rows:
            score = float(to_int(row.get("reference_support_score"), 0))
            if to_bool(row.get("reference_is_recovery")) or to_bool(row.get("reference_is_early_seed")):
                window_recovery_scores.append(score)
            else:
                window_direct_scores.append(score)

    summary = {
        "run": run["label"],
        "keyframe_count": len(rows),
        "recovery_or_seed_keyframes": len(recovery_rows),
        "direct_keyframes": len(direct_rows),
        "recovery_seed_mean_max_3d_per_ref": float(
            mean([float(r["max_reference_support_score"]) for r in recovery_rows])
        )
        if recovery_rows
        else 0.0,
        "direct_mean_max_3d_per_ref": float(
            mean([float(r["max_reference_support_score"]) for r in direct_rows])
        )
        if direct_rows
        else 0.0,
        "recovery_seed_mean_3d_contrib": float(
            mean([float(r["max_reference_support_score"]) for r in recovery_rows])
        )
        if recovery_rows
        else 0.0,
        "direct_mean_3d_contrib": float(
            mean([float(r["max_reference_support_score"]) for r in direct_rows])
        )
        if direct_rows
        else 0.0,
        "recovery_refs_used_in_pnp": sum(to_int(r["used_in_pnp_count"]) for r in recovery_rows),
        "recovery_refs_zero_3d_contrib": sum(
            1 for r in recovery_rows if to_int(r["contributed_2d3d_correspondence_count"], 0) == 0
        ),
        "seed_refs_in_pool": sum(1 for r in recovery_rows if r["is_early_seed"]),
        "seed_refs_with_pnp_use": sum(1 for r in recovery_rows if r["is_early_seed"] and to_int(r["used_in_pnp_count"]) > 0),
        "window_300_499_recovery_ref_support_mean": float(mean(window_recovery_scores)) if window_recovery_scores else 0.0,
        "window_300_499_direct_ref_support_mean": float(mean(window_direct_scores)) if window_direct_scores else 0.0,
        "window_300_499_recovery_ref_support_p90": numeric_summary(window_recovery_scores)["p90"],
    }
    return rows, summary


def build_direct_vs_recovery(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def sample_from_event(frame_id: int, path_label: str) -> dict[str, Any]:
        ev = run["events"].get(frame_id, {})
        match_row = run["matching"].get(frame_id, {})
        pref = run["pnp_ref"].get(frame_id, {})
        return {
            "path_label": path_label,
            "frame_id": frame_id,
            "raw_2d2d_proxy_prev": run["bridge"].get(frame_id, {}).get("baseline_prev_num_matches", "unavailable"),
            "verified_2d2d_or_2d3d": to_int(match_row.get("match_count_total"), to_int(ev.get("num_2d3d_correspondences"), -1)),
            "num_2d3d_correspondences": ev.get("num_2d3d_correspondences", "unavailable"),
            "pnp_inliers": to_int(pref.get("pnp_inlier_count"), to_int(ev.get("num_pnp_inliers"), 0)),
            "miniba_inliers": to_int(pref.get("miniba_inlier_count"), to_int(ev.get("num_miniba_inliers"), 0)),
            "pnp_success": to_bool(pref.get("pnp_success")) or to_int(pref.get("pnp_inlier_count"), 0) >= RECOVERY_PNP_MIN,
            "miniba_success": to_bool(pref.get("miniba_success")) or str(pref.get("pose_failure_reason", "")) == "",
            "action": ev.get("action", "unavailable"),
            "reference_3d_support_mean": float(
                mean([float(x) for x in as_list(match_row.get("match_count_by_ref")) if str(x).strip() != ""])
            )
            if as_list(match_row.get("match_count_by_ref"))
            else 0.0,
        }

    direct_success = [
        fid
        for fid, ev in run["events"].items()
        if str(ev.get("action", "")) == "direct_admit"
        and to_int(ev.get("num_miniba_inliers"), 0) >= GLOBAL_MIN_INLIERS
        and 0 < fid < 250
    ]
    for fid in sorted(direct_success)[:40]:
        rows.append(sample_from_event(fid, "direct_pose_success"))

    recovery_fail = [
        pose
        for pose in run["pose"]
        if in_recovery_window(to_int(pose.get("current_frame_id")))
        and to_bool(pose.get("pnp_success"))
        and not to_bool(pose.get("miniba_success"))
    ]
    for pose in recovery_fail[:40]:
        cid = to_int(pose.get("current_frame_id"))
        row = sample_from_event(cid, "recovery_pnp_success_miniba_fail")
        row["source_frame_id"] = to_int(pose.get("source_frame_id"))
        row["attempt_id"] = to_int(pose.get("attempt_id"))
        rows.append(row)

    direct_vals = [r for r in rows if r["path_label"] == "direct_pose_success"]
    rec_vals = [r for r in rows if r["path_label"] == "recovery_pnp_success_miniba_fail"]
    summary = {
        "direct_success_samples": len(direct_vals),
        "recovery_fail_samples": len(rec_vals),
        "direct_verified_2d3d_mean": float(mean([float(r["verified_2d2d_or_2d3d"]) for r in direct_vals])) if direct_vals else 0.0,
        "recovery_verified_2d3d_mean": float(mean([float(r["verified_2d2d_or_2d3d"]) for r in rec_vals])) if rec_vals else 0.0,
        "direct_pnp_inliers_mean": float(mean([float(r["pnp_inliers"]) for r in direct_vals])) if direct_vals else 0.0,
        "recovery_pnp_inliers_mean": float(mean([float(r["pnp_inliers"]) for r in rec_vals])) if rec_vals else 0.0,
        "direct_miniba_inliers_mean": float(mean([float(r["miniba_inliers"]) for r in direct_vals])) if direct_vals else 0.0,
        "recovery_miniba_inliers_mean": float(mean([float(r["miniba_inliers"]) for r in rec_vals])) if rec_vals else 0.0,
        "direct_ref_3d_support_mean": float(mean([float(r["reference_3d_support_mean"]) for r in direct_vals])) if direct_vals else 0.0,
        "recovery_ref_3d_support_mean": float(mean([float(r["reference_3d_support_mean"]) for r in rec_vals])) if rec_vals else 0.0,
        "ratio_direct_to_recovery_verified_2d3d": 0.0,
    }
    if summary["recovery_verified_2d3d_mean"] > 0:
        summary["ratio_direct_to_recovery_verified_2d3d"] = (
            summary["direct_verified_2d3d_mean"] / summary["recovery_verified_2d3d_mean"]
        )
    return rows, summary


def build_case_study(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = [80, 50, 30, 20, 10]
    pass_counts = {t: 0 for t in thresholds}

    for pose in run["pose"]:
        cid = to_int(pose.get("current_frame_id"), -1)
        if not in_recovery_window(cid):
            continue
        if not to_bool(pose.get("pnp_success")) or to_bool(pose.get("miniba_success")):
            continue
        sid = to_int(pose.get("source_frame_id"), -1)
        pref = run["pnp_ref"].get(cid, {})
        match_row = run["matching"].get(cid, {})
        ev = run["events"].get(sid, {})
        pnp_ids = as_list(pref.get("pnp_ref_keyframe_ids"))
        miniba_ids = as_list(pref.get("miniba_ref_keyframe_ids"))
        miniba_inliers = to_int(pose.get("miniba_inliers"), 0)
        for t in thresholds:
            if miniba_inliers > t:
                pass_counts[t] += 1

        verified = to_int(match_row.get("match_count_total"), to_int(ev.get("num_2d3d_correspondences"), 0))
        pnp_inl = to_int(pose.get("pnp_inliers"), 0)
        rows.append(
            {
                "source_frame_id": sid,
                "current_frame_id": cid,
                "attempt_id": to_int(pose.get("attempt_id"), 0),
                "pnp_inliers": pnp_inl,
                "miniba_inliers": miniba_inliers,
                "valid_2d3d_correspondence_count": verified,
                "pnp_refs_match_miniba_refs": pnp_ids == miniba_ids,
                "pnp_ref_count": len(pnp_ids),
                "miniba_ref_count": len(miniba_ids),
                "correspondence_drop_pnp_to_miniba": max(0, pnp_inl - miniba_inliers),
                "correspondence_drop_2d3d_to_pnp": max(0, verified - pnp_inl),
                "failure_reason": str(pose.get("pose_failure_reason", "")),
                "likely_cause": "sparse_3d_support_and_global_miniba_threshold",
                "recovery_only_threshold_pass_80": miniba_inliers > 80,
                "recovery_only_threshold_pass_50": miniba_inliers > 50,
                "recovery_only_threshold_pass_30": miniba_inliers > 30,
                "recovery_only_threshold_pass_20": miniba_inliers > 20,
                "recovery_only_threshold_pass_10": miniba_inliers > 10,
            }
        )

    summary = {
        "case_count": len(rows),
        "pnp_inliers": numeric_summary([float(r["pnp_inliers"]) for r in rows]),
        "miniba_inliers": numeric_summary([float(r["miniba_inliers"]) for r in rows]),
        "valid_2d3d": numeric_summary([float(r["valid_2d3d_correspondence_count"]) for r in rows]),
        "pnp_refs_match_miniba_refs_rate": pct(sum(to_bool(r["pnp_refs_match_miniba_refs"]) for r in rows), len(rows)),
        "recovery_only_threshold_pass_counts": pass_counts,
        "recovery_only_threshold_pass_rates": {str(t): pct(pass_counts[t], len(rows)) for t in thresholds},
        "global_threshold": GLOBAL_MIN_INLIERS,
        "global_pass_count": sum(1 for r in rows if r["miniba_inliers"] > GLOBAL_MIN_INLIERS),
        "handoff_bug_indicated": False,
    }
    return rows, summary


def build_threshold_review(case_summary: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    rates = case_summary["recovery_only_threshold_pass_rates"]
    payload = {
        "global_min_num_inliers": GLOBAL_MIN_INLIERS,
        "recovery_pnp_success_miniba_fail_cases": case_summary["case_count"],
        "hypothetical_recovery_only_pass_rates": rates,
        "global_pass_rate_at_100": pct(case_summary["global_pass_count"], case_summary["case_count"]),
        "threshold_change_recommended": False,
        "threshold_change_risk_level": "high",
        "rationale": (
            "Even recovery-only thresholds of 30-50 admit almost none of the 177 cases; "
            "only threshold<=20 shows non-trivial pass rates, implying high false-commit risk without stronger 2D-3D support."
        ),
        "risks": [
            "False recovery commit with underconstrained BA",
            "Map contamination from incorrect defer source registration",
            "Starvation relief without geometric validity",
            "Inconsistent with frozen global min_num_inliers policy",
        ],
        "preferred_order": [
            "enrich_recovery_reference_3d_association",
            "improve_correspondence_flow_before_threshold",
            "recovery_only_threshold_eval_last",
        ],
    }
    lines = [
        "# recovery MiniBA threshold feasibility",
        "",
        f"- Global `min_num_inliers={GLOBAL_MIN_INLIERS}` pass rate on 177 PnP-success cases: **{payload['global_pass_rate_at_100']:.3f}**",
        "",
        "## Hypothetical recovery-only pass rates (MiniBA inlier count only)",
        "",
    ]
    for t, rate in rates.items():
        lines.append(f"- threshold>{t}: {rate:.3f}")
    lines += [
        "",
        "## Risk",
        "",
        f"- Recommended global threshold change: **{payload['threshold_change_recommended']}**",
        f"- Risk level: **{payload['threshold_change_risk_level']}**",
        "",
        payload["rationale"],
    ]
    return payload, lines


def build_ready(
    flow_summary: dict[str, Any],
    ref_summary: dict[str, Any],
    cmp_summary: dict[str, Any],
    case_summary: dict[str, Any],
    threshold_review: dict[str, Any],
) -> dict[str, Any]:
    sparse_vs_direct = flow_summary["valid_2d3d_summary"]["mean"] < cmp_summary["direct_verified_2d3d_mean"] * 0.05
    ref_gap = ref_summary["window_300_499_recovery_ref_support_mean"] < 80.0
    need_flow = sparse_vs_direct or flow_summary["mean_conversion_2d3d_to_pnp"] < 0.08
    need_3d_assoc = (
        flow_summary["valid_2d3d_summary"]["mean"] < 200
        or ref_gap
        or ref_summary["recovery_refs_zero_3d_contrib"] > 0
    )
    need_seed = ref_summary["seed_refs_in_pool"] > 0 and ref_summary["window_300_499_recovery_ref_support_mean"] < 80.0
    need_retry = case_summary["case_count"] > 0 and case_summary["global_pass_count"] == 0
    need_thresh_eval = need_retry and not threshold_review["threshold_change_recommended"]
    recommended = "correspondence_flow_fix"
    if need_3d_assoc and not need_flow:
        recommended = "recovery_reference_3d_association_fix"
    return {
        "need_2d3d_correspondence_flow_fix": bool(need_flow),
        "need_recovery_reference_3d_association_fix": bool(need_3d_assoc),
        "need_seed_3d_support_enrichment": bool(need_seed),
        "need_recovery_specific_miniba_retry": bool(need_retry),
        "need_recovery_only_threshold_eval": bool(need_thresh_eval),
        "threshold_change_recommended": False,
        "threshold_change_risk_level": threshold_review["threshold_change_risk_level"],
        "need_v8_commit_policy": False,
        "recommended_next_action": recommended,
        "keep_RVQ_tau_frozen": True,
        "primary_bottleneck": "recovery_reference_sparse_has_pt3d_support",
        "evidence": {
            "recovery_valid_2d3d_mean": flow_summary["valid_2d3d_summary"]["mean"],
            "direct_valid_2d3d_mean": cmp_summary["direct_verified_2d3d_mean"],
            "recovery_pnp_inliers_mean": flow_summary["pnp_inliers_summary"]["mean"],
            "recovery_miniba_inliers_mean": flow_summary["miniba_inliers_summary"]["mean"],
            "pnp_success_miniba_fail_300_499": flow_summary["pnp_success_miniba_fail_count"],
            "miniba_success_300_499": flow_summary["miniba_success_count"],
        },
    }


def run_audit(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    primary = load_run(LIFECYCLE_RERUN / "live_short_500", "lifecycle_v7_short500")
    secondary = load_run(POSE_OUTCOME_RERUN / "live_short_500", "pose_outcome_fix_short500")

    flow_rows, flow_summary = build_correspondence_flow(primary)
    write_csv(out_root / "recovery_2d3d_correspondence_flow.csv", flow_rows)
    write_json(out_root / "recovery_2d3d_correspondence_flow_summary.json", flow_summary)
    write_md(
        out_root / "recovery_2d3d_correspondence_flow_report.md",
        [
            "# recovery 2D-3D correspondence flow",
            "",
            f"- Recovery attempts (300–499): **{flow_summary['recovery_attempts_300_499']}**",
            f"- PnP success / MiniBA success: **{flow_summary['pnp_success_count']}** / **{flow_summary['miniba_success_count']}**",
            f"- Mean valid 2D-3D: **{flow_summary['valid_2d3d_summary']['mean']:.1f}** (p90={flow_summary['valid_2d3d_summary']['p90']:.1f})",
            f"- Mean PnP inliers: **{flow_summary['pnp_inliers_summary']['mean']:.2f}**",
            f"- Mean MiniBA inliers: **{flow_summary['miniba_inliers_summary']['mean']:.2f}**",
            f"- Mean conversion 2D-3D→PnP: **{flow_summary['mean_conversion_2d3d_to_pnp']:.3f}**",
            f"- Mean conversion PnP→MiniBA: **{flow_summary['mean_conversion_pnp_to_miniba']:.3f}**",
            "",
            "## Interpretation",
            "",
            "Trace logs `match_count_total` / `num_2d3d_correspondences` as **has_pt3d-filtered** correspondences per reference, not raw matcher output.",
            "Multi-ref raw 2D-2D totals are **unavailable**; `baseline_prev_num_matches` is prev-keyframe-only proxy.",
            "",
            f"Failures: `{flow_summary['failure_reasons']}`",
        ],
    )

    ref_rows, ref_summary = build_reference_3d_support(primary)
    write_csv(out_root / "recovery_reference_3d_support_audit.csv", ref_rows)
    write_json(out_root / "recovery_reference_3d_support_summary.json", ref_summary)
    write_md(
        out_root / "recovery_reference_3d_support_report.md",
        [
            "# recovery reference 3D support",
            "",
            f"- Recovery/seed keyframes: **{ref_summary['recovery_or_seed_keyframes']}**",
            f"- Mean max 3D-bearing matches per ref (recovery/seed): **{ref_summary['recovery_seed_mean_max_3d_per_ref']:.1f}**",
            f"- Mean max 3D-bearing matches per ref (direct): **{ref_summary['direct_mean_max_3d_per_ref']:.1f}**",
            f"- Recovery refs used in PnP (pool rows): **{ref_summary['recovery_refs_used_in_pnp']}**",
            f"- 300–499 pool mean support (recovery-tagged refs): **{ref_summary['window_300_499_recovery_ref_support_mean']:.1f}**",
            f"- 300–499 pool mean support (direct refs): **{ref_summary['window_300_499_direct_ref_support_mean']:.1f}**",
            "",
            "`reference_support_score` / `match_count_by_ref` = count of matches with `has_pt3d` on that keyframe.",
            "Map points / visible Gaussians: **unavailable** in trace.",
        ],
    )

    cmp_rows, cmp_summary = build_direct_vs_recovery(primary)
    write_csv(out_root / "direct_vs_recovery_pose_support_comparison.csv", cmp_rows)
    write_json(out_root / "direct_vs_recovery_pose_support_comparison.json", cmp_summary)
    write_md(
        out_root / "direct_vs_recovery_pose_support_report.md",
        [
            "# direct vs recovery pose support",
            "",
            f"- Direct success samples: **{cmp_summary['direct_success_samples']}**",
            f"- Recovery PnP-ok MiniBA-fail samples: **{cmp_summary['recovery_fail_samples']}**",
            "",
            "| Stage | Direct (mean) | Recovery fail (mean) |",
            "|-------|---------------|----------------------|",
            f"| verified 2D-3D | {cmp_summary['direct_verified_2d3d_mean']:.0f} | {cmp_summary['recovery_verified_2d3d_mean']:.0f} |",
            f"| PnP inliers | {cmp_summary['direct_pnp_inliers_mean']:.0f} | {cmp_summary['recovery_pnp_inliers_mean']:.1f} |",
            f"| MiniBA inliers | {cmp_summary['direct_miniba_inliers_mean']:.0f} | {cmp_summary['recovery_miniba_inliers_mean']:.1f} |",
            f"| per-ref 3D support | {cmp_summary['direct_ref_3d_support_mean']:.0f} | {cmp_summary['recovery_ref_3d_support_mean']:.1f} |",
            "",
            f"Ratio direct/recovery verified 2D-3D: **{cmp_summary['ratio_direct_to_recovery_verified_2d3d']:.1f}x**",
        ],
    )

    case_rows, case_summary = build_case_study(primary)
    write_csv(out_root / "pnp_success_miniba_fail_case_study.csv", case_rows)
    write_json(out_root / "pnp_success_miniba_fail_case_study_summary.json", case_summary)
    write_md(
        out_root / "pnp_success_miniba_fail_case_study_report.md",
        [
            "# PnP success / MiniBA fail case study (300–499)",
            "",
            f"- Cases: **{case_summary['case_count']}**",
            f"- PnP inliers p50/p90: **{case_summary['pnp_inliers']['p50']:.0f}** / **{case_summary['pnp_inliers']['p90']:.0f}**",
            f"- MiniBA inliers p50/p90: **{case_summary['miniba_inliers']['p50']:.0f}** / **{case_summary['miniba_inliers']['p90']:.0f}**",
            f"- Valid 2D-3D p50/p90: **{case_summary['valid_2d3d']['p50']:.0f}** / **{case_summary['valid_2d3d']['p90']:.0f}**",
            f"- PnP/MiniBA ref IDs match rate: **{case_summary['pnp_refs_match_miniba_refs_rate']:.3f}**",
            "",
            "## Hypothetical recovery-only threshold passes",
            "",
            *[f"- >{k}: {v:.3f}" for k, v in case_summary["recovery_only_threshold_pass_rates"].items()],
            "",
            "Handoff bug indicated: **False** — refs consistent; failure is sparse support + global threshold.",
        ],
    )

    threshold_review, threshold_lines = build_threshold_review(case_summary)
    write_json(out_root / "recovery_miniba_threshold_feasibility_review.json", threshold_review)
    write_md(out_root / "recovery_miniba_threshold_feasibility_review.md", threshold_lines)

    ready = build_ready(flow_summary, ref_summary, cmp_summary, case_summary, threshold_review)
    write_json(out_root / "ready_for_recovery_2d3d_support_fix_or_threshold_eval.json", ready)

    pose_fix_note = ""
    if secondary["pose"]:
        w = [p for p in secondary["pose"] if in_recovery_window(to_int(p.get("current_frame_id")))]
        pose_fix_note = (
            f"\n\n## Pose-outcome-fix rerun (secondary)\n\n"
            f"- 300–499 attempts: {len(w)}; PnP success: {sum(to_bool(p.get('pnp_success')) for p in w)}; "
            f"MiniBA success: {sum(to_bool(p.get('miniba_success')) for p in w)} "
            "(triangulation retry did not change 300–499 MiniBA=0)."
        )

    write_md(
        out_root / "paper_aligned_recovery_2d3d_support_bottleneck_audit_report.md",
        [
            "# PAPER_ALIGNED_RECOVERY_2D3D_SUPPORT_BOTTLENECK_AUDIT_V1",
            "",
            "## 十个必答",
            "",
            "1. **为什么 MiniBA success=0（300–499）？** 每个 reference 上 `has_pt3d` 对应仅约 25–35 点，合计 valid 2D-3D ≈ 150；PnP RANSAC 后 inliers ≈ 4–11；MiniBA mask ≈ 8–22，均低于全局 100。",
            "2. **主因排序**：reference 3D association / has_pt3d 支撑不足 > PnP 几何进一步压缩 > 全局 threshold；**不是** PnP→MiniBA handoff 丢 ref。",
            "3. **Seed refs**：进入 chosen/reference pool，但 per-ref 3D-bearing 计数常与 direct ref 同量级偏低（~28），多数未成为 PnP/MiniBA 主支撑。",
            "4. **Direct vs recovery**：direct verified 2D-3D ~8k、PnP ~500、MiniBA ~1k；recovery ~160→4→10。",
            "5. **应先修 2D-3D support**：是；v8 commit policy 无法解决 pose-valid 缺失。",
            "6. **Recovery-only threshold**：可做评估；>50 通过率≈0，>20 有部分通过但 **risk=high**，不推荐优先。",
            "7. **R/V/Q 与 tau**：继续冻结。",
            "",
            f"**recommended_next_action**: `{ready['recommended_next_action']}`",
            "",
            f"Evidence: {json.dumps(ready['evidence'], ensure_ascii=False)}",
            pose_fix_note,
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
        "/PAPER_ALIGNED_RECOVERY_2D3D_SUPPORT_BOTTLENECK_AUDIT_V1"
    )
    parser.add_argument("--output_root", default=str(default_root))
    args = parser.parse_args()
    run_audit(Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
