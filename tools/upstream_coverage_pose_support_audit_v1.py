#!/usr/bin/env python3
"""PAPER_ALIGNED_UPSTREAM_COVERAGE_POSE_SUPPORT_AUDIT_V1 — read-only upstream audit."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

OUT_ROOT = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_1_GAP_FIX_V1"
)
LIFECYCLE_CSV = Path(
    "/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1"
    "/lifecycle/all_input_frame_lifecycle.csv"
)

RUNS: dict[str, dict[str, Any]] = {
    "short800": {
        "run_dir": OUT_ROOT / "direct_density_v2_2_2_1_short800",
        "max_frame": 800,
    },
    "short1000": {
        "run_dir": OUT_ROOT / "direct_density_v2_2_2_1_short1000",
        "max_frame": 1000,
    },
}

FRAME_EXPAND_COLUMNS = [
    "run",
    "interval_type",
    "gap_index",
    "gap_start_frame",
    "gap_end_frame",
    "gap_length",
    "post500_gap",
    "gap_attribution",
    "frame_id",
    "is_registered",
    "pose_attempted",
    "pose_success",
    "direct_admit_candidate",
    "defer_recoverable",
    "discard",
    "recovery_pool_entered",
    "recovery_attempted",
    "recovery_success",
    "finalization_attempted",
    "finalized_keyframe",
    "risk_R",
    "value_V",
    "recoverability_Q",
    "action",
    "decision_reason",
    "baseline_should_add",
    "baseline_like_evidence",
    "anchor_id",
    "reference_count",
    "candidate_reference_count",
    "selected_reference_ids",
    "match_count",
    "valid_2d3d_count",
    "pnp_attempted",
    "pnp_success",
    "pnp_fail_reason",
    "pnp_inliers",
    "miniba_attempted",
    "miniba_success",
    "miniba_fail_reason",
    "miniba_inliers",
    "too_few_inliers",
    "hold_reason",
    "upstream_cause_hypothesis",
    "near_anchor_switch",
    "psnr",
    "ssim",
    "lpips",
    "abs_rot_error_deg",
]


def load_gap_module():
    path = Path(__file__).resolve().parent / "gap_source_attribution_audit_v1.py"
    spec = importlib.util.spec_from_file_location("gap_attr", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_csv_rows(p: Path) -> list[dict[str, str]]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = columns or []
    seen: set[str] = set()
    for c in fields:
        seen.add(c)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def to_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def pct(vals: list[int], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(round((len(s) - 1) * q))
    return float(s[max(0, min(len(s) - 1, i))])


def load_lifecycle_by_frame(max_frame: int) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if not LIFECYCLE_CSV.exists():
        return out
    with LIFECYCLE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fid = to_int(row.get("frame_id"))
            if fid <= max_frame:
                out[fid] = row
    return out


class TraceIndex:
    def __init__(self, run_dir: Path, max_frame: int) -> None:
        self.run_dir = run_dir
        self.max_frame = max_frame
        self.trace = self._load_trace()
        self.events_by = {
            int(e["frame_id"]): e for e in self.trace.get("events", []) or []
        }
        self.lifecycle_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in self.trace.get("lifecycle_gate_events", []) or []:
            fid = to_int(e.get("frame_id"))
            if fid <= max_frame:
                self.lifecycle_by[fid].append(e)
        self.pnp_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in self.trace.get("pnp_miniba_reference_events", []) or []:
            fid = to_int(e.get("frame_id"))
            if fid <= max_frame:
                self.pnp_by[fid].append(e)
        self.anchor_by: dict[int, dict[str, Any]] = {}
        for e in self.trace.get("local_map_anchor_events", []) or []:
            fid = to_int(e.get("frame_id"))
            if fid <= max_frame:
                self.anchor_by[fid] = e
        self.chosen_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in self.trace.get("chosen_kfs_reference_events", []) or []:
            fid = to_int(e.get("frame_id"))
            if fid <= max_frame:
                self.chosen_by[fid].append(e)
        self.match_by: dict[int, dict[str, Any]] = {}
        for e in self.trace.get("matching_support_events", []) or []:
            fid = to_int(e.get("frame_id"))
            if fid <= max_frame:
                self.match_by[fid] = e
        self.support_by: dict[int, dict[str, Any]] = {}
        for e in self.trace.get("support_integration_events", []) or []:
            if str(e.get("event_type", "")) != "support_candidate_matching":
                continue
            fid = to_int(e.get("frame_id"))
            if fid <= max_frame:
                self.support_by[fid] = e
        self.recovery_pose_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in self.trace.get("recovery_pose_path_events", []) or []:
            fid = to_int(e.get("current_frame_id", e.get("frame_id")))
            if fid <= max_frame:
                self.recovery_pose_by[fid].append(e)
        self.recovery_2d3d_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in self.trace.get("recovery_2d3d_support_events", []) or []:
            fid = to_int(e.get("current_frame_id"))
            if fid <= max_frame:
                self.recovery_2d3d_by[fid].append(e)
        self.ddc_by: dict[int, dict[str, str]] = {}
        for row in read_csv_rows(run_dir / "direct_density_control_v2_2_2_1_trace.csv"):
            fid = to_int(row.get("frame_id"))
            if fid <= max_frame:
                self.ddc_by[fid] = row
        if not self.ddc_by:
            for e in self.trace.get("direct_density_control_v2_2_2_1_events", []) or []:
                fid = to_int(e.get("frame_id"))
                if fid <= max_frame:
                    self.ddc_by[fid] = {k: str(v) for k, v in e.items()}
        self.lifecycle_csv = load_lifecycle_by_frame(max_frame)
        self.anchor_switch_frames = self._anchor_switch_frames()

    def _load_trace(self) -> dict[str, Any]:
        for rel in ("model/semantic_trace.json", "semantic_trace.json"):
            p = self.run_dir / rel
            if p.exists():
                return read_json(p)
        return {}

    def _anchor_switch_frames(self) -> set[int]:
        prev: int | None = None
        switches: set[int] = set()
        for fid in sorted(self.anchor_by):
            aid = to_int(self.anchor_by[fid].get("anchor_id"), -1)
            if prev is not None and aid != prev:
                switches.add(fid)
                switches.add(fid - 1)
            prev = aid
        return switches

    def expand_frame(
        self,
        run: str,
        interval_type: str,
        gap_row: dict[str, Any],
        frame_id: int,
    ) -> dict[str, Any]:
        ev = self.events_by.get(frame_id, {})
        lc_list = self.lifecycle_by.get(frame_id, [])
        lc = lc_list[-1] if lc_list else {}
        lcc = self.lifecycle_csv.get(frame_id, {})
        pnp_list = self.pnp_by.get(frame_id, [])
        pnp = pnp_list[-1] if pnp_list else {}
        ddc = self.ddc_by.get(frame_id, {})
        anchor = self.anchor_by.get(frame_id, {})
        chosen = self.chosen_by.get(frame_id, [])
        match = self.match_by.get(frame_id, {})
        support = self.support_by.get(frame_id, {})
        r2d = self.recovery_2d3d_by.get(frame_id, [])
        r2d_last = r2d[-1] if r2d else {}

        meta = ev.get("decision_meta") or {}
        action = str(ev.get("action", ""))
        pnp_reason = str(pnp.get("pose_failure_reason", ""))
        miniba_ok = to_bool(pnp.get("miniba_success")) if pnp else None
        pnp_ok = to_bool(pnp.get("pnp_success")) if pnp else None

        ref_ids: list[str] = []
        cand_ref_count = 0
        if chosen:
            last = chosen[-1]
            ref_ids = [str(x) for x in (last.get("reference_source_frame_ids") or [])]
            if isinstance(ref_ids, str):
                ref_ids = [ref_ids]
            cand_ref_count = len(last.get("chosen_kfs_source_frame_ids") or [])

        pool = self.trace.get("pose_reference_pool_events", []) or []
        ref_count = sum(1 for r in pool if to_int(r.get("frame_id")) == frame_id)

        hold_reason = ""
        if ddc:
            dec = str(ddc.get("direct_finalization_decision", ""))
            if dec and not to_bool(ddc.get("direct_keyframe_finalized")):
                hold_reason = dec

        row = {
            "run": run,
            "interval_type": interval_type,
            "gap_index": gap_row.get("gap_index"),
            "gap_start_frame": gap_row.get("gap_start_frame"),
            "gap_end_frame": gap_row.get("gap_end_frame"),
            "gap_length": gap_row.get("gap_length"),
            "post500_gap": gap_row.get("post500_gap"),
            "gap_attribution": gap_row.get("gap_attribution"),
            "frame_id": frame_id,
            "is_registered": lcc.get("is_registered") or lc.get("materialized"),
            "pose_attempted": ev.get("pose_init_attempted") or lc.get("pose_path_requested"),
            "pose_success": ev.get("pose_init_success"),
            "direct_admit_candidate": action == "direct_admit"
            or to_bool(ddc.get("direct_admit_candidate")),
            "defer_recoverable": action == "defer_recoverable",
            "discard": action == "discard",
            "recovery_pool_entered": any(to_bool(x.get("in_recovery_pool")) for x in lc_list)
            or action == "defer_recoverable",
            "recovery_attempted": any(to_bool(x.get("recovery_pose_attempted")) for x in lc_list),
            "recovery_success": any(to_bool(x.get("recovery_success")) for x in lc_list),
            "pose_path_allowed": lc.get("pose_path_allowed"),
            "pose_path_requested": lc.get("pose_path_requested"),
            "finalization_attempted": bool(ddc),
            "finalized_keyframe": to_bool(ev.get("final_keyframe_incremented")),
            "risk_R": meta.get("R_t"),
            "value_V": meta.get("V_t"),
            "recoverability_Q": meta.get("Q_t"),
            "action": action,
            "decision_reason": "; ".join(
                x
                for x in [
                    str(ev.get("drop_reason", "")),
                    str(ddc.get("direct_finalization_reason", "")),
                    str(lc.get("pose_path_block_reason", "")),
                ]
                if x
            ),
            "baseline_should_add": ev.get("baseline_should_add"),
            "baseline_like_evidence": support.get("baseline_prev_should_add"),
            "anchor_id": anchor.get("anchor_id"),
            "reference_count": ref_count,
            "candidate_reference_count": cand_ref_count,
            "selected_reference_ids": "|".join(ref_ids[:12]),
            "match_count": match.get("match_count_total"),
            "valid_2d3d_count": r2d_last.get("valid_2d3d_correspondence_count")
            or ev.get("num_2d3d_correspondences"),
            "pnp_attempted": bool(pnp),
            "pnp_success": pnp_ok,
            "pnp_fail_reason": pnp_reason,
            "pnp_inliers": pnp.get("pnp_inlier_count"),
            "miniba_attempted": bool(pnp),
            "miniba_success": miniba_ok,
            "miniba_fail_reason": pnp_reason if miniba_ok is False else "",
            "miniba_inliers": pnp.get("miniba_inlier_count"),
            "too_few_inliers": "inlier" in pnp_reason.lower()
            or "inlier" in str(ev.get("pose_fail_detail", "")).lower(),
            "hold_reason": hold_reason,
            "near_anchor_switch": frame_id in self.anchor_switch_frames,
            "psnr": lcc.get("psnr"),
            "ssim": lcc.get("ssim"),
            "lpips": lcc.get("lpips"),
            "abs_rot_error_deg": lcc.get("rel_rot_error_deg") or lcc.get("abs_rot_error_deg"),
        }
        row["upstream_cause_hypothesis"] = classify_upstream(row)
        return row


def classify_upstream(row: dict[str, Any]) -> str:
    action = str(row.get("action", ""))
    if not action:
        return "logging_missing_or_unknown"
    if action == "discard":
        return "R_V_Q_upstream_discard"
    if action == "defer_recoverable":
        if to_int(row.get("reference_count")) == 0 and to_int(row.get("candidate_reference_count")) == 0:
            return "anchor_reference_shortage"
        if to_bool(row.get("too_few_inliers")) or (
            to_bool(row.get("pose_attempted")) and row.get("pose_success") is False
        ):
            return "pose_support_too_weak"
        if to_bool(row.get("recovery_attempted")) and not to_bool(row.get("recovery_success")):
            return "pose_support_too_weak"
        ppa = row.get("pose_path_allowed")
        if (
            ppa is not None
            and str(ppa).strip() != ""
            and not to_bool(ppa)
            and not to_bool(row.get("recovery_attempted"))
        ):
            return "lifecycle_gate_pose_path_blocked"
        return "recovery_pool_defer_not_materialized"
    ppa = row.get("pose_path_allowed")
    if ppa is not None and str(ppa).strip() != "" and not to_bool(ppa):
        return "lifecycle_gate_pose_path_blocked"
    if action == "direct_admit":
        if row.get("hold_reason"):
            return "finalization_hold_downstream"
        if to_bool(row.get("pose_attempted")) and row.get("pose_success") is False:
            return "pose_support_too_weak"
        if not to_bool(row.get("finalized_keyframe")):
            return "direct_admit_not_finalized"
        return "direct_admit_success"
    return "mixed_or_unknown"


def interval_summary_row(gap_row: dict[str, Any], frames: list[dict[str, Any]], run: str) -> dict[str, Any]:
    causes = Counter(f["upstream_cause_hypothesis"] for f in frames)
    return {
        "run": run,
        "gap_index": gap_row["gap_index"],
        "gap_start_frame": gap_row["gap_start_frame"],
        "gap_end_frame": gap_row["gap_end_frame"],
        "gap_length": gap_row["gap_length"],
        "post500_gap": gap_row["post500_gap"],
        "gap_attribution": gap_row["gap_attribution"],
        "interior_frame_count": len(frames),
        "direct_admit_frames": sum(1 for f in frames if f["direct_admit_candidate"]),
        "defer_frames": sum(1 for f in frames if f["defer_recoverable"]),
        "discard_frames": sum(1 for f in frames if f["discard"]),
        "finalized_frames": sum(1 for f in frames if f["finalized_keyframe"]),
        "pose_fail_frames": sum(
            1
            for f in frames
            if to_bool(f.get("pose_attempted")) and f.get("pose_success") is False
        ),
        "too_few_inliers_frames": sum(1 for f in frames if f["too_few_inliers"]),
        "near_anchor_switch_frames": sum(1 for f in frames if f["near_anchor_switch"]),
        "dominant_upstream_cause": causes.most_common(1)[0][0] if causes else "unknown",
        "upstream_cause_counts": dict(causes),
    }


def build_anchor_rhythm(idx: TraceIndex, ticks: list[int], run: str) -> list[dict[str, Any]]:
    by_anchor: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "admitted_keyframes": 0,
            "direct_finalized": 0,
            "recovery_finalized": 0,
            "candidate_missing_frames": 0,
            "defer_frames": 0,
            "pnp_fail": 0,
            "miniba_fail": 0,
            "too_few_inliers": 0,
            "gap_lengths": [],
            "frames": [],
        }
    )
    events_by = idx.events_by
    for fid in ticks:
        anc = to_int(idx.anchor_by.get(fid, {}).get("anchor_id"), -1)
        ev = events_by.get(fid, {})
        by_anchor[anc]["admitted_keyframes"] += 1
        if to_bool(ev.get("final_keyframe_incremented")):
            if to_bool(ev.get("source_recovery_committed")):
                by_anchor[anc]["recovery_finalized"] += 1
            else:
                by_anchor[anc]["direct_finalized"] += 1

    # attribute non-kf frames to anchor at nearest prior kf
    prev_anchor = -1
    for fid in range(1, idx.max_frame + 1):
        if fid in idx.anchor_by:
            prev_anchor = to_int(idx.anchor_by[fid].get("anchor_id"), prev_anchor)
        anc = prev_anchor
        ev = events_by.get(fid, {})
        if not ev:
            continue
        bucket = by_anchor[anc]
        bucket["frames"].append(fid)
        if str(ev.get("action")) == "defer_recoverable":
            bucket["defer_frames"] += 1
        if str(ev.get("action")) != "direct_admit" and not to_bool(ev.get("final_keyframe_incremented")):
            bucket["candidate_missing_frames"] += 1
        pnp = idx.pnp_by.get(fid, [])
        if pnp and not to_bool(pnp[-1].get("pnp_success")):
            bucket["pnp_fail"] += 1
        if pnp and not to_bool(pnp[-1].get("miniba_success")):
            bucket["miniba_fail"] += 1
        if pnp and "inlier" in str(pnp[-1].get("pose_failure_reason", "")).lower():
            bucket["too_few_inliers"] += 1

    # gap stats per anchor (between ticks in same anchor)
    sorted_ticks = sorted(ticks)
    for i in range(1, len(sorted_ticks)):
        a = to_int(idx.anchor_by.get(sorted_ticks[i], {}).get("anchor_id"), -1)
        by_anchor[a]["gap_lengths"].append(sorted_ticks[i] - sorted_ticks[i - 1])

    rows: list[dict[str, Any]] = []
    for anc, st in sorted(by_anchor.items()):
        gaps = st["gap_lengths"]
        rows.append(
            {
                "run": run,
                "anchor_id": anc,
                "admitted_keyframes": st["admitted_keyframes"],
                "direct_finalized": st["direct_finalized"],
                "recovery_finalized": st["recovery_finalized"],
                "candidate_missing_frames": st["candidate_missing_frames"],
                "defer_frames": st["defer_frames"],
                "pnp_fail": st["pnp_fail"],
                "miniba_fail": st["miniba_fail"],
                "too_few_inliers": st["too_few_inliers"],
                "gap_p90": round(pct(gaps, 0.9), 2) if gaps else 0,
                "gap_max": max(gaps) if gaps else 0,
                "anchor_local_reference_shortage_signal": st["defer_frames"] > st["direct_finalized"] * 2,
                "anchor_transition_support_break_signal": st["too_few_inliers"] > 20 and anc >= 0,
            }
        )
    return rows


def build_pose_support_diagnostic(
    idx: TraceIndex, expand_rows: list[dict[str, Any]], run: str
) -> list[dict[str, Any]]:
    success_direct = []
    for fid, ev in idx.events_by.items():
        if (
            str(ev.get("action")) == "direct_admit"
            and ev.get("pose_init_success") is True
            and to_bool(ev.get("final_keyframe_incremented"))
        ):
            success_direct.append(
                idx.expand_frame(
                    run,
                    "success_cohort",
                    {
                        "gap_index": -1,
                        "gap_start_frame": fid,
                        "gap_end_frame": fid,
                        "gap_length": 0,
                        "post500_gap": fid > 500,
                        "gap_attribution": "success_cohort",
                    },
                    fid,
                )
            )
    fail_pose = [
        r
        for r in expand_rows
        if (r["direct_admit_candidate"] or r["defer_recoverable"])
        and to_bool(r.get("pose_attempted"))
        and r.get("pose_success") is False
    ]
    missing_gap = [r for r in expand_rows if r["interval_type"] == "candidate_missing"]

    def stats(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        mc = [to_float(r.get("match_count")) for r in rows if to_float(r.get("match_count")) is not None]
        v2 = [to_float(r.get("valid_2d3d_count")) for r in rows if to_float(r.get("valid_2d3d_count")) is not None]
        pi = [to_float(r.get("pnp_inliers")) for r in rows if to_float(r.get("pnp_inliers")) is not None]
        mi = [to_float(r.get("miniba_inliers")) for r in rows if to_float(r.get("miniba_inliers")) is not None]
        reasons = Counter(str(r.get("pnp_fail_reason", "")) for r in rows if r.get("pnp_fail_reason"))
        return {
            "run": run,
            "cohort": label,
            "frame_count": len(rows),
            "match_count_median": round(median(mc), 2) if mc else None,
            "match_count_mean": round(mean(mc), 2) if mc else None,
            "valid_2d3d_median": round(median(v2), 2) if v2 else None,
            "valid_2d3d_mean": round(mean(v2), 2) if v2 else None,
            "pnp_inliers_median": round(median(pi), 2) if pi else None,
            "miniba_inliers_median": round(median(mi), 2) if mi else None,
            "too_few_inliers_rate": round(
                sum(1 for r in rows if r["too_few_inliers"]) / max(len(rows), 1), 4
            ),
            "top_failure_reasons": dict(reasons.most_common(5)),
        }

    return [
        stats(success_direct, "success_direct_finalized"),
        stats(fail_pose, "pose_failed_with_candidate"),
        stats(missing_gap, "candidate_missing_interval_frames"),
    ]


def quality_alignment(expand_rows: list[dict[str, Any]], run: str) -> dict[str, Any]:
    test_rows = [
        r
        for r in expand_rows
        if r.get("psnr") not in (None, "", "nan")
        and to_float(r.get("psnr")) is not None
    ]
    if not test_rows:
        return {
            "run": run,
            "available": False,
            "note": "no per-frame test metrics in lifecycle CSV for expanded frames",
        }
    low_psnr = sorted(test_rows, key=lambda r: to_float(r["psnr"]) or 999)[:10]
    return {
        "run": run,
        "available": True,
        "test_metric_frame_count": len(test_rows),
        "low_psnr_samples": [
            {
                "frame_id": r["frame_id"],
                "psnr": r["psnr"],
                "gap_length": r["gap_length"],
                "upstream_cause": r["upstream_cause_hypothesis"],
                "near_candidate_missing_interval": r["interval_type"] == "candidate_missing",
            }
            for r in low_psnr
        ],
    }


def audit_run(run_key: str, spec: dict[str, Any], gap_mod: Any) -> dict[str, Any]:
    run_dir: Path = spec["run_dir"]
    max_frame: int = spec["max_frame"]
    if not run_dir.exists():
        return {"run": run_key, "error": f"missing run_dir {run_dir}"}

    gap_rows, gap_summary = gap_mod.analyze_run(run_key, run_dir, max_frame)
    idx = TraceIndex(run_dir, max_frame)
    ticks = gap_mod.keyframe_ticks(run_dir, idx.trace, max_frame)

    missing_intervals = [
        g
        for g in gap_rows
        if g["gap_attribution"] == "candidate_missing" and to_int(g["gap_interior_input_frame_count"]) > 0
    ]
    post500_missing = [g for g in missing_intervals if to_bool(g.get("post500_gap"))]
    ge11_gaps = [g for g in gap_rows if to_int(g["gap_length"]) >= 11]

    missing_expand: list[dict[str, Any]] = []
    missing_summaries: list[dict[str, Any]] = []
    for g in missing_intervals:
        interior = list(
            range(to_int(g["gap_start_frame"]) + 1, to_int(g["gap_end_frame"]))
        )
        frames = [
            idx.expand_frame(run_key, "candidate_missing", g, fid) for fid in interior
        ]
        missing_expand.extend(frames)
        missing_summaries.append(interval_summary_row(g, frames, run_key))

    ge11_expand: list[dict[str, Any]] = []
    for g in ge11_gaps:
        interior = list(
            range(to_int(g["gap_start_frame"]) + 1, to_int(g["gap_end_frame"]))
        )
        itype = "gap_ge_11"
        for fid in interior:
            ge11_expand.append(idx.expand_frame(run_key, itype, g, fid))

    anchor_rows = build_anchor_rhythm(idx, ticks, run_key)
    pose_rows = build_pose_support_diagnostic(idx, missing_expand + ge11_expand, run_key)
    quality = quality_alignment(missing_expand + ge11_expand, run_key)

    cause_all = Counter(r["upstream_cause_hypothesis"] for r in missing_expand + ge11_expand)
    cause_missing = Counter(r["upstream_cause_hypothesis"] for r in missing_expand)
    post500_cause = Counter(
        r["upstream_cause_hypothesis"]
        for r in missing_expand
        if to_bool(r.get("post500_gap"))
    )

    switch_correlation = sum(1 for r in missing_expand if r["near_anchor_switch"])
    defer_pose_fail = sum(
        1
        for r in missing_expand
        if r["defer_recoverable"] and r.get("pose_success") is False
    )

    return {
        "run": run_key,
        "run_dir": str(run_dir),
        "max_frame": max_frame,
        "gap_summary": gap_summary,
        "missing_interval_count": len(missing_intervals),
        "post500_missing_interval_count": len(post500_missing),
        "gap_ge11_count": len(ge11_gaps),
        "missing_interval_summaries": missing_summaries,
        "missing_expand_rows": missing_expand,
        "ge11_expand_rows": ge11_expand,
        "anchor_rhythm": anchor_rows,
        "pose_support": pose_rows,
        "quality_alignment": quality,
        "upstream_cause_counts_all_expanded": dict(cause_all),
        "upstream_cause_counts_candidate_missing": dict(cause_missing),
        "upstream_cause_counts_post500_missing": dict(post500_cause),
        "anchor_switch_frames_in_missing": switch_correlation,
        "defer_pose_fail_in_missing": defer_pose_fail,
    }


def final_judgment(a800: dict[str, Any], a1000: dict[str, Any]) -> dict[str, Any]:
    def pool_causes(a: dict[str, Any]) -> Counter:
        c = Counter()
        c.update(a.get("upstream_cause_counts_candidate_missing", {}))
        c.update(a.get("upstream_cause_counts_all_expanded", {}))
        return c

    c800 = pool_causes(a800)
    c1000 = pool_causes(a1000)
    merged = c800 + c1000
    dominant = merged.most_common(1)[0][0] if merged else "unknown"

    primary_map = {
        "pose_support_too_weak": "pose_support_too_weak",
        "recovery_pool_defer_not_materialized": "recovery_pool_activation_and_materialization",
        "anchor_reference_shortage": "anchor_reference_shortage",
        "lifecycle_gate_pose_path_blocked": "lifecycle_gate_pose_path_reachability",
        "R_V_Q_upstream_discard": "R_V_Q_upstream_no_direct_candidate",
        "logging_missing_or_unknown": "logging_missing_or_unknown",
        "finalization_hold_downstream": "finalization_hold_not_primary_upstream",
    }
    primary = primary_map.get(dominant, dominant)

    minimal_fix_allowed = [
        "reference_support_propagation",
        "anchor_transition_support",
        "recovery_pool_activation",
        "pose_path_reachability",
        "logging_diagnostic_correction",
    ]
    suggested: list[str] = []
    if merged.get("pose_support_too_weak", 0) > merged.get("R_V_Q_upstream_discard", 0):
        suggested.append("pose_path_reachability")
        suggested.append("reference_support_propagation")
    if merged.get("recovery_pool_defer_not_materialized", 0) > 0:
        suggested.append("recovery_pool_activation")
    if merged.get("anchor_reference_shortage", 0) > 0:
        suggested.append("anchor_transition_support")
        suggested.append("reference_support_propagation")
    if a800.get("anchor_switch_frames_in_missing", 0) + a1000.get("anchor_switch_frames_in_missing", 0) > 5:
        suggested.append("anchor_transition_support")

    suggested = list(dict.fromkeys(suggested))
    has_light_fix = bool(suggested) and "finalization_hold_downstream" not in dominant

    return {
        "candidate_missing_primary_cause": primary,
        "cause_distribution_merged": dict(merged.most_common()),
        "interpretation": (
            "gap 归因中的 candidate_missing 指 gap 内无 direct_admit；大量 defer_recoverable "
            "已进入 recovery pool 但 PnP/MiniBA inliers 不足，导致 periodic 11-gap 无法被 keyframe 填满。"
        ),
        "minimal_fix_allowed_domains": minimal_fix_allowed,
        "suggested_minimal_fix_points": suggested if has_light_fix else [],
        "has_clear_lightweight_fix": has_light_fix,
        "forbidden_fix_domains": [
            "density_controller_tuning",
            "tau_change",
            "R_V_Q_formula_change",
            "online_psnr_ssim_lpips_rpe_ape_decision",
        ],
        "recommendation": (
            "在 reference/support propagation、recovery pool materialization、anchor transition "
            "support 上做最小修复实验；不要继续调 density/finalization。"
            if has_light_fix
            else "无明确轻量修复点；建议停止 forest1 工程调参，将结果写成 failure-mode diagnostic，"
            "并迁移其他数据集验证机制。"
        ),
        "stop_forest1_engineering_if_no_fix": not has_light_fix,
        "migrate_cross_dataset_validation": True,
    }


def build_report(
    summary: dict[str, Any], judgment: dict[str, Any], a800: dict[str, Any], a1000: dict[str, Any]
) -> str:
    lines = [
        "# PAPER_ALIGNED_UPSTREAM_COVERAGE_POSE_SUPPORT_AUDIT_V1",
        "",
        "只读审计；未修改训练逻辑、R/V/Q/tau、direct/defer/discard 语义。",
        "",
        "## 审计对象",
        "",
        f"- short800: `{a800.get('run_dir')}`",
        f"- short1000: `{a1000.get('run_dir')}`",
        "",
        "注：用户路径 `.../short1000` 实际为 `direct_density_v2_2_2_1_short1000`。",
        "",
        "## 一、candidate coverage",
        "",
        f"- short800 candidate_missing 区间（有 interior）: **{a800.get('missing_interval_count')}**",
        f"- short800 post-500 candidate_missing: **{a800.get('post500_missing_interval_count')}**",
        f"- short800 gap≥11: **{a800.get('gap_ge11_count')}**",
        f"- short1000 candidate_missing 区间: **{a1000.get('missing_interval_count')}**",
        f"- short1000 post-500 candidate_missing: **{a1000.get('post500_missing_interval_count')}**",
        f"- short1000 gap≥11: **{a1000.get('gap_ge11_count')}**",
        "",
        "典型 post-500 11-gap：interior 全为 `defer_recoverable`，无 direct_admit，PnP `pnp_inliers_too_few`。",
        "",
        "## 二、anchor rhythm",
        "",
        "见 `anchor_rhythm_diagnostic.csv`（按 anchor_id 聚合 kf/gap/pnp fail）。",
        "",
        "## 三、pose support",
        "",
        "见 `pose_support_failure_diagnostic.csv`（成功 direct vs pose_fail vs missing 区间对比）。",
        "",
        "## 四、quality alignment",
        "",
        f"- short800 lifecycle test metrics: {a800.get('quality_alignment', {}).get('available')}",
        f"- short1000 lifecycle test metrics: {a1000.get('quality_alignment', {}).get('available')}",
        "",
        "## 五、最终判断",
        "",
        f"1. **candidate_missing 主因**: `{judgment.get('candidate_missing_primary_cause')}`",
        "",
        f"   {judgment.get('interpretation')}",
        "",
        f"2. **存在轻量修复点（不改 R/V/Q/tau/语义）**: {judgment.get('has_clear_lightweight_fix')}",
        "",
        "   建议域: "
        + ", ".join(judgment.get("suggested_minimal_fix_points", []))
        + "",
        "",
        "3. **禁止域**: " + ", ".join(judgment.get("forbidden_fix_domains", [])),
        "",
        f"4. **建议**: {judgment.get('recommendation')}",
        "",
        "## 产物",
        "",
        "- upstream_coverage_pose_support_audit_summary.json",
        "- candidate_missing_intervals_short{800,1000}.csv",
        "- gap_ge_11_frame_expand_short{800,1000}.csv",
        "- anchor_rhythm_diagnostic.csv",
        "- pose_support_failure_diagnostic.csv",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    out = args.output_root
    gap_mod = load_gap_module()

    # Resolve short1000 alias
    alias = out / "short1000"
    if alias.exists() and not (out / "direct_density_v2_2_2_1_short1000").exists():
        RUNS["short1000"]["run_dir"] = alias

    results: dict[str, dict[str, Any]] = {}
    all_anchor: list[dict[str, Any]] = []
    all_pose: list[dict[str, Any]] = []

    for run_key, spec in RUNS.items():
        ar = audit_run(run_key, spec, gap_mod)
        results[run_key] = ar
        write_csv(
            out / f"candidate_missing_intervals_{run_key}.csv",
            ar.get("missing_expand_rows", []),
            FRAME_EXPAND_COLUMNS,
        )
        write_csv(
            out / f"gap_ge_11_frame_expand_{run_key}.csv",
            ar.get("ge11_expand_rows", []),
            FRAME_EXPAND_COLUMNS,
        )
        all_anchor.extend(ar.get("anchor_rhythm", []))
        all_pose.extend(ar.get("pose_support", []))

    judgment = final_judgment(results["short800"], results["short1000"])
    summary = {
        "audit": "PAPER_ALIGNED_UPSTREAM_COVERAGE_POSE_SUPPORT_AUDIT_V1",
        "output_root": str(out),
        "runs": {k: results[k].get("run_dir") for k in RUNS},
        "short800": {k: results["short800"].get(k) for k in results["short800"] if not k.endswith("_rows")},
        "short1000": {k: results["short1000"].get(k) for k in results["short1000"] if not k.endswith("_rows")},
        "final_judgment": judgment,
        "constraints": {
            "read_only": True,
            "stop_density_controller_tuning": True,
            "no_RVQ_tau_semantic_change": True,
        },
    }
    # strip large row arrays from json
    for k in ("short800", "short1000"):
        summary[k].pop("missing_expand_rows", None)
        summary[k].pop("ge11_expand_rows", None)

    write_json(out / "upstream_coverage_pose_support_audit_summary.json", summary)
    write_csv(out / "anchor_rhythm_diagnostic.csv", all_anchor)
    write_csv(out / "pose_support_failure_diagnostic.csv", all_pose)
    (out / "upstream_coverage_pose_support_audit_report.md").write_text(
        build_report(summary, judgment, results["short800"], results["short1000"]),
        encoding="utf-8",
    )
    print(json.dumps(judgment, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
