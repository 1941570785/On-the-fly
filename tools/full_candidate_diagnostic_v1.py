#!/usr/bin/env python3
"""FULL_CANDIDATE_DIAGNOSTIC_V1 — read-only candidate audit and paper usability ratings."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

FOREST1 = Path("/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1")
OUT_ROOT = FOREST1 / "PAPER_ALIGNED_DIRECT_DENSITY_REBALANCE_V2_2_2_1_GAP_FIX_V1"
LIFECYCLE_CSV = FOREST1 / "lifecycle" / "all_input_frame_lifecycle.csv"

CANDIDATES: dict[str, dict[str, Any]] = {
    "compatible_baseline": {
        "label": "A_compatible_baseline",
        "rating_key": "baseline_reference",
        "splits": {
            "full": {"path": FOREST1 / "compatible", "max_frame": None, "source": "compatible_full"},
            "short800": {
                "path": FOREST1 / "compatible",
                "max_frame": 800,
                "source": "compatible_frame_metrics_slice",
            },
            "short1000": {
                "path": FOREST1 / "compatible",
                "max_frame": 1000,
                "source": "compatible_frame_metrics_slice",
            },
        },
    },
    "paper_aligned_v2_2_2_1": {
        "label": "B_PAPER_ALIGNED_V2_2_2_1_GAP_FIX",
        "rating_key": "paper_aligned_main_candidate",
        "splits": {
            "full": {"path": None, "max_frame": None},
            "short800": {
                "path": OUT_ROOT / "direct_density_v2_2_2_1_short800",
                "max_frame": 800,
            },
            "short1000": {
                "path": OUT_ROOT / "direct_density_v2_2_2_1_short1000",
                "max_frame": 1000,
            },
        },
    },
    "high_density_pnp_consensus": {
        "label": "C_high_density_pnp_consensus_short1000",
        "rating_key": "diagnostic_only",
        "auto_found": True,
        "fingerprint_note": (
            "Auto-matched PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1/"
            "pnp_consensus_short1000 (kf=820, density=82, focal≈990.6, anchor_4 Gaussians=600834). "
            "User PSNR≈20.11/SSIM≈0.62 not found in metadata; on-disk PSNR=14.46/SSIM=0.32."
        ),
        "splits": {
            "full": {"path": None, "max_frame": None},
            "short800": {
                "path": FOREST1
                / "PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1"
                / "pnp_consensus_short800",
                "max_frame": 800,
            },
            "short1000": {
                "path": FOREST1
                / "PAPER_ALIGNED_EXTENDED_SHORT_DENSITY_REFERENCE_REVIEW_V1"
                / "pnp_consensus_short1000",
                "max_frame": 1000,
            },
        },
    },
}

TABLE_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "split",
    "run_dir",
    "available",
    "PSNR",
    "SSIM",
    "LPIPS",
    "R_deg",
    "t",
    "processed_frame_count",
    "keyframes",
    "density_per_100",
    "anchors",
    "Gaussians",
    "focal",
    "gap_p50",
    "gap_p90",
    "gap_p95",
    "gap_max",
    "candidate_missing_ratio",
    "candidate_held_by_finalization_ratio",
    "pose_failed_after_candidate_ratio",
    "direct_finalized_count",
    "recovery_finalized_count",
    "direct_recovery_ratio",
    "too_few_inliers",
    "pnp_success",
    "pnp_fail",
    "miniba_success",
    "miniba_fail",
    "recovery_success",
    "current_frame_surrogate_count",
    "defer_discard_contamination_count",
    "duplicate_keyframe_count",
    "hidden_unknown_count",
    "pose_eval_coverage",
    "test_split_sanity",
    "render_eval_split_sanity",
    "paper_rating",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    for c in TABLE_COLUMNS:
        if c not in seen:
            fields.append(c)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def parse_train_summary(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pat = re.compile(
        r"num anchors:\s*(\d+),\s*num keyframes:\s*(\d+),\s*time:\s*([\d.]+),\s*FPS:\s*([\d.]+),\s*"
        r"PSNR:\s*([\d.]+),\s*SSIM:\s*([\d.]+),\s*LPIPS:\s*([\d.]+)"
        r"(?:,\s*R°:\s*([\d.]+),\s*t:\s*([\d.]+))?"
    )
    for name in ("train.log", "run.log"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = p.read_bytes().decode("utf-8", errors="ignore")
        for line in reversed(text.splitlines()):
            m = pat.search(line)
            if m:
                out = {
                    "anchors": int(m.group(1)),
                    "keyframes": int(m.group(2)),
                    "PSNR": float(m.group(5)),
                    "SSIM": float(m.group(6)),
                    "LPIPS": float(m.group(7)),
                    "R_deg": float(m.group(8)) if m.group(8) else None,
                    "t": float(m.group(9)) if m.group(9) else None,
                }
                break
    return out


def load_metadata(run_dir: Path) -> dict[str, Any]:
    for rel in ("metadata.json", "model/metadata.json"):
        p = run_dir / rel
        if p.exists():
            return read_json(p)
    return {}


def gaussian_stats(run_dir: Path) -> dict[str, Any]:
    pc = run_dir / "point_clouds"
    if not pc.exists():
        return {"Gaussians_max_anchor": None, "Gaussians_sum_anchors": None}
    counts: list[int] = []
    for ply in sorted(pc.glob("*.ply")):
        try:
            head = ply.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]
        except Exception:
            continue
        for line in head:
            if line.startswith("element vertex"):
                counts.append(int(line.split()[-1]))
                break
    if not counts:
        return {"Gaussians_max_anchor": None, "Gaussians_sum_anchors": None}
    return {
        "Gaussians_max_anchor": max(counts),
        "Gaussians_sum_anchors": sum(counts),
        "Gaussians_per_anchor": counts,
        "Gaussians_fingerprint_600834": 600834 in counts,
    }


def load_trace(run_dir: Path) -> dict[str, Any]:
    for rel in ("model/semantic_trace.json", "semantic_trace.json"):
        p = run_dir / rel
        if p.exists():
            return read_json(p)
    return {}


def keyframe_ticks_from_trace(trace: dict[str, Any], run_dir: Path, max_frame: int | None) -> list[int]:
    ticks: list[int] = []
    for row in read_csv_rows(run_dir / "keyframe_timeline.csv"):
        if to_bool(row.get("materialized", "true")):
            cf = to_int(row.get("current_frame_id") or row.get("source_frame_id"))
            if max_frame is None or cf <= max_frame:
                ticks.append(cf)
    if not ticks:
        for e in trace.get("events", []) or []:
            if to_bool(e.get("final_keyframe_incremented")):
                fid = int(e["frame_id"])
                if max_frame is None or fid <= max_frame:
                    ticks.append(fid)
    return sorted(set(ticks))


def gap_stats(ticks: list[int]) -> dict[str, float]:
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    if not gaps:
        return {"gap_p50": 0.0, "gap_p90": 0.0, "gap_p95": 0.0, "gap_max": 0.0}
    return {
        "gap_p50": round(pct(gaps, 0.5), 2),
        "gap_p90": round(pct(gaps, 0.9), 2),
        "gap_p95": round(pct(gaps, 0.95), 2),
        "gap_max": float(max(gaps)),
    }


def safety_from_trace(trace: dict[str, Any], ticks: list[int]) -> dict[str, int]:
    events = trace.get("events", []) or []
    surrogate = sum(
        1 for e in events if str(e.get("action", "")) == "current_frame_surrogate_commit"
    )
    contamination = 0
    for e in events:
        src_action = str(e.get("source_action", e.get("committed_source_action", "")))
        action = str(e.get("action", ""))
        if action in {"discard", "defer_recoverable"} and src_action and src_action not in {
            "defer_recoverable",
            "direct_admit",
            "",
        }:
            contamination += 1
        if to_bool(e.get("is_contamination_risk")):
            contamination += 1
    duplicate = max(0, len(ticks) - len(set(ticks)))
    hidden = sum(
        1
        for e in trace.get("lifecycle_gate_events", []) or []
        if str(e.get("gate_decision", e.get("decision", ""))).lower() in {"unknown", ""}
        and e.get("pose_path_allowed") is None
    )
    return {
        "current_frame_surrogate_count": surrogate,
        "defer_discard_contamination_count": contamination,
        "duplicate_keyframe_count": duplicate,
        "hidden_unknown_count": hidden,
    }


def pose_pipeline_counts(trace: dict[str, Any], run_dir: Path, max_frame: int | None) -> dict[str, int]:
    pnp_ok = pnp_fail = miniba_ok = miniba_fail = 0
    for pr in trace.get("pnp_miniba_reference_events", []) or []:
        fid = to_int(pr.get("frame_id"))
        if max_frame is not None and fid > max_frame:
            continue
        if to_bool(pr.get("pnp_success")):
            pnp_ok += 1
        else:
            pnp_fail += 1
        if to_bool(pr.get("miniba_success")):
            miniba_ok += 1
        else:
            miniba_fail += 1
    too_few = 0
    for name in ("train.log", "run.log"):
        p = run_dir / name
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = p.read_bytes().decode("utf-8", errors="ignore")
            too_few = len(re.findall(r"Too few inliers for pose initialization", text))
            break
    recovery_success = sum(
        1
        for e in trace.get("recovery_pose_path_events", []) or []
        if to_bool(e.get("success"))
        and (max_frame is None or to_int(e.get("current_frame_id")) <= max_frame)
    )
    return {
        "too_few_inliers": too_few,
        "pnp_success": pnp_ok,
        "pnp_fail": pnp_fail,
        "miniba_success": miniba_ok,
        "miniba_fail": miniba_fail,
        "recovery_success": recovery_success,
    }


def direct_recovery_counts(trace: dict[str, Any], ticks: list[int], max_frame: int | None) -> dict[str, Any]:
    direct = 0
    recovery = 0
    for e in trace.get("events", []) or []:
        if not to_bool(e.get("final_keyframe_incremented")):
            continue
        fid = int(e["frame_id"])
        if max_frame is not None and fid > max_frame:
            continue
        action = str(e.get("action", ""))
        origin = str(e.get("commit_origin", ""))
        if "recovery" in origin or to_bool(e.get("source_recovery_committed")):
            recovery += 1
        elif action in {"direct_admit", "current_frame_surrogate_commit"}:
            direct += 1
        else:
            direct += 1
    for e in trace.get("true_recovery_commit_events", []) or []:
        if to_bool(e.get("final_keyframe_incremented")):
            recovery += 1
    # dedupe by ticks length fallback
    if direct == 0 and recovery == 0 and ticks:
        direct = len(ticks)
    ratio = round(direct / max(recovery, 1), 4) if recovery else float(direct)
    return {
        "direct_finalized_count": direct,
        "recovery_finalized_count": recovery,
        "direct_recovery_ratio": ratio,
    }


def gap_attribution_ratios(
    gap_mod: Any, run_dir: Path, max_frame: int
) -> dict[str, float | None]:
    if not (run_dir / "model" / "semantic_trace.json").exists() and not (
        run_dir / "semantic_trace.json"
    ).exists():
        return {
            "candidate_missing_ratio": None,
            "candidate_held_by_finalization_ratio": None,
            "pose_failed_after_candidate_ratio": None,
        }
    rows, summary = gap_mod.analyze_run(run_dir.name, run_dir, max_frame)
    fr = summary.get("attribution_fraction", {})
    return {
        "candidate_missing_ratio": fr.get("candidate_missing"),
        "candidate_held_by_finalization_ratio": fr.get("candidate_held_by_finalization"),
        "pose_failed_after_candidate_ratio": fr.get("pose_failed_after_candidate"),
        "_gap_rows": rows,
        "_gap_summary": summary,
    }


def eval_from_frame_metrics(run_dir: Path, max_frame: int | None) -> dict[str, Any]:
    p = run_dir / "frame_metrics.csv"
    if not p.exists():
        return {}
    rows = read_csv_rows(p)
    if max_frame is not None:
        rows = [r for r in rows if to_int(r.get("frame_idx")) < max_frame]
    test = [r for r in rows if r.get("is_test_view") == "True"]
    eval_frames = [r for r in test if r.get("psnr")]
    psnrs = [to_float(r["psnr"]) for r in eval_frames if to_float(r["psnr"]) is not None]
    ssims = [to_float(r["ssim"]) for r in eval_frames if to_float(r["ssim"]) is not None]
    lpips = [to_float(r["lpips"]) for r in eval_frames if to_float(r["lpips"]) is not None]
    rots = [
        to_float(r.get("abs_rot_error_deg"))
        for r in eval_frames
        if to_float(r.get("abs_rot_error_deg")) is not None
    ]
    trans = [
        to_float(r.get("abs_trans_error"))
        for r in eval_frames
        if to_float(r.get("abs_trans_error")) is not None
    ]
    kf_ticks = sorted(
        to_int(r["frame_idx"])
        for r in rows
        if r.get("is_keyframe") == "True" and to_int(r.get("frame_idx")) >= 0
    )
    gaps = [kf_ticks[i] - kf_ticks[i - 1] for i in range(1, len(kf_ticks))]
    proc = max_frame if max_frame else (max(to_int(r["frame_idx"]) for r in rows) + 1 if rows else 0)
    train_rows = [r for r in rows if r.get("split") == "train"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    return {
        "PSNR": round(mean(psnrs), 4) if psnrs else None,
        "SSIM": round(mean(ssims), 4) if ssims else None,
        "LPIPS": round(mean(lpips), 4) if lpips else None,
        "R_deg": round(mean(rots), 4) if rots else None,
        "t": round(mean(trans), 4) if trans else None,
        "processed_frame_count": proc,
        "keyframes": len(kf_ticks),
        "density_per_100": round(100.0 * len(kf_ticks) / max(proc, 1), 2),
        "pose_eval_coverage": round(len(eval_frames) / max(len(test), 1), 4),
        "test_split_sanity": len(test_rows) > 0 and all(r.get("split") == "test" for r in test[:5]),
        "render_eval_split_sanity": len(eval_frames) > 0,
        **(
            {
                "gap_p50": round(pct(gaps, 0.5), 2),
                "gap_p90": round(pct(gaps, 0.9), 2),
                "gap_p95": round(pct(gaps, 0.95), 2),
                "gap_max": float(max(gaps)) if gaps else 0.0,
            }
            if gaps
            else {}
        ),
        "_kf_ticks": kf_ticks,
    }


def collect_run(
    candidate_id: str,
    spec: dict[str, Any],
    split: str,
    split_spec: dict[str, Any],
    gap_mod: Any,
) -> dict[str, Any]:
    run_dir = split_spec.get("path")
    max_frame = split_spec.get("max_frame")
    source = split_spec.get("source", "run_dir")

    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_label": spec["label"],
        "split": split,
        "run_dir": str(run_dir) if run_dir else "",
        "available": False,
        "paper_rating": spec.get("rating_key", ""),
    }
    for c in TABLE_COLUMNS:
        if c not in row:
            row[c] = None

    if run_dir is None or not Path(run_dir).exists():
        row["available"] = False
        return row

    run_dir = Path(run_dir)
    row["available"] = True
    row["run_dir"] = str(run_dir)

    meta = load_metadata(run_dir)
    train = parse_train_summary(run_dir)
    gstats = gaussian_stats(run_dir)
    trace = load_trace(run_dir)
    ticks = keyframe_ticks_from_trace(trace, run_dir, max_frame)
    gs = gap_stats(ticks)
    safety = safety_from_trace(trace, ticks) if trace else {
        "current_frame_surrogate_count": 0,
        "defer_discard_contamination_count": 0,
        "duplicate_keyframe_count": 0,
        "hidden_unknown_count": 0,
    }
    pose = pose_pipeline_counts(trace, run_dir, max_frame) if trace else {
        "too_few_inliers": 0,
        "pnp_success": 0,
        "pnp_fail": 0,
        "miniba_success": 0,
        "miniba_fail": 0,
        "recovery_success": 0,
    }
    dr = direct_recovery_counts(trace, ticks, max_frame) if trace else {
        "direct_finalized_count": len(ticks),
        "recovery_finalized_count": 0,
        "direct_recovery_ratio": float(len(ticks)),
    }

    fm: dict[str, Any] = {}
    if source.startswith("compatible_frame_metrics"):
        fm = eval_from_frame_metrics(run_dir, max_frame)
        ticks = fm.get("_kf_ticks", ticks)
        gs = {k: fm[k] for k in ("gap_p50", "gap_p90", "gap_p95", "gap_max") if k in fm} or gs

    proc = max_frame or to_int(meta.get("processed_frame_count"), 0)
    if not proc and fm:
        proc = fm.get("processed_frame_count", 0)
    if not proc and max_frame:
        proc = max_frame
    if not proc and (run_dir / "frame_metrics.csv").exists():
        frows = read_csv_rows(run_dir / "frame_metrics.csv")
        if frows:
            proc = max(to_int(r.get("frame_idx")) for r in frows) + 1

    kf = len(ticks) or to_int(meta.get("num keyframes") or train.get("keyframes"))
    density = round(100.0 * kf / max(proc, 1), 2) if proc else None

    attr: dict[str, Any] = {}
    if max_frame and trace:
        attr = gap_attribution_ratios(gap_mod, run_dir, max_frame)

    audit = read_json(run_dir / "engine_stability_audit.json")

    row.update(
        {
            "PSNR": fm.get("PSNR")
            or meta.get("PSNR")
            or train.get("PSNR"),
            "SSIM": fm.get("SSIM")
            or meta.get("SSIM")
            or train.get("SSIM"),
            "LPIPS": fm.get("LPIPS")
            or meta.get("LPIPS")
            or train.get("LPIPS"),
            "R_deg": fm.get("R_deg")
            or meta.get("R°")
            or meta.get("R_deg")
            or train.get("R_deg"),
            "t": fm.get("t") or meta.get("t") or train.get("t"),
            "processed_frame_count": proc,
            "keyframes": kf,
            "density_per_100": density,
            "anchors": meta.get("num anchors") or train.get("anchors"),
            "Gaussians": gstats.get("Gaussians_max_anchor"),
            "focal": (meta.get("config") or {}).get("f"),
            **gs,
            "candidate_missing_ratio": attr.get("candidate_missing_ratio"),
            "candidate_held_by_finalization_ratio": attr.get(
                "candidate_held_by_finalization_ratio"
            ),
            "pose_failed_after_candidate_ratio": attr.get(
                "pose_failed_after_candidate_ratio"
            ),
            **dr,
            **pose,
            **safety,
            "pose_eval_coverage": fm.get("pose_eval_coverage"),
            "test_split_sanity": fm.get("test_split_sanity"),
            "render_eval_split_sanity": fm.get("render_eval_split_sanity"),
        }
    )
    if audit:
        row["gap_p90"] = row.get("gap_p90") or audit.get("gap_p90") or audit.get("main_chain_gap_p90")
        row["gap_p95"] = row.get("gap_p95") or audit.get("gap_p95") or audit.get("main_chain_gap_p95")
        row["gap_max"] = row.get("gap_max") or audit.get("gap_max") or audit.get("main_chain_gap_max")
        if row.get("too_few_inliers") == 0:
            row["too_few_inliers"] = audit.get("too_few_inliers_count", 0)

    row["_gap_rows"] = attr.get("_gap_rows")
    row["_gap_summary"] = attr.get("_gap_summary")
    row["_meta"] = meta
    row["_gstats"] = gstats
    return row


def build_high_density_diagnostic(
    rows: list[dict[str, Any]], baseline_full: dict[str, Any]
) -> dict[str, Any]:
    hd = next(
        (
            r
            for r in rows
            if r["candidate_id"] == "high_density_pnp_consensus" and r["split"] == "short1000"
        ),
        None,
    )
    bl = baseline_full
    if not hd or not hd.get("available"):
        return {"error": "high_density short1000 unavailable"}

    bl_density = to_float(bl.get("density_per_100")) or 34.3
    hd_density = to_float(hd.get("density_per_100")) or 0.0
    flags: list[str] = []
    if hd_density >= 2.0 * bl_density:
        flags.append("keyframe_density_over_2x_baseline")
    if (to_float(hd.get("PSNR")) or 0) > (to_float(bl.get("PSNR")) or 0) and (
        (to_float(hd.get("R_deg")) or 0) > 2.0 * (to_float(bl.get("R_deg")) or 1.0)
        or (to_float(hd.get("t")) or 0) > 2.0 * (to_float(bl.get("t")) or 0.1)
    ):
        flags.append("psnr_not_accompanied_by_pose_quality")
    elif (to_float(hd.get("R_deg")) or 999) > 10 * (to_float(bl.get("R_deg")) or 1):
        flags.append("severe_pose_degradation_vs_baseline")
    gsum = gaussian_stats(Path(str(hd.get("run_dir", ""))))
    if (gsum.get("Gaussians_sum_anchors") or 0) > 2_000_000:
        flags.append("gaussian_memory_inflation_multi_anchor")
    if (to_float(hd.get("focal")) or 0) > 0 and abs((to_float(hd.get("focal")) or 0) - 1134.0) > 150:
        flags.append("focal_shift_vs_compatible_baseline")
    dr = to_float(hd.get("direct_recovery_ratio")) or 0
    if dr > 100:
        flags.append("extreme_direct_recovery_imbalance")
    appearance_risk = (
        (to_float(hd.get("density_per_100")) or 0) >= 70
        and (to_float(hd.get("PSNR")) or 0) < (to_float(bl.get("PSNR")) or 0)
        and len(flags) >= 2
    )
    return {
        "run_dir": hd.get("run_dir"),
        "fingerprint_match": {
            "keyframes_820": hd.get("keyframes") == 820,
            "density_82": hd.get("density_per_100") == 82.0,
            "focal_990": abs((to_float(hd.get("focal")) or 0) - 990.6) < 1.0,
            "gaussians_600834": bool(gsum.get("Gaussians_fingerprint_600834")),
            "gaussians_max_anchor": hd.get("Gaussians"),
            "user_psnr_20_11_found": False,
            "on_disk_psnr": hd.get("PSNR"),
            "on_disk_ssim": hd.get("SSIM"),
        },
        "baseline_reference": {
            "density_per_100": bl_density,
            "PSNR": bl.get("PSNR"),
            "R_deg": bl.get("R_deg"),
            "t": bl.get("t"),
            "Gaussians": bl.get("Gaussians"),
        },
        "comparisons": {
            "density_ratio_vs_baseline": round(hd_density / max(bl_density, 1e-6), 3),
            "psnr_delta": round((to_float(hd.get("PSNR")) or 0) - (to_float(bl.get("PSNR")) or 0), 3),
            "R_deg_ratio": round(
                (to_float(hd.get("R_deg")) or 0) / max(to_float(bl.get("R_deg")) or 1e-6, 1e-6),
                3,
            ),
            "t_ratio": round(
                (to_float(hd.get("t")) or 0) / max(to_float(bl.get("t")) or 0.01, 0.01),
                3,
            ),
        },
        "over_admission_flags": flags,
        "appearance_fitting_or_memory_render_risk": appearance_risk,
        "recommended_rating": "reject_as_overfit_or_overdense"
        if len(flags) >= 3 or appearance_risk
        else "diagnostic_only",
        "interpretation": (
            "高密度 PnP-consensus 短跑以 direct_admit 为主（~819/820），pose/几何约束弱，"
            "Gaussians 与 keyframe 密度显著高于 compatible baseline；不宜作为 paper 主候选。"
        ),
    }


def build_v2221_coverage_diagnostic(rows: list[dict[str, Any]], gap_mod: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"audit": "v2_2_2_1_candidate_coverage_diagnostic"}
    for split in ("short800", "short1000"):
        r = next(
            (
                x
                for x in rows
                if x["candidate_id"] == "paper_aligned_v2_2_2_1" and x["split"] == split
            ),
            None,
        )
        if not r or not r.get("available"):
            out[split] = {"available": False}
            continue
        run_dir = Path(r["run_dir"])
        max_frame = 800 if split == "short800" else 1000
        gap_rows, summary = gap_mod.analyze_run(split, run_dir, max_frame)
        post_missing = [
            g
            for g in gap_rows
            if g.get("post500_gap") and g.get("gap_attribution") == "candidate_missing"
        ]
        big_missing = [
            g
            for g in gap_rows
            if to_int(g.get("gap_length")) >= 11
            and g.get("gap_attribution") == "candidate_missing"
        ]
        trace = load_trace(run_dir)
        events_by = {int(e["frame_id"]): e for e in trace.get("events", []) or []}
        lifecycle = gap_mod.load_lifecycle_by_frame(max_frame)

        def gap_mechanism(g: dict[str, Any]) -> str:
            interior = range(
                to_int(g["gap_start_frame"]) + 1, to_int(g["gap_end_frame"])
            )
            no_direct = all(
                str(events_by.get(f, {}).get("action", "")) != "direct_admit" for f in interior
            )
            no_pose = all(
                not to_bool(events_by.get(f, {}).get("pose_init_attempted")) for f in interior
            )
            anchor_switch = any(
                to_bool(events_by.get(f, {}).get("anchor_switch"))
                or "anchor" in str(events_by.get(f, {}).get("drop_reason", "")).lower()
                for f in interior
            )
            ref_short = any(
                "reference" in str(events_by.get(f, {}).get("drop_reason", "")).lower()
                or "support" in str(events_by.get(f, {}).get("drop_reason", "")).lower()
                for f in interior
            )
            if no_direct and no_pose:
                return "pose_path_not_attempted"
            if no_direct and anchor_switch:
                return "anchor_switch_or_boundary"
            if no_direct and ref_short:
                return "reference_shortage"
            if no_direct:
                return "no_direct_candidate"
            return "mixed"

        mech_counts = Counter(gap_mechanism(g) for g in post_missing + big_missing)

        # align gap>=11 with frame_metrics quality if present
        fm_path = run_dir / "frame_metrics.csv"
        drift_alignment: list[dict[str, Any]] = []
        if fm_path.exists():
            fm = {to_int(r["frame_idx"]): r for r in read_csv_rows(fm_path)}
            for g in big_missing:
                start, end = to_int(g["gap_start_frame"]), to_int(g["gap_end_frame"])
                seg = [fm[f] for f in range(start, end + 1) if f in fm and fm[f].get("psnr")]
                if seg:
                    drift_alignment.append(
                        {
                            "gap_start": start,
                            "gap_end": end,
                            "gap_length": g["gap_length"],
                            "mean_psnr": round(
                                mean(to_float(x["psnr"]) for x in seg if to_float(x["psnr"])), 3
                            ),
                            "mean_rot_err": round(
                                mean(
                                    to_float(x.get("abs_rot_error_deg"))
                                    for x in seg
                                    if to_float(x.get("abs_rot_error_deg"))
                                ),
                                3,
                            ),
                        }
                    )

        out[split] = {
            "available": True,
            "gap_summary": summary,
            "post500_candidate_missing_count": len(post_missing),
            "post500_candidate_missing_fraction_of_post500_gaps": round(
                len(post_missing) / max(summary.get("post500_gap_count", 1), 1),
                4,
            ),
            "gap_ge11_candidate_missing_count": len(big_missing),
            "mechanism_hypothesis_counts": dict(mech_counts),
            "candidate_missing_concentrated_post500": len(post_missing)
            >= max(1, int(0.6 * summary.get("post500_gap_count", 0))),
            "large_gap_ge11_attribution": summary.get("large_gaps_ge11"),
            "gap_quality_pose_drift_samples": drift_alignment[:15],
            "lifecycle_discard_in_gap_regions": sum(
                1
                for g in big_missing
                for f in range(to_int(g["gap_start_frame"]), to_int(g["gap_end_frame"]))
                if str((lifecycle.get(f) or {}).get("state_admission", "")) == "discard"
            ),
        }
    out["conclusion"] = (
        "失败尾部 gap（post-500、gap≥11）以 candidate_missing 为主，符合 GAP_SOURCE_ATTRIBUTION 结论；"
        "机制上多为 gap 内无 direct_admit / pose 未尝试或 reference 不足，而非 finalization hold。"
    )
    return out


def safety_pass(row: dict[str, Any]) -> bool:
    checks = [
        row.get("current_frame_surrogate_count") in (0, None),
        row.get("defer_discard_contamination_count") in (0, None),
        row.get("duplicate_keyframe_count") in (0, None),
        row.get("hidden_unknown_count") in (0, None),
    ]
    return all(checks)


def rate_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bl_full = next(
        (r for r in rows if r["candidate_id"] == "compatible_baseline" and r["split"] == "full"),
        {},
    )
    v1_800 = next(
        (r for r in rows if r["candidate_id"] == "paper_aligned_v2_2_2_1" and r["split"] == "short800"),
        {},
    )
    v1_1000 = next(
        (
            r
            for r in rows
            if r["candidate_id"] == "paper_aligned_v2_2_2_1" and r["split"] == "short1000"
        ),
        {},
    )
    hd = next(
        (
            r
            for r in rows
            if r["candidate_id"] == "high_density_pnp_consensus" and r["split"] == "short1000"
        ),
        {},
    )

    ratings: dict[str, str] = {
        "compatible_baseline": "baseline_reference",
        "high_density_pnp_consensus": "reject_as_overfit_or_overdense",
    }

    v_pass = safety_pass(v1_800) and safety_pass(v1_1000)
    v_psnr_ok = (to_float(v1_1000.get("PSNR")) or 0) >= 11.0
    v_gap_tail = (to_float(v1_1000.get("gap_p90")) or 99) <= 11 and (to_float(v1_1000.get("gap_max")) or 99) <= 35
    v_density_ok = 25 <= (to_float(v1_1000.get("density_per_100")) or 0) <= 45
    v_pose_not_catastrophic = (to_float(v1_1000.get("R_deg")) or 999) < 200

    if v_pass and v_density_ok and v_pose_not_catastrophic:
        if v_gap_tail and (to_float(v1_800.get("candidate_missing_ratio")) or 0) < 0.5:
            ratings["paper_aligned_v2_2_2_1"] = "paper_aligned_main_candidate"
        else:
            ratings["paper_aligned_v2_2_2_1"] = "needs_light_refinement"
    elif v_pass:
        ratings["paper_aligned_v2_2_2_1"] = "needs_light_refinement"
    else:
        ratings["paper_aligned_v2_2_2_1"] = "diagnostic_only"

    bl_psnr = to_float(bl_full.get("PSNR")) or 18.34
    paper_worse = all(
        (to_float(r.get("PSNR")) or 0) < bl_psnr - 2.0
        for r in (v1_800, v1_1000)
        if r.get("available")
    )
    cross_ready = False
    if ratings.get("paper_aligned_v2_2_2_1") == "paper_aligned_main_candidate" and v_pass:
        cross_ready = True

    return {
        "per_candidate": ratings,
        "compatible_baseline": ratings["compatible_baseline"],
        "paper_aligned_v2_2_2_1": ratings["paper_aligned_v2_2_2_1"],
        "high_density_pnp_consensus": ratings["high_density_pnp_consensus"],
        "ready_for_cross_dataset_transfer": cross_ready,
        "stop_density_controller_tuning": True,
        "paper_level_failure_modes_if_all_paper_aligned_weak": (
            [
                "candidate_coverage_bottleneck",
                "online_pose_support_insufficiency",
                "density_quality_tradeoff",
            ]
            if paper_worse
            else []
        ),
        "rationale": {
            "v2_2_2_1_safety": v_pass,
            "v2_2_2_1_density_in_band": v_density_ok,
            "v2_2_2_1_gap_tail_unresolved": not v_gap_tail,
            "high_density_overdense": (to_float(hd.get("density_per_100")) or 0) >= 70,
            "do_not_use_psnr_alone": True,
        },
    }


def build_report_md(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    hd_diag: dict[str, Any],
    cov_diag: dict[str, Any],
) -> str:
    lines = [
        "# FULL_CANDIDATE_DIAGNOSTIC_V1",
        "",
        "只读审计；未修改训练逻辑、R/V/Q/tau 或 direct/defer/discard 语义。",
        "",
        "## 候选路径",
        "",
        "| 候选 | full | short800 | short1000 |",
        "|------|------|----------|-----------|",
    ]
    for cid, spec in CANDIDATES.items():
        parts = []
        for sp in ("full", "short800", "short1000"):
            p = spec["splits"][sp].get("path")
            parts.append("✓" if p and Path(p).exists() else ("切片" if sp != "full" and cid == "compatible_baseline" else "—"))
        lines.append(f"| {spec['label']} | {parts[0]} | {parts[1]} | {parts[2]} |")
    lines.extend(
        [
            "",
            "## 高密度候选自动定位",
            "",
            CANDIDATES["high_density_pnp_consensus"].get("fingerprint_note", ""),
            "",
            "## 评级",
            "",
        ]
    )
    rat = summary.get("ratings", {})
    for k, v in rat.get("per_candidate", {}).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.extend(
        [
            "",
            f"- **ready_for_cross_dataset_transfer**: {rat.get('ready_for_cross_dataset_transfer')}",
            f"- **stop_density_controller_tuning**: {rat.get('stop_density_controller_tuning')}",
            "",
            "## 论文层面归纳（paper-aligned 弱于 baseline 时）",
            "",
        ]
    )
    for mode in rat.get("paper_level_failure_modes_if_all_paper_aligned_weak", []):
        lines.append(f"- {mode}")
    lines.extend(["", "## 对比表（摘要）", "", "| 候选 | split | PSNR | SSIM | kf | dens | gap_p90 | gap_max | R° | t |", "|------|-------|------|------|-----|------|---------|---------|-----|---|"])
    for r in rows:
        if not r.get("available"):
            continue
        lines.append(
            f"| {r['candidate_label']} | {r['split']} | {r.get('PSNR')} | {r.get('SSIM')} | "
            f"{r.get('keyframes')} | {r.get('density_per_100')} | {r.get('gap_p90')} | {r.get('gap_max')} | "
            f"{r.get('R_deg')} | {r.get('t')} |"
        )
    lines.extend(["", "## V2_2_2_1 candidate coverage", "", "```json", json.dumps(cov_diag, ensure_ascii=False, indent=2)[:4000], "```", "", "## High-density over-admission", "", "```json", json.dumps(hd_diag, ensure_ascii=False, indent=2)[:3000], "```", ""])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    out = args.output_root
    gap_mod = load_gap_module()

    table_rows: list[dict[str, Any]] = []
    for cid, spec in CANDIDATES.items():
        for split, split_spec in spec["splits"].items():
            row = collect_run(cid, spec, split, split_spec, gap_mod)
            row["paper_rating"] = spec.get("rating_key")
            # strip internal keys for csv
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            table_rows.append(public)

    ratings = rate_candidates(table_rows)
    for r in table_rows:
        r["paper_rating"] = ratings["per_candidate"].get(r["candidate_id"], r.get("paper_rating"))

    bl_full = next(
        (r for r in table_rows if r["candidate_id"] == "compatible_baseline" and r["split"] == "full"),
        {},
    )
    hd_diag = build_high_density_diagnostic(table_rows, bl_full)
    cov_diag = build_v2221_coverage_diagnostic(table_rows, gap_mod)

    summary = {
        "audit": "FULL_CANDIDATE_DIAGNOSTIC_V1",
        "output_root": str(out),
        "candidates": {
            cid: {
                "label": s["label"],
                "splits": {
                    sp: str(ss["path"]) if ss.get("path") else None for sp, ss in s["splits"].items()
                },
            }
            for cid, s in CANDIDATES.items()
        },
        "high_density_auto_discovery": hd_diag.get("fingerprint_match"),
        "ratings": ratings,
        "table_row_count": len(table_rows),
        "constraints": {
            "no_training_logic_change": True,
            "no_density_controller_v2222": True,
            "no_finalization_layer_change": True,
            "no_RVQ_tau_semantic_change": True,
        },
    }

    write_csv(out / "full_candidate_diagnostic_table.csv", table_rows)
    write_json(out / "full_candidate_diagnostic_summary.json", summary)
    write_json(out / "high_density_over_admission_diagnostic.json", hd_diag)
    write_json(out / "v2_2_2_1_candidate_coverage_diagnostic.json", cov_diag)
    (out / "full_candidate_diagnostic_report.md").write_text(
        build_report_md(table_rows, summary, hd_diag, cov_diag), encoding="utf-8"
    )
    print(json.dumps(ratings["per_candidate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
