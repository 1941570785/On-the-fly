#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile_int(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = int(round((len(arr) - 1) * p))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def compute_gap_stats(ticks: list[int]) -> dict[str, float]:
    if len(ticks) < 2:
        return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    gaps = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    return {
        "p50": percentile_int(gaps, 0.5),
        "p75": percentile_int(gaps, 0.75),
        "p90": percentile_int(gaps, 0.9),
        "max": float(max(gaps) if gaps else 0),
    }


def simulate_mode(
    mode: str,
    direct_ticks: list[int],
    recovery_candidates: list[dict[str, Any]],
    total_frames: int,
    baseline_density_per_100: float,
    window_size: int = 30,
    max_per_window: int = 5,
) -> dict[str, Any]:
    # candidates sorted by recovery attempt tick
    rec_candidates = sorted(recovery_candidates, key=lambda x: (x["commit_tick"], x["source_frame_id"]))
    accepted: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    accepted_ticks = set(direct_ticks)
    accepted_recovery_ticks: list[int] = []
    timeline_rows: list[dict[str, Any]] = []

    for cand in rec_candidates:
        tick = int(cand["commit_tick"])
        lo = max(1, tick - window_size + 1)
        recent_recovery = sum(1 for t in accepted_recovery_ticks if lo <= t <= tick)
        current_density = 100.0 * (len(accepted_ticks)) / max(tick, 1)
        gap_stats = compute_gap_stats(sorted(accepted_ticks))
        gap_p90 = gap_stats["p90"]

        allow = True
        reason = "accepted_current"
        if mode == "conservative_window_control":
            if recent_recovery >= max_per_window:
                allow = False
                reason = "window_rate_limit"
            elif current_density > baseline_density_per_100 * 1.6 and gap_p90 <= 2.0:
                allow = False
                reason = "density_guard"
        elif mode == "adaptive_gap_density_control":
            adaptive_limit = max_per_window
            if gap_p90 >= 6.0:
                adaptive_limit += 2
            elif gap_p90 >= 4.0:
                adaptive_limit += 1
            elif current_density > baseline_density_per_100 * 1.6:
                adaptive_limit = max(2, max_per_window - 2)
            elif current_density > baseline_density_per_100 * 1.4:
                adaptive_limit = max(3, max_per_window - 1)

            if recent_recovery >= adaptive_limit:
                allow = False
                reason = "adaptive_window_limit"
            elif current_density > baseline_density_per_100 * 1.8 and gap_p90 <= 2.0:
                allow = False
                reason = "adaptive_density_guard"

        if mode == "current_no_control":
            allow = True
            reason = "accepted_current"

        if allow:
            accepted.append(cand)
            accepted_ticks.add(tick)
            accepted_recovery_ticks.append(tick)
        else:
            # preflight只区分hold，不做永久reject
            held.append(cand | {"hold_reason": reason})

        timeline_rows.append(
            {
                "mode": mode,
                "tick": tick,
                "recent_recovery_in_window": recent_recovery,
                "accepted_recovery_count_so_far": len(accepted),
                "held_recovery_count_so_far": len(held),
                "current_keyframe_density_per_100": current_density,
                "current_main_chain_gap_p90": gap_p90,
                "decision": "commit" if allow else "hold",
                "decision_reason": reason,
                "V_t": cand.get("V_t"),
                "Q_t": cand.get("Q_t"),
            }
        )

    final_ticks = sorted(direct_ticks + [int(x["commit_tick"]) for x in accepted])
    gap_stats = compute_gap_stats(final_ticks)
    expected_total_keyframes = len(direct_ticks) + len(accepted)
    expected_density = 100.0 * expected_total_keyframes / max(total_frames, 1)
    retained_high_value = sum(
        1
        for x in accepted
        if (x.get("V_t", 0.0) >= 0.55 and x.get("Q_t", 0.0) >= 0.10)
    )
    retained_low_value = sum(
        1
        for x in accepted
        if (x.get("V_t", 1.0) < 0.35 or x.get("Q_t", 1.0) < 0.20)
    )

    return {
        "mode": mode,
        "accepted": accepted,
        "held": held,
        "rejected": rejected,
        "timeline_rows": timeline_rows,
        "expected_recovery_commit_count": len(accepted),
        "expected_total_keyframes": expected_total_keyframes,
        "expected_keyframes_per_100_frames": expected_density,
        "expected_recovery_fraction": len(accepted) / max(expected_total_keyframes, 1),
        "expected_main_chain_gap_stats": gap_stats,
        "dropped_or_held_recovery_count": len(held),
        "retained_high_value_recovery_count": retained_high_value,
        "retained_low_value_recovery_count": retained_low_value,
        "expected_optimization_dilution_risk": expected_density > baseline_density_per_100 * 1.6,
        "expected_under_support_risk": gap_stats["p90"] > 3.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--diagnosis_dir", default="")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stability = read_json(in_dir / "full_engine_stability_audit.json")
    key_timeline = read_csv(in_dir / "keyframe_timeline.csv")
    true_trace = read_csv(in_dir / "true_recovery_commit_trace.csv")
    score_rows = read_csv(in_dir / "engine_risk_value_recovery_score_table.csv")
    main_gap_rows = read_csv(in_dir / "main_chain_gap_timeline.csv")

    contrib_map: dict[int, dict[str, Any]] = {}
    if args.diagnosis_dir:
        contrib_path = Path(args.diagnosis_dir).resolve() / "recovery_commit_quality_contribution.csv"
        for r in read_csv(contrib_path):
            try:
                contrib_map[int(float(r.get("source_frame_id", "0") or 0))] = r
            except Exception:
                pass

    score_map: dict[int, dict[str, Any]] = {}
    for r in score_rows:
        try:
            fid = int(float(r.get("frame_id", "0") or 0))
            score_map[fid] = {
                "R_t": float(r.get("R_t", 0.0) or 0.0),
                "V_t": float(r.get("V_t", 0.0) or 0.0),
                "Q_t": float(r.get("Q_t", 0.0) or 0.0),
            }
        except Exception:
            continue

    direct_ticks = sorted(
        int(float(r.get("tick", "0") or 0))
        for r in key_timeline
        if str(r.get("action_type", "")) == "direct_admit"
    )

    recovery_candidates: list[dict[str, Any]] = []
    for r in true_trace:
        try:
            sid = int(float(r.get("source_frame_id", "0") or 0))
            tick = int(float(r.get("recovery_attempt_tick", "0") or 0))
            scores = score_map.get(sid, {})
            contrib = contrib_map.get(sid, {})
            recovery_candidates.append(
                {
                    "source_frame_id": sid,
                    "source_input_index": int(float(r.get("source_input_index", sid) or sid)),
                    "commit_tick": tick,
                    "commit_delay": tick - int(float(r.get("source_input_index", sid) or sid)),
                    "R_t": float(scores.get("R_t", 0.0)),
                    "V_t": float(scores.get("V_t", 0.0)),
                    "Q_t": float(scores.get("Q_t", 0.0)),
                    "pnp_inliers": contrib.get("pnp_inliers"),
                    "miniba_inliers": contrib.get("miniba_inliers"),
                    "suspected_positive_or_negative_contribution": contrib.get(
                        "suspected_positive_or_negative_contribution", ""
                    ),
                }
            )
        except Exception:
            continue

    total_frames = int(stability.get("processed_frame_count", 1194) or 1194)
    baseline_density_per_100 = 100.0 * 343.0 / max(total_frames, 1)

    modes = [
        "current_no_control",
        "conservative_window_control",
        "adaptive_gap_density_control",
    ]
    sims = [
        simulate_mode(
            m,
            direct_ticks=direct_ticks,
            recovery_candidates=recovery_candidates,
            total_frames=total_frames,
            baseline_density_per_100=baseline_density_per_100,
            window_size=30,
            max_per_window=5,
        )
        for m in modes
    ]

    sim_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    density_rows: list[dict[str, Any]] = []
    expected_keyframes: dict[str, Any] = {"baseline_target_range": [410, 550], "modes": {}}
    gap_rows: list[dict[str, Any]] = []
    for sim in sims:
        sim_rows.extend(
            [
                {
                    "mode": sim["mode"],
                    "source_frame_id": c["source_frame_id"],
                    "commit_tick": c["commit_tick"],
                    "decision": "commit",
                    "R_t": c.get("R_t"),
                    "V_t": c.get("V_t"),
                    "Q_t": c.get("Q_t"),
                    "commit_delay": c.get("commit_delay"),
                }
                for c in sim["accepted"]
            ]
            + [
                {
                    "mode": sim["mode"],
                    "source_frame_id": c["source_frame_id"],
                    "commit_tick": c["commit_tick"],
                    "decision": "hold",
                    "R_t": c.get("R_t"),
                    "V_t": c.get("V_t"),
                    "Q_t": c.get("Q_t"),
                    "commit_delay": c.get("commit_delay"),
                }
                for c in sim["held"]
            ]
        )
        timeline_rows.extend(sim["timeline_rows"])
        density_rows.append(
            {
                "mode": sim["mode"],
                "expected_recovery_commit_count": sim["expected_recovery_commit_count"],
                "expected_total_keyframes": sim["expected_total_keyframes"],
                "expected_keyframes_per_100_frames": sim["expected_keyframes_per_100_frames"],
                "expected_recovery_fraction": sim["expected_recovery_fraction"],
                "expected_optimization_dilution_risk": sim["expected_optimization_dilution_risk"],
                "expected_under_support_risk": sim["expected_under_support_risk"],
            }
        )
        expected_keyframes["modes"][sim["mode"]] = {
            "expected_total_keyframes": sim["expected_total_keyframes"],
            "expected_recovery_commit_count": sim["expected_recovery_commit_count"],
            "expected_keyframes_per_100_frames": sim["expected_keyframes_per_100_frames"],
        }
        gap_rows.append(
            {
                "mode": sim["mode"],
                "expected_main_chain_gap_p50": sim["expected_main_chain_gap_stats"]["p50"],
                "expected_main_chain_gap_p75": sim["expected_main_chain_gap_stats"]["p75"],
                "expected_main_chain_gap_p90": sim["expected_main_chain_gap_stats"]["p90"],
                "expected_main_chain_gap_max": sim["expected_main_chain_gap_stats"]["max"],
            }
        )

    write_csv(
        out_dir / "recovery_commit_control_offline_simulation.csv",
        sim_rows,
        [
            "mode",
            "source_frame_id",
            "commit_tick",
            "decision",
            "R_t",
            "V_t",
            "Q_t",
            "commit_delay",
        ],
    )
    write_csv(
        out_dir / "recovery_commit_control_timeline.csv",
        timeline_rows,
        [
            "mode",
            "tick",
            "recent_recovery_in_window",
            "accepted_recovery_count_so_far",
            "held_recovery_count_so_far",
            "current_keyframe_density_per_100",
            "current_main_chain_gap_p90",
            "decision",
            "decision_reason",
            "V_t",
            "Q_t",
        ],
    )
    write_csv(
        out_dir / "recovery_commit_density_comparison.csv",
        density_rows,
        list(density_rows[0].keys()) if density_rows else ["mode"],
    )
    write_json(out_dir / "expected_keyframe_count_after_control.json", expected_keyframes)
    write_csv(
        out_dir / "expected_main_chain_gap_after_control.csv",
        gap_rows,
        list(gap_rows[0].keys()) if gap_rows else ["mode"],
    )

    summary = {
        "input_total_frames": total_frames,
        "baseline_density_per_100": baseline_density_per_100,
        "current_main_chain_gap_p90": float(
            mean([float(r.get("gap", "0") or 0) for r in main_gap_rows]) if main_gap_rows else 0.0
        ),
        "modes": [
            {
                "mode": sim["mode"],
                "expected_recovery_commit_count": sim["expected_recovery_commit_count"],
                "expected_total_keyframes": sim["expected_total_keyframes"],
                "expected_keyframes_per_100_frames": sim["expected_keyframes_per_100_frames"],
                "expected_recovery_fraction": sim["expected_recovery_fraction"],
                "expected_main_chain_gap_p50": sim["expected_main_chain_gap_stats"]["p50"],
                "expected_main_chain_gap_p75": sim["expected_main_chain_gap_stats"]["p75"],
                "expected_main_chain_gap_p90": sim["expected_main_chain_gap_stats"]["p90"],
                "expected_main_chain_gap_max": sim["expected_main_chain_gap_stats"]["max"],
                "dropped_or_held_recovery_count": sim["dropped_or_held_recovery_count"],
                "retained_high_value_recovery_count": sim["retained_high_value_recovery_count"],
                "retained_low_value_recovery_count": sim["retained_low_value_recovery_count"],
                "expected_optimization_dilution_risk": sim["expected_optimization_dilution_risk"],
                "expected_under_support_risk": sim["expected_under_support_risk"],
            }
            for sim in sims
        ],
    }
    write_json(out_dir / "recovery_commit_control_offline_summary.json", summary)

    preferred = None
    for sim in sims:
        kf = sim["expected_total_keyframes"]
        gap_p90 = sim["expected_main_chain_gap_stats"]["p90"]
        if 410 <= kf <= 550 and gap_p90 <= 3.5:
            preferred = sim["mode"]
            break
    if preferred is None:
        preferred = "conservative_window_control"

    ready = {
        "ready_for_controlled_short_run": True,
        "preferred_control_mode": "adaptive"
        if preferred == "adaptive_gap_density_control"
        else "conservative",
        "preferred_simulation_mode": preferred,
    }
    write_json(out_dir / "ready_for_controlled_short_run.json", ready)

    report_lines = [
        "# recovery commit control preflight",
        "",
        f"- input_total_frames: {total_frames}",
        f"- baseline_density_per_100: {baseline_density_per_100:.2f}",
        f"- preferred_simulation_mode: {preferred}",
        f"- preferred_runtime_mode: {ready['preferred_control_mode']}",
        "",
        "## mode summary",
    ]
    for m in summary["modes"]:
        report_lines.append(
            f"- {m['mode']}: recovery={m['expected_recovery_commit_count']}, "
            f"keyframes={m['expected_total_keyframes']}, "
            f"density={m['expected_keyframes_per_100_frames']:.2f}, "
            f"gap_p90={m['expected_main_chain_gap_p90']:.2f}"
        )
    report_lines.append("")
    report_lines.append(
        "- 结论：控制层仅作用于 recovery_success 后 commit eligibility/rate，不改变 R/V/Q 与 tau。"
    )
    (out_dir / "recovery_commit_control_preflight_report.md").write_text(
        "\n".join(report_lines).rstrip() + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
