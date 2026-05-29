#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    short = root / "short_run"
    r300 = read_json(short / "live_short_300" / "engine_stability_audit.json")
    r500 = read_json(short / "live_short_500" / "engine_stability_audit.json")
    kf300 = int(r300.get("final_keyframe_count", 0) or 0)
    kf500 = int(r500.get("final_keyframe_count", 0) or 0)
    growth = kf500 - kf300
    ready = bool(
        bool(r300.get("short_run_stable", False))
        and bool(r500.get("short_run_stable", False))
        and kf300 >= 90
        and kf500 >= 140
        and growth >= 40
        and float(r500.get("keyframes_per_100_frames", 0.0) or 0.0) >= 28.0
        and float(r500.get("keyframes_per_100_frames", 999.0) or 999.0) <= 45.0
        and float(r500.get("main_chain_gap_p90", 999.0) or 999.0) <= 5.0
        and float(r500.get("main_chain_gap_p95", 999.0) or 999.0) <= 7.0
        and float(r500.get("main_chain_gap_max", 999.0) or 999.0) <= 20.0
        and (not bool(r500.get("starvation_risk", False)))
        and (not bool(r500.get("keyframe_growth_plateau", False)))
    )
    write_csv(short / "rescue_v5_short_run_comparison.csv", [{"run": "live_short_300", **r300}, {"run": "live_short_500", **r500}])
    write_json(
        short / "rescue_v5_short_run_comparison.json",
        {
            "live_short_300": r300,
            "live_short_500": r500,
            "keyframe_growth_300_to_500": growth,
            "ready_for_rescue_v5_full_run": ready,
        },
    )
    write_json(short / "ready_for_rescue_v5_full_run.json", {"ready_for_rescue_v5_full_run": ready})
    (short / "paper_aligned_rescue_v5_short_run_report.md").write_text(
        "\n".join(
            [
                "# rescue_v5 short run report",
                "",
                f"- short300 keyframes/density: {kf300}/{r300.get('keyframes_per_100_frames', None)}",
                f"- short500 keyframes/density: {kf500}/{r500.get('keyframes_per_100_frames', None)}",
                f"- growth(300->500): {growth}",
                f"- short500 gap p90/p95/max: {r500.get('main_chain_gap_p90', None)}/{r500.get('main_chain_gap_p95', None)}/{r500.get('main_chain_gap_max', None)}",
                f"- ready_for_rescue_v5_full_run: {ready}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
