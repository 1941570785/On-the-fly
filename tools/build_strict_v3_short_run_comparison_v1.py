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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    short = root / "short_run"
    r300 = read_json(short / "live_short_300" / "engine_stability_audit.json")
    r500 = read_json(short / "live_short_500" / "engine_stability_audit.json")

    rows = [
        {"run": "live_short_300", **r300},
        {"run": "live_short_500", **r500},
    ]
    write_csv(short / "strict_v3_short_run_comparison.csv", rows)
    comparison = {
        "live_short_300": r300,
        "live_short_500": r500,
        "ready_for_strict_v3_full_run": bool(
            r500.get("short_run_stable", False)
            and int(r500.get("final_keyframe_count", 0) or 0) >= 300
            and float(r500.get("keyframes_per_100_frames", 999.0) or 999.0) <= 42.0
            and float(r500.get("main_chain_gap_p90", 999.0) or 999.0) <= 5.0
            and float(r500.get("main_chain_gap_p95", 999.0) or 999.0) <= 7.0
            and float(r500.get("main_chain_gap_max", 999.0) or 999.0) <= 20.0
        ),
    }
    write_json(short / "strict_v3_short_run_comparison.json", comparison)
    write_json(
        short / "ready_for_strict_v3_full_run.json",
        {"ready_for_strict_v3_full_run": bool(comparison["ready_for_strict_v3_full_run"])},
    )
    report_lines = [
        "# strict_v3 short run report",
        "",
        f"- live_short_300 stable: {r300.get('short_run_stable', False)}",
        f"- live_short_300 keyframes_per_100: {r300.get('keyframes_per_100_frames', None)}",
        f"- live_short_300 gap p90/p95/max: {r300.get('main_chain_gap_p90', None)}/{r300.get('main_chain_gap_p95', None)}/{r300.get('main_chain_gap_max', None)}",
        f"- live_short_500 stable: {r500.get('short_run_stable', False)}",
        f"- live_short_500 keyframes_per_100: {r500.get('keyframes_per_100_frames', None)}",
        f"- live_short_500 gap p90/p95/max: {r500.get('main_chain_gap_p90', None)}/{r500.get('main_chain_gap_p95', None)}/{r500.get('main_chain_gap_max', None)}",
        f"- ready_for_strict_v3_full_run: {comparison['ready_for_strict_v3_full_run']}",
    ]
    (short / "paper_aligned_strict_v3_short_run_report.md").write_text(
        "\n".join(report_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
