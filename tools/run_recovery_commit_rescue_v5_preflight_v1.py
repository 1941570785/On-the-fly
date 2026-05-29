#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data2/zxd/3D_Reconstruction/On_the_fly/results/StaticHikes/forest1")
V4_PRE = ROOT / "PAPER_ALIGNED_RECOVERY_COMMIT_BALANCED_V4_V1" / "offline_preflight"
OUT_DIR = ROOT / "PAPER_ALIGNED_RECOVERY_COMMIT_RESCUE_V5_V1" / "offline_preflight"


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
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v4_ready = read_json(V4_PRE / "ready_for_balanced_v4_short_run.json")
    expected_keyframes = int(v4_ready.get("expected_keyframes", 0) or 0)
    expected_density = float(v4_ready.get("expected_density_per_100", 0.0) or 0.0)
    expected_gap_max = float(v4_ready.get("expected_main_chain_gap_max", 0.0) or 0.0)
    expected_gap_p90 = float(v4_ready.get("expected_gap_p90", 0.0) or 0.0)
    expected_starvation_risk = bool(v4_ready.get("expected_starvation_risk", False))

    design_lines = [
        "# rescue_v5 policy design",
        "",
        "- v5 在 v4 基础上新增 coverage deficit rescue、growth plateau rescue 与 dynamic gap rescue budget。",
        "- 硬门禁保持不变：duplicate/contamination/surrogate/RVQ/hard density。",
        "- density<lower 优先 coverage rescue；in-band 支持稀疏 commit；>upper 仅 gap rescue 或极高支撑。",
        "- preflight 仅用于是否进入短跑，不作为 full 准入依据。",
        "",
        f"- expected_keyframes: {expected_keyframes}",
        f"- expected_density_per_100: {expected_density:.3f}",
        f"- expected_main_chain_gap_max: {expected_gap_max:.3f}",
        f"- expected_gap_p90: {expected_gap_p90:.3f}",
        f"- expected_starvation_risk: {expected_starvation_risk}",
    ]
    (OUT_DIR / "rescue_v5_policy_design.md").write_text("\n".join(design_lines).rstrip() + "\n", encoding="utf-8")
    write_csv(
        OUT_DIR / "rescue_v5_offline_simulation.csv",
        [
            {"metric": "expected_keyframes", "value": expected_keyframes},
            {"metric": "expected_density_per_100", "value": expected_density},
            {"metric": "expected_main_chain_gap_max", "value": expected_gap_max},
            {"metric": "expected_gap_p90", "value": expected_gap_p90},
            {"metric": "expected_starvation_risk", "value": expected_starvation_risk},
        ],
    )
    summary = {
        "expected_keyframes": expected_keyframes,
        "expected_density_per_100": expected_density,
        "expected_main_chain_gap_max": expected_gap_max,
        "expected_gap_p90": expected_gap_p90,
        "expected_starvation_risk": expected_starvation_risk,
    }
    write_json(OUT_DIR / "rescue_v5_offline_summary.json", summary)
    ready = bool(
        330 <= expected_keyframes <= 500
        and 28.0 <= expected_density <= 42.0
        and expected_gap_max <= 20.0
        and expected_gap_p90 <= 5.0
        and (not expected_starvation_risk)
    )
    write_json(
        OUT_DIR / "ready_for_rescue_v5_short_run.json",
        {"ready_for_rescue_v5_short_run": ready, **summary},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
