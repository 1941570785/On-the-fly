#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path("/data2/zxd/3D_Reconstruction/On_the_fly")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault(
    "OTF_EXPERIMENT_LABEL", "pose-rep-active-memory-v27-utility-controller_all"
)
os.environ.setdefault("OTF_DIRECT_DENSITY_MODE", "pose_rep_active_memory_v27")

from tools import run_ssm_viewpoint_coverage_v1_all as runner  # noqa: E402


UTILITY_FIELDS = [
    "utility_tracking_only_count",
    "utility_representation_role_count",
    "utility_recovery_pressure_context_count",
    "utility_pose_reference_mean",
    "utility_representation_mean",
    "utility_coverage_gain_mean",
    "utility_recovery_gain_mean",
    "utility_compute_cost_mean",
    "utility_drift_risk_mean",
    "utility_total_mean",
    "utility_view_change_mean",
]


def _float_values(events: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for event in events:
        try:
            values.append(float(event.get(key, 0.0) or 0.0))
        except Exception:
            pass
    return values


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


_base_trace_stats = runner.trace_stats


def trace_stats(model_dir: Path) -> dict[str, Any]:
    row = _base_trace_stats(model_dir)
    path = model_dir / "semantic_trace.json"
    if not path.exists():
        return row
    with path.open("r", encoding="utf-8") as f:
        trace = json.load(f)
    events = trace.get("direct_density_control_events", [])
    row.update(
        {
            "utility_tracking_only_count": sum(
                e.get("utility_frame_role") == "tracking_only" for e in events
            ),
            "utility_representation_role_count": sum(
                bool(e.get("utility_representation_role")) for e in events
            ),
            "utility_recovery_pressure_context_count": sum(
                bool(e.get("utility_recovery_pressure_context")) for e in events
            ),
            "utility_pose_reference_mean": _mean(
                _float_values(events, "utility_pose_reference")
            ),
            "utility_representation_mean": _mean(
                _float_values(events, "utility_representation")
            ),
            "utility_coverage_gain_mean": _mean(
                _float_values(events, "utility_coverage_gain")
            ),
            "utility_recovery_gain_mean": _mean(
                _float_values(events, "utility_recovery_gain")
            ),
            "utility_compute_cost_mean": _mean(
                _float_values(events, "utility_compute_cost")
            ),
            "utility_drift_risk_mean": _mean(
                _float_values(events, "utility_drift_risk")
            ),
            "utility_total_mean": _mean(_float_values(events, "utility_total")),
            "utility_view_change_mean": _mean(
                _float_values(events, "utility_view_change")
            ),
        }
    )
    return row


runner.trace_stats = trace_stats
for field in reversed(UTILITY_FIELDS):
    if field not in runner.FIELDNAMES:
        runner.FIELDNAMES.insert(-2, field)

selected = os.environ.get("OTF_DATASETS", "").strip()
if selected:
    wanted = {name.strip() for name in selected.split(",") if name.strip()}
    runner.DATASETS = [row for row in runner.DATASETS if row[0] in wanted]


if __name__ == "__main__":
    raise SystemExit(runner.main())
