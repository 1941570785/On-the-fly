from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

try:
    from paper_aligned_policy.config import (
        COUPLED_INNOVATION_MODE,
        COUPLED_RUNTIME_MODE,
        CoupledInnovationConfig,
    )
except Exception:  # pragma: no cover - keeps this utility importable in partial tooling contexts.
    COUPLED_INNOVATION_MODE = "on_the_fly_innovation_v1"
    COUPLED_RUNTIME_MODE = "paper_aligned_semantic_v1"
    CoupledInnovationConfig = None  # type: ignore[assignment]


OFFLINE_STAGE_METRIC_FIELDS = [
    "PSNR",
    "SSIM",
    "LPIPS",
    "absolute_relative_translation_error",
    "absolute_relative_rotation_error",
    "keyframe_gap",
    "direct_admit_density",
    "defer_recoverable_count",
    "recovery_attempt_count",
    "true_source_materialized_count",
    "support_2d3d",
    "pnp_inliers",
    "miniba_inliers",
    "risk_bucket",
    "recoverability_bucket",
]

QUALITY_ALIASES = {
    "psnr": ("psnr", "PSNR"),
    "ssim": ("ssim", "SSIM"),
    "lpips": ("lpips", "LPIPS"),
    "absolute_relative_translation_error": (
        "absolute_relative_translation_error",
        "abs_rel_translation_error",
        "translation_error",
        "t",
    ),
    "absolute_relative_rotation_error": (
        "absolute_relative_rotation_error",
        "abs_rel_rotation_error",
        "rotation_error_deg",
        "R_deg",
    ),
}

QUALITY_DECISION_FIELD_NAMES = {
    alias
    for aliases in QUALITY_ALIASES.values()
    for alias in aliases
}


def build_stage_metric_evaluation(
    trace: dict[str, Any],
    *,
    lifecycle_rows: list[dict[str, Any]] | None = None,
    run_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build offline stage metrics from runtime trace and lifecycle rows.

    The runtime trace owns process decisions; lifecycle/run-quality artifacts own
    PSNR/SSIM/LPIPS and absolute pose errors. This keeps stage proof metrics out
    of online admission decisions.
    """

    lifecycle_by_frame = _index_lifecycle_rows(lifecycle_rows or [])
    events = list(trace.get("events") or [])
    frame_rows: list[dict[str, Any]] = []

    for index, event in enumerate(events, start=1):
        frame_id = as_int(event.get("frame_id"), index)
        action = str(event.get("action") or "unknown")
        lifecycle = lifecycle_by_frame.get(frame_id, {})
        frame_rows.append(
            _build_process_frame_row(
                stage_name=action,
                frame_id=frame_id,
                event=event,
                lifecycle=lifecycle,
            )
        )

    recovery_events = list(trace.get("recovery_pose_path_events") or [])
    for event in recovery_events:
        source_frame_id = as_int(event.get("source_frame_id"), as_int(event.get("frame_id"), 0))
        lifecycle = lifecycle_by_frame.get(source_frame_id, {})
        frame_rows.append(
            _build_process_frame_row(
                stage_name="recovery_attempt",
                frame_id=source_frame_id,
                event=event,
                lifecycle=lifecycle,
                extra={
                    "current_frame_id": as_int(event.get("current_frame_id"), 0),
                    "recovery_attempt_count": 1,
                    "pose_attempt_count": 1,
                    "pose_success_count": 1 if as_bool(event.get("pnp_success")) else 0,
                },
            )
        )

    materialization_events = list(trace.get("recovery_commit_materialization_events") or [])
    materialized_events = [
        event
        for event in materialization_events
        if as_bool(event.get("materialized"))
        and not as_bool(event.get("source_equals_current_frame"))
    ]
    for event in materialized_events:
        source_frame_id = as_int(event.get("source_frame_id"), as_int(event.get("frame_id"), 0))
        lifecycle = lifecycle_by_frame.get(source_frame_id, {})
        frame_rows.append(
            _build_process_frame_row(
                stage_name="true_source_materialized",
                frame_id=source_frame_id,
                event=event,
                lifecycle=lifecycle,
                extra={
                    "current_frame_id": as_int(event.get("current_tick_frame_id"), 0),
                    "true_source_materialized_count": 1,
                    "final_keyframe_count": 1,
                },
            )
        )

    stage_rows = _aggregate_stage_rows(frame_rows)
    overall_summary = _build_overall_summary(
        events=events,
        frame_rows=frame_rows,
        recovery_attempt_count=len(recovery_events),
        true_source_materialized_count=len(materialized_events),
        run_quality=run_quality or {},
    )

    return {
        "metric_contract": _metric_contract(),
        "online_decision_metric_fields_seen": _online_decision_quality_fields(events),
        "overall_summary": overall_summary,
        "stage_rows": stage_rows,
        "frame_rows": frame_rows,
    }


def load_lifecycle_csv(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(fields)
    seen = set(all_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                all_fields.append(key)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _encode_cell(row.get(key)) for key in all_fields})


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


def _metric_contract() -> dict[str, list[str]]:
    if CoupledInnovationConfig is not None:
        return CoupledInnovationConfig(
            enabled=True,
            requested_mode=COUPLED_INNOVATION_MODE,
            runtime_mode=COUPLED_RUNTIME_MODE,
        ).stage_metric_contract()
    return {
        "online_quality_metric_fields": [],
        "offline_stage_metric_fields": list(OFFLINE_STAGE_METRIC_FIELDS),
    }


def _index_lifecycle_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame_id = as_int(row.get("frame_id"), 0)
        if frame_id:
            indexed[frame_id] = row
    return indexed


def _build_process_frame_row(
    *,
    stage_name: str,
    frame_id: int,
    event: dict[str, Any],
    lifecycle: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision_meta = event.get("decision_meta", {}) or {}
    row = {
        "stage_name": stage_name,
        "frame_id": frame_id,
        "action": str(event.get("action") or stage_name),
        "lifecycle_state": str(lifecycle.get("lifecycle_state") or lifecycle.get("final_state") or ""),
        "risk_bucket": str(lifecycle.get("state_risk_bucket") or lifecycle.get("risk_bucket") or ""),
        "recoverability_bucket": str(lifecycle.get("state_recoverability_bucket") or ""),
        "final_keyframe_count": 1 if as_bool(event.get("final_keyframe_incremented")) else 0,
        "pose_attempt_count": 1 if as_bool(event.get("pose_init_attempted")) else 0,
        "pose_success_count": 1 if as_bool(event.get("pose_init_success")) else 0,
        "valid_2d3d": _first_float(
            event,
            lifecycle,
            "num_2d3d_correspondences",
            "incremental_num_correspondences_2d3d",
            "num_matches",
        ),
        "pnp_inliers": _first_float(
            event,
            lifecycle,
            "num_pnp_inliers",
            "pnp_inliers",
            "incremental_pnp_inliers",
        ),
        "miniba_inliers": _first_float(
            event,
            lifecycle,
            "num_miniba_inliers",
            "miniba_inliers",
            "incremental_miniba_inliers",
        ),
        "psnr": _metric_value("psnr", event, lifecycle),
        "ssim": _metric_value("ssim", event, lifecycle),
        "lpips": _metric_value("lpips", event, lifecycle),
        "absolute_relative_translation_error": _metric_value(
            "absolute_relative_translation_error", event, lifecycle
        ),
        "absolute_relative_rotation_error": _metric_value(
            "absolute_relative_rotation_error", event, lifecycle
        ),
        "risk_score": as_float(decision_meta.get("R_t")),
        "visibility_score": as_float(decision_meta.get("V_t")),
        "quality_proxy_score": as_float(decision_meta.get("Q_t")),
        "recovery_attempt_count": 0,
        "true_source_materialized_count": 0,
    }
    if extra:
        row.update(extra)
    return row


def _aggregate_stage_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[str(row.get("stage_name") or "unknown")].append(row)

    rows: list[dict[str, Any]] = []
    for stage_name in sorted(grouped):
        rows.append(_aggregate_one_stage(stage_name, grouped[stage_name]))
    return rows


def _aggregate_one_stage(stage_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    lifecycle_counts = Counter(str(row.get("lifecycle_state") or "") for row in rows)
    risk_counts = Counter(str(row.get("risk_bucket") or "") for row in rows)
    lifecycle_counts.pop("", None)
    risk_counts.pop("", None)

    return {
        "stage_name": stage_name,
        "frame_count": len(rows),
        "final_keyframe_count": sum(as_int(row.get("final_keyframe_count"), 0) for row in rows),
        "pose_attempt_count": sum(as_int(row.get("pose_attempt_count"), 0) for row in rows),
        "pose_success_count": sum(as_int(row.get("pose_success_count"), 0) for row in rows),
        "recovery_attempt_count": sum(as_int(row.get("recovery_attempt_count"), 0) for row in rows),
        "true_source_materialized_count": sum(
            as_int(row.get("true_source_materialized_count"), 0) for row in rows
        ),
        "psnr_mean": _mean(_numbers(rows, "psnr")),
        "ssim_mean": _mean(_numbers(rows, "ssim")),
        "lpips_mean": _mean(_numbers(rows, "lpips")),
        "absolute_relative_translation_error_mean": _mean(
            _numbers(rows, "absolute_relative_translation_error")
        ),
        "absolute_relative_rotation_error_mean": _mean(
            _numbers(rows, "absolute_relative_rotation_error")
        ),
        "valid_2d3d_median": _median(_numbers(rows, "valid_2d3d")),
        "pnp_inliers_median": _median(_numbers(rows, "pnp_inliers")),
        "miniba_inliers_median": _median(_numbers(rows, "miniba_inliers")),
        "risk_score_mean": _mean(_numbers(rows, "risk_score")),
        "visibility_score_mean": _mean(_numbers(rows, "visibility_score")),
        "quality_proxy_score_mean": _mean(_numbers(rows, "quality_proxy_score")),
        "lifecycle_state_counts": dict(sorted(lifecycle_counts.items())),
        "risk_bucket_counts": dict(sorted(risk_counts.items())),
    }


def _build_overall_summary(
    *,
    events: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    recovery_attempt_count: int,
    true_source_materialized_count: int,
    run_quality: dict[str, Any],
) -> dict[str, Any]:
    final_keyframe_ids = [
        as_int(event.get("frame_id"), index)
        for index, event in enumerate(events, start=1)
        if as_bool(event.get("final_keyframe_incremented"))
    ]
    final_keyframe_ids = sorted(frame_id for frame_id in final_keyframe_ids if frame_id)
    gaps = [
        current - previous
        for previous, current in zip(final_keyframe_ids, final_keyframe_ids[1:])
        if current > previous
    ]
    stage_counts = Counter(str(row.get("stage_name") or "unknown") for row in frame_rows)

    summary = {
        "frame_count": len(events),
        "stage_row_count": len(frame_rows),
        "final_keyframe_count": len(final_keyframe_ids),
        "final_keyframe_ids": final_keyframe_ids,
        "keyframe_gap_mean": _mean([float(gap) for gap in gaps]),
        "keyframe_gap_median": _median([float(gap) for gap in gaps]),
        "keyframe_gap_p90": _percentile([float(gap) for gap in gaps], 0.90),
        "recovery_attempt_count": recovery_attempt_count,
        "true_source_materialized_count": true_source_materialized_count,
        "stage_counts": dict(sorted(stage_counts.items())),
        "psnr_mean": _mean(_numbers(frame_rows, "psnr")),
        "ssim_mean": _mean(_numbers(frame_rows, "ssim")),
        "lpips_mean": _mean(_numbers(frame_rows, "lpips")),
        "absolute_relative_translation_error_mean": _mean(
            _numbers(frame_rows, "absolute_relative_translation_error")
        ),
        "absolute_relative_rotation_error_mean": _mean(
            _numbers(frame_rows, "absolute_relative_rotation_error")
        ),
    }
    if run_quality:
        summary["run_quality"] = {
            key: value
            for key, value in sorted(run_quality.items())
            if key in QUALITY_DECISION_FIELD_NAMES
        }
    return summary


def _online_decision_quality_fields(events: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for event in events:
        _collect_quality_keys(event.get("decision_meta", {}) or {}, seen)
    return sorted(seen)


def _collect_quality_keys(value: Any, seen: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in QUALITY_DECISION_FIELD_NAMES:
                seen.add(str(key))
            _collect_quality_keys(child, seen)
    elif isinstance(value, list):
        for child in value:
            _collect_quality_keys(child, seen)


def _metric_value(name: str, event: dict[str, Any], lifecycle: dict[str, Any]) -> float | None:
    return _first_float(event, lifecycle, *QUALITY_ALIASES[name])


def _first_float(event: dict[str, Any], lifecycle: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = as_float(lifecycle.get(key))
        if value is not None:
            return value
    for key in keys:
        value = as_float(event.get(key))
        if value is not None:
            return value
    return None


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    index = max(0, min(len(ordered) - 1, index))
    return float(ordered[index])


def _encode_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value
