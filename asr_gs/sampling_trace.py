from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def effective_bernoulli_probability(value: Any) -> Any:
    """Return the probability represented by a raw Bernoulli intensity."""

    if hasattr(value, "clamp"):
        return value.clamp(0.0, 1.0)
    return np.clip(np.asarray(value), 0.0, 1.0)


def sampling_trace_matches(
    *,
    target: str,
    frame_name: str,
    applied: bool,
) -> bool:
    normalized = str(target or "").strip()
    if not normalized:
        return False
    if normalized == "all":
        return True
    if normalized == "applied":
        return bool(applied)
    return Path(normalized).name == Path(frame_name).name


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return _json_safe(_as_numpy(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_sampling_trace(
    *,
    output_root: Path,
    target: str,
    frame_name: str,
    base_probability: Any,
    final_probability: Any,
    sample_mask: Any,
    debug: dict[str, Any],
) -> Path | None:
    applied = bool(debug.get("applied", False))
    if not sampling_trace_matches(
        target=target,
        frame_name=frame_name,
        applied=applied,
    ):
        return None

    base = _as_numpy(base_probability).astype(np.float32, copy=False)
    final = _as_numpy(final_probability).astype(np.float32, copy=False)
    sampled = _as_numpy(sample_mask).astype(bool, copy=False)
    if base.shape != final.shape or base.shape != sampled.shape:
        raise ValueError("sampling trace arrays must have identical shapes")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(final)):
        raise ValueError("sampling probabilities must be finite")
    if float(final.min(initial=0.0)) < 0.0 or float(
        final.max(initial=0.0)
    ) > 1.0:
        raise ValueError("final Bernoulli probabilities must lie in [0, 1]")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(frame_name).stem
    archive_path = output_root / f"{stem}.npz"
    np.savez_compressed(
        archive_path,
        base_probability=base,
        final_probability=final,
        sample_mask=sampled,
    )
    metadata = {
        "frame_name": Path(frame_name).name,
        "shape": list(base.shape),
        "base_expected_samples": float(base.sum(dtype=np.float64)),
        "final_expected_samples": float(final.sum(dtype=np.float64)),
        "realized_samples": int(sampled.sum()),
        "debug": _json_safe(debug),
    }
    archive_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return archive_path
