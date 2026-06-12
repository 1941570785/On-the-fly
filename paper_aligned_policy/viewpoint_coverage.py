from __future__ import annotations

import math
from typing import Any

import torch


def _as_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach()
    try:
        return torch.as_tensor(value)
    except Exception:
        return None


def _rotation_matrix(value: Any) -> torch.Tensor | None:
    tensor = _as_tensor(value)
    if tensor is None or tensor.numel() < 9:
        return None
    if tensor.ndim == 2 and tensor.shape[0] >= 3 and tensor.shape[1] >= 3:
        return tensor[:3, :3].float().cpu()
    if tensor.ndim == 1 and tensor.numel() >= 9:
        return tensor[:9].reshape(3, 3).float().cpu()
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def rotation_degrees_between(a: Any, b: Any) -> float:
    """Return relative rotation angle in degrees for two Rt/R matrices."""
    r_a = _rotation_matrix(a)
    r_b = _rotation_matrix(b)
    if r_a is None or r_b is None:
        return 0.0
    rel = r_b.transpose(0, 1).matmul(r_a)
    trace = float(torch.trace(rel).item())
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(math.degrees(math.acos(cos_theta)))


def _grid_cell_ids(kpts: Any, width: int, height: int, grid_size: int = 4) -> list[int]:
    tensor = _as_tensor(kpts)
    if tensor is None or tensor.numel() == 0:
        return []
    if tensor.ndim != 2 or tensor.shape[1] < 2:
        return []
    grid = max(1, int(grid_size))
    width = max(1, int(width))
    height = max(1, int(height))
    pts = tensor[:, :2].float().cpu()
    xs = torch.clamp((pts[:, 0] / float(width) * grid).floor().long(), 0, grid - 1)
    ys = torch.clamp((pts[:, 1] / float(height) * grid).floor().long(), 0, grid - 1)
    return (ys * grid + xs).tolist()


def grid_coverage(kpts: Any, width: int, height: int, grid_size: int = 4) -> float:
    cells = _grid_cell_ids(kpts, width, height, grid_size)
    if not cells:
        return 0.0
    total_cells = max(1, int(grid_size) * int(grid_size))
    return float(len(set(cells)) / total_cells)


def grid_entropy(kpts: Any, width: int, height: int, grid_size: int = 4) -> float:
    cells = _grid_cell_ids(kpts, width, height, grid_size)
    if not cells:
        return 0.0
    counts: dict[int, int] = {}
    for cell in cells:
        counts[int(cell)] = counts.get(int(cell), 0) + 1
    total = float(len(cells))
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        if p > 0:
            entropy -= p * math.log(p)
    max_entropy = math.log(max(2, int(grid_size) * int(grid_size)))
    return _clamp01(entropy / max_entropy)


def _history_frame_and_pose(item: Any) -> tuple[int, Any] | None:
    if isinstance(item, dict):
        frame_id = item.get("frame_id", item.get("source_frame_id", -1))
        pose = item.get("Rt", item.get("pose", item.get("current_Rt")))
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        frame_id, pose = item[0], item[1]
    else:
        return None
    try:
        frame_id_int = int(frame_id)
    except Exception:
        return None
    if _rotation_matrix(pose) is None:
        return None
    return frame_id_int, pose


def windowed_rotation_degrees(
    *,
    current_Rt: Any,
    pose_history: Any,
    current_frame_id: int,
    windows: tuple[int, ...] = (20, 50, 100),
) -> dict[str, Any]:
    history: list[tuple[int, Any]] = []
    for item in pose_history or []:
        parsed = _history_frame_and_pose(item)
        if parsed is None:
            continue
        frame_id, pose = parsed
        if frame_id < int(current_frame_id):
            history.append((frame_id, pose))

    out: dict[str, Any] = {}
    max_rot = 0.0
    max_window = 0
    max_source = -1
    for window in windows:
        window = max(1, int(window))
        target_frame = int(current_frame_id) - window
        eligible = [(fid, pose) for fid, pose in history if fid <= target_frame]
        if eligible:
            source_frame, source_pose = max(eligible, key=lambda row: row[0])
            rotation = rotation_degrees_between(current_Rt, source_pose)
        else:
            source_frame = -1
            rotation = 0.0
        out[f"viewpoint_rotation_deg_window_{window}"] = float(rotation)
        out[f"viewpoint_rotation_window_source_{window}"] = int(source_frame)
        if rotation > max_rot:
            max_rot = float(rotation)
            max_window = int(window)
            max_source = int(source_frame)

    out["viewpoint_rotation_deg_window_max"] = float(max_rot)
    out["viewpoint_rotation_window_max_size"] = int(max_window)
    out["viewpoint_rotation_window_max_source"] = int(max_source)
    return out


def build_viewpoint_coverage_event(
    *,
    frame_id: int,
    current_Rt: Any,
    last_keyframe_Rt: Any = None,
    active_anchor_Rt: Any = None,
    inlier_kpts: Any = None,
    image_width: int = 1,
    image_height: int = 1,
    pose_debug: dict[str, Any] | None = None,
    active_anchor_keyframe_count: int = 0,
    selected_reference_count: int = 0,
    grid_size: int = 4,
    pose_history: Any = None,
    rotation_windows: tuple[int, ...] = (20, 50, 100),
) -> dict[str, Any]:
    pose_debug = dict(pose_debug or {})
    kpt_count = len(_grid_cell_ids(inlier_kpts, image_width, image_height, grid_size))
    coverage = grid_coverage(inlier_kpts, image_width, image_height, grid_size)
    entropy = grid_entropy(inlier_kpts, image_width, image_height, grid_size)
    support_concentration = 1.0 - entropy
    rot_last = rotation_degrees_between(current_Rt, last_keyframe_Rt)
    rot_anchor = rotation_degrees_between(current_Rt, active_anchor_Rt)
    windowed_rotation = windowed_rotation_degrees(
        current_Rt=current_Rt,
        pose_history=pose_history,
        current_frame_id=int(frame_id),
        windows=rotation_windows,
    )
    windowed_rotation_max = float(
        windowed_rotation.get("viewpoint_rotation_deg_window_max", 0.0) or 0.0
    )
    match_total = max(
        _safe_int(pose_debug.get("match_count_total", 0)),
        _safe_int(pose_debug.get("num_2d3d_correspondences", 0)),
        kpt_count,
        1,
    )
    pose_inliers = max(
        _safe_int(pose_debug.get("num_miniba_inliers", 0)),
        _safe_int(pose_debug.get("num_pnp_inliers", 0)),
    )
    inlier_ratio = _clamp01(float(pose_inliers) / float(match_total))
    reference_health = _clamp01(float(selected_reference_count) / 4.0)
    anchor_size_health = _clamp01(float(active_anchor_keyframe_count) / 8.0)
    anchor_health = _clamp01(
        0.55 * inlier_ratio + 0.25 * reference_health + 0.20 * anchor_size_health
    )
    rotation_score = _clamp01(max(rot_last, rot_anchor, windowed_rotation_max) / 90.0)
    new_view_event_score = _clamp01(
        0.45 * rotation_score
        + 0.20 * (1.0 - anchor_health)
        + 0.20 * (1.0 - coverage)
        + 0.15 * support_concentration
    )
    return {
        "frame_id": int(frame_id),
        "viewpoint_rotation_deg_to_last_keyframe": float(rot_last),
        "viewpoint_rotation_deg_to_active_anchor": float(rot_anchor),
        **windowed_rotation,
        "inlier_grid_coverage": float(coverage),
        "inlier_grid_entropy": float(entropy),
        "support_concentration": float(support_concentration),
        "anchor_health_score": float(anchor_health),
        "new_view_event_score": float(new_view_event_score),
        "inlier_keypoint_count": int(kpt_count),
        "pose_inlier_ratio": float(inlier_ratio),
        "active_anchor_keyframe_count": int(active_anchor_keyframe_count),
        "selected_reference_count": int(selected_reference_count),
        "viewpoint_grid_size": int(grid_size),
    }
