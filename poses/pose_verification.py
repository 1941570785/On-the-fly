from __future__ import annotations

import math
from typing import Any

import torch


def deterministic_pose_sampling_seed(frame_id: int, *, stream: int = 0) -> int:
    """Return a stable, frame-specific seed for an isolated pose RNG stream."""
    modulus = (1 << 63) - 1
    return int(
        (
            0x9E3779B1
            + 1_000_003 * int(frame_id)
            + 9_176 * int(stream)
        )
        % modulus
    )


def async_pose_update_enabled(
    mode: str,
    *,
    run_until_interrupt: bool,
    iteration: int,
    base_iterations: int,
) -> bool:
    """Keep opportunistic asynchronous steps from changing verified poses."""
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode == "off":
        return True
    if normalized_mode != "fixed_joint_budget_v1":
        raise ValueError(f"unsupported async pose protection mode: {mode}")
    if not bool(run_until_interrupt):
        return True
    return int(iteration) < max(0, int(base_iterations))


def decide_failed_verification_reference_quarantine(
    risk_event: dict[str, Any] | None,
    *,
    policy: str,
    risk_threshold: float,
    frame_id: int,
    last_quarantine_frame_id: int,
    cooldown_frames: int,
) -> dict[str, Any]:
    """Keep an unsafe verified pose out of future registration references."""
    event = dict(risk_event or {})
    normalized_policy = str(policy or "off").strip().lower()
    adaptive_threshold = float(event.get("adaptive_threshold", 0.0) or 0.0)
    effective_threshold = max(float(risk_threshold), adaptive_threshold)
    cooldown = max(0, int(cooldown_frames))
    cooldown_active = bool(
        int(last_quarantine_frame_id) >= 0
        and int(frame_id) - int(last_quarantine_frame_id) < cooldown
    )
    debug = {
        "policy": normalized_policy,
        "quarantine": False,
        "reason": "",
        "risk_score": float(event.get("risk_score", 0.0) or 0.0),
        "effective_threshold": float(effective_threshold),
        "cooldown_active": cooldown_active,
        "last_quarantine_frame_id": int(last_quarantine_frame_id),
    }
    if normalized_policy == "off":
        debug["reason"] = "policy_off"
    elif normalized_policy not in {
        "conservative_quarantine_v1",
        "conservative_high_risk_v2",
    }:
        raise ValueError(
            f"Unsupported pose verification reference policy: {policy}"
        )
    elif (
        normalized_policy == "conservative_quarantine_v1"
        and bool(event.get("verification_accepted", False))
    ):
        debug["reason"] = "candidate_accepted"
    elif not (
        bool(event.get("eligible", False))
        and bool(event.get("warmed_up", False))
        and bool(event.get("verification_trigger", False))
    ):
        debug["reason"] = "verification_not_triggered"
    elif not (
        bool(event.get("multi_signal_risk", False))
        or bool(event.get("severe_pose_risk", False))
    ):
        debug["reason"] = "insufficient_risk_evidence"
    elif debug["risk_score"] < effective_threshold:
        debug["reason"] = "risk_below_threshold"
    elif cooldown_active:
        debug["reason"] = "cooldown"
    else:
        debug["quarantine"] = True
        if bool(event.get("verification_accepted", False)):
            debug["reason"] = "verified_high_risk_reference"
        elif normalized_policy == "conservative_high_risk_v2":
            debug["reason"] = "failed_high_risk_reference"
        else:
            debug["reason"] = "failed_verification_high_risk"
    return debug


def sample_stable_pose_anchor_candidates(
    records: list[dict[str, Any]],
    *,
    base_reference_ids: set[int],
    current_frame_id: int,
    min_age_frames: int,
    min_support_count: int,
    max_risk_score: float,
    max_pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter and uniformly subsample causal, low-risk pose anchors."""
    base_ids = {int(value) for value in base_reference_ids}
    eligible: list[dict[str, Any]] = []
    rejected = {
        "base_reference": 0,
        "noncausal_or_too_recent": 0,
        "insufficient_support": 0,
        "unsafe": 0,
    }
    for record in records:
        keyframe_id = int(record.get("keyframe_id", -1))
        source_frame_id = int(record.get("source_frame_id", -1))
        if keyframe_id in base_ids:
            rejected["base_reference"] += 1
            continue
        if (
            source_frame_id < 0
            or source_frame_id >= int(current_frame_id)
            or int(current_frame_id) - source_frame_id < int(min_age_frames)
        ):
            rejected["noncausal_or_too_recent"] += 1
            continue
        if int(record.get("support_count", 0) or 0) < int(min_support_count):
            rejected["insufficient_support"] += 1
            continue
        if (
            bool(record.get("isolated", False))
            or bool(record.get("quarantined", False))
            or float(record.get("risk_score", 0.0) or 0.0)
            > float(max_risk_score)
        ):
            rejected["unsafe"] += 1
            continue
        eligible.append(record)

    eligible.sort(
        key=lambda record: (
            int(record.get("source_frame_id", -1)),
            int(record.get("keyframe_id", -1)),
        )
    )
    limit = max(0, int(max_pool_size))
    sampled = eligible
    if limit == 0:
        sampled = []
    elif len(eligible) > limit:
        if limit == 1:
            positions = [0]
        else:
            positions = [
                round(index * (len(eligible) - 1) / (limit - 1))
                for index in range(limit)
            ]
        sampled = [eligible[position] for position in positions]
    return sampled, {
        "candidate_count": len(records),
        "eligible_count": len(eligible),
        "sampled_count": len(sampled),
        **{f"rejected_{key}": value for key, value in rejected.items()},
    }


def rank_stable_pose_anchor_records(
    records: list[dict[str, Any]],
    *,
    max_references: int,
    min_match_score: float,
    min_source_separation: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select strongly matched anchors while suppressing temporal duplicates."""
    above_threshold = [
        record
        for record in records
        if float(record.get("match_score", float("-inf")))
        >= float(min_match_score)
    ]
    above_threshold.sort(
        key=lambda record: (
            -float(record.get("match_score", float("-inf"))),
            float(record.get("risk_score", 0.0) or 0.0),
            int(record.get("source_frame_id", -1)),
            int(record.get("keyframe_id", -1)),
        )
    )
    selected: list[dict[str, Any]] = []
    diversity_rejected = 0
    separation = max(0, int(min_source_separation))
    for record in above_threshold:
        if len(selected) >= max(0, int(max_references)):
            break
        source_frame_id = int(record.get("source_frame_id", -1))
        if any(
            abs(source_frame_id - int(item.get("source_frame_id", -1)))
            < separation
            for item in selected
        ):
            diversity_rejected += 1
            continue
        selected.append(record)
    return selected, {
        "scored_count": len(records),
        "above_match_threshold_count": len(above_threshold),
        "selected_count": len(selected),
        "diversity_rejected_count": diversity_rejected,
    }


def resolve_stable_pose_anchor_probe_references(
    base_references: list[Any],
    anchor_references: list[Any],
    *,
    candidate_scope: str,
) -> tuple[list[Any], dict[str, Any]]:
    """Resolve an anchor-only probe with a safe combined fallback."""
    scope = str(candidate_scope or "combined_v1").strip().lower()
    if scope not in {"combined_v1", "anchor_only_v1"}:
        raise ValueError(f"Unsupported anchor candidate scope: {candidate_scope}")

    def unique(references: list[Any]) -> list[Any]:
        output: list[Any] = []
        seen: set[int] = set()
        for reference in references:
            reference_id = int(reference.index)
            if reference_id not in seen:
                output.append(reference)
                seen.add(reference_id)
        return output

    anchors = unique(list(anchor_references))
    combined = unique(list(base_references) + anchors)
    if scope == "anchor_only_v1" and len(anchors) >= 2:
        selected = anchors
        resolved_scope = "anchor_only_v1"
    else:
        selected = combined
        resolved_scope = (
            "combined_fallback"
            if scope == "anchor_only_v1"
            else "combined_v1"
        )
    return selected, {
        "requested_scope": scope,
        "resolved_scope": resolved_scope,
        "base_reference_count": len(unique(list(base_references))),
        "anchor_reference_count": len(anchors),
        "probe_reference_count": len(selected),
        "probe_reference_ids": [
            int(reference.index) for reference in selected
        ],
    }


def _pose_4x4(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = torch.eye(4, dtype=pose.dtype, device=pose.device)
        out[:3] = pose
        return out
    raise ValueError(f"Expected a 3x4 or 4x4 pose, got {tuple(pose.shape)}")


def geometry_anchor_pose(
    current_pose: torch.Tensor,
    info: dict[str, Any] | None,
) -> torch.Tensor:
    """Return the opt-in verified geometry anchor or the current pose."""
    current = _pose_4x4(current_pose)
    metadata = info or {}
    if (
        str(
            metadata.get(
                "_pose_verification_geometry_anchor_mode",
                "off",
            )
        )
        != "freeze_v1"
    ):
        return current.clone()
    raw_anchor = metadata.get("_pose_verification_geometry_anchor_Rt")
    if raw_anchor is None:
        return current.clone()
    try:
        anchor = torch.as_tensor(
            raw_anchor,
            dtype=current.dtype,
            device=current.device,
        )
        anchor = _pose_4x4(anchor)
    except (TypeError, ValueError):
        return current.clone()
    if not bool(torch.isfinite(anchor).all()):
        return current.clone()
    return anchor.clone()


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )


def _batch_skew(vectors: torch.Tensor) -> torch.Tensor:
    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def _so3_log(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if float(theta.abs().item()) < 1e-6:
        return 0.5 * vee
    sine = torch.sin(theta)
    if float(sine.abs().item()) < 1e-5:
        symmetric = 0.5 * (rotation + torch.eye(3, **{
            "dtype": rotation.dtype,
            "device": rotation.device,
        }))
        axis = torch.sqrt(torch.diagonal(symmetric).clamp_min(0.0))
        signs = torch.sign(
            torch.stack((rotation[2, 1], rotation[0, 2], rotation[1, 0]))
        )
        axis = axis * torch.where(signs == 0, torch.ones_like(signs), signs)
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-8)
        return theta * axis
    return theta * vee / (2.0 * sine)


def _so3_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(axis_angle)
    identity = torch.eye(
        3, dtype=axis_angle.dtype, device=axis_angle.device
    )
    if float(theta.item()) < 1e-6:
        skew = _skew(axis_angle)
        return identity + skew + 0.5 * (skew @ skew)
    axis = axis_angle / theta
    skew = _skew(axis)
    return (
        identity
        + torch.sin(theta) * skew
        + (1.0 - torch.cos(theta)) * (skew @ skew)
    )


def interpolate_world_to_camera_pose(
    initial_pose: torch.Tensor,
    candidate_pose: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Interpolate rotation and camera centre between world-to-camera poses."""
    initial = _pose_4x4(initial_pose)
    candidate = _pose_4x4(candidate_pose)
    amount = max(0.0, min(1.0, float(alpha)))
    if amount <= 0.0:
        return initial.clone()
    if amount >= 1.0:
        return candidate.clone()

    relative_rotation = candidate[:3, :3] @ initial[:3, :3].transpose(0, 1)
    rotation = _so3_exp(amount * _so3_log(relative_rotation)) @ initial[:3, :3]
    initial_centre = -(initial[:3, :3].transpose(0, 1) @ initial[:3, 3])
    candidate_centre = -(candidate[:3, :3].transpose(0, 1) @ candidate[:3, 3])
    centre = (1.0 - amount) * initial_centre + amount * candidate_centre

    output = torch.eye(4, dtype=initial.dtype, device=initial.device)
    output[:3, :3] = rotation
    output[:3, 3] = -(rotation @ centre)
    return output


def _pose_history_entries(
    pose_history: list[object] | None,
    *,
    like_pose: torch.Tensor,
) -> list[tuple[int, torch.Tensor]]:
    entries: list[tuple[int, torch.Tensor]] = []
    for item in pose_history or []:
        frame_id: int | None = None
        raw_pose: object | None = None
        if isinstance(item, dict):
            frame_id = int(item.get("frame_id", -1))
            raw_pose = item.get("Rt")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            frame_id = int(item[0])
            raw_pose = item[1]
        if raw_pose is None or frame_id is None:
            continue
        try:
            tensor = torch.as_tensor(
                raw_pose,
                dtype=like_pose.dtype,
                device=like_pose.device,
            )
            entries.append((frame_id, _pose_4x4(tensor)))
        except (TypeError, ValueError):
            continue
    return entries


def predict_constant_velocity_pose(
    pose_history: list[object] | None,
    *,
    current_frame_id: int,
    like_pose: torch.Tensor,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Predict a causal world-to-camera pose from the latest two poses."""
    entries = _pose_history_entries(pose_history, like_pose=like_pose)
    if len(entries) < 2:
        return None, {"available": False, "reason": "insufficient_history"}
    previous_id, previous = entries[-2]
    last_id, last = entries[-1]
    history_gap = max(last_id - previous_id, 1)
    current_gap = max(int(current_frame_id) - last_id, 1)
    gap_ratio = float(current_gap) / float(history_gap)

    previous_rotation_c2w = previous[:3, :3].transpose(0, 1)
    last_rotation_c2w = last[:3, :3].transpose(0, 1)
    previous_centre = -(
        previous_rotation_c2w @ previous[:3, 3]
    )
    last_centre = -(last_rotation_c2w @ last[:3, 3])
    predicted_centre = last_centre + gap_ratio * (
        last_centre - previous_centre
    )
    rotation_step = (
        previous_rotation_c2w.transpose(0, 1) @ last_rotation_c2w
    )
    predicted_rotation_c2w = last_rotation_c2w @ _so3_exp(
        gap_ratio * _so3_log(rotation_step)
    )
    predicted_rotation_w2c = predicted_rotation_c2w.transpose(0, 1)
    prediction = torch.eye(
        4, dtype=like_pose.dtype, device=like_pose.device
    )
    prediction[:3, :3] = predicted_rotation_w2c
    prediction[:3, 3] = -(
        predicted_rotation_w2c @ predicted_centre
    )
    return prediction, {
        "available": True,
        "history_frame_gap": history_gap,
        "current_frame_gap": current_gap,
        "gap_ratio": gap_ratio,
    }


def temporal_pose_consistency(
    pose: torch.Tensor,
    prediction: torch.Tensor,
    pose_history: list[object] | None,
) -> dict[str, float]:
    """Measure prediction residual normalized by recent motion scales."""
    entries = _pose_history_entries(pose_history, like_pose=pose)[-10:]
    translation_steps: list[float] = []
    rotation_steps: list[float] = []
    for (_, previous), (_, current) in zip(entries[:-1], entries[1:]):
        previous_centre = -(
            previous[:3, :3].transpose(0, 1) @ previous[:3, 3]
        )
        current_centre = -(
            current[:3, :3].transpose(0, 1) @ current[:3, 3]
        )
        translation_steps.append(
            float(torch.linalg.vector_norm(current_centre - previous_centre).item())
        )
        relative_rotation = (
            current[:3, :3] @ previous[:3, :3].transpose(0, 1)
        )
        rotation_steps.append(
            float(torch.linalg.vector_norm(_so3_log(relative_rotation)).item())
        )
    translation_scale = max(
        float(torch.tensor(translation_steps).median().item())
        if translation_steps
        else 0.0,
        1e-4,
    )
    rotation_scale = max(
        float(torch.tensor(rotation_steps).median().item())
        if rotation_steps
        else 0.0,
        math.radians(0.05),
    )
    pose_centre = -(pose[:3, :3].transpose(0, 1) @ pose[:3, 3])
    prediction_centre = -(
        prediction[:3, :3].transpose(0, 1) @ prediction[:3, 3]
    )
    translation_residual = float(
        torch.linalg.vector_norm(pose_centre - prediction_centre).item()
    )
    relative_rotation = (
        pose[:3, :3] @ prediction[:3, :3].transpose(0, 1)
    )
    rotation_residual = float(
        torch.linalg.vector_norm(_so3_log(relative_rotation)).item()
    )
    score = math.sqrt(
        (translation_residual / translation_scale) ** 2
        + (rotation_residual / rotation_scale) ** 2
    )
    return {
        "score": score,
        "translation_residual": translation_residual,
        "rotation_residual_rad": rotation_residual,
        "translation_scale": translation_scale,
        "rotation_scale_rad": rotation_scale,
    }


def compute_epipolar_sampson_errors(
    current_pose: torch.Tensor,
    current_uv: torch.Tensor,
    reference_uv: torch.Tensor,
    reference_poses: torch.Tensor,
    *,
    focal: torch.Tensor,
    centre: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return square-root Sampson errors in pixels for pose-conditioned matches."""
    count = int(current_uv.shape[0])
    if current_uv.shape != (count, 2) or reference_uv.shape != (count, 2):
        raise ValueError("Current and reference observations must have shape [N, 2]")
    if reference_poses.shape not in {(count, 3, 4), (count, 4, 4)}:
        raise ValueError("Reference poses must have shape [N, 3, 4] or [N, 4, 4]")

    current = _pose_4x4(current_pose)
    references = reference_poses.to(dtype=current.dtype, device=current.device)
    current_points = current_uv.to(dtype=current.dtype, device=current.device)
    reference_points = reference_uv.to(dtype=current.dtype, device=current.device)
    focal_value = focal.to(dtype=current.dtype, device=current.device).reshape(-1)[0]
    centre_value = centre.to(dtype=current.dtype, device=current.device).reshape(2)

    current_h = torch.cat(
        (
            (current_points - centre_value) / focal_value,
            torch.ones((count, 1), dtype=current.dtype, device=current.device),
        ),
        dim=-1,
    )
    reference_h = torch.cat(
        (
            (reference_points - centre_value) / focal_value,
            torch.ones((count, 1), dtype=current.dtype, device=current.device),
        ),
        dim=-1,
    )
    relative_rotation = (
        current[:3, :3].unsqueeze(0)
        @ references[:, :3, :3].transpose(-2, -1)
    )
    relative_translation = current[:3, 3].unsqueeze(0) - (
        relative_rotation @ references[:, :3, 3].unsqueeze(-1)
    ).squeeze(-1)
    essential = _batch_skew(relative_translation) @ relative_rotation
    epipolar_in_current = (
        essential @ reference_h.unsqueeze(-1)
    ).squeeze(-1)
    epipolar_in_reference = (
        essential.transpose(-2, -1) @ current_h.unsqueeze(-1)
    ).squeeze(-1)
    numerator = torch.sum(current_h * epipolar_in_current, dim=-1).abs()
    denominator = (
        epipolar_in_current[:, :2].square().sum(dim=-1)
        + epipolar_in_reference[:, :2].square().sum(dim=-1)
    )
    baseline = torch.linalg.vector_norm(relative_translation, dim=-1)
    valid = (
        torch.isfinite(numerator)
        & torch.isfinite(denominator)
        & (denominator > 1e-12)
        & (baseline > 1e-8)
        & torch.isfinite(focal_value)
        & (focal_value.abs() > 1e-8)
    )
    errors = numerator / denominator.clamp_min(1e-12).sqrt()
    errors = errors * focal_value.abs()
    errors = torch.where(valid, errors, torch.full_like(errors, float("inf")))
    return errors, valid


def estimate_opencv_pnp_candidates(
    points_3d: torch.Tensor,
    image_points: torch.Tensor,
    *,
    focal: torch.Tensor,
    centre: torch.Tensor,
    initial_pose: torch.Tensor,
    max_reprojection_error: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate deterministic robust PnP hypotheses using independent solvers."""
    count = int(points_3d.shape[0])
    if points_3d.shape != (count, 3) or image_points.shape != (count, 2):
        raise ValueError("PnP inputs must have shape [N, 3] and [N, 2]")
    debug: dict[str, Any] = {
        "input_count": count,
        "attempted_solvers": 0,
        "successful_candidates": 0,
        "solver_inlier_counts": {},
    }
    if count < 6:
        debug["reason"] = "insufficient_support"
        return [], debug

    try:
        import cv2
        import numpy as np
    except ImportError:
        debug["reason"] = "opencv_unavailable"
        return [], debug

    pose = _pose_4x4(initial_pose)
    object_points = (
        points_3d.detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    observations = (
        image_points.detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    focal_values = (
        torch.as_tensor(focal).detach().to(dtype=torch.float64, device="cpu")
        .reshape(-1)
        .numpy()
    )
    centre_values = (
        torch.as_tensor(centre).detach().to(dtype=torch.float64, device="cpu")
        .reshape(-1)
        .numpy()
    )
    fx = float(focal_values[0])
    fy = float(focal_values[1] if len(focal_values) > 1 else focal_values[0])
    camera_matrix = np.array(
        [
            [fx, 0.0, float(centre_values[0])],
            [0.0, fy, float(centre_values[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    initial_rotation = (
        pose[:3, :3].detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    initial_translation = (
        pose[:3, 3].detach().to(dtype=torch.float64, device="cpu").numpy()
        .reshape(3, 1)
    )
    initial_rvec, _ = cv2.Rodrigues(initial_rotation)
    solver_flags = (
        ("epnp", cv2.SOLVEPNP_EPNP),
        ("ap3p", cv2.SOLVEPNP_AP3P),
        ("iterative", cv2.SOLVEPNP_ITERATIVE),
    )
    output: list[dict[str, Any]] = []
    for solver_index, (solver_name, solver_flag) in enumerate(solver_flags):
        debug["attempted_solvers"] += 1
        try:
            cv2.setRNGSeed(1701 + solver_index)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                observations,
                camera_matrix,
                None,
                rvec=initial_rvec.copy(),
                tvec=initial_translation.copy(),
                useExtrinsicGuess=True,
                iterationsCount=200,
                reprojectionError=max(0.25, float(max_reprojection_error)),
                confidence=0.999,
                flags=solver_flag,
            )
        except cv2.error:
            continue
        inlier_count = 0 if inliers is None else int(len(inliers))
        debug["solver_inlier_counts"][solver_name] = inlier_count
        if not success or inlier_count < 6:
            continue
        inlier_indices = inliers.reshape(-1)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                observations[inlier_indices],
                camera_matrix,
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
        rotation, _ = cv2.Rodrigues(rvec)
        candidate = torch.eye(
            4, dtype=pose.dtype, device=pose.device
        )
        candidate[:3, :3] = torch.as_tensor(
            rotation, dtype=pose.dtype, device=pose.device
        )
        candidate[:3, 3] = torch.as_tensor(
            tvec.reshape(3), dtype=pose.dtype, device=pose.device
        )
        if not bool(torch.isfinite(candidate).all()):
            continue
        duplicate = any(
            pose_correction_magnitude(item["pose"], candidate)["translation"]
            < 1e-7
            and pose_correction_magnitude(item["pose"], candidate)[
                "rotation_deg"
            ]
            < 1e-5
            for item in output
        )
        if not duplicate:
            output.append(
                {
                    "name": f"opencv_{solver_name}",
                    "pose": candidate,
                    "inlier_count": inlier_count,
                }
            )
    debug["successful_candidates"] = len(output)
    debug["reason"] = "ok" if output else "all_solvers_failed"
    return output, debug


def select_pose_candidate_by_reprojection(
    candidates: list[dict[str, Any]],
    points_3d: torch.Tensor,
    image_points: torch.Tensor,
    *,
    focal: torch.Tensor,
    centre: torch.Tensor,
    max_reprojection_error: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select a PnP hypothesis using only deterministic geometric support."""
    threshold = max(0.25, float(max_reprojection_error))
    ranked: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        pose = candidate.get("pose")
        if not isinstance(pose, torch.Tensor):
            continue
        errors, valid = compute_reprojection_errors(
            pose,
            points_3d,
            image_points,
            focal=focal,
            centre=centre,
        )
        inlier_mask = valid & torch.isfinite(errors) & (errors <= threshold)
        inlier_count = int(inlier_mask.sum().item())
        if inlier_count < 4:
            continue
        inlier_errors = errors[inlier_mask]
        median = float(torch.median(inlier_errors).item())
        p90 = float(torch.quantile(inlier_errors, 0.90).item())
        name = str(candidate.get("name", f"candidate_{index:02d}"))
        selected = dict(candidate)
        selected["name"] = name
        selected["inlier_mask"] = inlier_mask
        row = {
            "name": name,
            "inlier_count": inlier_count,
            "median": median,
            "p90": p90,
        }
        rank = (-float(inlier_count), median, p90, float(index))
        ranked.append((rank, selected, row))
    ranked.sort(key=lambda item: item[0])
    debug = {
        "candidate_count": len(candidates),
        "valid_candidate_count": len(ranked),
        "candidates": [item[2] for item in ranked],
        "selected_name": ranked[0][1]["name"] if ranked else "",
    }
    return (ranked[0][1] if ranked else None), debug


def estimate_multireference_relative_pose_candidates(
    current_uv: torch.Tensor,
    reference_uv: torch.Tensor,
    reference_poses: torch.Tensor,
    reference_ids: torch.Tensor,
    *,
    focal: torch.Tensor,
    centre: torch.Tensor,
    initial_pose: torch.Tensor,
    max_epipolar_error: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover an absolute pose from multiple independent 2D-2D baselines."""
    count = int(current_uv.shape[0])
    if (
        current_uv.shape != (count, 2)
        or reference_uv.shape != (count, 2)
        or reference_ids.shape != (count,)
        or reference_poses.shape
        not in {(count, 3, 4), (count, 4, 4)}
    ):
        raise ValueError("Relative-pose evidence has inconsistent shapes")
    debug: dict[str, Any] = {
        "input_count": count,
        "reference_count": 0,
        "usable_references": 0,
        "reference_inliers": {},
    }
    if count < 16:
        debug["reason"] = "insufficient_support"
        return [], debug
    try:
        import cv2
        import numpy as np
    except ImportError:
        debug["reason"] = "opencv_unavailable"
        return [], debug

    pose = _pose_4x4(initial_pose)
    current_points = (
        current_uv.detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    reference_points = (
        reference_uv.detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    reference_pose_values = (
        reference_poses.detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    reference_id_values = reference_ids.detach().to(device="cpu").numpy()
    focal_values = (
        torch.as_tensor(focal).detach().to(dtype=torch.float64, device="cpu")
        .reshape(-1)
        .numpy()
    )
    centre_values = (
        torch.as_tensor(centre).detach().to(dtype=torch.float64, device="cpu")
        .reshape(-1)
        .numpy()
    )
    fx = float(focal_values[0])
    fy = float(focal_values[1] if len(focal_values) > 1 else focal_values[0])
    camera_matrix = np.array(
        [
            [fx, 0.0, float(centre_values[0])],
            [0.0, fy, float(centre_values[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    initial_rotation = (
        pose[:3, :3].detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    initial_translation = (
        pose[:3, 3].detach().to(dtype=torch.float64, device="cpu").numpy()
    )
    initial_centre = -(initial_rotation.T @ initial_translation)
    hypotheses: list[dict[str, Any]] = []
    unique_ids = sorted(
        int(value)
        for value in np.unique(reference_id_values)
        if int(value) >= 0
    )
    debug["reference_count"] = len(unique_ids)
    essential_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    for order, reference_id in enumerate(unique_ids):
        indices = np.flatnonzero(reference_id_values == reference_id)
        if len(indices) < 12:
            continue
        points_ref = reference_points[indices]
        points_cur = current_points[indices]
        try:
            cv2.setRNGSeed(3101 + order)
            essential, inlier_mask = cv2.findEssentialMat(
                points_ref,
                points_cur,
                camera_matrix,
                method=essential_method,
                prob=0.999,
                threshold=max(0.25, float(max_epipolar_error)),
            )
        except cv2.error:
            continue
        if essential is None:
            continue
        essential_matrices = [
            essential[row : row + 3]
            for row in range(0, int(essential.shape[0]), 3)
            if essential[row : row + 3].shape == (3, 3)
        ]
        best_recovery: tuple[int, Any, Any] | None = None
        for essential_matrix in essential_matrices:
            try:
                inlier_count, relative_rotation, relative_translation, _ = (
                    cv2.recoverPose(
                        essential_matrix,
                        points_ref,
                        points_cur,
                        camera_matrix,
                        mask=inlier_mask.copy(),
                    )
                )
            except cv2.error:
                continue
            if best_recovery is None or int(inlier_count) > best_recovery[0]:
                best_recovery = (
                    int(inlier_count),
                    relative_rotation,
                    relative_translation,
                )
        if best_recovery is None:
            continue
        inlier_count, relative_rotation, relative_translation = best_recovery
        debug["reference_inliers"][str(reference_id)] = inlier_count
        if inlier_count < max(12, int(math.ceil(0.20 * len(indices)))):
            continue
        reference_pose = reference_pose_values[indices[0]]
        reference_rotation = reference_pose[:3, :3]
        reference_translation = reference_pose[:3, 3]
        candidate_rotation = relative_rotation @ reference_rotation
        reference_centre = -(
            reference_rotation.T @ reference_translation
        )
        direction = -(
            candidate_rotation.T @ relative_translation.reshape(3)
        )
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm < 1e-8:
            continue
        direction = direction / direction_norm
        hypotheses.append(
            {
                "reference_id": reference_id,
                "rotation": candidate_rotation,
                "centre": reference_centre,
                "direction": direction,
                "weight": float(
                    inlier_count
                    * math.sqrt(inlier_count / max(len(indices), 1))
                ),
            }
        )
    debug["usable_references"] = len(hypotheses)
    if len(hypotheses) < 2:
        debug["reason"] = "insufficient_usable_references"
        return [], debug

    rotations = [item["rotation"] for item in hypotheses]
    weights = np.asarray(
        [float(item["weight"]) for item in hypotheses], dtype=np.float64
    )

    def rotation_distance(first: Any, second: Any) -> float:
        cosine = float(
            np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0)
        )
        return math.acos(cosine)

    pairwise = np.asarray(
        [
            [
                rotation_distance(first, second)
                for second in rotations
            ]
            for first in rotations
        ],
        dtype=np.float64,
    )
    medoid = int(np.argmin(pairwise @ weights))
    medoid_distances = pairwise[medoid]
    distance_median = float(np.median(medoid_distances))
    distance_mad = float(
        np.median(np.abs(medoid_distances - distance_median))
    )
    rotation_cutoff = max(
        math.radians(2.0),
        distance_median + 2.5 * 1.4826 * distance_mad,
    )
    keep = medoid_distances <= rotation_cutoff
    if int(np.count_nonzero(keep)) < 2:
        keep[:] = True
    hypotheses = [
        item for item, selected in zip(hypotheses, keep) if bool(selected)
    ]
    weights = np.asarray(
        [float(item["weight"]) for item in hypotheses], dtype=np.float64
    )
    rotation_sum = sum(
        weight * item["rotation"]
        for weight, item in zip(weights, hypotheses)
    )
    left, _, right = np.linalg.svd(rotation_sum)
    consensus_rotation = left @ right
    if float(np.linalg.det(consensus_rotation)) < 0.0:
        left[:, -1] *= -1.0
        consensus_rotation = left @ right

    ray_matrix = np.zeros((3, 3), dtype=np.float64)
    ray_vector = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for weight, item in zip(weights, hypotheses):
        direction = item["direction"]
        projector = identity - np.outer(direction, direction)
        ray_matrix += weight * projector
        ray_vector += weight * projector @ item["centre"]
    weight_sum = max(float(weights.sum()), 1e-8)
    output: list[dict[str, Any]] = []
    for prior_strength in (0.0, 0.02, 0.10):
        regularizer = prior_strength * weight_sum
        system = ray_matrix + regularizer * identity
        target = ray_vector + regularizer * initial_centre
        try:
            candidate_centre = np.linalg.solve(system, target)
        except np.linalg.LinAlgError:
            continue
        if not bool(np.isfinite(candidate_centre).all()):
            continue
        candidate = torch.eye(4, dtype=pose.dtype, device=pose.device)
        candidate[:3, :3] = torch.as_tensor(
            consensus_rotation, dtype=pose.dtype, device=pose.device
        )
        candidate[:3, 3] = torch.as_tensor(
            -(consensus_rotation @ candidate_centre),
            dtype=pose.dtype,
            device=pose.device,
        )
        output.append(
            {
                "name": f"relative_rays_prior_{prior_strength:.2f}",
                "pose": candidate,
                "inlier_count": int(sum(debug["reference_inliers"].values())),
            }
        )

    rotation_only = torch.eye(4, dtype=pose.dtype, device=pose.device)
    rotation_only[:3, :3] = torch.as_tensor(
        consensus_rotation, dtype=pose.dtype, device=pose.device
    )
    rotation_only[:3, 3] = torch.as_tensor(
        -(consensus_rotation @ initial_centre),
        dtype=pose.dtype,
        device=pose.device,
    )
    output.append(
        {
            "name": "relative_rotation_only",
            "pose": rotation_only,
            "inlier_count": int(sum(debug["reference_inliers"].values())),
        }
    )
    debug["consensus_references"] = len(hypotheses)
    debug["ray_condition_number"] = float(np.linalg.cond(ray_matrix))
    debug["successful_candidates"] = len(output)
    debug["reason"] = "ok"
    return output, debug


def select_balanced_correspondence_indices(
    errors: torch.Tensor,
    confidence: torch.Tensor,
    uv: torch.Tensor,
    reference_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    width: int,
    height: int,
    max_points: int,
    grid_rows: int = 4,
    grid_cols: int = 6,
    max_reference_fraction: float = 0.4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select deterministic spatially and reference-balanced correspondences."""
    count = int(errors.shape[0])
    if (
        errors.shape != (count,)
        or confidence.shape != (count,)
        or uv.shape != (count, 2)
        or reference_ids.shape != (count,)
        or valid_mask.shape != (count,)
    ):
        raise ValueError("Correspondence inputs must share the same leading size")

    eligible = (
        valid_mask.to(dtype=torch.bool)
        & torch.isfinite(errors)
        & torch.isfinite(confidence)
        & torch.isfinite(uv).all(dim=-1)
        & (reference_ids >= 0)
    )
    eligible_indices = torch.where(eligible)[0]
    target = min(max(0, int(max_points)), int(eligible_indices.numel()))
    if target <= 0:
        return torch.empty(0, dtype=torch.long, device=errors.device), {
            "eligible_count": int(eligible_indices.numel()),
            "selected_count": 0,
            "selected_cell_count": 0,
            "selected_reference_count": 0,
            "reference_cap": 0,
            "reference_cap_relaxed": False,
        }

    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    x = torch.floor(uv[:, 0] * cols / max(1, int(width)))
    y = torch.floor(uv[:, 1] * rows / max(1, int(height)))
    cells = (
        y.to(torch.long).clamp(0, rows - 1) * cols
        + x.to(torch.long).clamp(0, cols - 1)
    )
    index_values = eligible_indices.detach().cpu().tolist()
    error_values = errors[eligible_indices].detach().cpu().tolist()
    confidence_values = confidence[eligible_indices].detach().cpu().tolist()
    cell_values = cells[eligible_indices].detach().cpu().tolist()
    reference_values = reference_ids[eligible_indices].detach().cpu().tolist()
    records = sorted(
        (
            (
                float(error),
                -float(score),
                int(index),
                int(cell),
                int(reference_id),
            )
            for error, score, index, cell, reference_id in zip(
                error_values,
                confidence_values,
                index_values,
                cell_values,
                reference_values,
            )
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    unique_references = sorted({record[4] for record in records})
    if len(unique_references) <= 1:
        reference_cap = target
    else:
        even_share = math.ceil(target / len(unique_references))
        fraction_share = math.ceil(
            target * max(0.0, min(1.0, float(max_reference_fraction)))
        )
        reference_cap = max(1, even_share, fraction_share)

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_cells: set[int] = set()
    reference_counts = {reference_id: 0 for reference_id in unique_references}

    def add(record: tuple[float, float, int, int, int], *, enforce_cap: bool) -> bool:
        _, _, index, cell, reference_id = record
        if index in selected_set or len(selected) >= target:
            return False
        if enforce_cap and reference_counts[reference_id] >= reference_cap:
            return False
        selected.append(index)
        selected_set.add(index)
        selected_cells.add(cell)
        reference_counts[reference_id] += 1
        return True

    for cell in sorted({record[3] for record in records}):
        for record in records:
            if record[3] == cell and add(record, enforce_cap=True):
                break
        if len(selected) >= target:
            break

    for reference_id in unique_references:
        if reference_counts[reference_id] > 0:
            continue
        for record in records:
            if record[4] == reference_id and add(record, enforce_cap=True):
                break

    for record in records:
        add(record, enforce_cap=True)
        if len(selected) >= target:
            break

    cap_relaxed = len(selected) < target
    if cap_relaxed:
        for record in records:
            add(record, enforce_cap=False)
            if len(selected) >= target:
                break

    selected_tensor = torch.tensor(
        selected, dtype=torch.long, device=errors.device
    )
    selected_references = {
        record[4] for record in records if record[2] in selected_set
    }
    return selected_tensor, {
        "eligible_count": int(eligible_indices.numel()),
        "selected_count": len(selected),
        "selected_cell_count": len(selected_cells),
        "selected_reference_count": len(selected_references),
        "reference_cap": int(reference_cap),
        "reference_cap_relaxed": bool(cap_relaxed),
        "max_selected_per_reference": max(reference_counts.values(), default=0),
    }


def compute_reprojection_errors(
    pose: torch.Tensor,
    xyz: torch.Tensor,
    uv: torch.Tensor,
    *,
    focal: torch.Tensor,
    centre: torch.Tensor,
    min_depth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-correspondence pixel errors for a world-to-camera pose."""
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if uv.shape != (xyz.shape[0], 2):
        raise ValueError("uv must have shape [N, 2]")
    pose4 = _pose_4x4(pose)
    camera_xyz = xyz @ pose4[:3, :3].transpose(0, 1) + pose4[:3, 3]
    depth = camera_xyz[:, 2]
    finite = (
        torch.isfinite(camera_xyz).all(dim=-1)
        & torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(pose4).all()
    )
    valid = finite & (depth > float(min_depth))
    focal_value = focal.reshape(-1)[0].to(dtype=xyz.dtype, device=xyz.device)
    centre_value = centre.reshape(2).to(dtype=xyz.dtype, device=xyz.device)
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    projected = camera_xyz[:, :2] * (focal_value / safe_depth[:, None]) + centre_value
    errors = torch.linalg.vector_norm(projected - uv, dim=-1)
    errors = torch.where(valid & torch.isfinite(errors), errors, torch.full_like(errors, float("inf")))
    return errors, valid


def summarize_reprojection_errors(
    errors: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float | int]:
    valid = valid_mask.to(dtype=torch.bool) & torch.isfinite(errors)
    values = errors[valid]
    if values.numel() == 0:
        return {
            "valid_count": 0,
            "mean": float("inf"),
            "median": float("inf"),
            "p90": float("inf"),
            "max": float("inf"),
        }
    return {
        "valid_count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "max": float(values.max().item()),
    }


def validation_candidate_rank(
    pre: dict[str, Any],
    post: dict[str, Any],
) -> tuple[float, float, float]:
    """Rank accepted candidates without overfitting one error quantile."""

    def ratio(field: str) -> float:
        before = float(pre.get(field, float("inf")))
        after = float(post.get(field, float("inf")))
        if not math.isfinite(before) or not math.isfinite(after):
            return float("inf")
        return after / max(abs(before), 1e-8)

    median_ratio = ratio("median")
    mean_ratio = ratio("mean")
    p90_ratio = ratio("p90")
    robust_score = (
        0.45 * median_ratio
        + 0.35 * mean_ratio
        + 0.20 * p90_ratio
    )
    return (
        robust_score,
        max(median_ratio, mean_ratio, p90_ratio),
        median_ratio,
    )


def split_pose_verification_evidence(
    corr_ref_ids: torch.Tensor,
    uv: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    width: int,
    height: int,
    min_solve_support: int = 4,
    min_validation_support: int = 4,
    grid_rows: int = 4,
    grid_cols: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Create disjoint solve and validation evidence for pose verification."""
    if corr_ref_ids.ndim != 1:
        raise ValueError("corr_ref_ids must have shape [N]")
    if uv.shape != (corr_ref_ids.shape[0], 2):
        raise ValueError("uv must have shape [N, 2]")
    if valid_mask.shape != corr_ref_ids.shape:
        raise ValueError("valid_mask must have shape [N]")

    eligible = (
        valid_mask.to(dtype=torch.bool)
        & torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(corr_ref_ids.to(dtype=torch.float32))
    )
    solve = torch.zeros_like(eligible)
    validation = torch.zeros_like(eligible)
    min_solve = max(4, int(min_solve_support))
    min_validation = max(4, int(min_validation_support))
    eligible_indices = torch.where(eligible)[0]
    unique_refs = torch.unique(corr_ref_ids[eligible]).detach().cpu().tolist()
    strategy = "reference_holdout"
    spatial_balance_fallback = False

    if len(unique_refs) >= 2:
        counts = {
            int(ref_id): int((eligible & (corr_ref_ids == int(ref_id))).sum().item())
            for ref_id in unique_refs
        }
        ordered_refs = sorted(counts, key=lambda ref_id: (-counts[ref_id], ref_id))
        solve_refs = {ordered_refs[0]}
        validation_refs: set[int] = set()
        solve_count = counts[ordered_refs[0]]
        validation_count = 0
        for ref_id in ordered_refs[1:]:
            if validation_count <= solve_count:
                validation_refs.add(ref_id)
                validation_count += counts[ref_id]
            else:
                solve_refs.add(ref_id)
                solve_count += counts[ref_id]
        for ref_id in solve_refs:
            solve |= eligible & (corr_ref_ids == ref_id)
        for ref_id in validation_refs:
            validation |= eligible & (corr_ref_ids == ref_id)
    elif len(unique_refs) == 1:
        strategy = "single_reference_spatial_holdout"
        rows = max(1, int(grid_rows))
        cols = max(1, int(grid_cols))
        safe_width = max(1, int(width))
        safe_height = max(1, int(height))
        x = torch.floor(uv[:, 0] * cols / safe_width).to(torch.long).clamp(0, cols - 1)
        y = torch.floor(uv[:, 1] * rows / safe_height).to(torch.long).clamp(0, rows - 1)
        validation = eligible & (((y * cols + x) % 2) == 1)
        solve = eligible & ~validation
        if int(solve.sum()) < min_solve or int(validation.sum()) < min_validation:
            spatial_balance_fallback = True
            solve.zero_()
            validation.zero_()
            validation[eligible_indices[1::2]] = True
            solve[eligible_indices[0::2]] = True

    solve_count = int(solve.sum().item())
    validation_count = int(validation.sum().item())
    disjoint = not bool((solve & validation).any())
    complete = bool(torch.equal(solve | validation, eligible))
    split_valid = bool(
        solve_count >= min_solve
        and validation_count >= min_validation
        and disjoint
        and complete
    )
    solve_reference_ids = sorted(
        {int(value) for value in corr_ref_ids[solve].detach().cpu().tolist()}
    )
    validation_reference_ids = sorted(
        {int(value) for value in corr_ref_ids[validation].detach().cpu().tolist()}
    )
    return solve, validation, {
        "valid": split_valid,
        "reason": "ready" if split_valid else "insufficient_independent_support",
        "strategy": strategy,
        "eligible_count": int(eligible.sum().item()),
        "solve_count": solve_count,
        "validation_count": validation_count,
        "min_solve_support": min_solve,
        "min_validation_support": min_validation,
        "solve_reference_ids": solve_reference_ids,
        "validation_reference_ids": validation_reference_ids,
        "reference_count": len(unique_refs),
        "spatial_balance_fallback": spatial_balance_fallback,
        "disjoint": disjoint,
        "complete": complete,
    }


def select_robust_correspondences(
    errors: torch.Tensor,
    uv: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    width: int,
    height: int,
    mad_scale: float = 2.5,
    max_cutoff: float = float("inf"),
    min_support: int = 24,
    grid_rows: int = 4,
    grid_cols: int = 6,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """MAD-filter correspondences while preserving spatially distributed support."""
    if errors.ndim != 1 or uv.shape != (errors.shape[0], 2):
        raise ValueError("errors and uv must have shapes [N] and [N, 2]")
    valid = (
        valid_mask.to(dtype=torch.bool)
        & torch.isfinite(errors)
        & torch.isfinite(uv).all(dim=-1)
    )
    selected = torch.zeros_like(valid)
    values = errors[valid]
    if values.numel() == 0:
        return selected, {
            "valid_count": 0,
            "selected_count": 0,
            "rejected_count": 0,
            "median": float("inf"),
            "mad": float("inf"),
            "cutoff": float("inf"),
            "occupied_cells": 0,
            "selected_cells": 0,
            "restored_for_min_support": 0,
        }

    median = values.median()
    mad = (values - median).abs().median()
    robust_sigma = 1.4826 * mad
    cutoff_value = max(1.0, float((median + max(0.0, mad_scale) * robust_sigma).item()))
    if math.isfinite(max_cutoff) and max_cutoff > 0:
        cutoff_value = min(cutoff_value, float(max_cutoff))
    selected = valid & (errors <= cutoff_value)

    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    x = torch.floor(uv[:, 0] * cols / safe_width).to(torch.long).clamp(0, cols - 1)
    y = torch.floor(uv[:, 1] * rows / safe_height).to(torch.long).clamp(0, rows - 1)
    cells = y * cols + x
    bounded = valid
    if math.isfinite(max_cutoff) and max_cutoff > 0:
        bounded = bounded & (errors <= float(max_cutoff))
    occupied = torch.unique(cells[bounded])
    for cell in occupied:
        indices = torch.where(bounded & (cells == cell))[0]
        if indices.numel() > 0:
            best = indices[torch.argmin(errors[indices])]
            selected[best] = True

    before_restore = int(selected.sum().item())
    target = min(max(0, int(min_support)), int(bounded.sum().item()))
    if before_restore < target:
        candidates = torch.where(bounded)[0]
        order = torch.argsort(errors[candidates])
        selected[candidates[order[:target]]] = True
    selected_cells = int(torch.unique(cells[selected]).numel()) if bool(selected.any()) else 0
    selected_count = int(selected.sum().item())
    valid_count = int(valid.sum().item())
    return selected, {
        "valid_count": valid_count,
        "selected_count": selected_count,
        "rejected_count": valid_count - selected_count,
        "median": float(median.item()),
        "mad": float(mad.item()),
        "cutoff": float(cutoff_value),
        "occupied_cells": int(occupied.numel()),
        "selected_cells": selected_cells,
        "restored_for_min_support": max(0, selected_count - before_restore),
    }


def pose_correction_magnitude(
    initial_pose: torch.Tensor,
    refined_pose: torch.Tensor,
) -> dict[str, float]:
    initial = _pose_4x4(initial_pose)
    refined = _pose_4x4(refined_pose)
    relative_rotation = refined[:3, :3] @ initial[:3, :3].transpose(0, 1)
    cosine = ((torch.trace(relative_rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_deg = torch.rad2deg(torch.acos(cosine))
    initial_centre = -(initial[:3, :3].transpose(0, 1) @ initial[:3, 3])
    refined_centre = -(refined[:3, :3].transpose(0, 1) @ refined[:3, 3])
    translation = torch.linalg.vector_norm(refined_centre - initial_centre)
    return {
        "rotation_deg": float(rotation_deg.item()),
        "translation": float(translation.item()),
    }


def decide_pose_refinement(
    *,
    pre: dict[str, Any],
    post: dict[str, Any],
    correction: dict[str, Any],
    max_rotation_deg: float,
    max_translation: float,
    min_relative_median_improvement: float = 0.02,
    max_p90_ratio: float = 1.01,
    min_support_ratio: float = 0.80,
) -> dict[str, float | int | bool | str]:
    pre_median = float(pre.get("median", float("inf")))
    post_median = float(post.get("median", float("inf")))
    pre_mean = float(pre.get("mean", pre_median))
    post_mean = float(post.get("mean", post_median))
    pre_p90 = float(pre.get("p90", float("inf")))
    post_p90 = float(post.get("p90", float("inf")))
    pre_count = int(pre.get("valid_count", 0) or 0)
    post_count = int(post.get("valid_count", 0) or 0)
    rotation_deg = float(correction.get("rotation_deg", float("inf")))
    translation = float(correction.get("translation", float("inf")))
    finite = all(
        math.isfinite(value)
        for value in (
            pre_mean,
            post_mean,
            pre_median,
            post_median,
            pre_p90,
            post_p90,
            rotation_deg,
            translation,
        )
    )
    relative_improvement = (
        (pre_median - post_median) / max(abs(pre_median), 1e-6)
        if finite
        else float("-inf")
    )
    support_ratio = post_count / max(pre_count, 1)
    relative_mean_improvement = (
        (pre_mean - post_mean) / max(abs(pre_mean), 1e-6)
        if finite
        else float("-inf")
    )

    reason = "accepted"
    if not finite:
        reason = "non_finite_diagnostics"
    elif pre_count <= 0 or post_count <= 0:
        reason = "insufficient_valid_support"
    elif relative_improvement < float(min_relative_median_improvement):
        reason = "median_not_improved"
    elif post_mean > pre_mean:
        reason = "mean_degraded"
    elif post_p90 > pre_p90 * float(max_p90_ratio):
        reason = "p90_degraded"
    elif support_ratio < float(min_support_ratio):
        reason = "support_collapsed"
    elif rotation_deg > float(max_rotation_deg):
        reason = "rotation_correction_too_large"
    elif translation > float(max_translation):
        reason = "translation_correction_too_large"
    return {
        "accepted": reason == "accepted",
        "reason": reason,
        "relative_median_improvement": float(relative_improvement),
        "relative_mean_improvement": float(relative_mean_improvement),
        "support_ratio": float(support_ratio),
        "rotation_correction_deg": rotation_deg,
        "translation_correction": translation,
        "max_rotation_deg": float(max_rotation_deg),
        "max_translation": float(max_translation),
    }


def choose_verified_pose(
    initial_pose: torch.Tensor,
    refined_pose: torch.Tensor,
    decision: dict[str, Any],
) -> torch.Tensor:
    return (refined_pose if bool(decision.get("accepted", False)) else initial_pose).clone()
