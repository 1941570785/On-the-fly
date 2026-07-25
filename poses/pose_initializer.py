# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 位姿初始化器，用于初始化关键帧的位姿
# 参考自：https://github.com/verlab/accelerated_features


import torch
import math
import time

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from poses.triangulator import matches_to_points
from utils import fov2focal, depth2points, sixD2mtx
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType
from poses.delayed_pose_verification import summarize_candidate_validation
from poses.pose_verification import (
    choose_verified_pose,
    compute_epipolar_sampson_errors,
    compute_reprojection_errors,
    decide_pose_refinement,
    estimate_multireference_relative_pose_candidates,
    estimate_opencv_pnp_candidates,
    interpolate_world_to_camera_pose,
    pose_correction_magnitude,
    predict_constant_velocity_pose,
    select_balanced_correspondence_indices,
    select_pose_candidate_by_reprojection,
    select_robust_correspondences,
    split_pose_verification_evidence,
    summarize_reprojection_errors,
    temporal_pose_consistency,
    validation_candidate_rank,
)
from experiment_reproducibility import experiment_cuda_graphs_enabled


def should_retry_direct_pose_initialization_v18(
    debug: dict[str, object],
    *,
    is_test: bool,
    retry_test_frames: bool = True,
    min_2d3d: int = 500,
    min_pnp_inliers: int = 20,
) -> bool:
    """Gate v18 retries to supported MiniBA failures."""
    if is_test and not retry_test_frames:
        return False
    if str(debug.get("failure_reason", "") or "") != "miniba_inliers_too_few":
        return False
    valid_2d3d = int(debug.get("num_2d3d_correspondences", 0) or 0)
    pnp_inliers = int(debug.get("num_pnp_inliers", 0) or 0)
    return valid_2d3d >= int(min_2d3d) and pnp_inliers >= int(min_pnp_inliers)


def should_run_direct_pose_multi_hypothesis_v18(
    debug: dict[str, object],
    *,
    min_2d3d: int = 2000,
    min_pnp_inliers: int = 700,
    max_pnp_inlier_ratio: float = 0.45,
    max_miniba_residual: float = 1.0,
) -> bool:
    """Detect successful but weak v18 poses that need re-sampling."""
    if str(debug.get("failure_reason", "") or ""):
        return False
    valid_2d3d = int(debug.get("num_2d3d_correspondences", 0) or 0)
    if valid_2d3d < int(min_2d3d):
        return False
    pnp_inliers = int(debug.get("num_pnp_inliers", 0) or 0)
    sampled = int(debug.get("num_pnp_candidate_correspondences", 0) or 0)
    if sampled <= 0:
        sampled = min(
            valid_2d3d,
            int(debug.get("num_pts_pnpransac", valid_2d3d) or valid_2d3d),
        )
    pnp_ratio = float(pnp_inliers) / max(float(sampled), 1.0)
    miniba_residual = float(
        debug.get("direct_pose_miniba_residual", 0.0) or 0.0
    )
    if (
        math.isfinite(miniba_residual)
        and miniba_residual > float(max_miniba_residual)
    ):
        return True
    return (
        pnp_inliers < int(min_pnp_inliers)
        or pnp_ratio < float(max_pnp_inlier_ratio)
    )


def score_direct_pose_candidate_v18(debug: dict[str, object]) -> float:
    """Rank v18 hypotheses by support, residual, and reference motion."""
    pnp_inliers = float(debug.get("num_pnp_inliers", 0) or 0)
    miniba_inliers = float(debug.get("num_miniba_inliers", 0) or 0)
    rotation_deg = float(
        debug.get("direct_pose_motion_rotation_deg", 0.0) or 0.0
    )
    translation = float(
        debug.get("direct_pose_motion_translation", 0.0) or 0.0
    )
    miniba_residual = float(
        debug.get("direct_pose_miniba_residual", 0.0) or 0.0
    )
    support_score = pnp_inliers + 0.50 * miniba_inliers
    motion_penalty = rotation_deg + 25.0 * translation
    residual_penalty = (
        400.0 * miniba_residual if math.isfinite(miniba_residual) else 0.0
    )
    return support_score - motion_penalty - residual_penalty


def should_accept_direct_pose_candidate_v18(
    current_best: dict[str, object],
    candidate: dict[str, object],
    *,
    min_support_gain: float = 0.15,
    min_residual_gain: float = 0.20,
    rotation_margin_deg: float = 4.0,
    translation_margin: float = 0.12,
    translation_scale: float = 2.5,
) -> bool:
    """Accept a v18 replacement only with a supported, bounded gain."""
    best_support = float(
        current_best.get("num_pnp_inliers", 0) or 0
    ) + 0.50 * float(current_best.get("num_miniba_inliers", 0) or 0)
    candidate_support = float(
        candidate.get("num_pnp_inliers", 0) or 0
    ) + 0.50 * float(candidate.get("num_miniba_inliers", 0) or 0)
    best_residual = float(
        current_best.get("direct_pose_miniba_residual", float("inf"))
        or float("inf")
    )
    candidate_residual = float(
        candidate.get("direct_pose_miniba_residual", float("inf"))
        or float("inf")
    )
    support_gain_ok = candidate_support > best_support * (
        1.0 + float(min_support_gain)
    )
    residual_gain_ok = (
        math.isfinite(best_residual)
        and math.isfinite(candidate_residual)
        and candidate_residual
        < best_residual * (1.0 - float(min_residual_gain))
        and candidate_support >= best_support * 0.75
    )
    if not support_gain_ok and not residual_gain_ok:
        return False

    best_rotation = float(
        current_best.get("direct_pose_motion_rotation_deg", 0.0) or 0.0
    )
    candidate_rotation = float(
        candidate.get("direct_pose_motion_rotation_deg", 0.0) or 0.0
    )
    best_translation = float(
        current_best.get("direct_pose_motion_translation", 0.0) or 0.0
    )
    candidate_translation = float(
        candidate.get("direct_pose_motion_translation", 0.0) or 0.0
    )
    max_rotation = max(
        8.0,
        best_rotation + float(rotation_margin_deg),
        best_rotation * 1.75,
    )
    max_translation = max(
        0.35,
        best_translation * float(translation_scale)
        + float(translation_margin),
    )
    if candidate_rotation > max_rotation:
        return False
    if candidate_translation > max_translation:
        return False
    return score_direct_pose_candidate_v18(
        candidate
    ) > score_direct_pose_candidate_v18(current_best)


class PoseInitializer():
    """
    【位姿估计模块】位姿初始化器
    
    负责两种姿态初始化模式：
    1. Bootstrap模式：同时估计多个关键帧的初始位姿和焦距
    2. 增量模式：使用PnP-RANSAC和Mini-BA估计新关键帧的位姿
    
    使用Mini-BA（小规模Bundle Adjustment）进行快速优化。
    """
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args):
        """
        【位姿估计模块】初始化位姿初始化器
        
        Args:
            width: 图像宽度
            height: 图像高度
            triangulator: 三角化器
            matcher: 特征匹配器
            max_pnp_error: PnP-RANSAC的最大误差
            args: 训练参数
        """
        # 相机尺寸与模块引用
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers

        # Initialize the focal length
        # 选择初始焦距：优先用户给定，其次 FOV，最后默认 0.7*width
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        make_cuda_graph = experiment_cuda_graphs_enabled(
            bool(getattr(args, "experiment_deterministic", False))
        )
        self.miniba_bootstrap = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  not args.fix_focal, True,
            make_cuda_graph=make_cuda_graph, iters=args.iters_miniba_bootstrap)
        self.miniba_rebooting = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
            make_cuda_graph=make_cuda_graph, iters=args.iters_miniba_bootstrap)
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=make_cuda_graph, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)
        self.last_incremental_debug: dict[str, object] = {}
        self.last_recovery_pose_outcome_fix: dict[str, object] = {}
        self.last_recovery_2d3d_support: dict[str, object] = {}
        self.last_recovery_pnp_consensus: dict[str, object] = {}
        self.last_recovery_ref_subset: list[dict[str, object]] = []
        self.last_incremental_pose_support: dict[str, torch.Tensor] = {}
        self.last_incremental_pose_candidates: dict[str, torch.Tensor] = {}
        self.recovery_defer_source_frame_id: int = -1
        self._last_pnp_Rt: torch.Tensor | None = None
        self.pose_direct_retry_mode = str(
            getattr(args, "pose_direct_retry_mode", "off") or "off"
        ).strip().lower()
        if self.pose_direct_retry_mode not in {"off", "pose_safe_v18"}:
            raise ValueError(
                f"Unsupported pose direct retry mode: {self.pose_direct_retry_mode}"
            )
        self.direct_pose_retry_attempts = 2
        self.direct_pose_retry_min_2d3d = 500
        self.direct_pose_retry_min_pnp_inliers = 20
        self.direct_pose_retry_test_frames = True
        self.direct_pose_multi_hypothesis_attempts = 2
        self.direct_pose_multi_hypothesis_min_2d3d = 2000
        self.direct_pose_multi_hypothesis_min_pnp_inliers = 700
        self.direct_pose_multi_hypothesis_max_pnp_ratio = 0.45
        self.direct_pose_multi_hypothesis_max_miniba_residual = 1.0
        self.recovery_miniba_retry_min_2d3d = 500
        self.recovery_miniba_retry_min_pnp_inliers = 20
        self.recovery_consensus_target_refs = 10
        self.recovery_consensus_max_refs = 12
        self.recovery_consensus_min_refs = 6
        self.recovery_consensus_min_total_valid_2d3d = 120
        self.recovery_probe_min_inlier_ratio = 0.03

    @staticmethod
    def _keyframe_geometry_Rt(keyframe: Keyframe) -> torch.Tensor:
        getter = getattr(keyframe, "get_geometry_Rt", None)
        if callable(getter):
            return getter()
        return keyframe.get_Rt()

    @staticmethod
    def _keyframe_pose_reference_geometry(
        keyframe: Keyframe,
        *,
        verification_evidence: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = str(
            getattr(
                keyframe,
                "_pose_verification_geometry_snapshot_mode",
                "off",
            )
        )
        if not verification_evidence and mode == "frozen_verification_only_v2":
            return (
                keyframe.desc_kpts.pts3d,
                keyframe.desc_kpts.pts_conf,
                keyframe.desc_kpts.has_pt3d,
            )
        getter = getattr(
            keyframe,
            "get_pose_verification_geometry",
            None,
        )
        if callable(getter):
            return getter()
        return (
            keyframe.desc_kpts.pts3d,
            keyframe.desc_kpts.pts_conf,
            keyframe.desc_kpts.has_pt3d,
        )

    @staticmethod
    def _keyframe_pose_reference_bundle(
        keyframe: Keyframe,
        matched_indices: torch.Tensor,
        *,
        verification_evidence: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
        dict,
    ]:
        live_geometry = (
            keyframe.desc_kpts.pts3d,
            keyframe.desc_kpts.pts_conf,
            keyframe.desc_kpts.has_pt3d,
        )
        live_Rt = PoseInitializer._keyframe_geometry_Rt(keyframe)
        mode = str(
            getattr(
                keyframe,
                "_pose_verification_geometry_snapshot_mode",
                "off",
            )
        )

        if mode not in {
            "guarded_frozen_v3",
            "guarded_frozen_live_pose_v4",
            "guarded_frozen_homogeneous_v5",
        }:
            geometry = PoseInitializer._keyframe_pose_reference_geometry(
                keyframe,
                verification_evidence=verification_evidence,
            )
            snapshot = getattr(
                keyframe,
                "_pose_verification_geometry_snapshot",
                None,
            )
            used_snapshot = bool(
                snapshot is not None
                and mode
                in {
                    "frozen_first_valid_v1",
                    "frozen_verification_only_v2",
                    "frozen_global_support_guard_v6",
                }
                and (
                    verification_evidence
                    or mode != "frozen_verification_only_v2"
                )
            )
            return (
                *geometry,
                live_Rt,
                used_snapshot,
                {
                    "reason": (
                        "legacy_frozen_geometry"
                        if used_snapshot
                        else "live_geometry"
                    ),
                    "live_support": None,
                    "frozen_support": None,
                    "required_frozen_support": None,
                },
            )

        snapshot_marker = getattr(
            keyframe,
            "_pose_verification_geometry_snapshot",
            ...
        )
        snapshot_getter = getattr(
            keyframe,
            "get_pose_verification_geometry",
            None,
        )
        if snapshot_marker is None or not callable(snapshot_getter):
            return (
                *live_geometry,
                live_Rt,
                False,
                {
                    "reason": "frozen_snapshot_unavailable",
                    "live_support": None,
                    "frozen_support": 0,
                    "required_frozen_support": None,
                },
            )

        frozen_geometry = snapshot_getter()
        frozen_points, frozen_confidence, frozen_mask = frozen_geometry
        live_points, live_confidence, live_mask = live_geometry
        if (
            frozen_mask.shape[0] != live_mask.shape[0]
            or matched_indices.numel() == 0
        ):
            return (
                live_points,
                live_confidence,
                live_mask,
                live_Rt,
                False,
                {
                    "reason": (
                        "frozen_geometry_shape_mismatch"
                        if frozen_mask.shape[0] != live_mask.shape[0]
                        else "no_current_matches"
                    ),
                    "live_support": 0,
                    "frozen_support": 0,
                    "required_frozen_support": None,
                },
            )

        live_support = int(live_mask[matched_indices].sum().item())
        frozen_support = int(frozen_mask[matched_indices].sum().item())
        minimum_support = max(
            0,
            int(
                getattr(
                    keyframe,
                    "_pose_verification_frozen_min_match_support",
                    24,
                )
            ),
        )
        minimum_live_ratio = max(
            0.0,
            float(
                getattr(
                    keyframe,
                    "_pose_verification_frozen_min_live_ratio",
                    0.50,
                )
            ),
        )
        ratio_support = int(math.ceil(live_support * minimum_live_ratio))
        required_support = max(minimum_support, ratio_support)
        debug = {
            "live_support": live_support,
            "frozen_support": frozen_support,
            "required_frozen_support": required_support,
            "minimum_match_support": minimum_support,
            "minimum_live_ratio": minimum_live_ratio,
        }
        if frozen_support < required_support:
            return (
                live_points,
                live_confidence,
                live_mask,
                live_Rt,
                False,
                {
                    **debug,
                    "reason": "frozen_support_too_low",
                },
            )

        frozen_Rt_getter = getattr(
            keyframe,
            "get_pose_verification_geometry_Rt",
            None,
        )
        frozen_Rt = (
            frozen_Rt_getter()
            if callable(frozen_Rt_getter)
            else live_Rt
        )
        selected_Rt = (
            live_Rt
            if mode
            in {
                "guarded_frozen_live_pose_v4",
                "guarded_frozen_homogeneous_v5",
            }
            else frozen_Rt
        )
        if mode == "guarded_frozen_live_pose_v4":
            reason = "frozen_support_guard_passed_live_pose"
        elif mode == "guarded_frozen_homogeneous_v5":
            reason = "frozen_support_guard_passed_homogeneous_candidate"
        else:
            reason = "frozen_support_guard_passed"
        return (
            frozen_points,
            frozen_confidence,
            frozen_mask,
            selected_Rt,
            True,
            {
                **debug,
                "reason": reason,
            },
        )

    @staticmethod
    def _select_guarded_frozen_frame_policy(
        candidate_supports: list[int],
        *,
        total_reference_count: int,
        min_reference_count: int,
        min_total_support: int,
    ) -> tuple[str, list[int], dict]:
        normalized_supports = [
            max(0, int(value)) for value in candidate_supports
        ]
        candidate_indices = [
            index
            for index, support in enumerate(normalized_supports)
            if support > 0
        ]
        candidate_total_support = sum(
            normalized_supports[index] for index in candidate_indices
        )
        required_references = max(1, int(min_reference_count))
        required_total_support = max(4, int(min_total_support))
        use_frozen_subset = bool(
            len(candidate_indices) >= required_references
            and candidate_total_support >= required_total_support
        )
        debug = {
            "mode": (
                "frozen_subset"
                if use_frozen_subset
                else "live_fallback"
            ),
            "candidate_reference_count": len(candidate_indices),
            "candidate_total_support": candidate_total_support,
            "required_reference_count": required_references,
            "required_total_support": required_total_support,
            "total_reference_count": max(0, int(total_reference_count)),
        }
        if use_frozen_subset:
            return "frozen_subset", candidate_indices, debug
        return (
            "live_fallback",
            list(range(max(0, int(total_reference_count)))),
            debug,
        )

    @staticmethod
    def _select_frozen_global_support_policy(
        candidate_supports: list[int],
        *,
        all_snapshots_available: bool,
        total_reference_count: int,
        min_total_support: int,
    ) -> tuple[str, list[int], dict]:
        normalized_supports = [
            max(0, int(value)) for value in candidate_supports
        ]
        reference_count = max(0, int(total_reference_count))
        candidate_total_support = sum(normalized_supports)
        required_total_support = max(4, int(min_total_support))
        use_all_frozen = bool(
            all_snapshots_available
            and len(normalized_supports) == reference_count
            and candidate_total_support >= required_total_support
        )
        debug = {
            "mode": "frozen_all" if use_all_frozen else "live_fallback",
            "candidate_total_support": candidate_total_support,
            "required_total_support": required_total_support,
            "total_reference_count": reference_count,
            "all_snapshots_available": bool(all_snapshots_available),
        }
        indices = list(range(reference_count))
        return (
            "frozen_all" if use_all_frozen else "live_fallback",
            indices,
            debug,
        )

    def _record_incremental_pose_support(
        self,
        match_indices: torch.Tensor,
        pts3d: torch.Tensor,
        pts_conf: torch.Tensor,
        uvs: torch.Tensor,
        corr_ref_ids: torch.Tensor,
    ) -> None:
        self.last_incremental_pose_support = {
            "match_indices": match_indices.detach().clone(),
            "pts3d": pts3d.detach().clone(),
            "pts_conf": pts_conf.detach().clone(),
            "uvs": uvs.detach().clone(),
            "corr_ref_ids": corr_ref_ids.detach().clone(),
        }

    def _record_incremental_pose_candidates(
        self,
        match_indices: torch.Tensor,
        pts3d: torch.Tensor,
        pts_conf: torch.Tensor,
        uvs: torch.Tensor,
        corr_ref_ids: torch.Tensor,
        ref_uvs: torch.Tensor | None = None,
        ref_Rts: torch.Tensor | None = None,
    ) -> None:
        self.last_incremental_pose_candidates = {
            "match_indices": match_indices.detach().clone(),
            "pts3d": pts3d.detach().clone(),
            "pts_conf": pts_conf.detach().clone(),
            "uvs": uvs.detach().clone(),
            "corr_ref_ids": corr_ref_ids.detach().clone(),
        }
        if isinstance(ref_uvs, torch.Tensor):
            self.last_incremental_pose_candidates["ref_uvs"] = (
                ref_uvs.detach().clone()
            )
        if isinstance(ref_Rts, torch.Tensor):
            self.last_incremental_pose_candidates["ref_Rts"] = (
                ref_Rts.detach().clone()
            )

    def _run_incremental_pose_with_direct_retry_v18(
        self,
        *,
        keyframes: list[Keyframe],
        index: int,
        is_test: bool,
        pnp_xyz: torch.Tensor,
        pnp_uvs: torch.Tensor,
        pnp_confs: torch.Tensor,
        pnp_match_indices: torch.Tensor,
        pnp_corr_ref_ids: torch.Tensor,
        reference_initial_Rt: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the locked pose-safe v18 retry and hypothesis policy."""
        rotation_init = reference_initial_Rt[:3, :2]
        translation_init = reference_initial_Rt[:3, 3]

        def clone_support() -> dict[str, torch.Tensor]:
            return {
                key: (
                    value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in self.last_incremental_pose_support.items()
            }

        def restore_support(support: dict[str, torch.Tensor]) -> None:
            self.last_incremental_pose_support = {
                key: (
                    value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in support.items()
            }

        def motion_debug(pose: torch.Tensor) -> dict[str, float]:
            relative_rotation = (
                pose[:3, :3] @ reference_initial_Rt[:3, :3].transpose(0, 1)
            )
            trace = torch.trace(relative_rotation).clamp(-1.0, 3.0)
            cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
            rotation_deg = float(
                torch.rad2deg(torch.arccos(cosine)).item()
            )
            translation = float(
                torch.linalg.vector_norm(
                    pose[:3, 3] - reference_initial_Rt[:3, 3]
                ).item()
            )
            return {
                "direct_pose_motion_rotation_deg": rotation_deg,
                "direct_pose_motion_translation": translation,
            }

        def select_pnp_sample() -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            if len(pnp_xyz) > self.num_pts_pnpransac:
                selected = torch.multinomial(
                    pnp_confs,
                    self.num_pts_miniba_incr,
                    replacement=False,
                )
                return (
                    pnp_xyz[selected],
                    pnp_uvs[selected],
                    pnp_confs[selected],
                    pnp_match_indices[selected],
                    pnp_corr_ref_ids[selected],
                )
            return (
                pnp_xyz,
                pnp_uvs,
                pnp_confs,
                pnp_match_indices,
                pnp_corr_ref_ids,
            )

        def run_pose_attempt(attempt_id: int) -> torch.Tensor | None:
            self.last_incremental_debug[
                "direct_pose_retry_attempt_id"
            ] = int(attempt_id)
            (
                xyz,
                uvs,
                confs,
                match_indices,
                corr_ref_ids,
            ) = select_pnp_sample()
            self.last_incremental_debug[
                "num_pnp_candidate_correspondences"
            ] = int(len(xyz))
            self.last_incremental_debug["num_pts_pnpransac"] = int(
                self.num_pts_pnpransac
            )
            try:
                pnp_Rt, inliers = self.PnPRANSAC(
                    uvs,
                    xyz,
                    self.f,
                    self.centre,
                    rotation_init,
                    translation_init,
                    confs,
                )
            except Exception:
                self.last_incremental_debug[
                    "failure_reason"
                ] = "pnp_ransac_exception"
                return None

            inliers = inliers.to(dtype=torch.bool)
            self._last_pnp_Rt = pnp_Rt.clone()
            self.last_incremental_debug["pnp_candidate_Rt"] = (
                pnp_Rt.detach().cpu().tolist()
            )
            xyz = xyz[inliers]
            uvs = uvs[inliers]
            confs = confs[inliers]
            match_indices = match_indices[inliers]
            corr_ref_ids = corr_ref_ids[inliers]
            self.last_incremental_debug["num_pnp_inliers"] = int(len(xyz))
            pnp_reference_ids = sorted(
                {int(value) for value in corr_ref_ids.detach().cpu().tolist()}
            )
            self.last_incremental_debug[
                "pnp_ref_keyframe_ids"
            ] = pnp_reference_ids
            self.last_incremental_debug[
                "pnp_ref_source_frame_ids"
            ] = [
                int(
                    keyframe.info.get(
                        "_paper_aligned_source_frame_id",
                        keyframe.index,
                    )
                )
                for keyframe in keyframes
                if int(keyframe.index) in pnp_reference_ids
            ]
            self.last_incremental_debug["pnp_ref_contains_seed"] = any(
                bool(
                    keyframe.info.get(
                        "_paper_aligned_is_v7_early_seed",
                        False,
                    )
                )
                for keyframe in keyframes
                if int(keyframe.index) in pnp_reference_ids
            )
            if len(xyz) < 4:
                self.last_incremental_debug[
                    "failure_reason"
                ] = "pnp_inliers_too_few"
                return None

            selected_indices: torch.Tensor | None = None
            if len(xyz) >= self.num_pts_miniba_incr:
                selected_indices = torch.topk(
                    torch.rand_like(xyz[..., 0]),
                    self.num_pts_miniba_incr,
                    dim=0,
                    largest=False,
                )[1]
                xyz_ba = xyz[selected_indices]
                uvs_ba = uvs[selected_indices]
                confs_ba = confs[selected_indices]
                match_indices_ba = match_indices[selected_indices]
                corr_ref_ids_ba = corr_ref_ids[selected_indices]
                miniba_reference_ids_tensor = corr_ref_ids_ba
            else:
                padding = self.num_pts_miniba_incr - len(xyz)
                xyz_ba = torch.cat(
                    [
                        xyz,
                        torch.zeros(
                            padding,
                            3,
                            device=xyz.device,
                            dtype=xyz.dtype,
                        ),
                    ],
                    dim=0,
                )
                uvs_ba = torch.cat(
                    [
                        uvs,
                        -torch.ones(
                            padding,
                            2,
                            device=uvs.device,
                            dtype=uvs.dtype,
                        ),
                    ],
                    dim=0,
                )
                confs_ba = torch.cat(
                    [
                        confs,
                        torch.zeros(
                            padding,
                            device=confs.device,
                            dtype=confs.dtype,
                        ),
                    ],
                    dim=0,
                )
                match_indices_ba = torch.cat(
                    [
                        match_indices,
                        -torch.ones(
                            padding,
                            device=match_indices.device,
                            dtype=match_indices.dtype,
                        ),
                    ],
                    dim=0,
                )
                corr_ref_ids_ba = torch.cat(
                    [
                        corr_ref_ids,
                        -torch.ones(
                            padding,
                            device=corr_ref_ids.device,
                            dtype=corr_ref_ids.dtype,
                        ),
                    ],
                    dim=0,
                )
                miniba_reference_ids_tensor = corr_ref_ids

            miniba_reference_ids = sorted(
                {
                    int(value)
                    for value in miniba_reference_ids_tensor.detach()
                    .cpu()
                    .tolist()
                }
            )
            self.last_incremental_debug[
                "miniba_ref_keyframe_ids"
            ] = miniba_reference_ids
            self.last_incremental_debug[
                "miniba_ref_source_frame_ids"
            ] = [
                int(
                    keyframe.info.get(
                        "_paper_aligned_source_frame_id",
                        keyframe.index,
                    )
                )
                for keyframe in keyframes
                if int(keyframe.index) in miniba_reference_ids
            ]
            self.last_incremental_debug["miniba_ref_contains_seed"] = any(
                bool(
                    keyframe.info.get(
                        "_paper_aligned_is_v7_early_seed",
                        False,
                    )
                )
                for keyframe in keyframes
                if int(keyframe.index) in miniba_reference_ids
            )

            rotations = pnp_Rt[:3, :2][None]
            translations = pnp_Rt[:3, 3][None]
            (
                rotations,
                translations,
                _,
                _,
                miniba_residuals,
                _,
                miniba_mask,
            ) = self.miniBA_incr(
                rotations,
                translations,
                self.f,
                xyz_ba,
                self.centre,
                uvs_ba.view(-1),
            )
            miniba_count = int(miniba_mask.sum().item())
            self.last_incremental_debug[
                "num_miniba_inliers"
            ] = miniba_count
            mask_sum = miniba_mask.sum()
            if int(mask_sum.item()) > 0:
                miniba_residual = float(
                    (
                        (miniba_residuals * miniba_mask).abs().sum()
                        / mask_sum
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            else:
                miniba_residual = float("inf")
            self.last_incremental_debug[
                "direct_pose_miniba_residual"
            ] = miniba_residual

            pose = torch.eye(
                4,
                device=pnp_Rt.device,
                dtype=pnp_Rt.dtype,
            )
            pose[:3, :3] = sixD2mtx(rotations)[0]
            pose[:3, 3] = translations[0]
            self.last_incremental_debug["miniba_candidate_Rt"] = (
                pose.detach().cpu().tolist()
            )
            self.last_incremental_debug.update(motion_debug(pose))
            self.last_incremental_debug[
                "direct_pose_candidate_score"
            ] = score_direct_pose_candidate_v18(
                self.last_incremental_debug
            )

            if miniba_count > self.min_num_inliers:
                valid_ba = uvs_ba[:, 0] >= 0
                if bool(valid_ba.any()):
                    self._record_incremental_pose_support(
                        match_indices_ba[valid_ba],
                        xyz_ba[valid_ba],
                        confs_ba[valid_ba],
                        uvs_ba[valid_ba],
                        corr_ref_ids_ba[valid_ba],
                    )
                self.last_incremental_debug["failure_reason"] = ""
                return pose

            self.last_incremental_debug[
                "failure_reason"
            ] = "miniba_inliers_too_few"
            return None

        self.last_incremental_debug["pose_direct_retry_mode"] = (
            "pose_safe_v18"
        )
        self.last_incremental_debug["direct_pose_retry_applied"] = False
        self.last_incremental_debug["direct_pose_retry_success"] = False
        self.last_incremental_debug["direct_pose_retry_attempts"] = 0
        self.last_incremental_debug[
            "direct_pose_multi_hypothesis_applied"
        ] = False
        self.last_incremental_debug[
            "direct_pose_multi_hypothesis_attempts"
        ] = 0

        pose = run_pose_attempt(0)
        if pose is not None:
            best_pose = pose
            best_debug = dict(self.last_incremental_debug)
            best_support = clone_support()
            if should_run_direct_pose_multi_hypothesis_v18(
                best_debug,
                min_2d3d=self.direct_pose_multi_hypothesis_min_2d3d,
                min_pnp_inliers=(
                    self.direct_pose_multi_hypothesis_min_pnp_inliers
                ),
                max_pnp_inlier_ratio=(
                    self.direct_pose_multi_hypothesis_max_pnp_ratio
                ),
                max_miniba_residual=(
                    self.direct_pose_multi_hypothesis_max_miniba_residual
                ),
            ):
                best_score = score_direct_pose_candidate_v18(best_debug)
                for retry_id in range(
                    1,
                    self.direct_pose_multi_hypothesis_attempts + 1,
                ):
                    self.last_incremental_debug = dict(best_debug)
                    candidate_pose = run_pose_attempt(retry_id)
                    self.last_incremental_debug[
                        "direct_pose_multi_hypothesis_attempts"
                    ] = retry_id
                    if candidate_pose is None:
                        continue
                    candidate_debug = dict(self.last_incremental_debug)
                    candidate_score = score_direct_pose_candidate_v18(
                        candidate_debug
                    )
                    if (
                        candidate_score > best_score
                        and should_accept_direct_pose_candidate_v18(
                            best_debug,
                            candidate_debug,
                        )
                    ):
                        best_score = candidate_score
                        best_pose = candidate_pose
                        best_debug = candidate_debug
                        best_support = clone_support()
                self.last_incremental_debug = dict(best_debug)
                restore_support(best_support)
                self.last_incremental_debug[
                    "direct_pose_multi_hypothesis_applied"
                ] = True
                self.last_incremental_debug[
                    "direct_pose_multi_hypothesis_attempts"
                ] = self.direct_pose_multi_hypothesis_attempts
                self.last_incremental_debug[
                    "direct_pose_multi_hypothesis_selected_score"
                ] = score_direct_pose_candidate_v18(
                    self.last_incremental_debug
                )
            return best_pose

        initial_failure_reason = str(
            self.last_incremental_debug.get("failure_reason", "") or ""
        )
        initial_pnp_inliers = int(
            self.last_incremental_debug.get("num_pnp_inliers", 0) or 0
        )
        initial_miniba_inliers = int(
            self.last_incremental_debug.get("num_miniba_inliers", 0) or 0
        )
        if should_retry_direct_pose_initialization_v18(
            self.last_incremental_debug,
            is_test=is_test,
            retry_test_frames=self.direct_pose_retry_test_frames,
            min_2d3d=self.direct_pose_retry_min_2d3d,
            min_pnp_inliers=self.direct_pose_retry_min_pnp_inliers,
        ):
            self.last_incremental_debug["direct_pose_retry_applied"] = True
            self.last_incremental_debug[
                "direct_pose_retry_initial_failure_reason"
            ] = initial_failure_reason
            self.last_incremental_debug[
                "direct_pose_retry_initial_pnp_inliers"
            ] = initial_pnp_inliers
            self.last_incremental_debug[
                "direct_pose_retry_initial_miniba_inliers"
            ] = initial_miniba_inliers
            for retry_id in range(1, self.direct_pose_retry_attempts + 1):
                self.last_incremental_debug[
                    "direct_pose_retry_attempts"
                ] = retry_id
                pose = run_pose_attempt(retry_id)
                if pose is not None:
                    self.last_incremental_debug[
                        "direct_pose_retry_success"
                    ] = True
                    self.last_incremental_debug[
                        "direct_pose_retry_success_attempt"
                    ] = retry_id
                    return pose
                if not should_retry_direct_pose_initialization_v18(
                    self.last_incremental_debug,
                    is_test=is_test,
                    retry_test_frames=self.direct_pose_retry_test_frames,
                    min_2d3d=self.direct_pose_retry_min_2d3d,
                    min_pnp_inliers=self.direct_pose_retry_min_pnp_inliers,
                ):
                    break

        if (
            str(
                self.last_incremental_debug.get("failure_reason", "") or ""
            )
            == "miniba_inliers_too_few"
        ):
            print("Too few inliers for pose initialization")
        for keyframe in keyframes:
            keyframe.desc_kpts.matches.pop(index, None)
        return None

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        # 将多视角匹配组织成 miniBA 所需的 uvs / xyz_indices
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        unused_kpts_mask = torch.ones((n_cams, desc_kpts_list[0].kpts.shape[0]), device='cuda', dtype=torch.bool)
        for k in range(n_primary_cam):
            # 统计当前主视角与其他视角的匹配出现次数
            idx_occurrences = torch.zeros(self.num_kpts, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            selected_indices = torch.multinomial(idx_occurrences.float(), npts_per_primary_cam, replacement=False)

            selected_mask = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    # 主视角自身的关键点坐标直接填充
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        # 将主视角关键点与其他视角对齐到同一 3D 点槽位
                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                # 仅填充未被占用的槽位
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        # 若某个 3D 点有效匹配过少则剔除
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    @torch.no_grad()
    def initialize_bootstrap(self, desc_kpts_list: list[DescribedKeypoints], rebooting=False):
        """
        【位姿估计模块】Bootstrap位姿初始化
        
        同时估计多个关键帧的初始位姿和焦距。
        使用Mini-BA进行联合优化，确保所有位姿和焦距的一致性。
        
        Args:
            desc_kpts_list: 关键帧的描述关键点列表
            rebooting: 是否为重启模式（重启时不优化焦距）
        
        Returns:
            Rts: 估计的位姿矩阵列表 [N, 4, 4]
            f: 估计的焦距
            final_residual: 最终残差（用于验证收敛性）
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        # 全连接匹配，以获得稳定的多视角约束
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
        
        ## Build the problem by organizing matches
        uvs, xyz_indices = self.build_problem(desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams)))

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        # 3D 点用单位深度回投影初始化，带随机缩放
        f_init = (torch.tensor([self.f_init], device="cuda"))
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        ## Run miniBA, estimating 3D points, camera focal and poses
        # rebooting 时不再优化焦距
        if rebooting:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(Rs6D_init, ts_init, self.f, xyz_init, self.centre, uvs.view(-1))
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1))
        final_residual = (r * mask).abs().sum()/mask.sum()

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0)

        ## Scale to 0.1 average translation
        # 归一化尺度，避免尺度漂移
        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        ts *= scale
        xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts

        return Rts, f, final_residual

    @torch.no_grad()
    def initialize_incremental(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
        sampling_seed: int | None = None,
        registration_solver_mode: str = "baseline_cuda_v1",
    ):
        """
        【位姿估计模块】增量位姿初始化
        
        使用历史关键帧估计新关键帧的位姿。
        流程：
        1. 匹配当前帧与历史关键帧
        2. 使用PnP-RANSAC估计初始位姿
        3. 使用Mini-BA优化位姿
        
        Args:
            keyframes: 历史关键帧列表
            curr_desc_kpts: 当前帧的描述关键点
            index: 当前帧索引
            is_test: 是否为测试帧
            curr_img: 当前图像（未使用，保留接口）
        
        Returns:
            Rt: 估计的位姿矩阵 [4, 4]，如果失败返回None
        """
        self.last_incremental_pose_support = {}
        self.last_incremental_pose_candidates = {}
        normalized_registration_solver_mode = str(
            registration_solver_mode or "baseline_cuda_v1"
        ).strip().lower()
        if normalized_registration_solver_mode not in {
            "baseline_cuda_v1",
            "deterministic_opencv_v2",
        }:
            raise ValueError(
                "Unsupported pose registration solver mode: "
                f"{registration_solver_mode}"
            )
        self.last_incremental_debug = {
            "failure_reason": "",
            "num_2d3d_correspondences": 0,
            "num_pnp_inliers": 0,
            "num_miniba_inliers": 0,
            "ref_keyframe_ids": [],
            "ref_source_frame_ids": [],
            "ref_commit_origin": [],
            "ref_is_recovery": [],
            "ref_is_seed": [],
            "ref_is_support_eligible": [],
            "match_count_by_ref": [],
            "match_count_total": 0,
            "match_count_to_seed_keyframes": 0,
            "best_match_keyframe_id": -1,
            "best_match_is_seed": False,
            "best_match_num_matches": 0,
            "pnp_ref_keyframe_ids": [],
            "pnp_ref_source_frame_ids": [],
            "pnp_ref_contains_seed": False,
            "miniba_ref_keyframe_ids": [],
            "miniba_ref_source_frame_ids": [],
            "miniba_ref_contains_seed": False,
            "reference_geometry_mode_by_ref": [],
            "reference_geometry_guard_by_ref": [],
            "frozen_reference_count": 0,
            "guarded_frozen_reference_count": 0,
            "guarded_live_fallback_count": 0,
            "guarded_frame_geometry_policy": "per_reference",
            "guarded_frame_geometry_debug": {},
            "sampling_seed": (
                int(sampling_seed) if sampling_seed is not None else None
            ),
            "registration_solver_mode": normalized_registration_solver_mode,
        }

        # Match the current frame with previous keyframes
        # 收集可用于 PnP 的 2D-3D 对应
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        corr_ref_ids = []
        verification_xyz = []
        verification_uvs = []
        verification_confs = []
        verification_match_indices = []
        verification_corr_ref_ids = []
        verification_ref_uvs = []
        verification_ref_Rts = []
        initial_ref_Rts = []
        global_support_mode = any(
            str(
                getattr(
                    keyframe,
                    "_pose_verification_geometry_snapshot_mode",
                    "off",
                )
            )
            == "frozen_global_support_guard_v6"
            for keyframe in keyframes
        )
        homogeneous_mode = global_support_mode or any(
            str(
                getattr(
                    keyframe,
                    "_pose_verification_geometry_snapshot_mode",
                    "off",
                )
            )
            == "guarded_frozen_homogeneous_v5"
            for keyframe in keyframes
        )
        homogeneous_records = None
        if homogeneous_mode:
            candidate_records = []
            for keyframe in keyframes:
                matches = self.matcher(
                    curr_desc_kpts,
                    keyframe.desc_kpts,
                    remove_outliers=True,
                    update_kpts_flag="all",
                    kID=index,
                    kID_other=keyframe.index,
                )
                reference_bundle = self._keyframe_pose_reference_bundle(
                    keyframe,
                    matches.idx_other,
                    verification_evidence=False,
                )
                verification_bundle = self._keyframe_pose_reference_bundle(
                    keyframe,
                    matches.idx_other,
                    verification_evidence=True,
                )
                candidate_records.append(
                    (
                        keyframe,
                        matches,
                        reference_bundle,
                        verification_bundle,
                    )
                )

            candidate_supports = [
                int(
                    record[2][2][record[1].idx_other]
                    .sum()
                    .item()
                )
                if bool(record[2][4])
                else 0
                for record in candidate_records
            ]
            policy_keyframe = keyframes[0]
            if global_support_mode:
                (
                    frame_policy,
                    selected_record_indices,
                    frame_policy_debug,
                ) = self._select_frozen_global_support_policy(
                    candidate_supports,
                    all_snapshots_available=all(
                        bool(record[2][4])
                        for record in candidate_records
                    ),
                    total_reference_count=len(candidate_records),
                    min_total_support=int(
                        getattr(
                            policy_keyframe,
                            "_pose_verification_frozen_min_total_support",
                            24,
                        )
                    ),
                )
            else:
                (
                    frame_policy,
                    selected_record_indices,
                    frame_policy_debug,
                ) = self._select_guarded_frozen_frame_policy(
                    candidate_supports,
                    total_reference_count=len(candidate_records),
                    min_reference_count=int(
                        getattr(
                            policy_keyframe,
                            "_pose_verification_frozen_min_reference_count",
                            2,
                        )
                    ),
                    min_total_support=int(
                        getattr(
                            policy_keyframe,
                            "_pose_verification_frozen_min_total_support",
                            48,
                        )
                    ),
                )
            self.last_incremental_debug[
                "guarded_frame_geometry_policy"
            ] = frame_policy
            self.last_incremental_debug[
                "guarded_frame_geometry_debug"
            ] = frame_policy_debug
            if frame_policy in {"frozen_subset", "frozen_all"}:
                homogeneous_records = [
                    candidate_records[record_index]
                    for record_index in selected_record_indices
                ]
            else:
                homogeneous_records = []
                for (
                    keyframe,
                    matches,
                    reference_bundle,
                    verification_bundle,
                ) in candidate_records:
                    live_geometry = (
                        keyframe.desc_kpts.pts3d,
                        keyframe.desc_kpts.pts_conf,
                        keyframe.desc_kpts.has_pt3d,
                    )
                    live_Rt = self._keyframe_geometry_Rt(keyframe)
                    reference_debug = {
                        **reference_bundle[5],
                        "candidate_reason": reference_bundle[5].get(
                            "reason",
                            "",
                        ),
                        "reason": "frame_homogeneous_live_fallback",
                    }
                    verification_debug = {
                        **verification_bundle[5],
                        "candidate_reason": verification_bundle[5].get(
                            "reason",
                            "",
                        ),
                        "reason": "frame_homogeneous_live_fallback",
                    }
                    homogeneous_records.append(
                        (
                            keyframe,
                            matches,
                            (
                                *live_geometry,
                                live_Rt,
                                False,
                                reference_debug,
                            ),
                            (
                                *live_geometry,
                                live_Rt,
                                False,
                                verification_debug,
                            ),
                        )
                    )

        reference_records = (
            homogeneous_records
            if homogeneous_records is not None
            else (
                (
                    keyframe,
                    self.matcher(
                        curr_desc_kpts,
                        keyframe.desc_kpts,
                        remove_outliers=True,
                        update_kpts_flag="all",
                        kID=index,
                        kID_other=keyframe.index,
                    ),
                    None,
                    None,
                )
                for keyframe in keyframes
            )
        )
        for (
            keyframe,
            matches,
            reference_bundle,
            verification_bundle,
        ) in reference_records:
            # 匹配当前帧与历史关键帧并过滤外点
            if reference_bundle is None:
                reference_bundle = self._keyframe_pose_reference_bundle(
                    keyframe,
                    matches.idx_other,
                    verification_evidence=False,
                )
                verification_bundle = self._keyframe_pose_reference_bundle(
                    keyframe,
                    matches.idx_other,
                    verification_evidence=True,
                )

            (
                ref_points,
                ref_confidence,
                ref_has_point,
                reference_Rt,
                reference_used_snapshot,
                reference_guard_debug,
            ) = reference_bundle
            (
                verification_points,
                verification_confidence,
                verification_has_point,
                verification_reference_Rt,
                verification_used_snapshot,
                verification_guard_debug,
            ) = verification_bundle
            snapshot_mode = str(
                getattr(
                    keyframe,
                    "_pose_verification_geometry_snapshot_mode",
                    "off",
                )
            )
            reference_geometry_mode = (
                snapshot_mode
                if reference_used_snapshot or verification_used_snapshot
                else "live"
            )
            self.last_incremental_debug[
                "reference_geometry_mode_by_ref"
            ].append(reference_geometry_mode)
            self.last_incremental_debug[
                "reference_geometry_guard_by_ref"
            ].append(
                {
                    "initial": reference_guard_debug,
                    "verification": verification_guard_debug,
                }
            )
            if reference_geometry_mode != "live":
                self.last_incremental_debug["frozen_reference_count"] += 1
            if snapshot_mode in {
                "guarded_frozen_v3",
                "guarded_frozen_live_pose_v4",
                "guarded_frozen_homogeneous_v5",
                "frozen_global_support_guard_v6",
            }:
                if reference_used_snapshot:
                    self.last_incremental_debug[
                        "guarded_frozen_reference_count"
                    ] += 1
                else:
                    self.last_incremental_debug[
                        "guarded_live_fallback_count"
                    ] += 1
            initial_ref_Rts.append(reference_Rt)
            mask = ref_has_point[matches.idx_other]
            verification_mask = verification_has_point[matches.idx_other]
            valid_count = int(mask.sum().item())
            verification_valid_count = int(verification_mask.sum().item())
            source_frame_id = int(keyframe.info.get("_paper_aligned_source_frame_id", keyframe.index))
            commit_origin = str(keyframe.info.get("_paper_aligned_commit_origin", "unknown"))
            is_recovery = commit_origin in {"true_recovery_commit", "early_seed_recovery_commit"}
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            self.last_incremental_debug["ref_keyframe_ids"].append(int(keyframe.index))
            self.last_incremental_debug["ref_source_frame_ids"].append(source_frame_id)
            self.last_incremental_debug["ref_commit_origin"].append(commit_origin)
            self.last_incremental_debug["ref_is_recovery"].append(is_recovery)
            self.last_incremental_debug["ref_is_seed"].append(is_seed)
            self.last_incremental_debug["ref_is_support_eligible"].append(
                bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
            )
            self.last_incremental_debug["match_count_by_ref"].append(valid_count)
            self.last_incremental_debug["match_count_total"] += valid_count
            if is_seed:
                self.last_incremental_debug["match_count_to_seed_keyframes"] += valid_count
            if valid_count > int(self.last_incremental_debug["best_match_num_matches"]):
                self.last_incremental_debug["best_match_num_matches"] = valid_count
                self.last_incremental_debug["best_match_keyframe_id"] = int(keyframe.index)
                self.last_incremental_debug["best_match_is_seed"] = is_seed
            xyz.append(ref_points[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(ref_confidence[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])
            corr_ref_ids.append(torch.full((valid_count,), int(keyframe.index), device="cuda", dtype=torch.long))
            verification_xyz.append(
                verification_points[matches.idx_other[verification_mask]]
            )
            verification_uvs.append(matches.kpts[verification_mask])
            verification_confs.append(
                verification_confidence[
                    matches.idx_other[verification_mask]
                ]
            )
            verification_match_indices.append(matches.idx[verification_mask])
            verification_corr_ref_ids.append(
                torch.full(
                    (verification_valid_count,),
                    int(keyframe.index),
                    device="cuda",
                    dtype=torch.long,
                )
            )
            verification_ref_uvs.append(
                keyframe.desc_kpts.kpts[
                    matches.idx_other[verification_mask]
                ]
            )
            verification_ref_Rts.append(
                verification_reference_Rt[None].expand(
                    verification_valid_count,
                    -1,
                    -1,
                )
            )

        if len(xyz) == 0:
            self.last_incremental_debug["failure_reason"] = "no_2d3d_correspondences"
            return None
        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)
        corr_ref_ids = torch.cat(corr_ref_ids, dim=0)
        verification_xyz = torch.cat(verification_xyz, dim=0)
        verification_uvs = torch.cat(verification_uvs, dim=0)
        verification_confs = torch.cat(verification_confs, dim=0)
        verification_match_indices = torch.cat(
            verification_match_indices,
            dim=0,
        )
        verification_corr_ref_ids = torch.cat(
            verification_corr_ref_ids,
            dim=0,
        )
        verification_ref_uvs = torch.cat(verification_ref_uvs, dim=0)
        verification_ref_Rts = torch.cat(verification_ref_Rts, dim=0)
        sampling_generator = None
        if sampling_seed is not None:
            sampling_generator = torch.Generator(device=xyz.device)
            sampling_generator.manual_seed(int(sampling_seed))
        self.last_incremental_debug["num_2d3d_correspondences"] = int(len(xyz))
        self._record_incremental_pose_candidates(
            verification_match_indices,
            verification_xyz,
            verification_confs,
            verification_uvs,
            verification_corr_ref_ids,
            verification_ref_uvs,
            verification_ref_Rts,
        )
        if self.pose_direct_retry_mode == "pose_safe_v18":
            return self._run_incremental_pose_with_direct_retry_v18(
                keyframes=keyframes,
                index=index,
                is_test=is_test,
                pnp_xyz=xyz,
                pnp_uvs=uvs,
                pnp_confs=confs,
                pnp_match_indices=match_indices,
                pnp_corr_ref_ids=corr_ref_ids,
                reference_initial_Rt=initial_ref_Rts[0],
            )

        # Subsample the points if there are too many
        # 先按置信度采样控制 PnP 输入规模
        if len(xyz) > self.num_pts_pnpransac:
            # 按置信度随机下采样，避免单帧点过多
            selected_indices = torch.multinomial(
                confs,
                self.num_pts_miniba_incr,
                replacement=False,
                generator=sampling_generator,
            )
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]
            corr_ref_ids = corr_ref_ids[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        # 使用上一关键帧作为初始位姿
        reference_initial_Rt = initial_ref_Rts[0]
        Rs6D_init = reference_initial_Rt[:3, :2]
        ts_init = reference_initial_Rt[:3, 3]
        if len(xyz) < 4:
            self.last_incremental_debug["failure_reason"] = "insufficient_correspondences_for_pnp"
            return None
        Rt = None
        inliers = None
        if normalized_registration_solver_mode == "deterministic_opencv_v2":
            try:
                opencv_candidates, opencv_debug = (
                    estimate_opencv_pnp_candidates(
                        xyz,
                        uvs,
                        focal=self.f,
                        centre=self.centre,
                        initial_pose=reference_initial_Rt,
                        max_reprojection_error=float(self.max_pnp_error),
                    )
                )
                selected_candidate, ranking_debug = (
                    select_pose_candidate_by_reprojection(
                        opencv_candidates,
                        xyz,
                        uvs,
                        focal=self.f,
                        centre=self.centre,
                        max_reprojection_error=float(self.max_pnp_error),
                    )
                )
                self.last_incremental_debug["opencv_registration"] = {
                    **opencv_debug,
                    "ranking": ranking_debug,
                }
                if selected_candidate is not None:
                    Rt = selected_candidate["pose"]
                    inliers = selected_candidate["inlier_mask"]
                    self.last_incremental_debug[
                        "registration_solver_selected"
                    ] = str(selected_candidate["name"])
            except Exception as exc:
                self.last_incremental_debug["opencv_registration"] = {
                    "reason": "exception",
                    "exception_type": type(exc).__name__,
                }
        if Rt is None or inliers is None:
            self.last_incremental_debug[
                "registration_solver_fallback"
            ] = bool(
                normalized_registration_solver_mode
                == "deterministic_opencv_v2"
            )
            try:
                pnp_kwargs = (
                    {"generator": sampling_generator}
                    if sampling_generator is not None
                    else {}
                )
                Rt, inliers = self.PnPRANSAC(
                    uvs,
                    xyz,
                    self.f,
                    self.centre,
                    Rs6D_init,
                    ts_init,
                    confs,
                    **pnp_kwargs,
                )
                self.last_incremental_debug[
                    "registration_solver_selected"
                ] = "baseline_cuda_pnp"
            except Exception:
                self.last_incremental_debug[
                    "failure_reason"
                ] = "pnp_ransac_exception"
                return None
        self._last_pnp_Rt = Rt.clone()
        self.last_incremental_debug["pnp_candidate_Rt"] = (
            Rt.detach().cpu().tolist()
        )

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]
        corr_ref_ids = corr_ref_ids[inliers]
        self.last_incremental_debug["num_pnp_inliers"] = int(len(xyz))
        pnp_ref_ids = sorted({int(x) for x in corr_ref_ids.detach().cpu().tolist()})
        self.last_incremental_debug["pnp_ref_keyframe_ids"] = pnp_ref_ids
        self.last_incremental_debug["pnp_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        ]
        self.last_incremental_debug["pnp_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        )
        if len(xyz) < 4:
            self.last_incremental_debug["failure_reason"] = "pnp_inliers_too_few"
            return None

        # Subsample the points if there are too many
        # 为 miniBA 填充固定数量的点
        if len(xyz) >= self.num_pts_miniba_incr:
            random_scores = torch.rand(
                xyz[..., 0].shape,
                device=xyz.device,
                dtype=xyz.dtype,
                generator=sampling_generator,
            )
            selected_indices = torch.topk(
                random_scores,
                self.num_pts_miniba_incr,
                dim=0,
                largest=False,
            )[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
            miniba_ref_ids_tensor = corr_ref_ids[selected_indices]
            corr_ref_ids_ba = miniba_ref_ids_tensor
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)
            miniba_ref_ids_tensor = corr_ref_ids
            corr_ref_ids_ba = torch.cat(
                [
                    corr_ref_ids,
                    -torch.ones(
                        self.num_pts_miniba_incr - len(corr_ref_ids),
                        device=corr_ref_ids.device,
                        dtype=corr_ref_ids.dtype,
                    ),
                ],
                dim=0,
            )
        miniba_ref_ids = sorted({int(x) for x in miniba_ref_ids_tensor.detach().cpu().tolist()})
        self.last_incremental_debug["miniba_ref_keyframe_ids"] = miniba_ref_ids
        self.last_incremental_debug["miniba_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        ]
        self.last_incremental_debug["miniba_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        )

        # Run the initialization
        # 以 PnP 结果为初始化，执行小规模 BA 微调
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        self.last_incremental_debug["num_miniba_inliers"] = int(mask.sum().item())
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]
        self.last_incremental_debug["miniba_candidate_Rt"] = (
            Rt.detach().cpu().tolist()
        )

        # Check if we have sufficiently many inliers
        # 训练阶段要求足够内点以避免错误注册
        if is_test or mask.sum() > self.min_num_inliers:
            valid_ba = uvs_ba[:, 0] >= 0
            if bool(valid_ba.any()):
                support_match_indices = match_indices[selected_indices][valid_ba] if len(xyz) >= self.num_pts_miniba_incr else match_indices
                support_pts3d = xyz_ba[valid_ba]
                support_conf = confs[selected_indices][valid_ba] if len(xyz) >= self.num_pts_miniba_incr else confs
                self._record_incremental_pose_support(
                    support_match_indices,
                    support_pts3d,
                    support_conf,
                    uvs_ba[valid_ba],
                    corr_ref_ids_ba[valid_ba],
                )
            # Return the pose of the current frame
            self.last_incremental_debug["failure_reason"] = ""
            return Rt
        else:
            print("Too few inliers for pose initialization")
            self.last_incremental_debug["failure_reason"] = "miniba_inliers_too_few"
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None

    @staticmethod
    def _pose_verification_motion_limits(
        pose_history: list[object] | None,
    ) -> tuple[float, float, dict[str, float]]:
        poses = []
        for item in pose_history or []:
            pose = item.get("Rt") if isinstance(item, dict) else item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else None
            if isinstance(pose, torch.Tensor):
                poses.append(pose.to(dtype=torch.float32))
        poses = poses[-9:]
        translation_steps = []
        rotation_steps = []
        for previous, current in zip(poses[:-1], poses[1:]):
            delta = pose_correction_magnitude(previous, current)
            if math.isfinite(delta["translation"]):
                translation_steps.append(delta["translation"])
            if math.isfinite(delta["rotation_deg"]):
                rotation_steps.append(delta["rotation_deg"])
        median_translation = float(torch.tensor(translation_steps).median().item()) if translation_steps else 0.0
        median_rotation = float(torch.tensor(rotation_steps).median().item()) if rotation_steps else 0.0
        max_translation = max(0.05, 4.0 * median_translation)
        max_rotation = min(15.0, max(3.0, 3.0 * median_rotation + 2.0))
        return max_rotation, max_translation, {
            "historical_translation_step_median": median_translation,
            "historical_rotation_step_median_deg": median_rotation,
        }

    @torch.no_grad()
    def verify_incremental_pose(
        self,
        initial_Rt: torch.Tensor,
        risk_event: dict[str, object] | None,
        *,
        image_width: int,
        image_height: int,
        pose_history: list[object] | None,
        mad_scale: float = 2.5,
        min_support: int = 24,
        min_relative_median_improvement: float = 0.02,
        max_p90_ratio: float = 1.01,
        min_support_ratio: float = 0.80,
        independent_validation: bool = False,
        candidate_mode: str = "single_v2",
        max_temporal_score_ratio: float = float("inf"),
        pose_evidence_override: dict[str, object] | None = None,
        pose_evidence_pre_error_scale: float = 1.0,
        sampling_seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Risk-triggered pose-only MiniBA verification with exact fallback."""
        started = time.perf_counter()
        event = dict(risk_event or {})
        normalized_candidate_mode = str(candidate_mode or "single_v2").strip().lower()
        if normalized_candidate_mode not in {
            "single_v2",
            "balanced_step_v21",
            "balanced_epipolar_v22",
            "multihypothesis_v23",
            "multiview_relative_v24",
        }:
            raise ValueError(
                f"Unsupported pose verification candidate mode: {candidate_mode}"
            )
        debug: dict[str, object] = {
            "triggered": bool(event.get("verification_trigger", False)),
            "attempted": False,
            "accepted": False,
            "reason": "risk_not_triggered",
            "risk_score": float(event.get("risk_score", 0.0) or 0.0),
            "independent_validation": bool(independent_validation),
            "candidate_mode": normalized_candidate_mode,
            "sampling_seed": (
                int(sampling_seed) if sampling_seed is not None else None
            ),
        }
        if not debug["triggered"]:
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug

        candidates = dict(
            getattr(self, "last_incremental_pose_candidates", {}) or {}
        )
        support = dict(getattr(self, "last_incremental_pose_support", {}) or {})
        override = dict(pose_evidence_override or {})
        if override:
            candidate_source = "stable_anchor_2d3d"
            pose_evidence = override
        else:
            candidate_source = "full_2d3d" if candidates else "miniba_support"
            pose_evidence = candidates or support
        xyz = pose_evidence.get("pts3d")
        uvs = pose_evidence.get("uvs")
        confs = pose_evidence.get("pts_conf")
        corr_ref_ids = pose_evidence.get("corr_ref_ids")
        ref_uvs = pose_evidence.get("ref_uvs")
        ref_Rts = pose_evidence.get("ref_Rts")
        debug["candidate_source"] = candidate_source
        if not all(isinstance(value, torch.Tensor) for value in (xyz, uvs, confs)):
            debug["reason"] = "pose_support_unavailable"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug
        if len(xyz) < 4 or len(uvs) != len(xyz) or len(confs) != len(xyz):
            debug["reason"] = "insufficient_pose_support"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug
        if independent_validation and (
            not isinstance(corr_ref_ids, torch.Tensor)
            or corr_ref_ids.shape != (len(xyz),)
        ):
            debug["reason"] = "reference_ids_unavailable"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug
        epipolar_validation = normalized_candidate_mode == "balanced_epipolar_v22"
        relative_multiview = (
            normalized_candidate_mode == "multiview_relative_v24"
        )
        if (epipolar_validation or relative_multiview) and (
            not isinstance(ref_uvs, torch.Tensor)
            or ref_uvs.shape != (len(xyz), 2)
            or not isinstance(ref_Rts, torch.Tensor)
            or ref_Rts.shape not in {(len(xyz), 3, 4), (len(xyz), 4, 4)}
        ):
            debug["reason"] = "epipolar_evidence_unavailable"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug

        pre_errors, pre_valid = compute_reprojection_errors(
            initial_Rt, xyz, uvs, focal=self.f, centre=self.centre
        )
        pre_error_scale = max(
            1.0,
            min(8.0, float(pose_evidence_pre_error_scale)),
        )
        pre_error_limit = float(self.max_pnp_error) * pre_error_scale
        debug["pose_evidence_pre_error_scale"] = pre_error_scale
        debug["pose_evidence_pre_error_limit"] = pre_error_limit
        evaluation_mask = pre_valid & (pre_errors <= pre_error_limit)
        solve_input_mask = pre_valid
        validation_mask = evaluation_mask
        validation_input_mask = evaluation_mask
        validation_minimum = max(4, int(min_support) // 2)
        if independent_validation:
            split_minimum = validation_minimum
            solve_input_mask, validation_input_mask, split_debug = (
                split_pose_verification_evidence(
                    corr_ref_ids,
                    uvs,
                    evaluation_mask,
                    width=image_width,
                    height=image_height,
                    min_solve_support=split_minimum,
                    min_validation_support=split_minimum,
                )
            )
            debug.update(
                {
                    "split_strategy": split_debug["strategy"],
                    "split_valid": split_debug["valid"],
                    "split_reason": split_debug["reason"],
                    "solve_count": split_debug["solve_count"],
                    "validation_count": split_debug["validation_count"],
                    "solve_reference_ids": split_debug["solve_reference_ids"],
                    "validation_reference_ids": split_debug[
                        "validation_reference_ids"
                    ],
                    "split_reference_count": split_debug["reference_count"],
                    "split_spatial_balance_fallback": split_debug[
                        "spatial_balance_fallback"
                    ],
                }
            )
            if not bool(split_debug["valid"]):
                debug["reason"] = str(split_debug["reason"])
                debug["runtime_seconds"] = time.perf_counter() - started
                return initial_Rt.clone(), debug
            validation_mask, validation_cleaning = select_robust_correspondences(
                pre_errors,
                uvs,
                validation_input_mask,
                width=image_width,
                height=image_height,
                mad_scale=mad_scale,
                max_cutoff=pre_error_limit,
                min_support=split_minimum,
            )
            debug.update(
                {
                    f"validation_cleaning_{key}": value
                    for key, value in validation_cleaning.items()
                }
            )
            if int(validation_mask.sum().item()) < split_minimum:
                debug["reason"] = "insufficient_independent_support"
                debug["runtime_seconds"] = time.perf_counter() - started
                return initial_Rt.clone(), debug
        pre_stats = summarize_reprojection_errors(pre_errors, validation_mask)
        decision_pre_stats = pre_stats
        epipolar_validation_mask: torch.Tensor | None = None
        if epipolar_validation:
            pre_epipolar_errors, pre_epipolar_valid = (
                compute_epipolar_sampson_errors(
                    initial_Rt,
                    uvs,
                    ref_uvs,
                    ref_Rts,
                    focal=self.f,
                    centre=self.centre,
                )
            )
            epipolar_validation_mask = (
                validation_input_mask & pre_epipolar_valid
            )
            if (
                int(epipolar_validation_mask.sum().item())
                < validation_minimum
            ):
                debug["reason"] = "insufficient_epipolar_support"
                debug["runtime_seconds"] = time.perf_counter() - started
                return initial_Rt.clone(), debug
            decision_pre_stats = summarize_reprojection_errors(
                pre_epipolar_errors,
                epipolar_validation_mask,
            )
            debug.update(
                {
                    f"pre_epipolar_{key}": value
                    for key, value in decision_pre_stats.items()
                }
            )
        selected, cleaning = select_robust_correspondences(
            pre_errors,
            uvs,
            solve_input_mask,
            width=image_width,
            height=image_height,
            mad_scale=mad_scale,
            max_cutoff=pre_error_limit,
            min_support=min_support,
        )
        debug.update({f"cleaning_{key}": value for key, value in cleaning.items()})
        debug.update({f"pre_reprojection_{key}": value for key, value in pre_stats.items()})
        if int(selected.sum().item()) < 4:
            debug["reason"] = "insufficient_robust_support"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug

        max_pnp_points = int(
            getattr(self, "num_pts_pnpransac", int(selected.sum().item()))
        )
        balanced_candidate = normalized_candidate_mode in {
            "balanced_step_v21",
            "balanced_epipolar_v22",
            "multihypothesis_v23",
            "multiview_relative_v24",
        }
        if balanced_candidate:
            effective_ref_ids = (
                corr_ref_ids
                if isinstance(corr_ref_ids, torch.Tensor)
                else torch.zeros(
                    len(xyz), dtype=torch.long, device=xyz.device
                )
            )
            selected_indices, balance_debug = (
                select_balanced_correspondence_indices(
                    pre_errors,
                    confs,
                    uvs,
                    effective_ref_ids,
                    selected,
                    width=image_width,
                    height=image_height,
                    max_points=max_pnp_points,
                    grid_rows=4,
                    grid_cols=6,
                    max_reference_fraction=0.40,
                )
            )
            debug.update(
                {
                    f"pnp_balance_{key}": value
                    for key, value in balance_debug.items()
                }
            )
        else:
            selected_indices = torch.where(selected)[0]
            if len(selected_indices) > max_pnp_points:
                strongest = torch.topk(
                    confs[selected_indices], max_pnp_points, largest=True
                ).indices
                selected_indices = selected_indices[strongest]
        xyz_pnp = xyz[selected_indices]
        uvs_pnp = uvs[selected_indices]
        confs_pnp = confs[selected_indices]
        debug["attempted"] = True
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(initial_Rt.device) if initial_Rt.is_cuda else None
        )
        sampling_generator = None
        if sampling_seed is not None:
            sampling_generator = torch.Generator(device=xyz_pnp.device)
            sampling_generator.manual_seed(int(sampling_seed))
        try:
            pnp_kwargs = (
                {"generator": sampling_generator}
                if sampling_generator is not None
                else {}
            )
            pnp_Rt, pnp_inliers = self.PnPRANSAC(
                uvs_pnp,
                xyz_pnp,
                self.f,
                self.centre,
                initial_Rt[:3, :2],
                initial_Rt[:3, 3],
                confs_pnp,
                **pnp_kwargs,
            )
        except Exception:
            debug["reason"] = "verification_pnp_exception"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, initial_Rt.device)
        pnp_inliers = pnp_inliers.to(dtype=torch.bool)
        debug["verification_pnp_inliers"] = int(pnp_inliers.sum().item())
        debug["verification_pnp_Rt"] = pnp_Rt.detach().cpu().tolist()
        if int(pnp_inliers.sum().item()) < 4:
            debug["reason"] = "verification_pnp_inliers_too_few"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug
        xyz_selected = xyz_pnp[pnp_inliers]
        uvs_selected = uvs_pnp[pnp_inliers]
        confs_selected = confs_pnp[pnp_inliers]
        if balanced_candidate:
            source_indices = selected_indices[pnp_inliers]
            effective_ref_ids = (
                corr_ref_ids
                if isinstance(corr_ref_ids, torch.Tensor)
                else torch.zeros(
                    len(xyz), dtype=torch.long, device=xyz.device
                )
            )
            ba_indices, ba_balance_debug = (
                select_balanced_correspondence_indices(
                    pre_errors[source_indices],
                    confs_selected,
                    uvs_selected,
                    effective_ref_ids[source_indices],
                    torch.ones(
                        len(source_indices), dtype=torch.bool, device=xyz.device
                    ),
                    width=image_width,
                    height=image_height,
                    max_points=self.num_pts_miniba_incr,
                    grid_rows=4,
                    grid_cols=6,
                    max_reference_fraction=0.40,
                )
            )
            xyz_selected = xyz_selected[ba_indices]
            uvs_selected = uvs_selected[ba_indices]
            confs_selected = confs_selected[ba_indices]
            debug.update(
                {
                    f"miniba_balance_{key}": value
                    for key, value in ba_balance_debug.items()
                }
            )
        elif len(xyz_selected) > self.num_pts_miniba_incr:
            strongest = torch.topk(
                confs_selected, self.num_pts_miniba_incr, largest=True
            ).indices
            xyz_selected = xyz_selected[strongest]
            uvs_selected = uvs_selected[strongest]
        if len(xyz_selected) < self.num_pts_miniba_incr:
            pad = self.num_pts_miniba_incr - len(xyz_selected)
            xyz_ba = torch.cat(
                [xyz_selected, torch.zeros(pad, 3, device=xyz.device, dtype=xyz.dtype)]
            )
            uvs_ba = torch.cat(
                [uvs_selected, -torch.ones(pad, 2, device=uvs.device, dtype=uvs.dtype)]
            )
        else:
            xyz_ba = xyz_selected
            uvs_ba = uvs_selected

        if initial_Rt.is_cuda:
            torch.cuda.synchronize(initial_Rt.device)
        solver_started = time.perf_counter()
        rotations = pnp_Rt[:3, :2][None]
        translations = pnp_Rt[:3, 3][None]
        rotations, translations, _, _, _, _, solver_mask = self.miniBA_incr(
            rotations, translations, self.f, xyz_ba, self.centre, uvs_ba.view(-1)
        )
        if initial_Rt.is_cuda:
            torch.cuda.synchronize(initial_Rt.device)
        debug["solver_runtime_seconds"] = time.perf_counter() - solver_started
        refined_Rt = torch.eye(4, device=initial_Rt.device, dtype=initial_Rt.dtype)
        refined_Rt[:3, :3] = sixD2mtx(rotations)[0]
        refined_Rt[:3, 3] = translations[0]
        solver_candidates: list[dict[str, object]] = [
            {"name": "cuda_pnp_miniba", "pose": refined_Rt}
        ]
        if normalized_candidate_mode in {
            "multihypothesis_v23",
            "multiview_relative_v24",
        }:
            opencv_candidates, opencv_debug = estimate_opencv_pnp_candidates(
                xyz_pnp,
                uvs_pnp,
                focal=self.f,
                centre=self.centre,
                initial_pose=initial_Rt,
                max_reprojection_error=float(self.max_pnp_error),
            )
            solver_candidates.extend(opencv_candidates)
            debug.update(
                {
                    f"opencv_pnp_{key}": value
                    for key, value in opencv_debug.items()
                }
            )
        if relative_multiview:
            relative_indices = torch.where(solve_input_mask)[0]
            relative_candidates, relative_debug = (
                estimate_multireference_relative_pose_candidates(
                    uvs[relative_indices],
                    ref_uvs[relative_indices],
                    ref_Rts[relative_indices],
                    corr_ref_ids[relative_indices],
                    focal=self.f,
                    centre=self.centre,
                    initial_pose=initial_Rt,
                    max_epipolar_error=min(
                        2.0, max(0.5, float(self.max_pnp_error) * 0.25)
                    ),
                )
            )
            solver_candidates.extend(relative_candidates)
            debug.update(
                {
                    f"relative_pose_{key}": value
                    for key, value in relative_debug.items()
                }
            )
        if normalized_candidate_mode in {
            "multihypothesis_v23",
            "multiview_relative_v24",
        }:
            debug["solver_candidates"] = [
                {
                    "name": str(item["name"]),
                    "pose": item["pose"].detach().cpu().tolist(),
                    "inlier_count": int(item.get("inlier_count", 0)),
                }
                for item in solver_candidates
                if isinstance(item.get("pose"), torch.Tensor)
            ]
        max_rotation, max_translation, history_debug = self._pose_verification_motion_limits(
            pose_history
        )
        temporal_prediction, temporal_prediction_debug = (
            predict_constant_velocity_pose(
                pose_history,
                current_frame_id=int(event.get("frame_id", -1)),
                like_pose=initial_Rt,
            )
        )
        debug.update(
            {
                f"temporal_prediction_{key}": value
                for key, value in temporal_prediction_debug.items()
            }
        )
        initial_temporal_stats: dict[str, float] | None = None
        if temporal_prediction is not None:
            initial_temporal_stats = temporal_pose_consistency(
                initial_Rt,
                temporal_prediction,
                pose_history,
            )
            debug.update(
                {
                    f"pre_temporal_{key}": value
                    for key, value in initial_temporal_stats.items()
                }
            )
        debug["max_temporal_score_ratio"] = float(
            max_temporal_score_ratio
        )
        selected_alpha = 1.0
        selected_candidate = refined_Rt
        selected_candidate_source = "cuda_pnp_miniba"
        step_candidates: list[dict[str, object]] = []
        if balanced_candidate:
            evaluated_candidates: list[
                tuple[
                    tuple[float, float, float],
                    str,
                    float,
                    torch.Tensor,
                    dict[str, object],
                    dict[str, float | int],
                ]
            ] = []
            for solver_candidate in solver_candidates:
                candidate_source = str(solver_candidate["name"])
                base_candidate = solver_candidate["pose"]
                if not isinstance(base_candidate, torch.Tensor):
                    continue
                for alpha in (0.25, 0.50, 0.75, 1.00):
                    candidate_Rt = interpolate_world_to_camera_pose(
                        initial_Rt, base_candidate, alpha
                    )
                    candidate_errors, candidate_valid = compute_reprojection_errors(
                        candidate_Rt, xyz, uvs, focal=self.f, centre=self.centre
                    )
                    candidate_reprojection_stats = summarize_reprojection_errors(
                        candidate_errors, candidate_valid & validation_mask
                    )
                    candidate_stats = candidate_reprojection_stats
                    if epipolar_validation:
                        candidate_epipolar_errors, candidate_epipolar_valid = (
                            compute_epipolar_sampson_errors(
                                candidate_Rt,
                                uvs,
                                ref_uvs,
                                ref_Rts,
                                focal=self.f,
                                centre=self.centre,
                            )
                        )
                        candidate_stats = summarize_reprojection_errors(
                            candidate_epipolar_errors,
                            candidate_epipolar_valid
                            & epipolar_validation_mask,
                        )
                    candidate_correction = pose_correction_magnitude(
                        initial_Rt, candidate_Rt
                    )
                    candidate_decision = decide_pose_refinement(
                        pre=decision_pre_stats,
                        post=candidate_stats,
                        correction=candidate_correction,
                        max_rotation_deg=max_rotation,
                        max_translation=max_translation,
                        min_relative_median_improvement=min_relative_median_improvement,
                        max_p90_ratio=max_p90_ratio,
                        min_support_ratio=min_support_ratio,
                    )
                    candidate_temporal_stats: dict[str, float] | None = None
                    if (
                        temporal_prediction is not None
                        and initial_temporal_stats is not None
                    ):
                        candidate_temporal_stats = temporal_pose_consistency(
                            candidate_Rt,
                            temporal_prediction,
                            pose_history,
                        )
                        temporal_ratio = float(
                            candidate_temporal_stats["score"]
                            / max(initial_temporal_stats["score"], 1e-8)
                        )
                        candidate_decision["temporal_score_ratio"] = (
                            temporal_ratio
                        )
                        if (
                            bool(candidate_decision.get("accepted", False))
                            and temporal_ratio
                            > float(max_temporal_score_ratio)
                        ):
                            candidate_decision["accepted"] = False
                            candidate_decision["reason"] = (
                                "temporal_inconsistent"
                            )
                    row = {
                        "source": candidate_source,
                        "alpha": float(alpha),
                        **{
                            f"post_{key}": value
                            for key, value in candidate_stats.items()
                        },
                        **(
                            {
                                f"post_reprojection_{key}": value
                                for key, value in candidate_reprojection_stats.items()
                            }
                            if epipolar_validation
                            else {}
                        ),
                        **candidate_decision,
                        **(
                            {
                                f"temporal_{key}": value
                                for key, value in candidate_temporal_stats.items()
                            }
                            if candidate_temporal_stats is not None
                            else {}
                        ),
                    }
                    step_candidates.append(row)
                    if bool(candidate_decision.get("accepted", False)):
                        if normalized_candidate_mode in {
                            "multihypothesis_v23",
                            "multiview_relative_v24",
                        }:
                            rank = validation_candidate_rank(
                                decision_pre_stats,
                                candidate_stats,
                            )
                        else:
                            rank = (
                                float(
                                    candidate_stats.get(
                                        "median", float("inf")
                                    )
                                ),
                                float(
                                    candidate_stats.get(
                                        "mean", float("inf")
                                    )
                                ),
                                float(
                                    candidate_stats.get(
                                        "p90", float("inf")
                                    )
                                ),
                            )
                        evaluated_candidates.append(
                            (
                                rank,
                                candidate_source,
                                float(alpha),
                                candidate_Rt,
                                candidate_decision,
                                candidate_stats,
                            )
                        )
            if evaluated_candidates:
                (
                    _,
                    selected_candidate_source,
                    selected_alpha,
                    selected_candidate,
                    decision,
                    post_stats,
                ) = min(evaluated_candidates, key=lambda item: item[0])
            else:
                primary_rows = [
                    row
                    for row in step_candidates
                    if row.get("source") == "cuda_pnp_miniba"
                ]
                full_row = primary_rows[-1]
                decision = {
                    key: value
                    for key, value in full_row.items()
                    if key not in {"source", "alpha"}
                    and not key.startswith("post_")
                }
                post_stats = {
                    "valid_count": int(full_row["post_valid_count"]),
                    "mean": float(full_row["post_mean"]),
                    "median": float(full_row["post_median"]),
                    "p90": float(full_row["post_p90"]),
                    "max": float(full_row["post_max"]),
                }
                selected_candidate = refined_Rt
                selected_alpha = 1.0
                selected_candidate_source = "cuda_pnp_miniba"
        else:
            post_errors, post_valid = compute_reprojection_errors(
                refined_Rt, xyz, uvs, focal=self.f, centre=self.centre
            )
            post_stats = summarize_reprojection_errors(
                post_errors, post_valid & validation_mask
            )
            correction = pose_correction_magnitude(initial_Rt, refined_Rt)
            decision = decide_pose_refinement(
                pre=pre_stats,
                post=post_stats,
                correction=correction,
                max_rotation_deg=max_rotation,
                max_translation=max_translation,
                min_relative_median_improvement=min_relative_median_improvement,
                max_p90_ratio=max_p90_ratio,
                min_support_ratio=min_support_ratio,
            )
        debug.update(decision)
        debug.update(history_debug)
        selected_reprojection_errors, selected_reprojection_valid = (
            compute_reprojection_errors(
                selected_candidate,
                xyz,
                uvs,
                focal=self.f,
                centre=self.centre,
            )
        )
        selected_reprojection_stats = summarize_reprojection_errors(
            selected_reprojection_errors,
            selected_reprojection_valid & validation_mask,
        )
        debug.update(
            {
                f"post_reprojection_{key}": value
                for key, value in selected_reprojection_stats.items()
            }
        )
        if epipolar_validation:
            debug.update(
                {
                    f"post_epipolar_{key}": value
                    for key, value in post_stats.items()
                }
            )
        debug["verification_miniba_inlier_coordinates"] = int(solver_mask.sum().item())
        debug["initial_Rt"] = initial_Rt.detach().cpu().tolist()
        debug["solver_refined_Rt"] = refined_Rt.detach().cpu().tolist()
        debug["selected_step_alpha"] = float(selected_alpha)
        debug["selected_candidate_source"] = selected_candidate_source
        debug["selected_temporal_score_ratio"] = float(
            decision.get("temporal_score_ratio", float("inf"))
        )
        debug["step_candidates"] = step_candidates
        debug["refined_Rt"] = selected_candidate.detach().cpu().tolist()
        selected_Rt = choose_verified_pose(
            initial_Rt, selected_candidate, decision
        )
        debug["final_Rt"] = selected_Rt.detach().cpu().tolist()
        debug["runtime_seconds"] = time.perf_counter() - started
        return selected_Rt, debug

    @torch.no_grad()
    def _collect_delayed_validation_evidence(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz: list[torch.Tensor] = []
        uvs: list[torch.Tensor] = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            if bool(mask.any()):
                xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
                uvs.append(matches.kpts[mask])
        if not xyz:
            device = curr_desc_kpts.kpts.device
            return (
                torch.empty((0, 3), device=device),
                torch.empty((0, 2), device=device),
            )
        return torch.cat(xyz, dim=0), torch.cat(uvs, dim=0)

    @torch.no_grad()
    def verify_delayed_incremental_pose(
        self,
        initial_Rt: torch.Tensor,
        solve_keyframes: list[Keyframe],
        validation_keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        *,
        index: int,
        curr_img: torch.Tensor,
        min_support: int = 24,
        min_relative_improvement: float = 0.03,
        max_mean_ratio: float = 0.99,
        max_p90_ratio: float = 1.01,
        max_translation: float = 0.10,
        max_rotation_deg: float = 3.0,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Re-section one pose using mature points and disjoint validation refs."""
        started = time.perf_counter()
        debug: dict[str, object] = {
            "attempted": False,
            "accepted": False,
            "reason": "insufficient_reference_groups",
            "solve_reference_ids": [int(kf.index) for kf in solve_keyframes],
            "validation_reference_ids": [
                int(kf.index) for kf in validation_keyframes
            ],
        }
        if len(solve_keyframes) < 2 or len(validation_keyframes) < 2:
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug

        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(initial_Rt.device)
        try:
            seed = 0xA51E + int(index)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            candidate_Rt = self.initialize_incremental(
                solve_keyframes,
                curr_desc_kpts,
                int(index),
                True,
                curr_img,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, initial_Rt.device)
        debug["attempted"] = True
        debug["candidate_pose_debug"] = dict(self.last_incremental_debug)
        if candidate_Rt is None:
            debug["reason"] = "candidate_generation_failed"
            debug["runtime_seconds"] = time.perf_counter() - started
            return initial_Rt.clone(), debug

        xyz, uvs = self._collect_delayed_validation_evidence(
            validation_keyframes,
            curr_desc_kpts,
            int(index),
        )
        pre_errors, pre_valid = compute_reprojection_errors(
            initial_Rt, xyz, uvs, focal=self.f, centre=self.centre
        )
        post_errors, post_valid = compute_reprojection_errors(
            candidate_Rt, xyz, uvs, focal=self.f, centre=self.centre
        )
        correction = pose_correction_magnitude(initial_Rt, candidate_Rt)
        decision = summarize_candidate_validation(
            pre_errors,
            post_errors,
            pre_valid,
            post_valid,
            max_error=float(self.max_pnp_error),
            min_support=int(min_support),
            min_relative_improvement=float(min_relative_improvement),
            max_mean_ratio=float(max_mean_ratio),
            max_p90_ratio=float(max_p90_ratio),
            correction_translation=float(correction["translation"]),
            correction_rotation_deg=float(correction["rotation_deg"]),
            max_translation=float(max_translation),
            max_rotation_deg=float(max_rotation_deg),
        )
        debug.update(decision)
        debug["initial_Rt"] = initial_Rt.detach().cpu().tolist()
        debug["candidate_Rt"] = candidate_Rt.detach().cpu().tolist()
        debug["final_Rt"] = (
            candidate_Rt if bool(decision["accepted"]) else initial_Rt
        ).detach().cpu().tolist()
        debug["runtime_seconds"] = time.perf_counter() - started
        return (
            candidate_Rt if bool(decision["accepted"]) else initial_Rt.clone(),
            debug,
        )

    def _keyframe_has_pt3d_count(self, keyframe: Keyframe) -> int:
        return int(keyframe.desc_kpts.has_pt3d.sum().item())

    def _sort_recovery_reference_keyframes(
        self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints, index: int
    ) -> list[Keyframe]:
        scored: list[tuple[int, int, int, int, Keyframe]] = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            valid_count = int(mask.sum().item())
            raw_count = int(len(matches.kpts))
            has_pt3d_total = self._keyframe_has_pt3d_count(keyframe)
            is_seed = int(bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False)))
            is_support = int(
                bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
            )
            scored.append((has_pt3d_total, valid_count, is_support, is_seed, raw_count, keyframe))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4]))
        return [item[5] for item in scored]

    def _ref_conf_weight_from_probe_stat(self, stat: dict[str, object]) -> float:
        ratio = float(stat.get("ref_pnp_inlier_ratio", 0.0) or 0.0)
        valid = float(stat.get("ref_valid_2d3d_count", 0) or 0.0)
        score = float(stat.get("ref_consensus_score", 0.0) or 0.0)
        return float(
            max(
                0.12,
                min(
                    1.0,
                    0.20 + 0.35 * min(score, 2.0) / 2.0 + 0.30 * min(ratio * 12.0, 1.0) + 0.15 * min(valid / 200.0, 1.0),
                ),
            )
        )

    def _per_ref_correspondence_cap(self, stat: dict[str, object]) -> int | None:
        valid = int(stat.get("ref_valid_2d3d_count", 0) or 0)
        if valid <= 0:
            return 0
        ratio = float(stat.get("ref_pnp_inlier_ratio", 0.0) or 0.0)
        has_pt3d = int(stat.get("ref_has_pt3d_count", 0) or 0)
        if has_pt3d > 5000 and ratio < 0.01:
            return max(16, int(valid * 0.08))
        if ratio < 0.02:
            return max(24, int(valid * min(0.35, 0.10 + ratio * 10.0)))
        return None

    def _collect_recovery_correspondences(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        Rt_guess_for_triangulation: torch.Tensor | None,
        ref_conf_weights: dict[int, float] | None = None,
        ref_correspondence_caps: dict[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        support_debug: dict[str, object] = {
            "raw_2d2d_match_count": 0,
            "verified_2d2d_match_count": 0,
            "has_pt3d_match_count": 0,
            "temporary_3d_support_used": False,
            "temporary_3d_support_count": 0,
            "valid_2d3d_from_direct_refs": 0,
            "valid_2d3d_from_recovery_refs": 0,
            "valid_2d3d_from_seed_refs": 0,
            "per_ref_3d_bearing_counts": [],
            "per_ref_raw_match_counts": [],
            "selected_refs_by_3d_support": [int(kf.index) for kf in keyframes],
        }
        xyz: list[torch.Tensor] = []
        uvs: list[torch.Tensor] = []
        confs: list[torch.Tensor] = []
        corr_ref_ids: list[torch.Tensor] = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            raw_count = int(len(matches.kpts))
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            valid_count = int(mask.sum().item())
            commit_origin = str(keyframe.info.get("_paper_aligned_commit_origin", "unknown"))
            is_recovery = commit_origin in {"true_recovery_commit", "early_seed_recovery_commit"}
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            support_debug["raw_2d2d_match_count"] = int(support_debug["raw_2d2d_match_count"]) + raw_count
            support_debug["has_pt3d_match_count"] = int(support_debug["has_pt3d_match_count"]) + valid_count
            support_debug["per_ref_3d_bearing_counts"].append(valid_count)
            support_debug["per_ref_raw_match_counts"].append(raw_count)
            if is_recovery:
                support_debug["valid_2d3d_from_recovery_refs"] = int(
                    support_debug["valid_2d3d_from_recovery_refs"]
                ) + valid_count
            elif is_seed:
                support_debug["valid_2d3d_from_seed_refs"] = int(support_debug["valid_2d3d_from_seed_refs"]) + valid_count
            else:
                support_debug["valid_2d3d_from_direct_refs"] = int(
                    support_debug["valid_2d3d_from_direct_refs"]
                ) + valid_count
            if valid_count == 0:
                continue
            ref_pts_conf = keyframe.desc_kpts.pts_conf[matches.idx_other[mask]]
            ref_xyz = keyframe.desc_kpts.pts3d[matches.idx_other[mask]]
            ref_uvs = matches.kpts[mask]
            kid = int(keyframe.index)
            if ref_correspondence_caps is not None and kid in ref_correspondence_caps:
                cap = int(ref_correspondence_caps[kid])
                if cap <= 0:
                    continue
                if valid_count > cap:
                    top_idx = torch.topk(ref_pts_conf, cap, largest=True).indices
                    ref_xyz = ref_xyz[top_idx]
                    ref_uvs = ref_uvs[top_idx]
                    ref_pts_conf = ref_pts_conf[top_idx]
                    valid_count = cap
            if ref_conf_weights is not None:
                ref_pts_conf = ref_pts_conf * float(ref_conf_weights.get(kid, 0.5))
            xyz.append(ref_xyz)
            uvs.append(ref_uvs)
            confs.append(ref_pts_conf)
            corr_ref_ids.append(
                torch.full((valid_count,), kid, device="cuda", dtype=torch.long)
            )
        if Rt_guess_for_triangulation is not None:
            extra_xyz, extra_uvs, extra_confs = self._triangulate_recovery_correspondences(
                keyframes, curr_desc_kpts, index, Rt_guess_for_triangulation
            )
            if len(extra_xyz) > 0:
                support_debug["temporary_3d_support_used"] = True
                support_debug["temporary_3d_support_count"] = int(len(extra_xyz))
                xyz.append(extra_xyz)
                uvs.append(extra_uvs)
                confs.append(extra_confs)
                corr_ref_ids.append(
                    torch.full((len(extra_xyz),), int(keyframes[0].index), device="cuda", dtype=torch.long)
                )
        if not xyz:
            support_debug["verified_2d2d_match_count"] = 0
            return (
                torch.zeros(0, 3, device="cuda"),
                torch.zeros(0, 2, device="cuda"),
                torch.zeros(0, device="cuda"),
                torch.zeros(0, device="cuda", dtype=torch.long),
                support_debug,
            )
        xyz_cat = torch.cat(xyz, dim=0)
        uvs_cat = torch.cat(uvs, dim=0)
        confs_cat = torch.cat(confs, dim=0)
        corr_ref_ids_cat = torch.cat(corr_ref_ids, dim=0)
        support_debug["verified_2d2d_match_count"] = int(len(xyz_cat))
        return xyz_cat, uvs_cat, confs_cat, corr_ref_ids_cat, support_debug

    def _run_pose_from_correspondences(
        self,
        keyframes: list[Keyframe],
        xyz_cat: torch.Tensor,
        uvs_cat: torch.Tensor,
        confs_cat: torch.Tensor,
        corr_ref_ids_cat: torch.Tensor,
        index: int,
        is_test: bool,
    ) -> torch.Tensor | None:
        self.last_incremental_debug["num_2d3d_correspondences"] = int(len(xyz_cat))
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "insufficient_correspondences_for_pnp"
            return None
        if len(xyz_cat) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(
                confs_cat, min(self.num_pts_pnpransac, len(xyz_cat)), replacement=False
            )
            xyz_cat = xyz_cat[selected_indices]
            uvs_cat = uvs_cat[selected_indices]
            confs_cat = confs_cat[selected_indices]
            corr_ref_ids_cat = corr_ref_ids_cat[selected_indices]
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        try:
            Rt, inliers = self.PnPRANSAC(uvs_cat, xyz_cat, self.f, self.centre, Rs6D_init, ts_init, confs_cat)
        except Exception:
            self.last_incremental_debug["failure_reason"] = "pnp_ransac_exception"
            return None
        xyz_cat = xyz_cat[inliers]
        uvs_cat = uvs_cat[inliers]
        confs_cat = confs_cat[inliers]
        corr_ref_ids_cat = corr_ref_ids_cat[inliers]
        self.last_incremental_debug["num_pnp_inliers"] = int(len(xyz_cat))
        pnp_ref_ids = sorted({int(x) for x in corr_ref_ids_cat.detach().cpu().tolist()})
        self.last_incremental_debug["pnp_ref_keyframe_ids"] = pnp_ref_ids
        self.last_incremental_debug["pnp_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        ]
        self.last_incremental_debug["pnp_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        )
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "pnp_inliers_too_few"
            self._last_pnp_Rt = None
            return None
        self._last_pnp_Rt = Rt.clone()
        if len(xyz_cat) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(
                torch.rand_like(xyz_cat[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False
            )[1]
            xyz_ba = xyz_cat[selected_indices]
            uvs_ba = uvs_cat[selected_indices]
            miniba_ref_ids_tensor = corr_ref_ids_cat[selected_indices]
        else:
            xyz_ba = torch.cat(
                [xyz_cat, torch.zeros(self.num_pts_miniba_incr - len(xyz_cat), 3, device="cuda")], dim=0
            )
            uvs_ba = torch.cat(
                [uvs_cat, -torch.ones(self.num_pts_miniba_incr - len(uvs_cat), 2, device="cuda")], dim=0
            )
            miniba_ref_ids_tensor = corr_ref_ids_cat
        miniba_ref_ids = sorted({int(x) for x in miniba_ref_ids_tensor.detach().cpu().tolist()})
        self.last_incremental_debug["miniba_ref_keyframe_ids"] = miniba_ref_ids
        self.last_incremental_debug["miniba_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        ]
        self.last_incremental_debug["miniba_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        )
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(
            Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1)
        )
        self.last_incremental_debug["num_miniba_inliers"] = int(mask.sum().item())
        Rt_out = torch.eye(4, device="cuda")
        Rt_out[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt_out[:3, 3] = ts[0]
        if is_test or mask.sum() > self.min_num_inliers:
            self.last_incremental_debug["failure_reason"] = ""
            return Rt_out
        self.last_incremental_debug["failure_reason"] = "miniba_inliers_too_few"
        for keyframe in keyframes:
            keyframe.desc_kpts.matches.pop(index, None)
        return None

    def _triangulate_recovery_correspondences(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        Rt_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extra_xyz: list[torch.Tensor] = []
        extra_uvs: list[torch.Tensor] = []
        extra_confs: list[torch.Tensor] = []
        max_error = float(self.triangulator.max_error.item())
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            if len(matches.kpts) == 0:
                continue
            missing = ~keyframe.desc_kpts.has_pt3d[matches.idx_other]
            if int(missing.sum().item()) == 0:
                continue
            uv_src = matches.kpts[missing]
            uv_ref = matches.kpts_other[missing]
            Rt_ref = keyframe.get_Rt()
            rel_Rt = Rt_source @ torch.linalg.inv_ex(Rt_ref)[0]
            pts3d, dis, err = matches_to_points(
                uv_src, uv_ref, rel_Rt[:3, :3], rel_Rt[:3, 3], self.f, self.centre
            )
            valid = (
                (pts3d[:, 2] > 1e-6)
                & (dis > 0.0)
                & (err < max_error)
                & (~pts3d.isnan().any(dim=-1))
            )
            if int(valid.sum().item()) == 0:
                continue
            extra_xyz.append(pts3d[valid])
            extra_uvs.append(uv_src[valid])
            extra_confs.append(torch.ones(int(valid.sum().item()), device="cuda"))
        if not extra_xyz:
            return (
                torch.zeros(0, 3, device="cuda"),
                torch.zeros(0, 2, device="cuda"),
                torch.zeros(0, device="cuda"),
            )
        return torch.cat(extra_xyz, dim=0), torch.cat(extra_uvs, dim=0), torch.cat(extra_confs, dim=0)

    def _run_incremental_pose_core(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
    ) -> torch.Tensor | None:
        self.last_incremental_debug = {
            "failure_reason": "",
            "num_2d3d_correspondences": 0,
            "num_pnp_inliers": 0,
            "num_miniba_inliers": 0,
            "ref_keyframe_ids": [],
            "ref_source_frame_ids": [],
            "ref_commit_origin": [],
            "ref_is_recovery": [],
            "ref_is_seed": [],
            "ref_is_support_eligible": [],
            "match_count_by_ref": [],
            "match_count_total": 0,
            "match_count_to_seed_keyframes": 0,
            "best_match_keyframe_id": -1,
            "best_match_is_seed": False,
            "best_match_num_matches": 0,
            "pnp_ref_keyframe_ids": [],
            "pnp_ref_source_frame_ids": [],
            "pnp_ref_contains_seed": False,
            "miniba_ref_keyframe_ids": [],
            "miniba_ref_source_frame_ids": [],
            "miniba_ref_contains_seed": False,
        }
        xyz: list[torch.Tensor] = []
        uvs: list[torch.Tensor] = []
        confs: list[torch.Tensor] = []
        match_indices: list[torch.Tensor] = []
        corr_ref_ids: list[torch.Tensor] = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            valid_count = int(mask.sum().item())
            source_frame_id = int(keyframe.info.get("_paper_aligned_source_frame_id", keyframe.index))
            commit_origin = str(keyframe.info.get("_paper_aligned_commit_origin", "unknown"))
            is_recovery = commit_origin in {"true_recovery_commit", "early_seed_recovery_commit"}
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            self.last_incremental_debug["ref_keyframe_ids"].append(int(keyframe.index))
            self.last_incremental_debug["ref_source_frame_ids"].append(source_frame_id)
            self.last_incremental_debug["ref_commit_origin"].append(commit_origin)
            self.last_incremental_debug["ref_is_recovery"].append(is_recovery)
            self.last_incremental_debug["ref_is_seed"].append(is_seed)
            self.last_incremental_debug["ref_is_support_eligible"].append(
                bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
            )
            self.last_incremental_debug["match_count_by_ref"].append(valid_count)
            self.last_incremental_debug["match_count_total"] += valid_count
            if is_seed:
                self.last_incremental_debug["match_count_to_seed_keyframes"] += valid_count
            if valid_count > int(self.last_incremental_debug["best_match_num_matches"]):
                self.last_incremental_debug["best_match_num_matches"] = valid_count
                self.last_incremental_debug["best_match_keyframe_id"] = int(keyframe.index)
                self.last_incremental_debug["best_match_is_seed"] = is_seed
            if valid_count == 0:
                continue
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])
            corr_ref_ids.append(
                torch.full((valid_count,), int(keyframe.index), device="cuda", dtype=torch.long)
            )

        if len(xyz) == 0:
            self.last_incremental_debug["failure_reason"] = "no_2d3d_correspondences"
            return None
        xyz_cat = torch.cat(xyz, dim=0)
        uvs_cat = torch.cat(uvs, dim=0)
        confs_cat = torch.cat(confs, dim=0)
        match_indices_cat = torch.cat(match_indices, dim=0)
        corr_ref_ids_cat = torch.cat(corr_ref_ids, dim=0)
        self.last_incremental_debug["num_2d3d_correspondences"] = int(len(xyz_cat))

        if len(xyz_cat) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(confs_cat, self.num_pts_miniba_incr, replacement=False)
            xyz_cat = xyz_cat[selected_indices]
            uvs_cat = uvs_cat[selected_indices]
            confs_cat = confs_cat[selected_indices]
            match_indices_cat = match_indices_cat[selected_indices]
            corr_ref_ids_cat = corr_ref_ids_cat[selected_indices]

        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "insufficient_correspondences_for_pnp"
            return None
        try:
            Rt, inliers = self.PnPRANSAC(uvs_cat, xyz_cat, self.f, self.centre, Rs6D_init, ts_init, confs_cat)
        except Exception:
            self.last_incremental_debug["failure_reason"] = "pnp_ransac_exception"
            return None

        xyz_cat = xyz_cat[inliers]
        uvs_cat = uvs_cat[inliers]
        confs_cat = confs_cat[inliers]
        match_indices_cat = match_indices_cat[inliers]
        corr_ref_ids_cat = corr_ref_ids_cat[inliers]
        self.last_incremental_debug["num_pnp_inliers"] = int(len(xyz_cat))
        pnp_ref_ids = sorted({int(x) for x in corr_ref_ids_cat.detach().cpu().tolist()})
        self.last_incremental_debug["pnp_ref_keyframe_ids"] = pnp_ref_ids
        self.last_incremental_debug["pnp_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        ]
        self.last_incremental_debug["pnp_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        )
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "pnp_inliers_too_few"
            self._last_pnp_Rt = None
            return None

        self._last_pnp_Rt = Rt.clone()

        if len(xyz_cat) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(
                torch.rand_like(xyz_cat[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False
            )[1]
            xyz_ba = xyz_cat[selected_indices]
            uvs_ba = uvs_cat[selected_indices]
            miniba_ref_ids_tensor = corr_ref_ids_cat[selected_indices]
        else:
            xyz_ba = torch.cat(
                [xyz_cat, torch.zeros(self.num_pts_miniba_incr - len(xyz_cat), 3, device="cuda")], dim=0
            )
            uvs_ba = torch.cat(
                [uvs_cat, -torch.ones(self.num_pts_miniba_incr - len(uvs_cat), 2, device="cuda")], dim=0
            )
            miniba_ref_ids_tensor = corr_ref_ids_cat
        miniba_ref_ids = sorted({int(x) for x in miniba_ref_ids_tensor.detach().cpu().tolist()})
        self.last_incremental_debug["miniba_ref_keyframe_ids"] = miniba_ref_ids
        self.last_incremental_debug["miniba_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        ]
        self.last_incremental_debug["miniba_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        )

        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        self.last_incremental_debug["num_miniba_inliers"] = int(mask.sum().item())
        Rt_out = torch.eye(4, device="cuda")
        Rt_out[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt_out[:3, 3] = ts[0]
        if is_test or mask.sum() > self.min_num_inliers:
            self.last_incremental_debug["failure_reason"] = ""
            return Rt_out
        self.last_incremental_debug["failure_reason"] = "miniba_inliers_too_few"
        for keyframe in keyframes:
            keyframe.desc_kpts.matches.pop(index, None)
        return None

    def _ref_source_frame_id(self, keyframe: Keyframe) -> int:
        return int(keyframe.info.get("_paper_aligned_source_frame_id", keyframe.index))

    def _ref_anchor_id(self, keyframe: Keyframe) -> int:
        return int(keyframe.info.get("_paper_aligned_anchor_id", -1))

    @torch.no_grad()
    def _probe_ref_pnp_inliers(
        self,
        keyframe: Keyframe,
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        init_keyframe: Keyframe,
    ) -> tuple[int, int, float]:
        matches = self.matcher(
            curr_desc_kpts,
            keyframe.desc_kpts,
            remove_outliers=True,
            update_kpts_flag="all",
            kID=index,
            kID_other=keyframe.index,
        )
        mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
        valid_count = int(mask.sum().item())
        if valid_count < 4:
            return 0, valid_count, 0.0
        xyz = keyframe.desc_kpts.pts3d[matches.idx_other[mask]]
        uvs = matches.kpts[mask]
        confs = keyframe.desc_kpts.pts_conf[matches.idx_other[mask]]
        try:
            _, inliers = self.PnPRANSAC(
                uvs,
                xyz,
                self.f,
                self.centre,
                init_keyframe.rW2C,
                init_keyframe.tW2C,
                confs,
            )
            inlier_count = int(inliers.sum().item())
        except Exception:
            return 0, valid_count, 0.0
        ratio = float(inlier_count) / max(valid_count, 1)
        return inlier_count, valid_count, ratio

    def _recovery_ref_consensus_score(
        self,
        keyframe: Keyframe,
        valid_2d3d: int,
        pnp_inliers: int,
        pnp_ratio: float,
        defer_source_frame_id: int,
        anchor_mode_id: int,
    ) -> float:
        has_pt3d_total = self._keyframe_has_pt3d_count(keyframe)
        src_dist = abs(self._ref_source_frame_id(keyframe) - int(defer_source_frame_id))
        dist_weight = 1.0 / (1.0 + float(src_dist) / 50.0)
        anchor_match = 1.0 if self._ref_anchor_id(keyframe) == anchor_mode_id and anchor_mode_id >= 0 else 0.5
        is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
        is_support = bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
        seed_bonus = 0.05 if is_seed or is_support else 0.0
        if bool(keyframe.info.get("_paper_aligned_bridge_ref", False)):
            seed_bonus += 0.30
        high_pt3d_penalty = 0.25 if has_pt3d_total > 3000 and pnp_ratio < self.recovery_probe_min_inlier_ratio else 0.0
        return (
            0.35 * min(float(valid_2d3d) / 500.0, 2.0)
            + 0.45 * min(pnp_ratio * 10.0, 2.0)
            + 0.10 * dist_weight
            + 0.10 * anchor_match
            + seed_bonus
            - high_pt3d_penalty
        )

    def _select_coherent_ref_subset(
        self,
        keyframes: list[Keyframe],
        ref_stats: list[dict[str, object]],
        defer_source_frame_id: int,
    ) -> tuple[list[Keyframe], list[dict[str, object]], list[int], list[str], str]:
        if not keyframes:
            return [], [], [], [], "no_candidate_refs"
        stat_by_kid = {int(s.get("ref_keyframe_id", -1)): s for s in ref_stats}
        ranked = sorted(
            zip(keyframes, ref_stats),
            key=lambda item: -float(item[1].get("consensus_score", 0.0)),
        )
        selected: list[Keyframe] = []
        subset_trace: list[dict[str, object]] = []
        excluded_ids: list[int] = []
        excluded_reasons: list[str] = []
        has_direct_or_support = False

        def is_support_ref(keyframe: Keyframe) -> bool:
            return bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False)) or str(
                keyframe.info.get("_paper_aligned_commit_origin", "")
            ) in {"direct_admit", "direct"}

        for keyframe, stat in ranked:
            kid = int(keyframe.index)
            ratio = float(stat.get("ref_pnp_inlier_ratio", 0.0) or 0.0)
            valid = int(stat.get("ref_valid_2d3d_count", 0) or 0)
            pnp_inl = int(stat.get("ref_pnp_inlier_count", 0) or 0)
            has_pt3d = int(stat.get("ref_has_pt3d_count", 0) or 0)
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            is_support = is_support_ref(keyframe)
            extreme_noise = has_pt3d > 6000 and ratio < 0.005 and pnp_inl == 0
            if extreme_noise:
                excluded_ids.append(kid)
                excluded_reasons.append("extreme_high_haspt3d_zero_inlier")
                subset_trace.append(
                    {**stat, "ref_selected": False, "ref_exclusion_reason": "extreme_high_haspt3d_zero_inlier"}
                )
                continue
            if len(selected) >= self.recovery_consensus_target_refs:
                excluded_ids.append(kid)
                excluded_reasons.append("beyond_target_ref_count")
                subset_trace.append(
                    {**stat, "ref_selected": False, "ref_exclusion_reason": "beyond_target_ref_count"}
                )
                continue
            is_bridge = bool(keyframe.info.get("_paper_aligned_bridge_ref", False))
            soft_ok = (
                valid >= 8
                or pnp_inl >= 2
                or is_support
                or is_seed
                or (is_bridge and (valid >= 4 or pnp_inl >= 1))
            )
            if not soft_ok and valid < 4:
                excluded_ids.append(kid)
                excluded_reasons.append("insufficient_probe_support")
                subset_trace.append(
                    {**stat, "ref_selected": False, "ref_exclusion_reason": "insufficient_probe_support"}
                )
                continue
            selected.append(keyframe)
            if is_support or is_seed:
                has_direct_or_support = True
            subset_trace.append({**stat, "ref_selected": True, "ref_exclusion_reason": ""})

        if not has_direct_or_support:
            for keyframe, stat in ranked:
                if is_support_ref(keyframe):
                    if keyframe not in selected:
                        selected.insert(0, keyframe)
                        subset_trace.append({**stat, "ref_selected": True, "ref_exclusion_reason": "promoted_support_ref"})
                    break

        if len(selected) < self.recovery_consensus_min_refs:
            for keyframe, stat in ranked:
                if keyframe in selected:
                    continue
                selected.append(keyframe)
                subset_trace.append({**stat, "ref_selected": True, "ref_exclusion_reason": "filled_to_min_refs"})
                if len(selected) >= self.recovery_consensus_min_refs:
                    break

        selected = selected[: self.recovery_consensus_max_refs]
        selected_ids = {int(kf.index) for kf in selected}
        for keyframe, stat in ranked:
            kid = int(keyframe.index)
            if kid in selected_ids:
                continue
            if len(subset_trace) >= len(keyframes) * 2:
                break
        selection_reason = "top_scored_soft_filter_with_probe_weights"
        return selected, subset_trace, excluded_ids, excluded_reasons, selection_reason

    def _run_pnp_inlier_count_only(
        self,
        keyframes: list[Keyframe],
        xyz_cat: torch.Tensor,
        uvs_cat: torch.Tensor,
        confs_cat: torch.Tensor,
    ) -> int:
        if len(xyz_cat) < 4:
            return 0
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        try:
            _, inliers = self.PnPRANSAC(uvs_cat, xyz_cat, self.f, self.centre, Rs6D_init, ts_init, confs_cat)
            return int(inliers.sum().item())
        except Exception:
            return 0

    def _init_recovery_incremental_debug(self, keyframes: list[Keyframe]) -> None:
        self.last_incremental_debug = {
            "failure_reason": "",
            "num_2d3d_correspondences": 0,
            "num_pnp_inliers": 0,
            "num_miniba_inliers": 0,
            "ref_keyframe_ids": [int(kf.index) for kf in keyframes],
            "ref_source_frame_ids": [
                int(kf.info.get("_paper_aligned_source_frame_id", kf.index)) for kf in keyframes
            ],
            "ref_commit_origin": [str(kf.info.get("_paper_aligned_commit_origin", "unknown")) for kf in keyframes],
            "ref_is_recovery": [
                str(kf.info.get("_paper_aligned_commit_origin", ""))
                in {"true_recovery_commit", "early_seed_recovery_commit"}
                for kf in keyframes
            ],
            "ref_is_seed": [bool(kf.info.get("_paper_aligned_is_v7_early_seed", False)) for kf in keyframes],
            "ref_is_support_eligible": [
                bool(kf.info.get("_paper_aligned_support_eligible_recovery_keyframe", False)) for kf in keyframes
            ],
            "match_count_by_ref": list(self.last_recovery_2d3d_support.get("per_ref_3d_bearing_counts", []) or []),
            "match_count_total": int(self.last_recovery_2d3d_support.get("verified_2d2d_match_count", 0) or 0),
            "match_count_to_seed_keyframes": int(
                self.last_recovery_2d3d_support.get("valid_2d3d_from_seed_refs", 0) or 0
            ),
            "best_match_keyframe_id": int(keyframes[0].index) if keyframes else -1,
            "best_match_is_seed": bool(keyframes[0].info.get("_paper_aligned_is_v7_early_seed", False))
            if keyframes
            else False,
            "best_match_num_matches": int(
                max(self.last_recovery_2d3d_support.get("per_ref_3d_bearing_counts", [0]) or [0])
            ),
            "pnp_ref_keyframe_ids": [],
            "pnp_ref_source_frame_ids": [],
            "pnp_ref_contains_seed": False,
            "miniba_ref_keyframe_ids": [],
            "miniba_ref_source_frame_ids": [],
            "miniba_ref_contains_seed": False,
        }

    @torch.no_grad()
    def initialize_incremental_recovery(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
    ):
        sorted_keyframes = self._sort_recovery_reference_keyframes(keyframes, curr_desc_kpts, index)
        defer_sid = int(getattr(self, "recovery_defer_source_frame_id", -1))
        init_kf = sorted_keyframes[0]
        anchor_kf = max(sorted_keyframes, key=self._keyframe_has_pt3d_count)
        anchor_mode_id = self._ref_anchor_id(anchor_kf)
        Rt_guess = anchor_kf.get_Rt().clone()

        xyz_all, uvs_all, confs_all, _, support_all = self._collect_recovery_correspondences(
            sorted_keyframes, curr_desc_kpts, index, Rt_guess
        )
        pnp_before = self._run_pnp_inlier_count_only(sorted_keyframes, xyz_all, uvs_all, confs_all)
        valid_before = int(len(xyz_all))
        ratio_before = float(pnp_before) / max(valid_before, 1)

        ref_stats: list[dict[str, object]] = []
        for keyframe in sorted_keyframes:
            pnp_inl, valid_2d3d, pnp_ratio = self._probe_ref_pnp_inliers(
                keyframe, curr_desc_kpts, index, init_kf
            )
            score = self._recovery_ref_consensus_score(
                keyframe, valid_2d3d, pnp_inl, pnp_ratio, defer_sid, anchor_mode_id
            )
            ref_stats.append(
                {
                    "ref_keyframe_id": int(keyframe.index),
                    "ref_source_frame_id": self._ref_source_frame_id(keyframe),
                    "ref_commit_origin": str(keyframe.info.get("_paper_aligned_commit_origin", "")),
                    "ref_anchor_id": self._ref_anchor_id(keyframe),
                    "ref_valid_2d3d_count": valid_2d3d,
                    "ref_pnp_inlier_count": pnp_inl,
                    "ref_pnp_inlier_ratio": round(pnp_ratio, 6),
                    "ref_consensus_score": round(score, 6),
                    "ref_has_pt3d_count": self._keyframe_has_pt3d_count(keyframe),
                    "ref_source_distance_to_defer": abs(self._ref_source_frame_id(keyframe) - defer_sid),
                }
            )

        selected, subset_trace, excluded_ids, excluded_reasons, subset_selection_reason = (
            self._select_coherent_ref_subset(sorted_keyframes, ref_stats, defer_sid)
        )
        self.last_recovery_ref_subset = list(subset_trace)
        stat_by_kid = {int(s.get("ref_keyframe_id", -1)): s for s in ref_stats}
        selected_stats = [stat_by_kid[int(kf.index)] for kf in selected if int(kf.index) in stat_by_kid]
        selected_valid_sum = sum(int(s.get("ref_valid_2d3d_count", 0) or 0) for s in selected_stats)
        selected_pnp_sum = sum(int(s.get("ref_pnp_inlier_count", 0) or 0) for s in selected_stats)
        max_selected_pnp = max((int(s.get("ref_pnp_inlier_count", 0) or 0) for s in selected_stats), default=0)
        coherent = bool(
            len(selected) >= self.recovery_consensus_min_refs
            and selected_valid_sum >= self.recovery_consensus_min_total_valid_2d3d
            and (selected_pnp_sum >= 8 or max_selected_pnp >= 4)
        )
        selection_reason = subset_selection_reason if coherent else "no_coherent_subset"

        self.last_recovery_pnp_consensus = {
            "candidate_ref_count": len(sorted_keyframes),
            "candidate_ref_ids": [int(kf.index) for kf in sorted_keyframes],
            "candidate_ref_valid_2d3d_counts": [int(s["ref_valid_2d3d_count"]) for s in ref_stats],
            "candidate_ref_pnp_inlier_counts": [int(s["ref_pnp_inlier_count"]) for s in ref_stats],
            "candidate_ref_pnp_inlier_ratios": [float(s["ref_pnp_inlier_ratio"]) for s in ref_stats],
            "candidate_ref_anchor_ids": [int(s["ref_anchor_id"]) for s in ref_stats],
            "candidate_ref_source_distances": [int(s["ref_source_distance_to_defer"]) for s in ref_stats],
            "selected_consensus_ref_ids": [int(kf.index) for kf in selected],
            "selected_consensus_ref_count": len(selected),
            "selection_reason": selection_reason,
            "excluded_ref_ids": excluded_ids,
            "excluded_ref_reasons": excluded_reasons,
            "coherent_subset_found": coherent,
            "pnp_inliers_before_consensus": pnp_before,
            "pnp_inlier_ratio_before_consensus": round(ratio_before, 6),
            "valid_2d3d_before_consensus": valid_before,
        }

        if not coherent:
            self.last_recovery_2d3d_support = dict(support_all)
            self._init_recovery_incremental_debug(sorted_keyframes)
            self.last_incremental_debug["failure_reason"] = "no_coherent_ref_subset"
            self.last_recovery_pnp_consensus["failure_reason_after_consensus"] = "no_coherent_ref_subset"
            self.last_recovery_pose_outcome_fix = {
                "recovery_pose_mode": True,
                "correspondence_flow_fix_applied": False,
                "recovery_miniba_retry_applied": False,
                "failure_reason_after_fix": "no_coherent_ref_subset",
            }
            return None

        ref_conf_weights = {
            int(kf.index): self._ref_conf_weight_from_probe_stat(stat_by_kid[int(kf.index)])
            for kf in selected
            if int(kf.index) in stat_by_kid
        }
        ref_correspondence_caps: dict[int, int] = {}
        for kf in selected:
            kid = int(kf.index)
            if kid not in stat_by_kid:
                continue
            cap = self._per_ref_correspondence_cap(stat_by_kid[kid])
            if cap is not None:
                ref_correspondence_caps[kid] = int(cap)
        xyz_cat, uvs_cat, confs_cat, corr_ref_ids_cat, support_debug = self._collect_recovery_correspondences(
            selected,
            curr_desc_kpts,
            index,
            Rt_guess,
            ref_conf_weights=ref_conf_weights,
            ref_correspondence_caps=ref_correspondence_caps or None,
        )
        self.last_recovery_2d3d_support = dict(support_debug)
        self._init_recovery_incremental_debug(selected)
        self.last_recovery_pose_outcome_fix = {
            "recovery_pose_mode": True,
            "handoff_fix_applied": True,
            "reference_consistency_fix_applied": True,
            "pnp_geometric_consensus_fix_applied": True,
            "correspondence_flow_fix_applied": bool(support_debug.get("temporary_3d_support_used", False)),
            "recovery_miniba_retry_applied": False,
            "triangulation_augmented_correspondence_count": int(
                support_debug.get("temporary_3d_support_count", 0) or 0
            ),
            "miniba_success_before_fix": False,
            "miniba_inliers_before_fix": 0,
            "miniba_success_after_fix": False,
            "miniba_inliers_after_fix": 0,
            "failure_reason_before_fix": "",
            "failure_reason_after_fix": "",
        }
        Rt = self._run_pose_from_correspondences(
            selected, xyz_cat, uvs_cat, confs_cat, corr_ref_ids_cat, index, is_test
        )
        after_debug = dict(self.last_incremental_debug)
        pnp_after = int(after_debug.get("num_pnp_inliers", 0) or 0)
        valid_after = int(after_debug.get("num_2d3d_correspondences", 0) or 0)
        ratio_after = float(pnp_after) / max(valid_after, 1)
        self.last_recovery_pnp_consensus.update(
            {
                "pnp_inliers_after_consensus": pnp_after,
                "pnp_inlier_ratio_after_consensus": round(ratio_after, 6),
                "miniba_inliers_before_consensus": 0,
                "miniba_inliers_after_consensus": int(after_debug.get("num_miniba_inliers", 0) or 0),
                "miniba_success_after_consensus": bool(Rt is not None),
                "recovery_success_after_consensus": bool(Rt is not None),
                "failure_reason_after_consensus": str(after_debug.get("failure_reason", "") or ""),
            }
        )
        self.last_recovery_pose_outcome_fix["miniba_success_before_fix"] = bool(Rt is not None)
        self.last_recovery_pose_outcome_fix["miniba_inliers_before_fix"] = int(
            after_debug.get("num_miniba_inliers", 0) or 0
        )
        if Rt is not None:
            self.last_recovery_pose_outcome_fix["miniba_success_after_fix"] = True
            self.last_recovery_pose_outcome_fix["miniba_inliers_after_fix"] = int(
                after_debug.get("num_miniba_inliers", 0) or 0
            )
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = ""
            return Rt

        failure_reason = str(after_debug.get("failure_reason", "") or "")
        if (
            failure_reason != "miniba_inliers_too_few"
            or pnp_after < self.recovery_miniba_retry_min_pnp_inliers
            or valid_after < self.recovery_miniba_retry_min_2d3d
        ):
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = failure_reason
            return None

        self.last_recovery_pose_outcome_fix["recovery_miniba_retry_applied"] = True
        Rt_seed = self._last_pnp_Rt
        if Rt_seed is None:
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = "pnp_pose_unavailable"
            return None
        xyz_retry, uvs_retry, confs_retry, corr_retry, retry_support = self._collect_recovery_correspondences(
            selected,
            curr_desc_kpts,
            index,
            Rt_seed,
            ref_conf_weights=ref_conf_weights,
            ref_correspondence_caps=ref_correspondence_caps or None,
        )
        if len(xyz_retry) < self.recovery_miniba_retry_min_2d3d or pnp_after < self.recovery_miniba_retry_min_pnp_inliers:
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = "recovery_retry_gates_not_met"
            return None
        self._init_recovery_incremental_debug(selected)
        Rt_retry = self._run_pose_from_correspondences(
            selected, xyz_retry, uvs_retry, confs_retry, corr_retry, index, is_test
        )
        if Rt_retry is not None:
            self.last_recovery_pose_outcome_fix["miniba_success_after_fix"] = True
            self.last_recovery_pose_outcome_fix["miniba_inliers_after_fix"] = int(
                self.last_incremental_debug.get("num_miniba_inliers", 0) or 0
            )
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = ""
            return Rt_retry
        self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = str(
            self.last_incremental_debug.get("failure_reason", "") or "miniba_inliers_too_few"
        )
        return None
