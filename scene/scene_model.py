# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 场景模型类，用于管理3D高斯点云和关键帧
# 参考自：https://github.com/verlab/accelerated_features


from argparse import Namespace
import gc
import os
import json
import math
import threading
import time
import warnings
import cv2
import torch
import torch.nn.functional as F
import numpy as np

import lpips
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distIndex2
from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.guided_mvs import GuidedMVS
from poses.pose_verification import async_pose_update_enabled
from scene.optimizers import SparseGaussianAdam
from scene.keyframe import Keyframe
from scene.anchor import Anchor
from scene.pose_eval_utils import select_pose_eval_pairs
from scene.frame_metrics import (
    build_frame_metric_row,
    infer_dataset_scene_from_output_dir,
    write_frame_metrics_csv,
)
from scene.exposure_harmonization import (
    adaptive_source_time_exposure_scene_decision,
    dark_scene_test_exposure_scene_decision,
    harmonized_test_exposure,
)
from scene.pose_render_edge_loss import pose_render_gradient_loss
from scene.pose_render_extra_optimization import (
    EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE,
    EXTRA_REFINEMENT_STATE_KEY,
    pose_render_extra_optimization_decision,
    updated_render_response,
)
from scene.pose_render_init_weighting import pose_render_init_weighting_decision
from scene.pose_render_keyframe_sampling import choose_pose_render_keyframe_id
from scene.pose_render_pre_refine import pose_render_pre_refine_decision
from scene.pose_render_psnr_loss import (
    pose_render_mse_loss,
    pose_render_structure_alignment_score,
)
from scene.pose_render_texture_sampling import (
    SCENE_LOW_RESPONSE_KEY,
    TEXTURE_SAMPLING_COVERAGE_KEY,
    TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE,
    TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
    TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,
    TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
    TEXTURE_SAMPLING_SCENE_GUARD_KEY,
    residual_edge_guided_sampling_probability,
)
from scene.pose_render_update_gate import pose_render_update_gate_decision
from scene.pose_risk_utility_admission import (
    filter_pose_reference_indices,
    pose_review_acceptance,
    restore_optimizer_parameter_learning_rates,
    restore_optimizer_parameter_state,
    scale_optimizer_parameter_learning_rates,
    snapshot_optimizer_parameter_state,
)
from scene.test_render_calibration import calibrate_test_render
from scene.transactional_gaussian_refinement import (
    overwrite_selected_gaussian_value_snapshot,
    refinement_candidate_acceptance,
    refinement_reference_guard,
    refinement_time_budget_seconds,
    rendering_response_gap,
    restore_selected_gaussian_state,
    scale_gaussian_gradients,
    select_refinement_reference_indices,
    snapshot_selected_gaussian_state,
)
from utils import (
    RGB2SH,
    depth2points,
    focal2fov,
    get_lapla_norm,
    getProjectionMatrix,
    inverse_sigmoid,
    align_poses,
    make_torch_sampler,
    psnr,
    rotation_distance,
)
from dataloaders.read_write_model import write_model


def _metric_float(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


LPIPS_CPU_FALLBACK_MIN_FREE_BYTES = 1024 * 1024 * 1024


def _lpips_cpu_fallback_needed(
    free_cuda_bytes: int,
    min_free_bytes: int = LPIPS_CPU_FALLBACK_MIN_FREE_BYTES,
) -> bool:
    return int(free_cuda_bytes) < int(min_free_bytes)


def _module_parameters_on_cuda(module) -> bool:
    try:
        return any(param.is_cuda for param in module.parameters())
    except Exception:
        return False


class SceneModel:
    """
    【场景表示模块】场景模型类

    这是整个3D重建系统的核心类，负责管理：
    1. 3D高斯点云（Gaussians）：场景的几何和外观表示
    2. 关键帧（Keyframes）：包含图像、位姿、深度等信息
    3. 锚点（Anchors）：用于大尺度场景的分块管理
    4. 渲染和优化：从任意视角渲染场景，并优化高斯参数

    主要功能：
    - 添加新关键帧并初始化高斯点
    - 渲染场景（支持多分辨率）
    - 优化高斯参数（位置、颜色、不透明度、尺度、旋转）
    - 管理锚点（用于处理大尺度场景）
    - 保存和加载场景
    """
    def __init__(
        self,
        width: int,
        height: int,
        args: Namespace,
        matcher: Matcher = None,
        inference_mode: bool = False,
    ):
        """
        【场景表示模块】初始化场景模型

        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            args: 训练参数配置（包含学习率、损失权重、锚点重叠等）
            matcher: 特征匹配器（用于关键帧匹配，推理模式下可为None）
            inference_mode: 是否为推理模式（True=仅渲染，False=训练模式）
        """
        # ========== 基础参数初始化 ==========
        self.width = width
        self.height = height
        self.matcher = matcher
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device="cuda")  # 图像中心点
        self.anchor_overlap = args.anchor_overlap  # 锚点重叠区域大小（用于平滑融合）
        self.optimization_thread = None  # 异步优化线程（流式模式下使用）
        self.pose_verification_async_pose_protection_mode = str(
            getattr(args, "pose_verification_async_pose_protection_mode", "off")
            or "off"
        )
        self.pose_verification_async_pose_protection_stats = {
            "mode": self.pose_verification_async_pose_protection_mode,
            "joint_pose_steps": 0,
            "protected_gaussian_steps": 0,
        }
        self.risk_admission_mode = str(getattr(args, "risk_admission_mode", "off") or "off")
        self.recovery_commit_bridge = str(
            getattr(args, "paper_aligned_recovery_commit_bridge", "true_source_commit") or "true_source_commit"
        )
        self.paper_aligned_direct_density_control = str(
            getattr(args, "paper_aligned_direct_density_control", "off") or "off"
        )
        self.paper_aligned_render_frame_policy = str(
            getattr(args, "paper_aligned_render_frame_policy", "off") or "off"
        )
        self.paper_aligned_training_background_mode = str(
            getattr(args, "paper_aligned_training_background_mode", "random_v1")
            or "random_v1"
        )
        self.paper_aligned_test_exposure_harmonization = str(
            getattr(args, "paper_aligned_test_exposure_harmonization", "neighbor_average_v1")
            or "neighbor_average_v1"
        )
        self.paper_aligned_test_exposure_adaptive_stats = {
            "mode": self.paper_aligned_test_exposure_harmonization,
            "selected_mode": self.paper_aligned_test_exposure_harmonization,
            "reason": "not_adaptive",
        }
        self.paper_aligned_test_exposure_guard_max_delta = float(
            getattr(args, "paper_aligned_test_exposure_guard_max_delta", 0.0)
            or 0.0
        )
        self.paper_aligned_test_render_calibration = str(
            getattr(args, "paper_aligned_test_render_calibration", "off") or "off"
        )
        self._training_background_step = 0
        self._training_background_dark_scene_threshold = 0.34
        self._training_background_scene_mean_decay = 0.95
        self._training_background_scene_mean = None
        self._training_background_scene_dark_decision = None
        self.training_background_stats = self._new_training_background_stats()
        self.pose_render_texture_sampling = str(
            getattr(args, "paper_aligned_pose_render_texture_sampling", "off") or "off"
        )
        self.pose_render_texture_sampling_alpha = float(
            getattr(args, "paper_aligned_pose_render_texture_sampling_alpha", 0.12)
            or 0.12
        )
        self.pose_render_texture_sampling_min_selectivity = float(
            getattr(args, "paper_aligned_pose_render_texture_sampling_min_selectivity", 0.18)
            or 0.18
        )
        self.pose_render_texture_sampling_stats = {
            "mode": self.pose_render_texture_sampling,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "render_frame_policy": self.paper_aligned_render_frame_policy,
            "events": 0,
            "applied": 0,
            "low_selectivity": 0,
            "disabled": 0,
            "zero_budget": 0,
            "scene_guard_disabled": 0,
            "coverage_sufficient": 0,
            "scene_sampling_quota": 0,
            "alpha_sum": 0.0,
            "selectivity_sum": 0.0,
            "budget_shift_abs_sum": 0.0,
            "coverage_deficit_sum": 0.0,
            "coverage_deficit_applied_sum": 0.0,
            "response_evaluated": 0,
            "response_good": 0,
            "response_bad": 0,
            "response_pending": 0,
            "response_degraded": 0,
            "response_low": 0,
            "scene_guard_disabled_flag": False,
            "scene_guard_reason": "warming_up",
            "scene_guard_min_evaluated": 3,
            "scene_guard_max_bad_ratio": 0.67,
            "scene_guard_min_coverage_deficit": 0.08,
            "coverage_bypass_latched": False,
            "coverage_bypass_reason": "",
            "coverage_bypass_min_events": 32,
            "coverage_bypass_max_mean_deficit": 0.015,
            "coverage_bypass_max_applied_ratio": 0.02,
        }
        self.pose_render_keyframe_sampling = str(
            getattr(args, "paper_aligned_pose_render_keyframe_sampling", "off") or "off"
        )
        self.pose_render_keyframe_sampling_min_weight = float(
            getattr(args, "paper_aligned_pose_render_keyframe_sampling_min_weight", 0.35)
            or 0.35
        )
        self.pose_render_keyframe_sampling_max_pose_risk = float(
            getattr(args, "paper_aligned_pose_render_keyframe_sampling_max_pose_risk", 0.38)
            or 0.38
        )
        self.pose_render_keyframe_sampling_max_utility_drift = float(
            getattr(
                args,
                "paper_aligned_pose_render_keyframe_sampling_max_utility_drift",
                0.75,
            )
            or 0.75
        )
        self.pose_render_keyframe_sampling_min_pose_support = float(
            getattr(
                args,
                "paper_aligned_pose_render_keyframe_sampling_min_pose_support",
                0.35,
            )
            or 0.35
        )
        self.pose_render_keyframe_sampling_min_match_support = float(
            getattr(
                args,
                "paper_aligned_pose_render_keyframe_sampling_min_match_support",
                0.35,
            )
            or 0.35
        )
        self.pose_render_keyframe_sampling_stats = {
            "mode": self.pose_render_keyframe_sampling,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "min_weight": self.pose_render_keyframe_sampling_min_weight,
            "max_pose_risk": self.pose_render_keyframe_sampling_max_pose_risk,
            "max_utility_drift": self.pose_render_keyframe_sampling_max_utility_drift,
            "min_pose_support": self.pose_render_keyframe_sampling_min_pose_support,
            "min_match_support": self.pose_render_keyframe_sampling_min_match_support,
            "events": 0,
            "applied": 0,
            "off": 0,
            "unknown_mode": 0,
            "direct_density_mismatch": 0,
            "empty_candidates": 0,
            "candidate_weight_mismatch": 0,
            "pose_confidence_temporal": 0,
            "candidate_count_sum": 0,
            "high_risk_candidates_sum": 0,
            "missing_pose_render_gate_sum": 0,
            "chosen_probability_sum": 0.0,
            "probability_min_observed": 1.0,
            "probability_max_observed": 0.0,
            "weight_min_observed": 1.0,
            "weight_max_observed": 0.0,
            "weight_mean_sum": 0.0,
            "confidence_mean_sum": 0.0,
        }
        self.pose_render_edge_loss = str(
            getattr(args, "paper_aligned_pose_render_edge_loss", "off") or "off"
        )
        self.pose_render_edge_loss_weight = float(
            getattr(args, "paper_aligned_pose_render_edge_loss_weight", 0.03)
            or 0.03
        )
        self.pose_render_edge_loss_max_pose_risk = float(
            getattr(args, "paper_aligned_pose_render_edge_loss_max_pose_risk", 0.50)
            or 0.50
        )
        self.pose_render_edge_loss_max_utility_drift = float(
            getattr(args, "paper_aligned_pose_render_edge_loss_max_utility_drift", 0.65)
            or 0.65
        )
        self.pose_render_edge_loss_min_pose_support = float(
            getattr(args, "paper_aligned_pose_render_edge_loss_min_pose_support", 0.45)
            or 0.45
        )
        self.pose_render_edge_loss_min_match_support = float(
            getattr(args, "paper_aligned_pose_render_edge_loss_min_match_support", 0.45)
            or 0.45
        )
        self.pose_render_edge_loss_min_weight_scale = float(
            getattr(args, "paper_aligned_pose_render_edge_loss_min_weight_scale", 0.25)
            or 0.25
        )
        self.pose_render_edge_loss_target_raw_loss = float(
            getattr(args, "paper_aligned_pose_render_edge_loss_target_raw_loss", 0.02)
            or 0.02
        )
        self.pose_render_edge_loss_risk_free_threshold = float(
            getattr(
                args,
                "paper_aligned_pose_render_edge_loss_risk_free_threshold",
                0.25,
            )
            or 0.25
        )
        self.pose_render_edge_loss_stats = {
            "mode": self.pose_render_edge_loss,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "max_pose_risk": self.pose_render_edge_loss_max_pose_risk,
            "max_utility_drift": self.pose_render_edge_loss_max_utility_drift,
            "min_pose_support": self.pose_render_edge_loss_min_pose_support,
            "min_match_support": self.pose_render_edge_loss_min_match_support,
            "min_weight_scale": self.pose_render_edge_loss_min_weight_scale,
            "target_raw_loss": self.pose_render_edge_loss_target_raw_loss,
            "risk_free_threshold": self.pose_render_edge_loss_risk_free_threshold,
            "events": 0,
            "applied": 0,
            "disabled": 0,
            "gradient_edge_l1_adaptive": 0,
            "shape_mismatch": 0,
            "invalid_spatial_shape": 0,
            "invalid_target_mask": 0,
            "unknown_target_mask_mode": 0,
            "target_mask_applied": 0,
            "target_valid_ratio_sum": 0.0,
            "support_weight_scale_sum": 0.0,
            "support_correspondence_count_sum": 0.0,
            "support_final_pose_inliers_sum": 0.0,
            "raw_pose_support_scaled": 0,
            "raw_pose_support_gate_passed": 0,
            "low_raw_pose_support": 0,
            "missing_raw_support": 0,
            "unknown_support_weight_mode": 0,
            "pose_safe_gate_passed": 0,
            "pose_safe_gate_rejected": 0,
            "missing_pose_render_gate": 0,
            "direct_keyframe_not_finalized": 0,
            "pose_risk_high": 0,
            "pose_risk_reference": 0,
            "pose_risk_score_high": 0,
            "utility_drift_risk_high": 0,
            "low_pose_match_support": 0,
            "weight_sum": 0.0,
            "base_weight_sum": 0.0,
            "adaptive_weight_scale_sum": 0.0,
            "residual_clip_scale_sum": 0.0,
            "risk_weight_scale_sum": 0.0,
            "drift_weight_scale_sum": 0.0,
            "support_weight_scale_sum": 0.0,
            "context_weight_scale_sum": 0.0,
            "context_weight_boosted": 0,
            "mask_aware_no_mask_boost": 0,
            "mask_aware_no_mask_raw_response_boost": 0,
            "mask_aware_no_mask_raw_response_low": 0,
            "mask_aware_no_mask_raw_response_low_gate": 0,
            "mask_aware_masked": 0,
            "missing_or_unmatched_context": 0,
            "unknown_context_weight_mode": 0,
            "raw_loss_sum": 0.0,
            "weighted_loss_sum": 0.0,
        }
        self.pose_render_psnr_loss = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss", "off") or "off"
        )
        self.pose_render_psnr_loss_weight = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_weight", 0.10)
        )
        self.pose_render_psnr_loss_max_pose_risk = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_max_pose_risk", 0.30)
        )
        self.pose_render_psnr_loss_max_utility_drift = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_max_utility_drift", 0.55)
        )
        self.pose_render_psnr_loss_min_pose_support = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_min_pose_support", 0.45)
        )
        self.pose_render_psnr_loss_min_match_support = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_min_match_support", 0.45)
        )
        self.pose_render_psnr_loss_target_mask = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss_target_mask", "off") or "off"
        )
        self.pose_render_psnr_loss_support_weight = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss_support_weight", "off") or "off"
        )
        self.pose_render_psnr_loss_context_weight = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss_context_weight", "off")
            or "off"
        )
        self.pose_render_psnr_loss_support_min_scale = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_support_min_scale", 0.45)
            or 0.45
        )
        self.pose_render_psnr_loss_support_corr_low = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_support_corr_low", 3000.0)
            or 3000.0
        )
        self.pose_render_psnr_loss_support_corr_high = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_support_corr_high", 8000.0)
            or 8000.0
        )
        self.pose_render_psnr_loss_support_inlier_low = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_support_inlier_low", 900.0)
            or 900.0
        )
        self.pose_render_psnr_loss_support_inlier_high = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_support_inlier_high", 2500.0)
            or 2500.0
        )
        self.pose_render_psnr_loss_max_applied_ratio = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_max_applied_ratio", 1.0)
            or 1.0
        )
        self.pose_render_psnr_loss_ambiguity_low = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_ambiguity_low", 0.0)
            or 0.0
        )
        self.pose_render_psnr_loss_ambiguity_high = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_ambiguity_high", 0.0)
            or 0.0
        )
        self.pose_render_psnr_loss_ambiguity_min_applied_ratio = float(
            getattr(
                args,
                "paper_aligned_pose_render_psnr_loss_ambiguity_min_applied_ratio",
                1.0,
            )
            or 1.0
        )
        self.pose_render_psnr_loss_scene_guard = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss_scene_guard", "off")
            or "off"
        )
        self.pose_render_psnr_loss_scene_guard_raw_low = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_scene_guard_raw_low", 0.0)
            or 0.0
        )
        self.pose_render_psnr_loss_scene_guard_raw_high = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_scene_guard_raw_high", 0.0)
            or 0.0
        )
        self.pose_render_psnr_loss_scene_guard_min_ratio = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_scene_guard_min_ratio", 1.0)
            or 1.0
        )
        self.pose_render_psnr_loss_scene_guard_min_events = int(
            getattr(args, "paper_aligned_pose_render_psnr_loss_scene_guard_min_events", 0)
            or 0
        )
        self.pose_render_psnr_loss_scene_guard_min_applied = int(
            getattr(args, "paper_aligned_pose_render_psnr_loss_scene_guard_min_applied", 0)
            or 0
        )
        self.pose_render_psnr_loss_scene_guard_max_events = int(
            getattr(args, "paper_aligned_pose_render_psnr_loss_scene_guard_max_events", 0)
            or 0
        )
        self.pose_render_psnr_loss_structure_gate = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss_structure_gate", "off")
            or "off"
        )
        self.pose_render_psnr_loss_structure_min_score = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_structure_min_score", 0.0)
            or 0.0
        )
        self.pose_render_psnr_loss_structure_min_raw_mean = float(
            getattr(
                args,
                "paper_aligned_pose_render_psnr_loss_structure_min_raw_mean",
                0.0,
            )
            or 0.0
        )
        self.pose_render_psnr_loss_late_raw_stop = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss_late_raw_stop", "off")
            or "off"
        )
        self.pose_render_psnr_loss_late_raw_stop_min_mean = float(
            getattr(
                args,
                "paper_aligned_pose_render_psnr_loss_late_raw_stop_min_mean",
                0.0,
            )
            or 0.0
        )
        self.pose_render_psnr_loss_health_gate = str(
            getattr(args, "paper_aligned_pose_render_psnr_loss_health_gate", "off")
            or "off"
        )
        self.pose_render_psnr_loss_health_min_score = float(
            getattr(args, "paper_aligned_pose_render_psnr_loss_health_min_score", 0.0)
            or 0.0
        )
        self.pose_render_psnr_loss_health_min_raw_loss = float(
            getattr(
                args,
                "paper_aligned_pose_render_psnr_loss_health_min_raw_loss",
                0.0,
            )
            or 0.0
        )
        self.pose_render_psnr_loss_scene_guard_triggered = False
        self.pose_render_psnr_scene_low_response_latched = False
        self.pose_render_psnr_loss_stats = {
            "mode": self.pose_render_psnr_loss,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "render_frame_policy": self.paper_aligned_render_frame_policy,
            "training_background_mode": self.paper_aligned_training_background_mode,
            "max_applied_ratio": self.pose_render_psnr_loss_max_applied_ratio,
            "ambiguity_low": self.pose_render_psnr_loss_ambiguity_low,
            "ambiguity_high": self.pose_render_psnr_loss_ambiguity_high,
            "ambiguity_min_applied_ratio": (
                self.pose_render_psnr_loss_ambiguity_min_applied_ratio
            ),
            "scene_guard_mode": self.pose_render_psnr_loss_scene_guard,
            "scene_guard_raw_low": self.pose_render_psnr_loss_scene_guard_raw_low,
            "scene_guard_raw_high": self.pose_render_psnr_loss_scene_guard_raw_high,
            "scene_guard_min_ratio": self.pose_render_psnr_loss_scene_guard_min_ratio,
            "scene_guard_min_events": self.pose_render_psnr_loss_scene_guard_min_events,
            "scene_guard_min_applied": (
                self.pose_render_psnr_loss_scene_guard_min_applied
            ),
            "scene_guard_max_events": self.pose_render_psnr_loss_scene_guard_max_events,
            "structure_gate_mode": self.pose_render_psnr_loss_structure_gate,
            "structure_min_score": self.pose_render_psnr_loss_structure_min_score,
            "structure_min_raw_mean": self.pose_render_psnr_loss_structure_min_raw_mean,
            "late_raw_stop_mode": self.pose_render_psnr_loss_late_raw_stop,
            "late_raw_stop_min_mean": (
                self.pose_render_psnr_loss_late_raw_stop_min_mean
            ),
            "health_gate_mode": self.pose_render_psnr_loss_health_gate,
            "health_min_score": self.pose_render_psnr_loss_health_min_score,
            "health_min_raw_loss": self.pose_render_psnr_loss_health_min_raw_loss,
            "scene_guard_triggered": False,
            "scene_guard_trigger_events": 0,
            "scene_guard_trigger_applied": 0,
            "scene_guard_trigger_ratio": 0.0,
            "scene_guard_trigger_raw_loss_mean": 0.0,
            "scene_guard_candidate_count": 0,
            "scene_guard_candidate_raw_loss_sum": 0.0,
            "scene_low_response_latched": False,
            "scene_low_response_events": 0,
            "scene_low_response_raw_loss_sum": 0.0,
            "scene_low_response_trigger_events": 0,
            "scene_low_response_trigger_raw_loss_mean": 0.0,
            "max_pose_risk": self.pose_render_psnr_loss_max_pose_risk,
            "max_utility_drift": self.pose_render_psnr_loss_max_utility_drift,
            "min_pose_support": self.pose_render_psnr_loss_min_pose_support,
            "min_match_support": self.pose_render_psnr_loss_min_match_support,
            "target_mask_mode": self.pose_render_psnr_loss_target_mask,
            "support_weight_mode": self.pose_render_psnr_loss_support_weight,
            "support_weight_min_scale": self.pose_render_psnr_loss_support_min_scale,
            "support_corr_low": self.pose_render_psnr_loss_support_corr_low,
            "support_corr_high": self.pose_render_psnr_loss_support_corr_high,
            "support_inlier_low": self.pose_render_psnr_loss_support_inlier_low,
            "support_inlier_high": self.pose_render_psnr_loss_support_inlier_high,
            "events": 0,
            "applied": 0,
            "disabled": 0,
            "direct_density_mismatch": 0,
            "mse_rgb_pose_safe": 0,
            "robust_mse_rgb_pose_safe": 0,
            "freq_mse_rgb_pose_safe": 0,
            "shape_mismatch": 0,
            "invalid_spatial_shape": 0,
            "invalid_target_mask": 0,
            "unknown_target_mask_mode": 0,
            "psnr_budget_exhausted": 0,
            "psnr_residual_ambiguity_band": 0,
            "psnr_scene_guard_triggered": 0,
            "psnr_scene_guard_blocked": 0,
            "psnr_scene_low_response_gate": 0,
            "psnr_scene_guard_precommit_wait": 0,
            "psnr_scene_guard_late_window_expired": 0,
            "psnr_mask_aware_dark_background_blocked": 0,
            "psnr_structure_gate_low": 0,
            "unknown_structure_gate_mode": 0,
            "psnr_structure_gate_window_wait": 0,
            "psnr_structure_gate_raw_mean_low": 0,
            "psnr_late_raw_mean_stop": 0,
            "unknown_late_raw_stop_mode": 0,
            "psnr_health_gate_low": 0,
            "unknown_health_gate_mode": 0,
            "health_gate_raw_wait": 0,
            "health_score_sum": 0.0,
            "health_score_count": 0,
            "structure_score_sum": 0.0,
            "structure_score_count": 0,
            "target_mask_applied": 0,
            "target_valid_ratio_sum": 0.0,
            "support_weight_scale_sum": 0.0,
            "support_correspondence_count_sum": 0.0,
            "support_final_pose_inliers_sum": 0.0,
            "raw_pose_support_scaled": 0,
            "raw_pose_support_gate_passed": 0,
            "low_raw_pose_support": 0,
            "missing_raw_support": 0,
            "unknown_support_weight_mode": 0,
            "pose_safe_gate_passed": 0,
            "pose_safe_gate_rejected": 0,
            "missing_pose_render_gate": 0,
            "direct_keyframe_not_finalized": 0,
            "pose_risk_high": 0,
            "pose_risk_reference": 0,
            "pose_risk_score_high": 0,
            "utility_drift_risk_high": 0,
            "low_pose_match_support": 0,
            "weight_sum": 0.0,
            "base_weight_sum": 0.0,
            "context_weight_scale_sum": 0.0,
            "context_weight_boosted": 0,
            "mask_aware_no_mask_boost": 0,
            "mask_aware_no_mask_raw_response_boost": 0,
            "mask_aware_no_mask_raw_response_low": 0,
            "mask_aware_no_mask_raw_response_low_gate": 0,
            "mask_aware_masked": 0,
            "missing_or_unmatched_context": 0,
            "unknown_context_weight_mode": 0,
            "raw_loss_sum": 0.0,
            "weighted_loss_sum": 0.0,
        }
        self.pose_render_extra_optimization = str(
            getattr(args, "paper_aligned_pose_render_extra_optimization", "off") or "off"
        )
        self.pose_render_extra_optimization_min_confidence = float(
            getattr(
                args,
                "paper_aligned_pose_render_extra_optimization_min_confidence",
                0.75,
            )
            or 0.75
        )
        self.pose_render_extra_optimization_fraction = float(
            getattr(
                args,
                "paper_aligned_pose_render_extra_optimization_fraction",
                0.25,
            )
            or 0.25
        )
        self.pose_render_extra_optimization_max_extra = int(
            getattr(args, "paper_aligned_pose_render_extra_optimization_max_extra", 8)
            or 0
        )
        self.pose_render_extra_optimization_max_pose_risk = float(
            getattr(
                args,
                "paper_aligned_pose_render_extra_optimization_max_pose_risk",
                0.35,
            )
            or 0.35
        )
        self.pose_render_extra_optimization_max_utility_drift = float(
            getattr(
                args,
                "paper_aligned_pose_render_extra_optimization_max_utility_drift",
                0.60,
            )
            or 0.60
        )
        self.pose_render_extra_optimization_min_pose_support = float(
            getattr(
                args,
                "paper_aligned_pose_render_extra_optimization_min_pose_support",
                0.55,
            )
            or 0.55
        )
        self.pose_render_extra_optimization_min_match_support = float(
            getattr(
                args,
                "paper_aligned_pose_render_extra_optimization_min_match_support",
                0.55,
            )
            or 0.55
        )
        self.pose_render_extra_optimization_stats = {
            "mode": self.pose_render_extra_optimization,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "min_confidence": self.pose_render_extra_optimization_min_confidence,
            "fraction": self.pose_render_extra_optimization_fraction,
            "max_extra": self.pose_render_extra_optimization_max_extra,
            "max_pose_risk": self.pose_render_extra_optimization_max_pose_risk,
            "max_utility_drift": self.pose_render_extra_optimization_max_utility_drift,
            "min_pose_support": self.pose_render_extra_optimization_min_pose_support,
            "min_match_support": self.pose_render_extra_optimization_min_match_support,
            "events": 0,
            "applied": 0,
            "off": 0,
            "unknown_mode": 0,
            "direct_density_mismatch": 0,
            "test_frame": 0,
            "missing_pose_render_gate": 0,
            "pose_risk_high": 0,
            "pose_risk_score_high": 0,
            "utility_drift_risk_high": 0,
            "low_pose_match_support": 0,
            "confidence_low": 0,
            "zero_base_iterations": 0,
            "zero_extra_budget": 0,
            "coverage_sufficient": 0,
            "scene_coverage_sufficient": 0,
            "scene_pressure_low": 0,
            "scene_pressure_warming_up": 0,
            "scene_response_warming_up": 0,
            "scene_response_unstable": 0,
            "pose_confident_extra_optimization": 0,
            "render_response_pending": 0,
            "render_response_low": 0,
            "render_response_degraded": 0,
            "render_response_extra_optimization": 0,
            "extra_iterations_sum": 0,
            "confidence_sum": 0.0,
            "transaction_attempted": 0,
            "transaction_committed": 0,
            "transaction_rolled_back": 0,
            "transaction_early_stopped": 0,
            "transaction_budget_exhausted": 0,
            "transaction_reference_guard_rejected": 0,
            "transaction_best_iteration_sum": 0,
            "transaction_runtime_seconds": 0.0,
            "base_runtime_seconds": 0.0,
            "refinement_time_target_ratio": 0.08,
        }
        self.pose_render_update_gate = str(
            getattr(args, "paper_aligned_pose_render_update_gate", "off") or "off"
        )
        self.pose_render_update_gate_min_confidence = float(
            getattr(args, "paper_aligned_pose_render_update_gate_min_confidence", 0.35)
            or 0.35
        )
        self.pose_render_update_gate_max_pose_risk = float(
            getattr(args, "paper_aligned_pose_render_update_gate_max_pose_risk", 0.55)
            or 0.55
        )
        self.pose_render_update_gate_max_utility_drift = float(
            getattr(args, "paper_aligned_pose_render_update_gate_max_utility_drift", 0.75)
            or 0.75
        )
        self.pose_render_update_gate_soft_min_scale = float(
            getattr(args, "paper_aligned_pose_render_update_gate_soft_min_scale", 0.65)
            or 0.65
        )
        self.pose_render_update_gate_psnr_health_min_score = float(
            getattr(
                args,
                "paper_aligned_pose_render_update_gate_psnr_health_min_score",
                0.0,
            )
            or 0.0
        )
        self.pose_render_update_gate_psnr_health_min_raw_loss = float(
            getattr(
                args,
                "paper_aligned_pose_render_update_gate_psnr_health_min_raw_loss",
                0.0,
            )
            or 0.0
        )
        self.pose_render_update_gate_psnr_health_scale = float(
            getattr(
                args,
                "paper_aligned_pose_render_update_gate_psnr_health_scale",
                1.0,
            )
            or 1.0
        )
        self.pose_render_update_gate_stats = {
            "mode": self.pose_render_update_gate,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "render_frame_policy": self.paper_aligned_render_frame_policy,
            "min_confidence": self.pose_render_update_gate_min_confidence,
            "max_pose_risk": self.pose_render_update_gate_max_pose_risk,
            "max_utility_drift": self.pose_render_update_gate_max_utility_drift,
            "soft_min_scale": self.pose_render_update_gate_soft_min_scale,
            "psnr_health_min_score": (
                self.pose_render_update_gate_psnr_health_min_score
            ),
            "psnr_health_min_raw_loss": (
                self.pose_render_update_gate_psnr_health_min_raw_loss
            ),
            "psnr_health_scale": self.pose_render_update_gate_psnr_health_scale,
            "events": 0,
            "allowed": 0,
            "blocked": 0,
            "off": 0,
            "unknown_mode": 0,
            "direct_density_mismatch": 0,
            "test_frame": 0,
            "missing_gate_allow": 0,
            "pose_confidence_ok": 0,
            "pose_confidence_low": 0,
            "pose_confidence_soft_scaled": 0,
            "psnr_health_alias_scaled": 0,
            "pose_risk_high": 0,
            "pose_risk_score_high": 0,
            "pose_risk_reference": 0,
            "confidence_sum": 0.0,
            "gaussian_update_scale_sum": 0.0,
            "gaussian_update_scaled": 0,
            "psnr_health_score_sum": 0.0,
            "psnr_health_raw_loss_sum": 0.0,
            "psnr_health_alias_scale_sum": 0.0,
            "pose_risk_sum": 0.0,
            "pose_risk_max": 0.0,
            "confidence_min": 1.0,
            "utility_drift_sum": 0.0,
        }
        self.pose_render_pre_refine = str(
            getattr(args, "paper_aligned_pose_render_pre_refine", "off") or "off"
        )
        self.pose_render_pre_refine_iterations = int(
            getattr(args, "paper_aligned_pose_render_pre_refine_iterations", 2) or 0
        )
        self.pose_render_pre_refine_min_pose_risk = float(
            getattr(args, "paper_aligned_pose_render_pre_refine_min_pose_risk", 0.08)
            or 0.08
        )
        self.pose_render_pre_refine_max_pose_risk = float(
            getattr(args, "paper_aligned_pose_render_pre_refine_max_pose_risk", 0.38)
            or 0.38
        )
        self.pose_render_pre_refine_min_existing_gaussians = int(
            getattr(
                args,
                "paper_aligned_pose_render_pre_refine_min_existing_gaussians",
                5000,
            )
            or 0
        )
        self.pose_render_pre_refine_min_existing_keyframes = int(
            getattr(
                args,
                "paper_aligned_pose_render_pre_refine_min_existing_keyframes",
                8,
            )
            or 0
        )
        self.pose_render_pre_refine_min_render_coverage = float(
            getattr(
                args,
                "paper_aligned_pose_render_pre_refine_min_render_coverage",
                0.18,
            )
            or 0.18
        )
        self.pose_render_pre_refine_stats = {
            "mode": self.pose_render_pre_refine,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "iterations": self.pose_render_pre_refine_iterations,
            "min_pose_risk": self.pose_render_pre_refine_min_pose_risk,
            "max_pose_risk": self.pose_render_pre_refine_max_pose_risk,
            "min_existing_gaussians": self.pose_render_pre_refine_min_existing_gaussians,
            "min_existing_keyframes": self.pose_render_pre_refine_min_existing_keyframes,
            "min_render_coverage": self.pose_render_pre_refine_min_render_coverage,
            "events": 0,
            "applied": 0,
            "off": 0,
            "unknown_mode": 0,
            "direct_density_mismatch": 0,
            "test_frame": 0,
            "zero_iterations": 0,
            "direct_keyframe_not_finalized": 0,
            "map_not_mature_keyframes": 0,
            "map_not_mature_gaussians": 0,
            "pose_risk_too_high": 0,
            "pose_risk_low": 0,
            "pose_only_pre_refine": 0,
            "insufficient_render_coverage": 0,
            "no_gaussians": 0,
            "loss_sum": 0.0,
            "pose_risk_sum": 0.0,
            "render_coverage_sum": 0.0,
            "rotation_delta_deg_sum": 0.0,
            "translation_delta_sum": 0.0,
        }
        self.pose_render_init_weighting = str(
            getattr(args, "paper_aligned_pose_render_init_weighting", "off") or "off"
        )
        self.pose_render_init_weighting_min_pose_risk = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_min_pose_risk",
                0.16,
            )
            or 0.16
        )
        self.pose_render_init_weighting_max_pose_risk = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_max_pose_risk",
                0.38,
            )
            or 0.38
        )
        self.pose_render_init_weighting_min_sample_opacity_scale = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_min_sample_opacity_scale",
                0.82,
            )
            or 0.82
        )
        self.pose_render_init_weighting_min_match_opacity_scale = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_min_match_opacity_scale",
                0.92,
            )
            or 0.92
        )
        self.pose_render_init_weighting_max_representation_value = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_max_representation_value",
                0.65,
            )
            or 0.65
        )
        self.pose_render_init_weighting_max_novelty_value = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_max_novelty_value",
                0.65,
            )
            or 0.65
        )
        self.pose_render_init_weighting_adaptive_min_pose_risk = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_adaptive_min_pose_risk",
                0.20,
            )
            or 0.20
        )
        self.pose_render_init_weighting_adaptive_max_pose_risk = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_adaptive_max_pose_risk",
                0.32,
            )
            or 0.32
        )
        self.pose_render_init_weighting_adaptive_min_sample_opacity_scale = float(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_min_sample_opacity_scale"
                ),
                0.88,
            )
            or 0.88
        )
        self.pose_render_init_weighting_adaptive_min_match_opacity_scale = float(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_min_match_opacity_scale"
                ),
                0.95,
            )
            or 0.95
        )
        self.pose_render_init_weighting_adaptive_novelty_threshold = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_adaptive_novelty_threshold",
                0.45,
            )
            or 0.45
        )
        self.pose_render_init_weighting_adaptive_risk_override_alpha = float(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_adaptive_risk_override_alpha",
                1.0,
            )
            or 1.0
        )
        self.pose_render_init_weighting_adaptive_scene_min_events = int(
            getattr(
                args,
                "paper_aligned_pose_render_init_weighting_adaptive_scene_min_events",
                16,
            )
            or 16
        )
        self.pose_render_init_weighting_adaptive_scene_gentle_pose_risk_mean = float(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_scene_gentle_pose_risk_mean"
                ),
                0.20,
            )
            or 0.20
        )
        self.pose_render_init_weighting_adaptive_scene_gentle_novelty_mean = float(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_scene_gentle_novelty_mean"
                ),
                0.35,
            )
            or 0.35
        )
        adaptive_scene_max_risk_alpha_mean = getattr(
            args,
            (
                "paper_aligned_pose_render_init_weighting_"
                "adaptive_scene_max_risk_alpha_mean"
            ),
            1.0,
        )
        self.pose_render_init_weighting_adaptive_scene_max_risk_alpha_mean = (
            1.0
            if adaptive_scene_max_risk_alpha_mean is None
            else float(adaptive_scene_max_risk_alpha_mean)
        )
        adaptive_scene_low_pose_risk_mean = getattr(
            args,
            (
                "paper_aligned_pose_render_init_weighting_"
                "adaptive_scene_low_pose_risk_mean"
            ),
            -1.0,
        )
        self.pose_render_init_weighting_adaptive_scene_low_pose_risk_mean = (
            -1.0
            if adaptive_scene_low_pose_risk_mean is None
            else float(adaptive_scene_low_pose_risk_mean)
        )
        adaptive_scene_low_novelty_mean = getattr(
            args,
            (
                "paper_aligned_pose_render_init_weighting_"
                "adaptive_scene_low_novelty_mean"
            ),
            -1.0,
        )
        self.pose_render_init_weighting_adaptive_scene_low_novelty_mean = (
            -1.0
            if adaptive_scene_low_novelty_mean is None
            else float(adaptive_scene_low_novelty_mean)
        )
        self.pose_render_init_weighting_adaptive_scene_low_risk_min_events = int(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_scene_low_risk_min_events"
                ),
                64,
            )
            or 0
        )
        self.pose_render_init_weighting_adaptive_scene_stable_low_risk_min_events = int(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_scene_stable_low_risk_min_events"
                ),
                32,
            )
            or 0
        )
        self.pose_render_init_weighting_adaptive_scene_stable_low_pose_risk_mean = float(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_scene_stable_low_pose_risk_mean"
                ),
                0.06,
            )
            or 0.06
        )
        self.pose_render_init_weighting_adaptive_scene_stable_low_novelty_mean = float(
            getattr(
                args,
                (
                    "paper_aligned_pose_render_init_weighting_"
                    "adaptive_scene_stable_low_novelty_mean"
                ),
                0.12,
            )
            or 0.12
        )
        adaptive_scene_high_uncertainty_bypass_risk_alpha_mean = getattr(
            args,
            (
                "paper_aligned_pose_render_init_weighting_"
                "adaptive_scene_high_uncertainty_bypass_risk_alpha_mean"
            ),
            2.0,
        )
        self.pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_risk_alpha_mean = (
            2.0
            if adaptive_scene_high_uncertainty_bypass_risk_alpha_mean is None
            else float(adaptive_scene_high_uncertainty_bypass_risk_alpha_mean)
        )
        adaptive_scene_high_uncertainty_bypass_novelty_mean = getattr(
            args,
            (
                "paper_aligned_pose_render_init_weighting_"
                "adaptive_scene_high_uncertainty_bypass_novelty_mean"
            ),
            2.0,
        )
        self.pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_novelty_mean = (
            2.0
            if adaptive_scene_high_uncertainty_bypass_novelty_mean is None
            else float(adaptive_scene_high_uncertainty_bypass_novelty_mean)
        )
        self.pose_render_init_weighting_stats = {
            "mode": self.pose_render_init_weighting,
            "direct_density_mode": self.paper_aligned_direct_density_control,
            "min_pose_risk": self.pose_render_init_weighting_min_pose_risk,
            "max_pose_risk": self.pose_render_init_weighting_max_pose_risk,
            "min_sample_opacity_scale": (
                self.pose_render_init_weighting_min_sample_opacity_scale
            ),
            "min_match_opacity_scale": (
                self.pose_render_init_weighting_min_match_opacity_scale
            ),
            "max_representation_value": self.pose_render_init_weighting_max_representation_value,
            "max_novelty_value": self.pose_render_init_weighting_max_novelty_value,
            "adaptive_min_pose_risk": (
                self.pose_render_init_weighting_adaptive_min_pose_risk
            ),
            "adaptive_max_pose_risk": (
                self.pose_render_init_weighting_adaptive_max_pose_risk
            ),
            "adaptive_min_sample_opacity_scale": (
                self.pose_render_init_weighting_adaptive_min_sample_opacity_scale
            ),
            "adaptive_min_match_opacity_scale": (
                self.pose_render_init_weighting_adaptive_min_match_opacity_scale
            ),
            "adaptive_novelty_threshold": (
                self.pose_render_init_weighting_adaptive_novelty_threshold
            ),
            "adaptive_risk_override_alpha": (
                self.pose_render_init_weighting_adaptive_risk_override_alpha
            ),
            "adaptive_scene_min_events": (
                self.pose_render_init_weighting_adaptive_scene_min_events
            ),
            "adaptive_scene_gentle_pose_risk_mean": (
                self.pose_render_init_weighting_adaptive_scene_gentle_pose_risk_mean
            ),
            "adaptive_scene_gentle_novelty_mean": (
                self.pose_render_init_weighting_adaptive_scene_gentle_novelty_mean
            ),
            "adaptive_scene_max_risk_alpha_mean": (
                self.pose_render_init_weighting_adaptive_scene_max_risk_alpha_mean
            ),
            "adaptive_scene_low_pose_risk_mean": (
                self.pose_render_init_weighting_adaptive_scene_low_pose_risk_mean
            ),
            "adaptive_scene_low_novelty_mean": (
                self.pose_render_init_weighting_adaptive_scene_low_novelty_mean
            ),
            "adaptive_scene_low_risk_min_events": (
                self.pose_render_init_weighting_adaptive_scene_low_risk_min_events
            ),
            "adaptive_scene_stable_low_risk_min_events": (
                self.pose_render_init_weighting_adaptive_scene_stable_low_risk_min_events
            ),
            "adaptive_scene_stable_low_pose_risk_mean": (
                self.pose_render_init_weighting_adaptive_scene_stable_low_pose_risk_mean
            ),
            "adaptive_scene_stable_low_novelty_mean": (
                self.pose_render_init_weighting_adaptive_scene_stable_low_novelty_mean
            ),
            "adaptive_scene_gentle": 0,
            "adaptive_scene_high_uncertainty_bypass": 0,
            "adaptive_scene_low_risk_bypass": 0,
            "adaptive_scene_stable_low_risk_bypass": 0,
            "adaptive_scene_mature_low_risk_bypass": 0,
            "adaptive_scene_stable_low_risk_latched": 0,
            "adaptive_scene_stable_low_risk_latch_events": 0,
            "adaptive_scene_high_uncertainty_bypass_risk_alpha_mean": (
                self.pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_risk_alpha_mean
            ),
            "adaptive_scene_high_uncertainty_bypass_novelty_mean": (
                self.pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_novelty_mean
            ),
            "events": 0,
            "applied": 0,
            "off": 0,
            "unknown_mode": 0,
            "direct_density_mismatch": 0,
            "test_frame": 0,
            "pose_risk_low": 0,
            "representation_value_high": 0,
            "novelty_value_high": 0,
            "risk_opacity_scale": 0,
            "adaptive_novelty_preserve": 0,
            "risk_opacity_adaptive_scale": 0,
            "adaptive_risk_override": 0,
            "pose_risk_sum": 0.0,
            "risk_alpha_sum": 0.0,
            "representation_value_sum": 0.0,
            "novelty_value_sum": 0.0,
            "effective_min_pose_risk_sum": 0.0,
            "effective_max_pose_risk_sum": 0.0,
            "effective_min_sample_opacity_scale_sum": 0.0,
            "effective_min_match_opacity_scale_sum": 0.0,
            "sample_opacity_scale_sum": 0.0,
            "match_opacity_scale_sum": 0.0,
            "early_trace": [],
        }
        self.defer_recovery_bridge = None
        self._recovery_source_frame_id = -1
        self._last_recovery_bridge_meta: dict[str, object] = {}
        if not inference_mode:
            from paper_aligned_policy.defer_recovery_support_bridge import (
                DeferRecoverySupportBridge,
                bridge_enabled,
            )

            if bridge_enabled(args):
                self.defer_recovery_bridge = DeferRecoverySupportBridge(args)
        self.min_num_inliers = int(getattr(args, "min_num_inliers", 0) or 0)
        self.last_prev_keyframes_debug = {}

        # ========== LPIPS评估器初始化 ==========
        # 用于评估渲染质量（感知损失）
        try:
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            warnings.filterwarnings("ignore")
            self.lpips = lpips.LPIPS(net="vgg").cuda()
            sys.stdout = original_stdout
        except:
            self.lpips = None

        # ========== 训练模式下的初始化 ==========
        if not inference_mode:
            self.num_prev_keyframes_check = args.num_prev_keyframes_check  # 用于匹配的关键帧搜索窗口
            self.active_sh_degree = args.sh_degree  # 当前使用的球谐函数阶数
            self.max_sh_degree = args.sh_degree  # 最大球谐函数阶数
            self.lambda_dssim = args.lambda_dssim  # DSSIM损失权重
            self.init_proba_scaler = args.init_proba_scaler  # 高斯初始化概率缩放因子
            self.max_active_keyframes = args.max_active_keyframes  # 最大活跃关键帧数（超过后移到CPU）
            self.use_last_frame_proba = args.use_last_frame_proba  # 使用最新帧进行训练的概率
            self.active_frames_cpu = []  # CPU上的关键帧索引
            self.active_frames_gpu = []  # GPU上的关键帧索引
            self.guided_mvs = GuidedMVS(args)  # 【场景表示模块】引导多视图立体匹配（用于深度估计）

            # 学习率配置（位置参数使用衰减学习率）
            self.lr_dict = {
                "xyz": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            # ========== 高斯参数初始化 ==========
            # 3D高斯点云的所有可优化参数
            self.gaussian_params = {
                "xyz": {
                    "val": torch.empty(0, 3, device="cuda"),  # 3D位置 [N, 3]
                    "lr": args.position_lr_init,
                },
                "f_dc": {
                    "val": torch.empty(0, 1, 3, device="cuda"),  # 球谐函数DC项（基础颜色）[N, 1, 3]
                    "lr": args.feature_lr,
                },
                "f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device="cuda",
                    ),  # 球谐函数高阶项（视角相关颜色）[N, SH_rest, 3]
                    "lr": args.feature_lr / 20.0,
                },
                "scaling": {
                    "val": torch.empty(0, 3, device="cuda"),  # 尺度（log空间）[N, 3]
                    "lr": args.scaling_lr,
                },
                "rotation": {
                    "val": torch.empty(0, 4, device="cuda"),  # 旋转四元数 [N, 4]
                    "lr": args.rotation_lr,
                },
                "opacity": {
                    "val": torch.empty(0, 1, device="cuda"),  # 不透明度（logit空间）[N, 1]
                    "lr": args.opacity_lr,
                },
            }
            # 创建第一个活跃锚点（包含所有高斯点）
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors = [self.active_anchor]
            # 初始化优化器
            self.reset_optimizer()

        # ========== 关键帧和锚点管理 ==========
        self.keyframes = []  # 所有关键帧列表
        self.anchor_weights = [1.0]  # 锚点混合权重（用于多锚点融合）
        self.f = 0.7 * width  # 初始焦距（像素单位，基于图像宽度的经验值）
        self.init_intrinsics()  # 初始化内参（FoV、投影矩阵等）

        # ========== 位姿管理 ==========
        self.approx_cam_centres = None  # 所有关键帧的近似相机中心（用于距离计算）
        self.gt_Rts = torch.empty(0, 4, 4, device="cuda")  # 真实位姿矩阵（4x4，用于评估）
        self.gt_Rts_mask = torch.empty(0, device="cuda", dtype=bool)  # 真实位姿有效性掩码
        self.gt_f = self.f  # 真实焦距
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")  # 缓存的优化后位姿（避免重复计算）
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)  # 位姿缓存有效性标记
        self.sorted_frame_indices = None  # 按距离排序的关键帧索引（用于匹配搜索）
        self.last_trained_id = 0  # 最后一次训练的关键帧ID
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)  # 关键帧有效性掩码
        self.lock = threading.Lock()  # 线程锁（用于多线程场景下的高斯参数访问）
        self.inference_mode = inference_mode  # 是否为推理模式

        # ========== 高斯初始化辅助工具 ==========
        # 圆盘卷积核：用于计算图像的Laplacian范数（检测纹理/边缘区域）
        radius = 3  # 卷积核半径
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        # 创建圆形掩码（距离中心<=radius+0.5的像素置为1）
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()  # 归一化（平均池化）

        # 像素坐标网格：每个像素的(u,v)坐标，用于深度估计时的坐标查询
        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )

    def reset_optimizer(self):
        """
        【优化模块】重置优化器

        重新初始化Adam优化器，确保所有高斯参数都启用梯度计算。
        用于在锚点切换或优化器重置时调用。
        """
        # 确保所有高斯参数都启用梯度计算
        for key in self.gaussian_params:
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        # 创建新的稀疏高斯Adam优化器
        # (0.5, 0.99): Adam的beta参数（动量衰减率）
        self.optimizer = SparseGaussianAdam(
            self.gaussian_params, (0.5, 0.99), lr_dict=self.lr_dict
        )

    @property
    def xyz(self):
        """
        【属性】获取高斯点的3D位置 [N, 3]

        Returns:
            torch.Tensor: 高斯点的世界坐标系位置
        """
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        """
        【属性】获取球谐函数DC项（基础颜色）[N, 1, 3]

        Returns:
            torch.Tensor: 球谐函数的基础颜色系数
        """
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        """
        【属性】获取球谐函数高阶项（视角相关颜色）[N, SH_rest, 3]

        Returns:
            torch.Tensor: 球谐函数的高阶系数（控制视角相关的外观变化）
        """
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        """
        【属性】获取高斯点的尺度 [N, 3]（从log空间转换）

        Returns:
            torch.Tensor: 高斯点在三个轴上的尺度（指数空间）
        """
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        """
        【属性】获取高斯点的旋转四元数 [N, 4]（归一化）

        Returns:
            torch.Tensor: 归一化的四元数旋转
        """
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        """
        【属性】获取高斯点的不透明度 [N, 1]（从logit空间转换）

        Returns:
            torch.Tensor: 不透明度值 [0, 1]
        """
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def n_active_gaussians(self):
        """
        【属性】获取当前活跃的高斯点数量

        Returns:
            int: 高斯点数量
        """
        return self.xyz.shape[0]

    @classmethod
    def from_scene(cls, scene_dir: str, args):
        """
        【场景表示模块】从保存的场景目录加载场景模型

        从场景目录读取元数据和锚点，重建场景模型。
        用于推理模式下的场景加载。

        Args:
            scene_dir: 场景目录路径（包含metadata.json和point_clouds/）
            args: 配置参数

        Returns:
            SceneModel: 加载的场景模型实例
        """
        # 读取元数据文件
        with open(os.path.join(scene_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # 从元数据获取图像尺寸
        width = metadata["config"]["width"]
        height = metadata["config"]["height"]
        # 创建场景模型（推理模式）
        scene_model = cls(width, height, args, inference_mode=True)
        scene_model.active_sh_degree = metadata["config"]["sh_degree"]
        scene_model.max_sh_degree = metadata["config"]["sh_degree"]

        # ========== 加载锚点 ==========
        # 从PLY文件加载所有锚点的高斯点云
        scene_model.anchors = []
        for i in range(len(metadata["anchors"])):
            scene_model.anchors.append(
                Anchor.from_ply(
                    os.path.join(scene_dir, "point_clouds", f"anchor_{i}.ply"),
                    torch.tensor(metadata["anchors"][i]["position"]),
                    metadata["config"]["sh_degree"],
                )
            )

        # 设置第一个锚点为活跃锚点
        scene_model.active_anchor = scene_model.anchors[0]

        # ========== 加载关键帧 ==========
        # 从JSON文件加载所有关键帧的位姿和图像信息
        for i in range(len(metadata["keyframes"])):
            keyframe = Keyframe.from_json(metadata["keyframes"][i], i, width, height)
            scene_model.add_keyframe(keyframe)

        return scene_model

    @property
    def first_active_frame(self):
        """
        【属性】获取活跃锚点中第一个关键帧的索引

        Returns:
            int: 第一个关键帧的索引
        """
        return self.active_anchor.keyframe_ids[0]

    @property
    def last_active_frame(self):
        """
        【属性】获取活跃锚点中最后一个关键帧的索引

        Returns:
            int: 最后一个关键帧的索引
        """
        return self.active_anchor.keyframe_ids[-1]

    @property
    def n_active_keyframes(self):
        """
        【属性】获取活跃关键帧的数量

        Returns:
            int: 活跃关键帧数量
        """
        return self.last_active_frame - self.first_active_frame + 1

    def _deterministic_training_background_for_keyframe(
        self,
        keyframe: Keyframe,
        lvl: int,
        target: torch.Tensor,
        preserve_global_rng_stream: bool = False,
    ) -> torch.Tensor:
        if preserve_global_rng_stream:
            torch.rand(3, device=target.device)
        step = int(getattr(self, "_training_background_step", 0) or 0)
        self._training_background_step = step + 1
        keyframe_index = int(getattr(keyframe, "index", 0) or 0)
        seed = (
            (step + 1) * 1664525
            + (keyframe_index + 1) * 1013904223
            + (int(lvl) + 1) * 374761393
        ) & 0xFFFFFFFF

        def unit_float(salt: int) -> float:
            value = (seed ^ int(salt)) & 0xFFFFFFFF
            value = (1664525 * value + 1013904223) & 0xFFFFFFFF
            return float(value & 0x00FFFFFF) / float(0x01000000)

        return torch.tensor(
            [
                unit_float(0x9E3779B9),
                unit_float(0x7F4A7C15),
                unit_float(0x94D049BB),
            ],
            device=target.device,
            dtype=target.dtype,
        )

    def _training_background_for_keyframe(self, keyframe: Keyframe, lvl: int):
        mode = str(self.paper_aligned_training_background_mode or "random_v1")
        target = keyframe.image_pyr[lvl]
        if mode == "fixed_black_v1":
            return target.new_zeros(3)
        if mode in {
            "mask_aware_dark_scene_fixed_black_v1",
            "mask_aware_dark_scene_deterministic_random_v1",
        }:
            return self._mask_aware_dark_scene_training_background_for_keyframe(
                keyframe,
                lvl,
                target,
            )
        if mode == "mask_aware_fixed_black_v1":
            stats = self._ensure_training_background_stats()
            if getattr(keyframe, "mask_pyr", None) is not None:
                stats["mask_aware_fixed_black_applied"] = (
                    int(stats.get("mask_aware_fixed_black_applied", 0) or 0) + 1
                )
                return target.new_zeros(3)
            stats["mask_aware_random_fallback"] = (
                int(stats.get("mask_aware_random_fallback", 0) or 0) + 1
            )
            return torch.rand(3, device=target.device)
        if mode == "dark_scene_fixed_black_v1":
            return self._dark_scene_training_background_for_keyframe(
                keyframe,
                lvl,
                target,
            )
        if mode == "deterministic_random_v1":
            return self._deterministic_training_background_for_keyframe(
                keyframe,
                lvl,
                target,
            )
        if mode != "target_mean_v1":
            return torch.rand(3, device=target.device)

        if keyframe.mask_pyr is not None:
            mask = keyframe.mask_pyr[lvl].bool()
            if mask.ndim == target.ndim - 1:
                mask = mask.unsqueeze(0)
            mask = mask.expand_as(target)
            if int(mask.sum().item()) > 0:
                return target.detach()[mask].view(3, -1).mean(dim=1)
        return target.detach().mean(dim=(-2, -1))

    def _record_training_background_frame_decision(
        self,
        keyframe: Keyframe,
        *,
        decision: str,
        target_mean: float | None = None,
        valid_ratio: float | None = None,
    ) -> None:
        if not isinstance(getattr(keyframe, "info", None), dict):
            return
        frame_info: dict[str, object] = {
            "mode": str(self.paper_aligned_training_background_mode or "random_v1"),
            "decision": str(decision),
        }
        if target_mean is not None:
            frame_info["target_mean"] = float(target_mean)
        if valid_ratio is not None:
            frame_info["valid_ratio"] = float(valid_ratio)
        keyframe.info["_paper_aligned_training_background_frame"] = frame_info

    def _mask_aware_dark_scene_training_background_for_keyframe(
        self,
        keyframe: Keyframe,
        lvl: int,
        target: torch.Tensor,
    ):
        stats = self._ensure_training_background_stats()
        threshold = float(stats.get("mask_aware_dark_threshold", 0.48) or 0.48)
        decay = float(stats.get("mask_aware_dark_decay", 0.95) or 0.95)
        if getattr(keyframe, "mask_pyr", None) is None:
            stats["mask_aware_dark_random_fallback"] = (
                int(stats.get("mask_aware_dark_random_fallback", 0) or 0) + 1
            )
            self._record_training_background_frame_decision(
                keyframe,
                decision="mask_aware_dark_no_mask_random",
            )
            if (
                str(self.paper_aligned_training_background_mode or "")
                == "mask_aware_dark_scene_deterministic_random_v1"
            ):
                return self._deterministic_training_background_for_keyframe(
                    keyframe,
                    lvl,
                    target,
                    preserve_global_rng_stream=True,
                )
            return torch.rand(3, device=target.device)

        mask = keyframe.mask_pyr[lvl].bool()
        spatial_mask = mask[0] if mask.ndim == target.ndim else mask
        valid_count = int(spatial_mask.sum().item())
        total_count = int(spatial_mask.numel())
        valid_ratio = float(valid_count) / float(max(total_count, 1))
        if valid_count > 0:
            channel_mask = spatial_mask
            if channel_mask.ndim == target.ndim - 1:
                channel_mask = channel_mask.unsqueeze(0)
            channel_mask = channel_mask.expand_as(target)
            target_mean = float(target.detach()[channel_mask].mean().item())
        else:
            target_mean = float(target.detach().mean().item())

        previous_mean = getattr(self, "_training_background_mask_aware_dark_mean", None)
        running_mean = (
            target_mean
            if previous_mean is None
            else decay * float(previous_mean) + (1.0 - decay) * target_mean
        )
        self._training_background_mask_aware_dark_mean = running_mean
        stats["mask_aware_dark_observations"] = (
            int(stats.get("mask_aware_dark_observations", 0) or 0) + 1
        )
        stats["mask_aware_dark_mean_sum"] = (
            float(stats.get("mask_aware_dark_mean_sum", 0.0) or 0.0) + target_mean
        )
        stats["mask_aware_dark_running_mean"] = running_mean
        stats["mask_aware_dark_valid_ratio_sum"] = (
            float(stats.get("mask_aware_dark_valid_ratio_sum", 0.0) or 0.0)
            + valid_ratio
        )

        if running_mean <= threshold:
            stats["mask_aware_dark_decision"] = "fixed_black"
            stats["mask_aware_dark_fixed_black_applied"] = (
                int(stats.get("mask_aware_dark_fixed_black_applied", 0) or 0) + 1
            )
            self._record_training_background_frame_decision(
                keyframe,
                decision="mask_aware_dark_fixed_black",
                target_mean=target_mean,
                valid_ratio=valid_ratio,
            )
            return target.new_zeros(3)

        stats["mask_aware_dark_decision"] = "random"
        stats["mask_aware_dark_random_fallback"] = (
            int(stats.get("mask_aware_dark_random_fallback", 0) or 0) + 1
        )
        self._record_training_background_frame_decision(
            keyframe,
            decision="mask_aware_dark_random",
            target_mean=target_mean,
            valid_ratio=valid_ratio,
        )
        if (
            str(self.paper_aligned_training_background_mode or "")
            == "mask_aware_dark_scene_deterministic_random_v1"
        ):
            return self._deterministic_training_background_for_keyframe(
                keyframe,
                lvl,
                target,
                preserve_global_rng_stream=True,
            )
        return torch.rand(3, device=target.device)

    def _new_training_background_stats(self) -> dict[str, object]:
        return {
            "mode": str(
                getattr(self, "paper_aligned_training_background_mode", "random_v1")
                or "random_v1"
            ),
            "dark_scene_threshold": float(
                getattr(self, "_training_background_dark_scene_threshold", 0.34)
            ),
            "dark_scene_decay": float(
                getattr(self, "_training_background_scene_mean_decay", 0.95)
            ),
            "dark_scene_observations": 0,
            "dark_scene_mean_sum": 0.0,
            "dark_scene_running_mean": None,
            "dark_scene_decision": None,
            "dark_scene_fixed_black_applied": 0,
            "dark_scene_random_fallback": 0,
            "dark_scene_mask_blocked": 0,
            "mask_aware_fixed_black_applied": 0,
            "mask_aware_random_fallback": 0,
            "mask_aware_dark_threshold": 0.48,
            "mask_aware_dark_decay": 0.95,
            "mask_aware_dark_observations": 0,
            "mask_aware_dark_mean_sum": 0.0,
            "mask_aware_dark_valid_ratio_sum": 0.0,
            "mask_aware_dark_running_mean": None,
            "mask_aware_dark_decision": None,
            "mask_aware_dark_fixed_black_applied": 0,
            "mask_aware_dark_random_fallback": 0,
        }

    def _ensure_training_background_stats(self) -> dict[str, object]:
        stats = getattr(self, "training_background_stats", None)
        if not isinstance(stats, dict):
            stats = self._new_training_background_stats()
            self.training_background_stats = stats
        for key, value in self._new_training_background_stats().items():
            stats.setdefault(key, value)
        return stats

    def _attach_training_background_info(self, keyframe: Keyframe) -> None:
        if isinstance(getattr(keyframe, "info", None), dict):
            keyframe.info["_paper_aligned_training_background"] = dict(
                self._ensure_training_background_stats()
            )

    def _dark_fixed_black_scene_latched(self) -> bool:
        if str(self.paper_aligned_training_background_mode or "") != "dark_scene_fixed_black_v1":
            return False
        stats = self._ensure_training_background_stats()
        return (
            getattr(self, "_training_background_scene_dark_decision", None) is True
            and str(stats.get("dark_scene_decision", "")) == "fixed_black"
            and int(stats.get("dark_scene_mask_blocked", 0) or 0) == 0
        )

    def _dark_scene_mask_blocked_scene_latched(self) -> bool:
        if str(self.paper_aligned_training_background_mode or "") != "dark_scene_fixed_black_v1":
            return False
        stats = self._ensure_training_background_stats()
        return (
            int(stats.get("dark_scene_mask_blocked", 0) or 0) > 0
            and int(stats.get("dark_scene_observations", 0) or 0) == 0
        )

    def _dark_scene_training_background_for_keyframe(
        self,
        keyframe: Keyframe,
        lvl: int,
        target: torch.Tensor,
    ):
        stats = self._ensure_training_background_stats()
        if getattr(keyframe, "mask_pyr", None) is not None:
            stats["dark_scene_mask_blocked"] = int(stats["dark_scene_mask_blocked"]) + 1
            stats["dark_scene_random_fallback"] = (
                int(stats["dark_scene_random_fallback"]) + 1
            )
            return torch.rand(3, device=target.device)

        latched_decision = getattr(
            self,
            "_training_background_scene_dark_decision",
            None,
        )
        if latched_decision is True:
            stats["dark_scene_fixed_black_applied"] = (
                int(stats["dark_scene_fixed_black_applied"]) + 1
            )
            return target.new_zeros(3)
        if latched_decision is False:
            stats["dark_scene_random_fallback"] = (
                int(stats["dark_scene_random_fallback"]) + 1
            )
            return torch.rand(3, device=target.device)

        threshold = float(
            getattr(self, "_training_background_dark_scene_threshold", 0.34)
        )
        decay = float(getattr(self, "_training_background_scene_mean_decay", 0.95))
        target_mean = float(target.detach().mean().item())
        previous_mean = getattr(self, "_training_background_scene_mean", None)
        running_mean = (
            target_mean
            if previous_mean is None
            else decay * float(previous_mean) + (1.0 - decay) * target_mean
        )
        self._training_background_scene_mean = running_mean
        stats["dark_scene_observations"] = int(stats["dark_scene_observations"]) + 1
        stats["dark_scene_mean_sum"] = float(stats["dark_scene_mean_sum"]) + target_mean
        stats["dark_scene_running_mean"] = running_mean

        if running_mean <= threshold:
            self._training_background_scene_dark_decision = True
            stats["dark_scene_decision"] = "fixed_black"
            stats["dark_scene_fixed_black_applied"] = (
                int(stats["dark_scene_fixed_black_applied"]) + 1
            )
            return target.new_zeros(3)

        self._training_background_scene_dark_decision = False
        stats["dark_scene_decision"] = "random"
        stats["dark_scene_random_fallback"] = int(stats["dark_scene_random_fallback"]) + 1
        return torch.rand(3, device=target.device)

    def optimization_step(self, finetuning=False, keyframe_id_override=None, update_pose=True):
        """
        【优化模块】执行一步优化

        这是训练的核心函数，执行以下步骤：
        1. 选择要优化的关键帧
        2. 从该关键帧视角渲染场景
        3. 计算损失（L1 + DSSIM + 深度损失）
        4. 反向传播并更新参数（高斯参数 + 关键帧位姿）

        Args:
            finetuning: 是否为微调模式（微调时随机选择关键帧）
        """
        if len(self.xyz) == 0:
            return

        # ========== 关键帧选择策略 ==========
        # 训练策略：以一定概率使用最新关键帧，否则随机选择
        # 这样可以平衡新区域的学习和旧区域的细化
        keyframe_sampling_debug = None
        if keyframe_id_override is not None:
            keyframe_id = int(keyframe_id_override)
        elif (
            np.random.rand() > self.use_last_frame_proba
            or self.last_trained_id == -1
            or finetuning
        ):
            if self.pose_render_keyframe_sampling == "off":
                keyframe_id = np.random.choice(self.active_frames_gpu)
            else:
                keyframe_id, keyframe_sampling_debug = choose_pose_render_keyframe_id(
                    candidate_ids=list(self.active_frames_gpu),
                    keyframe_infos=[
                        self.keyframes[int(index)].info
                        for index in self.active_frames_gpu
                    ],
                    mode=self.pose_render_keyframe_sampling,
                    direct_density_mode=self.paper_aligned_direct_density_control,
                    rng=np.random,
                    min_weight=self.pose_render_keyframe_sampling_min_weight,
                    max_pose_risk=self.pose_render_keyframe_sampling_max_pose_risk,
                    max_utility_drift=(
                        self.pose_render_keyframe_sampling_max_utility_drift
                    ),
                    min_pose_support=(
                        self.pose_render_keyframe_sampling_min_pose_support
                    ),
                    min_match_support=(
                        self.pose_render_keyframe_sampling_min_match_support
                    ),
                )
                self._record_pose_render_keyframe_sampling(keyframe_sampling_debug)
        else:
            keyframe_id = -1  # 使用最新关键帧
        keyframe = self.keyframes[keyframe_id]
        if keyframe_sampling_debug is not None:
            keyframe.info["_paper_aligned_pose_render_keyframe_sampling"] = (
                keyframe_sampling_debug
            )
        lvl = keyframe.pyr_lvl  # 当前使用的金字塔层级

        # ========== 梯度清零 ==========
        keyframe.zero_grad()
        self.optimizer.zero_grad()

        # ========== 渲染 ==========
        # 【渲染模块】从关键帧视角渲染图像和深度
        render_pkg = self.render_from_id(
            keyframe_id,
            pyr_lvl=lvl,
            bg=self._training_background_for_keyframe(keyframe, lvl),
        )
        image = render_pkg["render"]  # 渲染的RGB图像 [3, H, W]
        invdepth = render_pkg["invdepth"]  # 渲染的逆深度 [1, H, W]

        # 获取真实图像和单目深度
        gt_image = keyframe.image_pyr[lvl]
        mono_idepth = keyframe.get_mono_idepth(lvl)

        # ========== 掩码处理 ==========
        # 如果有关键帧掩码，应用掩码（排除无效区域）
        if keyframe.mask_pyr is not None:
            image = image * keyframe.mask_pyr[lvl]
            gt_image = gt_image * keyframe.mask_pyr[lvl]
            invdepth = invdepth * keyframe.mask_pyr[lvl]
            mono_idepth = mono_idepth * keyframe.mask_pyr[lvl]

        # ========== 损失计算 ==========
        # 【损失函数模块】计算多任务损失
        l1_loss = (image - gt_image).abs().mean()  # L1损失（像素级）
        ssim_loss = 1 - fused_ssim(image[None], gt_image[None])  # DSSIM损失（结构相似性）
        depth_loss = (invdepth - mono_idepth).abs().mean()  # 深度损失（与单目深度对齐）
        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
        )
        if (
            (
                self.pose_render_extra_optimization
                == "pose_confidence_render_response_v2"
                or self.pose_render_extra_optimization == "render_response_v3"
                or self.pose_render_texture_sampling
                in {
                    TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,
                    TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE,
                    TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
                    TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
                }
            )
            and not self._pose_render_response_tracking_bypassed()
            and self.keyframes
            and keyframe is self.keyframes[-1]
            and not bool(keyframe.info.get("is_test", False))
        ):
            rgb_mse = (image.detach() - gt_image.detach()).square().mean()
            representation_gap = None
            if self.pose_render_extra_optimization == "render_response_v3":
                valid_mask = (
                    keyframe.mask_pyr[lvl]
                    if keyframe.mask_pyr is not None
                    else None
                )
                representation_gap = rendering_response_gap(
                    image.detach() - gt_image.detach(),
                    valid_mask,
                )
            self._record_pose_render_response(
                keyframe,
                rgb_mse,
                representation_gap=representation_gap,
            )
        edge_loss, edge_loss_debug = pose_render_gradient_loss(
            image,
            gt_image,
            mode=self.pose_render_edge_loss,
            direct_density_mode=self.paper_aligned_direct_density_control,
            keyframe_info=keyframe.info,
            weight=self.pose_render_edge_loss_weight,
            max_pose_risk=self.pose_render_edge_loss_max_pose_risk,
            max_utility_drift=self.pose_render_edge_loss_max_utility_drift,
            min_pose_support=self.pose_render_edge_loss_min_pose_support,
            min_match_support=self.pose_render_edge_loss_min_match_support,
            min_weight_scale=self.pose_render_edge_loss_min_weight_scale,
            target_raw_loss=self.pose_render_edge_loss_target_raw_loss,
            risk_free_threshold=self.pose_render_edge_loss_risk_free_threshold,
        )
        if edge_loss_debug.get("applied", False):
            loss = loss + edge_loss
        if self.pose_render_edge_loss != "off":
            keyframe.info["_paper_aligned_pose_render_edge_loss"] = edge_loss_debug
            self._record_pose_render_edge_loss(edge_loss_debug)
        psnr_budget_ratio = 0.0
        psnr_events = int(self.pose_render_psnr_loss_stats.get("events", 0))
        if psnr_events > 0:
            psnr_budget_ratio = float(
                self.pose_render_psnr_loss_stats.get("applied", 0)
            ) / float(psnr_events)
        psnr_budget_exhausted = (
            self.pose_render_psnr_loss != "off"
            and self.pose_render_psnr_loss_max_applied_ratio < 1.0
            and psnr_events > 0
            and psnr_budget_ratio >= self.pose_render_psnr_loss_max_applied_ratio
        )
        psnr_scene_guard_blocked = (
            self.pose_render_psnr_loss != "off"
            and self.pose_render_psnr_loss_scene_guard != "off"
            and self.pose_render_psnr_loss_scene_guard_triggered
        )
        if psnr_scene_guard_blocked:
            psnr_loss = image.new_zeros(())
            psnr_loss_debug = {
                "mode": self.pose_render_psnr_loss,
                "direct_density_mode": self.paper_aligned_direct_density_control,
                "render_frame_policy": self.paper_aligned_render_frame_policy,
                "is_test": bool(keyframe.info.get("is_test", False)),
                "applied": False,
                "reason": "psnr_scene_guard_blocked",
                "base_weight": self.pose_render_psnr_loss_weight,
                "weight": 0.0,
                "target_mask_mode": self.pose_render_psnr_loss_target_mask,
                "support_weight_mode": self.pose_render_psnr_loss_support_weight,
                "support_weight_scale": 1.0,
                "support_weight_reason": "off",
                "context_weight_mode": self.pose_render_psnr_loss_context_weight,
                "context_weight_scale": 1.0,
                "context_weight_reason": "off",
                "raw_loss": 0.0,
                "weighted_loss": 0.0,
                "scene_guard_mode": self.pose_render_psnr_loss_scene_guard,
                "scene_guard_triggered": True,
                "scene_guard_trigger_events": self.pose_render_psnr_loss_stats.get(
                    "scene_guard_trigger_events", 0
                ),
                "scene_guard_trigger_applied": self.pose_render_psnr_loss_stats.get(
                    "scene_guard_trigger_applied", 0
                ),
                "scene_guard_trigger_ratio": self.pose_render_psnr_loss_stats.get(
                    "scene_guard_trigger_ratio", 0.0
                ),
                "scene_guard_trigger_raw_loss_mean": (
                    self.pose_render_psnr_loss_stats.get(
                        "scene_guard_trigger_raw_loss_mean", 0.0
                    )
                ),
            }
        elif psnr_budget_exhausted:
            psnr_loss = image.new_zeros(())
            psnr_loss_debug = {
                "mode": self.pose_render_psnr_loss,
                "direct_density_mode": self.paper_aligned_direct_density_control,
                "render_frame_policy": self.paper_aligned_render_frame_policy,
                "is_test": bool(keyframe.info.get("is_test", False)),
                "applied": False,
                "reason": "psnr_budget_exhausted",
                "base_weight": self.pose_render_psnr_loss_weight,
                "weight": 0.0,
                "max_applied_ratio": self.pose_render_psnr_loss_max_applied_ratio,
                "budget_applied_ratio": psnr_budget_ratio,
                "target_mask_mode": self.pose_render_psnr_loss_target_mask,
                "support_weight_mode": self.pose_render_psnr_loss_support_weight,
                "support_weight_scale": 1.0,
                "support_weight_reason": "off",
                "context_weight_mode": self.pose_render_psnr_loss_context_weight,
                "context_weight_scale": 1.0,
                "context_weight_reason": "off",
                "raw_loss": 0.0,
                "weighted_loss": 0.0,
            }
        else:
            psnr_loss, psnr_loss_debug = pose_render_mse_loss(
                image,
                gt_image,
                mode=self.pose_render_psnr_loss,
                direct_density_mode=self.paper_aligned_direct_density_control,
                render_frame_policy=self.paper_aligned_render_frame_policy,
                keyframe_info=keyframe.info,
                weight=self.pose_render_psnr_loss_weight,
                target_mask_mode=self.pose_render_psnr_loss_target_mask,
                support_weight_mode=self.pose_render_psnr_loss_support_weight,
                context_weight_mode=self.pose_render_psnr_loss_context_weight,
                support_weight_min_scale=self.pose_render_psnr_loss_support_min_scale,
                support_weight_correspondence_low=self.pose_render_psnr_loss_support_corr_low,
                support_weight_correspondence_high=self.pose_render_psnr_loss_support_corr_high,
                support_weight_inlier_low=self.pose_render_psnr_loss_support_inlier_low,
                support_weight_inlier_high=self.pose_render_psnr_loss_support_inlier_high,
                max_pose_risk=self.pose_render_psnr_loss_max_pose_risk,
                max_utility_drift=self.pose_render_psnr_loss_max_utility_drift,
                min_pose_support=self.pose_render_psnr_loss_min_pose_support,
                min_match_support=self.pose_render_psnr_loss_min_match_support,
            )
            ambiguity_gate_enabled = (
                self.pose_render_psnr_loss_ambiguity_high
                > self.pose_render_psnr_loss_ambiguity_low
            )
            raw_psnr_loss = float(psnr_loss_debug.get("raw_loss", 0.0) or 0.0)
            in_ambiguity_band = (
                self.pose_render_psnr_loss_ambiguity_low
                <= raw_psnr_loss
                <= self.pose_render_psnr_loss_ambiguity_high
            )
            if (
                bool(psnr_loss_debug.get("applied", False))
                and ambiguity_gate_enabled
                and psnr_budget_ratio
                >= self.pose_render_psnr_loss_ambiguity_min_applied_ratio
                and in_ambiguity_band
            ):
                psnr_loss = image.new_zeros(())
                psnr_loss_debug.update(
                    {
                        "applied": False,
                        "reason": "psnr_residual_ambiguity_band",
                        "weight": 0.0,
                        "weighted_loss": 0.0,
                        "ambiguity_low": self.pose_render_psnr_loss_ambiguity_low,
                        "ambiguity_high": self.pose_render_psnr_loss_ambiguity_high,
                        "ambiguity_min_applied_ratio": (
                            self.pose_render_psnr_loss_ambiguity_min_applied_ratio
                        ),
                        "budget_applied_ratio": psnr_budget_ratio,
                    }
                )
            scene_low_should_block, scene_low_debug = (
                self._pose_render_psnr_scene_low_response_should_block(psnr_loss_debug)
            )
            if scene_low_should_block:
                psnr_loss = image.new_zeros(())
                psnr_loss_debug.update(scene_low_debug)
                psnr_loss_debug.update(
                    {
                        "applied": False,
                        "weight": 0.0,
                        "weighted_loss": 0.0,
                    }
                )
            elif scene_low_debug:
                psnr_loss_debug.update(scene_low_debug)
            scene_guard_should_block, scene_guard_debug = (
                self._pose_render_psnr_scene_guard_should_block(psnr_loss_debug)
            )
            if scene_guard_should_block:
                psnr_loss = image.new_zeros(())
                psnr_loss_debug.update(scene_guard_debug)
                psnr_loss_debug.update(
                    {
                        "applied": False,
                        "weight": 0.0,
                        "weighted_loss": 0.0,
                    }
                )
            health_gate_mode = str(self.pose_render_psnr_loss_health_gate or "off")
            if (
                bool(psnr_loss_debug.get("applied", False))
                and health_gate_mode != "off"
            ):
                if raw_psnr_loss < self.pose_render_psnr_loss_health_min_raw_loss:
                    psnr_loss_debug["health_gate_raw_wait"] = True
                elif health_gate_mode == "gradient_correlation_v1":
                    health_score, health_debug = pose_render_structure_alignment_score(
                        image, gt_image
                    )
                    psnr_loss_debug.update(health_debug)
                    psnr_loss_debug.update(
                        {
                            "health_gate_mode": health_gate_mode,
                            "health_gate_score": health_score,
                            "health_min_score": (
                                self.pose_render_psnr_loss_health_min_score
                            ),
                            "health_min_raw_loss": (
                                self.pose_render_psnr_loss_health_min_raw_loss
                            ),
                        }
                    )
                    if health_score < self.pose_render_psnr_loss_health_min_score:
                        psnr_loss = image.new_zeros(())
                        psnr_loss_debug.update(
                            {
                                "applied": False,
                                "reason": "psnr_health_gate_low",
                                "weight": 0.0,
                                "weighted_loss": 0.0,
                            }
                        )
                else:
                    psnr_loss = image.new_zeros(())
                    psnr_loss_debug.update(
                        {
                            "applied": False,
                            "reason": "unknown_health_gate_mode",
                            "weight": 0.0,
                            "weighted_loss": 0.0,
                        }
                    )
            late_raw_stop_mode = str(self.pose_render_psnr_loss_late_raw_stop or "off")
            if (
                bool(psnr_loss_debug.get("applied", False))
                and late_raw_stop_mode != "off"
            ):
                late_stop_stats = self.pose_render_psnr_loss_stats
                late_stop_events = int(late_stop_stats.get("events", 0)) + 1
                late_stop_applied = int(late_stop_stats.get("applied", 0)) + 1
                late_stop_raw_sum = float(
                    late_stop_stats.get("raw_loss_sum", 0.0)
                ) + raw_psnr_loss
                late_stop_raw_mean = late_stop_raw_sum / float(
                    max(late_stop_applied, 1)
                )
                late_stop_max_events = int(
                    self.pose_render_psnr_loss_scene_guard_max_events or 0
                )
                late_stop_window_expired = (
                    late_stop_max_events > 0
                    and late_stop_events > late_stop_max_events
                )
                psnr_loss_debug.update(
                    {
                        "late_raw_stop_mode": late_raw_stop_mode,
                        "late_raw_stop_window_expired": late_stop_window_expired,
                        "late_raw_stop_projected_events": late_stop_events,
                        "late_raw_stop_projected_raw_loss_mean": late_stop_raw_mean,
                        "late_raw_stop_min_mean": (
                            self.pose_render_psnr_loss_late_raw_stop_min_mean
                        ),
                    }
                )
                if late_raw_stop_mode == "raw_mean_stop_v1":
                    if (
                        late_stop_window_expired
                        and late_stop_raw_mean
                        >= self.pose_render_psnr_loss_late_raw_stop_min_mean
                    ):
                        psnr_loss = image.new_zeros(())
                        psnr_loss_debug.update(
                            {
                                "applied": False,
                                "reason": "psnr_late_raw_mean_stop",
                                "weight": 0.0,
                                "weighted_loss": 0.0,
                            }
                        )
                else:
                    psnr_loss = image.new_zeros(())
                    psnr_loss_debug.update(
                        {
                            "applied": False,
                            "reason": "unknown_late_raw_stop_mode",
                            "weight": 0.0,
                            "weighted_loss": 0.0,
                        }
                    )
            structure_gate_mode = str(self.pose_render_psnr_loss_structure_gate or "off")
            if (
                bool(psnr_loss_debug.get("applied", False))
                and structure_gate_mode != "off"
            ):
                structure_stats = self.pose_render_psnr_loss_stats
                structure_events = int(structure_stats.get("events", 0)) + 1
                structure_applied = int(structure_stats.get("applied", 0)) + 1
                structure_raw_sum = float(
                    structure_stats.get("raw_loss_sum", 0.0)
                ) + raw_psnr_loss
                structure_raw_mean = structure_raw_sum / float(
                    max(structure_applied, 1)
                )
                max_scene_events = int(
                    self.pose_render_psnr_loss_scene_guard_max_events or 0
                )
                structure_window_expired = (
                    max_scene_events <= 0 or structure_events > max_scene_events
                )
                psnr_loss_debug.update(
                    {
                        "structure_gate_window_expired": structure_window_expired,
                        "structure_gate_projected_events": structure_events,
                        "structure_gate_projected_raw_loss_mean": structure_raw_mean,
                        "structure_min_raw_mean": (
                            self.pose_render_psnr_loss_structure_min_raw_mean
                        ),
                    }
                )
                if not structure_window_expired:
                    structure_stats["psnr_structure_gate_window_wait"] = int(
                        structure_stats.get("psnr_structure_gate_window_wait", 0)
                    ) + 1
                    structure_gate_mode = "off"
                elif (
                    structure_raw_mean
                    < self.pose_render_psnr_loss_structure_min_raw_mean
                ):
                    structure_stats["psnr_structure_gate_raw_mean_low"] = int(
                        structure_stats.get("psnr_structure_gate_raw_mean_low", 0)
                    ) + 1
                    structure_gate_mode = "off"
            if (
                bool(psnr_loss_debug.get("applied", False))
                and structure_gate_mode != "off"
            ):
                if structure_gate_mode == "gradient_correlation_v1":
                    structure_score, structure_debug = (
                        pose_render_structure_alignment_score(image, gt_image)
                    )
                    psnr_loss_debug.update(structure_debug)
                    psnr_loss_debug["structure_min_score"] = (
                        self.pose_render_psnr_loss_structure_min_score
                    )
                    if structure_score < self.pose_render_psnr_loss_structure_min_score:
                        psnr_loss = image.new_zeros(())
                        psnr_loss_debug.update(
                            {
                                "applied": False,
                                "reason": "psnr_structure_gate_low",
                                "weight": 0.0,
                                "weighted_loss": 0.0,
                            }
                        )
                else:
                    psnr_loss = image.new_zeros(())
                    psnr_loss_debug.update(
                        {
                            "applied": False,
                            "reason": "unknown_structure_gate_mode",
                            "weight": 0.0,
                            "weighted_loss": 0.0,
                            "structure_gate_mode": structure_gate_mode,
                        }
                    )
        if psnr_loss_debug.get("applied", False):
            loss = loss + psnr_loss
        if self.pose_render_psnr_loss != "off":
            keyframe.info["_paper_aligned_pose_render_psnr_loss"] = psnr_loss_debug
            self._record_pose_render_psnr_loss(psnr_loss_debug)
        update_gate_debug = pose_render_update_gate_decision(
            mode=self.pose_render_update_gate,
            direct_density_mode=self.paper_aligned_direct_density_control,
            keyframe_info=keyframe.info,
            render_frame_policy=self.paper_aligned_render_frame_policy,
            min_confidence=self.pose_render_update_gate_min_confidence,
            max_pose_risk=self.pose_render_update_gate_max_pose_risk,
            max_utility_drift=self.pose_render_update_gate_max_utility_drift,
            soft_min_scale=self.pose_render_update_gate_soft_min_scale,
            psnr_health_min_score=(
                self.pose_render_update_gate_psnr_health_min_score
            ),
            psnr_health_min_raw_loss=(
                self.pose_render_update_gate_psnr_health_min_raw_loss
            ),
            psnr_health_scale=self.pose_render_update_gate_psnr_health_scale,
        )
        if self.pose_render_update_gate != "off":
            keyframe.info["_paper_aligned_pose_render_update_gate"] = update_gate_debug
            self._record_pose_render_update_gate(update_gate_debug)
        loss.backward()

        # ========== 参数更新 ==========
        with torch.no_grad():
            # 【优化模块】更新关键帧位姿（6D表示 + 曝光 + 深度缩放/偏移）
            if bool(update_pose):
                keyframe.step()

            # 测试关键帧不参与场景优化（仅用于评估）
            if not keyframe.info["is_test"]:
                # 【优化模块】更新高斯参数（位置、颜色、不透明度、尺度、旋转）
                # visibility_filter: 可见性掩码（用于自适应密度控制）
                # radii.shape[0]: 高斯点数量（用于学习率调度）
                if bool(update_gate_debug.get("allow_gaussian_update", True)):
                    self._apply_pose_render_gaussian_update_scale(
                        update_gate_debug.get("gaussian_update_scale", 1.0)
                    )
                    self.optimizer.step(
                        render_pkg["visibility_filter"], render_pkg["radii"].shape[0]
                    )

            # 保存最新渲染的逆深度（用于后续的三角化）
            keyframe.latest_invdepth = render_pkg["invdepth"].detach()

        # 标记位姿缓存失效（需要重新计算）
        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id

    def _apply_pose_render_gaussian_update_scale(self, update_scale: object) -> None:
        try:
            scale = float(update_scale)
        except (TypeError, ValueError):
            scale = 1.0
        scale = min(max(scale, 0.0), 1.0)
        if scale >= 0.999:
            return
        for param_dict in self.gaussian_params.values():
            val = param_dict.get("val")
            if getattr(val, "grad", None) is not None:
                val.grad.mul_(scale)

    def _record_pose_render_keyframe_sampling(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_keyframe_sampling_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats:
            stats[reason] = int(stats.get(reason, 0)) + 1
        if bool(debug.get("target_mask_applied", False)):
            stats["target_mask_applied"] = int(stats.get("target_mask_applied", 0)) + 1
            stats["target_valid_ratio_sum"] = float(
                stats.get("target_valid_ratio_sum", 0.0)
            ) + float(debug.get("target_valid_ratio", 1.0) or 0.0)
        if "health_gate_score" in debug:
            stats["health_score_count"] = int(
                stats.get("health_score_count", 0)
            ) + 1
            stats["health_score_sum"] = float(
                stats.get("health_score_sum", 0.0)
            ) + float(debug.get("health_gate_score", 0.0) or 0.0)
        if bool(debug.get("health_gate_raw_wait", False)):
            stats["health_gate_raw_wait"] = int(
                stats.get("health_gate_raw_wait", 0)
            ) + 1
        if bool(debug.get("applied", False)):
            stats["applied"] = int(stats.get("applied", 0)) + 1
        stats["candidate_count_sum"] = int(stats.get("candidate_count_sum", 0)) + int(
            debug.get("candidate_count", 0) or 0
        )
        stats["high_risk_candidates_sum"] = int(
            stats.get("high_risk_candidates_sum", 0)
        ) + int(debug.get("high_risk_candidates", 0) or 0)
        stats["missing_pose_render_gate_sum"] = int(
            stats.get("missing_pose_render_gate_sum", 0)
        ) + int(debug.get("missing_pose_render_gate", 0) or 0)
        stats["chosen_probability_sum"] = float(
            stats.get("chosen_probability_sum", 0.0)
        ) + float(debug.get("chosen_probability", 0.0) or 0.0)
        if "probability_min" in debug:
            stats["probability_min_observed"] = min(
                float(stats.get("probability_min_observed", 1.0)),
                float(debug.get("probability_min", 1.0) or 1.0),
            )
        if "probability_max" in debug:
            stats["probability_max_observed"] = max(
                float(stats.get("probability_max_observed", 0.0)),
                float(debug.get("probability_max", 0.0) or 0.0),
            )
        if "weight_min" in debug:
            stats["weight_min_observed"] = min(
                float(stats.get("weight_min_observed", 1.0)),
                float(debug.get("weight_min", 1.0) or 1.0),
            )
        if "weight_max" in debug:
            stats["weight_max_observed"] = max(
                float(stats.get("weight_max_observed", 0.0)),
                float(debug.get("weight_max", 0.0) or 0.0),
            )
        stats["weight_mean_sum"] = float(stats.get("weight_mean_sum", 0.0)) + float(
            debug.get("weight_mean", 0.0) or 0.0
        )
        stats["confidence_mean_sum"] = float(
            stats.get("confidence_mean_sum", 0.0)
        ) + float(debug.get("confidence_mean", 0.0) or 0.0)

    def _pose_render_keyframe_sampling_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_keyframe_sampling_stats)
        events = int(stats.get("events", 0))
        if events > 0:
            stats["candidate_count_mean"] = float(
                stats.get("candidate_count_sum", 0)
            ) / float(events)
            stats["high_risk_candidates_mean"] = float(
                stats.get("high_risk_candidates_sum", 0)
            ) / float(events)
            stats["missing_pose_render_gate_mean"] = float(
                stats.get("missing_pose_render_gate_sum", 0)
            ) / float(events)
            stats["chosen_probability_mean"] = float(
                stats.get("chosen_probability_sum", 0.0)
            ) / float(events)
            stats["weight_mean"] = float(stats.get("weight_mean_sum", 0.0)) / float(
                events
            )
            stats["confidence_mean"] = float(
                stats.get("confidence_mean_sum", 0.0)
            ) / float(events)
        else:
            stats["candidate_count_mean"] = 0.0
            stats["high_risk_candidates_mean"] = 0.0
            stats["missing_pose_render_gate_mean"] = 0.0
            stats["chosen_probability_mean"] = 0.0
            stats["weight_mean"] = 0.0
            stats["confidence_mean"] = 0.0
        return stats

    def _record_pose_render_texture_sampling(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_texture_sampling_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats:
            stats[reason] = int(stats.get(reason, 0)) + 1
        if "coverage_deficit" in debug:
            stats["coverage_deficit_sum"] = float(
                stats.get("coverage_deficit_sum", 0.0)
            ) + float(debug.get("coverage_deficit", 0.0) or 0.0)
        if bool(debug.get("applied", False)):
            stats["applied"] = int(stats.get("applied", 0)) + 1
            stats["alpha_sum"] = float(stats.get("alpha_sum", 0.0)) + float(
                debug.get("alpha", 0.0) or 0.0
            )
            stats["selectivity_sum"] = float(stats.get("selectivity_sum", 0.0)) + float(
                debug.get("selectivity", 0.0) or 0.0
            )
            budget_before = float(debug.get("budget_before", 0.0) or 0.0)
            budget_after = float(debug.get("budget_after", 0.0) or 0.0)
            stats["budget_shift_abs_sum"] = float(
                stats.get("budget_shift_abs_sum", 0.0)
            ) + abs(budget_after - budget_before)
            if "coverage_deficit" in debug:
                stats["coverage_deficit_applied_sum"] = float(
                    stats.get("coverage_deficit_applied_sum", 0.0)
                ) + float(debug.get("coverage_deficit", 0.0) or 0.0)

    def _pose_render_texture_sampling_scene_guard(self) -> dict[str, object]:
        stats = self.pose_render_texture_sampling_stats
        evaluated = int(stats.get("response_evaluated", 0) or 0)
        bad = int(stats.get("response_bad", 0) or 0)
        events = int(stats.get("events", 0) or 0)
        applied = int(stats.get("applied", 0) or 0)
        min_evaluated = int(stats.get("scene_guard_min_evaluated", 3) or 3)
        max_bad_ratio = float(stats.get("scene_guard_max_bad_ratio", 0.67) or 0.67)
        min_coverage_deficit = float(
            stats.get("scene_guard_min_coverage_deficit", 0.08) or 0.08
        )
        bad_ratio = float(bad) / float(max(evaluated, 1))
        coverage_deficit_mean = float(stats.get("coverage_deficit_sum", 0.0)) / float(
            max(events, 1)
        )
        sampling_applied_ratio = float(applied) / float(max(events, 1))
        coverage_bypass_latched = bool(stats.get("coverage_bypass_latched", False))
        coverage_bypass_reason = str(stats.get("coverage_bypass_reason", "") or "")
        if self.pose_render_texture_sampling == TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE:
            min_bypass_events = int(stats.get("coverage_bypass_min_events", 32) or 32)
            max_mean_deficit = float(
                stats.get("coverage_bypass_max_mean_deficit", 0.015) or 0.015
            )
            max_applied_ratio = float(
                stats.get("coverage_bypass_max_applied_ratio", 0.02) or 0.02
            )
            if (
                not coverage_bypass_latched
                and events >= min_bypass_events
                and coverage_deficit_mean <= max_mean_deficit
                and sampling_applied_ratio <= max_applied_ratio
            ):
                coverage_bypass_latched = True
                coverage_bypass_reason = "high_coverage_low_sampling_pressure"
            stats["coverage_bypass_latched"] = bool(coverage_bypass_latched)
            stats["coverage_bypass_reason"] = coverage_bypass_reason
        disabled = evaluated >= min_evaluated and bad_ratio >= max_bad_ratio
        reason = "response_bad_ratio" if disabled else "active"
        if evaluated < min_evaluated:
            reason = "warming_up"
        stats["scene_guard_disabled_flag"] = bool(disabled)
        stats["scene_guard_reason"] = reason
        return {
            "disabled": bool(disabled),
            "reason": reason,
            "response_evaluated": evaluated,
            "response_bad": bad,
            "response_good": int(stats.get("response_good", 0) or 0),
            "response_bad_ratio": bad_ratio,
            "events": events,
            "applied": applied,
            "coverage_deficit_mean": coverage_deficit_mean,
            "sampling_applied_ratio": sampling_applied_ratio,
            "coverage_bypass_latched": bool(coverage_bypass_latched),
            "coverage_bypass_reason": coverage_bypass_reason,
            "coverage_bypass_min_events": int(
                stats.get("coverage_bypass_min_events", 32) or 32
            ),
            "coverage_bypass_max_mean_deficit": float(
                stats.get("coverage_bypass_max_mean_deficit", 0.015) or 0.015
            ),
            "coverage_bypass_max_applied_ratio": float(
                stats.get("coverage_bypass_max_applied_ratio", 0.02) or 0.02
            ),
            "min_evaluated": min_evaluated,
            "max_bad_ratio": max_bad_ratio,
            "min_coverage_deficit": min_coverage_deficit,
        }

    def _pose_render_response_tracking_bypassed(self) -> bool:
        return (
            (
                self.pose_render_texture_sampling
                == TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE
                and bool(
                    self.pose_render_texture_sampling_stats.get(
                        "coverage_bypass_latched", False
                    )
                )
            )
            or (
                self.pose_render_texture_sampling
                == TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE
                and self._dark_scene_mask_blocked_scene_latched()
            )
        )

    def _pose_render_texture_sampling_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_texture_sampling_stats)
        applied = int(stats.get("applied", 0))
        evaluated = int(stats.get("response_evaluated", 0) or 0)
        bad = int(stats.get("response_bad", 0) or 0)
        stats["response_bad_ratio"] = float(bad) / float(max(evaluated, 1))
        if applied > 0:
            stats["alpha_mean"] = float(stats.get("alpha_sum", 0.0)) / applied
            stats["selectivity_mean"] = float(stats.get("selectivity_sum", 0.0)) / applied
            stats["coverage_deficit_applied_mean"] = (
                float(stats.get("coverage_deficit_applied_sum", 0.0)) / applied
            )
        else:
            stats["alpha_mean"] = 0.0
            stats["selectivity_mean"] = 0.0
            stats["coverage_deficit_applied_mean"] = 0.0
        events = int(stats.get("events", 0))
        stats["coverage_deficit_mean"] = (
            float(stats.get("coverage_deficit_sum", 0.0)) / float(max(events, 1))
        )
        return stats

    def _record_pose_render_edge_loss(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_edge_loss_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats:
            stats[reason] = int(stats.get(reason, 0)) + 1
        if str(debug.get("pose_gate_mode", "")) == "pose_safe_v1":
            if bool(debug.get("pose_gate_passed", False)):
                stats["pose_safe_gate_passed"] = int(
                    stats.get("pose_safe_gate_passed", 0)
                ) + 1
            else:
                stats["pose_safe_gate_rejected"] = int(
                    stats.get("pose_safe_gate_rejected", 0)
                ) + 1
        if bool(debug.get("applied", False)):
            stats["applied"] = int(stats.get("applied", 0)) + 1
            stats["weight_sum"] = float(stats.get("weight_sum", 0.0)) + float(
                debug.get("weight", 0.0) or 0.0
            )
            stats["base_weight_sum"] = float(
                stats.get("base_weight_sum", 0.0)
            ) + float(debug.get("base_weight", debug.get("weight", 0.0)) or 0.0)
            stats["adaptive_weight_scale_sum"] = float(
                stats.get("adaptive_weight_scale_sum", 0.0)
            ) + float(debug.get("adaptive_weight_scale", 1.0) or 0.0)
            stats["residual_clip_scale_sum"] = float(
                stats.get("residual_clip_scale_sum", 0.0)
            ) + float(debug.get("residual_clip_scale", 1.0) or 0.0)
            stats["risk_weight_scale_sum"] = float(
                stats.get("risk_weight_scale_sum", 0.0)
            ) + float(debug.get("risk_weight_scale", 1.0) or 0.0)
            stats["drift_weight_scale_sum"] = float(
                stats.get("drift_weight_scale_sum", 0.0)
            ) + float(debug.get("drift_weight_scale", 1.0) or 0.0)
            stats["support_weight_scale_sum"] = float(
                stats.get("support_weight_scale_sum", 0.0)
            ) + float(debug.get("support_weight_scale", 1.0) or 0.0)
            stats["raw_loss_sum"] = float(stats.get("raw_loss_sum", 0.0)) + float(
                debug.get("raw_loss", 0.0) or 0.0
            )
            stats["weighted_loss_sum"] = float(
                stats.get("weighted_loss_sum", 0.0)
            ) + float(debug.get("weighted_loss", 0.0) or 0.0)

    def _pose_render_edge_loss_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_edge_loss_stats)
        applied = int(stats.get("applied", 0))
        if applied > 0:
            stats["weight_mean"] = float(stats.get("weight_sum", 0.0)) / applied
            stats["base_weight_mean"] = (
                float(stats.get("base_weight_sum", 0.0)) / applied
            )
            stats["adaptive_weight_scale_mean"] = (
                float(stats.get("adaptive_weight_scale_sum", 0.0)) / applied
            )
            stats["residual_clip_scale_mean"] = (
                float(stats.get("residual_clip_scale_sum", 0.0)) / applied
            )
            stats["risk_weight_scale_mean"] = (
                float(stats.get("risk_weight_scale_sum", 0.0)) / applied
            )
            stats["drift_weight_scale_mean"] = (
                float(stats.get("drift_weight_scale_sum", 0.0)) / applied
            )
            stats["support_weight_scale_mean"] = (
                float(stats.get("support_weight_scale_sum", 0.0)) / applied
            )
            stats["raw_loss_mean"] = float(stats.get("raw_loss_sum", 0.0)) / applied
            stats["weighted_loss_mean"] = (
                float(stats.get("weighted_loss_sum", 0.0)) / applied
            )
            stats["support_weight_scale_mean"] = (
                float(stats.get("support_weight_scale_sum", 0.0)) / applied
            )
            stats["support_correspondence_count_mean"] = (
                float(stats.get("support_correspondence_count_sum", 0.0)) / applied
            )
            stats["support_final_pose_inliers_mean"] = (
                float(stats.get("support_final_pose_inliers_sum", 0.0)) / applied
            )
        else:
            stats["weight_mean"] = 0.0
            stats["base_weight_mean"] = 0.0
            stats["adaptive_weight_scale_mean"] = 0.0
            stats["residual_clip_scale_mean"] = 0.0
            stats["risk_weight_scale_mean"] = 0.0
            stats["drift_weight_scale_mean"] = 0.0
            stats["support_weight_scale_mean"] = 0.0
            stats["raw_loss_mean"] = 0.0
            stats["weighted_loss_mean"] = 0.0
            stats["support_weight_scale_mean"] = 0.0
            stats["support_correspondence_count_mean"] = 0.0
            stats["support_final_pose_inliers_mean"] = 0.0
        return stats

    def _pose_render_psnr_scene_low_response_should_block(
        self, debug: dict[str, object]
    ) -> tuple[bool, dict[str, object]]:
        mode = str(getattr(self, "pose_render_psnr_loss_context_weight", "off") or "off")
        guard_debug: dict[str, object] = {
            "scene_low_response_mode": mode,
            "scene_low_response_latched": bool(
                getattr(self, "pose_render_psnr_scene_low_response_latched", False)
            ),
        }
        if mode != "mask_aware_no_mask_scene_low_gate_boost_v1":
            if mode != "mask_aware_no_mask_scene_low_fast_gate_boost_v1":
                return False, guard_debug
        if (
            str(debug.get("training_background_decision", ""))
            != "mask_aware_dark_no_mask_random"
        ):
            guard_debug["reason"] = "scene_low_response_context_mismatch"
            return False, guard_debug

        stats = self.pose_render_psnr_loss_stats
        raw_loss = float(debug.get("raw_loss", 0.0) or 0.0)
        projected_events = int(stats.get("scene_low_response_events", 0) or 0) + 1
        projected_raw_sum = (
            float(stats.get("scene_low_response_raw_loss_sum", 0.0) or 0.0)
            + raw_loss
        )
        projected_raw_mean = projected_raw_sum / float(max(projected_events, 1))
        min_events = int(self.pose_render_psnr_loss_scene_guard_min_applied or 0)
        if mode == "mask_aware_no_mask_scene_low_fast_gate_boost_v1":
            min_events = min(max(min_events, 1), 32)
        raw_low = float(self.pose_render_psnr_loss_scene_guard_raw_low or 0.0)
        guard_debug.update(
            {
                "scene_low_response_candidate": True,
                "scene_low_response_events": projected_events,
                "scene_low_response_raw_loss_sum": projected_raw_sum,
                "scene_low_response_raw_loss_mean": projected_raw_mean,
                "scene_low_response_min_events": min_events,
                "scene_low_response_raw_low": raw_low,
            }
        )
        if bool(getattr(self, "pose_render_psnr_scene_low_response_latched", False)):
            guard_debug["reason"] = "psnr_scene_low_response_gate"
            return True, guard_debug
        if min_events <= 0 or raw_low <= 0.0 or projected_events < min_events:
            guard_debug["reason"] = "scene_low_response_wait"
            return False, guard_debug
        if projected_raw_mean < raw_low:
            self.pose_render_psnr_scene_low_response_latched = True
            stats["scene_low_response_latched"] = True
            stats["scene_low_response_trigger_events"] = projected_events
            stats["scene_low_response_trigger_raw_loss_mean"] = projected_raw_mean
            guard_debug.update(
                {
                    "reason": "psnr_scene_low_response_gate",
                    "scene_low_response_latched": True,
                    "scene_low_response_trigger_events": projected_events,
                    "scene_low_response_trigger_raw_loss_mean": projected_raw_mean,
                }
            )
            return True, guard_debug

        guard_debug["reason"] = "psnr_scene_low_response_open"
        return False, guard_debug

    def _attach_pose_render_scene_low_response_info(self, keyframe: Keyframe) -> None:
        mode = str(getattr(self, "pose_render_psnr_loss_context_weight", "off") or "off")
        if mode not in {
            "mask_aware_no_mask_scene_low_gate_boost_v1",
            "mask_aware_no_mask_scene_low_fast_gate_boost_v1",
        }:
            return
        stats = self.pose_render_psnr_loss_stats
        events = int(stats.get("scene_low_response_events", 0) or 0)
        raw_mean = (
            float(stats.get("scene_low_response_raw_loss_sum", 0.0) or 0.0)
            / float(max(events, 1))
        )
        disabled = bool(
            getattr(self, "pose_render_psnr_scene_low_response_latched", False)
        )
        keyframe.info[SCENE_LOW_RESPONSE_KEY] = {
            "mode": mode,
            "disabled": disabled,
            "latched": disabled,
            "reason": "psnr_scene_low_response_gate" if disabled else "open",
            "events": events,
            "raw_loss_mean": raw_mean,
            "trigger_events": int(
                stats.get("scene_low_response_trigger_events", 0) or 0
            ),
            "trigger_raw_loss_mean": float(
                stats.get("scene_low_response_trigger_raw_loss_mean", 0.0) or 0.0
            ),
        }

    def _pose_render_psnr_scene_guard_should_block(
        self, debug: dict[str, object]
    ) -> tuple[bool, dict[str, object]]:
        mode = str(self.pose_render_psnr_loss_scene_guard or "off")
        guard_debug: dict[str, object] = {
            "scene_guard_mode": mode,
            "scene_guard_triggered": bool(
                self.pose_render_psnr_loss_scene_guard_triggered
            ),
        }
        render_gap_precommit_mode = (
            mode == "raw_loss_ratio_precommit_render_gap_guard_v1"
        )
        coverage_precommit_mode = (
            mode == "raw_loss_ratio_precommit_coverage_guard_v1"
        )
        mask_aware_non_dark_precommit_mode = (
            mode == "raw_loss_ratio_precommit_non_dark_mask_guard_v3"
        )
        mask_aware_dark_background_mode = mode == "mask_aware_dark_background_guard_v4"
        non_dark_precommit_mode = (
            mode == "raw_loss_ratio_precommit_non_dark_guard_v2"
            or mask_aware_non_dark_precommit_mode
            or mask_aware_dark_background_mode
        )
        precommit_mode = (
            mode == "raw_loss_ratio_precommit_guard_v1"
            or render_gap_precommit_mode
            or coverage_precommit_mode
            or non_dark_precommit_mode
        )
        if mode not in {
            "raw_loss_ratio_guard_v1",
            "raw_loss_ratio_precommit_guard_v1",
            "raw_loss_ratio_precommit_render_gap_guard_v1",
            "raw_loss_ratio_precommit_coverage_guard_v1",
            "raw_loss_ratio_precommit_non_dark_guard_v2",
            "raw_loss_ratio_precommit_non_dark_mask_guard_v3",
            "mask_aware_dark_background_guard_v4",
        }:
            return False, guard_debug
        if (
            mask_aware_dark_background_mode
            and str(debug.get("training_background_decision", ""))
            == "mask_aware_dark_fixed_black"
        ):
            guard_debug.update(
                {
                    "reason": "psnr_mask_aware_dark_background_blocked",
                    "scene_guard_mask_aware_dark_background": True,
                    "training_background_decision": debug.get(
                        "training_background_decision"
                    ),
                    "training_background_target_mean": debug.get(
                        "training_background_target_mean",
                        0.0,
                    ),
                }
            )
            return True, guard_debug
        if non_dark_precommit_mode and self._dark_fixed_black_scene_latched():
            guard_debug.update(
                {
                    "reason": "psnr_scene_guard_dark_fixed_bypass",
                    "scene_guard_non_dark_only": True,
                }
            )
            return False, guard_debug
        if (
            mask_aware_non_dark_precommit_mode
            and self._dark_scene_mask_blocked_scene_latched()
        ):
            guard_debug.update(
                {
                    "reason": "psnr_scene_guard_mask_blocked_bypass",
                    "scene_guard_mask_aware": True,
                    "scene_guard_non_dark_only": True,
                }
            )
            return False, guard_debug
        if self.pose_render_psnr_loss_scene_guard_triggered:
            guard_debug["reason"] = "psnr_scene_guard_blocked"
            return True, guard_debug
        if not bool(debug.get("applied", False)):
            return False, guard_debug

        stats = self.pose_render_psnr_loss_stats
        projected_events = int(stats.get("events", 0)) + 1
        raw_loss = float(debug.get("raw_loss", 0.0) or 0.0)
        if precommit_mode:
            projected_applied = int(stats.get("scene_guard_candidate_count", 0)) + 1
            projected_raw_sum = (
                float(stats.get("scene_guard_candidate_raw_loss_sum", 0.0))
                + raw_loss
            )
        else:
            projected_applied = int(stats.get("applied", 0)) + 1
            projected_raw_sum = float(stats.get("raw_loss_sum", 0.0)) + raw_loss
        projected_ratio = float(projected_applied) / float(max(projected_events, 1))
        projected_raw_mean = projected_raw_sum / float(max(projected_applied, 1))
        guard_debug.update(
            {
                "scene_guard_precommit": precommit_mode,
                "scene_guard_candidate": True,
                "scene_guard_candidate_count": projected_applied,
                "scene_guard_candidate_raw_loss_sum": projected_raw_sum,
                "scene_guard_candidate_raw_loss_mean": projected_raw_mean,
                "scene_guard_projected_events": projected_events,
                "scene_guard_projected_applied": projected_applied,
                "scene_guard_projected_ratio": projected_ratio,
                "scene_guard_projected_raw_loss_mean": projected_raw_mean,
                "scene_guard_raw_low": self.pose_render_psnr_loss_scene_guard_raw_low,
                "scene_guard_raw_high": self.pose_render_psnr_loss_scene_guard_raw_high,
                "scene_guard_min_ratio": self.pose_render_psnr_loss_scene_guard_min_ratio,
                "scene_guard_min_events": self.pose_render_psnr_loss_scene_guard_min_events,
                "scene_guard_min_applied": (
                    self.pose_render_psnr_loss_scene_guard_min_applied
                ),
                "scene_guard_max_events": (
                    self.pose_render_psnr_loss_scene_guard_max_events
                ),
            }
        )
        max_events = int(self.pose_render_psnr_loss_scene_guard_max_events or 0)
        if max_events > 0 and projected_events > max_events:
            stats["psnr_scene_guard_late_window_expired"] = int(
                stats.get("psnr_scene_guard_late_window_expired", 0)
            ) + 1
            guard_debug["reason"] = "psnr_scene_guard_late_window_expired"
            return False, guard_debug
        if (
            precommit_mode
            and projected_events < self.pose_render_psnr_loss_scene_guard_min_events
        ):
            guard_debug["reason"] = "psnr_scene_guard_precommit_wait"
            return True, guard_debug
        raw_band_enabled = (
            self.pose_render_psnr_loss_scene_guard_raw_high
            > self.pose_render_psnr_loss_scene_guard_raw_low
        )
        texture_stats = getattr(self, "pose_render_texture_sampling_stats", {}) or {}
        texture_events = int(texture_stats.get("events", 0) or 0)
        texture_applied = int(texture_stats.get("applied", 0) or 0)
        coverage_deficit_mean = float(
            texture_stats.get("coverage_deficit_sum", 0.0) or 0.0
        ) / float(max(texture_events, 1))
        sampling_applied_ratio = float(texture_applied) / float(max(texture_events, 1))
        render_gap_min_coverage_deficit = 0.075
        coverage_pressure_min_deficit = 0.065
        coverage_pressure_min_sampling_ratio = 0.15
        coverage_pressure_sufficient = (
            coverage_deficit_mean >= coverage_pressure_min_deficit
            or sampling_applied_ratio >= coverage_pressure_min_sampling_ratio
        )
        render_gap_trigger = (
            render_gap_precommit_mode
            and raw_band_enabled
            and projected_events >= self.pose_render_psnr_loss_scene_guard_min_events
            and projected_applied >= self.pose_render_psnr_loss_scene_guard_min_applied
            and projected_ratio >= self.pose_render_psnr_loss_scene_guard_min_ratio
            and projected_raw_mean > self.pose_render_psnr_loss_scene_guard_raw_high
            and coverage_deficit_mean >= render_gap_min_coverage_deficit
        )
        guard_debug.update(
            {
                "scene_guard_render_gap_triggered": bool(render_gap_trigger),
                "scene_guard_coverage_aware": bool(coverage_precommit_mode),
                "scene_guard_coverage_pressure_sufficient": bool(
                    coverage_pressure_sufficient
                ),
                "scene_guard_coverage_deficit_mean": coverage_deficit_mean,
                "scene_guard_sampling_applied_ratio": sampling_applied_ratio,
                "scene_guard_render_gap_min_coverage_deficit": (
                    render_gap_min_coverage_deficit
                ),
                "scene_guard_coverage_pressure_min_deficit": (
                    coverage_pressure_min_deficit
                ),
                "scene_guard_coverage_pressure_min_sampling_ratio": (
                    coverage_pressure_min_sampling_ratio
                ),
            }
        )
        raw_band_trigger = (
            raw_band_enabled
            and projected_events >= self.pose_render_psnr_loss_scene_guard_min_events
            and projected_applied >= self.pose_render_psnr_loss_scene_guard_min_applied
            and projected_ratio >= self.pose_render_psnr_loss_scene_guard_min_ratio
            and self.pose_render_psnr_loss_scene_guard_raw_low
            <= projected_raw_mean
            <= self.pose_render_psnr_loss_scene_guard_raw_high
        )
        if coverage_precommit_mode:
            raw_band_trigger = raw_band_trigger and coverage_pressure_sufficient
        should_trigger = raw_band_trigger or render_gap_trigger
        if not should_trigger:
            return False, guard_debug

        self.pose_render_psnr_loss_scene_guard_triggered = True
        trigger_reason = (
            "render_gap_high_residual" if render_gap_trigger else "raw_loss_band"
        )
        stats["scene_guard_triggered"] = True
        stats["scene_guard_trigger_reason"] = trigger_reason
        stats["scene_guard_trigger_events"] = projected_events
        stats["scene_guard_trigger_applied"] = projected_applied
        stats["scene_guard_trigger_ratio"] = projected_ratio
        stats["scene_guard_trigger_raw_loss_mean"] = projected_raw_mean
        stats["scene_guard_trigger_coverage_deficit_mean"] = coverage_deficit_mean
        guard_debug.update(
            {
                "reason": "psnr_scene_guard_triggered",
                "scene_guard_triggered": True,
                "scene_guard_trigger_reason": trigger_reason,
                "scene_guard_trigger_events": projected_events,
                "scene_guard_trigger_applied": projected_applied,
                "scene_guard_trigger_ratio": projected_ratio,
                "scene_guard_trigger_raw_loss_mean": projected_raw_mean,
                "scene_guard_trigger_coverage_deficit_mean": coverage_deficit_mean,
            }
        )
        return True, guard_debug

    def _record_pose_render_psnr_loss(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_psnr_loss_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats:
            stats[reason] = int(stats.get(reason, 0)) + 1
        if "structure_score" in debug:
            stats["structure_score_sum"] = float(
                stats.get("structure_score_sum", 0.0)
            ) + float(debug.get("structure_score", 0.0) or 0.0)
            stats["structure_score_count"] = int(
                stats.get("structure_score_count", 0)
            ) + 1
        if bool(debug.get("scene_guard_candidate", False)):
            stats["scene_guard_candidate_count"] = int(
                stats.get("scene_guard_candidate_count", 0)
            ) + 1
            stats["scene_guard_candidate_raw_loss_sum"] = float(
                stats.get("scene_guard_candidate_raw_loss_sum", 0.0)
            ) + float(debug.get("raw_loss", 0.0) or 0.0)
        if bool(debug.get("scene_low_response_candidate", False)):
            stats["scene_low_response_events"] = int(
                stats.get("scene_low_response_events", 0)
            ) + 1
            stats["scene_low_response_raw_loss_sum"] = float(
                stats.get("scene_low_response_raw_loss_sum", 0.0)
            ) + float(debug.get("raw_loss", 0.0) or 0.0)
        support_reason = str(debug.get("support_weight_reason", ""))
        if support_reason in stats:
            stats[support_reason] = int(stats.get(support_reason, 0)) + 1
        context_reason = str(debug.get("context_weight_reason", ""))
        if context_reason in stats:
            stats[context_reason] = int(stats.get(context_reason, 0)) + 1
        if str(debug.get("pose_gate_mode", "")) == "pose_safe_v1":
            if bool(debug.get("pose_gate_passed", False)):
                stats["pose_safe_gate_passed"] = int(
                    stats.get("pose_safe_gate_passed", 0)
                ) + 1
            else:
                stats["pose_safe_gate_rejected"] = int(
                    stats.get("pose_safe_gate_rejected", 0)
                ) + 1
        if bool(debug.get("target_mask_applied", False)):
            stats["target_mask_applied"] = int(stats.get("target_mask_applied", 0)) + 1
            stats["target_valid_ratio_sum"] = float(
                stats.get("target_valid_ratio_sum", 0.0)
            ) + float(debug.get("target_valid_ratio", 1.0) or 0.0)
        if "health_gate_score" in debug:
            stats["health_score_count"] = int(
                stats.get("health_score_count", 0)
            ) + 1
            stats["health_score_sum"] = float(
                stats.get("health_score_sum", 0.0)
            ) + float(debug.get("health_gate_score", 0.0) or 0.0)
        if bool(debug.get("health_gate_raw_wait", False)):
            stats["health_gate_raw_wait"] = int(
                stats.get("health_gate_raw_wait", 0)
            ) + 1
        if bool(debug.get("applied", False)):
            stats["applied"] = int(stats.get("applied", 0)) + 1
            stats["weight_sum"] = float(stats.get("weight_sum", 0.0)) + float(
                debug.get("weight", 0.0) or 0.0
            )
            stats["base_weight_sum"] = float(
                stats.get("base_weight_sum", 0.0)
            ) + float(debug.get("base_weight", 0.0) or 0.0)
            stats["support_weight_scale_sum"] = float(
                stats.get("support_weight_scale_sum", 0.0)
            ) + float(debug.get("support_weight_scale", 1.0) or 0.0)
            stats["context_weight_scale_sum"] = float(
                stats.get("context_weight_scale_sum", 0.0)
            ) + float(debug.get("context_weight_scale", 1.0) or 0.0)
            if float(debug.get("context_weight_scale", 1.0) or 1.0) > 1.0:
                stats["context_weight_boosted"] = int(
                    stats.get("context_weight_boosted", 0)
                ) + 1
            corr_count = float(debug.get("support_correspondence_count", -1) or -1)
            if corr_count >= 0:
                stats["support_correspondence_count_sum"] = float(
                    stats.get("support_correspondence_count_sum", 0.0)
                ) + corr_count
            inlier_count = float(debug.get("support_final_pose_inliers", -1) or -1)
            if inlier_count >= 0:
                stats["support_final_pose_inliers_sum"] = float(
                    stats.get("support_final_pose_inliers_sum", 0.0)
                ) + inlier_count
            stats["raw_loss_sum"] = float(stats.get("raw_loss_sum", 0.0)) + float(
                debug.get("raw_loss", 0.0) or 0.0
            )
            stats["weighted_loss_sum"] = float(
                stats.get("weighted_loss_sum", 0.0)
            ) + float(debug.get("weighted_loss", 0.0) or 0.0)

    def _pose_render_psnr_loss_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_psnr_loss_stats)
        applied = int(stats.get("applied", 0))
        events = int(stats.get("events", 0))
        stats["applied_ratio"] = float(applied) / float(max(events, 1))
        target_mask_applied = int(stats.get("target_mask_applied", 0))
        stats["target_mask_applied_ratio"] = float(target_mask_applied) / float(max(events, 1))
        stats["target_valid_ratio_mean"] = (
            float(stats.get("target_valid_ratio_sum", 0.0)) / target_mask_applied
            if target_mask_applied > 0
            else 1.0
        )
        structure_count = int(stats.get("structure_score_count", 0))
        stats["structure_score_mean"] = (
            float(stats.get("structure_score_sum", 0.0)) / structure_count
            if structure_count > 0
            else 0.0
        )
        health_count = int(stats.get("health_score_count", 0))
        stats["health_score_mean"] = (
            float(stats.get("health_score_sum", 0.0)) / health_count
            if health_count > 0
            else 0.0
        )
        scene_low_events = int(stats.get("scene_low_response_events", 0))
        stats["scene_low_response_raw_loss_mean"] = (
            float(stats.get("scene_low_response_raw_loss_sum", 0.0))
            / scene_low_events
            if scene_low_events > 0
            else 0.0
        )
        if applied > 0:
            stats["weight_mean"] = float(stats.get("weight_sum", 0.0)) / applied
            stats["base_weight_mean"] = (
                float(stats.get("base_weight_sum", 0.0)) / applied
            )
            stats["raw_loss_mean"] = float(stats.get("raw_loss_sum", 0.0)) / applied
            stats["weighted_loss_mean"] = (
                float(stats.get("weighted_loss_sum", 0.0)) / applied
            )
            stats["support_weight_scale_mean"] = (
                float(stats.get("support_weight_scale_sum", 0.0)) / applied
            )
            stats["context_weight_scale_mean"] = (
                float(stats.get("context_weight_scale_sum", 0.0)) / applied
            )
            stats["support_correspondence_count_mean"] = (
                float(stats.get("support_correspondence_count_sum", 0.0)) / applied
            )
            stats["support_final_pose_inliers_mean"] = (
                float(stats.get("support_final_pose_inliers_sum", 0.0)) / applied
            )
        else:
            stats["weight_mean"] = 0.0
            stats["base_weight_mean"] = 0.0
            stats["raw_loss_mean"] = 0.0
            stats["weighted_loss_mean"] = 0.0
            stats["support_weight_scale_mean"] = 0.0
            stats["context_weight_scale_mean"] = 0.0
            stats["support_correspondence_count_mean"] = 0.0
            stats["support_final_pose_inliers_mean"] = 0.0
        return stats

    def _record_pose_render_extra_optimization(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_extra_optimization_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats and reason not in {
            "transaction_committed",
            "transaction_rolled_back",
        }:
            stats[reason] = int(stats.get(reason, 0)) + 1
        if bool(debug.get("applied", False)):
            stats["applied"] = int(stats.get("applied", 0)) + 1
            stats["extra_iterations_sum"] = int(
                stats.get("extra_iterations_sum", 0)
            ) + int(debug.get("extra_iterations", 0) or 0)
            stats["confidence_sum"] = float(
                stats.get("confidence_sum", 0.0)
            ) + float(debug.get("confidence", 0.0) or 0.0)
        if bool(debug.get("transaction_attempted", False)):
            stats["transaction_attempted"] = int(
                stats.get("transaction_attempted", 0)
            ) + 1
        if bool(debug.get("committed", False)):
            stats["transaction_committed"] = int(
                stats.get("transaction_committed", 0)
            ) + 1
            stats["transaction_best_iteration_sum"] = int(
                stats.get("transaction_best_iteration_sum", 0)
            ) + int(debug.get("best_iteration", 0) or 0)
        if bool(debug.get("rolled_back", False)):
            stats["transaction_rolled_back"] = int(
                stats.get("transaction_rolled_back", 0)
            ) + 1
        if bool(debug.get("early_stopped", False)):
            stats["transaction_early_stopped"] = int(
                stats.get("transaction_early_stopped", 0)
            ) + 1
        if bool(debug.get("time_budget_exhausted", False)):
            stats["transaction_budget_exhausted"] = int(
                stats.get("transaction_budget_exhausted", 0)
            ) + 1
        stats["transaction_reference_guard_rejected"] = int(
            stats.get("transaction_reference_guard_rejected", 0)
        ) + int(debug.get("reference_guard_rejections", 0) or 0)
        stats["transaction_runtime_seconds"] = float(
            stats.get("transaction_runtime_seconds", 0.0)
        ) + float(debug.get("refinement_runtime_seconds", 0.0) or 0.0)

    def _pose_render_extra_optimization_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_extra_optimization_stats)
        applied = int(stats.get("applied", 0))
        events = int(stats.get("events", 0))
        stats["applied_ratio"] = float(applied) / float(max(events, 1))
        if applied > 0:
            stats["extra_iterations_mean"] = (
                float(stats.get("extra_iterations_sum", 0)) / applied
            )
            stats["confidence_mean"] = (
                float(stats.get("confidence_sum", 0.0)) / applied
            )
        else:
            stats["extra_iterations_mean"] = 0.0
            stats["confidence_mean"] = 0.0
        committed = int(stats.get("transaction_committed", 0))
        stats["transaction_best_iteration_mean"] = (
            float(stats.get("transaction_best_iteration_sum", 0))
            / float(max(committed, 1))
        )
        base_runtime = float(stats.get("base_runtime_seconds", 0.0))
        stats["refinement_to_base_time_ratio"] = (
            float(stats.get("transaction_runtime_seconds", 0.0))
            / float(max(base_runtime, 1e-8))
        )
        return stats

    def _record_pose_render_response(
        self,
        keyframe: Keyframe,
        rgb_mse: torch.Tensor,
        *,
        representation_gap: dict[str, object] | None = None,
    ) -> None:
        try:
            latest = float(rgb_mse.detach().cpu().item())
        except Exception:
            return
        gap = representation_gap if isinstance(representation_gap, dict) else {}
        response = keyframe.info.get("_paper_aligned_pose_render_response", None)
        if not isinstance(response, dict):
            response = {}
        keyframe.info["_paper_aligned_pose_render_response"] = updated_render_response(
            response,
            latest,
            representation_coverage_deficit=float(
                gap.get("coverage_deficit", 0.0) or 0.0
            ),
            representation_gap_selectivity=float(
                gap.get("selectivity", 0.0) or 0.0
            ),
        )
        self._update_pose_render_texture_sampling_response_guard(keyframe)

    def _update_pose_render_texture_sampling_response_guard(self, keyframe: Keyframe) -> None:
        if self.pose_render_texture_sampling not in {
            TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,
            TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE,
            TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
            TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
        }:
            return
        sampling = keyframe.info.get("_paper_aligned_pose_render_texture_sampling", None)
        if not isinstance(sampling, dict) or not bool(sampling.get("applied", False)):
            return
        if bool(
            keyframe.info.get(
                "_paper_aligned_pose_render_texture_sampling_response_evaluated",
                False,
            )
        ):
            return
        response = keyframe.info.get("_paper_aligned_pose_render_response", None)
        if not isinstance(response, dict):
            return
        observations = int(response.get("observations", 0) or 0)
        if observations < 4:
            self.pose_render_texture_sampling_stats["response_pending"] = int(
                self.pose_render_texture_sampling_stats.get("response_pending", 0)
            ) + 1
            return

        first = float(response.get("first_rgb_mse", 0.0) or 0.0)
        latest = float(response.get("latest_rgb_mse", 0.0) or 0.0)
        improvement = float(response.get("relative_improvement", 0.0) or 0.0)
        degraded = bool(first > 0.0 and latest > first * 1.02)
        low = bool(improvement < 0.01)
        stats = self.pose_render_texture_sampling_stats
        stats["response_evaluated"] = int(stats.get("response_evaluated", 0)) + 1
        if degraded or low:
            stats["response_bad"] = int(stats.get("response_bad", 0)) + 1
            if degraded:
                stats["response_degraded"] = int(stats.get("response_degraded", 0)) + 1
            if low:
                stats["response_low"] = int(stats.get("response_low", 0)) + 1
            verdict = "bad"
        else:
            stats["response_good"] = int(stats.get("response_good", 0)) + 1
            verdict = "good"
        keyframe.info["_paper_aligned_pose_render_texture_sampling_response_evaluated"] = {
            "verdict": verdict,
            "observations": observations,
            "relative_improvement": improvement,
            "first_rgb_mse": first,
            "latest_rgb_mse": latest,
        }

    def _record_pose_render_update_gate(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_update_gate_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats:
            stats[reason] = int(stats.get(reason, 0)) + 1
        if bool(debug.get("allow_gaussian_update", True)):
            stats["allowed"] = int(stats.get("allowed", 0)) + 1
        else:
            stats["blocked"] = int(stats.get("blocked", 0)) + 1
        stats["confidence_sum"] = float(stats.get("confidence_sum", 0.0)) + float(
            debug.get("confidence", 0.0) or 0.0
        )
        update_scale = float(debug.get("gaussian_update_scale", 1.0) or 1.0)
        stats["gaussian_update_scale_sum"] = float(
            stats.get("gaussian_update_scale_sum", 0.0)
        ) + update_scale
        if update_scale < 0.999:
            stats["gaussian_update_scaled"] = int(
                stats.get("gaussian_update_scaled", 0)
            ) + 1
        stats["psnr_health_score_sum"] = float(
            stats.get("psnr_health_score_sum", 0.0)
        ) + float(debug.get("psnr_health_score", 0.0) or 0.0)
        stats["psnr_health_raw_loss_sum"] = float(
            stats.get("psnr_health_raw_loss_sum", 0.0)
        ) + float(debug.get("psnr_health_raw_loss", 0.0) or 0.0)
        if bool(debug.get("psnr_health_alias_scaled", False)):
            stats["psnr_health_alias_scale_sum"] = float(
                stats.get("psnr_health_alias_scale_sum", 0.0)
            ) + update_scale
        stats["pose_risk_sum"] = float(stats.get("pose_risk_sum", 0.0)) + float(
            debug.get("pose_risk_score", 0.0) or 0.0
        )
        stats["pose_risk_max"] = max(
            float(stats.get("pose_risk_max", 0.0)),
            float(debug.get("pose_risk_score", 0.0) or 0.0),
        )
        stats["confidence_min"] = min(
            float(stats.get("confidence_min", 1.0)),
            float(debug.get("confidence", 1.0) or 0.0),
        )
        stats["utility_drift_sum"] = float(
            stats.get("utility_drift_sum", 0.0)
        ) + float(debug.get("utility_drift_risk", 0.0) or 0.0)

    def _pose_render_update_gate_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_update_gate_stats)
        events = int(stats.get("events", 0))
        if events > 0:
            stats["confidence_mean"] = float(stats.get("confidence_sum", 0.0)) / events
            stats["pose_risk_mean"] = float(stats.get("pose_risk_sum", 0.0)) / events
            stats["utility_drift_mean"] = (
                float(stats.get("utility_drift_sum", 0.0)) / events
            )
            stats["gaussian_update_scale_mean"] = (
                float(stats.get("gaussian_update_scale_sum", 0.0)) / events
            )
            stats["blocked_ratio"] = float(stats.get("blocked", 0)) / events
            stats["scaled_ratio"] = float(stats.get("gaussian_update_scaled", 0)) / events
            stats["psnr_health_score_mean"] = (
                float(stats.get("psnr_health_score_sum", 0.0)) / events
            )
            stats["psnr_health_raw_loss_mean"] = (
                float(stats.get("psnr_health_raw_loss_sum", 0.0)) / events
            )
            alias_scaled = int(stats.get("psnr_health_alias_scaled", 0))
            stats["psnr_health_alias_scaled_ratio"] = float(alias_scaled) / events
            stats["psnr_health_alias_scale_mean"] = (
                float(stats.get("psnr_health_alias_scale_sum", 0.0)) / alias_scaled
                if alias_scaled > 0
                else 1.0
            )
        else:
            stats["confidence_mean"] = 0.0
            stats["pose_risk_mean"] = 0.0
            stats["utility_drift_mean"] = 0.0
            stats["gaussian_update_scale_mean"] = 1.0
            stats["blocked_ratio"] = 0.0
            stats["scaled_ratio"] = 0.0
            stats["psnr_health_score_mean"] = 0.0
            stats["psnr_health_raw_loss_mean"] = 0.0
            stats["psnr_health_alias_scaled_ratio"] = 0.0
            stats["psnr_health_alias_scale_mean"] = 1.0
        return stats

    def _record_pose_render_pre_refine(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_pre_refine_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats:
            stats[reason] = int(stats.get(reason, 0)) + 1
        stats["pose_risk_sum"] = float(stats.get("pose_risk_sum", 0.0)) + float(
            debug.get("pose_risk_score", 0.0) or 0.0
        )
        stats["render_coverage_sum"] = float(
            stats.get("render_coverage_sum", 0.0)
        ) + float(debug.get("render_coverage", 0.0) or 0.0)
        if bool(debug.get("applied", False)):
            stats["applied"] = int(stats.get("applied", 0)) + 1
            stats["loss_sum"] = float(stats.get("loss_sum", 0.0)) + float(
                debug.get("loss_mean", 0.0) or 0.0
            )
            stats["rotation_delta_deg_sum"] = float(
                stats.get("rotation_delta_deg_sum", 0.0)
            ) + float(debug.get("rotation_delta_deg", 0.0) or 0.0)
            stats["translation_delta_sum"] = float(
                stats.get("translation_delta_sum", 0.0)
            ) + float(debug.get("translation_delta", 0.0) or 0.0)

    def _pose_render_pre_refine_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_pre_refine_stats)
        events = int(stats.get("events", 0))
        applied = int(stats.get("applied", 0))
        if events > 0:
            stats["applied_ratio"] = applied / events
            stats["pose_risk_mean"] = float(stats.get("pose_risk_sum", 0.0)) / events
            stats["render_coverage_mean"] = (
                float(stats.get("render_coverage_sum", 0.0)) / events
            )
        else:
            stats["applied_ratio"] = 0.0
            stats["pose_risk_mean"] = 0.0
            stats["render_coverage_mean"] = 0.0
        if applied > 0:
            stats["loss_mean"] = float(stats.get("loss_sum", 0.0)) / applied
            stats["rotation_delta_deg_mean"] = (
                float(stats.get("rotation_delta_deg_sum", 0.0)) / applied
            )
            stats["translation_delta_mean"] = (
                float(stats.get("translation_delta_sum", 0.0)) / applied
            )
        else:
            stats["loss_mean"] = 0.0
            stats["rotation_delta_deg_mean"] = 0.0
            stats["translation_delta_mean"] = 0.0
        return stats

    def _record_pose_render_init_weighting(self, debug: dict[str, object]) -> None:
        stats = self.pose_render_init_weighting_stats
        stats["events"] = int(stats.get("events", 0)) + 1
        reason = str(debug.get("reason", ""))
        if reason in stats:
            stats[reason] = int(stats.get(reason, 0)) + 1
        if bool(debug.get("applied", False)):
            stats["applied"] = int(stats.get("applied", 0)) + 1
        if bool(debug.get("adaptive_risk_override", False)):
            stats["adaptive_risk_override"] = int(
                stats.get("adaptive_risk_override", 0)
            ) + 1
        if bool(debug.get("adaptive_scene_gentle", False)):
            stats["adaptive_scene_gentle"] = int(
                stats.get("adaptive_scene_gentle", 0)
            ) + 1
        if bool(debug.get("adaptive_scene_stable_low_risk_bypass", False)):
            if not bool(stats.get("adaptive_scene_stable_low_risk_latched", 0)):
                stats["adaptive_scene_stable_low_risk_latch_events"] = int(
                    stats.get("adaptive_scene_stable_low_risk_latch_events", 0)
                ) + 1
            stats["adaptive_scene_stable_low_risk_latched"] = 1
        for bool_key in (
            "adaptive_scene_stable_low_risk_bypass",
            "adaptive_scene_mature_low_risk_bypass",
        ):
            if bool(debug.get(bool_key, False)):
                stats[bool_key] = int(stats.get(bool_key, 0)) + 1
        stats["pose_risk_sum"] = float(stats.get("pose_risk_sum", 0.0)) + float(
            debug.get("pose_risk_score", 0.0) or 0.0
        )
        stats["risk_alpha_sum"] = float(stats.get("risk_alpha_sum", 0.0)) + float(
            debug.get("risk_alpha", 0.0) or 0.0
        )
        stats["representation_value_sum"] = float(
            stats.get("representation_value_sum", 0.0)
        ) + float(debug.get("representation_value_score", 0.0) or 0.0)
        stats["novelty_value_sum"] = float(
            stats.get("novelty_value_sum", 0.0)
        ) + float(debug.get("novelty_value_score", 0.0) or 0.0)
        stats["effective_min_pose_risk_sum"] = float(
            stats.get("effective_min_pose_risk_sum", 0.0)
        ) + float(debug.get("min_pose_risk", 0.0) or 0.0)
        stats["effective_max_pose_risk_sum"] = float(
            stats.get("effective_max_pose_risk_sum", 0.0)
        ) + float(debug.get("max_pose_risk", 0.0) or 0.0)
        stats["effective_min_sample_opacity_scale_sum"] = float(
            stats.get("effective_min_sample_opacity_scale_sum", 0.0)
        ) + float(debug.get("min_sample_opacity_scale", 1.0) or 1.0)
        stats["effective_min_match_opacity_scale_sum"] = float(
            stats.get("effective_min_match_opacity_scale_sum", 0.0)
        ) + float(debug.get("min_match_opacity_scale", 1.0) or 1.0)
        stats["sample_opacity_scale_sum"] = float(
            stats.get("sample_opacity_scale_sum", 0.0)
        ) + float(debug.get("sample_opacity_scale", 1.0) or 1.0)
        stats["match_opacity_scale_sum"] = float(
            stats.get("match_opacity_scale_sum", 0.0)
        ) + float(debug.get("match_opacity_scale", 1.0) or 1.0)
        early_trace = stats.get("early_trace", [])
        if isinstance(early_trace, list) and len(early_trace) < 96:
            early_trace.append(
                {
                    "event": int(stats.get("events", 0)),
                    "reason": reason,
                    "applied": bool(debug.get("applied", False)),
                    "pose_risk_score": float(
                        debug.get("pose_risk_score", 0.0) or 0.0
                    ),
                    "risk_alpha": float(debug.get("risk_alpha", 0.0) or 0.0),
                    "novelty_value_score": float(
                        debug.get("novelty_value_score", 0.0) or 0.0
                    ),
                    "adaptive_scene_pose_risk_mean": float(
                        debug.get("adaptive_scene_pose_risk_mean", 0.0) or 0.0
                    ),
                    "adaptive_scene_novelty_mean": float(
                        debug.get("adaptive_scene_novelty_mean", 0.0) or 0.0
                    ),
                    "adaptive_scene_low_risk_bypass": bool(
                        debug.get("adaptive_scene_low_risk_bypass", False)
                    ),
                    "adaptive_scene_stable_low_risk_bypass": bool(
                        debug.get("adaptive_scene_stable_low_risk_bypass", False)
                    ),
                    "adaptive_scene_mature_low_risk_bypass": bool(
                        debug.get("adaptive_scene_mature_low_risk_bypass", False)
                    ),
                    "adaptive_scene_stable_low_risk_latched": bool(
                        debug.get("adaptive_scene_stable_low_risk_latched", False)
                    ),
                }
            )

    def _pose_render_init_weighting_summary(self) -> dict[str, object]:
        stats = dict(self.pose_render_init_weighting_stats)
        events = int(stats.get("events", 0))
        if events > 0:
            stats["applied_ratio"] = float(stats.get("applied", 0)) / events
            stats["pose_risk_mean"] = float(stats.get("pose_risk_sum", 0.0)) / events
            stats["risk_alpha_mean"] = float(stats.get("risk_alpha_sum", 0.0)) / events
            stats["representation_value_mean"] = (
                float(stats.get("representation_value_sum", 0.0)) / events
            )
            stats["novelty_value_mean"] = (
                float(stats.get("novelty_value_sum", 0.0)) / events
            )
            stats["effective_min_pose_risk_mean"] = (
                float(stats.get("effective_min_pose_risk_sum", 0.0)) / events
            )
            stats["effective_max_pose_risk_mean"] = (
                float(stats.get("effective_max_pose_risk_sum", 0.0)) / events
            )
            stats["effective_min_sample_opacity_scale_mean"] = (
                float(stats.get("effective_min_sample_opacity_scale_sum", 0.0))
                / events
            )
            stats["effective_min_match_opacity_scale_mean"] = (
                float(stats.get("effective_min_match_opacity_scale_sum", 0.0))
                / events
            )
            stats["sample_opacity_scale_mean"] = (
                float(stats.get("sample_opacity_scale_sum", 0.0)) / events
            )
            stats["match_opacity_scale_mean"] = (
                float(stats.get("match_opacity_scale_sum", 0.0)) / events
            )
        else:
            stats["applied_ratio"] = 0.0
            stats["pose_risk_mean"] = 0.0
            stats["risk_alpha_mean"] = 0.0
            stats["representation_value_mean"] = 0.0
            stats["novelty_value_mean"] = 0.0
            stats["effective_min_pose_risk_mean"] = 0.0
            stats["effective_max_pose_risk_mean"] = 0.0
            stats["effective_min_sample_opacity_scale_mean"] = 1.0
            stats["effective_min_match_opacity_scale_mean"] = 1.0
            stats["sample_opacity_scale_mean"] = 1.0
            stats["match_opacity_scale_mean"] = 1.0
        return stats

    def pose_render_pre_refine_keyframe(self, keyframe_id: int = -1) -> dict[str, object]:
        keyframe = self.keyframes[keyframe_id]
        existing_gaussians = int(self.xyz.shape[0]) if hasattr(self, "xyz") else 0
        debug = pose_render_pre_refine_decision(
            mode=self.pose_render_pre_refine,
            direct_density_mode=self.paper_aligned_direct_density_control,
            keyframe_info=keyframe.info,
            existing_gaussians=existing_gaussians,
            existing_keyframes=len(self.keyframes),
            iterations=self.pose_render_pre_refine_iterations,
            min_pose_risk=self.pose_render_pre_refine_min_pose_risk,
            max_pose_risk=self.pose_render_pre_refine_max_pose_risk,
            min_existing_gaussians=self.pose_render_pre_refine_min_existing_gaussians,
            min_existing_keyframes=self.pose_render_pre_refine_min_existing_keyframes,
        )
        keyframe.info["_paper_aligned_pose_render_pre_refine"] = debug
        if (
            str(
                keyframe.info.get(
                    "_pose_verification_geometry_anchor_mode",
                    "off",
                )
            )
            == "freeze_v1"
        ):
            debug["run_pre_refine"] = False
            debug["reason"] = "pose_geometry_anchor_frozen"
            self._record_pose_render_pre_refine(debug)
            return debug
        if not bool(debug.get("run_pre_refine", False)):
            self._record_pose_render_pre_refine(debug)
            return debug
        if existing_gaussians <= 0:
            debug["run_pre_refine"] = False
            debug["reason"] = "no_gaussians"
            self._record_pose_render_pre_refine(debug)
            return debug

        start_Rt = keyframe.get_Rt().detach().clone()
        pose_params = {"rW2C", "tW2C"}
        losses: list[float] = []
        render_coverages: list[float] = []
        for _ in range(int(debug.get("iterations", 0))):
            keyframe.zero_grad()
            self.optimizer.zero_grad()
            lvl = keyframe.pyr_lvl
            render_pkg = self.render_from_id(
                keyframe_id, pyr_lvl=lvl, bg=torch.rand(3, device="cuda")
            )
            image = render_pkg["render"]
            gt_image = keyframe.image_pyr[lvl]
            render_support = render_pkg["mainGaussID"][0] >= 0
            if keyframe.mask_pyr is not None:
                keyframe_mask = keyframe.mask_pyr[lvl]
                if keyframe_mask.ndim == 3:
                    keyframe_mask = keyframe_mask[0]
                render_support = render_support & keyframe_mask.bool()
            render_coverage = float(render_support.float().mean().detach().cpu().item())
            render_coverages.append(render_coverage)
            if render_coverage < self.pose_render_pre_refine_min_render_coverage:
                debug["early_stop_reason"] = "insufficient_render_coverage"
                break
            residual = (image - gt_image).abs().mean(dim=0)
            loss = residual[render_support].mean()
            loss.backward()
            with torch.no_grad():
                for name, param_dict in keyframe.optimizer.params.items():
                    if name not in pose_params:
                        param_dict["val"].grad = None
                keyframe.optimizer.step()
            self.optimizer.zero_grad()
            keyframe.zero_grad()
            losses.append(float(loss.detach().cpu().item()))

        if not losses:
            debug["run_pre_refine"] = False
            debug["applied"] = False
            debug["reason"] = str(
                debug.get("early_stop_reason", "") or "insufficient_render_coverage"
            )
            debug["render_coverage"] = (
                sum(render_coverages) / max(len(render_coverages), 1)
                if render_coverages
                else 0.0
            )
            keyframe.info["_paper_aligned_pose_render_pre_refine"] = debug
            self._record_pose_render_pre_refine(debug)
            return debug

        with torch.no_grad():
            end_Rt = keyframe.get_Rt().detach().clone()
            keyframe.approx_centre = keyframe.get_centre().detach()
            if self.approx_cam_centres is not None and len(self.approx_cam_centres) > 0:
                self.approx_cam_centres[keyframe_id] = keyframe.approx_centre
            if len(self.valid_Rt_cache) > 0:
                self.valid_Rt_cache[keyframe_id] = False
            rotation_delta = rotation_distance(
                start_Rt[:3, :3][None], end_Rt[:3, :3][None]
            ) * (180.0 / math.pi)
            translation_delta = torch.linalg.vector_norm(
                start_Rt[:3, 3] - end_Rt[:3, 3]
            )

        debug.update(
            {
                "applied": True,
                "loss_mean": sum(losses) / max(len(losses), 1),
                "loss_last": losses[-1] if losses else 0.0,
                "render_coverage": sum(render_coverages) / max(len(render_coverages), 1),
                "rotation_delta_deg": float(rotation_delta.detach().cpu().view(-1)[0].item()),
                "translation_delta": float(translation_delta.detach().cpu().item()),
            }
        )
        keyframe.info["_paper_aligned_pose_render_pre_refine"] = debug
        self._record_pose_render_pre_refine(debug)
        return debug

    @torch.no_grad()
    def probe_pose_risk_utility(
        self,
        *,
        image: torch.Tensor,
        Rt: torch.Tensor,
        mask: torch.Tensor | None = None,
        downsample: int = 4,
    ) -> dict[str, object]:
        scale = max(1, int(downsample))
        width = max(1, int(self.width) // scale)
        height = max(1, int(self.height) // scale)
        if not hasattr(self, "xyz") or int(self.xyz.shape[0]) <= 0:
            return {
                "available": False,
                "reason": "no_gaussians",
                "render_coverage": 0.0,
                "coverage_deficit": 1.0,
                "residual_mean": 0.0,
                "residual_selectivity": 0.0,
                "probe_width": width,
                "probe_height": height,
            }

        device = self.xyz.device
        target = image.detach().to(device)
        if target.ndim == 4:
            target = target[0]
        target = F.interpolate(
            target[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        view_matrix = Rt.detach().to(device).transpose(0, 1)
        render_pkg = self.render(
            width,
            height,
            view_matrix,
            1.0,
            torch.zeros(3, device=device),
        )
        rendered = render_pkg["render"].detach()
        support = render_pkg["mainGaussID"][0] >= 0
        valid = torch.ones_like(support, dtype=torch.bool)
        if mask is not None:
            valid_mask = mask.detach().to(device)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[0]
            valid_mask = F.interpolate(
                valid_mask[None, None].float(),
                size=(height, width),
                mode="nearest",
            )[0, 0] > 0.5
            valid &= valid_mask

        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            return {
                "available": False,
                "reason": "no_valid_pixels",
                "render_coverage": 0.0,
                "coverage_deficit": 1.0,
                "residual_mean": 0.0,
                "residual_selectivity": 0.0,
                "probe_width": width,
                "probe_height": height,
            }

        supported_valid = support & valid
        render_coverage = float(
            supported_valid.sum().float().div(float(valid_count)).item()
        )
        residual = (rendered - target).abs().mean(dim=0)
        dx = F.pad((residual[:, 1:] - residual[:, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((residual[1:, :] - residual[:-1, :]).abs(), (0, 0, 0, 1))
        response = 0.65 * residual + 0.35 * (dx + dy)
        response_values = response[valid]
        response_mean = response_values.mean().clamp_min(1e-6)
        normalized = (response_values / response_mean).clamp(0.25, 4.0)
        residual_selectivity = float(
            (torch.quantile(normalized, 0.90) - torch.quantile(normalized, 0.50))
            .detach()
            .cpu()
            .item()
        )
        return {
            "available": True,
            "reason": "ok",
            "render_coverage": render_coverage,
            "coverage_deficit": max(0.0, 1.0 - render_coverage),
            "residual_mean": float(response_values.mean().detach().cpu().item()),
            "residual_selectivity": max(0.0, residual_selectivity),
            "valid_ratio": float(valid.float().mean().detach().cpu().item()),
            "probe_width": width,
            "probe_height": height,
        }

    def review_pose_risk_keyframe(
        self,
        keyframe_id: int = -1,
        *,
        iterations: int = 2,
        min_render_coverage: float = 0.15,
        max_rotation_delta_deg: float = 1.5,
        max_translation_delta: float = 0.05,
        min_relative_loss_improvement: float = 0.0,
        min_validation_support_ratio: float = 0.95,
        learning_rate_scale: float = 1.0,
        trace_key: str = "_pose_risk_utility_review",
    ) -> dict[str, object]:
        keyframe = self.keyframes[keyframe_id]
        bounded_iterations = max(0, min(12, int(iterations)))
        debug: dict[str, object] = {
            "requested": True,
            "applied": False,
            "accepted": False,
            "iterations": bounded_iterations,
            "min_relative_loss_improvement": float(
                max(0.0, min_relative_loss_improvement)
            ),
            "min_validation_support_ratio": float(
                max(0.0, min(1.0, min_validation_support_ratio))
            ),
            "learning_rate_scale": float(max(0.0, learning_rate_scale)),
            "reason": "",
        }
        if bounded_iterations <= 0:
            debug["reason"] = "zero_iterations"
            return debug
        if not hasattr(self, "xyz") or int(self.xyz.shape[0]) <= 0:
            debug["reason"] = "no_gaussians"
            return debug

        start_Rt = keyframe.get_Rt().detach().clone()
        pose_params = {"rW2C", "tW2C"}
        optimizer_snapshot = snapshot_optimizer_parameter_state(
            keyframe.optimizer,
            pose_params,
        )
        learning_rate_snapshot = scale_optimizer_parameter_learning_rates(
            keyframe.optimizer,
            pose_params,
            scale=learning_rate_scale,
        )
        lvl = keyframe.pyr_lvl
        fixed_bg = torch.zeros(3, device=start_Rt.device)

        def render_residual_support():
            render_pkg = self.render_from_id(keyframe_id, pyr_lvl=lvl, bg=fixed_bg)
            support = render_pkg["mainGaussID"][0] >= 0
            if keyframe.mask_pyr is not None:
                keyframe_mask = keyframe.mask_pyr[lvl]
                if keyframe_mask.ndim == 3:
                    keyframe_mask = keyframe_mask[0]
                support &= keyframe_mask.bool()
            coverage = float(support.float().mean().detach().cpu().item())
            residual = (render_pkg["render"] - keyframe.image_pyr[lvl]).abs().mean(dim=0)
            return residual, support, coverage

        with torch.no_grad():
            initial_residual, initial_support, initial_coverage = (
                render_residual_support()
            )
        debug["render_coverage"] = initial_coverage
        if (
            initial_coverage < float(min_render_coverage)
            or not bool(initial_support.any())
        ):
            debug["reason"] = "insufficient_render_coverage"
            return debug

        height, width = initial_support.shape[-2:]
        rows = torch.arange(height, device=initial_support.device).view(-1, 1)
        columns = torch.arange(width, device=initial_support.device).view(1, -1)
        validation_selector = ((rows + columns) % 2) == 0
        initial_validation_support = initial_support & validation_selector
        initial_solve_support = initial_support & ~validation_selector
        initial_validation_count = int(initial_validation_support.sum().item())
        initial_solve_count = int(initial_solve_support.sum().item())
        debug["initial_validation_support"] = initial_validation_count
        debug["initial_solve_support"] = initial_solve_count
        if initial_validation_count <= 0 or initial_solve_count <= 0:
            debug["reason"] = "insufficient_split_support"
            return debug

        def solve_loss() -> tuple[torch.Tensor | None, float, float]:
            residual, support, coverage = render_residual_support()
            common_support = initial_solve_support & support
            support_ratio = float(
                common_support.sum().detach().cpu().item()
                / max(initial_solve_count, 1)
            )
            if (
                coverage < float(min_render_coverage)
                or not bool(common_support.any())
                or support_ratio < float(min_validation_support_ratio)
            ):
                return None, coverage, support_ratio
            return residual[common_support].mean(), coverage, support_ratio

        completed = 0
        solve_support_ratio = 1.0
        loop_failure_reason = ""
        try:
            for _ in range(bounded_iterations):
                keyframe.zero_grad()
                self.optimizer.zero_grad()
                loss, coverage, solve_support_ratio = solve_loss()
                if loss is None:
                    loop_failure_reason = "solve_support_lost"
                    break
                loss.backward()
                with torch.no_grad():
                    for name, param_dict in keyframe.optimizer.params.items():
                        if name not in pose_params:
                            param_dict["val"].grad = None
                    keyframe.optimizer.step()
                self.optimizer.zero_grad()
                keyframe.zero_grad()
                completed += 1
        finally:
            restore_optimizer_parameter_learning_rates(
                keyframe.optimizer,
                learning_rate_snapshot,
            )

        with torch.no_grad():
            end_residual, end_support, end_coverage = render_residual_support()
            end_Rt = keyframe.get_Rt().detach().clone()
            common_validation_support = initial_validation_support & end_support
            common_validation_count = int(common_validation_support.sum().item())
            validation_support_ratio = float(
                common_validation_count / max(initial_validation_count, 1)
            )
            if common_validation_count > 0:
                start_loss_tensor = initial_residual[common_validation_support].mean()
                end_loss_tensor = end_residual[common_validation_support].mean()
            else:
                start_loss_tensor = None
                end_loss_tensor = None
            rotation_delta = rotation_distance(
                start_Rt[:3, :3][None], end_Rt[:3, :3][None]
            ) * (180.0 / math.pi)
            translation_delta = torch.linalg.vector_norm(
                start_Rt[:3, 3] - end_Rt[:3, 3]
            )
        start_loss = (
            float(start_loss_tensor.detach().cpu().item())
            if start_loss_tensor is not None
            else float("inf")
        )
        end_loss = (
            float(end_loss_tensor.detach().cpu().item())
            if end_loss_tensor is not None
            else float("inf")
        )
        rotation_delta_deg = float(rotation_delta.detach().cpu().view(-1)[0].item())
        translation_delta_value = float(translation_delta.detach().cpu().item())
        accepted, reason = pose_review_acceptance(
            start_loss=start_loss,
            end_loss=end_loss,
            rotation_delta_deg=rotation_delta_deg,
            translation_delta=translation_delta_value,
            max_rotation_delta_deg=max_rotation_delta_deg,
            max_translation_delta=max_translation_delta,
            min_relative_loss_improvement=min_relative_loss_improvement,
            validation_support_ratio=validation_support_ratio,
            min_validation_support_ratio=min_validation_support_ratio,
        )
        if loop_failure_reason:
            accepted = False
            reason = loop_failure_reason
        restore_optimizer_parameter_state(
            keyframe.optimizer,
            optimizer_snapshot,
            restore_values=not accepted,
        )
        if not accepted:
            keyframe.set_Rt(start_Rt)
        keyframe.optimizer.zero_grad()
        self.optimizer.zero_grad()
        final_Rt = keyframe.get_Rt().detach().clone()
        keyframe.approx_centre = keyframe.get_centre().detach()
        if hasattr(self, "approx_cam_centres") and self.approx_cam_centres is not None:
            try:
                self.approx_cam_centres[keyframe_id] = keyframe.approx_centre
            except (IndexError, TypeError):
                pass
        if hasattr(self, "valid_Rt_cache") and len(self.valid_Rt_cache) > 0:
            try:
                self.valid_Rt_cache[keyframe_id] = False
            except (IndexError, TypeError):
                pass

        debug.update(
            {
                "applied": bool(completed > 0),
                "accepted": bool(accepted),
                "reason": reason,
                "completed_iterations": int(completed),
                "start_loss": start_loss,
                "end_loss": end_loss,
                "end_render_coverage": float(end_coverage),
                "validation_support_ratio": validation_support_ratio,
                "solve_support_ratio": float(solve_support_ratio),
                "rotation_delta_deg": rotation_delta_deg,
                "translation_delta": translation_delta_value,
                "initial_Rt": start_Rt.detach().cpu().tolist(),
                "refined_Rt": end_Rt.detach().cpu().tolist(),
                "final_Rt": final_Rt.detach().cpu().tolist(),
            }
        )
        keyframe.info[str(trace_key)] = dict(debug)
        return debug

    def _transactional_refinement_forward(
        self,
        keyframe_id: int,
        fixed_background: torch.Tensor,
    ):
        keyframe = self.keyframes[int(keyframe_id)]
        lvl = int(keyframe.pyr_lvl)
        render_pkg = self.render_from_id(
            int(keyframe_id),
            pyr_lvl=lvl,
            bg=fixed_background,
        )
        image = render_pkg["render"]
        invdepth = render_pkg["invdepth"]
        target = keyframe.image_pyr[lvl]
        mono_idepth = keyframe.get_mono_idepth(lvl)
        if keyframe.mask_pyr is not None:
            mask = keyframe.mask_pyr[lvl]
            image = image * mask
            target = target * mask
            invdepth = invdepth * mask
            mono_idepth = mono_idepth * mask

        l1_loss = (image - target).abs().mean()
        dssim_loss = 1 - fused_ssim(image[None], target[None])
        depth_loss = (invdepth - mono_idepth).abs().mean()
        total_loss = (
            self.lambda_dssim * dssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
        )
        metrics = {
            "total_loss": _metric_float(total_loss),
            "rgb_mse": _metric_float((image - target).square().mean()),
            "dssim": _metric_float(dssim_loss),
            "depth": _metric_float(depth_loss),
            "l1": _metric_float(l1_loss),
        }
        visibility = render_pkg["visibility_filter"].detach().bool()
        gaussian_count = int(render_pkg["radii"].shape[0])
        return total_loss, metrics, visibility, gaussian_count

    def _run_transactional_gaussian_refinement(
        self,
        debug: dict[str, object],
    ) -> dict[str, object]:
        result = dict(debug)
        requested_iterations = int(result.get("extra_iterations", 0) or 0)
        result.update(
            {
                "requested_extra_iterations": requested_iterations,
                "transaction_attempted": False,
                "committed": False,
                "rolled_back": False,
                "realized_iterations": 0,
                "best_iteration": 0,
                "early_stopped": False,
                "time_budget_exhausted": False,
                "refinement_runtime_seconds": 0.0,
            }
        )
        if (
            not bool(result.get("applied", False))
            or requested_iterations <= 0
            or not self.keyframes
            or int(self.xyz.shape[0]) == 0
        ):
            result["applied"] = False
            return result

        current_id = len(self.keyframes) - 1
        keyframe = self.keyframes[current_id]
        if bool(keyframe.info.get("is_test", False)):
            result.update({"applied": False, "reason": "test_frame"})
            return result
        reference_ids = select_refinement_reference_indices(
            [
                bool(frame.info.get("is_test", False))
                for frame in self.keyframes
            ],
            current_index=current_id,
            max_references=2,
        )
        result["reference_keyframe_ids"] = reference_ids
        if not reference_ids:
            result.update(
                {
                    "applied": False,
                    "reason": "no_historical_training_reference",
                }
            )
            return result

        stats = self.pose_render_extra_optimization_stats
        available_seconds = refinement_time_budget_seconds(
            float(stats.get("base_runtime_seconds", 0.0)),
            float(stats.get("transaction_runtime_seconds", 0.0)),
            target_ratio=float(stats.get("refinement_time_target_ratio", 0.08)),
        )
        result["refinement_time_budget_seconds"] = available_seconds
        if available_seconds <= 0.0:
            result.update(
                {
                    "applied": False,
                    "reason": "refinement_time_budget_exhausted",
                    "time_budget_exhausted": True,
                }
            )
            return result

        lvl = int(keyframe.pyr_lvl)
        fixed_background = keyframe.image_pyr[lvl].new_zeros(3)
        last_trained_before = self.last_trained_id
        cpu_rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.xyz.device) if self.xyz.is_cuda else None
        )
        state_snapshot = None
        initial_visibility = None
        initial_metrics = None
        initial_reference_metrics = None
        best_metrics = None
        best_reference_metrics = None
        best_iteration = 0
        completed = 0
        consecutive_rejections = 0
        reference_guard_rejections = 0
        last_acceptance_reason = ""
        if self.xyz.is_cuda:
            torch.cuda.synchronize(self.xyz.device)
        started = time.perf_counter()
        result["transaction_attempted"] = True
        try:
            keyframe.zero_grad()
            self.optimizer.zero_grad()
            current_loss, initial_metrics, current_visibility, gaussian_count = (
                self._transactional_refinement_forward(
                    current_id,
                    fixed_background,
                )
            )
            initial_reference_metrics = []
            with torch.no_grad():
                for reference_id in reference_ids:
                    _, reference_metrics, _, _ = (
                        self._transactional_refinement_forward(
                            reference_id,
                            fixed_background,
                        )
                    )
                    initial_reference_metrics.append(reference_metrics)
            initial_visibility = current_visibility.clone()
            if not bool(initial_visibility.any()):
                result.update(
                    {
                        "applied": False,
                        "reason": "no_visible_gaussians",
                        "rolled_back": True,
                    }
                )
            else:
                state_snapshot = snapshot_selected_gaussian_state(
                    self.gaussian_params,
                    initial_visibility,
                )
                for iteration in range(1, requested_iterations + 1):
                    current_loss.backward()
                    scale_gaussian_gradients(self.gaussian_params)
                    with torch.no_grad():
                        selected_visibility = (
                            current_visibility.bool() & initial_visibility
                        )
                        if bool(selected_visibility.any()):
                            self.optimizer.step(
                                selected_visibility,
                                gaussian_count,
                            )
                    self.optimizer.zero_grad()
                    keyframe.zero_grad()
                    completed = iteration

                    current_loss, candidate_metrics, current_visibility, gaussian_count = (
                        self._transactional_refinement_forward(
                            current_id,
                            fixed_background,
                        )
                    )
                    acceptance = refinement_candidate_acceptance(
                        initial_metrics,
                        candidate_metrics,
                    )
                    candidate_reference_metrics = []
                    with torch.no_grad():
                        for reference_id in reference_ids:
                            _, reference_metrics, _, _ = (
                                self._transactional_refinement_forward(
                                    reference_id,
                                    fixed_background,
                                )
                            )
                            candidate_reference_metrics.append(reference_metrics)
                    reference_acceptance = refinement_reference_guard(
                        initial_reference_metrics,
                        candidate_reference_metrics,
                    )
                    if not bool(reference_acceptance.get("accepted", False)):
                        reference_guard_rejections += 1
                    last_acceptance_reason = str(
                        (
                            reference_acceptance
                            if bool(acceptance.get("accepted", False))
                            else acceptance
                        ).get("reason", "")
                    )
                    candidate_is_best = (
                        bool(acceptance.get("accepted", False))
                        and bool(reference_acceptance.get("accepted", False))
                        and (
                        best_metrics is None
                        or float(candidate_metrics["total_loss"])
                        < float(best_metrics["total_loss"])
                        )
                    )
                    if candidate_is_best:
                        overwrite_selected_gaussian_value_snapshot(
                            self.gaussian_params,
                            initial_visibility,
                            state_snapshot,
                        )
                        best_metrics = dict(candidate_metrics)
                        best_reference_metrics = [
                            dict(metrics) for metrics in candidate_reference_metrics
                        ]
                        best_iteration = iteration
                        consecutive_rejections = 0
                    else:
                        consecutive_rejections += 1

                    elapsed = time.perf_counter() - started
                    if elapsed >= available_seconds:
                        result["time_budget_exhausted"] = True
                        break
                    if consecutive_rejections >= 2:
                        result["early_stopped"] = True
                        break

                restore_selected_gaussian_state(
                    self.gaussian_params,
                    initial_visibility,
                    state_snapshot,
                )
                committed = best_metrics is not None
                result.update(
                    {
                        "applied": committed,
                        "committed": committed,
                        "rolled_back": not committed,
                        "reason": (
                            "transaction_committed"
                            if committed
                            else "transaction_no_safe_candidate"
                        ),
                        "extra_iterations": best_iteration if committed else 0,
                        "realized_iterations": completed,
                        "best_iteration": best_iteration,
                        "pre_refinement_metrics": dict(initial_metrics),
                        "post_refinement_metrics": (
                            dict(best_metrics)
                            if best_metrics is not None
                            else dict(initial_metrics)
                        ),
                        "pre_refinement_reference_metrics": [
                            dict(metrics) for metrics in initial_reference_metrics
                        ],
                        "post_refinement_reference_metrics": (
                            best_reference_metrics
                            if best_reference_metrics is not None
                            else [
                                dict(metrics)
                                for metrics in initial_reference_metrics
                            ]
                        ),
                        "reference_guard_rejections": reference_guard_rejections,
                        "last_acceptance_reason": last_acceptance_reason,
                    }
                )
        finally:
            if (
                state_snapshot is not None
                and initial_visibility is not None
                and not bool(result.get("committed", False))
                and str(result.get("reason", "")) != "transaction_no_safe_candidate"
            ):
                restore_selected_gaussian_state(
                    self.gaussian_params,
                    initial_visibility,
                    state_snapshot,
                )
            self.optimizer.zero_grad()
            keyframe.zero_grad()
            torch.random.set_rng_state(cpu_rng_state)
            np.random.set_state(numpy_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, self.xyz.device)
            self.last_trained_id = last_trained_before

        if self.xyz.is_cuda:
            torch.cuda.synchronize(self.xyz.device)
        runtime_seconds = time.perf_counter() - started
        result["refinement_runtime_seconds"] = runtime_seconds
        result["last_trained_id_preserved"] = self.last_trained_id == last_trained_before
        result["rng_state_preserved"] = True
        return result

    def optimization_loop(self, n_iters: int, run_until_interupt: bool = False):
        """
        【优化模块】优化循环

        执行至少n_iters次优化步骤。
        如果run_until_interupt为True，会持续运行直到join_optimization_thread被调用
        （用于流式模式下持续优化直到添加下一个关键帧）。

        Args:
            n_iters: 最小优化迭代次数
            run_until_interupt: 是否持续运行直到中断信号
        """
        response_gated_extra_optimization = (
            self.pose_render_extra_optimization == "pose_confidence_render_response_v2"
            or self.pose_render_extra_optimization == "render_response_v3"
            or self.pose_render_extra_optimization == "render_response_v4"
            or self.pose_render_extra_optimization == "render_response_v5"
            or self.pose_render_extra_optimization == "render_response_mask_conservative_v6"
        )
        extra_optimization_debug = {
            "mode": self.pose_render_extra_optimization,
            "applied": False,
            "reason": "render_response_deferred",
            "extra_iterations": 0,
        }
        if not response_gated_extra_optimization:
            extra_optimization_debug = pose_render_extra_optimization_decision(
                mode=self.pose_render_extra_optimization,
                direct_density_mode=self.paper_aligned_direct_density_control,
                keyframe_info=self.keyframes[-1].info if self.keyframes else {},
                base_iterations=int(n_iters),
                render_frame_policy=self.paper_aligned_render_frame_policy,
                min_confidence=self.pose_render_extra_optimization_min_confidence,
                fraction=self.pose_render_extra_optimization_fraction,
                max_extra=self.pose_render_extra_optimization_max_extra,
                max_pose_risk=self.pose_render_extra_optimization_max_pose_risk,
                max_utility_drift=self.pose_render_extra_optimization_max_utility_drift,
                min_pose_support=self.pose_render_extra_optimization_min_pose_support,
                min_match_support=self.pose_render_extra_optimization_min_match_support,
            )
        if self.pose_render_extra_optimization != "off" and not response_gated_extra_optimization:
            if self.keyframes:
                self.keyframes[-1].info["_paper_aligned_pose_render_extra_optimization"] = (
                    extra_optimization_debug
                )
            self._record_pose_render_extra_optimization(extra_optimization_debug)

        # 重置中断标志
        self.interupt_optimization = False
        i = 0
        if self.xyz.is_cuda:
            torch.cuda.synchronize(self.xyz.device)
        base_loop_started = time.perf_counter()
        # 持续优化直到达到最小迭代次数，或收到中断信号
        while i < n_iters or (run_until_interupt and not self.interupt_optimization):
            pose_update_enabled = async_pose_update_enabled(
                self.pose_verification_async_pose_protection_mode,
                run_until_interrupt=run_until_interupt,
                iteration=i,
                base_iterations=n_iters,
            )
            if pose_update_enabled:
                self.pose_verification_async_pose_protection_stats[
                    "joint_pose_steps"
                ] += 1
            else:
                self.pose_verification_async_pose_protection_stats[
                    "protected_gaussian_steps"
                ] += 1
            self.optimization_step(update_pose=pose_update_enabled)
            i += 1
        if self.xyz.is_cuda:
            torch.cuda.synchronize(self.xyz.device)
        base_loop_runtime = time.perf_counter() - base_loop_started
        if (
            self.pose_render_extra_optimization
            == EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE
        ):
            self.pose_render_extra_optimization_stats["base_runtime_seconds"] = float(
                self.pose_render_extra_optimization_stats.get(
                    "base_runtime_seconds",
                    0.0,
                )
            ) + base_loop_runtime
        if (
            self.pose_render_extra_optimization == "pose_confidence_render_response_v2"
            or self.pose_render_extra_optimization == "render_response_v3"
            or self.pose_render_extra_optimization == "render_response_v4"
            or self.pose_render_extra_optimization == "render_response_v5"
            or self.pose_render_extra_optimization == "render_response_mask_conservative_v6"
        ):
            if (
                (
                    self.pose_render_extra_optimization == "render_response_v4"
                    or self.pose_render_extra_optimization == "render_response_v5"
                    or self.pose_render_extra_optimization
                    == "render_response_mask_conservative_v6"
                )
                and self.keyframes
            ):
                self.keyframes[-1].info[TEXTURE_SAMPLING_SCENE_GUARD_KEY] = self._pose_render_texture_sampling_scene_guard()
            extra_optimization_debug = pose_render_extra_optimization_decision(
                mode=self.pose_render_extra_optimization,
                direct_density_mode=self.paper_aligned_direct_density_control,
                keyframe_info=self.keyframes[-1].info if self.keyframes else {},
                base_iterations=int(n_iters),
                render_frame_policy=self.paper_aligned_render_frame_policy,
                min_confidence=self.pose_render_extra_optimization_min_confidence,
                fraction=self.pose_render_extra_optimization_fraction,
                max_extra=self.pose_render_extra_optimization_max_extra,
                max_pose_risk=self.pose_render_extra_optimization_max_pose_risk,
                max_utility_drift=self.pose_render_extra_optimization_max_utility_drift,
                min_pose_support=self.pose_render_extra_optimization_min_pose_support,
                min_match_support=self.pose_render_extra_optimization_min_match_support,
            )
        if not run_until_interupt:
            if (
                self.pose_render_extra_optimization
                == EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE
            ):
                if self.keyframes:
                    keyframe = self.keyframes[-1]
                    already_finalized = (
                        str(extra_optimization_debug.get("reason", ""))
                        == "refinement_already_finalized"
                    )
                    if not already_finalized:
                        keyframe.info[EXTRA_REFINEMENT_STATE_KEY] = {
                            "state": "ATTEMPTING",
                            "attempted": bool(
                                extra_optimization_debug.get("applied", False)
                            ),
                        }
                        extra_optimization_debug = (
                            self._run_transactional_gaussian_refinement(
                                extra_optimization_debug
                            )
                        )
                        keyframe.info[EXTRA_REFINEMENT_STATE_KEY] = {
                            "state": "DONE",
                            "attempted": bool(
                                extra_optimization_debug.get(
                                    "transaction_attempted",
                                    False,
                                )
                            ),
                            "committed": bool(
                                extra_optimization_debug.get("committed", False)
                            ),
                            "reason": str(
                                extra_optimization_debug.get("reason", "")
                            ),
                            "best_iteration": int(
                                extra_optimization_debug.get("best_iteration", 0)
                                or 0
                            ),
                        }
            else:
                for _ in range(
                    int(extra_optimization_debug.get("extra_iterations", 0) or 0)
                ):
                    self.optimization_step(
                        keyframe_id_override=-1,
                        update_pose=False,
                    )
        elif (
            self.pose_render_extra_optimization
            == EXTRA_OPTIMIZATION_RENDER_ONLY_RESPONSE_MODE
        ):
            extra_optimization_debug = dict(extra_optimization_debug)
            extra_optimization_debug.update(
                {
                    "applied": False,
                    "reason": "transaction_deferred_async",
                    "extra_iterations": 0,
                }
            )

        if response_gated_extra_optimization:
            if self.keyframes:
                self.keyframes[-1].info[
                    "_paper_aligned_pose_render_extra_optimization"
                ] = extra_optimization_debug
            self._record_pose_render_extra_optimization(extra_optimization_debug)

    def join_optimization_thread(self):
        """
        【优化模块】中断优化循环并等待线程结束

        发送中断信号给优化线程，并等待其完成当前迭代后退出。
        用于在添加新关键帧前确保优化线程已停止。
        """
        if self.optimization_thread is not None:
            # 设置中断标志
            self.interupt_optimization = True
            # 等待线程结束
            self.optimization_thread.join()
            self.optimization_thread = None

    def optimize_async(self, n_iters: int):
        """
        【优化模块】异步启动优化线程

        在后台线程中运行优化循环，至少执行n_iters次优化步骤。
        用于流式模式下在不阻塞主线程的情况下持续优化场景。

        Args:
            n_iters: 最小优化迭代次数
        """
        # 先停止之前的优化线程（如果存在）
        self.join_optimization_thread()
        # 创建新的优化线程
        self.optimization_thread = threading.Thread(
            target=self.optimization_loop, args=(n_iters, True)
        )
        self.optimization_thread.start()

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """
        【渲染模块】统一测试关键帧的曝光矩阵

        通过平均相邻关键帧的曝光值来统一测试关键帧的曝光。
        这样可以确保测试帧的渲染质量不受曝光差异影响。
        """
        mode = self.paper_aligned_test_exposure_harmonization
        if mode == "source_time_adaptive_v2":
            train_count = sum(1 for keyframe in self.keyframes if not keyframe.info["is_test"])
            test_count = sum(1 for keyframe in self.keyframes if keyframe.info["is_test"])
            texture_stats = self.pose_render_texture_sampling_stats
            events = int(texture_stats.get("events", 0) or 0)
            coverage_deficit_mean = texture_stats.get("coverage_deficit_mean", None)
            if coverage_deficit_mean is None and events > 0:
                coverage_deficit_mean = float(
                    texture_stats.get("coverage_deficit_sum", 0.0) or 0.0
                ) / float(events)
            decision = adaptive_source_time_exposure_scene_decision(
                train_keyframes=train_count,
                test_keyframes=test_count,
                texture_sampling_events=events,
                texture_sampling_applied=int(texture_stats.get("applied", 0) or 0),
                coverage_deficit_mean=coverage_deficit_mean,
            )
            self.paper_aligned_test_exposure_adaptive_stats = decision
            mode = str(decision.get("selected_mode", "neighbor_average_v1"))
        elif mode == "dark_scene_off_guarded_v1":
            decision = dark_scene_test_exposure_scene_decision(
                self._ensure_training_background_stats()
            )
            self.paper_aligned_test_exposure_adaptive_stats = decision
            mode = str(decision.get("selected_mode", "neighbor_average_v1"))
        else:
            self.paper_aligned_test_exposure_adaptive_stats = {
                "mode": mode,
                "selected_mode": mode,
                "reason": "not_adaptive",
            }
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.info["is_test"]:
                keyframe.exposure = harmonized_test_exposure(
                    self.keyframes,
                    index,
                    mode=mode,
                    max_neighbor_exposure_delta=(
                        self.paper_aligned_test_exposure_guard_max_delta
                    ),
                )

    @torch.no_grad()
    def _evaluate_lpips(self, image, gt_image):
        def compute_on_cpu():
            self.lpips = self.lpips.cpu()
            torch.cuda.empty_cache()
            image_cpu = image.detach().cpu()
            gt_image_cpu = gt_image.detach().cpu()
            try:
                return float(self.lpips(image_cpu[None], gt_image_cpu[None]).item())
            finally:
                del image_cpu, gt_image_cpu

        if torch.cuda.is_available():
            try:
                free_cuda_bytes, _ = torch.cuda.mem_get_info()
                if _lpips_cpu_fallback_needed(free_cuda_bytes):
                    return compute_on_cpu()
            except RuntimeError:
                pass

        try:
            if _module_parameters_on_cuda(self.lpips):
                torch.cuda.empty_cache()
                return float(self.lpips(image[None], gt_image[None]).item())
            return compute_on_cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return compute_on_cpu()

    @torch.no_grad()
    def evaluate(
        self,
        eval_poses=False,
        with_LPIPS=False,
        all=False,
        return_frame_metrics=False,
        frame_metrics_output_dir="",
    ):
        """
        【评估模块】评估场景质量

        计算渲染质量和位姿误差指标。

        Args:
            eval_poses: 是否计算位姿误差
            with_LPIPS: 是否计算LPIPS感知损失
            all: 是否评估所有关键帧（False时只评估活跃锚点的关键帧）

        Returns:
            dict: 包含PSNR、SSIM、LPIPS（可选）、位姿误差（可选）的字典
        """
        # ========== 统一测试关键帧曝光 ==========
        # 确保测试关键帧的曝光与相邻帧一致
        self.harmonize_test_exposure()

        # ========== 计算图像质量指标 ==========
        # 初始化指标字典
        metrics = {"PSNR": 0, "SSIM": 0}
        if with_LPIPS:
            metrics["LPIPS"] = 0
        n_test_frames = 0
        frame_quality_by_index = {}
        pose_error_by_index = {}
        # 确定评估的关键帧范围
        start_index = 0 if all else self.active_anchor.keyframe_ids[0]

        # 遍历测试关键帧并计算指标
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if keyframe.info["is_test"]:
                # 获取真实图像（全分辨率）
                gt_image = keyframe.image_pyr[0].cuda()
                # 渲染当前视角
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]

                # 应用掩码（如果有）
                mask = (
                    keyframe.mask_pyr[0].cuda()
                    if keyframe.mask_pyr is not None
                    else torch.ones_like(image[:1] > 0)
                )
                mask = mask.expand_as(image)
                image, _ = calibrate_test_render(
                    image,
                    gt_image,
                    mode=self.paper_aligned_test_render_calibration,
                    mask=mask,
                )
                image = image * mask
                gt_image = gt_image * mask

                # 计算PSNR（峰值信噪比）
                psnr_value = _metric_float(psnr(image[mask], gt_image[mask]))
                metrics["PSNR"] += psnr_value
                # 计算SSIM（结构相似性）
                ssim_value = float(
                    fused_ssim(image[None], gt_image[None], train=False).item()
                )
                metrics["SSIM"] += ssim_value
                # 计算LPIPS（感知损失，如果启用）
                lpips_value = None
                if with_LPIPS and self.lpips is not None:
                    lpips_value = self._evaluate_lpips(image, gt_image)
                    metrics["LPIPS"] += lpips_value
                frame_quality_by_index[int(keyframe.index)] = {
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                }
                del render_pkg, image, gt_image, mask
                torch.cuda.empty_cache()
                n_test_frames += 1

        # 计算平均指标
        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}

        # ========== 计算位姿误差 ==========
        if eval_poses:
            # 获取优化后的位姿和真实位姿
            all_Rts = self.get_Rts()
            Rts, gt_Rts = select_pose_eval_pairs(
                all_Rts,
                self.get_gt_Rts(align=False),
                self.gt_Rts_mask,
            )
            if len(Rts) == len(gt_Rts) and len(Rts) > 0:
                # 对齐位姿（计算相似变换）
                Rts_aligned = torch.linalg.inv(align_poses(Rts, gt_Rts))
                gt_Rts = torch.linalg.inv(gt_Rts)
                # 计算旋转误差（角度）
                R_error = rotation_distance(Rts_aligned[:, :3, :3], gt_Rts[:, :3, :3])
                # 计算平移误差（欧氏距离）
                t_error = (Rts_aligned[:, :3, 3] - gt_Rts[:, :3, 3]).norm(dim=-1)

                # 转换为度数和米
                R_error_deg = R_error * 180 / math.pi
                metrics["R°"] = R_error_deg.mean().item()
                metrics["t"] = t_error.mean().item()
                valid_indices = self._pose_eval_keyframe_indices(all_Rts)
                pair_count = min(len(valid_indices), len(R_error_deg), len(t_error))
                for offset, keyframe_index in enumerate(valid_indices[:pair_count]):
                    pose_error_by_index[int(keyframe_index)] = {
                        "abs_rot_error_deg": float(R_error_deg[offset].detach().cpu().item()),
                        "abs_trans_error": float(t_error[offset].detach().cpu().item()),
                    }

        if return_frame_metrics:
            return metrics, self._build_frame_metric_rows(
                frame_quality_by_index=frame_quality_by_index,
                pose_error_by_index=pose_error_by_index,
                output_dir=frame_metrics_output_dir,
            )
        return metrics

    def _pose_eval_keyframe_indices(self, all_Rts):
        n_masked_poses = min(int(self.gt_Rts_mask.shape[0]), int(all_Rts.shape[0]))
        if n_masked_poses <= 0:
            return []
        valid_mask = self.gt_Rts_mask[:n_masked_poses].to(
            device=all_Rts.device,
            dtype=torch.bool,
        )
        return torch.where(valid_mask)[0].detach().cpu().tolist()

    def _build_frame_metric_rows(
        self,
        *,
        frame_quality_by_index,
        pose_error_by_index,
        output_dir,
    ):
        dataset_name, scene_name = infer_dataset_scene_from_output_dir(output_dir)
        all_Rts = self.get_Rts().detach().cpu().tolist() if len(self.keyframes) > 0 else []
        rows = []
        for sequence_order, keyframe in enumerate(self.keyframes):
            keyframe_index = int(keyframe.index)
            rows.append(
                build_frame_metric_row(
                    dataset_name=dataset_name,
                    scene_name=scene_name,
                    frame_idx=keyframe_index,
                    original_image_name=str(keyframe.info.get("name", "")),
                    sequence_order=sequence_order,
                    stream_frame_idx=int(keyframe.info.get("_paper_aligned_source_frame_id", sequence_order)),
                    is_test_view=bool(keyframe.info.get("is_test", False)),
                    baseline_eval={
                        "frame": bool(
                            keyframe.info.get(
                                "_baseline_eval_frame",
                                keyframe.info.get("is_test", False),
                            )
                        ),
                        "hold": keyframe.info.get("_baseline_eval_hold", ""),
                        "sequence_index": keyframe.info.get(
                            "_baseline_eval_sequence_index", ""
                        ),
                        "original_index": keyframe.info.get(
                            "_baseline_eval_original_index", ""
                        ),
                    },
                    is_keyframe=True,
                    is_registered=True,
                    est_rt=all_Rts[keyframe_index] if keyframe_index < len(all_Rts) else None,
                    quality=frame_quality_by_index.get(keyframe_index, {}),
                    pose_error=pose_error_by_index.get(keyframe_index, {}),
                    output_dir=output_dir,
                )
            )
        return rows

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        """
        【评估模块】保存测试关键帧的渲染图像

        为所有测试关键帧渲染图像并保存到指定目录。
        用于生成评估结果的可视化。

        Args:
            out_dir: 输出目录路径
        """
        # 统一测试关键帧曝光，确保渲染质量
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)

        # 遍历所有关键帧，渲染并保存测试帧
        for keyframe in self.keyframes:
            if keyframe.info["is_test"]:
                # 渲染全分辨率图像
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                gt_image = keyframe.image_pyr[0].cuda()
                mask = (
                    keyframe.mask_pyr[0].cuda()
                    if keyframe.mask_pyr is not None
                    else torch.ones_like(render_pkg["render"][:1] > 0)
                )
                mask = mask.expand_as(render_pkg["render"])
                render_pkg["render"], _ = calibrate_test_render(
                    render_pkg["render"],
                    gt_image,
                    mode=self.paper_aligned_test_render_calibration,
                    mask=mask,
                )
                # 转换为8位RGB图像（0-255范围）
                image = torch.clamp(render_pkg["render"], 0, 1) * 255
                image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # 转换为BGR格式（OpenCV使用BGR）
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # 根据文件扩展名选择保存格式（JPEG或PNG）
                is_jpeg = os.path.splitext(keyframe.info["name"])[-1].lower() in [
                    ".jpg",
                    ".jpeg",
                ]
                # JPEG质量设置为100（无损）
                write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
                cv2.imwrite(
                    os.path.join(out_dir, keyframe.info["name"]), image, write_flag
                )

    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        scaling_modifier=1,
        bg=torch.zeros(3, device="cuda"),
    ):
        """
        【渲染模块】从指定关键帧ID渲染场景

        从给定关键帧视角渲染场景，支持多分辨率（金字塔层级）和曝光校正。

        Args:
            keyframe_id: 关键帧索引
            pyr_lvl: 金字塔层级（0=全分辨率，1=半分辨率，...）
            scaling_modifier: 高斯尺度缩放因子（用于可视化）
            bg: 背景颜色 [3]

        Returns:
            dict: 包含渲染图像、逆深度、主要高斯ID、半径等信息的字典
        """
        # 获取关键帧和视图矩阵
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().transpose(0, 1)

        # 根据金字塔层级计算分辨率
        scale = 2**pyr_lvl
        width, height = self.width // scale, self.height // scale

        # 调用底层渲染函数
        render_pkg = self.render(width, height, view_matrix, scaling_modifier, bg)

        # ========== 应用曝光校正 ==========
        # 曝光矩阵：[3x4]，包含颜色变换和平移
        # 将渲染图像从原始颜色空间转换到关键帧的曝光空间
        render_pkg["render"] = (
            keyframe.exposure[:3, :3] @ render_pkg["render"].view(3, -1)
        ) + keyframe.exposure[:3, 3, None]
        # 裁剪到[0,1]范围并恢复形状
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        return render_pkg

    def render(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = torch.zeros(3, device="cuda"),
        top_view: bool = False,
        fov_x: float = None,
        fov_y: float = None,
    ):
        """
        【渲染模块】底层渲染函数

        使用3D高斯光栅化渲染图像和深度。支持自定义分辨率和视场角。
        这是所有渲染功能的底层实现。

        Args:
            width: 渲染图像宽度
            height: 渲染图像高度
            view_matrix: 视图矩阵（4x4，世界到相机坐标变换）
            scaling_modifier: 高斯尺度缩放因子（1.0=正常，>1.0=放大）
            bg: 背景颜色 [3]
            top_view: 是否为顶视图模式（用于可视化高斯位置）
            fov_x: 水平视场角（弧度，可选，默认使用场景内参）
            fov_y: 垂直视场角（弧度，可选，默认使用场景内参）

        Returns:
            dict: 包含渲染图像、逆深度、主要高斯ID、半径等信息的字典
        """
        # 计算相机中心（视图矩阵的逆矩阵的第4列前3行）
        cam_centre = view_matrix.detach().inverse()[3, :3]

        # ========== 内参设置 ==========
        # 如果没有提供自定义FOV，使用场景的默认内参
        if fov_x is None and fov_y is None:
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        # 如果提供了自定义FOV，计算对应的投影参数
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)
                .transpose(0, 1)
                .cuda()
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        # ========== 光栅化设置 ==========
        # 创建高斯光栅化器配置
        raster_settings = GaussianRasterizationSettings(
            height,
            width,
            tanfovx,
            tanfovy,
            bg,
            1 if top_view else scaling_modifier,  # 顶视图模式下不使用缩放
            projection_matrix,
            self.active_sh_degree,  # 当前使用的球谐函数阶数
            cam_centre,
            False,  # 是否使用预计算的颜色
            False,  # 其他选项
        )
        rasterizer = GaussianRasterizer(raster_settings)

        # ========== 多线程安全的高斯参数访问 ==========
        with self.lock:
            # 【场景表示模块】推理模式下混合多个锚点的高斯参数
            # 策略：根据相机中心到锚点的距离加权混合（用于大尺度场景）
            if self.inference_mode and not top_view:
                self.gaussian_params, self.anchor_weights = Anchor.blend(
                    cam_centre, self.anchors, self.anchor_overlap
                )

            # 屏幕空间点（用于计算2D位置，需要梯度用于优化）
            screenspace_points = torch.zeros_like(self.xyz, requires_grad=True)

            if self.xyz.shape[0] > 0:
                # ========== 顶视图模式 ==========
                # 顶视图：使用固定尺度和不透明度，便于可视化高斯点位置
                if top_view:
                    scaling = torch.ones_like(self.scaling) * scaling_modifier
                    opacity = torch.ones_like(self.opacity)
                else:
                    # 正常渲染：使用优化后的高斯参数
                    scaling = self.scaling
                    opacity = self.opacity

                # ========== 执行光栅化 ==========
                # 【渲染模块】调用底层CUDA光栅化器进行渲染
                color, invdepth, mainGaussID, radii = rasterizer(
                    self.xyz,  # 3D位置
                    screenspace_points,  # 屏幕空间位置（输出）
                    opacity,  # 不透明度
                    self.f_dc,  # 球谐函数DC项
                    self.f_rest,  # 球谐函数高阶项
                    scaling,  # 尺度
                    self.rotation,  # 旋转四元数
                    view_matrix,  # 视图矩阵
                )
            else:
                # ========== 空场景处理 ==========
                # 如果没有高斯点，返回空张量
                color = torch.zeros(3, height, width, device="cuda")
                invdepth = torch.zeros(1, height, width, device="cuda")
                mainGaussID = torch.zeros(
                    1, height, width, device="cuda", dtype=torch.int32
                )
                radii = torch.zeros(1, height, width, device="cuda")

        # 返回渲染结果字典
        return {
            "render": color,  # RGB图像 [3, H, W]
            "invdepth": invdepth,  # 逆深度 [1, H, W]
            "mainGaussID": mainGaussID,  # 每个像素的主要高斯点ID [1, H, W]
            "radii": radii,  # 每个高斯点在屏幕空间的半径 [1, H, W]
            "visibility_filter": radii > 0,  # 可见性掩码（用于自适应密度控制）
            "screenspace_points": screenspace_points,  # 屏幕空间位置（用于优化）
        }

    def get_closest_by_cam(self, cam_centre, k=3):
        """
        【场景表示模块】根据相机中心获取最近的k个锚点

        用于推理模式下选择需要混合的锚点。根据相机中心到锚点的距离排序。

        Args:
            cam_centre: 相机中心位置 [3]
            k: 返回的锚点数量

        Returns:
            tuple: (最近锚点列表, 锚点ID列表)
        """
        closest_anchors = []
        closest_anchors_ids = []
        offset = 0
        # 克隆相机中心列表（用于标记已选中的锚点）
        approx_cam_centres = self.approx_cam_centres.clone()

        # 迭代选择k个最近的锚点
        for l in range(min(k, len(self.anchors))):
            if approx_cam_centres.shape[0] == 0:
                break
            # 计算到所有相机中心的距离
            dists = torch.linalg.norm(approx_cam_centres - cam_centre[None], dim=-1)
            min_dist, min_id = torch.min(dists, dim=0)

            # 如果找到有效距离（<1e9表示未被标记）
            if min_dist < 1e9:
                # 查找对应的锚点
                for anchor_id, anchor in enumerate(self.anchors):
                    if min_id in anchor.keyframe_ids:
                        closest_anchors.append(anchor)
                        closest_anchors_ids.append(anchor_id)
                        # 标记该锚点的所有关键帧为已选中（设置为大值）
                        approx_cam_centres[
                            anchor.keyframe_ids[0] : anchor.keyframe_ids[-1] + 1
                        ] = 1e9
                        break

        return closest_anchors, closest_anchors_ids

    @torch.no_grad()
    def get_prev_keyframes(
        self,
        n: int,
        update_3dpts: bool,
        desc_kpts: DescribedKeypoints = None,
        resolution_mode: str = "baseline",
        exclude_pose_quarantined: bool = False,
    ):
        """
        【场景表示模块】获取最近的n个关键帧

        用于深度估计和匹配。如果提供了特征点描述符，会基于特征匹配数量选择关键帧；
        否则基于空间距离选择。

        Args:
            n: 要返回的关键帧数量
            update_3dpts: 是否更新关键帧的3D点（重新三角化）
            desc_kpts: 特征点描述符（可选，用于基于匹配选择关键帧）

        Returns:
            list[Keyframe]: 最近的n个关键帧列表
        """
        # ========== 确保优化线程已停止 ==========
        # 避免在多线程环境下访问关键帧数据时出现冲突
        self.join_optimization_thread()

        # ========== 关键帧选择策略 ==========
        # 如果提供了特征点描述符，基于特征匹配数量选择关键帧
        if desc_kpts is not None and len(self.keyframes) > n:
            # 在搜索窗口内查找匹配数量最多的关键帧
            n_ckecks = min(self.num_prev_keyframes_check, len(self.keyframes))
            base_indices = [int(x) for x in self.sorted_frame_indices[:n_ckecks]]
            keyframes_indices_to_check = list(base_indices)
            support_indices = []
            if self.risk_admission_mode != "off" and self.recovery_commit_bridge == "true_source_commit":
                support_indices = [
                    int(i)
                    for i, keyframe in enumerate(self.keyframes)
                    if bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
                ]
                for index in support_indices:
                    if index not in keyframes_indices_to_check:
                        keyframes_indices_to_check.append(index)
            bridge_meta: dict[str, object] = {}
            bridge_ids: list[int] = []
            bridge = getattr(self, "defer_recovery_bridge", None)
            if resolution_mode == "paper_aligned_true_recovery" and bridge is not None:
                bridge_ids, bridge_meta = bridge.extend_recovery_candidates(
                    self, desc_kpts, keyframes_indices_to_check, int(getattr(self, "_recovery_source_frame_id", -1))
                )
                for idx in bridge_ids:
                    if int(idx) not in keyframes_indices_to_check:
                        keyframes_indices_to_check.append(int(idx))
                bridge.mark_bridge_keyframes(self, bridge_ids, bridge_meta)
                self._last_recovery_bridge_meta = dict(bridge_meta)
            if resolution_mode == "paper_aligned_true_recovery":
                has_pt3d_ranked = sorted(
                    [
                        (int(i), int(self.keyframes[int(i)].desc_kpts.has_pt3d.sum().item()))
                        for i in range(len(self.keyframes))
                    ],
                    key=lambda item: -item[1],
                )
                for index, _score in has_pt3d_ranked[: max(n * 2, 12)]:
                    if index not in keyframes_indices_to_check:
                        keyframes_indices_to_check.append(index)
            keyframes_indices_to_check = filter_pose_reference_indices(
                self.keyframes,
                keyframes_indices_to_check,
                enabled=bool(exclude_pose_quarantined),
            )
            n_matches = torch.zeros(len(keyframes_indices_to_check), device="cuda")
            has_pt3d_counts = torch.zeros(len(keyframes_indices_to_check), device="cuda")
            # 计算每个候选关键帧的匹配数量
            for i, index in enumerate(keyframes_indices_to_check):
                n_matches[i] = self.matcher.evaluate_match(
                    self.keyframes[index].desc_kpts, desc_kpts
                )
                has_pt3d_counts[i] = float(self.keyframes[int(index)].desc_kpts.has_pt3d.sum().item())
            selection_scores = n_matches
            if resolution_mode == "paper_aligned_true_recovery":
                match_norm = n_matches / n_matches.max().clamp_min(1.0)
                pt3d_norm = has_pt3d_counts / has_pt3d_counts.max().clamp_min(1.0)
                selection_scores = match_norm * 0.35 + pt3d_norm * 0.65
            # 选择匹配数量最多的n个关键帧
            top_count = min(n, len(keyframes_indices_to_check))
            _, top_indices = torch.topk(selection_scores, top_count)
            selected_indices = [keyframes_indices_to_check[int(i)] for i in top_indices.cpu()]
            sorted_match_ids = torch.argsort(selection_scores, descending=True).cpu().tolist()
            rank_by_index = {
                keyframes_indices_to_check[int(rank_idx)]: rank + 1
                for rank, rank_idx in enumerate(sorted_match_ids)
            }
            promotion_applied = False
            promotion_reason = ""
            if self.risk_admission_mode != "off" and self.recovery_commit_bridge == "true_source_commit":
                selected_set = set(int(x) for x in selected_indices)
                support_scored = [
                    (int(index), float(n_matches[i].item()))
                    for i, index in enumerate(keyframes_indices_to_check)
                    if int(index) in set(support_indices)
                ]
                unselected_support = [
                    (index, score) for index, score in support_scored if index not in selected_set
                ]
                if unselected_support and selected_indices:
                    best_support_index, best_support_score = max(unselected_support, key=lambda item: item[1])
                    median_score = float(torch.median(n_matches).item()) if len(n_matches) > 0 else 0.0
                    selected_scores = {
                        int(index): float(n_matches[keyframes_indices_to_check.index(int(index))].item())
                        for index in selected_indices
                    }
                    lowest_selected = min(selected_indices, key=lambda index: selected_scores[int(index)])
                    direct_support_insufficient = selected_scores[int(lowest_selected)] < float(
                        getattr(self, "min_num_inliers", 0) or 0
                    )
                    if best_support_score >= median_score or direct_support_insufficient:
                        selected_indices = [int(x) for x in selected_indices]
                        selected_indices[selected_indices.index(int(lowest_selected))] = int(best_support_index)
                        promotion_applied = True
                        promotion_reason = (
                            "support_score_above_candidate_median"
                            if best_support_score >= median_score
                            else "direct_reference_support_insufficient"
                        )
            prev_keyframes_indices = torch.tensor(selected_indices, device="cpu", dtype=torch.long)
            final_selected = set(int(x) for x in prev_keyframes_indices.tolist())
            candidate_rows = []
            for i, index in enumerate(keyframes_indices_to_check):
                keyframe = self.keyframes[int(index)]
                commit_origin = str(keyframe.info.get("_paper_aligned_commit_origin", "unknown"))
                is_recovery = commit_origin in {"true_recovery_commit", "early_seed_recovery_commit"}
                candidate_rows.append(
                    {
                        "candidate_keyframe_id": int(keyframe.index),
                        "candidate_source_frame_id": int(
                            keyframe.info.get("_paper_aligned_source_frame_id", keyframe.index)
                        ),
                        "candidate_commit_origin": commit_origin,
                        "candidate_is_recovery": bool(is_recovery),
                        "candidate_is_early_seed": bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False)),
                        "candidate_is_support_eligible": bool(
                            keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False)
                        ),
                        "candidate_rank_before_filter": int(rank_by_index.get(int(index), 0)),
                        "candidate_rank_after_filter": int(
                            list(prev_keyframes_indices.tolist()).index(int(index)) + 1
                            if int(index) in final_selected
                            else 0
                        ),
                        "candidate_selected": bool(int(index) in final_selected),
                        "filter_reason": "" if int(index) in final_selected else "topk_not_selected",
                        "promotion_applied": bool(promotion_applied and int(index) in final_selected),
                        "promotion_reason": promotion_reason if promotion_applied and int(index) in final_selected else "",
                        "reference_support_score": float(
                            selection_scores[i].item()
                            if resolution_mode == "paper_aligned_true_recovery"
                            else n_matches[i].item()
                        ),
                        "reference_has_pt3d_count": int(has_pt3d_counts[i].item()),
                        "reference_raw_match_count": float(n_matches[i].item()),
                    }
                )
            self.last_prev_keyframes_debug = {
                "candidate_rows": candidate_rows,
                "selected_keyframe_ids": [int(x) for x in prev_keyframes_indices.tolist()],
                "promotion_applied": bool(promotion_applied),
                "promotion_reason": promotion_reason,
                "bridge_meta": bridge_meta,
                "bridge_reference_ids": list(bridge_ids),
            }
        # 如果没有提供特征点描述符，直接选择距离最近的n个关键帧
        else:
            pose_candidate_indices = filter_pose_reference_indices(
                self.keyframes,
                [int(index) for index in self.sorted_frame_indices],
                enabled=bool(exclude_pose_quarantined),
            )
            prev_keyframes_indices = pose_candidate_indices[:n]
            self.last_prev_keyframes_debug = {
                "candidate_rows": [],
                "selected_keyframe_ids": [int(x) for x in prev_keyframes_indices],
                "promotion_applied": False,
                "promotion_reason": "",
            }
        prev_keyframes = [self.keyframes[i] for i in prev_keyframes_indices]

        # ========== 更新3D点 ==========
        # 如果需要，重新三角化关键帧的3D点（用于深度对齐）
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes, resolution_mode=resolution_mode)
        return prev_keyframes

    def get_Rts(self):
        """
        【场景表示模块】获取所有关键帧的位姿矩阵（带缓存）

        返回缓存的位姿矩阵，如果缓存失效则重新计算。
        用于提高渲染和评估时的性能。

        Returns:
            torch.Tensor: 所有关键帧的位姿矩阵 [N, 4, 4]
        """
        # 查找缓存失效的关键帧ID
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            # 重新计算失效的位姿并更新缓存
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_gt_Rts(self, align):
        """
        【评估模块】获取真实位姿矩阵

        Args:
            align: 是否对齐到优化后的位姿（用于计算误差）

        Returns:
            torch.Tensor: 真实位姿矩阵 [N, 4, 4]
        """
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        # 如果需要对齐，计算相似变换将真实位姿对齐到优化位姿
        if align and n_poses > 0:
            Rts, gt_Rts = select_pose_eval_pairs(
                self.get_Rts(),
                self.gt_Rts,
                self.gt_Rts_mask,
            )
            if len(Rts) == 0:
                return self.gt_Rts[:0]
            return align_poses(gt_Rts, Rts)
        else:
            return self.gt_Rts

    def make_dummy_ext_tensor(self):
        """
        【优化模块】创建空的高斯参数张量字典

        用于剪枝操作（只移除高斯点，不添加新点）。

        Returns:
            dict: 空的高斯参数字典（所有张量的第一维为0）
        """
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
        }

    def reset(self, keyframe_id: int = -1):
        """
        【优化模块】移除指定关键帧中可见的高斯点

        用于重置场景的特定区域（例如，当关键帧位姿发生大幅变化时）。

        Args:
            keyframe_id: 关键帧索引（-1表示最新关键帧）
        """
        # 初始掩码：保留不透明度>0.05的高斯点
        valid_mask = self.opacity[:, 0] > 0.05
        # 渲染关键帧，获取可见性掩码
        render_pkg = self.render_from_id(keyframe_id)
        # 将可见的高斯点标记为无效（移除）
        valid_mask[render_pkg["visibility_filter"]] = False
        # 执行剪枝（不添加新点，只移除无效点）
        self.optimizer.add_and_prune(self.make_dummy_ext_tensor(), valid_mask)

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """
        【场景表示模块】为新关键帧初始化3D高斯点

        这是高斯点云增长的核心函数，执行以下步骤：
        1. 对齐关键帧的单目深度到三角化深度
        2. 基于Laplacian概率采样候选像素位置
        3. 使用引导MVS估计深度
        4. 初始化高斯参数（位置、颜色、尺度、不透明度等）
        5. 剪枝遮挡和过大的高斯点

        Args:
            keyframe_id: 关键帧索引（-1表示最新关键帧）
        """
        keyframe = self.keyframes[keyframe_id]

        # ========== 深度对齐 ==========
        # 如果关键点还没有3D点，先进行三角化
        if keyframe.desc_kpts.has_pt3d.sum() == 0:
            resolution_mode = (
                "paper_aligned_true_recovery"
                if bool(keyframe.info.get("_paper_aligned_insertion_type") == "true_recovery_commit")
                else "baseline"
            )
            keyframe.update_3dpts(self.keyframes, resolution_mode=resolution_mode)
        # 【场景表示模块】对齐单目深度到三角化深度（通过缩放和偏移）
        keyframe.align_depth()

        # 测试关键帧不添加高斯点（仅用于评估）
        if keyframe.info["is_test"]:
            return

        # ========== 基于Laplacian的概率采样 ==========
        # 【场景表示模块】计算每个像素成为新高斯点的概率
        # 策略：在图像边缘/纹理丰富区域（高Laplacian）更可能添加高斯点
        img = keyframe.image_pyr[0]
        img = F.avg_pool2d(img, 2)  # 轻微下采样以减少噪声
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        init_proba = get_lapla_norm(img, self.disc_kernel)  # 公式1：Laplacian范数作为概率

        # 应用掩码（排除无效区域）
        if keyframe.mask_pyr is not None:
            dilated_mask = (
                F.conv2d(
                    keyframe.mask_pyr[0][None].float(), self.disc_kernel, padding="same"
                )[0, 0]
                >= 0.99
            )
            init_proba *= dilated_mask

        # ========== 渲染惩罚机制 ==========
        # 【场景表示模块】计算惩罚项：如果已有高斯点能很好地渲染该区域，则降低添加新点的概率
        # 这避免了在已有良好表示的区域重复添加高斯点
        penalty = 0
        rendered_depth = None
        render_residual_edge = None
        if self.xyz.shape[0] > 0:
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"]
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
            render_support = render_pkg["mainGaussID"][0] >= 0
            if keyframe.mask_pyr is not None:
                keyframe_mask = keyframe.mask_pyr[0]
                if keyframe_mask.ndim == 3:
                    keyframe_mask = keyframe_mask[0]
                render_support = render_support & keyframe_mask.bool()
            render_coverage = float(render_support.float().mean().detach().cpu().item())
            keyframe.info[TEXTURE_SAMPLING_COVERAGE_KEY] = {
                "coverage": render_coverage,
                "coverage_deficit": max(0.0, 1.0 - render_coverage),
            }
            penalty = get_lapla_norm(render, self.disc_kernel)  # 渲染图像的Laplacian作为惩罚
            texture_sampling_scene_guard = None
            if self.pose_render_texture_sampling in {
                TEXTURE_SAMPLING_RESPONSE_GUARD_MODE,
                TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE,
                TEXTURE_SAMPLING_RESPONSE_GUARD_NON_DARK_MODE,
                TEXTURE_SAMPLING_RESPONSE_GUARD_MASK_CONSERVATIVE_MODE,
            }:
                texture_sampling_scene_guard = (
                    self._pose_render_texture_sampling_scene_guard()
                )
                keyframe.info[TEXTURE_SAMPLING_SCENE_GUARD_KEY] = (
                    texture_sampling_scene_guard
                )
            coverage_bypass_latched = bool(
                isinstance(texture_sampling_scene_guard, dict)
                and texture_sampling_scene_guard.get("coverage_bypass_latched", False)
            )
            if not (
                self.pose_render_texture_sampling
                == TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE
                and coverage_bypass_latched
            ):
                render_residual = (render.detach() - img).abs().mean(dim=0)
                render_residual_edge = 0.65 * render_residual + 0.35 * get_lapla_norm(
                    render_residual[None], self.disc_kernel
                )

        # ========== 采样掩码生成 ==========
        # 公式3：最终采样概率 = init_proba - penalty
        # 在纹理丰富且渲染质量差的区域添加新高斯点
        init_proba *= self.init_proba_scaler
        penalty *= self.init_proba_scaler
        sample_proba = (init_proba - penalty).clamp_min(0.0)
        if render_residual_edge is not None:
            self._attach_training_background_info(keyframe)
            self._attach_pose_render_scene_low_response_info(keyframe)
            sample_proba, texture_sampling_debug = residual_edge_guided_sampling_probability(
                sample_proba,
                render_residual_edge,
                mode=self.pose_render_texture_sampling,
                direct_density_mode=self.paper_aligned_direct_density_control,
                render_frame_policy=self.paper_aligned_render_frame_policy,
                keyframe_info=keyframe.info,
                alpha=self.pose_render_texture_sampling_alpha,
                min_selectivity=self.pose_render_texture_sampling_min_selectivity,
            )
            if self.pose_render_texture_sampling != "off":
                keyframe.info["_paper_aligned_pose_render_texture_sampling"] = texture_sampling_debug
            self._record_pose_render_texture_sampling(texture_sampling_debug)
        elif (
            self.pose_render_texture_sampling == TEXTURE_SAMPLING_RESPONSE_GUARD_BYPASS_MODE
            and bool(
                keyframe.info.get(TEXTURE_SAMPLING_SCENE_GUARD_KEY, {}).get(
                    "coverage_bypass_latched", False
                )
            )
        ):
            coverage = keyframe.info.get(TEXTURE_SAMPLING_COVERAGE_KEY, {})
            texture_sampling_debug = {
                "mode": self.pose_render_texture_sampling,
                "direct_density_mode": self.paper_aligned_direct_density_control,
                "render_frame_policy": self.paper_aligned_render_frame_policy,
                "applied": False,
                "reason": "coverage_bypass_latched",
                "alpha": 0.0,
                "selectivity": 0.0,
                "coverage": float(coverage.get("coverage", 0.0) or 0.0),
                "coverage_deficit": float(
                    coverage.get("coverage_deficit", coverage.get("deficit", 0.0))
                    or 0.0
                ),
                "coverage_bypass_latched": True,
            }
            keyframe.info["_paper_aligned_pose_render_texture_sampling"] = (
                texture_sampling_debug
            )
            self._record_pose_render_texture_sampling(texture_sampling_debug)
        sample_mask = torch.rand_like(init_proba) < sample_proba

        # ========== 深度估计 ==========
        sampled_uv = self.uv[sample_mask]  # 采样像素坐标

        # 【场景表示模块】使用引导多视图立体匹配（Guided MVS）估计深度
        # 策略：利用历史关键帧的密集特征进行立体匹配
        prev_KFs = self.get_prev_keyframes(
            self.guided_mvs.n_cams + 1, update_3dpts=False
        )
        for i, prev_keyframe in enumerate(prev_KFs):
            if keyframe.index == prev_keyframe.index:
                prev_KFs.pop(i)
                break
        depth, accurate_mask = self.guided_mvs(sampled_uv, keyframe, prev_KFs)

        # 过滤：保留置信度高且深度有效的点
        valid_mask = (keyframe.sample_conf(sampled_uv) > 0.5) * (depth > 1e-6)
        sample_mask[sample_mask.clone()] = valid_mask
        depth = depth[valid_mask]
        sampled_uv = sampled_uv[valid_mask]
        accurate_mask = accurate_mask[valid_mask]

        # ========== 剪枝过粗的高斯点 ==========
        # 【场景表示模块】如果新点比现有高斯点更精细，则移除过粗的旧高斯点
        # 策略：统计每个旧高斯点被新点覆盖的次数，如果超过阈值则移除
        if len(self.xyz) > 0:
            main_gaussians_map = render_pkg["mainGaussID"]  # 每个像素对应的主要高斯点ID
            accurate_sample_mask = sample_mask.clone()
            accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
            selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
            ids, counts = torch.unique(
                selected_main_gaussians[selected_main_gaussians >= 0],
                return_counts=True,
            )
            valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            valid_gs_mask[ids] = counts < 10  # 被覆盖次数少于10次的保留
            with self.lock:
                self.optimizer.add_and_prune(
                    self.make_dummy_ext_tensor(), valid_gs_mask
                )
            render_pkg = self.render_from_id(keyframe_id)
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

        # ========== 遮挡检查 ==========
        # 【场景表示模块】移除被现有高斯点遮挡的新点（避免重复表示）
        if rendered_depth is not None:
            valid_mask = depth < rendered_depth[sample_mask]  # 新点深度必须小于渲染深度
            sample_mask[sample_mask.clone()] = valid_mask
            depth = depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]

        # ========== 3D位置初始化 ==========
        # 【场景表示模块】将像素坐标+深度转换为世界坐标系3D点
        new_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
        new_pts = (new_pts - keyframe.get_t()) @ keyframe.get_R()  # 转换到世界坐标系

        # 添加从特征匹配三角化得到的3D点（这些点通常更准确）
        match_pts = keyframe.desc_kpts.pts3d[keyframe.desc_kpts.has_pt3d]
        new_pts = torch.cat([new_pts, match_pts], dim=0)

        # ========== 颜色初始化 ==========
        # 【场景表示模块】从图像中采样颜色并转换为球谐函数表示
        f_dc = img[:, sample_mask]  # 采样像素的颜色
        match_sampler = keyframe.desc_kpts.kpts[keyframe.desc_kpts.has_pt3d]
        match_sampler = make_torch_sampler(match_sampler, self.width, self.height)
        match_colors = F.grid_sample(
            img[None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(3, -1)
        f_dc = torch.cat([f_dc, match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))  # RGB转球谐函数DC项

        # ========== 尺度初始化 ==========
        # 【场景表示模块】根据初始化概率和到相机距离设置高斯尺度
        # 公式4：尺度与初始化概率的平方根成反比，并考虑距离
        sampled_init_proba = init_proba[sample_mask]
        match_init_proba = F.grid_sample(
            init_proba[None, None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(-1)
        sampled_init_proba = torch.cat([sampled_init_proba, match_init_proba], dim=0)
        # 期望到最近邻的距离（公式4）
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)  # 限制尺度范围
        # 根据到相机中心的距离缩放
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            new_pts - keyframe.approx_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)  # log空间

        # ========== 不透明度初始化 ==========
        # 【场景表示模块】根据深度估计精度设置初始不透明度
        opacities = torch.ones(f_dc.shape[0], 1, device="cuda")
        # 不准确的点使用较低的不透明度（0.02），准确的点使用稍高的不透明度（0.07）
        opacities[: sampled_uv.shape[0]] *= (
            0.07 * accurate_mask[..., None] + 0.02 * ~accurate_mask[..., None]
        )
        # 三角化得到的点使用更高的不透明度（0.2）
        opacities[sampled_uv.shape[0] :] *= 0.2
        init_weighting_events = int(
            self.pose_render_init_weighting_stats.get("events", 0)
        )
        if init_weighting_events > 0:
            init_weighting_pose_mean = float(
                self.pose_render_init_weighting_stats.get("pose_risk_sum", 0.0)
            ) / float(init_weighting_events)
            init_weighting_novelty_mean = float(
                self.pose_render_init_weighting_stats.get("novelty_value_sum", 0.0)
            ) / float(init_weighting_events)
            init_weighting_risk_alpha_mean = float(
                self.pose_render_init_weighting_stats.get("risk_alpha_sum", 0.0)
            ) / float(init_weighting_events)
        else:
            init_weighting_pose_mean = 0.0
            init_weighting_novelty_mean = 0.0
            init_weighting_risk_alpha_mean = 0.0
        init_weighting_debug = pose_render_init_weighting_decision(
            mode=self.pose_render_init_weighting,
            direct_density_mode=self.paper_aligned_direct_density_control,
            keyframe_info=keyframe.info,
            min_pose_risk=self.pose_render_init_weighting_min_pose_risk,
            max_pose_risk=self.pose_render_init_weighting_max_pose_risk,
            min_sample_opacity_scale=self.pose_render_init_weighting_min_sample_opacity_scale,
            min_match_opacity_scale=self.pose_render_init_weighting_min_match_opacity_scale,
            max_representation_value=self.pose_render_init_weighting_max_representation_value,
            max_novelty_value=self.pose_render_init_weighting_max_novelty_value,
            adaptive_novelty_threshold=(
                self.pose_render_init_weighting_adaptive_novelty_threshold
            ),
            adaptive_risk_override_alpha=(
                self.pose_render_init_weighting_adaptive_risk_override_alpha
            ),
            adaptive_min_pose_risk=(
                self.pose_render_init_weighting_adaptive_min_pose_risk
            ),
            adaptive_max_pose_risk=(
                self.pose_render_init_weighting_adaptive_max_pose_risk
            ),
            adaptive_min_sample_opacity_scale=(
                self.pose_render_init_weighting_adaptive_min_sample_opacity_scale
            ),
            adaptive_min_match_opacity_scale=(
                self.pose_render_init_weighting_adaptive_min_match_opacity_scale
            ),
            adaptive_scene_event_count=init_weighting_events,
            adaptive_scene_pose_risk_mean=init_weighting_pose_mean,
            adaptive_scene_novelty_mean=init_weighting_novelty_mean,
            adaptive_scene_risk_alpha_mean=init_weighting_risk_alpha_mean,
            adaptive_scene_min_events=(
                self.pose_render_init_weighting_adaptive_scene_min_events
            ),
            adaptive_scene_gentle_pose_risk_mean=(
                self.pose_render_init_weighting_adaptive_scene_gentle_pose_risk_mean
            ),
            adaptive_scene_gentle_novelty_mean=(
                self.pose_render_init_weighting_adaptive_scene_gentle_novelty_mean
            ),
            adaptive_scene_max_risk_alpha_mean=(
                self.pose_render_init_weighting_adaptive_scene_max_risk_alpha_mean
            ),
            adaptive_scene_low_pose_risk_mean=(
                self.pose_render_init_weighting_adaptive_scene_low_pose_risk_mean
            ),
            adaptive_scene_low_novelty_mean=(
                self.pose_render_init_weighting_adaptive_scene_low_novelty_mean
            ),
            adaptive_scene_low_risk_min_events=(
                self.pose_render_init_weighting_adaptive_scene_low_risk_min_events
            ),
            adaptive_scene_stable_low_risk_min_events=(
                self.pose_render_init_weighting_adaptive_scene_stable_low_risk_min_events
            ),
            adaptive_scene_stable_low_pose_risk_mean=(
                self.pose_render_init_weighting_adaptive_scene_stable_low_pose_risk_mean
            ),
            adaptive_scene_stable_low_novelty_mean=(
                self.pose_render_init_weighting_adaptive_scene_stable_low_novelty_mean
            ),
            adaptive_scene_stable_low_risk_latched=bool(
                self.pose_render_init_weighting_stats.get(
                    "adaptive_scene_stable_low_risk_latched", 0
                )
            ),
            adaptive_scene_high_uncertainty_bypass_risk_alpha_mean=(
                self.pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_risk_alpha_mean
            ),
            adaptive_scene_high_uncertainty_bypass_novelty_mean=(
                self.pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_novelty_mean
            ),
        )
        if bool(init_weighting_debug.get("applied", False)):
            opacities[: sampled_uv.shape[0]] *= float(
                init_weighting_debug.get("sample_opacity_scale", 1.0)
            )
            opacities[sampled_uv.shape[0] :] *= float(
                init_weighting_debug.get("match_opacity_scale", 1.0)
            )
        if self.pose_render_init_weighting != "off":
            keyframe.info["_paper_aligned_pose_render_init_weighting"] = (
                init_weighting_debug
            )
            self._record_pose_render_init_weighting(init_weighting_debug)
        opacities = inverse_sigmoid(opacities)  # 转换到logit空间

        # ========== 其他参数初始化 ==========
        # 【场景表示模块】球谐函数高阶项初始化为0（视角相关颜色）
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device="cuda",
        )
        # 旋转初始化为单位四元数（无旋转）
        rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
        rots[:, 0] = 1

        # ========== 剪枝策略 ==========
        # 【场景表示模块】确定哪些现有高斯点应该被剪枝
        if self.xyz.shape[0] > 0:
            # 只保留不透明度足够高的高斯点（>0.05）
            valid_gs_mask = self.opacity[:, 0] > 0.05

            # 移除在屏幕上过大的高斯点（可能是异常值）
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist  # 屏幕空间大小
            valid_gs_mask *= screen_size < 0.5 * self.width  # 屏幕大小不能超过图像宽度的一半
        else:
            valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)

        # ========== 添加新高斯点 ==========
        # 【优化模块】将新高斯点添加到优化器中，同时剪枝无效的旧高斯点
        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        with self.lock:
            self.optimizer.add_and_prune(extension_tensors, valid_gs_mask)

    def init_intrinsics(self):
        """
        【渲染模块】初始化相机内参

        根据焦距和图像尺寸计算视场角（FoV）和投影矩阵。
        用于光栅化渲染时的坐标变换。
        """
        # 计算水平和垂直视场角（弧度）
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        # 计算半视场角的正切值（用于光栅化）
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        # 计算投影矩阵（OpenGL格式，4x4）
        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)  # 转置为列主序（OpenGL格式）
            .cuda()
        )

    def move_rand_keyframe_to_cpu(self):
        """
        【内存管理模块】将随机关键帧移动到CPU内存

        当活跃关键帧数量超过限制时，将部分关键帧移到CPU以节省GPU内存。
        保留最后n_kept_frames个关键帧始终在GPU上。
        """
        # 从GPU关键帧中随机选择一个（排除最后n_kept_frames个）
        frame_id = np.random.choice(self.active_frames_gpu[:-self.n_kept_frames])
        self.keyframes[frame_id].to("cpu")
        self.active_frames_cpu.append(frame_id)
        self.active_frames_gpu.remove(frame_id)

    def move_rand_keyframe_to_gpu(self):
        """
        【内存管理模块】将随机关键帧移动到GPU内存

        当需要更多关键帧参与训练时，从CPU加载关键帧到GPU。
        """
        if len(self.active_frames_cpu) > 0:
            frame_id = np.random.choice(self.active_frames_cpu)
            self.keyframes[frame_id].to("cuda")
            self.active_frames_gpu.insert(0, frame_id)  # 插入到列表开头（优先使用）
            self.active_frames_cpu.remove(frame_id)

    def add_keyframe(self, keyframe: Keyframe, f=None):
        """
        【场景表示模块】添加新关键帧到场景

        这是场景增长的核心函数，执行以下操作：
        1. 将关键帧添加到列表并更新索引
        2. 更新相机内参（如果提供新的焦距）
        3. 更新位姿缓存
        4. 将关键帧添加到活跃锚点
        5. 管理GPU/CPU内存（当关键帧过多时）

        Args:
            keyframe: 要添加的关键帧对象
            f: 新的焦距值（可选，如果提供则更新内参）
        """

        # ========== 确保训练线程已停止 ==========
        # 避免在添加关键帧时与优化线程冲突
        self.join_optimization_thread()

        # ========== 添加关键帧并更新索引 ==========
        # 将关键帧添加到列表
        self.keyframes.append(keyframe)
        # 更新近似相机中心列表（用于距离计算和锚点选择）
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        # 计算所有关键帧到最新关键帧的距离，并排序（用于匹配搜索）
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        # ========== 更新内参 ==========
        # 如果提供了新焦距，更新内参（用于处理变焦或焦距估计变化）
        if f is not None:
            self.f = f.item()
            self.init_intrinsics()

        # ========== 更新位姿缓存 ==========
        # 为新关键帧添加位姿缓存条目（初始标记为有效）
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device="cuda", dtype=torch.bool)], dim=0
        )
        # 如果有真实位姿（用于评估），添加到gt_Rts
        gt_pose = keyframe.info.get("Rt", None)
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0,
        )
        self.gt_f = keyframe.info.get("focal", self.f)

        # ========== 训练模式下的额外操作 ==========
        if not self.inference_mode:
            # 将关键帧添加到活跃锚点
            self.active_anchor.add_keyframe(keyframe)
            self.active_frames_gpu.append(keyframe.index)

            # ========== 内存管理 ==========
            # 如果活跃关键帧数量超过限制，将部分关键帧移到CPU
            if len(self.active_frames_gpu) > self.max_active_keyframes:
                self.move_rand_keyframe_to_cpu()
                # 每5个关键帧重排一次（保持GPU/CPU关键帧的平衡）
                if len(self.active_frames_cpu) % 5 == 0:
                    self.move_rand_keyframe_to_cpu()
                    self.move_rand_keyframe_to_gpu()
                    # 清理内存碎片
                    gc.collect()
                    torch.cuda.empty_cache()

    def enable_inference_mode(self):
        """
        【场景表示模块】启用推理模式

        切换到推理模式（停止训练），并更新锚点位置为活跃关键帧的平均位置。
        用于完成训练后的场景渲染。
        """
        self.inference_mode = True
        self.update_anchor()

    def update_anchor(self, n_left_frames: int = 0):
        """
        【场景表示模块】更新锚点位置

        将锚点位置设置为活跃关键帧相机中心的平均值，并可选地移除最后n_left_frames个关键帧。
        用于锚点固定（在创建新锚点前）。

        Args:
            n_left_frames: 要从活跃锚点移除的关键帧数量（从末尾移除）
        """
        # 计算活跃关键帧的相机中心平均值（排除最后n_left_frames个）
        anchor_position = self.approx_cam_centres[
            self.first_active_frame : self.last_active_frame - n_left_frames
        ].mean(dim=0)
        self.active_anchor.position = anchor_position
        # 如果指定了要移除的关键帧数量，从锚点中移除
        if n_left_frames > 0:
            self.active_anchor.keyframes = self.active_anchor.keyframes[:-n_left_frames]
            self.active_anchor.keyframe_ids = self.active_anchor.keyframe_ids[
                :-n_left_frames
            ]

    def place_anchor_if_needed(self):
        """
        【场景表示模块】根据高斯点大小判断是否需要创建新锚点

        当大部分高斯点在屏幕上显示很小时（大尺度场景），创建新锚点并合并细小的高斯点。
        这是大尺度场景管理的关键函数。

        策略：
        1. 检查屏幕空间大小<1的高斯点比例
        2. 如果超过阈值，固定当前锚点并创建新锚点
        3. 合并细小的高斯点（减少点数，提高效率）
        """
        small_prop_thresh = 0.4  # 细小高斯点比例阈值（超过40%则创建锚点）
        k = 3  # 每个高斯点合并的最近邻数量
        self.n_kept_frames = 20  # 在新锚点中保留的关键帧数量
        if (
            self.xyz.shape[0] > 0
            and self.first_active_frame < len(self.keyframes) - 2 * self.n_kept_frames
        ):
            with torch.no_grad():
                dist = torch.linalg.vector_norm(
                    self.xyz - self.approx_cam_centres[-1][None], dim=-1
                )
                screen_size = self.f * self.scaling.mean(dim=-1) / dist
                small_mask = screen_size < 1
                small_prop = small_mask.float().mean()

            if small_prop > small_prop_thresh:
                with torch.no_grad():
                    # 扩大细小高斯点的掩码（屏幕大小<1.5）
                    small_mask = screen_size < 1.5
                    # 【场景表示模块】更新锚点位置（使用用于优化的相机位姿）
                    # 固定当前锚点，移除最后n_kept_frames个关键帧（这些将用于新锚点）
                    self.update_anchor(self.n_kept_frames)

                    # ========== 合并细小高斯点 ==========
                    # 【优化模块】选择需要合并的高斯点及其最近邻
                    # 提取所有细小高斯点的参数
                    small_gaussians = {
                        name: self.gaussian_params[name]["val"][small_mask]
                        for name in self.gaussian_params
                    }
                    xyz = small_gaussians["xyz"].contiguous()
                    # 使用KNN查找每个细小高斯点的k个最近邻
                    _, nn_idx = distIndex2(xyz, k)
                    nn_idx = nn_idx.view(-1, k)
                    # 随机选择部分高斯点作为合并中心（每个组包含k+1个高斯点）
                    perm = torch.randperm(xyz.shape[0], device=xyz.device)
                    idx = perm[: (xyz.shape[0] // (k + 1))]
                    # 每个合并组包含：1个中心点 + k个最近邻点
                    selected_nn_idx = torch.cat([idx[..., None], nn_idx[idx]], dim=-1)

                    # ========== 计算合并权重 ==========
                    # 【优化模块】基于高斯点对渲染的贡献计算合并权重
                    # 权重 = 不透明度 * 屏幕空间大小的平方（表示渲染贡献）
                    weights = self.gaussian_params["opacity"]["val"][
                        selected_nn_idx, 0
                    ].sigmoid() * (screen_size[selected_nn_idx] ** 2)
                    # 归一化权重（使每个组的权重和为1）
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                    weights.unsqueeze_(-1)

                    # ========== 合并高斯点参数 ==========
                    # 【场景表示模块】通过加权平均合并高斯点参数
                    # 位置：加权平均
                    # 颜色（球谐函数）：加权平均
                    # 不透明度：加权平均（在logit空间）
                    # 尺度：加权平均（在指数空间），并考虑合并后的点数(k+1)
                    # 旋转：加权平均（四元数）
                    merged_gaussians = {
                        "xyz": (self.gaussian_params["xyz"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                        "f_dc": (self.gaussian_params["f_dc"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "f_rest": (self.gaussian_params["f_rest"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "opacity": inverse_sigmoid(self.gaussian_params["opacity"]['val'][selected_nn_idx, :].sigmoid() * weights).sum(dim=1),
                        "scaling": torch.log((torch.exp(self.gaussian_params["scaling"]['val'][selected_nn_idx, :]) * weights * (k+1)).sum(dim=1)),
                        "rotation": (self.gaussian_params["rotation"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                    }

                    # ========== 将旧锚点移到CPU ==========
                    # 【内存管理模块】复制参数字典并移动到CPU（释放GPU内存）
                    self.active_anchor.duplicate_param_dict()
                    self.active_anchor.to("cpu", with_keyframes=True)

                    # ========== 添加合并后的高斯点 ==========
                    # 【优化模块】将合并后的高斯点添加到优化器，同时移除细小高斯点
                    # ~small_mask: 保留非细小高斯点的掩码
                    with self.lock:
                        self.optimizer.add_and_prune(merged_gaussians, ~small_mask)

                    # ========== 创建新活跃锚点 ==========
                    # 【场景表示模块】使用合并后的高斯点创建新锚点
                    # 位置：最新关键帧的相机中心
                    # 关键帧：最近n_kept_frames个关键帧
                    self.active_anchor = Anchor(
                        self.gaussian_params,
                        self.approx_cam_centres[-1],
                        self.keyframes[-self.n_kept_frames :],
                    )
                    self.anchors.append(self.active_anchor)
                    # 更新活跃关键帧列表（新锚点的关键帧都在GPU上）
                    self.active_frames_gpu = [kf.index for kf in self.active_anchor.keyframes]
                    self.active_frames_cpu = []

                    # ========== 可视化权重设置 ==========
                    # 设置锚点混合权重（仅新锚点权重为1，其他为0）
                    self.anchor_weights = np.zeros(len(self.anchors))
                    self.anchor_weights[-1] = 1.0

                gc.collect()
                torch.cuda.empty_cache()

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0):
        """
        【评估模块】保存场景模型到磁盘

        将完整的场景模型保存到指定路径，包括：
        1. 所有锚点的高斯点云（PLY格式）
        2. 场景元数据（JSON格式：配置、锚点位置、关键帧信息）
        3. 测试关键帧的渲染图像
        4. COLMAP格式的相机参数和图像信息

        Args:
            path: 保存路径（如果为空字符串则跳过保存，仅返回指标）
            reconstruction_time: 重建耗时（秒），用于计算FPS
            n_frames: 处理的关键帧数量，用于计算FPS

        Returns:
            dict: 包含场景统计信息（锚点数量、关键帧数量、时间、FPS、质量指标）的字典
        """
        # ========== 计算场景指标 ==========
        # 【评估模块】收集场景统计信息
        metrics = {
            "num anchors": len(self.anchors),  # 锚点数量
            "num keyframes": len(self.keyframes),  # 关键帧数量
        }
        # 如果提供了重建时间，计算FPS
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
        # 计算渲染质量指标（PSNR、SSIM、LPIPS、位姿误差）
        eval_metrics, frame_metric_rows = self.evaluate(
            True,
            True,
            True,
            return_frame_metrics=True,
            frame_metrics_output_dir=path,
        )
        metrics.update(eval_metrics)

        # 如果路径为空，跳过保存，仅返回指标
        if path == "":
            print("No path provided, skipping save")
            return metrics

        # ========== 保存锚点点云 ==========
        # 【场景表示模块】将每个锚点的高斯点云保存为PLY文件
        pcd_path = os.path.join(path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=True)
        for index, anchor in enumerate(self.anchors):
            anchor.save_ply(os.path.join(pcd_path, f"anchor_{index}.ply"))

        # ========== 保存场景元数据 ==========
        # 【场景表示模块】保存场景配置、锚点位置和关键帧信息
        metadata = {
            "config": {
                "width": self.width,  # 图像宽度
                "height": self.height,  # 图像高度
                "sh_degree": self.max_sh_degree,  # 球谐函数阶数
                "f": self.f,  # 焦距
                "test_exposure_harmonization": (
                    self.paper_aligned_test_exposure_harmonization
                ),
                "test_exposure_guard_max_delta": (
                    self.paper_aligned_test_exposure_guard_max_delta
                ),
                "test_render_calibration": self.paper_aligned_test_render_calibration,
            },
            "anchors": [
                {
                    "position": anchor.position.cpu().numpy().tolist(),  # 锚点位置（世界坐标系）
                }
                for anchor in self.anchors
            ],
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],  # 关键帧信息（位姿、曝光等）
            "pose_render_keyframe_sampling": self._pose_render_keyframe_sampling_summary(),
            "pose_render_texture_sampling": self._pose_render_texture_sampling_summary(),
            "test_exposure_adaptive": self.paper_aligned_test_exposure_adaptive_stats,
            "pose_render_edge_loss": self._pose_render_edge_loss_summary(),
            "pose_render_psnr_loss": self._pose_render_psnr_loss_summary(),
            "pose_render_extra_optimization": self._pose_render_extra_optimization_summary(),
            "pose_verification_async_pose_protection": dict(
                self.pose_verification_async_pose_protection_stats
            ),
            "pose_render_update_gate": self._pose_render_update_gate_summary(),
            "pose_render_pre_refine": self._pose_render_pre_refine_summary(),
            "pose_render_init_weighting": self._pose_render_init_weighting_summary(),
            "training_background": dict(self._ensure_training_background_stats()),
        }
        # 合并指标到元数据
        metadata = {**metrics, **metadata}

        # 将元数据保存为JSON文件
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        write_frame_metrics_csv(os.path.join(path, "frame_metrics.csv"), frame_metric_rows)

        # ========== 保存测试关键帧渲染图像 ==========
        # 【评估模块】渲染所有测试关键帧并保存图像（用于可视化结果）
        self.save_test_frames(os.path.join(path, "test_images"))

        # ========== 保存COLMAP格式数据 ==========
        # 【场景表示模块】将关键帧转换为COLMAP格式（用于兼容性）
        # COLMAP格式包含相机参数（内参、外参）和图像信息
        images = {}
        cameras = {}
        colmap_save_path = os.path.join(path, "colmap")
        os.makedirs(colmap_save_path, exist_ok=True)
        for index, keyframe in enumerate(self.keyframes):
            camera, image = keyframe.to_colmap(index)
            cameras[index] = camera
            images[index] = image
        # 使用COLMAP的二进制格式保存（.bin文件）
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        return metrics

    def get_closest_keyframe(
        self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        """
        【场景表示模块】根据位置获取最近的关键帧

        计算给定位置到所有关键帧相机中心的距离，返回最近的count个关键帧。
        用于基于空间位置的关键帧查询（例如，查找特定区域的关键帧）。

        Args:
            position: 查询位置（世界坐标系）[3]
            count: 要返回的关键帧数量（默认为1）

        Returns:
            list[Keyframe]: 最近的关键帧列表（按距离从近到远排序）
        """
        # 计算到所有关键帧相机中心的欧氏距离
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        # 选择距离最近的count个关键帧
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]

    def finetune_epoch(self):
        """
        【优化模块】遍历所有锚点并逐个优化

        这是微调阶段的核心函数，用于在初始训练完成后进一步细化场景质量。
        逐个加载每个锚点到GPU，对其关键帧进行一轮优化，然后保存并卸载。

        策略：
        1. 按顺序处理每个锚点
        2. 将锚点加载到GPU并设置为活跃锚点
        3. 遍历锚点的所有关键帧，对每个关键帧执行一次优化步骤
        4. 更新锚点参数并卸载到CPU（节省内存）

        注意：这是微调模式（finetuning=True），优化时会随机选择关键帧，而不是优先选择最新帧。
        """
        # 初始化锚点混合权重（全部设为0，优化时只激活当前锚点）
        self.anchor_weights = np.zeros(len(self.anchors))

        # 遍历所有锚点
        for anchor_id, anchor in enumerate(self.anchors):
            # ========== 激活当前锚点 ==========
            # 【场景表示模块】设置当前锚点为活跃锚点
            self.active_anchor = anchor
            # 将锚点及其关键帧加载到GPU
            anchor.to("cuda", with_keyframes=True)
            # 将锚点的高斯参数设置为当前优化参数
            self.gaussian_params = anchor.gaussian_params
            # 激活当前锚点的权重（用于多锚点混合）
            self.anchor_weights[anchor_id] = 1
            # 重置优化器（确保梯度正确计算）
            self.reset_optimizer()

            # 注释：可选的内存优化（将前一个锚点移到CPU）
            # 实际未启用，因为可能导致频繁的GPU-CPU数据传输
            # # Ensure other anchors are on cpu to save memory
            # if anchor_id >= 1:
            #     self.anchors[anchor_id-1].to("cpu", with_keyframes=True)

            # ========== 优化当前锚点 ==========
            # 【优化模块】遍历锚点的所有关键帧，对每个关键帧执行一次优化步骤
            # finetuning=True: 微调模式，随机选择关键帧（不偏向最新帧）
            for _ in range(len(anchor.keyframes)):
                self.optimization_step(finetuning=True)

            # ========== 保存并卸载锚点 ==========
            # 【场景表示模块】将优化后的高斯参数保存到锚点
            anchor.gaussian_params = self.gaussian_params
            # 取消激活当前锚点的权重
            self.anchor_weights[anchor_id] = 0
