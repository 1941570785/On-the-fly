# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 场景重建主训练脚本
# 参考：https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py


import os
import time
import atexit
import copy
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler
from args import get_args
from threading import Thread
from dataloaders.image_dataset import ImageDataset
from dataloaders.stream_dataset import StreamDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.mono_depth import MonoDepthEstimator
from scene.scene_model import SceneModel
from gaussianviewer import GaussianViewer
from webviewer.webviewer import WebViewer
from graphdecoviewer.types import ViewerMode
from utils import align_mean_up_fwd, increment_runtime
from paper_aligned_policy.config import (
    BASELINE_RENDER_LOCK_INTRA_FRAME_V22_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V23_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V24_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V53_PROFILE,
    BASELINE_RENDER_LOCK_INTRA_FRAME_V54_PROFILE,
    COUPLED_INNOVATION_MODE,
    apply_coupled_innovation_defaults,
)
from paper_aligned_policy.runtime_gate import PaperAlignedRuntimeGate
from paper_aligned_policy.viewpoint_coverage import build_viewpoint_coverage_event
from scene.pose_render_posterior_risk import (
    augment_pose_render_payload_with_posterior_risk,
)
from scene.pose_initialization_risk import PoseInitializationRiskGate
from scene.pose_risk_utility_admission import (
    PoseRiskUtilityAdmissionGate,
    pose_risk_candidate,
)
from scene.keyframe import pop_chosen_kfs_resolution_events

if __name__ == "__main__":
    """
    主训练脚本：实现基于3D高斯点云的实时场景重建流程
    
    整体流程：
    1. 初始化阶段：加载数据、初始化模块、启动可视化
    2. Bootstrap阶段：累积前N帧，进行初始姿态和焦距估计
    3. 增量重建阶段：逐帧处理，提取特征、匹配、估计姿态、初始化高斯、优化
    4. 保存阶段：保存重建结果和评估指标
    """
    # ========== 初始化阶段 ==========
    # 固定随机种子，保证实验结果可复现
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # 解析命令行参数（数据路径、训练超参、可视化选项等）
    args = get_args()

    risk_mode = getattr(args, "risk_admission_mode", "off") or "off"
    profile_mode = str(
        getattr(args, "paper_aligned_pose_render_assimilation_profile", "off") or "off"
    )
    if (
        risk_mode == COUPLED_INNOVATION_MODE
        and profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V24_PROFILE
    ):
        setattr(args, "risk_admission_mode", "off")
        risk_mode = "off"
        print(
            f"[risk_admission_mode={COUPLED_INNOVATION_MODE}] profile-only passthrough "
            f"{profile_mode} resolved runtime mode: {risk_mode}"
        )
    elif (
        risk_mode == COUPLED_INNOVATION_MODE
        and (
            profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V22_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V23_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V29_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V30_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V31_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V32_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V33_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V34_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V35_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V36_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V37_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V52_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V53_PROFILE
            or profile_mode == BASELINE_RENDER_LOCK_INTRA_FRAME_V54_PROFILE
        )
    ):
        cfg = apply_coupled_innovation_defaults(args)
        risk_mode = getattr(args, "risk_admission_mode", "off") or "off"
        print(
            f"[risk_admission_mode={COUPLED_INNOVATION_MODE}] profile-only preset "
            f"{cfg.pose_render_assimilation_profile} resolved runtime mode: {risk_mode}"
        )

    runtime_gate = None
    if risk_mode != "off":
        print(
            f"[risk_admission_mode={risk_mode}] contract runtime gate enabled "
            "(off mode remains baseline path)."
        )
        runtime_gate = PaperAlignedRuntimeGate(args)
        effective_risk_mode = str(
            getattr(runtime_gate, "training_risk_mode", getattr(runtime_gate, "mode", risk_mode)) or risk_mode
        )
        if effective_risk_mode != risk_mode:
            print(
                f"[risk_admission_mode={risk_mode}] effective runtime mode: "
                f"{effective_risk_mode}"
            )
        risk_mode = effective_risk_mode
        atexit.register(runtime_gate.flush_trace)
        def _flush_chosen_kfs_resolution_events():
            events = pop_chosen_kfs_resolution_events()
            out = Path(args.model_path) / "chosen_kfs_resolution_events.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        atexit.register(_flush_chosen_kfs_resolution_events)

    pose_risk_utility_admission_mode = str(
        getattr(args, "pose_risk_utility_admission_mode", "off") or "off"
    ).strip().lower()
    pose_initialization_risk_gate = None
    pose_initialization_risk_mode = str(
        getattr(args, "pose_initialization_risk_mode", "off") or "off"
    ).strip().lower()
    if (
        pose_risk_utility_admission_mode != "off"
        and pose_initialization_risk_mode == "off"
    ):
        pose_initialization_risk_mode = "observe_v1"
    if pose_initialization_risk_mode != "off":
        pose_initialization_risk_gate = PoseInitializationRiskGate(
            mode=pose_initialization_risk_mode,
            absolute_threshold=float(
                getattr(args, "pose_initialization_risk_absolute_threshold", 0.10)
            ),
            adaptive_sigma=float(
                getattr(args, "pose_initialization_risk_adaptive_sigma", 2.0)
            ),
            warmup=int(getattr(args, "pose_initialization_risk_warmup", 8)),
            history_size=int(
                getattr(args, "pose_initialization_risk_history_size", 64)
            ),
            cooldown_frames=int(
                getattr(args, "pose_initialization_risk_cooldown_frames", 12)
            ),
        )
        pose_initialization_risk_trace_path = (
            Path(args.model_path) / "pose_initialization_risk_trace.json"
        )
        atexit.register(
            pose_initialization_risk_gate.flush,
            pose_initialization_risk_trace_path,
        )
        print(
            f"[pose_initialization_risk_mode={pose_initialization_risk_mode}] "
            "post-pose risk observer enabled."
        )

    pose_risk_utility_gate = None
    if pose_risk_utility_admission_mode != "off":
        pose_risk_utility_gate = PoseRiskUtilityAdmissionGate(
            mode=pose_risk_utility_admission_mode,
            utility_threshold=float(
                getattr(args, "pose_risk_utility_threshold", 0.24)
            ),
            selectivity_reference=float(
                getattr(args, "pose_risk_utility_selectivity_reference", 1.8)
            ),
            isolation_risk_margin=float(
                getattr(args, "pose_risk_utility_isolation_risk_margin", 0.04)
            ),
            isolation_cooldown_frames=int(
                getattr(
                    args,
                    "pose_risk_utility_isolation_cooldown_frames",
                    24,
                )
            ),
            quarantine_risk_margin=float(
                getattr(args, "pose_risk_utility_quarantine_risk_margin", 0.08)
            ),
            quarantine_cooldown_frames=int(
                getattr(
                    args,
                    "pose_risk_utility_quarantine_cooldown_frames",
                    64,
                )
            ),
        )
        pose_risk_utility_trace_path = (
            Path(args.model_path) / "pose_risk_utility_trace.json"
        )
        atexit.register(
            pose_risk_utility_gate.flush,
            pose_risk_utility_trace_path,
        )
        print(
            f"[pose_risk_utility_admission_mode={pose_risk_utility_admission_mode}] "
            "risk-utility joint admission enabled."
        )

    # 根据输入路径类型选择数据集加载器
    # - 流式数据集：URL格式（如rtsp://），用于实时视频流
    # - 本地数据集：本地图像文件夹，用于离线处理
    if "://" in args.source_path:
        dataset = StreamDataset(args.source_path, args.downsampling)
        is_stream = True
    else:
        dataset = ImageDataset(args)
        is_stream = False
    height, width = dataset.get_image_size()

    # ========== 核心模块初始化 ==========
    # 初始化所有核心模块并完成JIT编译（首次运行较慢，后续会缓存）
    print("Initializing modules and running just in time compilation, may take a while...")
    
    # 计算匹配误差阈值（基于图像宽度，确保尺度不变性）
    max_error = max(args.match_max_error * width, 1.5)
    min_displacement = max(args.min_displacement * width, 30)
    
    # 【姿态估计模块】特征匹配器：用于两帧间的特征点匹配和基础矩阵估计
    matcher = Matcher(args.fundmat_samples, max_error)
    
    # 【姿态估计模块】三角化器：将匹配点对三角化为3D点
    triangulator = Triangulator(
        args.num_kpts, args.num_prev_keyframes_miniba_incr, max_error
    )
    
    # 【姿态估计模块】姿态初始化器：负责初始化和增量姿态估计
    pose_initializer = PoseInitializer(
        width, height, triangulator, matcher, 2 * max_error, args
    )
    focal = pose_initializer.f_init
    
    # 【场景表示模块】密集特征提取器：提取图像的密集特征图，用于后续的MVS
    dense_extractor = DenseExtractor(width, height)
    
    # 【场景表示模块】单目深度估计器：使用Depth-Anything-V2模型估计单目深度
    depth_estimator = MonoDepthEstimator(width, height)
    
    # 【场景表示模块】场景模型：管理3D高斯点云、关键帧、锚点等，负责渲染和优化
    scene_model = SceneModel(width, height, args, matcher)
    
    # 【特征提取模块】特征检测器：使用XFeat提取稀疏关键点和描述子
    detector = Detector(args.num_kpts, width, height)

    # 初始化可视化前端：本地/服务器/网页三种模式
    if args.viewer_mode in ["server", "local"]:
        viewer_mode = ViewerMode.SERVER if args.viewer_mode == "server" else ViewerMode.LOCAL
        viewer = GaussianViewer.from_scene_model(scene_model, viewer_mode)
        viewer_thd = Thread(target=viewer.run, args=(args.ip, args.port), daemon=True)
        viewer_thd.start()
        viewer.throttling = True # Enable throttling when training
    elif args.viewer_mode == "web":
        ip = "0.0.0.0"
        server = TCPServer((ip, 8000), SimpleHTTPRequestHandler)
        server_thd = Thread(target=server.serve_forever, daemon=True)
        server_thd.start()
        print(f"Visit http://{ip}:8000/webviewer to for the viewer")

        viewer = WebViewer(scene_model, args.ip, args.port)
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

    # 记录关键帧和运行状态
    n_active_keyframes = 0
    n_keyframes = 0
    needs_reboot = False
    bootstrap_keyframe_dicts = []
    bootstrap_desc_kpts = []
    recent_pose_success = deque(maxlen=50)
    viewpoint_pose_history = deque(maxlen=160)
    last_viewpoint_coverage_event: dict[str, Any] = {}

    # Dict of runtimes for each step
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    metrics = {}

    runtimes = {key: [0, 0] for key in runtimes}
    ## 场景重建主循环
    print(f"Starting reconstruction for {args.source_path}")
    total_frames = len(dataset) if args.max_frames is None or args.max_frames < 0 else min(len(dataset), int(args.max_frames))
    pbar = tqdm(range(0, total_frames))
    reconstruction_start_time = time.time()
    recovery_pose_attempt_state: dict[int, dict[str, Any]] = {}
    recovery_pose_attempt_cache: dict[int, dict[str, Any]] = {}

    def _kf_source_frame_id(kf: Keyframe) -> int:
        return int(kf.info.get("_paper_aligned_source_frame_id", kf.index))

    def _kf_commit_origin(kf: Keyframe) -> str:
        return str(kf.info.get("_paper_aligned_commit_origin", "unknown"))

    def _kf_is_seed(kf: Keyframe) -> bool:
        return bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))

    def _support_eligible_recovery_keyframes() -> list[Keyframe]:
        return [
            keyframe
            for keyframe in scene_model.keyframes
            if bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
            and not bool(keyframe.info.get("is_test", False))
        ]

    def _active_anchor_keyframe_ids() -> list[int]:
        active_anchor = getattr(scene_model, "active_anchor", None)
        if active_anchor is None:
            return []
        return [int(x) for x in getattr(active_anchor, "keyframe_ids", [])]

    def _anchor_id_for_keyframe_id(keyframe_id: int) -> int:
        for anchor_id, anchor in enumerate(getattr(scene_model, "anchors", []) or []):
            if int(keyframe_id) in [int(x) for x in getattr(anchor, "keyframe_ids", [])]:
                return int(anchor_id)
        return -1

    def _append_keyframe_timeline(kf: Keyframe, current_frame_id: int, control_reason: str = "") -> None:
        if runtime_gate is None:
            return
        if any(int(e.get("keyframe_id", -999)) == int(kf.index) for e in runtime_gate.keyframe_timeline_events):
            return
        is_seed = _kf_is_seed(kf)
        source_frame_id = _kf_source_frame_id(kf)
        runtime_gate.append_keyframe_timeline_event(
            {
                "keyframe_id": int(kf.index),
                "source_frame_id": source_frame_id,
                "current_frame_id": int(current_frame_id),
                "image_name": str(kf.info.get("image_name", "")),
                "commit_origin": "early_seed_recovery_commit" if is_seed else _kf_commit_origin(kf),
                "commit_channel": str(kf.info.get("_paper_aligned_commit_channel", "")),
                "control_reason": str(control_reason or kf.info.get("_paper_aligned_control_reason", "")),
                "materialized": True,
                "anchor_id": _anchor_id_for_keyframe_id(int(kf.index)),
                "match_graph_id": "unavailable",
                "is_v7_early_seed": is_seed,
                "seed_source_frame_id": source_frame_id if is_seed else "",
                "seed_runtime_frame_id": int(kf.info.get("_paper_aligned_seed_runtime_frame_id", -1)) if is_seed else "",
            }
        )

    def _append_chosen_reference_trace(frame_id: int, source_frame_id: int, chosen: list[Keyframe]) -> None:
        if runtime_gate is None:
            return
        seed_dists = [abs(_kf_source_frame_id(kf) - int(source_frame_id)) for kf in chosen if _kf_is_seed(kf)]
        runtime_gate.append_chosen_kfs_reference_event(
            {
                "frame_id": int(frame_id),
                "source_frame_id": int(source_frame_id),
                "chosen_kfs_ids": [int(kf.index) for kf in chosen],
                "chosen_kfs_source_frame_ids": [_kf_source_frame_id(kf) for kf in chosen],
                "chosen_kfs_commit_origin": [_kf_commit_origin(kf) for kf in chosen],
                "chosen_kfs_contains_v7_seed": any(_kf_is_seed(kf) for kf in chosen),
                "chosen_kfs_seed_count": sum(1 for kf in chosen if _kf_is_seed(kf)),
                "nearest_seed_source_distance": min(seed_dists) if seed_dists else -1,
                "reference_ids": [int(kf.index) for kf in chosen],
                "reference_source_frame_ids": [_kf_source_frame_id(kf) for kf in chosen],
            }
        )

    def _append_candidate_trace(frame_id: int) -> None:
        if runtime_gate is None:
            return
        debug = getattr(scene_model, "last_prev_keyframes_debug", {}) or {}
        rows = []
        for row in debug.get("candidate_rows", []) or []:
            rows.append({"frame_id": int(frame_id), **dict(row)})
        if rows:
            runtime_gate.append_chosen_kfs_candidate_events(rows)

    def _append_pose_and_matching_traces(frame_id: int, pose_debug: dict[str, Any]) -> None:
        if runtime_gate is None:
            return
        ref_ids = [int(x) for x in pose_debug.get("ref_keyframe_ids", []) or []]
        ref_sources = [int(x) for x in pose_debug.get("ref_source_frame_ids", []) or []]
        ref_origins = [str(x) for x in pose_debug.get("ref_commit_origin", []) or []]
        ref_seed_flags = [bool(x) for x in pose_debug.get("ref_is_seed", []) or []]
        ref_recovery_flags = [bool(x) for x in pose_debug.get("ref_is_recovery", []) or []]
        ref_support_flags = [bool(x) for x in pose_debug.get("ref_is_support_eligible", []) or []]
        match_counts = [int(x) for x in pose_debug.get("match_count_by_ref", []) or []]
        runtime_gate.append_matching_support_event(
            {
                "frame_id": int(frame_id),
                "matched_keyframe_ids": ref_ids,
                "matched_keyframe_source_ids": ref_sources,
                "matched_keyframe_commit_origin": ref_origins,
                "matched_seed_keyframe_count": sum(1 for x in ref_seed_flags if x),
                "match_count_total": int(pose_debug.get("match_count_total", 0) or 0),
                "match_count_to_seed_keyframes": int(pose_debug.get("match_count_to_seed_keyframes", 0) or 0),
                "best_match_keyframe_id": int(pose_debug.get("best_match_keyframe_id", -1) or -1),
                "best_match_is_seed": bool(pose_debug.get("best_match_is_seed", False)),
                "best_match_num_matches": int(pose_debug.get("best_match_num_matches", 0) or 0),
                "match_count_by_ref": match_counts,
            }
        )
        pnp_ids = [int(x) for x in pose_debug.get("pnp_ref_keyframe_ids", []) or []]
        miniba_ids = [int(x) for x in pose_debug.get("miniba_ref_keyframe_ids", []) or []]
        runtime_gate.append_pnp_miniba_reference_event(
            {
                "frame_id": int(frame_id),
                "pnp_ref_keyframe_ids": pnp_ids,
                "pnp_ref_source_frame_ids": [int(x) for x in pose_debug.get("pnp_ref_source_frame_ids", []) or []],
                "pnp_ref_contains_seed": bool(pose_debug.get("pnp_ref_contains_seed", False)),
                "pnp_inlier_count": int(pose_debug.get("num_pnp_inliers", 0) or 0),
                "pnp_success": int(pose_debug.get("num_pnp_inliers", 0) or 0) >= 4,
                "miniba_ref_keyframe_ids": miniba_ids,
                "miniba_ref_source_frame_ids": [int(x) for x in pose_debug.get("miniba_ref_source_frame_ids", []) or []],
                "miniba_ref_contains_seed": bool(pose_debug.get("miniba_ref_contains_seed", False)),
                "miniba_inlier_count": int(pose_debug.get("num_miniba_inliers", 0) or 0),
                "miniba_success": str(pose_debug.get("failure_reason", "") or "") == "",
                "pose_failure_reason": str(pose_debug.get("failure_reason", "") or ""),
            }
        )
        pool_rows = []
        for i, ref_id in enumerate(ref_ids):
            pool_rows.append(
                {
                    "frame_id": int(frame_id),
                    "reference_keyframe_id": ref_id,
                    "reference_source_frame_id": ref_sources[i] if i < len(ref_sources) else -1,
                    "reference_commit_origin": ref_origins[i] if i < len(ref_origins) else "",
                    "reference_is_recovery": ref_recovery_flags[i] if i < len(ref_recovery_flags) else False,
                    "reference_is_early_seed": ref_seed_flags[i] if i < len(ref_seed_flags) else False,
                    "reference_is_support_eligible": ref_support_flags[i] if i < len(ref_support_flags) else False,
                    "reference_used_for_pnp": ref_id in pnp_ids,
                    "reference_used_for_miniba": ref_id in miniba_ids,
                    "reference_support_score": match_counts[i] if i < len(match_counts) else 0,
                }
            )
        if pool_rows:
            runtime_gate.append_pose_reference_pool_events(pool_rows)

    def _observe_defer_recovery_anchor_bridge() -> None:
        bridge = getattr(scene_model, "defer_recovery_bridge", None)
        if bridge is None or runtime_gate is None:
            return
        for ev in bridge.observe_anchor_count(scene_model):
            runtime_gate.append_anchor_transition_bridge_event(ev)

    def _append_local_map_anchor_trace(frame_id: int) -> None:
        if runtime_gate is None:
            return
        active_ids = _active_anchor_keyframe_ids()
        active_set = set(active_ids)
        active_kfs = [kf for kf in scene_model.keyframes if int(kf.index) in active_set]
        runtime_gate.append_local_map_anchor_event(
            {
                "frame_id": int(frame_id),
                "local_map_keyframe_ids": active_ids,
                "local_map_source_frame_ids": [_kf_source_frame_id(kf) for kf in active_kfs],
                "local_map_contains_seed": any(_kf_is_seed(kf) for kf in active_kfs),
                "anchor_id": _anchor_id_for_keyframe_id(active_ids[-1]) if active_ids else -1,
                "anchor_keyframe_ids": active_ids,
                "anchor_contains_seed": any(_kf_is_seed(kf) for kf in active_kfs),
                "match_graph_neighbor_ids": "unavailable",
                "match_graph_neighbor_contains_seed": "unavailable",
            }
        )

    def _append_frame_stage_trace(frame_id: int, stage: str, reason: str = "") -> None:
        if runtime_gate is None:
            return
        event = runtime_gate._get_event(int(frame_id))
        runtime_gate.append_frame_stage_reachability_event(
            {
                "frame_id": int(frame_id),
                "processed": bool(event is not None),
                "candidate_evaluated": bool(event is not None),
                "pose_path_requested": False,
                "pose_path_allowed": False,
                "chosen_kfs_built": False,
                "matching_executed": False,
                "pnp_attempted": False,
                "miniba_attempted": False,
                "recovery_success": False,
                "commit_control_reached": False,
                "runtime_attempted": False,
                "materialized": False,
                "final_stage": str(stage),
                "blocking_stage": str(stage),
                "blocking_reason": str(reason or ""),
            }
        )

    def _write_pose_render_coupling_info(
        info: dict[str, Any],
        payload: dict[str, Any],
        source: str,
    ) -> None:
        keep_keys = (
            "frame_id",
            "source_frame_id",
            "direct_keyframe_finalized",
            "direct_finalization_decision",
            "direct_finalization_reason",
            "pose_safe_tracking_only",
            "density_state",
            "pose_risk_score",
            "pose_risk_high",
            "semantic_pose_risk_score",
            "pose_render_posterior_risk_score",
            "pose_render_risk_score",
            "pose_render_pose_confidence",
            "pose_render_risk_high",
            "pose_render_risk_reason",
            "pose_render_support_ratio",
            "pose_render_inlier_ratio",
            "pose_render_pnp_inlier_ratio",
            "pose_render_miniba_inlier_ratio",
            "pose_render_correspondence_count",
            "pose_render_match_count",
            "pose_render_final_pose_inliers",
            "pose_render_grid_coverage",
            "pose_render_grid_entropy",
            "pose_render_support_concentration",
            "finalize_pose_risk_reference",
            "semantic_R_t",
            "semantic_Q_t",
            "semantic_B_R_t",
            "representation_value_score",
            "novelty_value_score",
            "high_novelty_score",
            "support_needed_score",
            "pose_reference_value_score",
            "utility_drift_risk",
            "utility_representation",
            "utility_coverage_gain",
            "pose_support_score",
            "match_support_score",
            "active_memory_stable_pose_reference",
            "active_memory_frame_role",
            "stream_memory_frame_identity",
            "stream_memory_write_action",
            "stream_memory_representation_need",
            "utility_tracking_safe_context",
            "utility_representation_role",
            "new_view_event_score",
            "anchor_health_score",
            "num_matches",
            "pose_inliers",
        )
        summary: dict[str, Any] = {"gate_source": str(source)}
        for key in keep_keys:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, (bool, int, float, str)):
                summary[key] = value
        if "source_frame_id" not in summary:
            summary["source_frame_id"] = int(info.get("_paper_aligned_source_frame_id", -1))
        if "frame_id" not in summary:
            summary["frame_id"] = int(summary.get("source_frame_id", -1))
        if "pose_risk_score" not in summary and "semantic_R_t" in summary:
            summary["pose_risk_score"] = float(summary["semantic_R_t"])
        info["_paper_aligned_pose_render_coupling"] = summary

    def _write_baseline_render_lock_pose_support_info(
        info: dict[str, Any],
        *,
        frame_id: int,
        pose_debug: dict[str, Any] | None,
        viewpoint_scores: dict[str, Any] | None,
        num_matches: int,
        pose_inliers: int,
    ) -> None:
        if (
            str(getattr(args, "paper_aligned_render_frame_policy", "off") or "off")
            != "baseline_keyframe_lock_v1"
        ):
            return
        if (
            str(
                getattr(
                    args,
                    "paper_aligned_pose_render_psnr_loss_support_weight",
                    "off",
                )
                or "off"
            )
            == "off"
        ):
            return
        pose_debug = dict(pose_debug or {})
        source_frame_id = int(info.get("_paper_aligned_source_frame_id", frame_id))
        payload = {
            "frame_id": int(frame_id),
            "source_frame_id": source_frame_id,
            "direct_keyframe_finalized": True,
            "direct_finalization_decision": "baseline_render_lock",
            "direct_finalization_reason": "baseline_render_keyframe_pose_support",
            "pose_safe_tracking_only": False,
            "density_state": "baseline_render_locked",
            "pose_risk_score": 0.0,
            "pose_risk_high": False,
            "finalize_pose_risk_reference": False,
            "semantic_R_t": 0.0,
            "semantic_Q_t": 1.0,
            "semantic_B_R_t": 1.0,
            "utility_drift_risk": 0.0,
            "pose_support_score": 1.0,
            "match_support_score": 1.0,
            "active_memory_stable_pose_reference": True,
            "active_memory_frame_role": "baseline_render_lock",
            "stream_memory_frame_identity": "render+pose",
            "stream_memory_write_action": "baseline_render_lock_pose_support",
            "utility_tracking_safe_context": True,
            "utility_representation_role": True,
            "num_matches": int(num_matches or 0),
            "pose_inliers": int(pose_inliers or 0),
        }
        augment_pose_render_payload_with_posterior_risk(
            payload,
            pose_debug=pose_debug,
            viewpoint_scores=dict(viewpoint_scores or {}),
            min_num_inliers=int(args.min_num_inliers),
        )
        _write_pose_render_coupling_info(
            info,
            payload,
            "baseline_render_lock_pose_support",
        )

    def _snapshot_pose_match_state(
        curr_desc_kpts: Any,
        ref_keyframes: list[Keyframe],
        current_pose_index: int,
    ) -> dict[str, dict[int, Any]]:
        state = {"curr": {}, "refs": {}}
        for keyframe in ref_keyframes:
            ref_id = int(keyframe.index)
            state["curr"][ref_id] = curr_desc_kpts.matches.get(ref_id, None)
            state["refs"][ref_id] = keyframe.desc_kpts.matches.get(
                int(current_pose_index), None
            )
        return state

    def _pose_matrix_for_trace(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        try:
            return np.asarray(value).tolist()
        except Exception:
            return None

    def _restore_pose_match_state(
        curr_desc_kpts: Any,
        ref_keyframes: list[Keyframe],
        current_pose_index: int,
        state: dict[str, dict[int, Any]],
    ) -> None:
        for keyframe in ref_keyframes:
            ref_id = int(keyframe.index)
            curr_match = state.get("curr", {}).get(ref_id, None)
            ref_match = state.get("refs", {}).get(ref_id, None)
            if curr_match is None:
                curr_desc_kpts.matches.pop(ref_id, None)
            else:
                curr_desc_kpts.matches[ref_id] = curr_match
            if ref_match is None:
                keyframe.desc_kpts.matches.pop(int(current_pose_index), None)
            else:
                keyframe.desc_kpts.matches[int(current_pose_index)] = ref_match

    def _clone_pose_support(support: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in dict(support or {}).items():
            if hasattr(value, "detach") and hasattr(value, "clone"):
                cloned[key] = value.detach().clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _snapshot_torch_rng_state() -> dict[str, Any]:
        return {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
        }

    def _restore_torch_rng_state(state: dict[str, Any] | None) -> None:
        if not state:
            return
        torch.random.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda"):
            torch.cuda.set_rng_state_all(state["cuda"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])

    def _pose_safe_rt_cpu(Rt: Any) -> torch.Tensor | None:
        if Rt is None:
            return None
        try:
            if torch.is_tensor(Rt):
                rt = Rt.detach().float().cpu()
            else:
                rt = torch.tensor(Rt, dtype=torch.float32)
        except Exception:
            return None
        if rt.ndim != 2 or rt.shape[0] < 3 or rt.shape[1] < 4:
            return None
        return rt

    def _pose_safe_camera_center(Rt: torch.Tensor) -> torch.Tensor:
        return -Rt[:3, :3].transpose(0, 1) @ Rt[:3, 3]

    def _pose_safe_rotation_delta_deg(Rt_a: torch.Tensor, Rt_b: torch.Tensor) -> float:
        rel = Rt_a[:3, :3] @ Rt_b[:3, :3].transpose(0, 1)
        cos_angle = ((torch.trace(rel) - 1.0) * 0.5).clamp(-1.0, 1.0)
        return float(torch.rad2deg(torch.acos(cos_angle)).item())

    def _pose_safe_pose_geometry_delta(
        Rt_baseline: Any,
        Rt_memory: Any,
        last_keyframe_Rt: Any,
        pose_history: list[tuple[int, Any]],
        frame_id: int,
    ) -> dict[str, Any]:
        baseline_rt = _pose_safe_rt_cpu(Rt_baseline)
        memory_rt = _pose_safe_rt_cpu(Rt_memory)
        if baseline_rt is None or memory_rt is None:
            return {"available": False}

        baseline_center = _pose_safe_camera_center(baseline_rt)
        memory_center = _pose_safe_camera_center(memory_rt)
        center_delta = float(torch.linalg.norm(memory_center - baseline_center).item())
        rotation_delta = _pose_safe_rotation_delta_deg(memory_rt, baseline_rt)
        out: dict[str, Any] = {
            "available": True,
            "rotation_delta_deg": rotation_delta,
            "center_delta": center_delta,
            "baseline_step": 0.0,
            "memory_step": 0.0,
            "center_delta_over_baseline_step": 0.0,
            "memory_step_over_baseline_step": 0.0,
            "baseline_motion_error": 0.0,
            "memory_motion_error": 0.0,
        }

        last_rt = _pose_safe_rt_cpu(last_keyframe_Rt)
        if last_rt is not None:
            last_center = _pose_safe_camera_center(last_rt)
            baseline_step = float(torch.linalg.norm(baseline_center - last_center).item())
            memory_step = float(torch.linalg.norm(memory_center - last_center).item())
            step_denom = max(baseline_step, 1e-4)
            out.update(
                {
                    "baseline_step": baseline_step,
                    "memory_step": memory_step,
                    "center_delta_over_baseline_step": center_delta / step_denom,
                    "memory_step_over_baseline_step": memory_step / step_denom,
                }
            )

        history: list[tuple[int, torch.Tensor]] = []
        for hist_frame_id, hist_rt_raw in list(pose_history):
            hist_rt = _pose_safe_rt_cpu(hist_rt_raw)
            if hist_rt is not None:
                history.append((int(hist_frame_id), hist_rt))
        if len(history) >= 2:
            prev_frame_id, prev_rt = history[-2]
            last_frame_id, hist_last_rt = history[-1]
            prev_center = _pose_safe_camera_center(prev_rt)
            hist_last_center = _pose_safe_camera_center(hist_last_rt)
            frame_gap = max(int(last_frame_id) - int(prev_frame_id), 1)
            current_gap = max(int(frame_id) - int(last_frame_id), 1)
            expected_center = hist_last_center + (hist_last_center - prev_center) * (
                float(current_gap) / float(frame_gap)
            )
            out["baseline_motion_error"] = float(
                torch.linalg.norm(baseline_center - expected_center).item()
            )
            out["memory_motion_error"] = float(
                torch.linalg.norm(memory_center - expected_center).item()
            )
        return out

    def _attempt_recovery_pose_path(
        recovered: dict[str, Any],
        current_frame_id: int,
        attempt_reason: str,
    ) -> dict[str, Any]:
        if runtime_gate is None:
            return {"allowed": False, "success": False, "block_reason": "runtime_gate_disabled"}
        source_payload = recovered.get("source_payload", {}) or {}
        source_frame_id = int(recovered.get("source_frame_id", -1))
        source_info = source_payload.get("source_info", None) or {}
        source_desc = source_payload.get("desc_kpts", None)
        source_image = source_payload.get("image_tensor", None)
        source_event = runtime_gate._get_event(source_frame_id) or {}
        lifecycle_state_before = str(source_event.get("action", "unknown"))
        state = recovery_pose_attempt_state.setdefault(
            source_frame_id, {"attempt_count": 0, "last_attempt_frame": -10_000, "last_success": False}
        )
        is_defer_recoverable = lifecycle_state_before == "defer_recoverable"
        is_density_hold_recoverable = bool(
            lifecycle_state_before == "direct_admit"
            and runtime_gate.is_recovery_pose_path_candidate(source_frame_id)
        )
        is_recovery_pose_candidate = bool(is_defer_recoverable or is_density_hold_recoverable)
        is_duplicate = bool(source_event.get("final_keyframe_incremented", False)) or runtime_gate._source_already_committed(source_frame_id)
        is_surrogate = bool(int(source_frame_id) == int(current_frame_id))
        missing_source = bool(source_frame_id < 0 or source_desc is None or source_image is None)
        cooldown_active = bool(
            int(state.get("attempt_count", 0)) > 0
            and int(current_frame_id) - int(state.get("last_attempt_frame", -10_000)) < 8
            and not bool(state.get("last_success", False))
        )
        retry_budget_exceeded = bool(int(state.get("attempt_count", 0)) >= 2 and not bool(state.get("last_success", False)))
        support_refs_available = bool(len(scene_model.keyframes) >= args.num_keyframes_miniba_bootstrap)
        if lifecycle_state_before == "discard":
            block_reason = "discard_semantics"
        elif lifecycle_state_before not in {"defer_recoverable", "direct_admit"}:
            block_reason = "unknown"
        elif is_duplicate:
            block_reason = "duplicate_source"
        elif missing_source:
            block_reason = "missing_source_frame"
        elif not support_refs_available:
            block_reason = "no_support_reference"
        elif cooldown_active:
            block_reason = "cooldown_active"
        elif retry_budget_exceeded:
            block_reason = "retry_budget_exceeded"
        else:
            block_reason = ""
        allowed = bool(
            risk_mode != "off"
            and getattr(args, "paper_aligned_recovery_commit_bridge", "true_source_commit") == "true_source_commit"
            and is_recovery_pose_candidate
            and not block_reason
        )
        lifecycle_event = {
            "frame_id": int(current_frame_id),
            "source_frame_id": int(source_frame_id),
            "lifecycle_state_before": lifecycle_state_before,
            "lifecycle_state_after": lifecycle_state_before,
            "is_defer_recoverable": bool(is_defer_recoverable),
            "is_defer_only": bool(is_defer_recoverable),
            "is_density_hold_recoverable": bool(is_density_hold_recoverable),
            "in_recovery_pool": True,
            "pose_path_requested": True,
            "pose_path_allowed": bool(allowed),
            "pose_path_request_reason": str(attempt_reason),
            "pose_path_block_reason": block_reason if not allowed else "",
            "recovery_pose_attempted": False,
            "recovery_pose_attempt_result": "",
            "recovery_success": False,
            "control_allow_commit": False,
            "runtime_commit_attempted": False,
            "materialized": False,
        }
        if not allowed:
            runtime_gate.append_lifecycle_gate_event(lifecycle_event)
            return {"allowed": False, "success": False, "block_reason": block_reason}

        state["attempt_count"] = int(state.get("attempt_count", 0)) + 1
        state["last_attempt_frame"] = int(current_frame_id)
        attempt_id = int(state["attempt_count"])
        lifecycle_event["recovery_pose_attempted"] = True
        runtime_gate.mark_pose_attempt(source_frame_id)
        pose_debug: dict[str, Any] = {}
        prev_keyframes_src: list[Keyframe] = []
        Rt_src = None
        failure_reason = ""
        try:
            recovery_ref_count = int(args.num_prev_keyframes_check)
            prev_keyframes_src = scene_model.get_prev_keyframes(
                recovery_ref_count,
                True,
                source_desc,
                resolution_mode="paper_aligned_true_recovery",
            )
            for kf in prev_keyframes_src:
                kf.info["_paper_aligned_anchor_id"] = _anchor_id_for_keyframe_id(int(kf.index))
            pose_initializer.recovery_defer_source_frame_id = int(source_frame_id)
            scene_model._recovery_source_frame_id = int(source_frame_id)
            _append_candidate_trace(int(current_frame_id))
            _append_chosen_reference_trace(int(current_frame_id), source_frame_id, prev_keyframes_src)
            Rt_src = pose_initializer.initialize_incremental_recovery(
                prev_keyframes_src,
                source_desc,
                n_keyframes,
                bool(source_info.get("is_test", False)),
                source_image,
            )
            pose_debug = dict(getattr(pose_initializer, "last_incremental_debug", {}) or {})
            fix_trace = dict(getattr(pose_initializer, "last_recovery_pose_outcome_fix", {}) or {})
            if fix_trace:
                runtime_gate.append_recovery_pose_outcome_fix_event(
                    {
                        "attempt_id": attempt_id,
                        "source_frame_id": int(source_frame_id),
                        "current_frame_id": int(current_frame_id),
                        "pnp_success": bool(int(pose_debug.get("num_pnp_inliers", 0) or 0) >= 4),
                        "pnp_inliers": int(pose_debug.get("num_pnp_inliers", 0) or 0),
                        "pnp_ref_ids": list(pose_debug.get("pnp_ref_keyframe_ids", []) or []),
                        "pnp_seed_ref_count": sum(
                            1 for flag in (pose_debug.get("ref_is_seed", []) or []) if bool(flag)
                        ),
                        "miniba_attempted": True,
                        "miniba_success_before_fix": bool(fix_trace.get("miniba_success_before_fix", False)),
                        "miniba_inliers_before_fix": int(fix_trace.get("miniba_inliers_before_fix", 0) or 0),
                        "miniba_ref_ids_before_fix": list(pose_debug.get("miniba_ref_keyframe_ids", []) or []),
                        "handoff_fix_applied": bool(fix_trace.get("handoff_fix_applied", False)),
                        "reference_consistency_fix_applied": bool(
                            fix_trace.get("reference_consistency_fix_applied", False)
                        ),
                        "correspondence_flow_fix_applied": bool(
                            fix_trace.get("correspondence_flow_fix_applied", False)
                        ),
                        "recovery_miniba_retry_applied": bool(fix_trace.get("recovery_miniba_retry_applied", False)),
                        "miniba_success_after_fix": bool(fix_trace.get("miniba_success_after_fix", False)),
                        "miniba_inliers_after_fix": int(fix_trace.get("miniba_inliers_after_fix", 0) or 0),
                        "miniba_ref_ids_after_fix": list(pose_debug.get("miniba_ref_keyframe_ids", []) or []),
                        "recovery_success_after_fix": bool(Rt_src is not None),
                        "failure_reason_after_fix": str(
                            fix_trace.get("failure_reason_after_fix", pose_debug.get("failure_reason", ""))
                            or ""
                        ),
                        "triangulation_augmented_correspondence_count": int(
                            fix_trace.get("triangulation_augmented_correspondence_count", 0) or 0
                        ),
                    }
                )
            failure_reason = str(pose_debug.get("failure_reason", "") or "")
        except Exception as exc:
            failure_reason = f"recovery_pose_exception:{type(exc).__name__}"
            pose_debug = {"failure_reason": failure_reason}

        success = bool(Rt_src is not None)
        state["last_success"] = success
        support_trace = dict(getattr(pose_initializer, "last_recovery_2d3d_support", {}) or {})
        consensus_trace = dict(getattr(pose_initializer, "last_recovery_pnp_consensus", {}) or {})
        if consensus_trace and runtime_gate is not None:
            runtime_gate.append_recovery_pnp_consensus_event(
                {
                    "attempt_id": attempt_id,
                    "source_frame_id": int(source_frame_id),
                    "current_frame_id": int(current_frame_id),
                    "valid_2d3d_correspondence_count": int(
                        pose_debug.get("num_2d3d_correspondences", 0) or 0
                    ),
                    **consensus_trace,
                }
            )
            for row in getattr(pose_initializer, "last_recovery_ref_subset", []) or []:
                runtime_gate.append_recovery_ref_subset_event(
                    {"attempt_id": attempt_id, "source_frame_id": int(source_frame_id), **dict(row)}
                )
        if support_trace and runtime_gate is not None:
            runtime_gate.append_recovery_2d3d_support_event(
                {
                    "attempt_id": attempt_id,
                    "source_frame_id": int(source_frame_id),
                    "current_frame_id": int(current_frame_id),
                    "reference_keyframe_ids": [int(kf.index) for kf in prev_keyframes_src],
                    "reference_commit_origins": [
                        str(kf.info.get("_paper_aligned_commit_origin", "")) for kf in prev_keyframes_src
                    ],
                    "reference_is_recovery_seed": [
                        bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
                        or str(kf.info.get("_paper_aligned_commit_origin", ""))
                        in {"true_recovery_commit", "early_seed_recovery_commit"}
                        for kf in prev_keyframes_src
                    ],
                    "raw_2d2d_match_count": int(support_trace.get("raw_2d2d_match_count", 0) or 0),
                    "verified_2d2d_match_count": int(support_trace.get("verified_2d2d_match_count", 0) or 0),
                    "has_pt3d_match_count": int(support_trace.get("has_pt3d_match_count", 0) or 0),
                    "valid_2d3d_correspondence_count": int(pose_debug.get("num_2d3d_correspondences", 0) or 0),
                    "valid_2d3d_from_direct_refs": int(support_trace.get("valid_2d3d_from_direct_refs", 0) or 0),
                    "valid_2d3d_from_recovery_refs": int(
                        support_trace.get("valid_2d3d_from_recovery_refs", 0) or 0
                    ),
                    "valid_2d3d_from_seed_refs": int(support_trace.get("valid_2d3d_from_seed_refs", 0) or 0),
                    "num_3d_observations_refs_total": int(support_trace.get("has_pt3d_match_count", 0) or 0),
                    "per_ref_3d_bearing_counts": list(support_trace.get("per_ref_3d_bearing_counts", []) or []),
                    "selected_refs_by_raw_match": "unavailable",
                    "selected_refs_by_3d_support": list(
                        support_trace.get("selected_refs_by_3d_support", []) or []
                    ),
                    "temporary_3d_support_used": bool(support_trace.get("temporary_3d_support_used", False)),
                    "temporary_3d_support_count": int(support_trace.get("temporary_3d_support_count", 0) or 0),
                    "pnp_correspondence_count": int(pose_debug.get("num_pnp_inliers", 0) or 0),
                    "pnp_inliers": int(pose_debug.get("num_pnp_inliers", 0) or 0),
                    "miniba_correspondence_count": int(pose_debug.get("num_miniba_inliers", 0) or 0),
                    "miniba_inliers": int(pose_debug.get("num_miniba_inliers", 0) or 0),
                    "miniba_success": bool(success),
                    "recovery_success": bool(success),
                    "failure_reason": str(pose_debug.get("failure_reason", "") or ""),
                }
            )
            per_ref_counts = list(support_trace.get("per_ref_3d_bearing_counts", []) or [])
            for i, kf in enumerate(prev_keyframes_src):
                runtime_gate.append_recovery_reference_3d_association_event(
                    {
                        "keyframe_id": int(kf.index),
                        "source_frame_id": int(_kf_source_frame_id(kf)),
                        "commit_origin": str(kf.info.get("_paper_aligned_commit_origin", "")),
                        "is_recovery": str(kf.info.get("_paper_aligned_commit_origin", ""))
                        in {"true_recovery_commit", "early_seed_recovery_commit"},
                        "is_early_seed": bool(kf.info.get("_paper_aligned_is_v7_early_seed", False)),
                        "has_pose": True,
                        "has_features": True,
                        "has_descriptors": True,
                        "has_anchor_id": "unavailable",
                        "has_match_graph_id": "unavailable",
                        "has_3d_observations": bool(int(kf.desc_kpts.has_pt3d.sum().item()) > 0),
                        "num_3d_observations": int(kf.desc_kpts.has_pt3d.sum().item()),
                        "num_visible_gaussians_if_available": "unavailable",
                        "num_map_points_if_available": "unavailable",
                        "used_as_recovery_ref_count": 1,
                        "contributed_2d3d_count": int(per_ref_counts[i]) if i < len(per_ref_counts) else 0,
                    }
                )
        runtime_gate.annotate_pose_debug(source_frame_id, pose_debug)
        runtime_gate.mark_pose_result(source_frame_id, success)
        _append_pose_and_matching_traces(int(current_frame_id), pose_debug)
        lifecycle_event["recovery_pose_attempt_result"] = "success" if success else "failure"
        lifecycle_event["recovery_success"] = success
        lifecycle_event["pose_path_block_reason"] = "" if success else (failure_reason or "pose_init_failed")
        runtime_gate.append_lifecycle_gate_event(lifecycle_event)
        bridge_meta = dict(getattr(scene_model, "_last_recovery_bridge_meta", {}) or {})
        if scene_model.defer_recovery_bridge is not None and bridge_meta:
            scene_model.defer_recovery_bridge.log_recovery_support_trace(
                runtime_gate,
                int(source_frame_id),
                int(current_frame_id),
                pose_debug,
                support_trace,
                consensus_trace,
                bridge_meta,
            )
            scene_model.defer_recovery_bridge.log_anchor_bridge_usage(
                runtime_gate,
                int(source_frame_id),
                bridge_meta,
                pose_debug,
                int(consensus_trace.get("valid_2d3d_before_consensus", 0) or 0),
                int(consensus_trace.get("pnp_inliers_before_consensus", 0) or 0),
                0,
            )
        runtime_gate.append_recovery_pose_path_event(
            {
                "source_frame_id": int(source_frame_id),
                "current_frame_id": int(current_frame_id),
                "attempt_id": attempt_id,
                "attempt_reason": str(attempt_reason),
                "chosen_kfs_built": bool(prev_keyframes_src),
                "chosen_kfs_ids": [int(kf.index) for kf in prev_keyframes_src],
                "chosen_kfs_source_frame_ids": [_kf_source_frame_id(kf) for kf in prev_keyframes_src],
                "chosen_kfs_contains_recovery_seed": any(_kf_is_seed(kf) for kf in prev_keyframes_src),
                "matching_executed": bool(pose_debug.get("ref_keyframe_ids", [])),
                "matching_candidate_count": len(pose_debug.get("ref_keyframe_ids", []) or []),
                "matching_seed_candidate_count": sum(1 for x in pose_debug.get("ref_is_seed", []) or [] if bool(x)),
                "pnp_attempted": bool(pose_debug.get("num_pnp_inliers", 0) is not None),
                "pnp_inliers": int(pose_debug.get("num_pnp_inliers", 0) or 0),
                "pnp_success": bool(int(pose_debug.get("num_pnp_inliers", 0) or 0) >= 4),
                "miniba_attempted": bool(pose_debug.get("num_miniba_inliers", 0) is not None),
                "miniba_inliers": int(pose_debug.get("num_miniba_inliers", 0) or 0),
                "miniba_success": bool(success),
                "pose_failure_reason": "" if success else (failure_reason or "pose_init_failed"),
            }
        )
        if success:
            recovered["_recovery_pose_attempt_success"] = True
            recovered["_recovery_pose_Rt"] = Rt_src
            recovered["_recovery_pose_debug"] = pose_debug
            recovery_pose_attempt_cache[source_frame_id] = {
                "Rt": Rt_src,
                "debug": pose_debug,
                "attempt_frame": int(current_frame_id),
                "attempt_reason": str(attempt_reason),
            }
        return {
            "allowed": True,
            "success": success,
            "block_reason": "" if success else (failure_reason or "pose_init_failed"),
            "lifecycle_event": lifecycle_event,
        }

    def _commit_true_source_frame(recovered: dict, control_decision: dict[str, Any] | None = None) -> None:
        global n_keyframes, prev_desc_kpts, prev_keyframe
        if runtime_gate is None:
            return
        source_payload = recovered.get("source_payload", {}) or {}
        source_frame_id = int(recovered.get("source_frame_id", -1))
        source_image = source_payload.get("image_tensor", None)
        source_info = source_payload.get("source_info", None)
        source_desc = source_payload.get("desc_kpts", None)
        recovery_trace = {
            "recovery_event_id": int(len(runtime_gate.true_recovery_commit_events) + 1),
            "source_frame_id": source_frame_id,
            "source_input_index": int(source_payload.get("source_input_index", source_frame_id)),
            "source_image_name": str(source_payload.get("source_image_name", "")),
            "current_tick_frame_id": int(recovered.get("current_tick_frame_id", -1)),
            "current_tick_image_name": str(recovered.get("current_tick_image_name", "")),
            "source_equals_current_frame": bool(
                int(recovered.get("current_tick_frame_id", -1)) == source_frame_id
            ),
            "pool_enter_tick": int(recovered.get("pool_enter_tick", -1)),
            "recovery_attempt_tick": int(recovered.get("recovery_attempt_tick", -1)),
            "recovery_attempt_count": int(recovered.get("recovery_attempt_count", 0)),
            "recovery_success": True,
            "recovery_success_reason": "semantic_policy_success",
            "materialized_source_keyframe": False,
            "materialization_failure_reason": "",
            "add_keyframe_called": False,
            "add_keyframe_success": False,
            "scene_keyframe_appended": False,
            "final_keyframe_incremented": False,
            "representation_update_called": False,
            "representation_update_success": False,
            "gaussian_update_called": False,
            "gaussian_update_success": False,
            "anchor_update_called": False,
            "anchor_update_success": False,
            "active_set_update_called": False,
            "active_set_update_success": False,
            "optimizer_received": False,
            "final_model_contains_source_frame": False,
            "failure_stage": "",
            "failure_reason": "",
            "commit_control_mode": str(getattr(args, "paper_aligned_recovery_commit_control", "off")),
            "commit_control_decision": "allow_commit",
            "commit_control_reason": "",
            "control_decision": "allow_commit",
            "runtime_commit_attempted": True,
            "runtime_commit_success": False,
            "source_resolution_success": False,
            "add_keyframe_attempted": False,
            "add_keyframe_success": False,
            "materialized": False,
            "final_timeline_recorded": False,
            "keyframe_count_at_failure": int(n_keyframes),
        }
        if isinstance(control_decision, dict):
            recovery_trace["commit_control_decision"] = str(
                control_decision.get("control_decision", "allow_commit")
            )
            recovery_trace["commit_control_reason"] = str(control_decision.get("decision_reason", ""))
            recovery_trace["commit_control_debug"] = dict(control_decision.get("debug", {}))
            control_decision["control_decision"] = "allow_commit"
            control_decision["runtime_commit_attempted"] = True

        def _finish_materialization(committed: bool, failure_reason: str = "") -> None:
            reason = str(failure_reason or "")
            recovery_trace["runtime_commit_success"] = bool(committed)
            recovery_trace["materialized"] = bool(committed)
            recovery_trace["final_timeline_recorded"] = bool(committed)
            recovery_trace["materialization_failure_reason"] = reason
            if committed and runtime_gate is not None:
                for row in reversed(runtime_gate.recovery_support_trace_events):
                    if int(row.get("source_frame_id", -1)) == int(source_frame_id):
                        row["true_source_materialized"] = True
                        row["materialization_fail_reason"] = ""
                        break
            if not committed:
                recovery_trace["keyframe_count_at_failure"] = int(n_keyframes)
            if isinstance(control_decision, dict):
                control_decision["runtime_commit_attempted"] = True
                control_decision["runtime_commit_success"] = bool(committed)
                control_decision["source_resolution_success"] = bool(
                    recovery_trace.get("source_resolution_success", False)
                )
                control_decision["add_keyframe_attempted"] = bool(
                    recovery_trace.get("add_keyframe_attempted", False)
                )
                control_decision["add_keyframe_success"] = bool(
                    recovery_trace.get("add_keyframe_success", False)
                )
                control_decision["materialized"] = bool(committed)
                control_decision["final_timeline_recorded"] = bool(committed)
                control_decision["materialization_failure_reason"] = reason
            runtime_gate.append_recovery_commit_materialization_event(recovery_trace)

        runtime_gate.mark_true_source_recovery_attempt(
            source_frame_id=source_frame_id,
            current_tick_frame_id=int(recovered.get("current_tick_frame_id", -1)),
            current_tick_image_name=str(recovered.get("current_tick_image_name", "")),
            pool_enter_tick=int(recovered.get("pool_enter_tick", -1)),
            recovery_attempt_tick=int(recovered.get("recovery_attempt_tick", -1)),
        )
        event = runtime_gate._get_event(source_frame_id)
        if event is None:
            recovery_trace["failure_stage"] = "source_resolution"
            recovery_trace["failure_reason"] = "source_mapping_failed"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="source_mapping_failed"
            )
            _finish_materialization(False, "source_mapping_failed")
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        recovery_trace["source_resolution_success"] = True
        if event is not None and bool(event.get("final_keyframe_incremented", False)):
            recovery_trace["failure_stage"] = "duplicate_gate"
            recovery_trace["failure_reason"] = "already_existing_keyframe"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="already_existing_keyframe"
            )
            _finish_materialization(False, "already_existing_keyframe")
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if n_keyframes < args.num_keyframes_miniba_bootstrap:
            recovery_trace["failure_stage"] = "bootstrap_gate"
            recovery_trace["failure_reason"] = "bootstrap_not_ready"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="bootstrap_not_ready"
            )
            _finish_materialization(False, "add_keyframe_internal_skip")
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if source_image is None or source_desc is None:
            recovery_trace["failure_stage"] = "source_payload"
            recovery_trace["failure_reason"] = "source_mapping_failed"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="source_mapping_failed"
            )
            _finish_materialization(False, "source_mapping_failed")
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if source_info is None:
            source_info = {
                "is_test": False,
                "image_name": str(source_payload.get("source_image_name", "")),
                "image_path": str(source_payload.get("image_path", "")),
            }
        source_info["_paper_aligned_source_frame_id"] = source_frame_id
        source_info["_paper_aligned_commit_origin"] = "true_recovery_commit"
        source_info["_paper_aligned_commit_channel"] = str(
            ((control_decision or {}).get("debug", {}) or {}).get("commit_channel", "")
        )
        source_info["_paper_aligned_control_reason"] = str(
            (control_decision or {}).get("decision_reason", "")
        )
        source_info["_paper_aligned_is_v7_early_seed"] = bool(
            source_info["_paper_aligned_commit_channel"] == "pre_collapse_early_seed"
            or source_info["_paper_aligned_control_reason"] == "v7_early_seed_commit"
        )
        source_info["_paper_aligned_seed_runtime_frame_id"] = int(recovered.get("current_tick_frame_id", -1))
        source_info["_paper_aligned_insertion_type"] = "true_recovery_commit"
        if bool(recovered.get("_recovery_pose_attempt_success", False)) and recovered.get("_recovery_pose_Rt") is not None:
            Rt_src = recovered.get("_recovery_pose_Rt")
            runtime_gate.mark_pose_attempt(source_frame_id)
            pose_debug = dict(recovered.get("_recovery_pose_debug", {}) or {})
            runtime_gate.annotate_pose_debug(source_frame_id, pose_debug)
            runtime_gate.mark_pose_result(source_frame_id, True)
        else:
            runtime_gate.mark_pose_attempt(source_frame_id)
            try:
                prev_keyframes_src = scene_model.get_prev_keyframes(
                    args.num_prev_keyframes_miniba_incr,
                    True,
                    source_desc,
                    resolution_mode="paper_aligned_true_recovery",
                )
                _append_candidate_trace(int(recovered.get("current_tick_frame_id", source_frame_id)))
                _append_chosen_reference_trace(
                    int(recovered.get("current_tick_frame_id", source_frame_id)),
                    source_frame_id,
                    prev_keyframes_src,
                )
                Rt_src = pose_initializer.initialize_incremental(
                    prev_keyframes_src, source_desc, n_keyframes, bool(source_info.get("is_test", False)), source_image
                )
            except Exception as exc:
                recovery_trace["failure_stage"] = "source_resolution"
                recovery_trace["failure_reason"] = f"chosen_kfs_invalid:{type(exc).__name__}"
                runtime_gate.mark_true_source_recovery_result(
                    source_frame_id, committed=False, reason="chosen_kfs_invalid"
                )
                _finish_materialization(False, "chosen_kfs_invalid")
                runtime_gate.append_true_recovery_commit_event(recovery_trace)
                return
            runtime_gate.annotate_pose_debug(
                source_frame_id, getattr(pose_initializer, "last_incremental_debug", {})
            )
            _append_pose_and_matching_traces(
                int(recovered.get("current_tick_frame_id", source_frame_id)),
                getattr(pose_initializer, "last_incremental_debug", {}),
            )
        runtime_gate.mark_pose_result(source_frame_id, Rt_src is not None)
        if Rt_src is None:
            pose_debug = getattr(pose_initializer, "last_incremental_debug", {})
            fail_reason = str(pose_debug.get("failure_reason", "") or "source_pose_init_failed")
            recovery_trace["failure_stage"] = "pose"
            recovery_trace["failure_reason"] = fail_reason
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason=fail_reason
            )
            _finish_materialization(False, "add_keyframe_internal_skip")
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if args.use_colmap_poses and "Rt" in source_info:
            Rt_src = source_info["Rt"]
        before_scene_keyframes = len(scene_model.keyframes)
        try:
            source_kf = Keyframe(
                source_image,
                source_info,
                source_desc,
                Rt_src,
                n_keyframes,
                f,
                dense_extractor,
                depth_estimator,
                triangulator,
                args,
            )
            recovery_trace["materialized_source_keyframe"] = True
            recovery_trace["add_keyframe_attempted"] = True
            scene_model.add_keyframe(source_kf)
        except Exception as exc:
            recovery_trace["failure_stage"] = "add_keyframe"
            recovery_trace["failure_reason"] = f"add_keyframe_exception:{type(exc).__name__}"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="add_keyframe_exception"
            )
            _finish_materialization(False, "add_keyframe_exception")
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        runtime_gate.mark_keyframe_add(source_frame_id)
        recovery_trace["add_keyframe_called"] = True
        recovery_trace["add_keyframe_success"] = bool(len(scene_model.keyframes) > before_scene_keyframes)
        recovery_trace["scene_keyframe_appended"] = bool(len(scene_model.keyframes) > before_scene_keyframes)
        if not recovery_trace["add_keyframe_success"]:
            recovery_trace["failure_stage"] = "add_keyframe"
            recovery_trace["failure_reason"] = "add_keyframe_internal_skip"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="add_keyframe_internal_skip"
            )
            _finish_materialization(False, "add_keyframe_internal_skip")
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        recovery_trace["active_set_update_called"] = True
        recovery_trace["active_set_update_success"] = True
        scene_model.add_new_gaussians()
        runtime_gate.mark_gaussian_update(source_frame_id)
        recovery_trace["representation_update_called"] = True
        recovery_trace["representation_update_success"] = True
        recovery_trace["gaussian_update_called"] = True
        recovery_trace["gaussian_update_success"] = True
        if is_stream:
            scene_model.optimize_async(args.num_iterations)
        else:
            scene_model.optimization_loop(args.num_iterations)
        recovery_trace["optimizer_received"] = True
        scene_model.place_anchor_if_needed()
        _observe_defer_recovery_anchor_bridge()
        runtime_gate.mark_anchor_update(source_frame_id)
        recovery_trace["anchor_update_called"] = True
        recovery_trace["anchor_update_success"] = True
        current_tick_frame_id = int(recovered.get("current_tick_frame_id", -1))
        source_action = str(event.get("action", ""))
        support_checks = {
            "materialized": True,
            "true_source_frame_id_valid": int(source_frame_id) >= 0,
            "pose_valid": Rt_src is not None,
            "features_ready": source_desc is not None,
            "not_duplicate": bool(len(scene_model.keyframes) > before_scene_keyframes),
            "not_surrogate": bool(int(source_frame_id) != current_tick_frame_id),
            "not_contamination": source_action in {"defer_recoverable", "direct_admit"},
            "not_test": not bool(source_info.get("is_test", False)),
        }
        is_support_eligible = all(bool(v) for v in support_checks.values())
        source_info["_paper_aligned_support_eligible_recovery_keyframe"] = bool(is_support_eligible)
        if is_support_eligible:
            prev_desc_kpts = source_desc
            prev_keyframe = source_kf
        runtime_gate.append_support_integration_event(
            {
                "event_type": "support_reference_registration",
                "source_frame_id": int(source_frame_id),
                "current_frame_id": current_tick_frame_id,
                "keyframe_id": int(source_kf.index),
                "image_name": str(source_info.get("image_name", "")),
                "commit_origin": str(source_info.get("_paper_aligned_commit_origin", "")),
                "commit_channel": str(source_info.get("_paper_aligned_commit_channel", "")),
                "is_v7_early_seed": bool(source_info.get("_paper_aligned_is_v7_early_seed", False)),
                "is_support_eligible_recovery_keyframe": bool(is_support_eligible),
                "prev_desc_kpts_updated": bool(is_support_eligible),
                "prev_keyframe_updated": bool(is_support_eligible),
                **support_checks,
                "failure_reason": "" if is_support_eligible else ";".join(k for k, v in support_checks.items() if not bool(v)),
            }
        )
        _append_keyframe_timeline(
            source_kf,
            int(recovered.get("current_tick_frame_id", source_frame_id)),
            recovery_trace["commit_control_reason"],
        )
        _append_local_map_anchor_trace(int(recovered.get("current_tick_frame_id", source_frame_id)))
        n_keyframes += 1
        runtime_gate.mark_final_keyframe_increment(source_frame_id)
        runtime_gate.mark_true_source_recovery_result(source_frame_id, committed=True, reason="")
        recovery_trace["final_keyframe_incremented"] = True
        recovery_trace["final_model_contains_source_frame"] = True
        _finish_materialization(True, "")
        runtime_gate.append_true_recovery_commit_event(recovery_trace)

    runtime_action = ""
    baseline_should_add_frame = False
    for frameID in pbar:
        start_time = time.time()
        runtime_action = ""
        baseline_should_add_frame = False
        pose_safe_tracking_only = False

        # ========== 网页端交互控制 ==========
        if args.viewer_mode == "web":
            viewer.trainer_state = "running"

            # 支持网页端暂停训练（用于调试和检查）
            while viewer.state == "stop":
                pbar.set_postfix_str(
                    "\033[31mPaused. Press the Start button in the webviewer\033[0m"
                )
                time.sleep(0.1)
            
            # 支持网页端提前结束训练
            if viewer.state == "finish":
                viewer.trainer_state = "finish"
                break
        
        # ========== 第一帧处理 ==========
        # 第一帧仅用于引导初始化，提取特征但不进行三角化
        if n_keyframes == 0:
            image, info = dataset.getnext()
            info["_paper_aligned_source_frame_id"] = int(frameID)
            info["_paper_aligned_commit_origin"] = "direct_admit"
            info["_paper_aligned_commit_channel"] = ""
            info["_paper_aligned_control_reason"] = ""
            info["_paper_aligned_is_v7_early_seed"] = False
            prev_desc_kpts = detector(image)  # 提取关键点和描述子
            bootstrap_keyframe_dicts = [{"image": image, "info": info}]
            bootstrap_desc_kpts = [prev_desc_kpts]
            n_keyframes += 1
            continue

        # ========== 特征提取与关键帧判断 ==========
        # 读取下一帧图像并提取特征
        image, info = dataset.getnext()
        info["_paper_aligned_source_frame_id"] = int(frameID)
        info["_paper_aligned_commit_origin"] = "direct_admit"
        info["_paper_aligned_commit_channel"] = ""
        info["_paper_aligned_control_reason"] = ""
        info["_paper_aligned_is_v7_early_seed"] = False
        desc_kpts = detector(image)  # 【特征提取模块】提取稀疏关键点和描述子
        
        # 【姿态估计模块】当前帧与上一帧做特征匹配
        curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)
        
        # 基于匹配点位移判断是否生成新关键帧
        # 关键帧选择策略：当相机运动足够大时才添加关键帧，避免冗余
        dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
        should_add_keyframe = (
            dist.median() > min_displacement  # 中位位移超过阈值
            and len(curr_prev_matches.kpts) > args.min_num_inliers  # 匹配点数量足够
        )
        support_bridge_trace = None
        support_bridge_overrides_keyframe_gate = False
        if runtime_gate is not None and risk_mode != "off":
            support_bridge_overrides_keyframe_gate = bool(
                runtime_gate.should_support_bridge_override_keyframe_gate()
            )
            support_candidates = _support_eligible_recovery_keyframes()
            best_support = {
                "keyframe_id": -1,
                "source_frame_id": -1,
                "num_matches": 0,
                "median_displacement": 0.0,
                "is_seed": False,
                "should_add": False,
                "matches": None,
                "dist": None,
            }
            for support_kf in support_candidates:
                support_matches = matcher(desc_kpts, support_kf.desc_kpts)
                support_dist = torch.norm(
                    support_matches.kpts - support_matches.kpts_other, dim=-1
                )
                support_num_matches = int(len(support_matches.kpts))
                support_median = (
                    float(support_dist.median().item()) if len(support_dist) > 0 else 0.0
                )
                support_should_add = bool(
                    support_median > min_displacement
                    and support_num_matches > args.min_num_inliers
                )
                if (
                    support_should_add
                    and (
                        not bool(best_support["should_add"])
                        or support_median > float(best_support["median_displacement"])
                    )
                ):
                    best_support = {
                        "keyframe_id": int(support_kf.index),
                        "source_frame_id": _kf_source_frame_id(support_kf),
                        "num_matches": support_num_matches,
                        "median_displacement": support_median,
                        "is_seed": _kf_is_seed(support_kf),
                        "should_add": True,
                        "matches": support_matches,
                        "dist": support_dist,
                    }
            support_bridge_trace = {
                "frame_id": int(frameID),
                "matched_seed_keyframe_count": 1 if bool(best_support["is_seed"]) else 0,
                "matched_seed_best_score": int(best_support["num_matches"]) if bool(best_support["is_seed"]) else 0,
                "support_candidate_count": len(support_candidates),
                "best_support_keyframe_id": int(best_support["keyframe_id"]),
                "best_support_source_frame_id": int(best_support["source_frame_id"]),
                "best_support_num_matches": int(best_support["num_matches"]),
                "best_support_median_displacement": float(best_support["median_displacement"]),
                "support_candidate_should_add": bool(best_support["should_add"]),
                "support_bridge_keyframe_gate_enabled": bool(
                    support_bridge_overrides_keyframe_gate
                ),
                "support_triggered_keyframe_gate": bool(
                    best_support["should_add"] and support_bridge_overrides_keyframe_gate
                ),
                "baseline_prev_num_matches": int(len(curr_prev_matches.kpts)),
                "baseline_prev_median_displacement": float(dist.median().item()) if len(dist) > 0 else 0.0,
                "baseline_prev_should_add": bool(should_add_keyframe),
                "seed_promoted_to_reference_count": 0,
                "seed_promoted_to_pnp_count": 0,
                "seed_promoted_to_miniba_count": 0,
                "bridge_block_reason": "",
            }
            if bool(best_support["should_add"] and support_bridge_overrides_keyframe_gate):
                curr_prev_matches = best_support["matches"]
                dist = best_support["dist"]
                should_add_keyframe = True
            elif bool(best_support["should_add"]):
                support_bridge_trace["bridge_block_reason"] = (
                    "support_bridge_keyframe_gate_disabled"
                )
        # 测试帧始终加入，用于姿态估计和评估（但不参与训练）
        should_add_keyframe |= info["is_test"]
        baseline_should_add = should_add_keyframe
        baseline_should_add_frame = bool(baseline_should_add)
        if runtime_gate is not None:
            if support_bridge_trace is not None:
                support_bridge_trace["final_should_add_after_support"] = bool(should_add_keyframe)
                runtime_gate.append_support_integration_event(
                    {"event_type": "support_candidate_matching", **support_bridge_trace}
                )
                runtime_gate.append_matching_to_pose_path_bridge_event(dict(support_bridge_trace))
            phase = (
                "bootstrap"
                if n_keyframes < args.num_keyframes_miniba_bootstrap
                else "incremental"
            )
            info["_image_tensor"] = image
            info["_desc_kpts"] = desc_kpts
            info["_inlier_evidence"] = {"num_matches": int(len(curr_prev_matches.kpts))}
            info["_local_context"] = {"phase": phase}
            active_anchor_ids_for_gate = _active_anchor_keyframe_ids()
            evidence = {
                "median_displacement": float(dist.median().item()) if len(dist) > 0 else 0.0,
                "displacement_threshold": float(min_displacement),
                "num_matches": int(len(curr_prev_matches.kpts)),
                "min_num_inliers_threshold": int(args.min_num_inliers),
                "is_test": bool(info.get("is_test", False)),
                "scene_anchor_count": int(len(getattr(scene_model, "anchors", []) or [])),
                "active_anchor_keyframe_count": int(len(active_anchor_ids_for_gate)),
                "recent_pose_fail_rate": (
                    1.0 - (sum(recent_pose_success) / max(len(recent_pose_success), 1))
                    if len(recent_pose_success) > 0
                    else 0.0
                ),
                "baseline_prev_should_add": bool(
                    (support_bridge_trace or {}).get("baseline_prev_should_add", baseline_should_add)
                ),
                "support_triggered_keyframe_gate": bool(
                    (support_bridge_trace or {}).get("support_triggered_keyframe_gate", False)
                ),
                "support_candidate_count": int(
                    (support_bridge_trace or {}).get("support_candidate_count", 0) or 0
                ),
                "best_support_num_matches": int(
                    (support_bridge_trace or {}).get("best_support_num_matches", 0) or 0
                ),
                "best_support_median_displacement": float(
                    (support_bridge_trace or {}).get("best_support_median_displacement", 0.0) or 0.0
                ),
                "recent_viewpoint_new_view_event_score": float(
                    last_viewpoint_coverage_event.get("new_view_event_score", 0.0) or 0.0
                ),
                "recent_viewpoint_anchor_health_score": float(
                    last_viewpoint_coverage_event.get("anchor_health_score", 0.0) or 0.0
                ),
                "recent_viewpoint_active_anchor_keyframe_count": int(
                    last_viewpoint_coverage_event.get("active_anchor_keyframe_count", 0) or 0
                ),
            }
            should_add_keyframe, runtime_action = runtime_gate.decide(
                frameID, info, bool(baseline_should_add), phase=phase, evidence=evidence
            )
            baseline_render_lock_forced = bool(
                runtime_gate.should_lock_baseline_render_keyframe(
                    action=str(runtime_action),
                    baseline_should_add=bool(baseline_should_add_frame),
                    phase=phase,
                )
            )
            pose_safe_baseline_skeleton_forced = False
            if baseline_render_lock_forced:
                should_add_keyframe = True
                runtime_action = "direct_admit"
                trace_ev = runtime_gate._get_event(frameID)
                if trace_ev is not None:
                    trace_ev["baseline_render_lock_forced"] = True
                    trace_ev["admit_to_chain"] = True
                    trace_ev["action_before_baseline_render_lock"] = str(
                        trace_ev.get("action", "")
                    )
                    trace_ev["action"] = "direct_admit"
            else:
                pose_safe_baseline_skeleton_forced = bool(
                    runtime_gate.should_pose_safe_preserve_baseline_keyframe(
                        action=str(runtime_action),
                        baseline_should_add=bool(baseline_should_add_frame),
                        phase=phase,
                    )
                )
                if pose_safe_baseline_skeleton_forced:
                    should_add_keyframe = True
                    runtime_action = "direct_admit"
                    trace_ev = runtime_gate._get_event(frameID)
                    if trace_ev is not None:
                        trace_ev["pose_safe_baseline_skeleton_forced"] = True
                        trace_ev["admit_to_chain"] = True
                        trace_ev["action_before_pose_safe_baseline_force"] = str(
                            trace_ev.get("action", "")
                        )
                        trace_ev["action"] = "direct_admit"
                else:
                    pose_safe_tracking_only = bool(
                        runtime_gate.should_pose_safe_track_deferred(
                            frame_id=int(frameID),
                            action=str(runtime_action),
                            phase=phase,
                            evidence=evidence,
                        )
                    )
                    if pose_safe_tracking_only:
                        should_add_keyframe = True
                        runtime_action = "direct_admit"
                        baseline_should_add_frame = False
                        trace_ev = runtime_gate._get_event(frameID)
                        if trace_ev is not None:
                            trace_ev["pose_safe_tracking_only"] = True
                            trace_ev["pose_safe_tracking_admitted_to_pose_path"] = True
            baseline_eval_frame = bool(
                info.get("_baseline_eval_frame", info.get("is_test", False))
            )
            if baseline_eval_frame and not should_add_keyframe:
                trace_ev = runtime_gate._get_event(frameID)
                if trace_ev is not None:
                    trace_ev["baseline_eval_frame_forced"] = True
                    trace_ev["action_before_baseline_eval_force"] = str(
                        trace_ev.get("action", "")
                    )
                    trace_ev["action"] = "direct_admit"
                    trace_ev["admit_to_chain"] = True
                should_add_keyframe = True
                runtime_action = "direct_admit"
            if (
                risk_mode == "paper_aligned_semantic_v1"
                and getattr(args, "paper_aligned_recovery_commit_bridge", "true_source_commit")
                == "true_source_commit"
                and runtime_gate.should_process_recovery_commits_for_frame(frameID)
            ):
                for recovered in runtime_gate.pop_pending_true_source_commits(current_tick_frame_id=frameID):
                    source_frame_id = int(recovered.get("source_frame_id", -1))
                    cached_pose = recovery_pose_attempt_cache.get(source_frame_id, {})
                    if cached_pose.get("Rt") is not None:
                        recovered["_recovery_pose_attempt_success"] = True
                        recovered["_recovery_pose_Rt"] = cached_pose.get("Rt")
                        recovered["_recovery_pose_debug"] = dict(cached_pose.get("debug", {}) or {})
                        pose_probe = {"allowed": True, "success": True, "block_reason": ""}
                        runtime_gate.append_lifecycle_gate_event(
                            {
                                "frame_id": int(frameID),
                                "source_frame_id": int(source_frame_id),
                                "lifecycle_state_before": str((runtime_gate._get_event(source_frame_id) or {}).get("action", "")),
                                "lifecycle_state_after": str((runtime_gate._get_event(source_frame_id) or {}).get("action", "")),
                                "is_defer_recoverable": True,
                                "is_defer_only": True,
                                "in_recovery_pool": True,
                                "pose_path_requested": True,
                                "pose_path_allowed": True,
                                "pose_path_request_reason": "recovery_pool_candidate",
                                "pose_path_block_reason": "",
                                "recovery_pose_attempted": False,
                                "recovery_pose_attempt_result": "cached_success",
                                "recovery_success": True,
                                "control_allow_commit": False,
                                "runtime_commit_attempted": False,
                                "materialized": False,
                            }
                        )
                    else:
                        pose_probe = _attempt_recovery_pose_path(
                            recovered,
                            current_frame_id=frameID,
                            attempt_reason="recovery_pool_candidate",
                        )
                    if not bool(pose_probe.get("allowed", False)) or not bool(pose_probe.get("success", False)):
                        hold_reason = str(pose_probe.get("block_reason", "recovery_pose_attempt_failed") or "recovery_pose_attempt_failed")
                        runtime_gate.hold_recovered_source(
                            recovered,
                            current_tick_frame_id=frameID,
                            reason=hold_reason,
                        )
                        runtime_gate.mark_true_source_recovery_result(
                            source_frame_id, committed=False, reason=hold_reason
                        )
                        runtime_gate.append_true_recovery_commit_event(
                            {
                                "recovery_event_id": int(len(runtime_gate.true_recovery_commit_events) + 1),
                                "source_frame_id": source_frame_id,
                                "source_input_index": int(recovered.get("source_input_index", source_frame_id)),
                                "source_image_name": str((recovered.get("source_payload", {}) or {}).get("source_image_name", "")),
                                "current_tick_frame_id": int(frameID),
                                "current_tick_image_name": str(info.get("image_name", "")),
                                "source_equals_current_frame": bool(source_frame_id == int(frameID)),
                                "pool_enter_tick": int(recovered.get("pool_enter_tick", -1)),
                                "recovery_attempt_tick": int(recovered.get("recovery_attempt_tick", -1)),
                                "recovery_attempt_count": int(recovered.get("recovery_attempt_count", 0)),
                                "recovery_success": True,
                                "recovery_success_reason": "semantic_policy_success",
                                "materialized_source_keyframe": False,
                                "materialization_failure_reason": "recovery_pose_attempt_not_ready",
                                "add_keyframe_called": False,
                                "add_keyframe_success": False,
                                "scene_keyframe_appended": False,
                                "final_keyframe_incremented": False,
                                "representation_update_called": False,
                                "representation_update_success": False,
                                "gaussian_update_called": False,
                                "gaussian_update_success": False,
                                "anchor_update_called": False,
                                "anchor_update_success": False,
                                "active_set_update_called": False,
                                "active_set_update_success": False,
                                "optimizer_received": False,
                                "final_model_contains_source_frame": False,
                                "failure_stage": "recovery_pose_path",
                                "failure_reason": hold_reason,
                                "commit_control_mode": str(getattr(args, "paper_aligned_recovery_commit_control", "off")),
                                "commit_control_decision": "not_reached",
                                "control_decision": "not_reached",
                                "runtime_commit_attempted": False,
                                "runtime_commit_success": False,
                                "source_resolution_success": False,
                                "add_keyframe_attempted": False,
                                "add_keyframe_success": False,
                                "materialized": False,
                                "final_timeline_recorded": False,
                            }
                        )
                        continue
                    control = runtime_gate.decide_recovered_commit(recovered, current_tick_frame_id=frameID)
                    runtime_gate.append_lifecycle_gate_event(
                        {
                            "frame_id": int(frameID),
                            "source_frame_id": int(source_frame_id),
                            "lifecycle_state_before": str((runtime_gate._get_event(source_frame_id) or {}).get("action", "")),
                            "lifecycle_state_after": str((runtime_gate._get_event(source_frame_id) or {}).get("action", "")),
                            "is_defer_recoverable": True,
                            "is_defer_only": True,
                            "in_recovery_pool": True,
                            "pose_path_requested": True,
                            "pose_path_allowed": True,
                            "pose_path_request_reason": "recovery_pool_candidate",
                            "pose_path_block_reason": "",
                            "recovery_pose_attempted": True,
                            "recovery_pose_attempt_result": "success",
                            "recovery_success": True,
                            "control_allow_commit": bool(str(control.get("decision")) == "commit" or str(control.get("control_decision")) == "allow_commit"),
                            "runtime_commit_attempted": False,
                            "materialized": False,
                        }
                    )
                    if str(control.get("decision")) == "commit":
                        _commit_true_source_frame(recovered, control_decision=control)
                    elif str(control.get("decision")) == "hold":
                        runtime_gate.hold_recovered_source(
                            recovered, current_tick_frame_id=frameID, reason=str(control.get("decision_reason", "hold"))
                        )
                        runtime_gate.mark_true_source_recovery_result(
                            source_frame_id, committed=False, reason=str(control.get("decision_reason", "hold"))
                        )
                        runtime_gate.append_true_recovery_commit_event(
                            {
                                "recovery_event_id": int(len(runtime_gate.true_recovery_commit_events) + 1),
                                "source_frame_id": source_frame_id,
                                "source_input_index": int(
                                    recovered.get("source_input_index", source_frame_id)
                                ),
                                "source_image_name": str(
                                    (recovered.get("source_payload", {}) or {}).get("source_image_name", "")
                                ),
                                "current_tick_frame_id": int(frameID),
                                "current_tick_image_name": str(info.get("image_name", "")),
                                "source_equals_current_frame": bool(source_frame_id == int(frameID)),
                                "pool_enter_tick": int(recovered.get("pool_enter_tick", -1)),
                                "recovery_attempt_tick": int(recovered.get("recovery_attempt_tick", -1)),
                                "recovery_attempt_count": int(recovered.get("recovery_attempt_count", 0)),
                                "recovery_success": True,
                                "recovery_success_reason": "semantic_policy_success",
                                "materialized_source_keyframe": False,
                                "materialization_failure_reason": "commit_control_hold",
                                "add_keyframe_called": False,
                                "add_keyframe_success": False,
                                "scene_keyframe_appended": False,
                                "final_keyframe_incremented": False,
                                "representation_update_called": False,
                                "representation_update_success": False,
                                "gaussian_update_called": False,
                                "gaussian_update_success": False,
                                "anchor_update_called": False,
                                "anchor_update_success": False,
                                "active_set_update_called": False,
                                "active_set_update_success": False,
                                "optimizer_received": False,
                                "final_model_contains_source_frame": False,
                                "failure_stage": "commit_control",
                                "failure_reason": str(control.get("decision_reason", "hold")),
                                "commit_control_mode": str(
                                    getattr(args, "paper_aligned_recovery_commit_control", "off")
                                ),
                                "commit_control_decision": "hold",
                                "control_decision": "hold",
                                "runtime_commit_attempted": False,
                                "runtime_commit_success": False,
                                "source_resolution_success": False,
                                "add_keyframe_attempted": False,
                                "add_keyframe_success": False,
                                "materialized": False,
                                "final_timeline_recorded": False,
                            }
                        )
                    else:
                        runtime_gate.reject_recovered_source(
                            recovered, reason=str(control.get("decision_reason", "reject"))
                        )
                        runtime_gate.mark_true_source_recovery_result(
                            source_frame_id, committed=False, reason=str(control.get("decision_reason", "reject"))
                        )
                        runtime_gate.append_true_recovery_commit_event(
                            {
                                "recovery_event_id": int(len(runtime_gate.true_recovery_commit_events) + 1),
                                "source_frame_id": source_frame_id,
                                "source_input_index": int(
                                    recovered.get("source_input_index", source_frame_id)
                                ),
                                "source_image_name": str(
                                    (recovered.get("source_payload", {}) or {}).get("source_image_name", "")
                                ),
                                "current_tick_frame_id": int(frameID),
                                "current_tick_image_name": str(info.get("image_name", "")),
                                "source_equals_current_frame": bool(source_frame_id == int(frameID)),
                                "pool_enter_tick": int(recovered.get("pool_enter_tick", -1)),
                                "recovery_attempt_tick": int(recovered.get("recovery_attempt_tick", -1)),
                                "recovery_attempt_count": int(recovered.get("recovery_attempt_count", 0)),
                                "recovery_success": True,
                                "recovery_success_reason": "semantic_policy_success",
                                "materialized_source_keyframe": False,
                                "materialization_failure_reason": "commit_control_reject",
                                "add_keyframe_called": False,
                                "add_keyframe_success": False,
                                "scene_keyframe_appended": False,
                                "final_keyframe_incremented": False,
                                "representation_update_called": False,
                                "representation_update_success": False,
                                "gaussian_update_called": False,
                                "gaussian_update_success": False,
                                "anchor_update_called": False,
                                "anchor_update_success": False,
                                "active_set_update_called": False,
                                "active_set_update_success": False,
                                "optimizer_received": False,
                                "final_model_contains_source_frame": False,
                                "failure_stage": "commit_control",
                                "failure_reason": str(control.get("decision_reason", "reject")),
                                "commit_control_mode": str(
                                    getattr(args, "paper_aligned_recovery_commit_control", "off")
                                ),
                                "commit_control_decision": "reject",
                                "control_decision": "reject",
                                "runtime_commit_attempted": False,
                                "runtime_commit_success": False,
                                "source_resolution_success": False,
                                "add_keyframe_attempted": False,
                                "add_keyframe_success": False,
                                "materialized": False,
                                "final_timeline_recorded": False,
                            }
                        )
        increment_runtime(runtimes["Load"], start_time)

        if should_add_keyframe:
            # ========== Bootstrap阶段：初始姿态和焦距估计 ==========
            # 累积前N帧用于初始姿态与焦距估计（通常N=8）
            if n_keyframes < args.num_keyframes_miniba_bootstrap:
                bootstrap_keyframe_dicts.append({"image": image, "info": info})
                bootstrap_desc_kpts.append(desc_kpts)

            # 当累积够N帧时，执行Bootstrap初始化
            if n_keyframes == args.num_keyframes_miniba_bootstrap - 1:
                start_time = time.time()
                # 【姿态估计模块】使用Mini-BA同时估计所有初始帧的位姿和焦距
                if runtime_gate is not None:
                    runtime_gate.mark_pose_attempt(frameID)
                Rts, f, _ = pose_initializer.initialize_bootstrap(bootstrap_desc_kpts)
                if runtime_gate is not None:
                    runtime_gate.mark_pose_result(frameID, True)
                focal = f.cpu().item()
                increment_runtime(runtimes["BAB"], start_time)
                
                # 为每个Bootstrap关键帧创建Keyframe对象并添加到场景
                for index, (keyframe_dict, desc_kpts, Rt) in enumerate(
                    zip(bootstrap_keyframe_dicts, bootstrap_desc_kpts, Rts)
                ):
                    start_time = time.time()
                    # 如果使用COLMAP位姿，则覆盖估计的位姿
                    if args.use_colmap_poses:
                        Rt = keyframe_dict["info"]["Rt"]
                        f = keyframe_dict["info"]["focal"]
                    if runtime_gate is not None:
                        bootstrap_source_frame_id = int(
                            keyframe_dict["info"].get(
                                "_paper_aligned_source_frame_id", index
                            )
                        )
                        _write_pose_render_coupling_info(
                            keyframe_dict["info"],
                            {
                                "frame_id": bootstrap_source_frame_id,
                                "source_frame_id": bootstrap_source_frame_id,
                                "direct_keyframe_finalized": True,
                                "direct_finalization_decision": "bootstrap_miniba",
                                "direct_finalization_reason": "bootstrap_pose_batch",
                                "pose_risk_score": 0.0,
                                "pose_risk_high": False,
                                "finalize_pose_risk_reference": False,
                                "semantic_R_t": 0.0,
                                "semantic_Q_t": 1.0,
                                "semantic_B_R_t": 1.0,
                                "utility_drift_risk": 0.0,
                                "pose_support_score": 1.0,
                                "match_support_score": 1.0,
                                "active_memory_stable_pose_reference": True,
                                "active_memory_frame_role": "bootstrap",
                                "stream_memory_frame_identity": "render+pose",
                                "stream_memory_write_action": "bootstrap_write",
                                "utility_tracking_safe_context": True,
                                "utility_representation_role": True,
                            },
                            "bootstrap_miniba",
                        )
                    # 【场景表示模块】创建关键帧对象（包含图像、深度、特征等）
                    keyframe = Keyframe(
                        keyframe_dict["image"],
                        keyframe_dict["info"],
                        desc_kpts,
                        Rt,
                        index,
                        f,
                        dense_extractor,
                        depth_estimator,
                        triangulator,
                        args,
                    )
                    scene_model.add_keyframe(keyframe, f)
                    if runtime_gate is not None:
                        runtime_gate.mark_keyframe_add(frameID)
                        _append_keyframe_timeline(
                            keyframe,
                            int(keyframe.info.get("_paper_aligned_source_frame_id", frameID)),
                            "bootstrap_direct_admit",
                        )
                    increment_runtime(runtimes["Add"], start_time)
                
                if args.viewer_mode not in ["none", "web"]:
                    viewer.reset_intrinsics("point_view")
                prev_keyframe = keyframe
                
                # 【场景表示模块】为每个Bootstrap关键帧初始化3D高斯点
                for index in range(args.num_keyframes_miniba_bootstrap):
                    start_time = time.time()
                    scene_model.add_new_gaussians(index)
                    if runtime_gate is not None:
                        runtime_gate.mark_gaussian_update(frameID)
                    increment_runtime(runtimes["Init"], start_time)
                
                start_time = time.time()
                # 【优化模块】初始优化：流式用异步优化（不阻塞主线程），离线直接同步优化
                if is_stream:
                    scene_model.optimize_async(args.num_iterations)
                else:
                    scene_model.optimization_loop(args.num_iterations)
                increment_runtime(runtimes["Opt"], start_time)
                last_reboot = n_keyframes

            # ========== Reboot机制：处理相机运动模式突变 ==========
            # 当相机运动模式发生突变时（如从平移变为旋转），需要重启重建
            if (
                args.enable_reboot
                and not bool(pose_safe_tracking_only)
                and scene_model.approx_cam_centres is not None
                and len(scene_model.anchors)
            ):
                # 检查最近20帧的相机中心间距变化
                # 如果间距过大（快速运动）或过小（几乎静止），可能需要重启
                last_centers = scene_model.approx_cam_centres[-20:]
                rel_dist = torch.norm(
                    last_centers[1:] - last_centers[:-1], dim=-1
                ).mean()
                needs_reboot = (
                    rel_dist > 0.1 * 5 or rel_dist < 0.1 / 3  # 运动模式异常
                ) and n_keyframes - last_reboot > 50  # 距离上次重启足够远
            
            if needs_reboot:
                # 【姿态估计模块】重启：对末尾8个关键帧重新做Bootstrap BA
                bs_kfs = scene_model.keyframes[-8:]
                bootstrap_desc_kpts = [bs_kf.desc_kpts for bs_kf in bs_kfs]
                in_Rts = torch.stack([kf.get_Rt() for kf in bs_kfs])
                Rts, _, final_residual = pose_initializer.initialize_bootstrap(
                    bootstrap_desc_kpts, rebooting=True
                )
                # 验证重启是否收敛（残差足够小）
                if final_residual < max_error * 0.5:
                    # 对齐重启后的位姿到原坐标系
                    Rts = align_mean_up_fwd(Rts, in_Rts)
                    for Rt, keyframe in zip(Rts, bs_kfs):
                        keyframe.set_Rt(Rt)
                    # 【场景表示模块】重置场景并重新初始化高斯点
                    scene_model.reset()
                    for i in range(3, 0, -1):
                        scene_model.add_new_gaussians(-i)
                    # 【优化模块】快速优化新初始化的高斯点
                    for _ in range(3 * args.num_iterations):
                        scene_model.optimization_step()
                    needs_reboot = False
                    last_reboot = n_keyframes

            # ========== 增量重建阶段：逐帧添加新关键帧 ==========
            # 当Bootstrap完成后，进入增量重建模式
            if n_keyframes >= args.num_keyframes_miniba_bootstrap:
                start_time = time.time()
                # 【场景表示模块】获取与当前帧最匹配的历史关键帧（用于三角化和姿态估计）
                prev_keyframes = scene_model.get_prev_keyframes(
                    args.num_prev_keyframes_miniba_incr,
                    True,
                    desc_kpts,
                    exclude_pose_quarantined=(
                        pose_risk_utility_admission_mode
                        == "pose_quarantine_v1"
                    ),
                )
                pose_only_refs = []
                if runtime_gate is not None:
                    pose_only_refs = runtime_gate.select_pose_only_references(
                        frame_id=int(frameID),
                        curr_desc_kpts=desc_kpts,
                        matcher=scene_model.matcher,
                    )
                prev_keyframes_for_pose = list(prev_keyframes) + list(pose_only_refs)
                if runtime_gate is not None:
                    _append_candidate_trace(frameID)
                increment_runtime(runtimes["tri"], start_time)

                pose_initialization_risk_match_before = (
                    _snapshot_pose_match_state(
                        desc_kpts,
                        prev_keyframes_for_pose,
                        n_keyframes,
                    )
                    if pose_initialization_risk_gate is not None
                    else None
                )
                start_time = time.time()
                # 【姿态估计模块】增量姿态初始化：使用PnP-RANSAC和Mini-BA估计新帧位姿
                if runtime_gate is not None:
                    runtime_gate.mark_pose_attempt(frameID)
                pose_safe_dual_candidate = bool(
                    runtime_gate is not None
                    and pose_only_refs
                    and runtime_gate.direct_density_controller.is_pose_safe_streaming_memory_v1
                )
                pose_safe_memory_choice: dict[str, Any] = {}
                pose_safe_pose_rng_before = (
                    _snapshot_torch_rng_state()
                    if runtime_gate is not None
                    and runtime_gate.direct_density_controller.is_pose_safe_streaming_memory_v1
                    and (pose_safe_dual_candidate or pose_safe_tracking_only)
                    else None
                )
                pose_safe_probe_refs: list[Keyframe] = []
                pose_safe_pose_match_before = None
                if pose_safe_pose_rng_before is not None:
                    pose_safe_probe_refs_by_id: dict[int, Keyframe] = {}
                    for ref in prev_keyframes_for_pose:
                        pose_safe_probe_refs_by_id[int(ref.index)] = ref
                    pose_safe_probe_refs = list(pose_safe_probe_refs_by_id.values())
                    pose_safe_pose_match_before = _snapshot_pose_match_state(
                        desc_kpts, pose_safe_probe_refs, n_keyframes
                    )
                if pose_safe_dual_candidate:
                    all_trial_refs = pose_safe_probe_refs
                    initial_match_state = pose_safe_pose_match_before

                    Rt_baseline = pose_initializer.initialize_incremental(
                        list(prev_keyframes), desc_kpts, n_keyframes, info["is_test"], image
                    )
                    baseline_debug = copy.deepcopy(
                        getattr(pose_initializer, "last_incremental_debug", {}) or {}
                    )
                    baseline_support = _clone_pose_support(
                        getattr(pose_initializer, "last_incremental_pose_support", {}) or {}
                    )
                    baseline_match_state = _snapshot_pose_match_state(
                        desc_kpts, all_trial_refs, n_keyframes
                    )
                    baseline_rng_after = _snapshot_torch_rng_state()
                    pose_safe_memory_probe = runtime_gate.should_probe_pose_safe_memory_pose(
                        frame_id=int(frameID),
                        current_keyframe_count=int(n_keyframes),
                        baseline_pose_success=Rt_baseline is not None,
                        baseline_debug=baseline_debug,
                        pose_only_references=pose_only_refs,
                    )
                    if bool(pose_safe_memory_probe.get("probe_memory_pose", False)):
                        _restore_pose_match_state(
                            desc_kpts, all_trial_refs, n_keyframes, initial_match_state or {}
                        )
                        _restore_torch_rng_state(pose_safe_pose_rng_before)
                        Rt_memory = pose_initializer.initialize_incremental(
                            prev_keyframes_for_pose, desc_kpts, n_keyframes, info["is_test"], image
                        )
                        memory_debug = copy.deepcopy(
                            getattr(pose_initializer, "last_incremental_debug", {}) or {}
                        )
                        memory_support = _clone_pose_support(
                            getattr(pose_initializer, "last_incremental_pose_support", {}) or {}
                        )
                        memory_match_state = _snapshot_pose_match_state(
                            desc_kpts, all_trial_refs, n_keyframes
                        )
                        memory_rng_after = _snapshot_torch_rng_state()

                        pose_safe_candidate_delta = _pose_safe_pose_geometry_delta(
                            Rt_baseline,
                            Rt_memory,
                            prev_keyframe.get_Rt() if "prev_keyframe" in locals() else None,
                            list(viewpoint_pose_history),
                            int(frameID),
                        )
                        pose_safe_memory_choice = runtime_gate.choose_pose_safe_memory_pose(
                            frame_id=int(frameID),
                            current_keyframe_count=int(n_keyframes),
                            baseline_pose_success=Rt_baseline is not None,
                            memory_pose_success=Rt_memory is not None,
                            baseline_debug=baseline_debug,
                            memory_debug=memory_debug,
                            pose_only_reference_ids=[int(ref.index) for ref in pose_only_refs],
                            candidate_pose_delta=pose_safe_candidate_delta,
                        )
                        pose_safe_memory_choice["memory_probe"] = True
                        pose_safe_memory_choice["memory_probe_reason"] = str(
                            pose_safe_memory_probe.get("reason", "")
                        )
                        if bool(pose_safe_memory_choice.get("use_memory_pose", False)):
                            Rt = Rt_memory
                            pose_initializer.last_incremental_debug = memory_debug
                            pose_initializer.last_incremental_pose_support = memory_support
                            _restore_pose_match_state(
                                desc_kpts, all_trial_refs, n_keyframes, memory_match_state
                            )
                            _restore_torch_rng_state(memory_rng_after)
                        else:
                            Rt = Rt_baseline
                            prev_keyframes_for_pose = list(prev_keyframes)
                            pose_initializer.last_incremental_debug = baseline_debug
                            pose_initializer.last_incremental_pose_support = baseline_support
                            _restore_pose_match_state(
                                desc_kpts, all_trial_refs, n_keyframes, baseline_match_state
                            )
                            _restore_torch_rng_state(baseline_rng_after)
                    else:
                        Rt = Rt_baseline
                        prev_keyframes_for_pose = list(prev_keyframes)
                        pose_initializer.last_incremental_debug = baseline_debug
                        pose_initializer.last_incremental_pose_support = baseline_support
                        _restore_pose_match_state(
                            desc_kpts, all_trial_refs, n_keyframes, baseline_match_state
                        )
                        _restore_torch_rng_state(baseline_rng_after)
                        pose_safe_memory_choice = {
                            "decision": "baseline_pose",
                            "reason": str(pose_safe_memory_probe.get("reason", "")),
                            "use_memory_pose": False,
                            "memory_probe": False,
                            "memory_probe_decision": dict(pose_safe_memory_probe),
                            "baseline_success": Rt_baseline is not None,
                            "memory_success": False,
                            "memory_reference_used": False,
                            "frame_id": int(frameID),
                            "current_keyframe_count": int(n_keyframes),
                        }
                    trace_ev = runtime_gate._get_event(frameID)
                    if trace_ev is not None:
                        trace_ev["pose_safe_dual_candidate"] = True
                        trace_ev["pose_safe_memory_pose_decision"] = dict(
                            pose_safe_memory_choice
                        )
                else:
                    Rt = pose_initializer.initialize_incremental(
                        prev_keyframes_for_pose, desc_kpts, n_keyframes, info["is_test"], image
                    )
                if runtime_gate is not None:
                    _append_chosen_reference_trace(frameID, frameID, prev_keyframes_for_pose)
                if runtime_gate is not None:
                    runtime_gate.annotate_pose_debug(
                        frameID, getattr(pose_initializer, "last_incremental_debug", {})
                    )
                    pose_debug = getattr(pose_initializer, "last_incremental_debug", {})
                    _append_pose_and_matching_traces(frameID, pose_debug)
                    if support_bridge_trace is not None:
                        ref_seed_count = sum(
                            1
                            for kf in prev_keyframes_for_pose
                            if _kf_is_seed(kf)
                        )
                        runtime_gate.append_matching_to_pose_path_bridge_event(
                            {
                                **support_bridge_trace,
                                "seed_promoted_to_reference_count": int(ref_seed_count),
                                "seed_promoted_to_pnp_count": int(bool(pose_debug.get("pnp_ref_contains_seed", False))),
                                "seed_promoted_to_miniba_count": int(bool(pose_debug.get("miniba_ref_contains_seed", False))),
                                "bridge_block_reason": "" if ref_seed_count > 0 else "seed_not_selected_as_pose_reference",
                            }
                        )
                    runtime_gate.mark_pose_result(frameID, Rt is not None)
                recent_pose_success.append(1 if Rt is not None else 0)
                increment_runtime(runtimes["BAI"], start_time)
                
                start_time = time.time()
                if Rt is not None:  # 姿态估计成功
                    # 如果使用COLMAP位姿，则覆盖估计的位姿
                    if args.use_colmap_poses:
                        Rt = info["Rt"]
                    viewpoint_coverage_event: dict[str, Any] = {}
                    if (
                        runtime_gate is not None
                        or pose_initialization_risk_gate is not None
                    ):
                        active_anchor_ids = _active_anchor_keyframe_ids()
                        active_anchor_Rt = None
                        if active_anchor_ids:
                            active_anchor_kf = next(
                                (
                                    kf
                                    for kf in scene_model.keyframes
                                    if int(kf.index) == int(active_anchor_ids[-1])
                                ),
                                None,
                            )
                            if active_anchor_kf is not None:
                                active_anchor_Rt = active_anchor_kf.get_Rt()
                        last_keyframe_Rt = (
                            prev_keyframe.get_Rt() if "prev_keyframe" in locals() else None
                        )
                        viewpoint_coverage_event = build_viewpoint_coverage_event(
                            frame_id=int(frameID),
                            current_Rt=Rt,
                            last_keyframe_Rt=last_keyframe_Rt,
                            active_anchor_Rt=active_anchor_Rt,
                            inlier_kpts=curr_prev_matches.kpts,
                            image_width=int(width),
                            image_height=int(height),
                            pose_debug=getattr(
                                pose_initializer, "last_incremental_debug", {}
                            )
                            or {},
                            active_anchor_keyframe_count=len(active_anchor_ids),
                            selected_reference_count=len(
                                prev_keyframes_for_pose
                                if "prev_keyframes_for_pose" in locals()
                                else []
                            ),
                            pose_history=list(viewpoint_pose_history),
                        )
                        if runtime_gate is not None:
                            pose_memory_pool_summary = runtime_gate.pose_only_reference_pool_summary()
                            pose_memory_candidate_pool_size = len(
                                getattr(runtime_gate, "_pending_true_source_commits", [])
                            ) + len(getattr(runtime_gate, "_held_true_source_commits", []))
                            viewpoint_coverage_event.update(
                                {
                                    "pose_memory_reference_count": int(len(pose_only_refs)),
                                    "pose_memory_pool_size": int(
                                        pose_memory_pool_summary.get("pool_size", 0)
                                    ),
                                    "pose_memory_candidate_pool_size": int(
                                        pose_memory_candidate_pool_size
                                    ),
                                }
                            )
                    pose_debug_incr = getattr(
                        pose_initializer, "last_incremental_debug", {}
                    ) or {}
                    pose_initialization_risk_decision = None
                    if pose_initialization_risk_gate is not None:
                        recent_pose_fail_rate = (
                            1.0
                            - (sum(recent_pose_success) / max(len(recent_pose_success), 1))
                            if recent_pose_success
                            else 0.0
                        )
                        pose_initialization_risk_decision = (
                            pose_initialization_risk_gate.evaluate(
                                frame_id=int(frameID),
                                pose_debug=pose_debug_incr,
                                viewpoint_scores=viewpoint_coverage_event,
                                min_num_inliers=int(args.min_num_inliers),
                                recent_pose_fail_rate=float(recent_pose_fail_rate),
                                current_Rt=Rt,
                                pose_history=list(viewpoint_pose_history),
                                baseline_selected=bool(baseline_should_add_frame),
                                is_test=bool(info.get("is_test", False)),
                                is_bootstrap=False,
                            )
                        )
                        pose_initialization_risk_decision.update(
                            {
                                "image_name": str(
                                    info.get("image_name", info.get("name", ""))
                                ),
                                "initial_estimated_Rt": _pose_matrix_for_trace(Rt),
                                "estimated_Rt": _pose_matrix_for_trace(Rt),
                                "gt_Rt": _pose_matrix_for_trace(info.get("Rt")),
                            }
                        )
                        info["_pose_initialization_risk"] = dict(
                            pose_initialization_risk_decision
                        )
                        if pose_initialization_risk_mode == "verify_v1":
                            Rt, pose_verification_debug = (
                                pose_initializer.verify_incremental_pose(
                                    Rt,
                                    pose_initialization_risk_decision,
                                    image_width=int(width),
                                    image_height=int(height),
                                    pose_history=list(viewpoint_pose_history),
                                    mad_scale=float(
                                        getattr(args, "pose_verification_mad_scale", 2.5)
                                    ),
                                    min_support=int(
                                        getattr(args, "pose_verification_min_support", 24)
                                    ),
                                    min_relative_median_improvement=float(
                                        getattr(args, "pose_verification_min_improvement", 0.02)
                                    ),
                                    max_p90_ratio=float(
                                        getattr(args, "pose_verification_max_p90_ratio", 1.01)
                                    ),
                                    min_support_ratio=float(
                                        getattr(args, "pose_verification_min_support_ratio", 0.80)
                                    ),
                                )
                            )
                            pose_verification_debug.update(
                                {
                                    "frame_id": int(frameID),
                                    "image_name": str(info.get("image_name", info.get("name", ""))),
                                    "gt_Rt": _pose_matrix_for_trace(info.get("Rt")),
                                }
                            )
                            pose_initialization_risk_decision.update(
                                {
                                    "verification_attempted": bool(
                                        pose_verification_debug.get("attempted", False)
                                    ),
                                    "verification_accepted": bool(
                                        pose_verification_debug.get("accepted", False)
                                    ),
                                    "verification_reason": str(
                                        pose_verification_debug.get("reason", "")
                                    ),
                                    "verification": pose_verification_debug,
                                    "estimated_Rt": _pose_matrix_for_trace(Rt),
                                }
                            )
                            info["_pose_initialization_risk"] = dict(
                                pose_initialization_risk_decision
                            )
                        if pose_risk_utility_gate is not None:
                            pose_initialization_risk_decision.update(
                                {
                                    "source_frame_id": int(
                                        info.get(
                                            "_paper_aligned_source_frame_id",
                                            frameID,
                                        )
                                    ),
                                    "image_name": str(info.get("name", "")),
                                    "new_view_event_score": float(
                                        viewpoint_coverage_event.get(
                                            "new_view_event_score", 0.0
                                        )
                                        or 0.0
                                    ),
                                    "estimated_Rt": _pose_matrix_for_trace(Rt),
                                    "gt_Rt": _pose_matrix_for_trace(info.get("gt_Rt")),
                                }
                            )
                    pose_risk_utility_decision = None
                    if pose_risk_utility_gate is not None:
                        render_probe = None
                        if pose_risk_candidate(pose_initialization_risk_decision):
                            render_probe = scene_model.probe_pose_risk_utility(
                                image=image,
                                Rt=Rt,
                                mask=info.get("mask"),
                                downsample=int(
                                    getattr(
                                        args,
                                        "pose_risk_utility_probe_downsample",
                                        4,
                                    )
                                ),
                            )
                            render_probe["new_view_event_score"] = float(
                                viewpoint_coverage_event.get(
                                    "new_view_event_score", 0.0
                                )
                                or 0.0
                            )
                        pose_risk_utility_decision = (
                            pose_risk_utility_gate.evaluate(
                                frame_id=int(frameID),
                                risk_event=pose_initialization_risk_decision,
                                render_probe=render_probe,
                                baseline_selected=bool(baseline_should_add_frame),
                                is_test=bool(info.get("is_test", False)),
                                is_bootstrap=False,
                            )
                        )
                        info["_pose_risk_utility_admission"] = dict(
                            pose_risk_utility_decision
                        )
                        if pose_risk_utility_decision[
                            "pose_reference_quarantined"
                        ]:
                            info["_pose_reference_quarantined"] = True
                    if (
                        runtime_gate is not None
                        or pose_initialization_risk_gate is not None
                    ):
                        try:
                            history_Rt = Rt.detach().cpu().clone()
                        except Exception:
                            history_Rt = Rt
                        viewpoint_pose_history.append((int(frameID), history_Rt))
                        last_viewpoint_coverage_event = dict(viewpoint_coverage_event)
                    direct_keyframe_finalized = True
                    fin_dec = None
                    if (
                        runtime_gate is not None
                        and risk_mode == "paper_aligned_semantic_v1"
                        and (
                            getattr(args, "paper_aligned_recovery_commit_bridge", "true_source_commit")
                            == "true_source_commit"
                            or runtime_gate.direct_density_controller.is_pose_only_baseline_repr_family
                        )
                        and str(getattr(args, "paper_aligned_direct_density_control", "off")) != "off"
                    ):
                        support_triggered = bool(
                            (support_bridge_trace or {}).get("support_triggered_keyframe_gate", False)
                        )
                        anchor_changed = len(scene_model.anchors) > int(
                            getattr(runtime_gate, "_anchor_count_at_last_direct_finalize", 0)
                        )
                        pose_inliers = int(
                            pose_debug_incr.get(
                                "num_miniba_inliers",
                                pose_debug_incr.get("num_pnp_inliers", 0),
                            )
                            or 0
                        )
                        fin_dec = runtime_gate.decide_direct_finalization(
                            frame_id=int(frameID),
                            runtime_action=str(runtime_action or "direct_admit"),
                            baseline_should_add=bool(baseline_should_add_frame),
                            is_test=bool(info.get("is_test", False)),
                            is_bootstrap_phase=False,
                            anchor_changed=anchor_changed,
                            support_triggered=support_triggered,
                            median_displacement=float(
                                dist.median().item() if len(dist) > 0 else 0.0
                            ),
                            displacement_threshold=float(min_displacement),
                            num_matches=int(len(curr_prev_matches.kpts)),
                            min_num_inliers=int(args.min_num_inliers),
                            pose_inliers=pose_inliers,
                            viewpoint_scores=viewpoint_coverage_event,
                        )
                        direct_keyframe_finalized = bool(fin_dec.finalize)
                        dbg = dict(fin_dec.debug or {})
                        baseline_eval_frame = bool(
                            info.get("_baseline_eval_frame", info.get("is_test", False))
                        )
                        if baseline_eval_frame and not direct_keyframe_finalized:
                            direct_keyframe_finalized = True
                            dbg["baseline_eval_frame_forced"] = True
                            dbg["baseline_eval_frame_force_reason"] = (
                                "preserve_official_test_hold_eval_frame"
                            )
                        density_before = float(dbg.get("density_before", 0.0))
                        density_after = (
                            100.0
                            * (runtime_gate._current_keyframe_count() + 1)
                            / float(max(frameID, 1))
                            if direct_keyframe_finalized
                            else density_before
                        )
                        trace_ev = runtime_gate._get_event(frameID)
                        density_state = str(dbg.get("density_state", ""))
                        update_prev_on_hold = runtime_gate.direct_density_controller.should_update_prev_desc_on_hold(
                            str(fin_dec.decision), density_state
                        )
                        update_prev_after_hold = runtime_gate.direct_density_controller.should_update_prev_desc_after_hold(
                            str(fin_dec.decision),
                            density_state,
                            is_test=bool(info.get("is_test", False)),
                            pose_safe_tracking_only=bool(pose_safe_tracking_only),
                        )
                        held_bridge = bool(
                            not direct_keyframe_finalized
                            and update_prev_after_hold
                            and runtime_gate.direct_density_controller.hold_tracking_bridge_mode != "none"
                        )
                        if trace_ev is not None:
                            trace_ev["direct_admit_candidate"] = True
                            trace_ev["direct_keyframe_finalized"] = bool(direct_keyframe_finalized)
                            trace_ev["direct_finalization_decision"] = str(fin_dec.decision)
                            trace_ev["direct_finalization_reason"] = str(fin_dec.reason)
                            trace_ev["stream_memory_controller_enabled"] = bool(
                                dbg.get("stream_memory_controller_enabled", False)
                            )
                            trace_ev["stream_memory_frame_identity"] = str(
                                dbg.get("stream_memory_frame_identity", "")
                            )
                            trace_ev["stream_memory_memory_identity"] = str(
                                dbg.get("stream_memory_memory_identity", "")
                            )
                            trace_ev["stream_memory_write_action"] = str(
                                dbg.get("stream_memory_write_action", "")
                            )
                            trace_ev["stream_memory_candidate_verification_required"] = bool(
                                dbg.get("stream_memory_candidate_verification_required", False)
                            )
                            if not direct_keyframe_finalized:
                                trace_ev["direct_admit_but_held_for_density"] = True
                            if bool(dbg.get("baseline_eval_frame_forced", False)):
                                trace_ev["baseline_eval_frame_forced"] = True
                                trace_ev["direct_keyframe_finalized"] = True
                                trace_ev["direct_finalization_reason"] = str(
                                    dbg.get("baseline_eval_frame_force_reason", "")
                                )
                        density_hold_recovery_enqueued = False
                        density_hold_recovery_bridge_tag = ""
                        if not direct_keyframe_finalized:
                            if runtime_gate.direct_density_controller.should_enqueue_hold_recovery(
                                str(fin_dec.decision)
                            ):
                                density_hold_recovery_enqueued = runtime_gate.enqueue_density_hold_recovery_candidate(
                                    frame_id=int(frameID),
                                    info=info,
                                    evidence=evidence,
                                    hold_decision=str(fin_dec.decision),
                                    hold_reason=str(fin_dec.reason),
                                    density_debug=dbg,
                                )
                                if trace_ev is not None:
                                    density_hold_recovery_bridge_tag = str(
                                        trace_ev.get("density_hold_recovery_bridge_tag", "")
                                    )
                            elif trace_ev is not None:
                                trace_ev["density_hold_recovery_enqueued"] = False
                                trace_ev["density_hold_recovery_bridge_tag"] = "pose_only_tracking_hold"
                                density_hold_recovery_bridge_tag = "pose_only_tracking_hold"
                        pose_only_reference_registered = False
                        pose_only_reference_pool_size = 0
                        pose_only_registration_debug = dict(dbg)
                        pose_only_registration_reason = ""
                        if pose_safe_tracking_only:
                            pose_only_registration_debug["active_memory_context"] = True
                            pose_only_registration_debug["active_memory_frame_role"] = "tracking_only"
                            pose_only_registration_debug["stream_memory_frame_identity"] = "pose-only"
                            pose_only_registration_debug["stream_memory_write_action"] = "pose_safe_tracking_only_write"
                            pose_only_registration_reason = "pose_safe_tracking_only"
                        elif held_bridge:
                            pose_only_registration_reason = "held_bridge"
                        should_register_pose_only_reference = bool(
                            not direct_keyframe_finalized
                            and (held_bridge or pose_safe_tracking_only)
                            and runtime_gate.direct_density_controller.is_pose_rep_active_memory
                        )
                        if should_register_pose_only_reference:
                            pose_only_reference_registered = runtime_gate.register_pose_only_reference(
                                frame_id=int(frameID),
                                info=info,
                                desc_kpts=desc_kpts,
                                Rt=Rt,
                                density_debug=pose_only_registration_debug,
                                pose_debug=pose_debug_incr,
                                pose_support=getattr(
                                    pose_initializer,
                                    "last_incremental_pose_support",
                                    {},
                                ),
                            )
                            pose_only_reference_pool_size = int(
                                runtime_gate.pose_only_reference_pool_summary()["pool_size"]
                            )
                            if trace_ev is not None:
                                trace_ev["pose_only_reference_registered"] = bool(
                                    pose_only_reference_registered
                                )
                                trace_ev["pose_only_reference_pool_size"] = int(
                                    pose_only_reference_pool_size
                                )
                                trace_ev["pose_only_registration_reason"] = str(
                                    pose_only_registration_reason
                                )
                        v2_payload = {
                            "frame_id": int(frameID),
                            "source_frame_id": int(frameID),
                            "direct_admit_candidate": True,
                            "pose_safe_tracking_only": bool(pose_safe_tracking_only),
                            "direct_keyframe_finalized": bool(direct_keyframe_finalized),
                            "direct_finalization_decision": str(fin_dec.decision),
                            "direct_finalization_reason": str(fin_dec.reason),
                            "density_state": density_state,
                            "density_before": density_before,
                            "density_after": density_after,
                            "local_density_before": float(dbg.get("local_density_before", density_before)),
                            "baseline_relative_density": float(
                                dbg.get("baseline_relative_density", 0.0)
                            ),
                            "keyframe_count_current": int(dbg.get("keyframe_count_current", 0)),
                            "expected_min_keyframes": float(dbg.get("expected_min_keyframes", 0.0)),
                            "keyframe_debt": float(dbg.get("keyframe_debt", 0.0)),
                            "recent_keyframe_growth": int(
                                dbg.get("recent_keyframe_growth", dbg.get("keyframe_growth_recent", 0))
                            ),
                            "keyframe_growth_recent": int(dbg.get("keyframe_growth_recent", 0)),
                            "starvation_risk": bool(dbg.get("starvation_risk", False)),
                            "starvation_preempts_hold": bool(dbg.get("starvation_preempts_hold", False)),
                            "hold_density_high_allowed": bool(dbg.get("hold_density_high_allowed", False)),
                            "hold_density_high_blocked_by_starvation": bool(
                                dbg.get("hold_density_high_blocked_by_starvation", False)
                            ),
                            "hold_redundant_allowed": bool(dbg.get("hold_redundant_allowed", False)),
                            "source_gap_to_last_keyframe": int(
                                dbg.get("source_gap_to_last_keyframe", 0)
                            ),
                            "main_chain_gap_before": float(dbg.get("main_chain_gap_before", 0.0)),
                            "main_chain_gap_after_if_hold": float(
                                dbg.get("main_chain_gap_after_if_hold", 0.0)
                            ),
                            "high_novelty_score": float(dbg.get("high_novelty_score", 0.0)),
                            "support_needed_score": float(dbg.get("support_needed_score", 0.0)),
                            "novelty_value_score": float(dbg.get("novelty_value_score", 0.0)),
                            "representation_value_score": float(
                                dbg.get("representation_value_score", 0.0)
                            ),
                            "pose_reference_value_score": float(
                                dbg.get("pose_reference_value_score", 0.0)
                            ),
                            "recovery_pool_size": int(dbg.get("recovery_pool_size", 0)),
                            "utility_pose_reference": float(
                                dbg.get("utility_pose_reference", 0.0)
                            ),
                            "utility_representation": float(
                                dbg.get("utility_representation", 0.0)
                            ),
                            "utility_coverage_gain": float(
                                dbg.get("utility_coverage_gain", 0.0)
                            ),
                            "utility_recovery_gain": float(
                                dbg.get("utility_recovery_gain", 0.0)
                            ),
                            "utility_recovery_pressure_score": float(
                                dbg.get("utility_recovery_pressure_score", 0.0)
                            ),
                            "utility_compute_cost": float(
                                dbg.get("utility_compute_cost", 0.0)
                            ),
                            "utility_drift_risk": float(
                                dbg.get("utility_drift_risk", 0.0)
                            ),
                            "utility_view_change": float(
                                dbg.get("utility_view_change", 0.0)
                            ),
                            "utility_gap_pressure": float(
                                dbg.get("utility_gap_pressure", 0.0)
                            ),
                            "utility_total": float(dbg.get("utility_total", 0.0)),
                            "utility_frame_role": str(
                                dbg.get("utility_frame_role", "")
                            ),
                            "utility_tracking_only_role": bool(
                                dbg.get("utility_tracking_only_role", False)
                            ),
                            "utility_representation_role": bool(
                                dbg.get("utility_representation_role", False)
                            ),
                            "utility_recovery_pressure_context": bool(
                                dbg.get("utility_recovery_pressure_context", False)
                            ),
                            "utility_hard_window_guard": bool(
                                dbg.get("utility_hard_window_guard", False)
                            ),
                            "utility_tracking_safe_context": bool(
                                dbg.get("utility_tracking_safe_context", False)
                            ),
                            "pose_memory_geometry_context_enabled": bool(
                                dbg.get("pose_memory_geometry_context_enabled", False)
                            ),
                            "pose_memory_geometry_context_score": float(
                                dbg.get("pose_memory_geometry_context_score", 0.0)
                            ),
                            "pose_memory_geometry_quality": float(
                                dbg.get("pose_memory_geometry_quality", 0.0)
                            ),
                            "pose_memory_geometry_new_view_risk": float(
                                dbg.get("pose_memory_geometry_new_view_risk", 0.0)
                            ),
                            "pose_memory_geometry_tracking_context": bool(
                                dbg.get("pose_memory_geometry_tracking_context", False)
                            ),
                            "pose_memory_geometry_guard": bool(
                                dbg.get("pose_memory_geometry_guard", False)
                            ),
                            "pose_memory_reference_count": int(
                                dbg.get("pose_memory_reference_count", 0)
                            ),
                            "pose_memory_pool_size": int(
                                dbg.get("pose_memory_pool_size", 0)
                            ),
                            "pose_memory_candidate_pool_size": int(
                                dbg.get("pose_memory_candidate_pool_size", 0)
                            ),
                            "pose_memory_context_without_candidate_pool": bool(
                                dbg.get("pose_memory_context_without_candidate_pool", False)
                            ),
                            "stream_memory_controller_enabled": bool(
                                dbg.get("stream_memory_controller_enabled", False)
                            ),
                            "stream_memory_frame_identity": str(
                                dbg.get("stream_memory_frame_identity", "")
                            ),
                            "stream_memory_memory_identity": str(
                                dbg.get("stream_memory_memory_identity", "")
                            ),
                            "stream_memory_write_action": str(
                                dbg.get("stream_memory_write_action", "")
                            ),
                            "stream_memory_new_view_risk": float(
                                dbg.get("stream_memory_new_view_risk", 0.0)
                            ),
                            "stream_memory_geometry_safety": float(
                                dbg.get("stream_memory_geometry_safety", 0.0)
                            ),
                            "stream_memory_representation_need": float(
                                dbg.get("stream_memory_representation_need", 0.0)
                            ),
                            "stream_memory_pose_need": float(
                                dbg.get("stream_memory_pose_need", 0.0)
                            ),
                            "stream_memory_candidate_score": float(
                                dbg.get("stream_memory_candidate_score", 0.0)
                            ),
                            "stream_memory_candidate_verification_required": bool(
                                dbg.get("stream_memory_candidate_verification_required", False)
                            ),
                            "stream_memory_candidate_budget_used": int(
                                dbg.get("stream_memory_candidate_budget_used", 0)
                            ),
                            "stream_memory_pose_only_context": bool(
                                dbg.get("stream_memory_pose_only_context", False)
                            ),
                            "stream_memory_sparse_write": bool(
                                dbg.get("stream_memory_sparse_write", False)
                            ),
                            "pose_risk_score": float(dbg.get("pose_risk_score", 0.0)),
                            "pose_risk_high": bool(dbg.get("pose_risk_high", False)),
                            "motion_value_score": float(dbg.get("motion_value_score", 0.0)),
                            "match_support_score": float(dbg.get("match_support_score", 0.0)),
                            "pose_support_score": float(dbg.get("pose_support_score", 0.0)),
                            "num_matches": int(dbg.get("num_matches", 0)),
                            "pose_inliers": int(dbg.get("pose_inliers", 0)),
                            "source_redundancy_score": float(
                                dbg.get("source_redundancy_score", 0.0)
                            ),
                            "representation_redundancy_penalty": float(
                                dbg.get("representation_redundancy_penalty", 0.0)
                            ),
                            "semantic_R_t": float(dbg.get("semantic_R_t", 0.0)),
                            "semantic_V_t": float(dbg.get("semantic_V_t", 0.0)),
                            "semantic_Q_t": float(dbg.get("semantic_Q_t", 0.0)),
                            "semantic_C_t": float(dbg.get("semantic_C_t", 0.0)),
                            "semantic_B_R_t": float(dbg.get("semantic_B_R_t", 0.0)),
                            "value_hold_allowed": bool(dbg.get("value_hold_allowed", False)),
                            "value_hold_budget_per_100": int(
                                dbg.get("value_hold_budget_per_100", 0)
                            ),
                            "value_hold_budget_used": int(
                                dbg.get("value_hold_budget_used", 0)
                            ),
                            "value_hold_budget_available": bool(
                                dbg.get("value_hold_budget_available", False)
                            ),
                            "bootstrap_value_hold_guard": bool(
                                dbg.get("bootstrap_value_hold_guard", False)
                            ),
                            "value_hold_block_reason": str(
                                dbg.get("value_hold_block_reason", "")
                            ),
                            "high_recent_growth_representation_guard": bool(
                                dbg.get("high_recent_growth_representation_guard", False)
                            ),
                            "low_semantic_coverage_representation_guard": bool(
                                dbg.get("low_semantic_coverage_representation_guard", False)
                            ),
                            "anchor_boundary_representation_guard": bool(
                                dbg.get("anchor_boundary_representation_guard", False)
                            ),
                            "long_sequence_maturity_guard": bool(
                                dbg.get("long_sequence_maturity_guard", False)
                            ),
                            "long_stream_low_growth_context": bool(
                                dbg.get("long_stream_low_growth_context", False)
                            ),
                            "active_memory_context": bool(
                                dbg.get("active_memory_context", False)
                            ),
                            "active_memory_context_candidate": bool(
                                dbg.get("active_memory_context_candidate", False)
                            ),
                            "active_memory_frame_role": str(
                                dbg.get("active_memory_frame_role", "")
                            ),
                            "active_memory_marginal_value": float(
                                dbg.get("active_memory_marginal_value", 0.0)
                            ),
                            "active_memory_redundancy_pressure": float(
                                dbg.get("active_memory_redundancy_pressure", 0.0)
                            ),
                            "active_memory_redundancy_pressure_high": bool(
                                dbg.get("active_memory_redundancy_pressure_high", False)
                            ),
                            "active_memory_low_parallax": bool(
                                dbg.get("active_memory_low_parallax", False)
                            ),
                            "active_memory_stable_pose_reference": bool(
                                dbg.get("active_memory_stable_pose_reference", False)
                            ),
                            "active_memory_low_representation_value": bool(
                                dbg.get("active_memory_low_representation_value", False)
                            ),
                            "active_memory_low_marginal_representation": bool(
                                dbg.get("active_memory_low_marginal_representation", False)
                            ),
                            "density_only_hold_disabled": bool(
                                dbg.get("density_only_hold_disabled", False)
                            ),
                            "hold_low_representation_value": bool(
                                dbg.get("hold_low_representation_value", False)
                            ),
                            "finalize_high_representation_value": bool(
                                dbg.get("finalize_high_representation_value", False)
                            ),
                            "finalize_pose_risk_reference": bool(
                                dbg.get("finalize_pose_risk_reference", False)
                            ),
                            "high_novelty_budget_used": int(dbg.get("high_novelty_budget_used", 0)),
                            "support_needed_budget_used": int(dbg.get("support_needed_budget_used", 0)),
                            "high_novelty_budget_exhausted": bool(
                                dbg.get("high_novelty_budget_exhausted", False)
                            ),
                            "support_needed_budget_exhausted": bool(
                                dbg.get("support_needed_budget_exhausted", False)
                            ),
                            "hold_redundant_allowed": bool(dbg.get("hold_redundant_allowed", False)),
                            "hold_redundant_blocked_by_lower_guard": bool(
                                dbg.get("hold_redundant_blocked_by_lower_guard", False)
                            ),
                            "hold_density_high": bool(dbg.get("hold_density_high", False)),
                            "finalize_high_novelty": bool(dbg.get("finalize_high_novelty", False)),
                            "finalize_support_needed": bool(dbg.get("finalize_support_needed", False)),
                            "finalize_gap_critical": bool(dbg.get("finalize_gap_critical", False)),
                            "finalize_growth_rescue": fin_dec.decision == "finalize_growth_rescue",
                            "prev_desc_updated_on_hold": False,
                            "held_frame_used_for_tracking_bridge": held_bridge,
                            "tracking_bridge_mode": str(
                                runtime_gate.direct_density_controller.hold_tracking_bridge_mode
                            ),
                            "keyframe_finalized": bool(direct_keyframe_finalized),
                            "representation_updated": bool(direct_keyframe_finalized),
                            "blocked_override_reason": str(dbg.get("blocked_override_reason", "")),
                            "density_hold_recovery_enqueued": bool(
                                density_hold_recovery_enqueued
                            ),
                            "density_hold_recovery_bridge_tag": density_hold_recovery_bridge_tag,
                            "pose_only_registration_reason": str(
                                pose_only_registration_reason
                            ),
                            "pose_only_reference_registered": bool(
                                pose_only_reference_registered
                            ),
                            "pose_only_reference_pool_size": int(
                                pose_only_reference_pool_size
                            ),
                            "pose_only_reference_selected_count": int(
                                len(pose_only_refs) if "pose_only_refs" in locals() else 0
                            ),
                            "viewpoint_rotation_deg_to_last_keyframe": float(
                                viewpoint_coverage_event.get(
                                    "viewpoint_rotation_deg_to_last_keyframe", 0.0
                                )
                            ),
                            "viewpoint_rotation_deg_to_active_anchor": float(
                                viewpoint_coverage_event.get(
                                    "viewpoint_rotation_deg_to_active_anchor", 0.0
                                )
                            ),
                            "viewpoint_rotation_deg_window_20": float(
                                viewpoint_coverage_event.get(
                                    "viewpoint_rotation_deg_window_20", 0.0
                                )
                            ),
                            "viewpoint_rotation_deg_window_50": float(
                                viewpoint_coverage_event.get(
                                    "viewpoint_rotation_deg_window_50", 0.0
                                )
                            ),
                            "viewpoint_rotation_deg_window_100": float(
                                viewpoint_coverage_event.get(
                                    "viewpoint_rotation_deg_window_100", 0.0
                                )
                            ),
                            "viewpoint_rotation_deg_window_max": float(
                                viewpoint_coverage_event.get(
                                    "viewpoint_rotation_deg_window_max", 0.0
                                )
                            ),
                            "viewpoint_rotation_window_max_size": int(
                                viewpoint_coverage_event.get(
                                    "viewpoint_rotation_window_max_size", 0
                                )
                            ),
                            "inlier_grid_coverage": float(
                                viewpoint_coverage_event.get("inlier_grid_coverage", 0.0)
                            ),
                            "inlier_grid_entropy": float(
                                viewpoint_coverage_event.get("inlier_grid_entropy", 0.0)
                            ),
                            "support_concentration": float(
                                viewpoint_coverage_event.get("support_concentration", 0.0)
                            ),
                            "anchor_health_score": float(
                                viewpoint_coverage_event.get("anchor_health_score", 0.0)
                            ),
                            "new_view_event_score": float(
                                viewpoint_coverage_event.get("new_view_event_score", 0.0)
                            ),
                        }
                        if viewpoint_coverage_event:
                            viewpoint_coverage_event.update(
                                {
                                    "source_frame_id": int(frameID),
                                    "direct_keyframe_finalized": bool(
                                        direct_keyframe_finalized
                                    ),
                                    "direct_finalization_decision": str(
                                        fin_dec.decision
                                    ),
                                    "direct_finalization_reason": str(fin_dec.reason),
                                    "active_memory_frame_role": str(
                                        dbg.get("active_memory_frame_role", "")
                                    ),
                                    "utility_frame_role": str(
                                        dbg.get("utility_frame_role", "")
                                    ),
                                    "utility_pose_reference": float(
                                        dbg.get("utility_pose_reference", 0.0)
                                    ),
                                    "utility_representation": float(
                                        dbg.get("utility_representation", 0.0)
                                    ),
                                    "utility_compute_cost": float(
                                        dbg.get("utility_compute_cost", 0.0)
                                    ),
                                    "utility_drift_risk": float(
                                        dbg.get("utility_drift_risk", 0.0)
                                    ),
                                    "utility_hard_window_guard": bool(
                                        dbg.get("utility_hard_window_guard", False)
                                    ),
                                    "utility_tracking_safe_context": bool(
                                        dbg.get("utility_tracking_safe_context", False)
                                    ),
                                    "stream_memory_controller_enabled": bool(
                                        dbg.get("stream_memory_controller_enabled", False)
                                    ),
                                    "stream_memory_frame_identity": str(
                                        dbg.get("stream_memory_frame_identity", "")
                                    ),
                                    "stream_memory_memory_identity": str(
                                        dbg.get("stream_memory_memory_identity", "")
                                    ),
                                    "stream_memory_write_action": str(
                                        dbg.get("stream_memory_write_action", "")
                                    ),
                                    "stream_memory_new_view_risk": float(
                                        dbg.get("stream_memory_new_view_risk", 0.0)
                                    ),
                                    "stream_memory_geometry_safety": float(
                                        dbg.get("stream_memory_geometry_safety", 0.0)
                                    ),
                                    "stream_memory_representation_need": float(
                                        dbg.get("stream_memory_representation_need", 0.0)
                                    ),
                                    "stream_memory_pose_need": float(
                                        dbg.get("stream_memory_pose_need", 0.0)
                                    ),
                                    "stream_memory_candidate_verification_required": bool(
                                        dbg.get(
                                            "stream_memory_candidate_verification_required",
                                            False,
                                        )
                                    ),
                                    "value_hold_block_reason": str(
                                        dbg.get("value_hold_block_reason", "")
                                    ),
                                }
                            )
                            runtime_gate.append_viewpoint_coverage_event(
                                viewpoint_coverage_event
                            )
                        augment_pose_render_payload_with_posterior_risk(
                            v2_payload,
                            pose_debug=pose_debug_incr,
                            viewpoint_scores=viewpoint_coverage_event,
                            min_num_inliers=int(args.min_num_inliers),
                        )
                        if runtime_gate.direct_density_controller.is_v2221:
                            v2_payload.update(
                                {
                                    "local_window_density": float(
                                        dbg.get("local_window_density", 0.0)
                                    ),
                                    "local_window_gap_before": float(
                                        dbg.get("local_window_gap_before", 0.0)
                                    ),
                                    "local_window_gap_after_if_hold": float(
                                        dbg.get("local_window_gap_after_if_hold", 0.0)
                                    ),
                                    "density_cap": float(dbg.get("density_cap", 0.0)),
                                    "density_cap_exception_for_hard_gap": bool(
                                        dbg.get("density_cap_exception_for_hard_gap", False)
                                    ),
                                    "post500_pre_gap_rescue_triggered": bool(
                                        dbg.get("post500_pre_gap_rescue_triggered", False)
                                    ),
                                    "post500_pre_gap_rescue_finalized": bool(
                                        dbg.get("post500_pre_gap_rescue_finalized", False)
                                    )
                                    or fin_dec.decision
                                    == "finalize_post500_pre_gap_rescue_v2_2_2_1",
                                    "post500_gap_rescue_budget_used": int(
                                        dbg.get("post500_gap_rescue_budget_used", 0)
                                    ),
                                    "post500_gap_rescue_budget_exhausted": bool(
                                        dbg.get("post500_gap_rescue_budget_exhausted", False)
                                    ),
                                    "hard_gap_rescue_triggered": bool(
                                        dbg.get("hard_gap_rescue_triggered", False)
                                    ),
                                    "hard_gap_rescue_finalized": bool(
                                        dbg.get("hard_gap_rescue_finalized", False)
                                    )
                                    or fin_dec.decision
                                    == "finalize_hard_gap_rescue_v2_2_2_1",
                                    "hard_gap_soft_budget_bypassed": bool(
                                        dbg.get("hard_gap_soft_budget_bypassed", False)
                                    ),
                                    "hard_gap_density_cap_exception": bool(
                                        dbg.get("hard_gap_density_cap_exception", False)
                                    ),
                                    "candidate_missing_context": bool(
                                        dbg.get("candidate_missing_context", False)
                                    ),
                                    "predicted_next_gap_if_no_future_candidate": float(
                                        dbg.get("predicted_next_gap_if_no_future_candidate", 0.0)
                                    ),
                                    "hold_gap_rescue_budget_exhausted": bool(
                                        dbg.get("gap_rescue_budget_exhausted", False)
                                    ),
                                    "hold_density_high_blocked_by_hard_gap": bool(
                                        dbg.get("hold_density_high_blocked_by_hard_gap", False)
                                    ),
                                    "hold_redundant_blocked_by_hard_gap": bool(
                                        dbg.get("hold_redundant_blocked_by_hard_gap", False)
                                    ),
                                    "hold_density_high_blocked_by_pre_gap": bool(
                                        dbg.get("hold_density_high_blocked_by_pre_gap", False)
                                    ),
                                }
                            )
                            runtime_gate.append_direct_density_control_v2_2_2_1_event(v2_payload)
                        elif runtime_gate.direct_density_controller.is_v222:
                            v2_payload.update(
                                {
                                    "local_window_density": float(
                                        dbg.get("local_window_density", 0.0)
                                    ),
                                    "local_window_gap_before": float(
                                        dbg.get("local_window_gap_before", 0.0)
                                    ),
                                    "local_window_gap_after_if_hold": float(
                                        dbg.get("local_window_gap_after_if_hold", 0.0)
                                    ),
                                    "soft_gap_threshold": int(dbg.get("soft_gap_threshold", 6)),
                                    "preemptive_gap_threshold": int(
                                        dbg.get("preemptive_gap_threshold", 18)
                                    ),
                                    "hard_gap_threshold": int(dbg.get("hard_gap_threshold", 20)),
                                    "soft_gap_rescue_triggered": bool(
                                        dbg.get("soft_gap_rescue_triggered", False)
                                    ),
                                    "preemptive_gap_rescue_triggered": bool(
                                        dbg.get("preemptive_gap_rescue_triggered", False)
                                    ),
                                    "hard_gap_rescue_triggered": bool(
                                        dbg.get("hard_gap_rescue_triggered", False)
                                    ),
                                    "density_cap_for_gap_rescue": float(
                                        dbg.get("density_cap_for_gap_rescue", 0.0)
                                    ),
                                    "density_cap_exceeded": bool(
                                        dbg.get("density_cap_exceeded", False)
                                    ),
                                    "gap_rescue_budget_used": int(
                                        dbg.get("gap_rescue_budget_used", 0)
                                    ),
                                    "gap_rescue_budget_exhausted": bool(
                                        dbg.get("gap_rescue_budget_exhausted", False)
                                    ),
                                    "finalize_gap_tail_rescue_v2_2_1": bool(
                                        dbg.get("finalize_gap_tail_rescue_v2_2_1", False)
                                    )
                                    or fin_dec.decision == "finalize_gap_tail_rescue_v2_2_1",
                                    "finalize_gap_tail_preemptive_v2_2_2": bool(
                                        dbg.get("finalize_gap_tail_preemptive_v2_2_2", False)
                                    )
                                    or fin_dec.decision
                                    == "finalize_gap_tail_preemptive_v2_2_2",
                                    "finalize_hard_gap_rescue_v2_2_2": bool(
                                        dbg.get("finalize_hard_gap_rescue_v2_2_2", False)
                                    )
                                    or fin_dec.decision == "finalize_hard_gap_rescue_v2_2_2",
                                    "hold_density_high_blocked_by_gap": bool(
                                        dbg.get("hold_density_high_blocked_by_gap", False)
                                    ),
                                    "hold_redundant_blocked_by_gap": bool(
                                        dbg.get("hold_redundant_blocked_by_gap", False)
                                    ),
                                    "hold_gap_rescue_blocked_by_density_cap": bool(
                                        dbg.get("hold_gap_rescue_blocked_by_density_cap", False)
                                    ),
                                }
                            )
                            runtime_gate.append_direct_density_control_v2_2_2_event(v2_payload)
                        elif runtime_gate.direct_density_controller.is_v221:
                            v2_payload.update(
                                {
                                    "local_window_density": float(
                                        dbg.get("local_window_density", 0.0)
                                    ),
                                    "local_window_gap_before": float(
                                        dbg.get("local_window_gap_before", 0.0)
                                    ),
                                    "local_window_gap_after_if_hold": float(
                                        dbg.get("local_window_gap_after_if_hold", 0.0)
                                    ),
                                    "soft_gap_threshold": int(dbg.get("soft_gap_threshold", 6)),
                                    "hard_gap_threshold": int(dbg.get("hard_gap_threshold", 20)),
                                    "gap_tail_rescue_triggered": bool(
                                        dbg.get("gap_tail_rescue_triggered", False)
                                    ),
                                    "hard_gap_rescue_triggered": bool(
                                        dbg.get("hard_gap_rescue_triggered", False)
                                    ),
                                    "gap_rescue_budget_used": int(
                                        dbg.get("gap_rescue_budget_used", 0)
                                    ),
                                    "gap_rescue_budget_exhausted": bool(
                                        dbg.get("gap_rescue_budget_exhausted", False)
                                    ),
                                    "finalize_gap_tail_rescue_v2_2_1": bool(
                                        dbg.get("finalize_gap_tail_rescue_v2_2_1", False)
                                    )
                                    or fin_dec.decision == "finalize_gap_tail_rescue_v2_2_1",
                                    "finalize_hard_gap_rescue_v2_2_1": bool(
                                        dbg.get("finalize_hard_gap_rescue_v2_2_1", False)
                                    )
                                    or fin_dec.decision == "finalize_hard_gap_rescue_v2_2_1",
                                    "hold_density_high_blocked_by_gap": bool(
                                        dbg.get("hold_density_high_blocked_by_gap", False)
                                    ),
                                    "hold_redundant_blocked_by_gap": bool(
                                        dbg.get("hold_redundant_blocked_by_gap", False)
                                    ),
                                }
                            )
                            runtime_gate.append_direct_density_control_v2_2_1_event(v2_payload)
                        elif runtime_gate.direct_density_controller.is_v22:
                            v2_payload.update(
                                {
                                    "local_window_density": float(
                                        dbg.get("local_window_density", 0.0)
                                    ),
                                    "local_window_keyframes": int(
                                        dbg.get("local_window_keyframes", 0)
                                    ),
                                    "local_window_gap_max": float(
                                        dbg.get("local_window_gap_max", 0.0)
                                    ),
                                    "local_window_gap_after_if_hold": float(
                                        dbg.get("local_window_gap_after_if_hold", 0.0)
                                    ),
                                    "gap_critical_triggered": bool(
                                        dbg.get("gap_critical_triggered", False)
                                    ),
                                    "gap_critical_finalized": bool(
                                        dbg.get("gap_critical_finalized", False)
                                    ),
                                    "early_rescue_triggered": bool(
                                        dbg.get("early_rescue_triggered", False)
                                    ),
                                    "early_rescue_budget_used": int(
                                        dbg.get("early_rescue_budget_used", 0)
                                    ),
                                    "early_rescue_budget_exhausted": bool(
                                        dbg.get("early_rescue_budget_exhausted", False)
                                    ),
                                    "finalize_early_growth_rescue": bool(
                                        dbg.get("finalize_early_growth_rescue", False)
                                    )
                                    or fin_dec.decision == "finalize_early_growth_rescue",
                                    "local_density_rescue_triggered": bool(
                                        dbg.get("local_density_rescue_triggered", False)
                                    ),
                                    "local_gap_rescue_triggered": bool(
                                        dbg.get("local_gap_rescue_triggered", False)
                                    ),
                                    "finalize_local_density_rescue": bool(
                                        dbg.get("finalize_local_density_rescue", False)
                                    )
                                    or fin_dec.decision == "finalize_local_density_rescue",
                                    "finalize_local_gap_rescue": bool(
                                        dbg.get("finalize_local_gap_rescue", False)
                                    )
                                    or fin_dec.decision == "finalize_local_gap_rescue",
                                    "hold_density_high_blocked_by_gap": bool(
                                        dbg.get("hold_density_high_blocked_by_gap", False)
                                    ),
                                    "hold_density_high_blocked_by_local_under_density": bool(
                                        dbg.get(
                                            "hold_density_high_blocked_by_local_under_density",
                                            False,
                                        )
                                    ),
                                    "hold_redundant_blocked_by_gap": bool(
                                        dbg.get("hold_redundant_blocked_by_gap", False)
                                    ),
                                }
                            )
                            runtime_gate.append_direct_density_control_v2_2_event(v2_payload)
                        elif runtime_gate.direct_density_controller.is_v21:
                            runtime_gate.append_direct_density_control_v2_1_event(v2_payload)
                        elif runtime_gate.direct_density_controller.mode == "target_band_v2":
                            runtime_gate.append_direct_density_control_v2_event(v2_payload)
                        else:
                            runtime_gate.append_direct_density_control_event(
                                {
                                    **v2_payload,
                                    "anchor_id": len(scene_model.anchors) - 1,
                                    "anchor_changed": bool(anchor_changed),
                                    "support_trend_state": (
                                        "support_triggered"
                                        if support_triggered
                                        else "baseline_prev_match"
                                    ),
                                    "novelty_proxy": float(dbg.get("novelty_proxy", 0.0)),
                                    "baseline_would_keep_if_available": bool(
                                        baseline_should_add_frame
                                    ),
                                    "hold_redundant": bool(dbg.get("hold_redundant", False)),
                                }
                            )
                        if not direct_keyframe_finalized:
                            if pose_safe_pose_match_before is not None:
                                _restore_pose_match_state(
                                    desc_kpts,
                                    pose_safe_probe_refs,
                                    n_keyframes,
                                    pose_safe_pose_match_before,
                                )
                                v2_payload["pose_safe_match_restored_on_hold"] = True
                                if trace_ev is not None:
                                    trace_ev["pose_safe_match_restored_on_hold"] = True
                            if pose_safe_tracking_only and pose_safe_pose_rng_before is not None:
                                _restore_torch_rng_state(pose_safe_pose_rng_before)
                                v2_payload["pose_safe_rng_restored_on_hold"] = True
                                if trace_ev is not None:
                                    trace_ev["pose_safe_rng_restored_on_hold"] = True
                            should_add_keyframe = False
                            if update_prev_after_hold:
                                prev_desc_kpts = desc_kpts
                                v2_payload["prev_desc_updated_on_hold"] = True
                                v2_payload["prev_desc_update_role"] = (
                                    "pose_safe_tracking_only"
                                    if pose_safe_tracking_only
                                    else "held_bridge"
                                )
                                if trace_ev is not None:
                                    trace_ev["prev_desc_updated_on_hold"] = True
                                    trace_ev["prev_desc_update_role"] = str(
                                        v2_payload["prev_desc_update_role"]
                                    )
                                if runtime_gate.direct_density_controller.is_v2221:
                                    runtime_gate.direct_density_control_v2_2_2_1_events[-1][
                                        "prev_desc_updated_on_hold"
                                    ] = True
                                elif runtime_gate.direct_density_controller.is_v222:
                                    runtime_gate.direct_density_control_v2_2_2_events[-1][
                                        "prev_desc_updated_on_hold"
                                    ] = True
                                elif runtime_gate.direct_density_controller.is_v221:
                                    runtime_gate.direct_density_control_v2_2_1_events[-1][
                                        "prev_desc_updated_on_hold"
                                    ] = True
                                elif runtime_gate.direct_density_controller.is_v22:
                                    runtime_gate.direct_density_control_v2_2_events[-1][
                                        "prev_desc_updated_on_hold"
                                    ] = True
                                elif runtime_gate.direct_density_controller.is_v21:
                                    runtime_gate.direct_density_control_v2_1_events[-1][
                                        "prev_desc_updated_on_hold"
                                    ] = True
                                elif runtime_gate.direct_density_controller.mode == "target_band_v2":
                                    runtime_gate.direct_density_control_v2_events[-1][
                                        "prev_desc_updated_on_hold"
                                    ] = True
                                elif runtime_gate.direct_density_control_events:
                                    runtime_gate.direct_density_control_events[-1][
                                        "prev_desc_updated_on_hold"
                                    ] = True
                    pose_initialization_isolated = bool(
                        pose_initialization_risk_decision is not None
                        and pose_initialization_risk_decision["isolated"]
                    )
                    pose_risk_utility_isolated = bool(
                        pose_risk_utility_decision is not None
                        and pose_risk_utility_decision["isolated"]
                    )
                    if pose_initialization_isolated or pose_risk_utility_isolated:
                        direct_keyframe_finalized = False
                        should_add_keyframe = False
                        if pose_initialization_isolated:
                            info["_pose_initialization_risk_isolated"] = True
                        if pose_risk_utility_isolated:
                            info["_pose_risk_utility_isolated"] = True
                        if pose_initialization_risk_match_before is not None:
                            _restore_pose_match_state(
                                desc_kpts,
                                prev_keyframes_for_pose,
                                n_keyframes,
                                pose_initialization_risk_match_before,
                            )
                        if runtime_gate is not None:
                            trace_ev = runtime_gate._get_event(frameID)
                            if trace_ev is not None:
                                trace_ev["pose_initialization_risk"] = dict(
                                    pose_initialization_risk_decision
                                )
                                trace_ev["pose_initialization_risk_isolated"] = True
                                trace_ev["admit_to_chain"] = False
                                trace_ev["drop_reason"] = (
                                    "pose_risk_utility_isolated"
                                    if pose_risk_utility_isolated
                                    else "pose_initialization_risk_isolated"
                                )
                    if direct_keyframe_finalized:
                        if runtime_gate is not None and fin_dec is not None:
                            _write_pose_render_coupling_info(
                                info,
                                v2_payload,
                                "direct_density_control",
                            )
                        elif runtime_gate is not None and fin_dec is None:
                            try:
                                baseline_lock_num_matches = int(len(curr_prev_matches.kpts))
                            except Exception:
                                baseline_lock_num_matches = int(
                                    pose_debug_incr.get("match_count_total", 0) or 0
                                )
                            baseline_lock_pose_inliers = int(
                                pose_debug_incr.get(
                                    "num_miniba_inliers",
                                    pose_debug_incr.get("num_pnp_inliers", 0),
                                )
                                or 0
                            )
                            _write_baseline_render_lock_pose_support_info(
                                info,
                                frame_id=int(frameID),
                                pose_debug=pose_debug_incr,
                                viewpoint_scores=(
                                    viewpoint_coverage_event
                                    if "viewpoint_coverage_event" in locals()
                                    else {}
                                ),
                                num_matches=baseline_lock_num_matches,
                                pose_inliers=baseline_lock_pose_inliers,
                            )
                        # 【场景表示模块】创建新关键帧对象
                        keyframe = Keyframe(
                            image,
                            info,
                            desc_kpts,
                            Rt,
                            n_keyframes,
                            f,
                            dense_extractor,
                            depth_estimator,
                            triangulator,
                            args,
                        )
                        scene_model.add_keyframe(keyframe)
                        if runtime_gate is not None:
                            runtime_gate.mark_keyframe_add(frameID)
                            runtime_gate._anchor_count_at_last_direct_finalize = len(
                                scene_model.anchors
                            )
                        prev_keyframe = keyframe
                        increment_runtime(runtimes["Add"], start_time)

                        if (
                            pose_risk_utility_decision is not None
                            and pose_risk_utility_decision["review"]
                        ):
                            start_time = time.time()
                            pose_review_result = (
                                scene_model.review_pose_risk_keyframe(
                                    -1,
                                    iterations=int(
                                        getattr(
                                            args,
                                            "pose_risk_utility_review_iterations",
                                            2,
                                        )
                                    ),
                                    min_render_coverage=float(
                                        getattr(
                                            args,
                                            "pose_risk_utility_review_min_coverage",
                                            0.15,
                                        )
                                    ),
                                    max_rotation_delta_deg=float(
                                        getattr(
                                            args,
                                            "pose_risk_utility_review_max_rotation_deg",
                                            1.5,
                                        )
                                    ),
                                    max_translation_delta=float(
                                        getattr(
                                            args,
                                            "pose_risk_utility_review_max_translation",
                                            0.05,
                                        )
                                    ),
                                )
                            )
                            pose_risk_utility_decision["pose_review"] = dict(
                                pose_review_result
                            )
                            pose_risk_utility_decision["estimated_Rt"] = (
                                _pose_matrix_for_trace(keyframe.get_Rt())
                            )
                            keyframe.info["_pose_risk_utility_admission"] = dict(
                                pose_risk_utility_decision
                            )
                            if viewpoint_pose_history:
                                viewpoint_pose_history[-1] = (
                                    int(frameID),
                                    keyframe.get_Rt().detach().cpu().clone(),
                                )
                            increment_runtime(runtimes["Opt"], start_time)

                        start_time = time.time()
                        scene_model.pose_render_pre_refine_keyframe(-1)
                        increment_runtime(runtimes["Opt"], start_time)

                        # 【场景表示模块】为新关键帧初始化3D高斯点
                        # 使用Laplacian概率采样 + 引导MVS深度估计
                        start_time = time.time()
                        scene_model.add_new_gaussians()
                        if runtime_gate is not None:
                            runtime_gate.mark_gaussian_update(frameID)
                        increment_runtime(runtimes["Init"], start_time)

                        start_time = time.time()
                        # 【优化模块】优化场景：流式使用异步优化，离线直接循环优化
                        if is_stream:
                            scene_model.optimize_async(args.num_iterations)
                        else:
                            scene_model.optimization_loop(args.num_iterations)
                        increment_runtime(runtimes["Opt"], start_time)
                else:
                    # 姿态估计失败，跳过该帧
                    should_add_keyframe = False
                    if runtime_gate is not None:
                        if pose_safe_pose_match_before is not None:
                            _restore_pose_match_state(
                                desc_kpts,
                                pose_safe_probe_refs,
                                n_keyframes,
                                pose_safe_pose_match_before,
                            )
                            trace_ev = runtime_gate._get_event(frameID)
                            if trace_ev is not None:
                                trace_ev["pose_safe_match_restored_on_pose_fail"] = True
                        if pose_safe_tracking_only and pose_safe_pose_rng_before is not None:
                            _restore_torch_rng_state(pose_safe_pose_rng_before)
                            trace_ev = runtime_gate._get_event(frameID)
                            if trace_ev is not None:
                                trace_ev["pose_safe_rng_restored_on_pose_fail"] = True
                        pose_debug = getattr(pose_initializer, "last_incremental_debug", {})
                        fail_reason = str(pose_debug.get("failure_reason", "") or "pose_init_failed")
                        runtime_gate.mark_drop_reason(frameID, fail_reason)

        if should_add_keyframe:
            # ========== 锚点管理：处理大尺度场景 ==========
            # 【场景表示模块】检查是否需要创建新锚点
            # 当高斯点在屏幕上变得过小时，创建新锚点并合并细粒度高斯点
            start_time = time.time()
            scene_model.place_anchor_if_needed()
            _observe_defer_recovery_anchor_bridge()
            if runtime_gate is not None:
                runtime_gate.mark_anchor_update(frameID)
                if "prev_keyframe" in locals():
                    _append_keyframe_timeline(prev_keyframe, frameID, "direct_admit")
                _append_local_map_anchor_trace(frameID)
            increment_runtime(runtimes["anc"], start_time)

            n_keyframes += 1
            if runtime_gate is not None:
                runtime_gate.mark_final_keyframe_increment(frameID)
            # 更新上一帧的描述子（用于下一帧的匹配）
            if not info["is_test"]:
                prev_desc_kpts = desc_kpts

            # ========== 中间评估 ==========
            # 定期评估重建质量（PSNR、SSIM、LPIPS等）
            if (
                n_keyframes % args.test_frequency == 0
                and args.test_frequency > 0
                and (args.test_hold > 0 or args.eval_poses)
            ):
                metrics = scene_model.evaluate(args.eval_poses)

            # ========== 中间保存 ==========
            # 定期保存重建进度（用于断点续训或检查）
            if (
                frameID % args.save_every == 0
                and args.save_every > 0
            ):
                scene_model.save(
                    os.path.join(args.model_path, "progress", f"{frameID:05d}")
                )

            # ========== 进度显示 ==========
            # 在进度条中显示评估指标、运行时间、场景统计等信息
            bar_postfix = []
            for key, value in metrics.items():
                bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            if args.display_runtimes:
                for key, value in runtimes.items():
                    if value[1] > 0:
                        bar_postfix += [
                            f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"  # 平均时间(ms)
                        ]
            bar_postfix += [
                f"\033[36mFocal:{focal:.1f}",
                f"\033[36mKeyframes:{n_keyframes}\033[0m",
                f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
                f"\033[36mAnchors:{len(scene_model.anchors)}\033[0m",
            ]
            pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    reconstruction_time = time.time() - reconstruction_start_time
    if runtime_gate is not None:
        runtime_gate.flush_trace()

    # ========== 重建完成后的处理 ==========
    # 【场景表示模块】切换为推理模式（停止优化，启用锚点融合，准备渲染）
    scene_model.enable_inference_mode()

    # 【保存模块】保存最终模型与评估指标
    print("Saving the reconstruction to:", args.model_path)
    metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset))
    print(
        ", ".join(
            f"{metric}: {value:.3f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
            for metric, value in metrics.items()
        )
    )

    # ========== 可选微调阶段 ==========
    # 初始重建完成后，可以对所有锚点进行全局微调以进一步提升质量
    if len(args.save_at_finetune_epoch) > 0:
        finetune_epochs = max(args.save_at_finetune_epoch)
        torch.cuda.empty_cache()
        scene_model.inference_mode = False
        pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
        for epoch in pbar:
            # 【优化模块】执行一轮全局微调（遍历所有锚点）
            epoch_start_time = time.time()
            scene_model.finetune_epoch()
            epoch_time = time.time() - epoch_start_time
            reconstruction_time += epoch_time
            # 按需保存微调结果（用于检查不同epoch的效果）
            if epoch + 1 in args.save_at_finetune_epoch:
                torch.cuda.empty_cache()
                scene_model.inference_mode = True
                metrics = scene_model.save(
                    os.path.join(args.model_path, str(epoch + 1)), reconstruction_time
                )
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                pbar.set_postfix_str(",".join(bar_postfix))
                scene_model.inference_mode = False
                torch.cuda.empty_cache()
                
        # 设置为推理模式以便正确渲染
        scene_model.inference_mode = True

    # ========== 保持可视化窗口运行 ==========
    # 训练完成后，保持可视化窗口运行以便用户查看结果
    if args.viewer_mode != "none":
        if args.viewer_mode == "web":
            # 网页模式：保持服务器运行
            while True:
                time.sleep(1)
        else:
            viewer.throttling = False  # 训练完成后禁用节流，提高渲染质量
            # 保持本地/服务器端viewer存活
            while viewer.running:
                time.sleep(1)
