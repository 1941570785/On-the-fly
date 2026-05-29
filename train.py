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
from paper_aligned_policy.runtime_gate import PaperAlignedRuntimeGate
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
    runtime_gate = None
    if risk_mode != "off":
        print(
            f"[risk_admission_mode={risk_mode}] contract runtime gate enabled "
            "(off mode remains baseline path)."
        )
        runtime_gate = PaperAlignedRuntimeGate(args)
        atexit.register(runtime_gate.flush_trace)
        def _flush_chosen_kfs_resolution_events():
            events = pop_chosen_kfs_resolution_events()
            out = Path(args.model_path) / "chosen_kfs_resolution_events.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        atexit.register(_flush_chosen_kfs_resolution_events)

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

    # Dict of runtimes for each step
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    metrics = {}

    runtimes = {key: [0, 0] for key in runtimes}
    ## 场景重建主循环
    print(f"Starting reconstruction for {args.source_path}")
    total_frames = len(dataset) if args.max_frames is None or args.max_frames < 0 else min(len(dataset), int(args.max_frames))
    pbar = tqdm(range(0, total_frames))
    reconstruction_start_time = time.time()

    def _commit_true_source_frame(recovered: dict, control_decision: dict[str, Any] | None = None) -> None:
        global n_keyframes
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
            "commit_control_decision": "commit",
            "commit_control_reason": "",
        }
        if isinstance(control_decision, dict):
            recovery_trace["commit_control_decision"] = str(control_decision.get("decision", "commit"))
            recovery_trace["commit_control_reason"] = str(control_decision.get("decision_reason", ""))
            recovery_trace["commit_control_debug"] = dict(control_decision.get("debug", {}))
        runtime_gate.mark_true_source_recovery_attempt(
            source_frame_id=source_frame_id,
            current_tick_frame_id=int(recovered.get("current_tick_frame_id", -1)),
            current_tick_image_name=str(recovered.get("current_tick_image_name", "")),
            pool_enter_tick=int(recovered.get("pool_enter_tick", -1)),
            recovery_attempt_tick=int(recovered.get("recovery_attempt_tick", -1)),
        )
        event = runtime_gate._get_event(source_frame_id)
        if event is not None and bool(event.get("final_keyframe_incremented", False)):
            recovery_trace["failure_stage"] = "duplicate_gate"
            recovery_trace["failure_reason"] = "source_already_final_keyframe"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="source_already_final_keyframe"
            )
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if n_keyframes < args.num_keyframes_miniba_bootstrap:
            recovery_trace["failure_stage"] = "bootstrap_gate"
            recovery_trace["failure_reason"] = "bootstrap_not_ready"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="bootstrap_not_ready"
            )
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if source_image is None or source_desc is None:
            recovery_trace["failure_stage"] = "source_payload"
            recovery_trace["failure_reason"] = "missing_source_image_or_features"
            recovery_trace["materialization_failure_reason"] = "missing_source_image_or_features"
            runtime_gate.mark_true_source_recovery_result(
                source_frame_id, committed=False, reason="missing_source_image_or_features"
            )
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if source_info is None:
            source_info = {
                "is_test": False,
                "image_name": str(source_payload.get("source_image_name", "")),
                "image_path": str(source_payload.get("image_path", "")),
            }
        source_info["_paper_aligned_insertion_type"] = "true_recovery_commit"
        runtime_gate.mark_pose_attempt(source_frame_id)
        prev_keyframes_src = scene_model.get_prev_keyframes(
            args.num_prev_keyframes_miniba_incr,
            True,
            source_desc,
            resolution_mode="paper_aligned_true_recovery",
        )
        Rt_src = pose_initializer.initialize_incremental(
            prev_keyframes_src, source_desc, n_keyframes, bool(source_info.get("is_test", False)), source_image
        )
        runtime_gate.annotate_pose_debug(
            source_frame_id, getattr(pose_initializer, "last_incremental_debug", {})
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
            runtime_gate.append_true_recovery_commit_event(recovery_trace)
            return
        if args.use_colmap_poses and "Rt" in source_info:
            Rt_src = source_info["Rt"]
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
        scene_model.add_keyframe(source_kf)
        runtime_gate.mark_keyframe_add(source_frame_id)
        recovery_trace["add_keyframe_called"] = True
        recovery_trace["add_keyframe_success"] = True
        recovery_trace["scene_keyframe_appended"] = True
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
        runtime_gate.mark_anchor_update(source_frame_id)
        recovery_trace["anchor_update_called"] = True
        recovery_trace["anchor_update_success"] = True
        n_keyframes += 1
        runtime_gate.mark_final_keyframe_increment(source_frame_id)
        runtime_gate.mark_true_source_recovery_result(source_frame_id, committed=True, reason="")
        recovery_trace["final_keyframe_incremented"] = True
        recovery_trace["final_model_contains_source_frame"] = True
        runtime_gate.append_true_recovery_commit_event(recovery_trace)

    for frameID in pbar:
        start_time = time.time()

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
            prev_desc_kpts = detector(image)  # 提取关键点和描述子
            bootstrap_keyframe_dicts = [{"image": image, "info": info}]
            bootstrap_desc_kpts = [prev_desc_kpts]
            n_keyframes += 1
            continue

        # ========== 特征提取与关键帧判断 ==========
        # 读取下一帧图像并提取特征
        image, info = dataset.getnext()
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
        # 测试帧始终加入，用于姿态估计和评估（但不参与训练）
        should_add_keyframe |= info["is_test"]
        baseline_should_add = should_add_keyframe
        if runtime_gate is not None:
            phase = (
                "bootstrap"
                if n_keyframes < args.num_keyframes_miniba_bootstrap
                else "incremental"
            )
            info["_image_tensor"] = image
            info["_desc_kpts"] = desc_kpts
            info["_inlier_evidence"] = {"num_matches": int(len(curr_prev_matches.kpts))}
            info["_local_context"] = {"phase": phase}
            evidence = {
                "median_displacement": float(dist.median().item()) if len(dist) > 0 else 0.0,
                "displacement_threshold": float(min_displacement),
                "num_matches": int(len(curr_prev_matches.kpts)),
                "min_num_inliers_threshold": int(args.min_num_inliers),
                "is_test": bool(info.get("is_test", False)),
                "recent_pose_fail_rate": (
                    1.0 - (sum(recent_pose_success) / max(len(recent_pose_success), 1))
                    if len(recent_pose_success) > 0
                    else 0.0
                ),
            }
            should_add_keyframe, _ = runtime_gate.decide(
                frameID, info, bool(baseline_should_add), phase=phase, evidence=evidence
            )
            if (
                risk_mode == "paper_aligned_semantic_v1"
                and getattr(args, "paper_aligned_recovery_commit_bridge", "true_source_commit")
                == "true_source_commit"
            ):
                for recovered in runtime_gate.pop_pending_true_source_commits(current_tick_frame_id=frameID):
                    control = runtime_gate.decide_recovered_commit(recovered, current_tick_frame_id=frameID)
                    source_frame_id = int(recovered.get("source_frame_id", -1))
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
                    args.num_prev_keyframes_miniba_incr, True, desc_kpts
                )
                increment_runtime(runtimes["tri"], start_time)
                
                start_time = time.time()
                # 【姿态估计模块】增量姿态初始化：使用PnP-RANSAC和Mini-BA估计新帧位姿
                if runtime_gate is not None:
                    runtime_gate.mark_pose_attempt(frameID)
                Rt = pose_initializer.initialize_incremental(
                    prev_keyframes, desc_kpts, n_keyframes, info["is_test"], image
                )
                if runtime_gate is not None:
                    runtime_gate.annotate_pose_debug(
                        frameID, getattr(pose_initializer, "last_incremental_debug", {})
                    )
                    runtime_gate.mark_pose_result(frameID, Rt is not None)
                recent_pose_success.append(1 if Rt is not None else 0)
                increment_runtime(runtimes["BAI"], start_time)
                
                start_time = time.time()
                if Rt is not None:  # 姿态估计成功
                    # 如果使用COLMAP位姿，则覆盖估计的位姿
                    if args.use_colmap_poses:
                        Rt = info["Rt"]
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
                    prev_keyframe = keyframe
                    increment_runtime(runtimes["Add"], start_time)
                    
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
                        pose_debug = getattr(pose_initializer, "last_incremental_debug", {})
                        fail_reason = str(pose_debug.get("failure_reason", "") or "pose_init_failed")
                        runtime_gate.mark_drop_reason(frameID, fail_reason)

        if should_add_keyframe:
            # ========== 锚点管理：处理大尺度场景 ==========
            # 【场景表示模块】检查是否需要创建新锚点
            # 当高斯点在屏幕上变得过小时，创建新锚点并合并细粒度高斯点
            start_time = time.time()
            scene_model.place_anchor_if_needed()
            if runtime_gate is not None:
                runtime_gate.mark_anchor_update(frameID)
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