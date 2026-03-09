# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 参数配置文件
# 参考：https://github.com/graphdeco-inria/gaussian-splatting/blob/main/args.py


import argparse
import ast
import os

def _parse_simple_config(config_path: str):
    """
    Parse a tiny YAML-like config file with `key: value`.
    Supports bool/int/float/str and comma-separated lists.
    """
    config = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                continue
            lowered = value.lower()
            if lowered in {"true", "false"}:
                config[key] = lowered == "true"
                continue
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = ast.literal_eval(value)
                except Exception:
                    parsed = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                config[key] = parsed
                continue
            try:
                config[key] = ast.literal_eval(value)
            except Exception:
                config[key] = value
    return config

def get_args():
    # 构建命令行解析器，集中管理训练/数据相关参数
    parser = argparse.ArgumentParser(description="Options for data loading and training")
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to a minimal YAML-like config file. Values override argparse defaults.",
    )

    ## Data and Images
    # 数据与图像路径配置
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help="Path to the data folder (should have sparse/0/ if using COLMAP or evaluating poses)")
    parser.add_argument('-i', '--images_dir', type=str, default="images",
                        help="source_path/images_dir is the path to the images (with extensions jpg, png or jpeg).")
    parser.add_argument('--masks_dir', type=str, default="", 
                        help="If set, source_path/masks_dir is the path to optional masks to apply to the images before computing loss (png).")
    parser.add_argument('--num_loader_threads', type=int, default=4,
                        help="Number of workers to load and prepare input images")
    parser.add_argument('--downsampling', type=float, default=-1.0, help="Downsampling ratio for input images")
    parser.add_argument('--pyr_levels', type=int, default=2,
                        help="Number of pyramid levels. Each level l will downsample the image 2^l times in width and height")
    parser.add_argument('--min_displacement', type=float, default=0.03,
                        help="Minimum median keypoint displacement for a new keyframe to be added. Relative to the image width")
    parser.add_argument('--start_at', type=int, default=0,
                        help="Number of frames to skip from the dataset.")
    
    # 球谐阶数（颜色表达）
    parser.add_argument('--sh_degree', default=3)

    ## COLMAP options
    # COLMAP 相关配置：用于对齐或使用外部位姿
    parser.add_argument('--eval_poses', action='store_true',
                        help="Compare poses to COLMAP")
    parser.add_argument('--use_colmap_poses', action='store_true',
                        help="Load COLMAP data for pose and intrinsics initialization")

    ## Learning Rates
    # 各模块学习率
    parser.add_argument('--lr_poses', type=float, default=1e-4, help="Pose learning rate")
    parser.add_argument('--lr_exposure', type=float, default=5e-4, 
                        help="Exposure compensation learning rate")
    parser.add_argument('--lr_depth_scale_offset', type=float, default=1e-4)
    parser.add_argument('--position_lr_init', type=float, default=0.00005, 
                        help="Initial position learning rate")
    parser.add_argument('--position_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for position learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.005, help="Feature learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="Opacity learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.01, help="Scaling learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.002, help="Rotation learning rate")

    ## Training schedule and losses
    # 训练迭代和损失权重
    parser.add_argument('--lambda_dssim', type=float, default=0.2, help="Weight for DSSIM loss")
    parser.add_argument('--num_iterations', type=int, default=30, 
                        help="Number of training iterations per keyframe")
    parser.add_argument('--depth_loss_weight_init', type=float, default=1e-2)
    parser.add_argument('--depth_loss_weight_decay', type=float, default=0.9, 
                        help="Weight decay for depth loss, multiply depth loss weight by this factor every iterations")
    parser.add_argument('--save_at_finetune_epoch', type=int, nargs='+', default=[], 
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified epochs when fine-tuning.")
    parser.add_argument('--use_last_frame_proba', type=float, default=0.2, 
                        help="Probability of using the last registered frame for each training iteration")

    ## Pose initialization options
    # Matching
    # 姿态初始化：特征匹配与 RANSAC 相关超参
    parser.add_argument('--num_kpts', type=int, default=int(4096*1.5),
                        help="Number of keypoints to extract from each image")
    parser.add_argument('--match_max_error', type=float, default=2e-3,
                        help="Maximum reprojection error for matching keypoints, proportion of the image width. This is used to filter outliers and discard points at triangulation.")
    parser.add_argument('--fundmat_samples', type=int, default=2000,
                        help="Maximum number of set of matches used to estimate the fundamental matrix for outlier removal")
    parser.add_argument('--min_num_inliers', type=int, default=100,
                        help="The keyframe will be added only if the number of inliers is greater than this value")
    # Initial mini bundle adjustment
    # 初始小规模 BA 超参
    parser.add_argument('--num_keyframes_miniba_bootstrap', type=int, default=8,
                        help="Number of first keyframes accumulated for pose and focal estimation before optimization")
    parser.add_argument('--num_pts_miniba_bootstrap', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_bootstrap', type=int, default=200)
    parser.add_argument('--enable_reboot', action='store_true')
    # Focal estimation
    parser.add_argument('--fix_focal', action='store_true', 
                        help="If set, will use init_focal or init_fov without reoptimizing focal")
    parser.add_argument('--init_focal', type=float, default=-1.0, 
                        help="Initial focal length in pixels. If not set, will use init_fov or be set as 0.7*width of the image if init_fov is also not set")
    parser.add_argument('--init_fov', type=float, default=-1.0, 
                        help="Initial horizontal FoV in degrees. Used only if init_focal is not set")
    # Incremental pose optimization
    # 增量 BA 超参
    parser.add_argument('--num_prev_keyframes_miniba_incr', type=int, default=6,
                        help="Number of previous keyframes for incremental pose initialization")
    parser.add_argument('--num_prev_keyframes_check', type=int, default=20,
                        help="Number of previous keyframes to check for matches with new keyframe")
    parser.add_argument('--pnpransac_samples', type=int, default=2000,
                        help="Maximum number of set of 2D-3D matches used to estimate the initial pose and outlier removal")
    parser.add_argument('--num_pts_miniba_incr', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_incr', type=int, default=20)
    parser.add_argument('--pose_refine_iters', type=int, default=25,
                        help="Number of pose refinement iterations after PnP initialization")
    parser.add_argument('--pose_refine_lr', type=float, default=2e-3,
                        help="Learning rate for pose refinement")
    parser.add_argument('--use_pose_reprojection_loss', action='store_true',
                        help="Enable reprojection loss branch for pose refinement")
    parser.add_argument('--pose_reprojection_weight', type=float, default=1.0,
                        help="Weight for pose reprojection loss")
    parser.add_argument('--use_pose_photometric_refine', action='store_true',
                        help="Enable photometric pose refinement branch with current renderer")
    parser.add_argument('--pose_photometric_weight', type=float, default=0.2,
                        help="Weight for pose photometric refinement")
    parser.add_argument('--pose_refine_downsample', type=int, default=4,
                        help="Downsample factor for photometric pose refinement")
    parser.add_argument('--use_correspondence_guided_pose_init', action='store_true',
                        help="Use rendered-depth-guided correspondences for 2D-3D pose initialization")
    parser.add_argument('--min_pnp_inliers', type=int, default=24,
                        help="Minimum inliers required for PnP pose initialization")
    parser.add_argument('--enable_pnp_fallback_global_refine', action='store_true',
                        help="Trigger global refine and retry when PnP fails")
    parser.add_argument('--fallback_global_refine_iters', type=int, default=60,
                        help="Iterations for fallback global pose+gaussian refine")

    ## Gaussian initialization options
    # 高斯初始化概率相关参数
    parser.add_argument('--init_proba_scaler', type=float, default=2,
                        help="Scale the laplacian-based probability of using a pixel to make a new Gaussian primitive. Set to 0 to only use triangulated points.")

    # Anchor management
    # 锚点融合相关
    parser.add_argument('--anchor_overlap', type=float, default=0.3,
                        help="Size of the overlapping regions when blending between anchors")
    parser.add_argument('--use_adaptive_octree_anchor', action='store_true',
                        help="Enable adaptive octree-like anchor/gaussian insertion")
    parser.add_argument('--octree_max_level', type=int, default=3,
                        help="Maximum level for adaptive octree insertion")
    parser.add_argument('--octree_split_threshold', type=int, default=12,
                        help="Density threshold used to keep finer octree cells")
    parser.add_argument('--octree_prune_threshold', type=int, default=2,
                        help="Minimum points per octree cell to keep")
    parser.add_argument('--anchor_overlap_prune', action='store_true',
                        help="Prune overlapping insertions in adaptive octree mode")
    parser.add_argument('--enable_new_region_unprojection', action='store_true',
                        help="Insert gaussians from newly visible regions via unprojection")
    parser.add_argument('--new_region_sample_cap', type=int, default=5000,
                        help="Maximum number of newly visible pixels to unproject per keyframe")
    parser.add_argument('--new_gaussian_conf_thresh', type=float, default=0.5,
                        help="Minimum confidence for new gaussian initialization")
    parser.add_argument('--aggressive_prune_opacity', type=float, default=0.02,
                        help="Opacity threshold for lightweight aggressive pruning")

    ## Keyframe management
    # 关键帧管理
    parser.add_argument('--max_active_keyframes', type=int, default=200,
                        help="Maximum number of keyframes to keep in GPU memory. Will start offloading keyframes to CPU if this number is exceeded.")
    parser.add_argument('--optimizer_mode', choices=['baseline', 'merged'], default='baseline',
                        help="Optimization schedule mode")
    parser.add_argument('--use_visibility_window', action='store_true',
                        help="Use visibility-adapted local optimization window")
    parser.add_argument('--global_refine_every_n_frames', type=int, default=0,
                        help="Run periodic global refinement every N keyframes (0 to disable)")
    parser.add_argument('--local_window_min', type=int, default=3,
                        help="Minimum size for visibility local window")
    parser.add_argument('--local_window_max', type=int, default=8,
                        help="Maximum size for visibility local window")
    parser.add_argument('--visibility_iou_threshold', type=float, default=0.2,
                        help="Visibility overlap threshold for local window selection")

    ## Evaluation
    # 测试频率与展示
    parser.add_argument('--test_hold', type=int, default=-1, 
                        help="Holdout for test set, will exclude every test_hold image from the Gaussian optimization and use them for testing. The test frames will still be used for training the pose. If set to -1, no keyframes will be excluded from training.")
    parser.add_argument('--test_frequency', type=int, default=-1, 
                        help="Test and get metrics every test_frequency keyframes")
    parser.add_argument('--display_runtimes', action='store_true', 
                        help="Display runtimes for each step in the tqdm bar")

    ## Checkpoint options
    # 输出与 checkpoint 相关
    parser.add_argument('-m', '--model_path', default="", 
                        help="Directory to store the renders from test view and checkpoints after training. If not set, will be set to results/xxxxxx.")
    parser.add_argument('--save_every', default=-1, type=int, 
                        help="Frequency of exporting renders w.r.t input frames.")

    ## Viewer
    # 前端可视化配置
    parser.add_argument('--viewer_mode', choices=['local', 'server', 'web', 'none'], default='none')
    parser.add_argument('--ip', type=str, default="0.0.0.0", 
                        help="IP address of the viewer client, if using server viewer_mode")
    parser.add_argument('--port', type=int, default=6009,
                        help="Port of the viewer client, if using server viewer_mode")

    args = parser.parse_args()

    if args.config_path:
        config_values = _parse_simple_config(args.config_path)
        for key, value in config_values.items():
            if hasattr(args, key):
                setattr(args, key, value)

    ## Set the output directory if not specified
    # 若未指定输出目录，则在 results 下自动递增创建
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args