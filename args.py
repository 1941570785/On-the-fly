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
import os

def get_args():
    # 构建命令行解析器，集中管理训练/数据相关参数
    parser = argparse.ArgumentParser(description="Options for data loading and training")

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

    ## Gaussian initialization options
    # 高斯初始化概率相关参数
    parser.add_argument('--init_proba_scaler', type=float, default=2,
                        help="Scale the laplacian-based probability of using a pixel to make a new Gaussian primitive. Set to 0 to only use triangulated points.")

    # Anchor management
    # 锚点融合相关
    parser.add_argument('--anchor_overlap', type=float, default=0.3,
                        help="Size of the overlapping regions when blending between anchors")

    ## Keyframe management
    # 关键帧管理
    parser.add_argument('--max_active_keyframes', type=int, default=200,
                        help="Maximum number of keyframes to keep in GPU memory. Will start offloading keyframes to CPU if this number is exceeded.")

    ## Evaluation
    # 测试频率与展示
    parser.add_argument('--test_hold', type=int, default=-1, 
                        help="Holdout for test set, will exclude every test_hold image from the Gaussian optimization and use them for testing. The test frames will still be used for training the pose. If set to -1, no keyframes will be excluded from training.")
    parser.add_argument('--test_frequency', type=int, default=-1, 
                        help="Test and get metrics every test_frequency keyframes")
    parser.add_argument('--display_runtimes', action='store_true', 
                        help="Display runtimes for each step in the tqdm bar")

    ## Paper-aligned risk admission (default off keeps baseline unchanged)
    parser.add_argument(
        '--risk_admission_mode',
        type=str,
        default='off',
        choices=[
            'off',
            'paper_aligned_baseline_passthrough',
            'paper_aligned_semantic_v1',
            'on_the_fly_innovation_v1',
        ],
        help='Risk admission mode. off keeps baseline runtime unchanged.',
    )
    parser.add_argument(
        '--paper_aligned_tau_R_low',
        type=float,
        default=None,
        help='Override coupled/semantic R low threshold.',
    )
    parser.add_argument(
        '--paper_aligned_tau_R_high',
        type=float,
        default=None,
        help='Override coupled/semantic R high threshold.',
    )
    parser.add_argument(
        '--paper_aligned_tau_V',
        type=float,
        default=None,
        help='Override coupled/semantic direct-admit V threshold.',
    )
    parser.add_argument(
        '--paper_aligned_tau_V_min',
        type=float,
        default=None,
        help='Override coupled/semantic defer minimum V threshold.',
    )
    parser.add_argument(
        '--paper_aligned_tau_B',
        type=float,
        default=None,
        help='Override coupled/semantic recoverable risk-band threshold.',
    )
    parser.add_argument(
        '--paper_aligned_tau_Q',
        type=float,
        default=None,
        help='Override coupled/semantic defer Q threshold.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_delay_frames',
        type=int,
        default=None,
        help='Override semantic recovery pool delay in frames.',
    )
    parser.add_argument(
        '--paper_aligned_semantic_recovery_max_attempts',
        type=int,
        default=None,
        help='Override semantic recovery maximum attempts per deferred source.',
    )
    parser.add_argument(
        '--paper_aligned_semantic_recovery_attempts_per_tick',
        type=int,
        default=None,
        help='Override semantic recovery attempts scheduled per tick.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_commit_bridge',
        type=str,
        default='true_source_commit',
        choices=['off', 'semantic_surrogate', 'true_source_commit'],
        help='Recovery commit bridge mode for semantic runtime.',
    )
    parser.add_argument(
        '--paper_aligned_defer_recovery_support_bridge',
        type=str,
        default=None,
        choices=['off', 'v1'],
        help='Defer-recoverable recovery support bridge (anchor transition + reference propagation).',
    )
    parser.add_argument(
        '--paper_aligned_bridge_max_refs',
        type=int,
        default=4,
        help='Max cross-anchor bridge references injected per recovery pose attempt.',
    )
    parser.add_argument(
        '--paper_aligned_bridge_top_k_per_anchor',
        type=int,
        default=6,
        help='Top-K materialized keyframes retained per anchor for bridge inventory.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_commit_control',
        type=str,
        default=None,
        choices=[
            'off',
            'conservative',
            'adaptive',
            'conservative_gap_aware',
            'conservative_gap_aware_v2',
            'recovery_commit_strict_v3',
            'recovery_commit_balanced_v4',
            'recovery_commit_rescue_v5',
            'recovery_commit_materialization_aware_v6',
            'recovery_commit_early_seed_v7',
        ],
        help='Post-success recovery commit control mode. off keeps prior behavior.',
    )
    parser.add_argument(
        '--paper_aligned_direct_density_control',
        type=str,
        default=None,
        choices=[
            'off',
            'conservative',
            'target_band_v1',
            'target_band_v2',
            'target_band_v2_1',
            'target_band_v2_2',
            'target_band_v2_2_1',
            'target_band_v2_2_2',
            'target_band_v2_2_2_1',
            'pose_rep_decouple_v1',
            'pose_rep_value_decouple_v2',
            'pose_rep_value_decouple_v3',
            'pose_rep_active_memory_v1',
            'pose_rep_active_memory_v2',
            'pose_rep_active_memory_v4',
            'pose_rep_active_memory_v5',
            'pose_rep_active_memory_v6',
            'pose_rep_active_memory_v8',
            'pose_rep_active_memory_v9',
            'pose_rep_active_memory_v24',
        ],
        help='Direct keyframe finalization density guard (not R/V/Q admission).',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_2_1_post500_gap_rescue_budget_per_100',
        type=int,
        default=4,
        help='v2.2.2.1 post-500 pre-gap rescue budget per 100 frames.',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_2_preemptive_gap_threshold',
        type=int,
        default=18,
        help='v2.2.2 preemptive hard gap rescue when gap_after_if_hold >= this.',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_1_soft_gap_threshold',
        type=int,
        default=6,
        help='v2.2.1 soft gap tail threshold for budgeted gap rescue.',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_1_hard_gap_threshold',
        type=int,
        default=20,
        help='v2.2.1 hard gap threshold (must finalize).',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_1_gap_rescue_budget_per_100',
        type=int,
        default=4,
        help='v2.2.1 max gap-tail rescues per 100 frames.',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_1_gap_rescue_density_upper_500',
        type=float,
        default=50.0,
        help='v2.2.1 gap rescue density cap for frames <=500.',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_1_gap_rescue_density_upper_later',
        type=float,
        default=45.0,
        help='v2.2.1 gap rescue density cap for frames >500.',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_early_rescue_budget_per_100',
        type=int,
        default=8,
        help='v2.2 early lower-bound rescue budget per 100 frames.',
    )
    parser.add_argument(
        '--paper_aligned_direct_v2_2_early_rescue_density_stop',
        type=float,
        default=32.0,
        help='v2.2 stop aggressive early rescue when density reaches this per-100.',
    )
    parser.add_argument(
        '--paper_aligned_direct_local_window_size',
        type=int,
        default=100,
        help='v2.2 sliding frame window for local density/gap guards.',
    )
    parser.add_argument(
        '--paper_aligned_direct_local_density_lower_per_100',
        type=float,
        default=20.0,
        help='v2.2 local window density lower trigger per 100 frames.',
    )
    parser.add_argument(
        '--paper_aligned_direct_density_lower_per_100',
        type=float,
        default=25.0,
        help='Lower density band for direct finalization (keyframes per 100 frames).',
    )
    parser.add_argument(
        '--paper_aligned_direct_density_target_per_100',
        type=float,
        default=35.0,
        help='Target density band for direct finalization.',
    )
    parser.add_argument(
        '--paper_aligned_direct_density_upper_per_100',
        type=float,
        default=45.0,
        help='Upper density band; above this only gap/novelty/support finalize.',
    )
    parser.add_argument(
        '--paper_aligned_direct_density_hard_upper_per_100',
        type=float,
        default=50.0,
        help='Hard upper density for direct finalization holds.',
    )
    parser.add_argument(
        '--paper_aligned_direct_gap_hard_limit',
        type=int,
        default=20,
        help='Force direct finalize if hold would exceed this main-chain gap.',
    )
    parser.add_argument(
        '--paper_aligned_direct_redundant_source_gap',
        type=int,
        default=3,
        help='Hold redundant direct finalize when source gap to last keyframe is at most this.',
    )
    parser.add_argument(
        '--paper_aligned_direct_high_novelty_budget_per_100',
        type=int,
        default=8,
        help='Max finalize_high_novelty per 100 frames (v2).',
    )
    parser.add_argument(
        '--paper_aligned_direct_support_needed_budget_per_100',
        type=int,
        default=6,
        help='Max finalize_support_needed per 100 frames (v2).',
    )
    parser.add_argument(
        '--paper_aligned_direct_density_hysteresis_margin',
        type=float,
        default=3.0,
        help='Density band hysteresis margin (v2).',
    )
    parser.add_argument(
        '--paper_aligned_direct_min_growth_per_100',
        type=float,
        default=20.0,
        help='Minimum recent keyframe growth per 100 frames (v2 anti-starvation).',
    )
    parser.add_argument(
        '--paper_aligned_direct_baseline_relative_lower_ratio',
        type=float,
        default=0.8,
        help='Min keyframe count ratio vs baseline-at-same-length (v2).',
    )
    parser.add_argument(
        '--paper_aligned_direct_baseline_density_per_100',
        type=float,
        default=27.0,
        help='Reference baseline density per 100 frames for v2 lower guard.',
    )
    parser.add_argument(
        '--paper_aligned_direct_update_prev_desc_on_hold',
        type=str,
        default=None,
        choices=['off', 'on', 'light'],
        help='Whether to update prev_desc_kpts when direct finalize is held.',
    )
    parser.add_argument(
        '--paper_aligned_direct_hold_tracking_bridge_mode',
        type=str,
        default='light',
        choices=['none', 'light', 'full'],
        help='Tracking bridge mode when hold uses light prev_desc update.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_window_size',
        type=int,
        default=30,
        help='Sliding window size for recovery commit rate control.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_commit_max_per_window',
        type=int,
        default=5,
        help='Maximum recovery commits per sliding window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_candidate_max_age',
        type=int,
        default=180,
        help='Maximum source age allowed for recovery commit candidate.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_hold_max_retries',
        type=int,
        default=4,
        help='Maximum hold retries before rejecting a recovery candidate.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_hold_retry_interval',
        type=int,
        default=6,
        help='Retry interval (ticks) for held recovery candidates.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_density_upper_per_100',
        type=float,
        default=46.0,
        help='Density upper bound (keyframes per 100 frames) for recovery commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_gap_override_threshold',
        type=float,
        default=5.0,
        help='Enable gap-aware override when local main-chain gap reaches this threshold.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_gap_override_max_per_interval',
        type=int,
        default=2,
        help='Maximum override commits in each high-risk gap interval.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_gap_override_density_upper_per_100',
        type=float,
        default=55.0,
        help='Density upper bound for allowing gap-aware overrides.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_gap_override_min_v',
        type=float,
        default=0.25,
        help='Minimum V_t for a gap-aware override candidate.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_gap_override_min_q',
        type=float,
        default=0.10,
        help='Minimum Q_t for a gap-aware override candidate.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v2_source_gap_trigger',
        type=int,
        default=20,
        help='Enable v2 override when source gap to last committed reaches this threshold.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v2_predicted_gap_trigger',
        type=int,
        default=20,
        help='Enable v2 override when predicted gap if hold exceeds this threshold.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v2_episode_override_budget',
        type=int,
        default=2,
        help='Maximum v2 overrides allowed in one long-gap episode.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_density_upper_per_100',
        type=float,
        default=42.0,
        help='Strict v3 density upper bound (keyframes per 100 frames).',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_gap_critical_trigger',
        type=int,
        default=18,
        help='Strict v3 source gap trigger for gap-critical recovery commit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_gap_hard_limit',
        type=int,
        default=20,
        help='Strict v3 hard predicted gap limit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_window_size',
        type=int,
        default=50,
        help='Strict v3 commit accounting window size.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_max_normal_per_window',
        type=int,
        default=1,
        help='Strict v3 max support-ranked normal commits per window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_max_gap_critical_per_window',
        type=int,
        default=2,
        help='Strict v3 max gap-critical commits per window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_min_source_gap',
        type=int,
        default=5,
        help='Strict v3 minimum source-frame gap to avoid local over-stacking.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_support_topk',
        type=int,
        default=1,
        help='Strict v3 support top-k kept for sparse normal commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_anchor_guard_enabled',
        type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y', 't'},
        default=True,
        help='Enable strict v3 anchor-aware guard.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_anchor_soft_limit',
        type=int,
        default=5,
        help='Strict v3 anchor soft limit for normal commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_min_v',
        type=float,
        default=0.25,
        help='Strict v3 minimum V_t for eligible recovery commit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_min_q',
        type=float,
        default=0.10,
        help='Strict v3 minimum Q_t for eligible recovery commit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v3_max_r',
        type=float,
        default=0.75,
        help='Strict v3 maximum R_t for eligible recovery commit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_density_lower_per_100',
        type=float,
        default=28.0,
        help='Balanced v4 density lower bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_density_target_per_100',
        type=float,
        default=35.0,
        help='Balanced v4 density target center.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_density_upper_per_100',
        type=float,
        default=42.0,
        help='Balanced v4 density soft upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_density_hard_upper_per_100',
        type=float,
        default=50.0,
        help='Balanced v4 density hard upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_gap_trigger',
        type=int,
        default=18,
        help='Balanced v4 source gap trigger for gap rescue.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_gap_hard_limit',
        type=int,
        default=20,
        help='Balanced v4 hard gap limit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_gap_rescue_budget_per_episode',
        type=int,
        default=4,
        help='Balanced v4 gap rescue budget per long-gap episode.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_window_size',
        type=int,
        default=50,
        help='Balanced v4 decision window size.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_normal_topk_below_lower',
        type=int,
        default=2,
        help='Balanced v4 support top-k when density below lower bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_normal_topk_in_band',
        type=int,
        default=1,
        help='Balanced v4 support top-k in target band.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_normal_topk_above_upper',
        type=int,
        default=0,
        help='Balanced v4 support top-k when density above upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_min_source_gap',
        type=int,
        default=4,
        help='Balanced v4 minimum source gap for non-gap-rescue commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_min_v',
        type=float,
        default=0.25,
        help='Balanced v4 minimum V_t.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_min_q',
        type=float,
        default=0.10,
        help='Balanced v4 minimum Q_t.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_max_r',
        type=float,
        default=0.75,
        help='Balanced v4 maximum R_t.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_anchor_target_min',
        type=int,
        default=4,
        help='Balanced v4 anchor target lower bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_anchor_target_max',
        type=int,
        default=5,
        help='Balanced v4 anchor target upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_anchor_soft_upper',
        type=int,
        default=6,
        help='Balanced v4 anchor soft upper guard.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_retry_extension_for_gap',
        type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y', 't'},
        default=True,
        help='Allow retry extension for gap-critical candidates in v4.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v4_retry_extension_for_coverage',
        type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y', 't'},
        default=True,
        help='Allow retry extension for coverage-floor candidates in v4.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_density_lower_per_100',
        type=float,
        default=28.0,
        help='Rescue v5 density lower bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_density_target_per_100',
        type=float,
        default=35.0,
        help='Rescue v5 density target center.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_density_upper_per_100',
        type=float,
        default=42.0,
        help='Rescue v5 density soft upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_density_hard_upper_per_100',
        type=float,
        default=50.0,
        help='Rescue v5 density hard upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_gap_trigger',
        type=int,
        default=18,
        help='Rescue v5 source gap trigger.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_gap_hard_limit',
        type=int,
        default=20,
        help='Rescue v5 hard gap limit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_gap_rescue_base_budget',
        type=int,
        default=4,
        help='Rescue v5 base gap rescue budget per episode.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_window_size',
        type=int,
        default=50,
        help='Rescue v5 ranking window size.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_normal_topk_below_lower',
        type=int,
        default=2,
        help='Rescue v5 normal top-k below lower density.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_normal_topk_in_band',
        type=int,
        default=1,
        help='Rescue v5 normal top-k in target band.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_normal_topk_above_upper',
        type=int,
        default=0,
        help='Rescue v5 normal top-k above upper density.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_coverage_topk',
        type=int,
        default=3,
        help='Rescue v5 relaxed top-k for coverage rescue.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_min_source_gap',
        type=int,
        default=4,
        help='Rescue v5 minimum source gap for normal commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_min_v',
        type=float,
        default=0.25,
        help='Rescue v5 minimum V_t.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_min_q',
        type=float,
        default=0.10,
        help='Rescue v5 minimum Q_t.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_max_r',
        type=float,
        default=0.75,
        help='Rescue v5 maximum R_t.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_anchor_target_min',
        type=int,
        default=4,
        help='Rescue v5 anchor target lower bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_anchor_target_max',
        type=int,
        default=5,
        help='Rescue v5 anchor target upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_anchor_soft_upper',
        type=int,
        default=6,
        help='Rescue v5 anchor soft upper bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_growth_window_short',
        type=int,
        default=100,
        help='Rescue v5 short growth window size.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_growth_window_long',
        type=int,
        default=200,
        help='Rescue v5 long growth window size.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_growth_plateau_min_short',
        type=int,
        default=10,
        help='Rescue v5 minimum keyframe growth in short window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_growth_plateau_min_long',
        type=int,
        default=20,
        help='Rescue v5 minimum keyframe growth in long window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_retry_extension_for_gap',
        type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y', 't'},
        default=True,
        help='Allow retry extension for gap rescue in v5.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v5_retry_extension_for_coverage',
        type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y', 't'},
        default=True,
        help='Allow retry extension for coverage rescue in v5.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_min_feasibility_normal',
        type=float,
        default=0.58,
        help='Materialization-aware v6 minimum feasibility for normal sparse commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_min_feasibility_rescue',
        type=float,
        default=0.52,
        help='Materialization-aware v6 minimum feasibility for coverage/gap rescue commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_min_matches_rescue',
        type=int,
        default=450,
        help='Materialization-aware v6 minimum match support for rescue commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_pose_fail_cooldown_keyframes',
        type=int,
        default=3,
        help='Materialization-aware v6 keyframes needed before retrying pose-failed source.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_attempt_budget_per_window',
        type=int,
        default=8,
        help='Materialization-aware v6 maximum non-materialized attempts per window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_materialized_budget_per_window',
        type=int,
        default=4,
        help='Materialization-aware v6 materialized-success budget reference per window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_early_coverage_margin',
        type=float,
        default=4.0,
        help='Trigger early coverage rescue when density is within this margin above lower bound.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_pose_fail_rate_trigger',
        type=float,
        default=0.50,
        help='Trigger early coverage rescue when recent pose-fail rate exceeds this value.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v6_materialization_rate_trigger',
        type=float,
        default=0.35,
        help='Trigger early coverage rescue when recent materialization rate drops below this value.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v7_early_seed_start',
        type=int,
        default=150,
        help='v7 early seed source-frame window start.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v7_early_seed_end',
        type=int,
        default=300,
        help='v7 early seed source-frame window end.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v7_seed_budget_total_short500',
        type=int,
        default=30,
        help='v7 total early seed budget for short500 audits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v7_min_seed_feasibility',
        type=float,
        default=0.38,
        help='v7 minimum materialization feasibility for early seed commit.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v7_min_seed_matches',
        type=int,
        default=300,
        help='v7 minimum match support for early seed commit.',
    )
    parser.add_argument(
        '--paper_aligned_contract_trace_path',
        type=str,
        default='',
        help='Optional output JSON path for runtime contract trace.',
    )
    parser.add_argument(
        '--paper_aligned_lifecycle_csv',
        type=str,
        default='',
        help='Optional lifecycle CSV path for paper-aligned semantic mode.',
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=-1,
        help='Debug-only frame limit for short runs. -1 means no limit.',
    )

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

    ## Set the output directory if not specified
    # 若未指定输出目录，则在 results 下自动递增创建
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args
