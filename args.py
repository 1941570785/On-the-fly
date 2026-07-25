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
    parser.add_argument(
        '--experiment_seed',
        type=int,
        default=0,
        help="Random seed used by paired reconstruction experiments.",
    )
    parser.add_argument(
        '--experiment_deterministic',
        action='store_true',
        help="Use deterministic library settings for controlled experiments.",
    )
    
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
        '--pose_initialization_risk_mode',
        type=str,
        default='off',
        choices=['off', 'observe_v1', 'isolate_v1', 'verify_v1', 'verify_v2'],
        help='Post-pose A-module mode. verify_v2 independently validates risky training and pose-only test frames.',
    )
    parser.add_argument(
        '--pose_initialization_risk_absolute_threshold',
        type=float,
        default=0.10,
        help='Absolute lower bound for the post-pose isolation threshold.',
    )
    parser.add_argument(
        '--pose_initialization_risk_adaptive_sigma',
        type=float,
        default=2.0,
        help='Robust-sigma multiplier used by the online adaptive risk threshold.',
    )
    parser.add_argument(
        '--pose_initialization_risk_warmup',
        type=int,
        default=8,
        help='Number of selected training frames observed before isolation can activate.',
    )
    parser.add_argument(
        '--pose_initialization_risk_history_size',
        type=int,
        default=64,
        help='Maximum number of recent selected-frame risk scores used for online calibration.',
    )
    parser.add_argument(
        '--pose_initialization_risk_cooldown_frames',
        type=int,
        default=12,
        help='Minimum source-frame interval between two representation-isolation decisions.',
    )
    parser.add_argument(
        '--pose_verification_mad_scale',
        type=float,
        default=2.5,
        help='MAD multiplier used to clean pose-verification correspondences.',
    )
    parser.add_argument(
        '--pose_verification_min_support',
        type=int,
        default=24,
        help='Target minimum 2D-3D support retained for a verification MiniBA.',
    )
    parser.add_argument(
        '--pose_verification_min_improvement',
        type=float,
        default=0.02,
        help='Minimum relative median reprojection improvement required to accept a verified pose.',
    )
    parser.add_argument(
        '--pose_verification_v2_min_improvement',
        type=float,
        default=0.0,
        help='Held-out median reprojection improvement required by verify_v2.',
    )
    parser.add_argument(
        '--pose_verification_candidate_mode',
        type=str,
        default='single_v2',
        choices=[
            'single_v2',
            'balanced_step_v21',
            'balanced_epipolar_v22',
            'multihypothesis_v23',
            'multiview_relative_v24',
        ],
        help='Internal verification candidate strategy; single_v2 preserves the existing solver.',
    )
    parser.add_argument(
        '--pose_verification_registration_sampling_mode',
        type=str,
        default='off',
        choices=['off', 'frame_deterministic_v1'],
        help='Optionally isolate pose registration sampling from the shared training RNG.',
    )
    parser.add_argument(
        '--pose_direct_retry_mode',
        type=str,
        default='off',
        choices=['off', 'pose_safe_v18'],
        help='Optional locked pose-safe v18 retry and multi-hypothesis policy.',
    )
    parser.add_argument(
        '--pose_verification_registration_solver_mode',
        type=str,
        default='baseline_cuda_v1',
        choices=['baseline_cuda_v1', 'deterministic_opencv_v2'],
        help='PnP implementation used by the A registration stage.',
    )
    parser.add_argument(
        '--pose_verification_async_pose_protection_mode',
        type=str,
        default='off',
        choices=['off', 'fixed_joint_budget_v1'],
        help='Optionally protect verified poses from schedule-dependent asynchronous updates.',
    )
    parser.add_argument(
        '--pose_verification_max_p90_ratio',
        type=float,
        default=1.01,
        help='Maximum post/pre p90 reprojection-error ratio for safe acceptance.',
    )
    parser.add_argument(
        '--pose_verification_max_temporal_score_ratio',
        type=float,
        default=float('inf'),
        help='Maximum candidate/initial causal motion-consistency score ratio.',
    )
    parser.add_argument(
        '--pose_verification_geometry_anchor_mode',
        type=str,
        default='off',
        choices=['off', 'freeze_v1'],
        help='Use the verified pose as the persistent geometry reference.',
    )
    parser.add_argument(
        '--pose_verification_reference_geometry_mode',
        type=str,
        default='off',
        choices=[
            'off',
            'frozen_first_valid_v1',
            'frozen_verification_only_v2',
            'guarded_frozen_v3',
            'guarded_frozen_live_pose_v4',
            'guarded_frozen_homogeneous_v5',
            'frozen_global_support_guard_v6',
        ],
        help='Use an immutable first-valid sparse geometry snapshot for pose verification.',
    )
    parser.add_argument(
        '--pose_verification_frozen_min_match_support',
        type=int,
        default=24,
        help='Minimum current-frame matches required before using frozen reference geometry.',
    )
    parser.add_argument(
        '--pose_verification_frozen_min_live_ratio',
        type=float,
        default=0.50,
        help='Minimum frozen/live matched-support ratio for guarded reference geometry.',
    )
    parser.add_argument(
        '--pose_verification_frozen_min_reference_count',
        type=int,
        default=2,
        help='Minimum guarded frozen references required for a homogeneous frame subset.',
    )
    parser.add_argument(
        '--pose_verification_frozen_min_total_support',
        type=int,
        default=48,
        help='Minimum total matched support required for a homogeneous frozen frame subset.',
    )
    parser.add_argument(
        '--pose_verification_anchor_reference_mode',
        type=str,
        default='off',
        choices=['off', 'stable_anchor_v1'],
        help='Expand only risk-triggered A verification with causal stable anchor references.',
    )
    parser.add_argument(
        '--pose_verification_anchor_pool_size',
        type=int,
        default=48,
        help='Maximum uniformly sampled stable-anchor candidates scored per verification.',
    )
    parser.add_argument(
        '--pose_verification_anchor_max_refs',
        type=int,
        default=4,
        help='Maximum additional stable references used by an A verification candidate.',
    )
    parser.add_argument(
        '--pose_verification_anchor_min_age_frames',
        type=int,
        default=48,
        help='Minimum causal source-frame age of an A verification anchor.',
    )
    parser.add_argument(
        '--pose_verification_anchor_min_support_count',
        type=int,
        default=400,
        help='Minimum sparse 3D support required from an A verification anchor.',
    )
    parser.add_argument(
        '--pose_verification_anchor_max_risk_score',
        type=float,
        default=0.08,
        help='Maximum recorded pose-risk score allowed for an A verification anchor.',
    )
    parser.add_argument(
        '--pose_verification_anchor_min_match_score',
        type=float,
        default=180.0,
        help='Minimum descriptor-match support required from an A verification anchor.',
    )
    parser.add_argument(
        '--pose_verification_anchor_min_source_separation',
        type=int,
        default=24,
        help='Minimum temporal separation between selected A verification anchors.',
    )
    parser.add_argument(
        '--pose_verification_anchor_candidate_scope',
        type=str,
        default='combined_v1',
        choices=['combined_v1', 'anchor_only_v1'],
        help='Generate A candidates from stable anchors alone when at least two are available.',
    )
    parser.add_argument(
        '--pose_verification_anchor_pre_error_scale',
        type=float,
        default=4.0,
        help='Pre-solve reprojection tolerance multiplier for stable-anchor evidence only.',
    )
    parser.add_argument(
        '--pose_verification_reference_policy',
        type=str,
        default='off',
        choices=[
            'off',
            'conservative_quarantine_v1',
            'conservative_high_risk_v2',
        ],
        help='Optionally exclude high-risk A poses only from future pose references.',
    )
    parser.add_argument(
        '--pose_verification_reference_risk_threshold',
        type=float,
        default=0.12,
        help='Minimum A risk score for conservative reference quarantine.',
    )
    parser.add_argument(
        '--pose_verification_reference_cooldown_frames',
        type=int,
        default=20,
        help='Minimum source-frame interval between A reference quarantines.',
    )
    parser.add_argument(
        '--pose_verification_min_support_ratio',
        type=float,
        default=0.80,
        help='Minimum post/pre valid correspondence ratio for safe acceptance.',
    )
    parser.add_argument(
        '--pose_delayed_verification_mode',
        type=str,
        default='off',
        choices=[
            'off',
            'final_resection_v1',
            'final_resection_v2_global_observe',
            'final_resection_v2_global',
        ],
        help='Optional end-of-stream A review using mature 3D points and held-out references.',
    )
    parser.add_argument(
        '--pose_delayed_verification_solve_refs',
        type=int,
        default=8,
        help='Number of non-test references used to generate each delayed pose candidate.',
    )
    parser.add_argument(
        '--pose_delayed_verification_validation_refs',
        type=int,
        default=6,
        help='Disjoint non-test references reserved for delayed candidate validation.',
    )
    parser.add_argument(
        '--pose_delayed_verification_min_point_count',
        type=int,
        default=16,
        help='Minimum mature 3D points required for a delayed pose reference.',
    )
    parser.add_argument(
        '--pose_delayed_verification_min_improvement',
        type=float,
        default=0.03,
        help='Minimum held-out median reprojection improvement for delayed acceptance.',
    )
    parser.add_argument(
        '--pose_delayed_verification_max_mean_ratio',
        type=float,
        default=0.99,
        help='Maximum delayed post/pre mean reprojection-error ratio.',
    )
    parser.add_argument(
        '--pose_delayed_verification_max_translation',
        type=float,
        default=0.10,
        help='Maximum accepted delayed translation correction in reconstruction units.',
    )
    parser.add_argument(
        '--pose_delayed_verification_max_rotation_deg',
        type=float,
        default=3.0,
        help='Maximum accepted delayed rotation correction in degrees.',
    )
    parser.add_argument(
        '--pose_delayed_verification_global_pool',
        type=int,
        default=32,
        help='Maximum sampled nonlocal references scored for global delayed review.',
    )
    parser.add_argument(
        '--pose_delayed_verification_global_min_distance',
        type=int,
        default=20,
        help='Minimum keyframe-index distance for a nonlocal delayed reference.',
    )
    parser.add_argument(
        '--pose_verification_photometric_review',
        action='store_true',
        help='Run bounded held-out photometric pose review for verify_v2 candidates.',
    )
    parser.add_argument(
        '--pose_verification_photometric_iterations',
        type=int,
        default=2,
        help='Pose-only optimization steps for the verify_v2 photometric candidate.',
    )
    parser.add_argument(
        '--pose_verification_photometric_min_coverage',
        type=float,
        default=0.15,
        help='Minimum historical-Gaussian coverage for photometric pose review.',
    )
    parser.add_argument(
        '--pose_verification_photometric_min_relative_improvement',
        type=float,
        default=0.002,
        help='Minimum held-out relative loss improvement for photometric acceptance.',
    )
    parser.add_argument(
        '--pose_verification_photometric_min_support_ratio',
        type=float,
        default=0.95,
        help='Minimum retained held-out support ratio for photometric acceptance.',
    )
    parser.add_argument(
        '--pose_verification_photometric_max_rotation_deg',
        type=float,
        default=0.5,
        help='Maximum photometric correction rotation in degrees.',
    )
    parser.add_argument(
        '--pose_verification_photometric_max_translation',
        type=float,
        default=0.01,
        help='Maximum photometric correction translation in scene units.',
    )
    parser.add_argument(
        '--pose_verification_photometric_scope',
        type=str,
        default='all',
        choices=['all', 'test_only'],
        help='Apply photometric verification to all verify_v2 candidates or only held-out test candidates.',
    )
    parser.add_argument(
        '--pose_verification_photometric_seed',
        type=str,
        default='post_geometry',
        choices=['post_geometry', 'raw'],
        help='Start photometric review from verify_v2 output or the unmodified incremental pose.',
    )
    parser.add_argument(
        '--pose_verification_photometric_lr_scale',
        type=float,
        default=1.0,
        help='Temporary pose-only learning-rate multiplier during bounded photometric review.',
    )
    parser.add_argument(
        '--pose_risk_utility_admission_mode',
        type=str,
        default='off',
        choices=[
            'off',
            'observe_v1',
            'active_v1',
            'pose_quarantine_v1',
            'pose_quarantine_utility_v1',
            'pose_quarantine_severe_v1',
        ],
        help='Joint post-pose admission using estimated pose risk and reduced-resolution rendering value.',
    )
    parser.add_argument(
        '--pose_risk_utility_threshold',
        type=float,
        default=0.24,
        help='Minimum representation utility for retaining a pose-risk candidate with pose-only review.',
    )
    parser.add_argument(
        '--pose_risk_utility_selectivity_reference',
        type=float,
        default=1.8,
        help='Residual-edge selectivity value mapped to full utility response.',
    )
    parser.add_argument(
        '--pose_risk_utility_probe_downsample',
        type=int,
        default=4,
        help='Spatial downsampling factor for the risk-candidate rendering probe.',
    )
    parser.add_argument(
        '--pose_risk_utility_isolation_risk_margin',
        type=float,
        default=0.04,
        help='Minimum risk excess above the adaptive threshold before low-utility isolation is allowed.',
    )
    parser.add_argument(
        '--pose_risk_utility_isolation_cooldown_frames',
        type=int,
        default=24,
        help='Minimum source-frame interval between low-utility isolation decisions.',
    )
    parser.add_argument(
        '--pose_risk_utility_quarantine_risk_margin',
        type=float,
        default=0.08,
        help='Minimum risk excess for excluding a frame only from future pose references.',
    )
    parser.add_argument(
        '--pose_risk_utility_quarantine_cooldown_frames',
        type=int,
        default=64,
        help='Minimum source-frame interval between pose-reference quarantine decisions.',
    )
    parser.add_argument(
        '--pose_risk_utility_review_iterations',
        type=int,
        default=2,
        help='Pose-only photometric review iterations for retained high-utility risk candidates.',
    )
    parser.add_argument(
        '--pose_risk_utility_use_verification_candidates',
        action='store_true',
        help='Route warmed-up verify_v2 candidates through the bounded photometric pose review.',
    )
    parser.add_argument(
        '--pose_risk_utility_review_test_candidates',
        action='store_true',
        help='Allow pose-only photometric review for selected test candidates.',
    )
    parser.add_argument(
        '--pose_risk_utility_review_min_coverage',
        type=float,
        default=0.15,
        help='Minimum established-Gaussian render coverage required for pose-only review.',
    )
    parser.add_argument(
        '--pose_risk_utility_review_max_rotation_deg',
        type=float,
        default=1.5,
        help='Maximum accepted rotation change from pose-only review, in degrees.',
    )
    parser.add_argument(
        '--pose_risk_utility_review_max_translation',
        type=float,
        default=0.05,
        help='Maximum accepted translation change from pose-only review.',
    )
    parser.add_argument(
        '--pose_risk_utility_review_min_relative_improvement',
        type=float,
        default=0.0,
        help='Minimum relative photometric-loss improvement required to retain a reviewed pose.',
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
        '--paper_aligned_recovery_commit_materialization',
        type=str,
        default='off',
        choices=['off', 'on', 'controlled'],
        help='Allow true-source recovery commits to materialize render keyframes. controlled routes them through recovery commit control.',
    )
    parser.add_argument(
        '--paper_aligned_defer_recovery_support_bridge',
        type=str,
        default=None,
        choices=['off', 'v1'],
        help='Defer-recoverable recovery support bridge (anchor transition + reference propagation).',
    )
    parser.add_argument(
        '--paper_aligned_support_bridge_keyframe_gate',
        type=str,
        default='off',
        choices=['off', 'on'],
        help='Allow support-bridge matches to override the baseline keyframe gate.',
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
            'recovery_commit_sparse_late_v8',
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
            'pose_rep_active_memory_v25',
            'pose_rep_active_memory_v26',
            'pose_rep_active_memory_v27',
            'pose_rep_active_memory_v28',
            'pose_rep_active_memory_v29',
            'pose_rep_active_memory_v30',
            'pose_rep_active_memory_v31',
            'pose_rep_active_memory_v33',
            'pose_rep_streaming_memory_v1',
            'pose_only_ssm_baseline_repr_v1',
            'pose_safe_streaming_memory_v1',
        ],
        help='Direct keyframe finalization density guard (not R/V/Q admission).',
    )
    parser.add_argument(
        '--paper_aligned_pose_memory_geometry_context',
        type=str,
        default='off',
        choices=['off', 'v1'],
        help='Use online pose-memory geometry context to calibrate pose/representation decoupling.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_assimilation_profile',
        type=str,
        default='off',
        choices=['off', 'render_frame_assimilation_v1', 'pose_only_render_skeleton_v1', 'pose_only_baseline_render_lock_v1', 'baseline_render_lock_intra_frame_v1', 'baseline_render_lock_intra_frame_v2', 'baseline_render_lock_intra_frame_v3', 'baseline_render_lock_intra_frame_v4', 'baseline_render_lock_intra_frame_v5', 'baseline_render_lock_intra_frame_v6', 'baseline_render_lock_intra_frame_v7', 'baseline_render_lock_intra_frame_v8', 'baseline_render_lock_intra_frame_v9', 'baseline_render_lock_intra_frame_v10', 'baseline_render_lock_intra_frame_v11', 'baseline_render_lock_intra_frame_v12', 'baseline_render_lock_intra_frame_v13', 'baseline_render_lock_intra_frame_v14', 'baseline_render_lock_intra_frame_v15', 'baseline_render_lock_intra_frame_v16', 'baseline_render_lock_intra_frame_v17', 'baseline_render_lock_intra_frame_v18', 'baseline_render_lock_intra_frame_v19', 'baseline_render_lock_intra_frame_v20', 'baseline_render_lock_intra_frame_v21', 'baseline_render_lock_intra_frame_v22', 'baseline_render_lock_intra_frame_v23', 'baseline_render_lock_intra_frame_v24', 'baseline_render_lock_intra_frame_v25', 'baseline_render_lock_intra_frame_v26', 'baseline_render_lock_intra_frame_v27', 'baseline_render_lock_intra_frame_v28', 'baseline_render_lock_intra_frame_v29', 'baseline_render_lock_intra_frame_v30', 'baseline_render_lock_intra_frame_v31', 'baseline_render_lock_intra_frame_v32', 'baseline_render_lock_intra_frame_v33', 'baseline_render_lock_intra_frame_v34', 'baseline_render_lock_intra_frame_v35', 'baseline_render_lock_intra_frame_v36', 'baseline_render_lock_intra_frame_v37', 'baseline_render_lock_intra_frame_v38', 'baseline_render_lock_intra_frame_v39', 'baseline_render_lock_intra_frame_v40', 'baseline_render_lock_intra_frame_v41', 'baseline_render_lock_intra_frame_v42', 'baseline_render_lock_intra_frame_v43', 'baseline_render_lock_intra_frame_v44', 'baseline_render_lock_intra_frame_v45', 'baseline_render_lock_intra_frame_v46', 'baseline_render_lock_intra_frame_v47', 'baseline_render_lock_intra_frame_v48', 'baseline_render_lock_intra_frame_v49', 'baseline_render_lock_intra_frame_v50', 'baseline_render_lock_intra_frame_v51', 'baseline_render_lock_intra_frame_v52', 'baseline_render_lock_intra_frame_v53', 'baseline_render_lock_intra_frame_v54', 'baseline_render_lock_intra_frame_v55', 'baseline_render_lock_intra_frame_v56', 'baseline_render_lock_intra_frame_v57', 'baseline_render_lock_intra_frame_v58', 'baseline_render_lock_intra_frame_v59', 'baseline_render_lock_intra_frame_v60', 'baseline_render_lock_intra_frame_v61', 'baseline_render_lock_intra_frame_v62', 'baseline_render_lock_intra_frame_v63', 'baseline_render_lock_intra_frame_v64'],
        help='Preset that keeps baseline render-frame selection while applying pose-risk-aware render assimilation.',
    )
    parser.add_argument(
        '--paper_aligned_v31_component_ablation',
        type=str,
        default='none',
        choices=[
            'none',
            'disable_response_sampling',
            'disable_extra_optimization',
        ],
        help='Disable exactly one final V31 render component for controlled ablation.',
    )
    parser.add_argument(
        '--paper_aligned_render_frame_policy',
        type=str,
        default='off',
        choices=['off', 'baseline_keyframe_lock_v1'],
        help='Render-frame selection policy used by paper-aligned runtime gates.',
    )
    parser.add_argument(
        '--paper_aligned_test_exposure_harmonization',
        type=str,
        default='neighbor_average_v1',
        choices=['off', 'neighbor_average_v1', 'source_time_interp_v1', 'source_time_guarded_v1', 'source_time_adaptive_v2', 'dark_scene_off_guarded_v1'],
        help='Exposure harmonization policy for held-out test render frames.',
    )
    parser.add_argument(
        '--paper_aligned_test_exposure_guard_max_delta',
        type=float,
        default=0.0,
        help='Max neighbor exposure matrix delta allowed before guarded source-time exposure falls back to baseline averaging.',
    )
    parser.add_argument(
        '--paper_aligned_test_render_calibration',
        type=str,
        default='off',
        choices=['off', 'diag_affine_v1'],
        help='Per-held-out-frame render calibration applied after rendering. off preserves baseline metrics.',
    )
    parser.add_argument(
        '--paper_aligned_training_background_mode',
        type=str,
        default='random_v1',
        choices=['random_v1', 'fixed_black_v1', 'target_mean_v1', 'deterministic_random_v1', 'dark_scene_fixed_black_v1', 'mask_aware_fixed_black_v1', 'mask_aware_dark_scene_fixed_black_v1', 'mask_aware_dark_scene_deterministic_random_v1'],
        help='Background color policy for online training renders; random_v1 preserves the baseline behavior.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_texture_sampling',
        type=str,
        default='off',
        choices=['off', 'residual_edge_v1', 'residual_edge_response_guard_v2', 'residual_edge_response_guard_v4', 'residual_edge_response_guard_non_dark_v5', 'residual_edge_response_guard_mask_conservative_v6'],
        help='Pose-render coupling for representation sampling. off keeps baseline/v8 sampling.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_texture_sampling_alpha',
        type=float,
        default=0.12,
        help='Max residual-edge sampling redistribution strength; preserves total sampling budget.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_texture_sampling_min_selectivity',
        type=float,
        default=0.18,
        help='Minimum residual-edge selectivity required before redistributing sampling probability.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_keyframe_sampling',
        type=str,
        default='off',
        choices=['off', 'pose_confidence_temporal_v1'],
        help='Pose-render confidence weighting for random training keyframe sampling. off keeps baseline sampling.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_keyframe_sampling_min_weight',
        type=float,
        default=0.35,
        help='Minimum random-sampling weight for pose-risky keyframes in pose_confidence_temporal_v1.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_keyframe_sampling_max_pose_risk',
        type=float,
        default=0.38,
        help='Pose risk above which pose_confidence_temporal_v1 uses the minimum sampling weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_keyframe_sampling_max_utility_drift',
        type=float,
        default=0.75,
        help='Utility drift above which pose_confidence_temporal_v1 uses the minimum sampling weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_keyframe_sampling_min_pose_support',
        type=float,
        default=0.35,
        help='Minimum pose support before pose_confidence_temporal_v1 uses the minimum sampling weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_keyframe_sampling_min_match_support',
        type=float,
        default=0.35,
        help='Minimum match support before pose_confidence_temporal_v1 uses the minimum sampling weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss',
        type=str,
        default='off',
        choices=['off', 'gradient_v1', 'gradient_pose_safe_v1', 'gradient_pose_adaptive_v1'],
        help='Pose-render coupling loss on image gradients. off keeps baseline/v8 optimization loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_weight',
        type=float,
        default=0.03,
        help='Weight for pose-render edge consistency loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_max_pose_risk',
        type=float,
        default=0.50,
        help='Maximum pose risk allowed by gradient_pose_safe_v1 edge loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_max_utility_drift',
        type=float,
        default=0.65,
        help='Maximum utility drift risk allowed by gradient_pose_safe_v1 edge loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_min_pose_support',
        type=float,
        default=0.45,
        help='Minimum pose support score required by gradient_pose_safe_v1 edge loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_min_match_support',
        type=float,
        default=0.45,
        help='Minimum match support score required by gradient_pose_safe_v1 edge loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_min_weight_scale',
        type=float,
        default=0.25,
        help='Minimum effective/base weight ratio for gradient_pose_adaptive_v1 edge loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_target_raw_loss',
        type=float,
        default=0.02,
        help='Raw gradient residual target used to clip gradient_pose_adaptive_v1 edge weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_edge_loss_risk_free_threshold',
        type=float,
        default=0.25,
        help='Posterior pose risk below which gradient_pose_adaptive_v1 preserves the base edge weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss',
        type=str,
        default='off',
        choices=['off', 'mse_pose_safe_v1', 'robust_mse_pose_risk_v1', 'freq_mse_pose_risk_v1'],
        help='Pose-safe RGB MSE auxiliary loss aligned with PSNR. off keeps baseline/v8 optimization loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_weight',
        type=float,
        default=0.10,
        help='Weight for the pose-safe RGB MSE auxiliary loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_max_pose_risk',
        type=float,
        default=0.30,
        help='Maximum posterior pose risk allowed by mse_pose_safe_v1 PSNR loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_max_utility_drift',
        type=float,
        default=0.55,
        help='Maximum utility drift risk allowed by mse_pose_safe_v1 PSNR loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_min_pose_support',
        type=float,
        default=0.45,
        help='Minimum pose support score required by mse_pose_safe_v1 PSNR loss.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_min_match_support',
        type=float,
        default=0.45,
        help='Minimum match support score required by mse_pose_safe_v1 PSNR loss.',
    )

    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_target_mask',
        type=str,
        default='off',
        choices=['off', 'nonzero_gt_v1', 'mask_aware_random_nonzero_gt_v1'],
        help='Optional target-validity mask for pose-safe RGB MSE. nonzero_gt_v1 aligns TUM-style black invalid pixels with evaluation masking.',
    )

    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_support_weight',
        type=str,
        default='off',
        choices=['off', 'raw_pose_support_v1', 'raw_pose_support_gate_v1'],
        help='Optional raw pose-support adaptive weight for pose-safe RGB MSE.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_context_weight',
        type=str,
        default='off',
        choices=['off', 'mask_aware_no_mask_boost_v1', 'mask_aware_no_mask_raw_response_boost_v1', 'mask_aware_no_mask_raw_response_gate_boost_v1', 'mask_aware_no_mask_scene_low_gate_boost_v1', 'mask_aware_no_mask_scene_low_fast_gate_boost_v1'],
        help='Optional scene-context adaptive weight for pose-safe RGB MSE.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_support_min_scale',
        type=float,
        default=0.45,
        help='Minimum multiplier used by raw_pose_support_v1 PSNR support weighting.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_support_corr_low',
        type=float,
        default=3000.0,
        help='Raw correspondence count where raw_pose_support_v1 starts lifting above the minimum scale.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_support_corr_high',
        type=float,
        default=8000.0,
        help='Raw correspondence count where raw_pose_support_v1 restores full PSNR loss weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_support_inlier_low',
        type=float,
        default=900.0,
        help='Final pose inlier count where raw_pose_support_v1 starts lifting above the minimum scale.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_support_inlier_high',
        type=float,
        default=2500.0,
        help='Final pose inlier count where raw_pose_support_v1 restores full PSNR loss weight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_max_applied_ratio',
        type=float,
        default=1.0,
        help='Maximum online applied/event ratio for pose-safe PSNR loss. Values below 1.0 turn the loss into a sparse correction budget.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_ambiguity_low',
        type=float,
        default=0.0,
        help='Lower raw-loss bound for skipping ambiguous pose-safe PSNR corrections. Disabled when high <= low.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_ambiguity_high',
        type=float,
        default=0.0,
        help='Upper raw-loss bound for skipping ambiguous pose-safe PSNR corrections. Disabled when high <= low.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_ambiguity_min_applied_ratio',
        type=float,
        default=1.0,
        help='Minimum online applied/event ratio before the ambiguous residual PSNR gate can skip corrections.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_scene_guard',
        type=str,
        default='off',
        choices=['off', 'raw_loss_ratio_guard_v1', 'raw_loss_ratio_precommit_guard_v1', 'raw_loss_ratio_precommit_render_gap_guard_v1', 'raw_loss_ratio_precommit_coverage_guard_v1', 'raw_loss_ratio_precommit_non_dark_guard_v2', 'raw_loss_ratio_precommit_non_dark_mask_guard_v3', 'mask_aware_dark_background_guard_v4'],
        help='Scene-level online guard for disabling unreliable pose-safe PSNR corrections.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_scene_guard_raw_low',
        type=float,
        default=0.0,
        help='Lower cumulative raw-loss mean bound for the scene-level PSNR guard.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_scene_guard_raw_high',
        type=float,
        default=0.0,
        help='Upper cumulative raw-loss mean bound for the scene-level PSNR guard.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_scene_guard_min_ratio',
        type=float,
        default=1.0,
        help='Minimum cumulative applied/event ratio required by the scene-level PSNR guard.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_scene_guard_min_events',
        type=int,
        default=0,
        help='Minimum cumulative PSNR-loss events required by the scene-level PSNR guard.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_scene_guard_min_applied',
        type=int,
        default=0,
        help='Minimum cumulative applied PSNR-loss events required by the scene-level PSNR guard.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_scene_guard_max_events',
        type=int,
        default=0,
        help='Latest cumulative PSNR-loss event index allowed to trigger the scene-level PSNR guard; 0 disables the limit.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_structure_gate',
        type=str,
        default='off',
        choices=['off', 'gradient_correlation_v1'],
        help='Low-cost full-frame structure agreement gate for pose-safe RGB MSE corrections.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_structure_min_score',
        type=float,
        default=0.0,
        help='Minimum gradient-correlation structure score required by the pose-safe RGB MSE gate.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_structure_min_raw_mean',
        type=float,
        default=0.0,
        help='Minimum projected cumulative raw PSNR loss mean before the structure gate can run.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_late_raw_stop',
        type=str,
        default='off',
        choices=['off', 'raw_mean_stop_v1'],
        help='Stop late PSNR corrections when the early scene guard window expired with high raw residuals.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_late_raw_stop_min_mean',
        type=float,
        default=0.0,
        help='Minimum projected cumulative raw PSNR loss mean required by the late raw stop gate.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_health_gate',
        type=str,
        default='off',
        choices=['off', 'gradient_correlation_v1'],
        help='Lightweight frame-internal health gate before applying pose-safe PSNR corrections.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_health_min_score',
        type=float,
        default=0.0,
        help='Minimum gradient-correlation score required by the PSNR health gate.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_psnr_loss_health_min_raw_loss',
        type=float,
        default=0.0,
        help='Minimum raw PSNR loss before the frame-internal health gate is evaluated.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization',
        type=str,
        default='off',
        choices=['off', 'pose_confidence_v1', 'pose_confidence_render_response_v2', 'render_response_v3', 'render_response_v4', 'render_response_v5', 'render_response_mask_conservative_v6'],
        help='Add bounded latest-keyframe optimization iterations when posterior pose confidence is high.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization_min_confidence',
        type=float,
        default=0.75,
        help='Minimum pose-render confidence required by pose_confidence_v1 extra optimization.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization_fraction',
        type=float,
        default=0.25,
        help='Fraction of base per-keyframe iterations used to compute extra optimization iterations.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization_max_extra',
        type=int,
        default=8,
        help='Maximum latest-keyframe extra optimization iterations per materialized keyframe.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization_max_pose_risk',
        type=float,
        default=0.35,
        help='Maximum posterior pose risk allowed by pose_confidence_v1 extra optimization.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization_max_utility_drift',
        type=float,
        default=0.60,
        help='Maximum utility drift risk allowed by pose_confidence_v1 extra optimization.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization_min_pose_support',
        type=float,
        default=0.55,
        help='Minimum pose support score required by pose_confidence_v1 extra optimization.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_extra_optimization_min_match_support',
        type=float,
        default=0.55,
        help='Minimum match support score required by pose_confidence_v1 extra optimization.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate',
        type=str,
        default='off',
        choices=['off', 'pose_confidence_v1', 'pose_confidence_soft_v1', 'pose_confidence_soft_psnr_health_v1'],
        help='Pose-confidence gate for Gaussian representation updates. Pose parameters still receive gradients.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate_min_confidence',
        type=float,
        default=0.35,
        help='Minimum pose-render confidence required to update Gaussian parameters.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate_max_pose_risk',
        type=float,
        default=0.55,
        help='Pose risk normalization threshold for pose_confidence_v1 update gate.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate_max_utility_drift',
        type=float,
        default=0.75,
        help='Utility drift normalization threshold for pose_confidence_v1 update gate.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate_soft_min_scale',
        type=float,
        default=0.65,
        help='Minimum Gaussian gradient scale for pose_confidence_soft_v1 update gate.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate_psnr_health_min_score',
        type=float,
        default=0.0,
        help='Minimum PSNR health score required before pose_confidence_soft_psnr_health_v1 caps Gaussian updates.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate_psnr_health_min_raw_loss',
        type=float,
        default=0.0,
        help='Minimum raw PSNR residual required before pose_confidence_soft_psnr_health_v1 caps Gaussian updates.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_update_gate_psnr_health_scale',
        type=float,
        default=1.0,
        help='Gaussian gradient scale used for high-structure high-residual PSNR health aliases.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_pre_refine',
        type=str,
        default='off',
        choices=['off', 'pose_only_v1'],
        help='Risk-aware pose-only render alignment before initializing new Gaussian points.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_pre_refine_iterations',
        type=int,
        default=2,
        help='Number of pose-only pre-refinement steps before adding new Gaussian points.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_pre_refine_min_pose_risk',
        type=float,
        default=0.08,
        help='Minimum posterior pose-render risk required for pose-only pre-refinement.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_pre_refine_max_pose_risk',
        type=float,
        default=0.38,
        help='Maximum posterior pose-render risk allowed for pose-only pre-refinement.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_pre_refine_min_existing_gaussians',
        type=int,
        default=5000,
        help='Minimum existing Gaussian count before pose-only pre-refinement may run.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_pre_refine_min_existing_keyframes',
        type=int,
        default=8,
        help='Minimum existing keyframe count before pose-only pre-refinement may run.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_pre_refine_min_render_coverage',
        type=float,
        default=0.18,
        help='Minimum current-render visible pixel coverage for pose-only pre-refinement.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting',
        type=str,
        default='off',
        choices=['off', 'risk_opacity_v1', 'risk_opacity_adaptive_v2'],
        help='Transfer posterior pose-render risk into new-Gaussian initialization confidence.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_min_pose_risk',
        type=float,
        default=0.16,
        help='Minimum posterior pose-render risk for new-Gaussian opacity downweighting.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_max_pose_risk',
        type=float,
        default=0.38,
        help='Risk value mapped to the maximum opacity downweight for new Gaussians.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_min_sample_opacity_scale',
        type=float,
        default=0.82,
        help='Minimum opacity scale for MVS/image-sampled new Gaussians under high risk.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_min_match_opacity_scale',
        type=float,
        default=0.92,
        help='Minimum opacity scale for triangulated match Gaussians under high risk.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_max_representation_value',
        type=float,
        default=0.65,
        help='Do not downweight new Gaussians when representation value is above this guard.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_max_novelty_value',
        type=float,
        default=0.65,
        help='Do not downweight new Gaussians when novelty/new-view value is above this guard.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_min_pose_risk',
        type=float,
        default=0.20,
        help='Adaptive v2 minimum pose risk for gentler new-Gaussian opacity downweighting.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_max_pose_risk',
        type=float,
        default=0.32,
        help='Adaptive v2 risk value mapped to the maximum gentler opacity downweight.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_min_sample_opacity_scale',
        type=float,
        default=0.88,
        help='Adaptive v2 minimum opacity scale for MVS/image-sampled new Gaussians.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_min_match_opacity_scale',
        type=float,
        default=0.95,
        help='Adaptive v2 minimum opacity scale for triangulated match Gaussians.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_novelty_threshold',
        type=float,
        default=0.45,
        help='Adaptive v2 preserves high-value new-view Gaussians above this novelty score.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_risk_override_alpha',
        type=float,
        default=1.0,
        help='Adaptive v2 still downweights high-novelty frames when risk alpha reaches this value.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_min_events',
        type=int,
        default=16,
        help='Adaptive v2 minimum observed init events before scene-level gentle calibration.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_gentle_pose_risk_mean',
        type=float,
        default=0.20,
        help='Adaptive v2 scene mean pose risk required for gentle high-novelty protection.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_gentle_novelty_mean',
        type=float,
        default=0.35,
        help='Adaptive v2 scene mean novelty required for gentle high-novelty protection.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_max_risk_alpha_mean',
        type=float,
        default=1.0,
        help='Optional adaptive v2 cap: disables gentle high-novelty protection when scene risk-alpha mean is above this value.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_low_pose_risk_mean',
        type=float,
        default=-1.0,
        help='Optional adaptive v2 bypass: pose-risk mean threshold; negative disables the bypass.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_low_novelty_mean',
        type=float,
        default=-1.0,
        help='Optional adaptive v2 bypass: novelty mean threshold; negative disables the bypass.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_low_risk_min_events',
        type=int,
        default=64,
        help='Optional adaptive v2 bypass: minimum observed init events before low-risk scene bypass.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_stable_low_risk_min_events',
        type=int,
        default=32,
        help='Optional adaptive v2 bypass: early window size for extremely stable low-risk scenes.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_stable_low_pose_risk_mean',
        type=float,
        default=0.06,
        help='Optional adaptive v2 bypass: stricter pose-risk mean for early stable low-risk bypass.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_stable_low_novelty_mean',
        type=float,
        default=0.12,
        help='Optional adaptive v2 bypass: stricter novelty mean for early stable low-risk bypass.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_risk_alpha_mean',
        type=float,
        default=2.0,
        help='Optional adaptive v2 bypass: risk-alpha mean threshold; values above 1 disable the bypass.',
    )
    parser.add_argument(
        '--paper_aligned_pose_render_init_weighting_adaptive_scene_high_uncertainty_bypass_novelty_mean',
        type=float,
        default=2.0,
        help='Optional adaptive v2 bypass: novelty mean threshold; values above 1 disable the bypass.',
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
        '--paper_aligned_recovery_v8_start_frame',
        type=int,
        default=500,
        help='Sparse-late v8 earliest runtime frame allowed to materialize recovery commits.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_sparse_density_upper_per_100',
        type=float,
        default=12.0,
        help='Sparse-late v8 maximum keyframe density before recovery materialization is considered sparse.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_materialized_budget_per_window',
        type=int,
        default=3,
        help='Sparse-late v8 maximum recovery materializations in the recent materialization window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_candidate_max_age',
        type=int,
        default=45,
        help='Sparse-late v8 maximum source age for recovery commit materialization.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_min_feasibility',
        type=float,
        default=0.52,
        help='Sparse-late v8 minimum materialization feasibility inherited from v6 rescue candidates.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_min_matches',
        type=int,
        default=450,
        help='Sparse-late v8 minimum feature match support for recovery materialization.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_min_inliers',
        type=int,
        default=1200,
        help='Sparse-late v8 bootstrap minimum pose inlier support before recovery materialization updates rendering.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_late_min_inliers',
        type=int,
        default=1200,
        help='Optional sparse-late v8 stricter pose inlier support after the bootstrap recovery window.',
    )
    parser.add_argument(
        '--paper_aligned_recovery_v8_bootstrap_end_frame',
        type=int,
        default=800,
        help='Sparse-late v8 frame index before which moderate-inlier recovery candidates may bootstrap the chain.',
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
