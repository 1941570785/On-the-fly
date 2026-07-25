from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from asr_gs.config import resolve_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="On-the-fly NVS and ASR-GS reconstruction"
    )

    data = parser.add_argument_group("data")
    data.add_argument("-s", "--source_path", required=True)
    data.add_argument("-i", "--images_dir", default="images")
    data.add_argument("--masks_dir", default="")
    data.add_argument("--num_loader_threads", type=int, default=4)
    data.add_argument("--downsampling", type=float, default=-1.0)
    data.add_argument("--pyr_levels", type=int, default=2)
    data.add_argument("--min_displacement", type=float, default=0.03)
    data.add_argument("--start_at", type=int, default=0)
    data.add_argument("--sh_degree", type=int, default=3)

    pose = parser.add_argument_group("camera and pose")
    pose.add_argument("--eval_poses", action="store_true")
    pose.add_argument("--use_colmap_poses", action="store_true")
    pose.add_argument("--num_kpts", type=int, default=int(4096 * 1.5))
    pose.add_argument("--match_max_error", type=float, default=2e-3)
    pose.add_argument("--fundmat_samples", type=int, default=2000)
    pose.add_argument("--min_num_inliers", type=int, default=100)
    pose.add_argument("--num_keyframes_miniba_bootstrap", type=int, default=8)
    pose.add_argument("--num_pts_miniba_bootstrap", type=int, default=2000)
    pose.add_argument("--iters_miniba_bootstrap", type=int, default=200)
    pose.add_argument("--enable_reboot", action="store_true")
    pose.add_argument("--fix_focal", action="store_true")
    pose.add_argument("--init_focal", type=float, default=-1.0)
    pose.add_argument("--init_fov", type=float, default=-1.0)
    pose.add_argument("--num_prev_keyframes_miniba_incr", type=int, default=6)
    pose.add_argument("--num_prev_keyframes_check", type=int, default=20)
    pose.add_argument("--pnpransac_samples", type=int, default=2000)
    pose.add_argument("--num_pts_miniba_incr", type=int, default=2000)
    pose.add_argument("--iters_miniba_incr", type=int, default=20)

    training = parser.add_argument_group("optimization")
    training.add_argument("--lr_poses", type=float, default=1e-4)
    training.add_argument("--lr_exposure", type=float, default=5e-4)
    training.add_argument("--lr_depth_scale_offset", type=float, default=1e-4)
    training.add_argument("--position_lr_init", type=float, default=0.00005)
    training.add_argument("--position_lr_decay", type=float, default=1 - 2e-5)
    training.add_argument("--feature_lr", type=float, default=0.005)
    training.add_argument("--opacity_lr", type=float, default=0.1)
    training.add_argument("--scaling_lr", type=float, default=0.01)
    training.add_argument("--rotation_lr", type=float, default=0.002)
    training.add_argument("--lambda_dssim", type=float, default=0.2)
    training.add_argument("--num_iterations", type=int, default=30)
    training.add_argument("--depth_loss_weight_init", type=float, default=1e-2)
    training.add_argument("--depth_loss_weight_decay", type=float, default=0.9)
    training.add_argument(
        "--save_at_finetune_epoch",
        type=int,
        nargs="+",
        default=[],
    )
    training.add_argument("--use_last_frame_proba", type=float, default=0.2)
    training.add_argument("--init_proba_scaler", type=float, default=2.0)

    scene = parser.add_argument_group("scene management")
    scene.add_argument("--anchor_overlap", type=float, default=0.3)
    scene.add_argument("--max_active_keyframes", type=int, default=200)

    method = parser.add_argument_group("method")
    method.add_argument(
        "--method",
        choices=("baseline", "asr-gs"),
        default="asr-gs",
    )
    method.add_argument("--ablate-a", action="store_true")
    method.add_argument("--ablate-b", action="store_true")
    method.add_argument("--ablate-c", action="store_true")
    method.add_argument(
        "--b-signal-mode",
        choices=("base", "r", "r_e", "r_d", "r_e_d"),
        default="r_e_d",
        help="Controlled B-module signal composition for internal ablation.",
    )
    method.add_argument("--experiment-seed", type=int, default=0)
    method.add_argument("--deterministic", action="store_true")

    evaluation = parser.add_argument_group("evaluation and output")
    evaluation.add_argument("--test_hold", type=int, default=-1)
    evaluation.add_argument("--test_frequency", type=int, default=-1)
    evaluation.add_argument("--display_runtimes", action="store_true")
    evaluation.add_argument("-m", "--model_path", default="")
    evaluation.add_argument("--save_every", type=int, default=-1)

    viewer = parser.add_argument_group("viewer")
    viewer.add_argument(
        "--viewer_mode",
        choices=("local", "server", "web", "none"),
        default="none",
    )
    viewer.add_argument("--ip", default="0.0.0.0")
    viewer.add_argument("--port", type=int, default=6009)
    return parser


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    ablations = {
        name
        for name, enabled in (
            ("a", args.ablate_a),
            ("b", args.ablate_b),
            ("c", args.ablate_c),
        )
        if enabled
    }
    if args.method == "baseline" and ablations:
        raise ValueError("component ablations require --method asr-gs")
    if args.method == "baseline" and args.b_signal_mode != "r_e_d":
        raise ValueError("B signal modes require --method asr-gs")
    args.asr_gs_config = resolve_config(
        args.method,
        ablations,
        sampling_mode=args.b_signal_mode,
    )

    if not args.model_path:
        index = 0
        while os.path.exists(f"results/{index:06d}"):
            index += 1
        args.model_path = f"results/{index:06d}"
    return args
