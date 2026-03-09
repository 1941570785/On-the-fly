import argparse
import os
import subprocess


SCENES_4 = [
    "StaticHikes/forest1",
    "TUM/desk2",
    "TUM/long_office_household",
    "MipNeRF360/garden",
]


VARIANTS = {
    "baseline": [
        "--no-enable_uncertainty_sampling",
        "--no-enable_residual_replay",
        "--no-enable_dynamic_suppression",
        "--no-rectify_colmap_cameras",
    ],
    "replay_only": [
        "--no-enable_uncertainty_sampling",
        "--enable_residual_replay",
        "--no-enable_dynamic_suppression",
        "--no-rectify_colmap_cameras",
    ],
    "uncertainty_replay": [
        "--enable_uncertainty_sampling",
        "--enable_residual_replay",
        "--no-enable_dynamic_suppression",
        "--no-rectify_colmap_cameras",
    ],
    "full": [
        "--enable_uncertainty_sampling",
        "--enable_residual_replay",
        "--enable_dynamic_suppression",
        "--rectify_colmap_cameras",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the four main scenes and the core innovation ablations."
    )
    p.add_argument("--base_dir", default="datasets")
    p.add_argument("--base_out_dir", default="results/innovation")
    p.add_argument("--downsampling", type=float, default=1.0)
    p.add_argument("--scenes", nargs="+", default=SCENES_4)
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--extra_train_args", type=str, default="")
    return p.parse_args()


def get_test_hold(scene: str) -> int:
    dataset = scene.split("/")[0]
    if dataset == "TUM":
        return 30
    if dataset == "MipNeRF360":
        return 8
    if dataset == "StaticHikes":
        return 10
    return -1


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    wrapper = os.path.join(repo_root, "scripts", "train_with_keyframe_export.py")
    diag_plotter = os.path.join(repo_root, "scripts", "plot_keyframe_diagnostics.py")

    for variant in args.variants:
        for scene in args.scenes:
            src = os.path.join(args.base_dir, scene)
            out = os.path.join(args.base_out_dir, variant, scene)
            os.makedirs(out, exist_ok=True)

            cmd = [
                "python",
                wrapper,
                "--export_keyframes",
                "--disable_after_inference",
                "--train_entry",
                "train.py",
                "-s",
                src,
                "--model_path",
                out,
                "--viewer_mode",
                "none",
                "--downsampling",
                str(args.downsampling),
            ]
            test_hold = get_test_hold(scene)
            if test_hold > 0:
                cmd += ["--test_hold", str(test_hold)]
            cmd += VARIANTS[variant]
            if args.extra_train_args.strip():
                cmd += args.extra_train_args.split()

            print("Running:", " ".join(cmd))
            if not args.dry_run:
                train_proc = subprocess.run(cmd, cwd=repo_root, check=False)
                if train_proc.returncode != 0:
                    print(
                        f"Skipping plots because training failed for {variant}:{scene} "
                        f"(exit code {train_proc.returncode})."
                    )
                    continue

                metrics_csv = os.path.join(out, "keyframe_metrics.csv")
                if not os.path.exists(metrics_csv):
                    print(
                        f"Skipping plots because metrics file was not created: {metrics_csv}"
                    )
                    continue
                plot_cmd = [
                    "python",
                    diag_plotter,
                    "--csv_path",
                    metrics_csv,
                    "--out_dir",
                    os.path.join(out, "diagnostic_plots"),
                    "--title",
                    f"{variant}:{scene}",
                ]
                print("Plotting:", " ".join(plot_cmd))
                subprocess.run(plot_cmd, cwd=repo_root, check=False)


if __name__ == "__main__":
    main()
