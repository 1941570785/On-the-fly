import argparse
import os
import subprocess


SCENES_9 = [
    "TUM/desk1",
    "TUM/desk2",
    "TUM/long_office_household",
    "MipNeRF360/garden",
    "MipNeRF360/bonsai",
    "MipNeRF360/counter",
    "StaticHikes/forest1",
    "StaticHikes/forest2",
    "StaticHikes/university2",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run 9 default scenes and export per-keyframe renders + metrics WITHOUT modifying train.py. "
            "This wraps train.py via runtime monkeypatch (see train_with_keyframe_export.py)."
        )
    )
    p.add_argument("--base_dir", default="data")
    p.add_argument("--base_out_dir", default="results")
    p.add_argument("--downsampling", default=1.0, type=float)
    p.add_argument("--extra_train_args", type=str, default="", help="Extra args appended to each train invocation")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--plot_diagnostics", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    wrapper = os.path.join(repo_root, "scripts", "train_with_keyframe_export.py")
    diag_plotter = os.path.join(repo_root, "scripts", "plot_keyframe_diagnostics.py")

    # Match the evaluation protocol described in the paper / train_eval_all.py
    test_hold = {
        "TUM": 30,
        "MipNeRF360": 8,
        "StaticHikes": 10,
    }

    if not args.skip_train:
        for scene in SCENES_9:
            dataset = scene.split("/")[0]
            th = test_hold.get(dataset, -1)

            src = os.path.join(args.base_dir, scene)
            out = os.path.join(args.base_out_dir, scene)
            cmd = [
                "python",
                wrapper,
                "--export_keyframes",
                "--disable_after_inference",
                "--train_entry",
                "train.py",
            ]

            # Pass-through args to train.py
            cmd += [
                "-s",
                src,
                "--model_path",
                out,
                "--viewer_mode",
                "none",
                "--downsampling",
                str(args.downsampling),
            ]
            if th > 0:
                cmd += ["--test_hold", str(th)]

            if args.extra_train_args.strip() != "":
                cmd += args.extra_train_args.split()

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=repo_root, check=False)

            if args.plot_diagnostics:
                metrics_csv = os.path.join(out, "keyframe_metrics.csv")
                diag_out_dir = os.path.join(out, "diagnostic_plots")
                plot_cmd = [
                    "python",
                    diag_plotter,
                    "--csv_path",
                    metrics_csv,
                    "--out_dir",
                    diag_out_dir,
                    "--title",
                    scene,
                ]
                print("Plotting:", " ".join(plot_cmd))
                subprocess.run(plot_cmd, cwd=repo_root, check=False)


if __name__ == "__main__":
    main()

