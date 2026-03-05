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
    "StaticHikes/university1",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run 9 default scenes and export per-keyframe renders + metrics WITHOUT modifying train.py. "
            "This wraps train.py via runtime monkeypatch (see run_train_with_keyframe_export.py)."
        )
    )
    p.add_argument("--base_dir", default="data")
    p.add_argument("--base_out_dir", default="results")
    p.add_argument("--downsampling", default=1.0, type=float)
    p.add_argument("--extra_train_args", type=str, default="", help="Extra args appended to each train invocation")
    p.add_argument("--skip_train", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    wrapper = os.path.join(repo_root, "scripts", "run_train_with_keyframe_export.py")

    # Match the evaluation protocol described in the paper / train_eval_all.py
    test_hold = {
        "TUM": 30,
        "MipNerf360": 8,
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


if __name__ == "__main__":
    main()

