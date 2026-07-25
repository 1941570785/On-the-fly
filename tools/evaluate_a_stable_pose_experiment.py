#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_a_v2_baseline_pose_experiment import evaluate_experiment
from tools.run_a_stable_pose_experiment import PRESETS, SCENES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--repeats", nargs="+", type=int, required=True)
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=tuple(PRESETS),
        required=True,
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENES),
        required=True,
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_root / "evaluation"
    evaluate_experiment(
        args.run_root,
        output_dir,
        repeats=args.repeats,
        variants=args.presets,
        scene_names=args.scenes,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
