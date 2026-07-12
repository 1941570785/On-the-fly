# V31+A Official Pose Evaluation Design

## Objective

Evaluate On-the-fly NVS, the archived V31 rendering model, and V31+A under one
official per-dataset protocol. The experiment must measure whether A's
pose-reference quarantine improves trajectory accuracy without sacrificing
rendering quality or runtime.

## Frozen Methods

- **On-the-fly NVS:** `main` implementation at commit `8fed9bf`.
- **V31:** the `baseline_render_lock_intra_frame_v31` profile with pose-risk and
  risk-utility admission disabled.
- **V31+A:** the same V31 profile with the post-pose observer enabled and
  `pose_quarantine_v1` admission. Rendering-frame admission remains unchanged;
  only extreme-risk frames are excluded from later pose-reference selection.

The active A configuration is fixed globally: absolute risk threshold 0.10,
adaptive sigma 2.0, warmup 8, history 64, quarantine margin 0.08, quarantine
cooldown 64 frames, and zero pose-review iterations.

## Official Dataset Protocol

All methods receive the same padded image streams in the same order. The
holdout interval is 8 for MipNeRF360, 10 for StaticHikes, and 30 for TUM.
All commands enable the official `--enable_reboot` policy used by
`scripts/train_eval_all.py`. Runs execute sequentially with exactly one
visible CUDA device.

Reference trajectories are deliberately distinguished:

- TUM uses the recovered official motion-capture trajectory in `sparse/GT`.
- MipNeRF360 and StaticHikes use the complete COLMAP reconstruction in
  `sparse/0` as pseudo-ground truth.

The paper and generated reports must call all nine trajectories "reference
poses" and reserve "ground truth" for TUM.

## Pose Evaluation

Image identities are canonicalized by numeric filename stem so padded and
unpadded names match without positional assumptions. Estimated matrices are
read from each model's `metadata.json`; reference matrices are loaded directly
from COLMAP or the recovered TUM arrays.

Primary three-way metrics are computed on the intersection of reference-valid
keyframes from all three methods. Each method is independently aligned to the
same reference trajectory with Umeyama Sim(3), which is required for monocular
scale ambiguity. The main table reports:

- ATE translation RMSE;
- absolute rotation mean in degrees;
- RPE translation RMSE at delta one;
- RPE rotation mean in degrees at delta one;
- common-frame count and per-method trajectory coverage.

A second paired table compares V31+A with V31 on their larger pairwise common
set, isolating A from baseline keyframe differences. Raw translation errors
are never macro-averaged across datasets because COLMAP scene units are not
commensurate with TUM metres.

## Rendering And A-Module Diagnostics

The same official-protocol runs also report PSNR, SSIM, LPIPS, runtime,
keyframe count, risk candidates, quarantined references, and cooldown admits.
Risk calibration is evaluated by Spearman correlation between risk score and
aligned translation/rotation error, error enrichment of flagged frames, and
the reference-error percentile of every quarantined frame.

The full nine-scene benchmark is followed by three paired V31/V31+A repeats on
`forest1` and `long_office`, using one GPU serially. These scenes test the A
module under sparse wide-baseline motion and long official-TUM trajectories.

## Deliverables

The run directory contains commands, logs, per-run metadata, completeness
manifests, CSV/JSON/Markdown pose tables, rendering tables, delta tables, and
A-module diagnostic tables. A failed or incomplete scene prevents aggregate
report generation.
