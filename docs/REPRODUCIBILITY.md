# Reproducibility Notes

## Fixed Release Identity

The release is valid only when all runs report the same
`config_fingerprint` for a given ablation. The final ASR-GS
configuration fixes:

- A: support/residual secondary pose review with at most two additional
  hypotheses;
- B: response strength `0.03` and selectivity threshold `1.8`;
- C: extra-iteration fraction `0.25`, upper bound `K=8`, pose frozen,
  and transactional rollback.

The paper's dataset-level pose macro row is stored in
`reproducibility/paper_pose_macro.json`. Rotations are reported in
radians and translations use `100 x` the reconstruction coordinate
unit after Sim(3) alignment, matching the evaluation protocol used to
produce that row.

## Required Reporting

Report every scene separately before taking the arithmetic mean over
the three scenes in each dataset. Include PSNR, SSIM, LPIPS, wall-clock
reconstruction time, seed, Git revision, and configuration fingerprint.
For A/B/C ablations, also report the activation counters from
`scene_metrics.csv`.

Use at least three repeated runs when estimating stochastic variance.
Do not combine outputs with different fingerprints in one aggregate.

## Baseline Control

Use `--method baseline` in this repository. It disables A, B, and C
while preserving the code revision, environment, input order, test
split, and anchor implementation. This is the controlled baseline for
component attribution.
