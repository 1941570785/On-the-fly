# V31 Risk-Utility Admission Results

## Scope

This experiment evaluates a conservative pose-reference quarantine policy on top
of the archived V31 rendering model. Rendering-frame admission remains identical
to V31. A frame is excluded only from future pose-reference selection when its
pose-risk margin is at least 0.08, and another quarantine is forbidden for the
next 64 source frames. No pose-review optimization is used.

All runs were executed sequentially on one GPU. The dataset protocols are
`test_hold=8` for bonsai and `test_hold=10` for forest1. Pose trajectories are
evaluated for V31 and the new model on one common frame set per scene after an
independent Sim(3) alignment to the same ground-truth trajectory. The true
On-the-fly-NVS baseline values are the preserved full-precision values behind
the paper tables. Its original trajectory artifact was overwritten by a later
rerun, so baseline APE/RPE are intentionally left unreported. Translation
errors use the native dataset coordinate scale and rotation errors are in
degrees.

## Three-Way Protocol-Matched Result

| Scene | Method | Quality/Pose frames | PSNR | SSIM | LPIPS | APE-t | APE-R | RPE-t | RPE-R | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai | On-the-fly-NVS | table/-- | 24.2541 | 0.8119 | 0.2499 | -- | -- | -- | -- | 67.059 |
| bonsai | archived V31 | 37/234 | 25.4770 | 0.8574 | 0.2283 | 0.03715 | 0.6551 | 0.00511 | 0.0645 | 65.717 |
| bonsai | pose quarantine | 37/234 | 25.5957 | 0.8629 | 0.2287 | 0.03987 | 0.6972 | 0.00530 | 0.0658 | 69.491 |
| forest1 | On-the-fly-NVS | table/-- | 17.7886 | 0.4846 | 0.4262 | -- | -- | -- | -- | 109.036 |
| forest1 | archived V31 | 120/343 | 17.9688 | 0.4885 | 0.4239 | 0.60230 | 2.7866 | 0.01411 | 0.4262 | 116.382 |
| forest1 | pose quarantine | 120/343 | 17.9553 | 0.4898 | 0.4224 | 0.49902 | 2.9334 | 0.01245 | 0.4788 | 122.741 |

Relative to the true On-the-fly-NVS baseline, the new model improves bonsai by
+1.3416 dB PSNR, +0.0510 SSIM, and -0.0212 LPIPS, and improves forest1 by
+0.1668 dB PSNR, +0.0052 SSIM, and -0.0039 LPIPS. Runtime increases by 2.43
seconds and 13.71 seconds, respectively.

The bonsai run produced one risk candidate but no quarantine, so its numerical
difference from V31 cannot be attributed to the admission action. Forest1
quarantined two pose references. Relative to archived V31, it improved SSIM,
LPIPS, APE-t, and RPE-t, while PSNR, APE-R, RPE-R, and runtime regressed.

## Forest1 Paired Repeats

The paired control uses the same branch, protocol, V31 profile, risk estimator,
and rendering probe. `Observe` records decisions without changing pose
references. `Quarantine` applies the conservative pose-reference exclusion.

| Variant (n=3) | PSNR | SSIM | LPIPS | APE-t | APE-R | RPE-t | RPE-R | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Observe | 17.7856 +/- 0.1789 | 0.4735 +/- 0.0105 | 0.4345 +/- 0.0130 | 0.58587 +/- 0.12024 | 3.5862 +/- 0.9841 | 0.01450 +/- 0.00245 | 0.4625 +/- 0.0545 | 115.807 +/- 1.368 |
| Quarantine | 17.9198 +/- 0.1128 | 0.4845 +/- 0.0096 | 0.4238 +/- 0.0082 | 0.52381 +/- 0.05719 | 3.2259 +/- 0.8086 | 0.01303 +/- 0.00132 | 0.4514 +/- 0.0263 | 117.967 +/- 4.386 |
| Paired delta (Q - Observe) | +0.1342 | +0.0110 | -0.0107 | -0.06206 | -0.3603 | -0.00147 | -0.0111 | +2.160 |

Quarantine improved all eight quality/pose quantities in the first pair and all
but RPE-R in the second pair. The third pair was a counterexample and regressed
all quality and pose quantities. The mean is favorable, and the quality/pose
standard deviations are lower, but the per-run direction is not stable.

Compared with archived V31, the three-run quarantine mean is -0.0490 dB PSNR,
-0.0040 SSIM, and -0.00001 LPIPS. It improves APE-t by 0.0785 and RPE-t by
0.00108, but worsens APE-R by 0.4392 degrees and RPE-R by 0.0252 degrees. Mean
runtime increases by 1.58 seconds.

## Risk Diagnostic

The protocol-matched forest1 run produced 25 risk candidates, two quarantines,
and five cooldown admissions. The quarantined frames were not the largest
translation-error frames: their aligned translation-error percentiles were 7.3
and 24.2. Their rotation-error percentiles were 47.5 and 88.0. Across the risk
candidates, risk-margin correlation was 0.052 with translation APE and 0.319
with rotation APE. The score therefore has limited calibration to true pose
error, especially for translation.

## Decision

The complete V31-plus-quarantine model exceeds the true On-the-fly-NVS baseline
on all three rendering metrics in both tested scenes. This establishes the
overall model gain over the correct baseline. The incremental contribution of
pose quarantine must be judged against archived V31, however. It is a useful
stability ablation that can protect a weak forest1 trajectory and reduce error
variance, but it does not provide a consistent per-run improvement or dominate
V31 across rendering quality, translation, rotation, and runtime
simultaneously. It should therefore remain an auxiliary robustness mechanism,
not replace response-guided sampling and bounded representation optimization as
the main contributions, unless the pose-risk score is recalibrated and the
policy is validated across the full nine-scene benchmark.

## Artifacts

- Protocol-matched run:
  `results/BRANCH_EXPERIMENTS_20260703/v31_risk_utility_pose_quarantine_protocol_20260711`
- Corrected three-way report:
  `results/BRANCH_EXPERIMENTS_20260703/v31_risk_utility_pose_quarantine_protocol_20260711/comparison_corrected_table_baseline/three_way_comparison.md`
- Paired observe run 1:
  `results/BRANCH_EXPERIMENTS_20260703/v31_risk_utility_observe_protocol_20260711`
- Paired observe/quarantine run 2:
  `results/BRANCH_EXPERIMENTS_20260703/v31_risk_utility_forest1_repeat2_protocol_20260711`
- Paired observe/quarantine run 3:
  `results/BRANCH_EXPERIMENTS_20260703/v31_risk_utility_forest1_repeat3_protocol_20260711`
