# V31+A Official-Protocol Pose Benchmark Results

## Frozen protocol

- Source order: zero-padded symlink streams over the original image bytes.
- Holdouts: 8 for MipNeRF360, 10 for StaticHikes, and 30 for TUM.
- Online recovery: `--enable_reboot` for baseline, V31, and V31+A, matching
  the baseline repository's official `scripts/train_eval_all.py` protocol.
- Execution: one visible GPU, all jobs serialized.
- Reference poses: COLMAP pseudo-ground truth for MipNeRF360 and StaticHikes;
  official motion-capture ground truth for TUM.
- Pose evaluation: one three-way common frame set per scene, independent
  Umeyama Sim(3) alignment per method, and delta-one relative pose error.

The earlier `official_pose_full9_20260712_160500` run omitted
`--enable_reboot`. It is diagnostic only and must not be used in the paper.

## Full nine-scene rendering results

Each cell is PSNR / SSIM / LPIPS / reconstruction seconds.

| Scene | Baseline | V31 | V31+A |
|---|---:|---:|---:|
| bonsai | 25.618 / 0.8566 / 0.2325 / 78.82 | 25.560 / 0.8640 / 0.2288 / 70.13 | 25.440 / 0.8609 / 0.2289 / 73.19 |
| counter | 25.903 / 0.8688 / 0.2252 / 62.69 | 26.115 / 0.8733 / 0.2203 / 62.41 | 26.025 / 0.8716 / 0.2215 / 63.75 |
| garden | 25.106 / 0.7588 / 0.2319 / 57.44 | 25.155 / 0.7648 / 0.2287 / 57.98 | 25.136 / 0.7640 / 0.2291 / 61.34 |
| forest1 | 18.258 / 0.5137 / 0.4029 / 152.47 | 17.713 / 0.4683 / 0.4356 / 120.60 | 18.247 / 0.5078 / 0.4045 / 122.76 |
| forest2 | 22.782 / 0.6529 / 0.3380 / 81.55 | 22.989 / 0.6548 / 0.3353 / 74.51 | 22.882 / 0.6536 / 0.3357 / 81.56 |
| university2 | 22.201 / 0.6905 / 0.2841 / 89.52 | 22.181 / 0.6893 / 0.2842 / 90.70 | 22.171 / 0.6936 / 0.2834 / 94.94 |
| desk | 21.584 / 0.7994 / 0.2761 / 34.77 | 21.847 / 0.8010 / 0.2684 / 36.91 | 21.228 / 0.7849 / 0.2814 / 35.79 |
| xyz | 27.278 / 0.8810 / 0.1706 / 100.04 | 27.564 / 0.8829 / 0.1694 / 57.11 | 27.284 / 0.8808 / 0.1705 / 56.89 |
| long_office | 21.206 / 0.7990 / 0.2722 / 88.90 | 21.569 / 0.8068 / 0.2681 / 75.23 | 21.043 / 0.8052 / 0.2663 / 90.39 |
| Macro mean | 23.326 / 0.7579 / 0.2704 / 82.91 | 23.410 / 0.7561 / 0.2710 / 71.73 | 23.273 / 0.7580 / 0.2690 / 75.62 |

V31+A beats baseline on 4/9 PSNR, 6/9 SSIM, and 7/9 LPIPS scenes. It
beats V31 on only 1/9 PSNR, 2/9 SSIM, and 3/9 LPIPS scenes. Its macro
SSIM/LPIPS advantage over V31 is therefore driven by a few large changes
rather than broad scene-wise consistency; macro PSNR is lower.

## Full nine-scene pose result

The table below reports V31+A minus V31. Negative values are better for every
metric.

| Scene | ATE-RMSE | APE-R deg | RPE-t RMSE | RPE-R deg |
|---|---:|---:|---:|---:|
| bonsai | -0.000540 | -0.0191 | -0.000180 | -0.0013 |
| counter | -0.000020 | +0.0021 | -0.000023 | +0.0005 |
| garden | +0.000066 | -0.0002 | +0.000045 | -0.0002 |
| forest1 | -0.391125 | -1.2226 | -0.006440 | -0.0120 |
| forest2 | -0.003634 | +0.2217 | -0.000285 | -0.0000 |
| university2 | -0.012934 | +0.1606 | -0.000438 | +0.0004 |
| desk | +0.003547 | +0.7826 | +0.000481 | +0.0432 |
| xyz | +0.001614 | -0.0753 | +0.000129 | -0.0000 |
| long_office | +0.063543 | +4.6047 | -0.002464 | +0.0219 |

A improves ATE-RMSE in 5/9 scenes, absolute rotation in 4/9, relative
translation in 6/9, and relative rotation in 5/9. The largest gain is the
forest1 recovery; desk degrades in all four pose metrics and all three
rendering metrics.

## Three paired official repeats

The full run is repeat one. Repeats two and three rerun V31 and V31+A on
desk, forest1, and long_office with the same reboot-enabled protocol. Desk is
a no-intervention control: all three A runs contain zero risk candidates and
zero quarantined references.

| Scene | Metric | V31 mean +/- std | V31+A mean +/- std | A-V31 delta | A wins |
|---|---|---:|---:|---:|---:|
| desk | PSNR | 22.0426 +/- 0.3557 | 21.3850 +/- 0.4081 | -0.6576 +/- 0.0794 | 0/3 |
| desk | SSIM | 0.8050 +/- 0.0067 | 0.7878 +/- 0.0102 | -0.0173 +/- 0.0043 | 0/3 |
| desk | LPIPS | 0.2638 +/- 0.0064 | 0.2804 +/- 0.0101 | +0.0166 +/- 0.0058 | 0/3 |
| desk | ATE-RMSE | 0.0694 +/- 0.0155 | 0.0884 +/- 0.0189 | +0.0190 +/- 0.0158 | 0/3 |
| desk | APE-R deg | 6.9146 +/- 1.5188 | 8.8782 +/- 1.0990 | +1.9636 +/- 1.0896 | 0/3 |
| forest1 | PSNR | 18.0092 +/- 0.2618 | 18.0736 +/- 0.1904 | +0.0644 +/- 0.4399 | 1/3 |
| forest1 | SSIM | 0.4894 +/- 0.0184 | 0.4945 +/- 0.0179 | +0.0051 +/- 0.0338 | 2/3 |
| forest1 | LPIPS | 0.4191 +/- 0.0145 | 0.4157 +/- 0.0128 | -0.0034 +/- 0.0263 | 2/3 |
| forest1 | ATE-RMSE | 0.5347 +/- 0.1642 | 0.4298 +/- 0.1432 | -0.1049 +/- 0.3059 | 2/3 |
| forest1 | APE-R deg | 2.4871 +/- 0.4002 | 2.1982 +/- 0.5437 | -0.2889 +/- 0.8580 | 2/3 |
| forest1 | RPE-t RMSE | 0.0198 +/- 0.0047 | 0.0203 +/- 0.0016 | +0.0005 +/- 0.0062 | 1/3 |
| forest1 | RPE-R deg | 0.4268 +/- 0.0017 | 0.4391 +/- 0.0268 | +0.0123 +/- 0.0274 | 1/3 |
| long_office | PSNR | 21.2965 +/- 0.2856 | 21.1477 +/- 0.0975 | -0.1488 +/- 0.3496 | 1/3 |
| long_office | SSIM | 0.8014 +/- 0.0057 | 0.8015 +/- 0.0032 | +0.0001 +/- 0.0033 | 1/3 |
| long_office | LPIPS | 0.2748 +/- 0.0061 | 0.2735 +/- 0.0065 | -0.0013 +/- 0.0037 | 2/3 |
| long_office | ATE-RMSE | 0.6790 +/- 0.1414 | 0.7347 +/- 0.1245 | +0.0557 +/- 0.0181 | 0/3 |
| long_office | APE-R deg | 13.3020 +/- 4.2249 | 15.4805 +/- 4.4466 | +2.1786 +/- 2.1133 | 0/3 |

## Desk order counterbalance

The first three desk pairs always ran V31 before V31+A. Three additional
pairs reversed this order. A wins 0/3 PSNR comparisons when it runs second
but 2/3 when it runs first, demonstrating a material order/non-determinism
confound. Pooling the six counterbalanced pairs gives:

| Metric | V31 mean +/- std | V31+A mean +/- std | A-V31 delta | A wins |
|---|---:|---:|---:|---:|
| PSNR | 21.9470 +/- 0.3487 | 21.7928 +/- 0.5804 | -0.1541 +/- 0.7459 | 2/6 |
| SSIM | 0.8021 +/- 0.0078 | 0.7981 +/- 0.0144 | -0.0040 +/- 0.0187 | 2/6 |
| LPIPS | 0.2660 +/- 0.0071 | 0.2702 +/- 0.0139 | +0.0041 +/- 0.0174 | 2/6 |
| ATE-RMSE | 0.0709 +/- 0.0206 | 0.0799 +/- 0.0164 | +0.0090 +/- 0.0284 | 1/6 |
| APE-R deg | 7.0559 +/- 1.4753 | 7.9477 +/- 1.3653 | +0.8918 +/- 2.0965 | 1/6 |
| RPE-t RMSE | 0.0237 +/- 0.0071 | 0.0244 +/- 0.0048 | +0.0007 +/- 0.0097 | 1/6 |
| RPE-R deg | 0.8812 +/- 0.0700 | 0.9247 +/- 0.1094 | +0.0436 +/- 0.1414 | 2/6 |

## Interpretation

The official protocol does not support treating A as a stable final-module
improvement. It can rescue a difficult forest1 trajectory, but the paired
effect is high variance, the relative pose metrics do not improve reliably,
and long_office absolute pose is worse in all three fixed-order repeats.

The desk control exposes a deeper experimental limitation. Its V31 and V31+A
trajectories already diverge during bootstrap, before the post-pose A decision
can run. All six A traces contain zero candidates and zero quarantines, yet
the paired metrics change substantially and depend on execution order. Thus
desk differences cannot be causally attributed to quarantine. A clean
mechanism ablation must compare a matched observer-only control against active
quarantine and counterbalance execution order.

For the final model, V31 should remain the rendering-method reference and A
should be reported as a diagnostic pose ablation unless its intervention is
redesigned and revalidated. The current overview must not claim that the A
module provides uniform pose or rendering gains.

## Artifacts

Formal full-run root:

`results/BRANCH_EXPERIMENTS_20260703/official_pose_full9_official_20260712_183200`

Primary reports:

- `evaluation/official_pose_report.md`
- `evaluation/pose_three_way.csv`
- `evaluation/pose_v31_vs_a.csv`
- `evaluation/rendering_summary.csv`
- `evaluation/a_module_diagnostics.csv`
- `evaluation/repeats/repeat_report.md`
- `evaluation/desk_counterbalanced/repeat_report.md`
