# Pose-Verification A Ablation Results

## Setup

- Branch: `codex/a-pose-verification-20260720`
- Training commit: `9bd05ea`
- Run root: `/data2/zxd/3D_Reconstruction/On_the_fly_pose_verification_a_20260720/results/BRANCH_EXPERIMENTS_20260703/pose_verification_a_full9_20260720_234111`
- Variants: `V31-Control`, `A-Observe`, and `A-Full`
- Protocol: identical v31 B/C/anchor settings, official test holds, one common
  reference-valid frame set per scene, and independent Sim(3) alignment.
- Execution: 27/27 jobs succeeded with at most three concurrent GPUs.

## Nine-Scene Macro Results

| Variant | PSNR | SSIM | LPIPS | Time (s) | APE-t RMSE | APE-R mean | RPE-t RMSE | RPE-R mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V31-Control | 23.4026 | 0.75743 | 0.26935 | 73.73 | 0.24529 | 3.7916 | 0.07904 | 0.6826 |
| A-Observe | 23.3754 | 0.75779 | 0.26918 | 68.59 | 0.23972 | 4.1647 | 0.07907 | 0.6877 |
| A-Full | 23.3925 | 0.75978 | 0.26833 | 70.12 | 0.22026 | 3.3291 | 0.07895 | 0.6769 |

Relative to V31-Control, A-Full reduces macro APE-t by 10.2%, APE-R by
12.2%, RPE-t by 0.11%, and RPE-R by 0.85%. Rendering changes are mixed: PSNR
changes by -0.010 dB, SSIM by +0.00235, and LPIPS by -0.00103.

## Dataset Macro Rendering

| Dataset | Variant | PSNR | SSIM | LPIPS | Time (s) |
|---|---|---:|---:|---:|---:|
| Mip-NeRF360 | V31-Control | 25.5848 | 0.83178 | 0.22735 | 60.33 |
| Mip-NeRF360 | A-Observe | 25.5813 | 0.83197 | 0.22612 | 60.86 |
| Mip-NeRF360 | A-Full | 25.5232 | 0.83095 | 0.22783 | 62.26 |
| StaticHikes | V31-Control | 21.0237 | 0.61067 | 0.34465 | 94.86 |
| StaticHikes | A-Observe | 21.0233 | 0.61318 | 0.34515 | 92.15 |
| StaticHikes | A-Full | 21.0916 | 0.61971 | 0.34074 | 94.10 |
| TUM RGB-D | V31-Control | 23.5993 | 0.82985 | 0.23607 | 66.01 |
| TUM RGB-D | A-Observe | 23.5217 | 0.82823 | 0.23628 | 52.76 |
| TUM RGB-D | A-Full | 23.5628 | 0.82868 | 0.23642 | 54.01 |

## Direct A Evidence

- The verifier observed 2,077 pose events and selected 195 candidates in all
  nine scenes.
- Scene-normalized candidate translation error is 1.168 times the
  non-candidate error (above one in 7/9 scenes). Rotation error is 1.224 times
  the non-candidate error (above one in 5/9 scenes).
- Twenty-four candidates were accepted in six scenes. Accepted updates reduce
  mean, median, and p90 reprojection error by 0.138, 0.225, and 0.110 pixels on
  average, respectively.
- All 171 rejected candidates return the original pose exactly.
- A uses 3.957 seconds over the nine full runs, or 20.3 ms per attempted
  verification. This is about 0.63% of A-Full's summed reconstruction time.

## Interpretation

The experiment supports A as a measurable auxiliary pose module: it identifies
harder pose estimates, safely verifies them, improves the nine-scene macro pose
metrics, and has bounded overhead. It does not support a claim of universal
rendering improvement. Accepted corrections improve GT translation for 12/24
events and GT rotation for 11/24 events; three scenes accept no correction.
Therefore A should be presented through candidate-error enrichment, safe
fallback, reprojection reduction, and macro trajectory accuracy, while B and C
remain the rendering-quality contributions.

Machine-readable results are in `evaluation/pose_scene.csv`,
`evaluation/render_scene.csv`, `evaluation/a_diagnostics_scene.csv`,
`evaluation/a_diagnostics_macro.csv`, and `evaluation/a_events.csv` under the
run root.
