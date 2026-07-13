# V31+A Final Component Ablation Design

## Objective

Measure the independent contribution of the three components in the final V31+A model without changing the official data split, baseline keyframe policy, Gaussian sampling budget, or training schedule.

## Components

- **A:** pose-risk observation and pose-reference quarantine.
- **S:** budget-preserving residual-and-edge response sampling.
- **O:** response-conditioned bounded Gaussian-only extra optimization.

## Experiment Matrix

The frozen On-the-fly-NVS official run is the zero-component reference. Four V31-family variants are rerun from the same final commit:

| Variant | A | S | O |
|---|---:|---:|---:|
| `w_o_pose_risk_a` | 0 | 1 | 1 |
| `w_o_response_sampling` | 1 | 0 | 1 |
| `w_o_extra_optimization` | 1 | 1 | 0 |
| `full` | 1 | 1 | 1 |

Each removal is implemented after resolving the unchanged V31 profile. Therefore it changes only the selected component mode and preserves `baseline_keyframe_lock_v1`, response-sampling budget, all A thresholds, and all remaining V31 settings.

## Protocol

- Run all nine scenes using the official padded datasets and holdouts 8/10/30 for MipNeRF360, StaticHikes, and TUM RGB-D.
- Enable the official reboot policy and disable the viewer.
- Execute serially with exactly one visible GPU.
- Record command provenance, Git commit, metadata, risk traces, and incremental manifests.
- Compare PSNR, SSIM, LPIPS, reconstruction time, APE translation/rotation, RPE translation/rotation, keyframes, anchors, response-sampling applications, extra iterations, risk candidates, and quarantines.

## Interpretation Boundary

A component receives causal credit only when its leave-one-out removal changes an actually triggered mechanism. Single-run differences in scenes with zero relevant triggers are treated as runtime variance, not module gain. The primary paper conclusion is based on the nine-scene macro result and per-scene direction; high-variance or intervention-active scenes are candidates for targeted repeats.
