# Pose-Only A: Risk-Triggered Geometric Verification

## Objective

Replace the final v31+A hard pose quarantine with a small pose-only module that
produces a measurably better pose prior for the unchanged Gaussian sampling (B)
and Gaussian-only extra optimization (C) stages.

The module must be useful through direct pose evidence. It is not required to
improve every rendering metric, but it must preserve the existing reconstruction
path whenever verification cannot demonstrate a safer pose.

## Scope

A operates after the baseline PnP-RANSAC and incremental MiniBA pose estimate and
before any new Gaussian is sampled. It may read only pose-estimation evidence:

- the initial world-to-camera pose;
- the 2D-3D correspondences already used by pose initialization;
- PnP and MiniBA support diagnostics;
- spatial support and recent pose history already used by the risk observer.

A does not alter frame admission, reference-frame membership, Gaussian sampling,
optimization losses, extra-optimization decisions, or anchor maintenance. There
is no SSM, deferred candidate pool, render probe, pixel reliability map, or
sampling adapter in this design.

## Per-Frame Flow

1. Baseline PnP-RANSAC and MiniBA produce the initial pose `T0`.
2. The existing observer computes continuous pose risk from geometric support,
   spatial support, anchor/reference support, recent failures, and rotation
   anomaly.
3. In `observe_v1`, the risk and initial-pose diagnostics are recorded but `T0`
   is returned unchanged.
4. Both `observe_v1` and `verify_v1` record the same risk-candidate decision;
   only `verify_v1` executes the pose solver.
5. Reprojection residuals under `T0` are robustly filtered using median/MAD.
   Spatial cells retain their best-supported correspondence so that filtering
   cannot collapse all support into one image region.
6. PnP-RANSAC is rerun on the cleaned, full-resolution correspondence set and
   its candidate is polished by the existing incremental MiniBA solver. The
   solver's built-in Huber weighting and MAD outlier mask remain the only
   optimization robustifier.
7. The refined pose `Tr` is accepted only when all safe-acceptance checks pass;
   otherwise A returns `T0` exactly.
8. The accepted pose `TA` is passed through the existing pose field to B and C.

## Risk Trigger

`verify_v1` uses the same online median/MAD-calibrated risk score as the existing
observer. After warmup, a frame becomes a verification candidate when its score
enters the adaptive upper tail and at least one pose, state-support, or temporal
signal is non-trivial. This lower diagnostic tail is intentionally separate from
the conservative hard-isolation threshold: A verifies questionable estimates
without dropping them. It has no cooldown, and the safe fallback prevents
unsupported changes.

## Robust Correspondence Selection

For each valid 3D point, A projects the point with `T0` and measures its pixel
reprojection error. Invalid or behind-camera points are rejected. The adaptive
cutoff is the median plus a configurable MAD multiplier, bounded below by one
pixel and above by the original PnP error threshold. If filtering removes too
much support, the lowest-residual correspondences are restored until the minimum
verification support is reached.

The image is divided into a small fixed grid. For every occupied cell, its
lowest-residual correspondence is retained. This protects geometric leverage
without adding a learned model or a scene-specific mask.

## Safe Acceptance

The refined pose is accepted only if:

- both poses and all diagnostics are finite;
- enough cleaned correspondences support the second PnP/MiniBA pass;
- both mean error and median error improve, with at least 2% relative median
  reduction;
- the refined p90 error stays within 1% of the initial p90;
- the valid-support ratio does not collapse;
- translation and rotation corrections stay below conservative limits derived
  from the current scene scale and recent motion.

A rejected candidate is an explicit, measurable outcome rather than a frame
drop. This makes the module monotonic with respect to its own geometric
acceptance criterion: downstream reconstruction receives either a verified
improvement or the original baseline pose.

## Telemetry and Evidence

Each frame records the mode-independent candidate status, solver trigger,
support counts and ratios, robust cutoff, spatial coverage, pre/post mean,
median and p90 reprojection errors, correction magnitudes, accept/reject reason,
and A runtime. The trace also stores the initial, final, and available dataset
pose matrices for direct GT evaluation. PnP's RNG state is restored after the
verification attempt so rejected checks do not perturb later baseline sampling.

Primary A evidence:

- APE translation and rotation;
- RPE translation and rotation;
- registration success rate;
- PnP/MiniBA support and inlier ratios;
- pre/post reprojection median and p90;
- trigger, acceptance, improvement, and harmful-correction rates;
- Spearman correlation between risk and GT pose error;
- A runtime overhead.

Rendering PSNR, SSIM, LPIPS, and total time remain side-effect checks.
