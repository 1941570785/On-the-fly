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
4. In `verify_v1`, a risk candidate triggers a second pose-only verification.
5. Reprojection residuals under `T0` are robustly filtered using median/MAD.
   Spatial cells retain their best-supported correspondence so that filtering
   cannot collapse all support into one image region.
6. The existing incremental MiniBA solver is rerun from `T0` on the cleaned
   correspondences. Its built-in Huber weighting and MAD outlier mask remain the
   only optimization robustifier.
7. The refined pose `Tr` is accepted only when all safe-acceptance checks pass;
   otherwise A returns `T0` exactly.
8. The accepted pose `TA` is passed through the existing pose field to B and C.

## Risk Trigger

`verify_v1` uses the same online median/MAD-calibrated risk score as the existing
observer. Verification is eligible after warmup when the score exceeds the
adaptive threshold and either multiple risk signals agree or pose uncertainty is
severe. Unlike `isolate_v1`, the trigger does not remove the frame and has no
cooldown: every eligible risky estimate may be checked, while the safe fallback
prevents unsupported changes.

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
- enough cleaned correspondences support the second MiniBA;
- the refined median reprojection error decreases by a minimum relative amount;
- the refined high-percentile error does not increase;
- the valid-support ratio does not collapse;
- translation and rotation corrections stay below conservative limits derived
  from the current scene scale and recent motion.

A rejected candidate is an explicit, measurable outcome rather than a frame
drop. This makes the module monotonic with respect to its own geometric
acceptance criterion: downstream reconstruction receives either a verified
improvement or the original baseline pose.

## Telemetry and Evidence

Each frame records risk, trigger status, support counts, robust cutoff, spatial
coverage, pre/post median and p90 reprojection errors, correction magnitudes,
accept/reject reason, and A runtime. The trace also stores the initial and final
pose matrices for direct GT evaluation.

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
