# V31 Risk-Utility Joint Admission Design

## Objective

Replace frame-level high-risk rejection with a rendering-aware admission decision while keeping the paper-table v31 reconstruction backbone fixed. The experiment asks whether pose-risk handling helps when a risky frame is rejected only when its expected representation value is also low.

## Fixed Backbone

- Baseline keyframe selection remains unchanged.
- The v31 residual-edge Gaussian sampling policy, sampling budget, bounded representation refinement, anchor management, and evaluation split remain unchanged.
- SSM, the historical candidate pool, and recovery commits remain disabled.
- The new policy is disabled by default and has no effect on the v31 control path.

## Joint Admission

The existing post-PnP/Mini-BA observer produces pose uncertainty, local state-support gap, temporal degradation, an aggregate risk score, and a robust online threshold. Only baseline-selected training frames above that threshold enter the utility path.

For a risk candidate, the current Gaussian scene is rendered once at reduced resolution from the estimated pose. The probe records three representation-value signals:

1. projected Gaussian coverage deficit;
2. residual-edge selectivity between the rendered view and the incoming image;
3. geometric new-view value already available from the online viewpoint statistics.

These signals form a bounded utility score. The decision has three outcomes:

- `admit`: the frame is not a risk candidate and follows exact v31 behavior;
- `review_admit`: the frame is risky but has high representation value, so it is retained and receives at most two pose-only photometric review steps before Gaussian growth;
- `isolate_low_utility`: the frame is risky and representation value is low, so its temporary match state is restored and it cannot update the keyframe, Gaussian, or anchor state.

The pose review optimizes only camera extrinsics against already-established Gaussians. It never updates Gaussian parameters. It is skipped when render support is insufficient, and its starting pose is restored if the review degrades supported photometric error or exceeds a bounded pose change. A temporal deferred pool is intentionally excluded from this version because delayed insertion would change keyframe indexing and confound the admission ablation.

## Calibration

An observe-only pass on `bonsai` and `forest1` records risk and utility distributions without changing reconstruction. Active thresholds are then fixed once for both scenes. No scene-specific thresholds are allowed.

## Evaluation

All runs use one GPU sequentially and `test_hold=8`. The new method is compared with:

1. the main-branch On-the-fly NVS baseline result;
2. the paper-table v31 result;
3. the new risk-utility joint-admission result.

PSNR, SSIM, LPIPS, reconstruction time, keyframe count, and admission statistics are reported. Pose evaluation uses the same common frame identities for all three methods, independent Sim(3) trajectory alignment to the same ground truth, and reports:

- absolute translation error (APE-t);
- absolute rotation error in degrees (APE-R);
- relative translation error at delta one (RPE-t);
- relative rotation error in degrees at delta one (RPE-R).

The report must include the common-frame count so that lower pose error cannot be claimed through silent frame removal.
