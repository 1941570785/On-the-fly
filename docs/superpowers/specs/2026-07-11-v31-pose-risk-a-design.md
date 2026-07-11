# V31 Pose-Initialization Risk A-Module Design

## Goal

Add the overview panel-(a) pose-risk path to the render-only v31 pipeline without restoring SSM, a recovery candidate pool, or recovery commits. The experiment must identify the value of pose-side risk handling independently from v31 response-guided Gaussian sampling and bounded representation refinement.

## Fixed V31 Backbone

- Render-frame policy: `baseline_keyframe_lock_v1`.
- Gaussian sampling: `residual_edge_response_guard_v2`, alpha `0.03`, selectivity threshold `1.8`.
- Extra optimization: `render_response_v3`.
- PSNR loss and update gate: off.
- Test split: `test_hold=8`.

The A module is disabled by default and therefore cannot alter the existing v31 path unless explicitly selected.

## Post-Pose Risk

Risk is evaluated after incremental PnP-RANSAC and Mini-BA succeed and before a selected training frame writes to the Gaussian representation.

1. **Pose uncertainty** combines absolute inlier support, inlier-to-correspondence consensus, and PnP-to-Mini-BA inlier retention.
2. **State-support gap** combines spatial inlier coverage, spatial entropy, active-anchor health, and selected-reference support.
3. **Temporal degradation** combines recent pose-failure rate with a robust anomaly score for the current rotation step relative to recent pose steps.

The aggregate score uses weights `0.45`, `0.35`, and `0.20`. Its online threshold is the larger of an absolute floor (`0.10`) and `median + 2.0 * 1.4826 * MAD` over the latest 64 eligible scores. Isolation starts after eight observations and requires elevated pose uncertainty plus a second risk source. A 12-frame cooldown prevents repeated isolation inside one difficult temporal burst.

## Runtime Modes

- `off`: exact v31 behavior.
- `observe_v1`: compute and trace risk without changing admission.
- `isolate_v1`: prevent a high-risk selected training frame from becoming a keyframe or updating Gaussians.

Test views and bootstrap frames are never isolated. When a post-pose frame is isolated, temporary descriptor-match state is restored so the reused keyframe index cannot contaminate the next frame.

## Evaluation

Run sequentially on one GPU for `bonsai` and `forest1`:

- `V31_control`
- `V31_A_observe`
- `V31_A_isolate`

Report PSNR, SSIM, LPIPS, reconstruction time, keyframes, anchors, rotation error, translation error, risk distribution, isolated frames, and cooldown-blocked events. Because the CUDA path is not bitwise deterministic, repeat the control and interpret active changes against the observed control range rather than a single run.
