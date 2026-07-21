# Latest-A C Extra-Round Sensitivity Design

## Objective

Determine the best maximum number of Gaussian-only extra-optimization rounds in
module C for the final model that uses the risk-triggered pose-verification A.
The experiment compares only round budgets within the same final model. It does
not compare against older model versions or external baselines.

## Frozen Model Contract

- Repository: `/data2/zxd/3D_Reconstruction/On_the_fly_pose_verification_a_20260720`.
- Branch: `codex/a-pose-verification-20260720`.
- Starting commit: `75bddb6`.
- A mode: `verify_v1` with the parameters used by the completed nine-scene A
  ablation.
- B and the assimilation profile: `baseline_render_lock_intra_frame_v31`.
- Base joint optimization: 30 iterations with the existing RGB L1, DSSIM, and
  aligned inverse-depth losses.
- C mode: `render_response_v3`; camera pose remains frozen during all extra
  iterations.
- Anchor maintenance, frame admission, test holds, datasets, image resolution,
  and all other arguments remain unchanged.

## Independent Variable

`K` is the maximum number of extra Gaussian-only iterations per response-gated
C event. The response gate and response scale remain enabled. For positive even
budgets, use `fraction=(K-0.5)/30` and `max_extra=K`; `K=0` disables only the
extra-optimization stage. Report both requested `K` and realized iteration
counts because the response scale can produce fewer than `K` iterations.

The initial sweep is:

```text
K = {0, 2, 4, 8, 12, 16, 20}
```

If the selected quality region reaches the right boundary, extend first to 24
and 30. If needed, continue in increments of ten. Do not reuse any old result or
checkpoint.

## Complete-Run Protocol

Every `(K, scene)` job starts from an empty model directory and runs the complete
online pipeline. A job is complete only when the training process returns zero,
`run_status.json` records success, and `model/metadata.json` is readable.

Run all nine scenes with experiment seed zero:

```text
Mip-NeRF360: bonsai, counter, garden
StaticHikes: forest1, forest2, university2
TUM RGB-D: desk, xyz, long_office
```

Use official test holds 8, 10, and 30 respectively. Each child process receives
one physical GPU through `CUDA_VISIBLE_DEVICES`. Up to six currently idle RTX
4090 GPUs may run concurrently. Do not install or update packages in `otf`.

## Metrics And Selection

Collect per scene PSNR, SSIM, LPIPS, reconstruction time, wall time, keyframes,
anchors, C events, accepted C events, total realized extra iterations, and mean
realized iterations per accepted event.

For every K, compute three dataset macro means and the equal-scene nine-scene
macro mean. PSNR is the primary quality metric; SSIM and LPIPS are mandatory
consistency checks. A smaller K is quality-equivalent to the maximum-PSNR K when
its macro PSNR is within 0.02 dB and SSIM and LPIPS do not both degrade.
Among quality-equivalent settings, select the smallest K on the quality-time
Pareto frontier.

After a provisional optimum, stop only after two larger tested budgets both:

1. lose more than 0.02 dB macro PSNR relative to the best observed K;
2. degrade at least one of SSIM or LPIPS relative to that best K; and
3. take more reconstruction time.

If metrics remain mixed, test one additional larger budget before stopping.

## Required Outputs

- Complete manifest and command/status/log files for every job.
- A per-scene table for every K.
- Mip-NeRF360, StaticHikes, and TUM RGB-D macro tables.
- A nine-scene macro quality/time table.
- Requested and realized C-iteration diagnostics.
- Selection and stopping-rule evidence in JSON and Markdown.
- Quality-versus-K and quality-versus-time plots.

The final report must identify the maximum-quality K and the recommended
quality-time K separately when they differ.
