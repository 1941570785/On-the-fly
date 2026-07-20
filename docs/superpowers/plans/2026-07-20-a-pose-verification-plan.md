# Pose-Only A Implementation and Ablation Plan

## Frozen Baseline

- Base commit: `40e7f0b`
- Worktree: `/data2/zxd/3D_Reconstruction/On_the_fly_pose_verification_a_20260720`
- Branch: `codex/a-pose-verification-20260720`
- Constraint: `scene/pose_render_texture_sampling.py` and
  `scene/pose_render_extra_optimization.py` remain unchanged.

## Task 1: Lock Existing Behavior

- Run focused pose-risk, v31 integration, final ablation, and official-runner
  tests.
- Record unrelated failures from the broad legacy suite without changing them.
- Snapshot the B/C source hashes at `40e7f0b`.

## Task 2: Add Pure Pose-Verification Decisions With TDD

Create failing tests for:

- reprojection residual projection and invalid-depth handling;
- median/MAD cleaning with spatial-cell preservation;
- minimum-support restoration;
- accepted refinement with lower median and bounded motion;
- rejection on p90 degradation, support collapse, or implausible motion;
- exact fallback to the original pose.

Implement a small `poses/pose_verification.py` module containing pure tensor and
scalar decision helpers. Keep CUDA-dependent solver calls outside the pure
module.

## Task 3: Reuse PnP-RANSAC and MiniBA for a Second Pose Candidate

Extend the pose initializer's existing support cache with the corresponding 2D
coordinates. Add a method that:

- exits without work unless `verify_v1` produces a risk trigger;
- cleans the cached correspondences;
- reruns PnP-RANSAC on cleaned full-resolution correspondences;
- starts from the PnP candidate and reruns MiniBA once;
- computes pre/post diagnostics and safe-acceptance decision;
- returns the accepted pose or the original pose exactly;
- records solver time and detailed telemetry.

## Task 4: Integrate Before Gaussian Construction

Add `verify_v1` to the A-mode CLI. Invoke pose verification immediately after
risk evaluation and before any keyframe/Gaussian construction. Update the risk
event and keyframe info with verification telemetry. Recompute only pose-derived
viewpoint telemetry after an accepted correction; do not call B or C from A.

## Task 5: Add the Three-Way Ablation Runner

Build a runner with `V31-Control`, `A-Observe`, and `A-Full`. Freeze all non-A
arguments and support up to three explicitly assigned GPUs. Add a summarizer that
joins official pose metrics, rendering metrics, risk traces, and A telemetry per
scene and as dataset macro averages.

## Task 6: Verify Implementation Boundaries

- Run focused unit/integration tests.
- Compile changed Python files.
- Confirm B/C source hashes and `git diff 40e7f0b` show no B/C changes.
- Smoke-test all three variants on a short scene prefix.
- Inspect traces for exact observe-mode pose passthrough and verify-mode fallback.

## Task 7: Official Nine-Scene Ablation

Run the three variants with the official protocol over:

- Mip-NeRF360: bonsai, counter, garden;
- StaticHikes: forest1, forest2, university2;
- TUM RGB-D: desk, xyz, long_office.

Use at most three GPUs concurrently. Report every scene separately, then the
three dataset arithmetic means and nine-scene macro mean. Compare `A-Full` to
both `V31-Control` and `A-Observe`, with pose evidence leading and rendering/time
as side-effect checks.
