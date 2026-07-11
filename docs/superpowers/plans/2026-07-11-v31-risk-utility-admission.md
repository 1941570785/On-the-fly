# V31 Risk-Utility Joint Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rendering-value-aware pose-risk admission policy to the fixed v31 pipeline and compare it with the main baseline and paper-table v31 using quality, APE, and RPE metrics.

**Architecture:** A pure policy module combines the existing post-pose risk event with a reduced-resolution render probe. Training invokes the probe only for risk candidates, isolates only low-value candidates, and gives retained high-value candidates a bounded pose-only photometric review before Gaussian growth. A separate evaluator aligns all three trajectories on one common frame set.

**Tech Stack:** Python 3.10/3.12, PyTorch, NumPy, unittest, existing CUDA Gaussian rasterizer.

---

### Task 1: Pure Risk-Utility Admission Policy

**Files:**
- Create: `scene/pose_risk_utility_admission.py`
- Create: `tests/test_pose_risk_utility_admission.py`

- [ ] **Step 1: Write failing policy tests**

Add tests that require: non-risk frames return `admit`; high-risk/high-utility frames return `review_admit`; high-risk/low-utility frames return `isolate_low_utility`; observe mode records the same scores without changing admission; test and bootstrap frames bypass the policy.

- [ ] **Step 2: Run the policy tests and verify RED**

Run:

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_pose_risk_utility_admission -v
```

Expected: import failure because `scene.pose_risk_utility_admission` does not exist.

- [ ] **Step 3: Implement the minimal pure policy**

Create a bounded utility score from coverage deficit, normalized residual-edge selectivity, and new-view score. Implement `PoseRiskUtilityAdmissionGate.evaluate(...)` with `off`, `observe_v1`, and `active_v1` modes, fixed scene-independent thresholds, event tracing, and summary output.

- [ ] **Step 4: Run the policy tests and verify GREEN**

Expected: all policy tests pass.

### Task 2: Reduced-Resolution Probe and Pose-Only Review

**Files:**
- Modify: `scene/scene_model.py`
- Create: `tests/test_pose_risk_utility_scene_integration.py`

- [ ] **Step 1: Write failing integration tests**

Require `SceneModel.probe_pose_risk_utility(...)` to render without Gaussian gradients and return coverage deficit plus residual selectivity. Require `SceneModel.review_pose_risk_keyframe(...)` to update only keyframe extrinsics, restore the starting pose after degradation or an excessive pose step, and cap iterations at two.

- [ ] **Step 2: Run the integration tests and verify RED**

Run:

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_pose_risk_utility_scene_integration -v
```

Expected: failures because both SceneModel methods are missing.

- [ ] **Step 3: Implement the probe and review methods**

The probe uses a fixed black background and quarter resolution. It derives support from `mainGaussID >= 0`, applies the incoming validity mask when present, and computes a robust normalized residual-edge selectivity. The review reuses the established Gaussian scene, takes at most two camera-only optimizer steps, checks supported photometric loss before and after, bounds rotation and translation changes, and restores the original pose on rejection.

- [ ] **Step 4: Run integration and existing pre-refine tests**

Run the new integration test plus `tests.test_pose_render_pre_refine` and expect all tests to pass.

### Task 3: Training and CLI Integration

**Files:**
- Modify: `args.py`
- Modify: `train.py`
- Modify: `tests/test_v31_pose_risk_a_integration.py`
- Create: `tests/test_v31_risk_utility_integration.py`

- [ ] **Step 1: Write failing CLI and source-path tests**

Require arguments for policy mode, utility threshold, probe scale, review iterations, and pose-step bounds. Require the training loop to evaluate pose risk first, probe only a risk candidate, restore match state on `isolate_low_utility`, and call pose review before `add_new_gaussians()` for `review_admit`.

- [ ] **Step 2: Run the tests and verify RED**

Expected: missing CLI options and training integration markers.

- [ ] **Step 3: Implement the minimal training path**

Keep `pose_initialization_risk_mode=observe_v1` as the risk estimator whenever the joint policy is active. Store estimated pose, GT pose, and source image identity in the trace. Preserve exact v31 behavior when the new policy is off. Do not change keyframe selection, texture-sampling configuration, extra-optimization configuration, or anchor logic.

- [ ] **Step 4: Run focused regression tests**

Run the A-module, joint-policy, texture-sampling, extra-optimization, keyframe-resolution, and v31 profile tests. Expected: all pass.

### Task 4: Three-Way Experiment and Pose Evaluation

**Files:**
- Create: `tools/run_v31_risk_utility_experiment.py`
- Create: `tools/compare_v31_risk_utility_results.py`
- Create: `tests/test_v31_risk_utility_runner.py`

- [ ] **Step 1: Write failing runner and evaluator tests**

Use synthetic metadata with differently padded image names. Require numeric-name normalization, one three-way common frame set, independent Sim(3) alignment to the same GT, and output fields for PSNR, SSIM, LPIPS, time, APE-t, APE-R, RPE-t, RPE-R, and common-frame count.

- [ ] **Step 2: Run tests and verify RED**

Expected: import failures because the runner and evaluator are missing.

- [ ] **Step 3: Implement runner and evaluator**

Fix scenes to `bonsai` and `forest1`, `test_hold=8`, v31 profile `baseline_render_lock_intra_frame_v31`, and one sequential process at a time. The evaluator reads the main baseline directories, the paper-table v31 directory, and the new output directory, then writes CSV, JSON, and Markdown summaries.

- [ ] **Step 4: Run runner tests and dry-run commands**

Expected: tests pass and dry-run commands vary only the joint-admission mode.

### Task 5: Calibration, Active Runs, and Final Audit

**Files:**
- Create: `docs/superpowers/specs/2026-07-11-v31-risk-utility-admission-results.md`

- [ ] **Step 1: Run observe-only calibration**

Use only CUDA device 7 and run scenes sequentially. Inspect one shared threshold against both utility distributions; do not tune per scene.

- [ ] **Step 2: Run active experiments**

Run `bonsai` and `forest1` sequentially on CUDA device 7. Preserve command provenance, logs, metadata, and admission traces.

- [ ] **Step 3: Generate the three-way report**

Report quality and pose metrics on common frame identities. Include result directories, keyframe counts, isolated/reviewed counts, and runtime.

- [ ] **Step 4: Verify and commit**

Run `py_compile`, focused unittests, the v31 profile semantic test, and `git diff --check`. Confirm no training process remains, write the results note, and commit the implementation without staging unrelated pre-existing untracked files.
