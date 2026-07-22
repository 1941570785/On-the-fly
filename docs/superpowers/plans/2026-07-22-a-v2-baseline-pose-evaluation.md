# A-v2 Baseline Pose Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently validated, test-frame-safe A-v2 pose verification and run a reproducible single-GPU, nine-scene baseline-protocol ablation.

**Architecture:** Preserve the existing A gate and verifier, adding a new `verify_v2` mode. Reuse recorded 2D-3D correspondence reference IDs to separate solve and validation evidence, preserve exact fallback, and reuse the existing paper Table 5 evaluator for final metrics.

**Tech Stack:** Python 3, PyTorch, NumPy, `unittest`, existing CUDA P4P RANSAC/MiniBA, existing experiment runners.

---

### Task 1: Add the A-v2 eligibility contract

**Files:**
- Modify: `scene/pose_initialization_risk.py`
- Modify: `args.py`
- Test: `tests/test_pose_initialization_risk.py`
- Test: `tests/test_pose_verification_integration.py`

- [ ] Add a failing test proving `verify_v1` still bypasses test frames while `verify_v2` can trigger on a selected, non-bootstrap test frame.
- [ ] Run the two test modules and confirm the failure is caused by the unsupported mode.
- [ ] Add `verify_v2` to the mode sets and CLI choices; compute eligibility as `baseline_selected and not is_bootstrap and (not is_test or mode == "verify_v2")`.
- [ ] Preserve all `verify_v1`, isolation, observation, and bootstrap behavior.
- [ ] Run the tests and commit.

### Task 2: Split solve and validation evidence

**Files:**
- Modify: `poses/pose_verification.py`
- Modify: `poses/pose_initializer.py`
- Test: `tests/test_pose_verification.py`
- Test: `tests/test_pose_verification_integration.py`

- [ ] Add failing tests for deterministic reference-level holdout, no overlap, minimum support, single-reference spatial fallback, and exact fallback when validation support is insufficient.
- [ ] Run the tests and confirm the missing splitter/verifier behavior fails.
- [ ] Extend incremental candidate/support recording with `corr_ref_ids`.
- [ ] Implement a pure splitter returning solve and validation masks plus diagnostics.
- [ ] In `verify_v2`, run PnP/MiniBA only on the solve mask and evaluate pre/post residual distributions only on the validation mask.
- [ ] Reuse existing correction magnitude, support ratio, tail, and exact-fallback rules.
- [ ] Run focused and regression tests and commit.

### Task 3: Integrate A-v2 and stage traces

**Files:**
- Modify: `train.py`
- Test: `tests/test_pose_verification_integration.py`

- [ ] Add a failing source-integration test proving both verify modes invoke the verifier and only v2 enables independent validation.
- [ ] Pass the mode into the verifier and record `initial_estimated_Rt`, `post_a_Rt`, test status, split strategy, solve references, and validation references.
- [ ] Verify test keyframes retain the existing no-Gaussian-update behavior and are not added to pose-reference memory.
- [ ] Run integration tests and commit.

### Task 4: Add the single-GPU paired runner

**Files:**
- Create: `tools/run_a_v2_baseline_pose_experiment.py`
- Test: `tests/test_run_a_v2_baseline_pose_experiment.py`

- [ ] Add failing tests for the three variants, identical non-A arguments, nine-scene ordering, five repeats, and rejection of multiple GPU IDs.
- [ ] Implement a resumable sequential runner using the current final K16 profile.
- [ ] Store commands, commit, GPU inventory, status, logs, and manifests atomically.
- [ ] Run runner tests and a dry run and commit.

### Task 5: Add final and stage evaluators

**Files:**
- Create: `tools/evaluate_a_v2_baseline_pose_experiment.py`
- Test: `tests/test_evaluate_a_v2_baseline_pose_experiment.py`

- [ ] Add failing synthetic tests for exact Table 5 field units, common-alignment stage deltas, success/retention rates, P95, catastrophic failures, and repeat mean/sample standard deviation.
- [ ] Reuse `/data2/zxd/3D_Reconstruction/comparison_tools/paper_table5_pose_eval.py` for final metrics without changing its formulas.
- [ ] Implement stage extraction from risk traces and final trajectories using one fixed anchor-derived Sim(3).
- [ ] Write per-scene, dataset-macro, repeat-summary CSV/JSON/Markdown outputs.
- [ ] Run evaluator tests and commit.

### Task 6: Verify and execute

**Files:**
- No production-file changes expected.

- [ ] Run all focused A, runner, evaluator, and paper-protocol unit tests.
- [ ] Run one-scene `a_v2` smoke test on one GPU and evaluate it.
- [ ] Inspect trace invariants: test frames may be verified but never update Gaussians or become references.
- [ ] Launch five sequential repeats of all three variants over nine scenes on one physical GPU.
- [ ] Evaluate each completed repeat and then aggregate all repeats.
- [ ] Report paths and results, separating paper-protocol final metrics from A-stage diagnostics.
