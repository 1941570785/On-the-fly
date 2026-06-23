# Pose-Only SSM Baseline Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate whether SSM/memory can improve online pose estimation while preserving the original baseline representation update set.

**Architecture:** Add a direct finalization mode, `pose_only_ssm_baseline_repr_v1`, that materializes only baseline/test/bootstrap representation frames. Non-baseline direct-admit frames may be estimated and registered as pose-only references, but they do not call `add_keyframe`, `add_new_gaussians`, or optimizer updates.

**Tech Stack:** Python, existing `PaperAlignedRuntimeGate`, `DirectDensityController`, unittest, on-the-fly-nvs training runner.

---

### Task 1: Policy Contract

**Files:**
- Create: `tests/test_pose_only_baseline_repr_policy.py`
- Modify: `paper_aligned_policy/direct_density_control.py`

- [x] Write a failing unittest proving non-baseline direct-admit frames return `finalize=False`.
- [x] Add `pose_only_ssm_baseline_repr_v1` mode and dispatch it before existing density/value logic.
- [x] Verify baseline/test/bootstrap frames still materialize.

### Task 2: Runtime Integration

**Files:**
- Modify: `args.py`
- Modify: `train.py`
- Modify: `paper_aligned_policy/runtime_gate.py`

- [x] Add the new mode to CLI choices.
- [x] Allow direct finalization to run even when recovery materialization is disabled for this mode.
- [x] Configure a bounded pose-only reference pool for this mode.

### Task 3: Experiment Runner

**Files:**
- Create: `tools/run_pose_only_ssm_baseline_repr_v1_all.py`

- [x] Add a 9-dataset runner using baseline-compatible test holds.
- [x] Set `--paper_aligned_recovery_commit_bridge off` and `--paper_aligned_defer_recovery_support_bridge off`.
- [x] Set `--paper_aligned_direct_density_control pose_only_ssm_baseline_repr_v1`.

### Task 4: Verification

**Commands:**
- [x] `/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_pose_only_baseline_repr_policy -v`
- [x] `/home/zxd/miniconda3/envs/otf/bin/python -m py_compile paper_aligned_policy/direct_density_control.py paper_aligned_policy/runtime_gate.py args.py train.py tools/run_pose_only_ssm_baseline_repr_v1_all.py`
- [x] `/home/zxd/miniconda3/envs/otf/bin/python -m unittest discover -s tests -p 'test_*.py'`
