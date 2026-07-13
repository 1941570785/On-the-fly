# V31+A Final Component Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a reproducible nine-scene leave-one-component-out ablation for the final V31+A model.

**Architecture:** Add a V31-only post-profile component-ablation switch, a serial official-protocol runner, and a general ablation evaluator that reuses the existing official pose metric implementation. Production behavior remains unchanged when the ablation switch is `none`.

**Tech Stack:** Python, argparse, unittest, NumPy, existing On-the-fly NVS training and official pose evaluation utilities.

---

### Task 1: Define component-removal contracts

**Files:**
- Modify: `args.py`
- Modify: `paper_aligned_policy/config.py`
- Test: `tests/test_v31_a_final_ablation.py`

- [ ] Add failing tests proving response sampling and extra optimization can be disabled independently after resolving V31.
- [ ] Add a constrained CLI option whose default preserves the final model exactly.
- [ ] Apply the option only to the V31 profile and reject accidental use with another profile.
- [ ] Run focused configuration tests.

### Task 2: Add the official serial ablation runner

**Files:**
- Create: `tools/run_v31_a_final_ablation.py`
- Test: `tests/test_v31_a_final_ablation.py`

- [ ] Add failing command-contract tests for all four variants.
- [ ] Reuse official scene definitions, A parameters, provenance, manifest, and one-GPU guard.
- [ ] Preserve official holdouts, reboot, viewer-off, and scene-major serial execution.
- [ ] Verify the dry run contains exactly 36 unique jobs.

### Task 3: Add rendering, pose, and trigger aggregation

**Files:**
- Create: `tools/evaluate_v31_a_final_ablation.py`
- Test: `tests/test_v31_a_final_ablation.py`

- [ ] Add tests for trigger extraction and five-method output contracts.
- [ ] Reuse official APE/RPE alignment functions and reference data.
- [ ] Export per-scene rendering, pose, trigger, and macro summary CSV files.

### Task 4: Execute and verify the benchmark

- [ ] Run focused tests and a 36-job dry run.
- [ ] Run all jobs serially on one GPU and stop on the first failure.
- [ ] Link the frozen official baseline into the comparison root.
- [ ] Run the evaluator and verify expected row counts and complete artifacts.
- [ ] Report per-scene and macro rendering, time, pose, and trigger comparisons.
