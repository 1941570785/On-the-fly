# Latest-A C Extra-Round Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a complete nine-scene sensitivity sweep that selects the best C extra-round budget for the final pose-verification-A model.

**Architecture:** Reuse the existing tested round-ablation runner and evaluator, then adapt only their experiment contract to the latest A branch, all nine scenes, and at most six GPUs. Training code and the production A/B/C implementations remain unchanged; the runner controls C through existing command-line arguments.

**Tech Stack:** Python, PyTorch, existing `otf` Conda environment, CSV/JSON manifests, Matplotlib, Git worktree.

---

### Task 1: Import the proven experiment tooling

**Files:**
- Create: `tools/run_extra_optimization_round_ablation.py`
- Create: `tools/evaluate_extra_optimization_round_ablation.py`
- Create: `tests/test_extra_optimization_round_runner.py`
- Create: `tests/test_extra_optimization_round_evaluation.py`

- [ ] Cherry-pick the existing runner, evaluator, and exact budget-mapping commits `390a0d1`, `6a5d9ab`, and `e46e17e`.
- [ ] Inspect the resulting diff and confirm that no production A/B/C training code changed.

### Task 2: Lock the runner to the latest A and six-GPU protocol

**Files:**
- Modify: `tests/test_extra_optimization_round_runner.py`
- Modify: `tools/run_extra_optimization_round_ablation.py`

- [ ] Add failing tests asserting `verify_v1`, the shared pose-verification parameters, all nine default scenes, and acceptance of six unique GPUs.
- [ ] Run `/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_extra_optimization_round_runner -v` and verify the new tests fail for the expected old-A/three-GPU behavior.
- [ ] Replace the old A constants with `A_SHARED_ARGS` plus `--pose_initialization_risk_mode verify_v1`, set the sensitivity scene set to all nine scenes, and raise only the runner GPU limit to six.
- [ ] Re-run the runner tests and the evaluator tests until both pass.

### Task 3: Verify commands and completion contracts

**Files:**
- Verify: `tools/run_extra_optimization_round_ablation.py`
- Verify: `tools/evaluate_extra_optimization_round_ablation.py`

- [ ] Run a dry run for all nine scenes and initial budgets and inspect every generated command for the latest A mode, V31 profile, one GPU, seed zero, and the requested K mapping.
- [ ] Run one complete smoke job and verify return code zero, readable metadata, requested budget, realized C iterations, and all four quality/time metrics.
- [ ] Commit the experiment-tool changes after tests and smoke validation pass.

### Task 4: Run the initial complete sweep

**Files:**
- Create under: `results/BRANCH_EXPERIMENTS_20260703/latest_a_c_round_sensitivity_<timestamp>/`

- [ ] Query GPU utilization and choose up to six idle RTX 4090 cards.
- [ ] Run all 63 initial `(scene, K)` jobs for `K={0,2,4,8,12,16,20}` with seed zero.
- [ ] Retry failed jobs individually with `--skip-existing`; never delete successful sibling runs.
- [ ] Validate that the manifest contains exactly 63 successful, non-dry-run rows and that every model has readable metadata.

### Task 5: Extend past the optimum

**Files:**
- Update under the same result root.

- [ ] Evaluate the initial nine-scene macro curve using the preregistered quality and Pareto rules.
- [ ] If the optimum is on or near the right boundary, run complete nine-scene jobs for K=24 and K=30.
- [ ] Continue with ten-round increments only when two larger settings have not yet met the stopping rule.
- [ ] Validate every extension batch before using it in the curve.

### Task 6: Produce the final report

**Files:**
- Create: `<result-root>/analysis_final/results.md`
- Create: `<result-root>/analysis_final/per_scene.csv`
- Create: `<result-root>/analysis_final/dataset_macro.csv`
- Create: `<result-root>/analysis_final/nine_scene_macro.csv`
- Create: `<result-root>/analysis_final/selection.json`

- [ ] Aggregate per-scene, per-dataset, and nine-scene metrics without importing prior-model results.
- [ ] Record maximum-quality K, recommended quality-time K, realized iterations, and exact stopping evidence.
- [ ] Generate quality-versus-K and quality-versus-time plots.
- [ ] Re-read every manifest and metadata file and verify no incomplete job contributes to the final tables.
