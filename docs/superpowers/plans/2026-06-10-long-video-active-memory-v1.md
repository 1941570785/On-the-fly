# Long-Video Active Memory V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TUM-oriented online frame-role controller that keeps low-parallax redundant frames as pose/tracking support instead of materializing them into the rendering representation.

**Architecture:** Extend the existing `DirectDensityController` v3 pose/representation decoupling path with a new mode, `pose_rep_active_memory_v1`. The mode reuses v3 thresholds and adds an online marginal representation value classifier, active-memory pressure debug fields, and trace output; it does not use offline PSNR/SSIM/LPIPS/APE/RPE during training.

**Tech Stack:** Python, existing `paper_aligned_policy` modules, `unittest`, existing `train.py` trace plumbing.

---

### Task 1: CLI and Mode Plumbing

**Files:**
- Modify: `args.py`
- Modify: `paper_aligned_policy/direct_density_control.py`
- Test: `tests/test_coupled_innovation_model.py`

- [ ] **Step 1: Write the failing CLI/mode test**

```python
def test_active_memory_v1_is_accepted_as_direct_density_mode(self):
    parser = _parser()
    args = parser.parse_args(
        [
            "--paper_aligned_direct_density_control",
            "pose_rep_active_memory_v1",
        ]
    )
    self.assertEqual(args.paper_aligned_direct_density_control, "pose_rep_active_memory_v1")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model
```

Expected: argparse rejects `pose_rep_active_memory_v1`.

- [ ] **Step 3: Add the mode to argparse and v3-family predicates**

Add `pose_rep_active_memory_v1` to `args.py` choices and to the v3-family mode sets in `DirectDensityController`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add args.py paper_aligned_policy/direct_density_control.py tests/test_coupled_innovation_model.py
git commit -m "long-video-active-memory-v1-mode"
```

### Task 2: Low-Value TUM Frame Role

**Files:**
- Modify: `paper_aligned_policy/direct_density_control.py`
- Test: `tests/test_coupled_innovation_model.py`

- [ ] **Step 1: Write failing behavior tests**

```python
def test_active_memory_v1_holds_low_parallax_redundant_tum_frame(self):
    controller = DirectDensityController(
        _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
    )
    decision = controller.decide(
        frame_id=900,
        runtime_action="direct_admit",
        baseline_should_add=True,
        is_test=False,
        is_bootstrap_phase=False,
        density_before=72.0,
        local_density_before=72.0,
        local_window_density=72.0,
        local_window_keyframes=72,
        local_window_gap_max=2.0,
        local_window_gap_after_if_hold=2.0,
        keyframe_growth_recent=28,
        baseline_relative_density=1.0,
        source_gap_to_last_keyframe=1,
        main_chain_gap_before=1.0,
        main_chain_gap_after_if_hold=2.0,
        anchor_changed=False,
        support_triggered=False,
        median_displacement=18.0,
        displacement_threshold=30.0,
        num_matches=2600,
        min_num_inliers=100,
        pose_inliers=1800,
        novelty_proxy=0.08,
        current_keyframe_count=650,
        semantic_scores={"R_t": 0.05, "V_t": 0.92, "Q_t": 0.96, "C_t": 0.96, "B_R_t": 0.95},
    )
    self.assertFalse(decision.finalize)
    self.assertEqual(decision.decision, "hold_low_representation_value")
    self.assertEqual(decision.debug["active_memory_frame_role"], "tracking_only")
```

```python
def test_active_memory_v1_keeps_high_novelty_or_gap_frame_as_representation(self):
    controller = DirectDensityController(
        _args(paper_aligned_direct_density_control="pose_rep_active_memory_v1")
    )
    decision = controller.decide(
        frame_id=900,
        runtime_action="direct_admit",
        baseline_should_add=True,
        is_test=False,
        is_bootstrap_phase=False,
        density_before=72.0,
        local_density_before=72.0,
        local_window_density=72.0,
        local_window_keyframes=72,
        local_window_gap_max=2.0,
        local_window_gap_after_if_hold=8.0,
        keyframe_growth_recent=28,
        baseline_relative_density=1.0,
        source_gap_to_last_keyframe=6,
        main_chain_gap_before=4.0,
        main_chain_gap_after_if_hold=8.0,
        anchor_changed=False,
        support_triggered=True,
        median_displacement=62.0,
        displacement_threshold=30.0,
        num_matches=1400,
        min_num_inliers=100,
        pose_inliers=800,
        novelty_proxy=0.78,
        current_keyframe_count=650,
        semantic_scores={"R_t": 0.18, "V_t": 0.88, "Q_t": 0.82, "C_t": 0.72, "B_R_t": 0.80},
    )
    self.assertTrue(decision.finalize)
    self.assertEqual(decision.debug["active_memory_frame_role"], "representation")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model
```

Expected: no active-memory debug fields and low-parallax frame is finalized.

- [ ] **Step 3: Implement minimal active-memory context**

Add an `is_pose_rep_active_memory_v1` property. Compute an active-memory score from low motion, high redundancy, high pose support, high semantic quality, high density, and low novelty. If this context is true, allow `value_hold_allowed` even outside v3's `long_stream_low_growth_context`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model
```

Expected: active-memory tests pass and existing v3 tests remain green.

- [ ] **Step 5: Commit**

```bash
git add paper_aligned_policy/direct_density_control.py tests/test_coupled_innovation_model.py
git commit -m "long-video-active-memory-v1-frame-role"
```

### Task 3: Trace and Experiment Runner

**Files:**
- Modify: `train.py`
- Create: `results/BRANCH_EXPERIMENTS_20260610/long-video-active-memory-v1_tum/run_tum_active_memory_v1.py`

- [ ] **Step 1: Write trace fields**

Add trace fields:

```python
"active_memory_context": bool(dbg.get("active_memory_context", False)),
"active_memory_frame_role": str(dbg.get("active_memory_frame_role", "")),
"active_memory_marginal_value": float(dbg.get("active_memory_marginal_value", 0.0)),
"active_memory_redundancy_pressure": float(dbg.get("active_memory_redundancy_pressure", 0.0)),
```

- [ ] **Step 2: Run compile check**

Run:

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m py_compile args.py train.py paper_aligned_policy/direct_density_control.py paper_aligned_policy/runtime_gate.py
```

Expected: exit code 0.

- [ ] **Step 3: Create TUM runner**

The runner executes `desk1`, `desk2`, and `long_office_household` with:

```bash
--risk_admission_mode on_the_fly_innovation_v1
--paper_aligned_recovery_commit_bridge true_source_commit
--paper_aligned_defer_recovery_support_bridge v1
--paper_aligned_recovery_commit_control off
--paper_aligned_direct_density_control pose_rep_active_memory_v1
--paper_aligned_direct_update_prev_desc_on_hold light
```

- [ ] **Step 4: Commit**

```bash
git add train.py results/BRANCH_EXPERIMENTS_20260610/long-video-active-memory-v1_tum/run_tum_active_memory_v1.py
git commit -m "long-video-active-memory-v1-tum-runner"
```

### Task 4: Verification and TUM Experiment

**Files:**
- Read: `results/BRANCH_EXPERIMENTS_20260610/long-video-active-memory-v1_tum/*`

- [ ] **Step 1: Run tests**

```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model tests.test_semantic_runtime_policy
```

Expected: all tests pass.

- [ ] **Step 2: Run TUM full experiment**

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/zxd/miniconda3/envs/otf/bin/python \
  results/BRANCH_EXPERIMENTS_20260610/long-video-active-memory-v1_tum/run_tum_active_memory_v1.py
```

Expected: three datasets complete with returncode 0.

- [ ] **Step 3: Compare with baseline/v3**

Check that materialized keyframes drop on TUM without catastrophic PSNR/SSIM/LPIPS loss, and that APE/RPE trends improve or at minimum do not worsen versus v3.

- [ ] **Step 4: Commit any report scripts**

```bash
git add -A
git commit -m "long-video-active-memory-v1-tum-results"
```
