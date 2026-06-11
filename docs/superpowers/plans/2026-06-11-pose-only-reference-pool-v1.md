# Pose-Only Reference Pool V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let active-memory tracking-only frames improve later pose estimation without participating in Gaussian rendering or keyframe representation growth.

**Architecture:** Store bounded lightweight pose-only reference objects in `PaperAlignedRuntimeGate`. Each reference contains only pose, source metadata, and a descriptor copy with verified 2D-3D support inherited from the successful pose initialization path. `train.py` appends at most two selected pose-only references to the PnP/MiniBA reference list, after normal `SceneModel.get_prev_keyframes()` returns, so representation and rendering code remain untouched.

**Tech Stack:** Existing Python runtime, PyTorch tensors already carried by `DescribedKeypoints`, current unittest suite, existing trace JSON contract.

---

### Task 1: Branch and Documentation

**Files:**
- Create: `docs/superpowers/plans/2026-06-11-pose-only-reference-pool-v1.md`

- [ ] **Step 1: Create branch**

Run:
```bash
git switch long-video-active-memory-v1
git switch -c pose-only-reference-pool-v1
```

Expected: branch `pose-only-reference-pool-v1` is active.

- [ ] **Step 2: Commit this plan**

Run:
```bash
git add docs/superpowers/plans/2026-06-11-pose-only-reference-pool-v1.md
git commit -m pose-only-reference-pool-v1-plan
```

Expected: plan is preserved before code changes.

### Task 2: Runtime Pool Behavior

**Files:**
- Modify: `paper_aligned_policy/runtime_gate.py`
- Test: `tests/test_coupled_innovation_model.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:
```python
gate.register_pose_only_reference(...)
selected = gate.select_pose_only_references(...)
```

must register only active-memory tracking-only frames, evict expired entries, cap the pool size, and return at most two references sorted by online match support.

- [ ] **Step 2: Run tests and observe RED**

Run:
```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model.CoupledInnovationModelTests.test_pose_only_reference_pool_registers_and_selects_tracking_only_frames
```

Expected: failure because the runtime gate has no pose-only reference pool API.

- [ ] **Step 3: Implement minimal pool API**

Add a lightweight `PoseOnlyReference` dataclass or namespace in `runtime_gate.py`, plus:
```python
register_pose_only_reference(...)
select_pose_only_references(...)
pose_only_reference_pool_summary()
```

The pool uses fixed constants: max size 64, TTL 360 frames, min 80 verified 3D points, max two selected references per query.

- [ ] **Step 4: Verify GREEN**

Run:
```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model
```

Expected: all coupled innovation tests pass.

### Task 3: Pose Support Extraction

**Files:**
- Modify: `poses/pose_initializer.py`
- Test: `tests/test_coupled_innovation_model.py`

- [ ] **Step 1: Write failing test**

Add a test or focused assertion that after a successful incremental pose initialization, `last_incremental_pose_support` exposes current-frame keypoint indices, inherited 3D points, and confidences.

- [ ] **Step 2: Implement support capture**

In `initialize_incremental()`, after PnP/MiniBA succeeds, store detached tensors:
```python
self.last_incremental_pose_support = {
    "match_indices": match_indices.detach().clone(),
    "pts3d": xyz.detach().clone(),
    "pts_conf": confs.detach().clone(),
}
```

Reset this field at the start of each call and leave baseline behavior unchanged.

### Task 4: Training Loop Integration

**Files:**
- Modify: `train.py`
- Modify: `tools/run_tum_active_memory_v1.py` or create `tools/run_tum_pose_only_reference_pool_v1.py`

- [ ] **Step 1: Select pose-only references before PnP**

After:
```python
prev_keyframes = scene_model.get_prev_keyframes(...)
```

call:
```python
pose_only_refs = runtime_gate.select_pose_only_references(frameID, desc_kpts, scene_model.matcher)
prev_keyframes_for_pose = prev_keyframes + pose_only_refs
```

Use `prev_keyframes_for_pose` only for `pose_initializer.initialize_incremental()` and trace.

- [ ] **Step 2: Register held tracking-only frames**

When `direct_keyframe_finalized` is false and `held_bridge` is true, copy the current `desc_kpts`, inject support tensors from `pose_initializer.last_incremental_pose_support`, and register it in the pool with the direct density debug fields.

- [ ] **Step 3: Add trace stats**

Trace per-frame selected pose-only reference count, pool size, registered count, selected count, PnP usage count, and MiniBA usage count.

### Task 5: Verification and TUM Experiment

**Files:**
- Create: `tools/run_tum_pose_only_reference_pool_v1.py`

- [ ] **Step 1: Run verification**

Run:
```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model tests.test_semantic_runtime_policy
/home/zxd/miniconda3/envs/otf/bin/python -m py_compile args.py train.py poses/pose_initializer.py paper_aligned_policy/runtime_gate.py paper_aligned_policy/direct_density_control.py
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Commit**

Run:
```bash
git add -A
git commit -m pose-only-reference-pool-v1
```

- [ ] **Step 3: Run TUM**

Run:
```bash
CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/zxd/miniconda3/envs/otf/bin/python tools/run_tum_pose_only_reference_pool_v1.py
```

Expected outputs under `results/BRANCH_EXPERIMENTS_20260611/pose-only-reference-pool-v1_tum/`.
