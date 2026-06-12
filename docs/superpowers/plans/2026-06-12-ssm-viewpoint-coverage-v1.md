# SSM Viewpoint Coverage V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lightweight per-frame SSM diagnostics for viewpoint change, match/support coverage, and anchor health without changing keyframe, pose, anchor, or optimization decisions.

**Architecture:** Introduce a small policy-side helper that computes scalar diagnostics from data already available during online training. `train.py` will call it after successful incremental pose initialization and copy the output into the existing direct density trace payload. `PaperAlignedRuntimeGate` will store a separate `viewpoint_coverage_events` list so analysis scripts can read the new signal directly.

**Tech Stack:** Python, existing `torch` tensors, existing `PaperAlignedRuntimeGate` JSON trace, existing `unittest` suite.

---

### Task 1: Runtime Trace Contract

**Files:**
- Modify: `paper_aligned_policy/runtime_gate.py`
- Test: `tests/test_coupled_innovation_model.py`

- [ ] **Step 1: Write the failing trace storage test**

```python
def test_runtime_gate_flushes_viewpoint_coverage_events(self):
    with tempfile.TemporaryDirectory() as td:
        trace_path = Path(td) / "trace.json"
        gate = PaperAlignedRuntimeGate(
            _args(paper_aligned_contract_trace_path=str(trace_path))
        )

        gate.append_viewpoint_coverage_event(
            {
                "frame_id": 42,
                "viewpoint_rotation_deg_to_last_keyframe": 37.5,
                "inlier_grid_coverage": 0.5,
                "anchor_health_score": 0.75,
            }
        )
        gate.flush_trace()

        payload = json.loads(trace_path.read_text())
        self.assertEqual(len(payload["viewpoint_coverage_events"]), 1)
        self.assertEqual(payload["viewpoint_coverage_events"][0]["frame_id"], 42)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest \
  tests.test_coupled_innovation_model.CoupledInnovationModelTests.test_runtime_gate_flushes_viewpoint_coverage_events
```

Expected: `AttributeError` for missing `append_viewpoint_coverage_event` or missing trace key.

- [ ] **Step 3: Add event storage and flush key**

Add `self.viewpoint_coverage_events = []`, `append_viewpoint_coverage_event()`, and include `"viewpoint_coverage_events"` in the trace payload.

- [ ] **Step 4: Re-run the test and verify it passes**

Expected: `OK`.

### Task 2: Lightweight Diagnostic Helper

**Files:**
- Create: `paper_aligned_policy/viewpoint_coverage.py`
- Test: `tests/test_coupled_innovation_model.py`

- [ ] **Step 1: Write failing tests for rotation and coverage metrics**

Tests should cover:
- identity rotation returns `0.0` degrees;
- 90 degree relative rotation is near `90.0`;
- keypoints covering two out of sixteen grid cells return `0.125`;
- concentrated inlier support has lower entropy than spread support.

- [ ] **Step 2: Run the tests and verify they fail**

Expected: import failure for `paper_aligned_policy.viewpoint_coverage`.

- [ ] **Step 3: Implement minimal helper functions**

Implement:
- `rotation_degrees_between(a, b)`;
- `grid_coverage(kpts, width, height, grid_size=4)`;
- `grid_entropy(kpts, width, height, grid_size=4)`;
- `build_viewpoint_coverage_event(...)`.

Use only existing tensors and Python math.

- [ ] **Step 4: Re-run tests and verify they pass**

Expected: `OK`.

### Task 3: Train Integration Without Decision Changes

**Files:**
- Modify: `train.py`
- Test: `tests/test_coupled_innovation_model.py`

- [ ] **Step 1: Write a failing integration-shape test**

The test should call `build_viewpoint_coverage_event()` with synthetic current pose, previous representation pose, active anchor ids, and inlier keypoints, then assert all fields expected by `train.py` exist:

```python
for key in (
    "viewpoint_rotation_deg_to_last_keyframe",
    "viewpoint_rotation_deg_to_active_anchor",
    "inlier_grid_coverage",
    "inlier_grid_entropy",
    "support_concentration",
    "anchor_health_score",
    "new_view_event_score",
):
    self.assertIn(key, event)
```

- [ ] **Step 2: Run the test and verify it fails**

Expected: missing helper or missing fields.

- [ ] **Step 3: Add train call site**

After `pose_initializer.initialize_incremental()` succeeds and before finalization decision is recorded, compute the event and:
- append it with `runtime_gate.append_viewpoint_coverage_event(event)`;
- merge scalar fields into `v2_payload`;
- do not read the event inside `DirectDensityController`.

- [ ] **Step 4: Re-run tests, compile, and diff check**

Run:
```bash
/home/zxd/miniconda3/envs/otf/bin/python -m unittest tests.test_coupled_innovation_model tests.test_semantic_runtime_policy
/home/zxd/miniconda3/envs/otf/bin/python -m py_compile train.py paper_aligned_policy/runtime_gate.py paper_aligned_policy/viewpoint_coverage.py
git diff --check
```

Expected: all pass.

### Task 4: SSM-Only Experiment

**Files:**
- Create: `tools/run_ssm_viewpoint_coverage_v1_all.py`

- [ ] **Step 1: Copy the existing all-dataset experiment runner pattern**

Use the same command options as prior innovation/v1.1 experiments so behavior is comparable.

- [ ] **Step 2: Add trace statistics extraction**

For each dataset, record:
- existing metrics: keyframes, PSNR, SSIM, LPIPS, R, t, train time;
- viewpoint metrics: mean/p90 rotation to last keyframe, low coverage count, high support concentration count, high new-view event count.

- [ ] **Step 3: Run all nine datasets**

Expected: reconstruction metrics should match v1.1 within normal nondeterministic noise because no decision logic changed.

- [ ] **Step 4: Commit and report**

Commit with:
```bash
git add -A
git commit -m ssm-viewpoint-coverage-v1
```

Report whether the new SSM signals cluster around forest1, university2, and long failure regions.
