# V31+A Official Pose Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run On-the-fly NVS, V31, and V31+A with the official 8/10/30 protocol on one GPU and produce reproducible nine-scene rendering, pose, and A-module diagnostic reports.

**Architecture:** A deterministic experiment runner builds method-specific commands, enforces one visible GPU, records provenance, and executes scenes serially. A separate evaluator loads estimated keyframe poses from metadata and external reference poses from COLMAP or TUM `sparse/GT`, then computes common-frame Sim(3)-aligned pose metrics and report artifacts.

**Tech Stack:** Python 3.10/3.12, NumPy, existing On-the-fly NVS training code, COLMAP model reader, CSV/JSON/Markdown, direct `unittest` execution.

---

### Task 1: External Reference Pose Evaluator

**Files:**
- Create: `tools/evaluate_official_pose_benchmark.py`
- Create: `tests/test_evaluate_official_pose_benchmark.py`

- [ ] **Step 1: Write failing tests for reference loading and common-frame evaluation**

```python
class OfficialPoseEvaluatorTest(unittest.TestCase):
    def test_canonical_frame_id_matches_padded_names(self):
        self.assertEqual(canonical_frame_id("000031.png"), "31")
        self.assertEqual(canonical_frame_id("31.png"), "31")

    def test_tum_reference_loader_uses_only_valid_rows(self):
        references = load_tum_reference(self.gt_dir)
        self.assertEqual(set(references), {"2", "3"})

    def test_three_way_evaluation_uses_one_common_frame_set(self):
        report = evaluate_scene(reference, methods, rpe_delta=1)
        self.assertEqual(report["three_way_common_frames"], 3)
        self.assertEqual(report["methods"]["v31_a"]["ape_count"], 3)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python tests/test_evaluate_official_pose_benchmark.py`

Expected: import failure because `tools/evaluate_official_pose_benchmark.py` does not exist.

- [ ] **Step 3: Implement canonical loading and pose evaluation**

Implement these public functions using existing helpers from
`tools.standard_pose_eval_report`:

```python
def canonical_frame_id(name: str) -> str: ...
def load_colmap_reference(images_path: Path) -> dict[str, np.ndarray]: ...
def load_tum_reference(gt_dir: Path) -> dict[str, np.ndarray]: ...
def load_metadata_trajectory(model_dir: Path) -> dict[str, np.ndarray]: ...
def evaluate_scene(reference, methods, *, rpe_delta: int = 1) -> dict: ...
```

For each method, fit an independent Umeyama Sim(3) on the exact three-way
intersection, then compute ATE translation RMSE, absolute rotation mean,
RPE translation RMSE, and RPE rotation mean. Also compute V31/V31+A pairwise
metrics and method coverage against all valid reference poses.

- [ ] **Step 4: Add report and risk-diagnostic tests**

```python
def test_quarantined_frame_percentiles_use_external_reference(self):
    diagnostics = evaluate_risk_trace(trace_path, reference, aligned_trajectory)
    self.assertEqual(diagnostics["quarantined_count"], 1)
    self.assertGreaterEqual(diagnostics["quarantined"][0]["rotation_percentile"], 0)
```

- [ ] **Step 5: Implement CSV, JSON, and Markdown outputs**

Write `pose_three_way.csv`, `pose_v31_vs_a.csv`, `rendering_summary.csv`,
`a_module_diagnostics.csv`, `official_pose_report.json`, and
`official_pose_report.md`. Refuse to aggregate when any requested method or
scene is missing.

- [ ] **Step 6: Run evaluator tests and existing metric tests**

Run:

```bash
python tests/test_evaluate_official_pose_benchmark.py
python tests/test_standard_pose_eval_report.py
```

Expected: all tests pass with exit code 0.

- [ ] **Step 7: Commit evaluator**

```bash
git add tools/evaluate_official_pose_benchmark.py tests/test_evaluate_official_pose_benchmark.py
git commit -m "feat: add official reference pose evaluator"
```

### Task 2: Official-Protocol Serial Runner

**Files:**
- Create: `tools/run_v31_a_official_pose_benchmark.py`
- Create: `tests/test_run_v31_a_official_pose_benchmark.py`

- [ ] **Step 1: Write failing command-construction tests**

```python
class OfficialRunnerTest(unittest.TestCase):
    def test_official_holdouts_are_dataset_specific(self):
        self.assertEqual(SCENES["bonsai"].test_hold, 8)
        self.assertEqual(SCENES["forest1"].test_hold, 10)
        self.assertEqual(SCENES["desk"].test_hold, 30)

    def test_a_command_preserves_v31_and_enables_quarantine(self):
        command = build_command(spec, variant="v31_a", python=Path("python"))
        self.assertIn("baseline_render_lock_intra_frame_v31", command)
        self.assertIn("pose_quarantine_v1", command)
        self.assertEqual(command[command.index("--pose_risk_utility_review_iterations") + 1], "0")

    def test_gpu_guard_rejects_multiple_visible_devices(self):
        with self.assertRaises(ValueError):
            validate_single_gpu("0,1")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python tests/test_run_v31_a_official_pose_benchmark.py`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement scene and variant specifications**

Use the nine padded source paths and exact holds. Baseline commands run in
`/data2/zxd/3D_Reconstruction/On_the_fly_main_true_baseline_20260703`.
V31 and V31+A commands run in the current worktree. V31 disables both risk
modes; V31+A uses the fixed parameters from the approved design.

- [ ] **Step 4: Implement serial execution and resumability**

The runner must:

```python
validate_single_gpu(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
for variant in requested_variants:
    for scene in requested_scenes:
        subprocess.run(command, cwd=variant_repo, check=False)
        update_manifest_after_each_run()
```

Support `--dry_run`, `--skip_existing`, `--only_scene`, `--only_variant`, and
`--repeat_index`. Save `command.json`, `train.log`, return code, repository
commit, holdout, source path, and completeness status for every run.

- [ ] **Step 5: Run runner tests and dry-run all 27 jobs**

Run:

```bash
python tests/test_run_v31_a_official_pose_benchmark.py
CUDA_VISIBLE_DEVICES=7 python tools/run_v31_a_official_pose_benchmark.py --dry_run
```

Expected: 27 unique commands, holds 8/10/30 by group, no duplicate model paths,
and no process launch.

- [ ] **Step 6: Commit runner**

```bash
git add tools/run_v31_a_official_pose_benchmark.py tests/test_run_v31_a_official_pose_benchmark.py
git commit -m "feat: add official protocol benchmark runner"
```

### Task 3: Preflight And Full Nine-Scene Execution

**Files:**
- Create under results: `official_pose_full9_<timestamp>/...`

- [ ] **Step 1: Verify source/reference completeness**

Run evaluator preflight and require nine image streams, six complete COLMAP
reference models, three TUM `sparse/GT` directories, and at least three valid
reference poses per scene.

- [ ] **Step 2: Verify one-GPU availability**

Run: `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`

Select one idle device and export only that device. Do not launch concurrent
training processes.

- [ ] **Step 3: Run all methods serially**

```bash
CUDA_VISIBLE_DEVICES=7 /home/zxd/miniconda3/envs/otf/bin/python \
  tools/run_v31_a_official_pose_benchmark.py \
  --output_root results/BRANCH_EXPERIMENTS_20260703/official_pose_full9_<timestamp>
```

Expected: baseline, V31, and V31+A complete for all nine scenes with 27/27
successful return codes.

- [ ] **Step 4: Validate artifacts before evaluation**

Require every model to contain readable `metadata.json`, finite 4x4 keyframe
poses, non-empty quality metrics, and a successful return code. Require A runs
to contain both risk trace files.

### Task 4: Reports And Paired Stability Repeats

**Files:**
- Create under the run root: `evaluation/*`
- Create under results: `official_pose_repeats_<timestamp>/...`

- [ ] **Step 1: Generate full-nine reports**

```bash
/home/zxd/miniconda3/envs/otf/bin/python \
  tools/evaluate_official_pose_benchmark.py \
  --run_root results/BRANCH_EXPERIMENTS_20260703/official_pose_full9_<timestamp> \
  --output_dir results/BRANCH_EXPERIMENTS_20260703/official_pose_full9_<timestamp>/evaluation
```

- [ ] **Step 2: Run three paired V31/V31+A repeats**

For repeat indices 1 through 3, run only `forest1` and `long_office`, keeping
the official holds and executing V31 followed by V31+A on the same GPU.

- [ ] **Step 3: Generate repeat mean, standard deviation, and paired deltas**

Report each pose/rendering metric as mean plus/minus standard deviation and the
per-repeat V31+A minus V31 delta. Never replace full-nine raw rows with repeat
averages.

- [ ] **Step 4: Fresh verification**

Run all new direct tests, `py_compile` on both tools, manifest validation,
finite-value checks, and a second evaluator invocation into a temporary output
directory. Compare the two generated JSON reports byte-for-byte.

- [ ] **Step 5: Commit code and experiment documentation**

```bash
git add docs/superpowers/specs/2026-07-12-v31-a-official-pose-evaluation-results.md
git commit -m "docs: record official pose benchmark results"
```
