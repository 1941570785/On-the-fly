# ASR-GS

ASR-GS is an adaptive on-the-fly 3D Gaussian reconstruction method for
ordered, unposed image streams. It extends
[On-the-fly NVS](https://github.com/graphdeco-inria/on-the-fly-nvs) with
three modules:

- **A - pose reliability review:** weak or failed PnP-RANSAC + MiniBA
  estimates receive a bounded secondary multi-hypothesis review. A
  replacement is committed only when geometric support or reprojection
  residual improves without an implausible motion increase.
- **B - response-guided Gaussian sampling:** photometric residual and
  spatial edge response redistribute the baseline Bernoulli sampling
  mass when the current projection has a coverage gap. A scene-level
  response guard stops interventions that repeatedly fail to improve
  rendering. The pre-clipping sampling mass is preserved and the final
  probability is explicitly clipped to `[0, 1]`.
- **C - bounded transactional refinement:** rendering response and
  representation coverage request at most **8** Gaussian-only
  iterations. Camera poses remain frozen, and a candidate update is
  committed only when current and recent reference views pass the loss
  guards; otherwise the Gaussian and optimizer states are rolled back.

Anchor-based active/stored scene maintenance and the baseline keyframe
policy are retained.

## Repository State

The paper configuration is immutable in
[`asr_gs/config.py`](asr_gs/config.py). Public method selection is limited
to:

```text
--method baseline
--method asr-gs
--ablate-a
--ablate-b
--ablate-c
```

Historical experiment profiles and internal version switches are not
part of this release. Every output stores the resolved configuration and
its SHA-256 fingerprint in `metadata.json`.

## Installation

The code was tested with Ubuntu 22.04, Python 3.12, PyTorch 2.7, and
CUDA 12.8.

```bash
git clone --recursive -b ASR-GS https://github.com/1941570785/On-the-fly.git
cd On-the-fly
conda create -n asr-gs python=3.12 -y
conda activate asr-gs
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install cupy-cuda12x
pip install -r requirements.official.txt
```

The pretrained feature and monocular-depth weights follow the upstream
On-the-fly NVS setup.

## Data

Each scene must contain ordered images under `images/`. COLMAP-format
poses and intrinsics can be placed under `sparse/0`; pose evaluation may
use a separate `sparse/GT` reconstruction. Numeric frame names are read
with natural sorting.

The nine-scene protocol uses:

| Dataset | Scenes | Test hold |
|---|---|---:|
| Mip-NeRF360 | bonsai, counter, garden | 8 |
| StaticHikes | forest1, forest2, university2 | 10 |
| TUM RGB-D | desk, xyz, long office | 30 |

## Reconstruction

ASR-GS is the default:

```bash
python train.py \
  -s /path/to/scene \
  -m /path/to/output \
  --test_hold 8 \
  --enable_reboot \
  --viewer_mode none
```

Run the unmodified algorithmic baseline from the same code and
environment:

```bash
python train.py \
  -s /path/to/scene \
  -m /path/to/baseline-output \
  --method baseline \
  --test_hold 8 \
  --enable_reboot \
  --viewer_mode none
```

## Nine-Scene Evaluation

The benchmark runner enforces the paper holdout settings and accepts at
most three distinct GPUs:

```bash
python tools/run_asr_gs_benchmark.py \
  --data-root /path/to/datasets \
  --output-root /path/to/results/asr-gs \
  --gpus 0 1 2 \
  --method asr-gs \
  --repeat 3 \
  --seed 0
```

For a component ablation, add exactly the module being removed:

```bash
python tools/run_asr_gs_benchmark.py \
  --data-root /path/to/datasets \
  --output-root /path/to/results/without-b \
  --gpus 0 \
  --method asr-gs \
  --ablate b
```

The runner writes:

- `run_manifest.json`: Git revision, working-tree state, full fixed
  configuration, and fingerprint;
- `scene_metrics.csv`: per-scene PSNR, SSIM, LPIPS, time, and A/B/C
  activation statistics;
- `dataset_macro_metrics.csv`: arithmetic means over the three scenes in
  each dataset;
- one `train.log`, `metadata.json`, and
  `pose_reliability_trace.json` per run.

## Viewer

For an optimized model:

```bash
python gaussianviewer.py local /path/to/output
```

For a remote GPU server, start the client locally and tunnel the port:

```bash
python gaussianviewer.py client --port 6009
ssh -L 6009:127.0.0.1:6009 USER@SERVER
```

Then run `python gaussianviewer.py server /path/to/output --port 6009`
on the server.

## Tests

The release contract can be checked without running a reconstruction:

```bash
python -m unittest discover -s tests -p "test_asr_gs_*.py" -v
```

The tests verify the fixed A policy, B probability bounds and baseline
bypass, C's `K=8` upper bound, public ablations, benchmark protocol, and
the three-GPU limit.

## Acknowledgments

This project is built on
[On-the-fly NVS](https://github.com/graphdeco-inria/on-the-fly-nvs).
Please retain its license and cite the original work:

```bibtex
@article{meuleman2025onthefly,
  title={On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images},
  author={Meuleman, Andreas and Shah, Ishaan and Lanvin, Alexandre and Kerbl, Bernhard and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={44},
  number={4},
  year={2025}
}
```

The ASR-GS citation will be added upon publication.
