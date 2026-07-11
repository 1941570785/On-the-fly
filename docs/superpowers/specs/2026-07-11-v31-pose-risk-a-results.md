# V31 Pose-Initialization Risk A-Module Results

## Protocol

- Branch: `codex/v31-pose-risk-a-20260711`
- GPU: one GPU, CUDA device 7, sequential execution
- Split: `test_hold=8`
- Scenes: `bonsai`, `forest1`
- Active configuration: absolute risk floor `0.10`, adaptive sigma `2.0`, warmup `8`, history `64`, cooldown `12`

## Raw Results

| Scene | Variant | PSNR | SSIM | LPIPS | Time (s) | R (deg) | t | Keyframes | Isolated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai | Control 1 | 25.4622 | 0.8564 | 0.2309 | 68.69 | 0.6831 | 0.03935 | 234 | 0 |
| bonsai | Control 2 | 25.4410 | 0.8595 | 0.2282 | 68.44 | 0.6932 | 0.03936 | 234 | 0 |
| bonsai | A + cooldown | 25.3846 | 0.8614 | 0.2274 | 68.24 | 0.6551 | 0.03724 | 233 | 1 |
| forest1 | Control 1 | 18.1555 | 0.5055 | 0.4084 | 119.06 | 4.5771 | 0.49506 | 369 | 0 |
| forest1 | Control 2 | 17.8577 | 0.4759 | 0.4356 | 115.98 | 4.2630 | 0.74607 | 369 | 0 |
| forest1 | A + cooldown | 17.7996 | 0.4766 | 0.4339 | 119.94 | 4.0065 | 0.71966 | 368 | 8 |

The no-cooldown active run isolated 1 `bonsai` frame and 21 `forest1` frames. It produced `25.3336/0.8588/0.2288` on `bonsai` and `18.1258/0.5010/0.4153` on `forest1` (PSNR/SSIM/LPIPS). The cooldown reduced the `forest1` isolation count from 21 to 8 but did not create a stable rendering gain.

## Finding

The A module has a measurable pose-side effect: the cooldown variant improves rotation error on both scenes, and also improves both pose errors on `bonsai`. It does not reliably transfer this benefit to rendering. `bonsai` gains SSIM and LPIPS but loses PSNR; `forest1` lies within the wide control variability for SSIM/LPIPS and loses PSNR. Runtime remains approximately unchanged.

The active A module should therefore remain an ablation/diagnostic branch rather than replacing the render-only v31 final model. The result supports the paper's central observation that better pose-side filtering is not sufficient for stable rendering improvement; the representation path still needs image-space response guidance.

## Result Directories

- Calibration/control/observe: `results/BRANCH_EXPERIMENTS_20260703/v31_pose_risk_a_calibration_hold8_bonsai_forest1_20260711`
- Active without cooldown: `results/BRANCH_EXPERIMENTS_20260703/v31_pose_risk_a_active_hold8_bonsai_forest1_20260711`
- Active with cooldown: `results/BRANCH_EXPERIMENTS_20260703/v31_pose_risk_a_cooldown12_hold8_bonsai_forest1_20260711`
- Repeated control: `results/BRANCH_EXPERIMENTS_20260703/v31_pose_risk_a_control_repeat2_hold8_20260711`
