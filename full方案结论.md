# full 方案 pass 结论

## 结论

- 决策：`full`（`enable_uncertainty_sampling + enable_residual_replay + enable_dynamic_suppression + rectify_colmap_cameras`）在当前实现下**直接 pass**。
- 原因：虽然已修复 CUDA 崩溃并可完整跑通，但关键场景渲染质量出现明显负提升，尤其 `TUM/long_office_household` 肉眼质量和坏帧占比显著恶化。
- 后续保留主线：`replay_only`。

## 对比证据（replay_only vs full）

> 说明：`replay_only` 指标来自 `results/innovation/replay_only/*/metadata.json`；`full` 指标来自本轮完整运行日志汇总。

### 1) StaticHikes/forest1

- replay_only: FPS `7.399`, PSNR `17.910`, SSIM `0.477`, LPIPS `0.438`, PSNR_std `1.940`, bad_16 `0.183`, bad_18 `0.425`
- full: FPS `5.596`, PSNR `17.700`, SSIM `0.489`, LPIPS `0.434`, PSNR_std `3.676`, bad_16 `0.317`, bad_18 `0.383`
- 解读：均值指标部分持平/小幅波动，但稳定性指标变差（`PSNR_std`、`bad_16` 上升），视觉观感不稳定，且速度下降。

### 2) TUM/long_office_household

- replay_only: FPS `29.809`, PSNR `20.331`, SSIM `0.772`, LPIPS `0.302`, PSNR_std `3.991`, bad_16 `0.184`, bad_18 `0.264`
- full: FPS `19.985`, PSNR `16.436`, SSIM `0.658`, LPIPS `0.449`, PSNR_std `3.292`, bad_16 `0.575`, bad_18 `0.701`
- 解读：核心画质指标全面退化（PSNR/SSIM 下降、LPIPS 升高），坏帧比例大幅上升，属于不可接受退化。

## 执行建议（已确认）

- 主线实验仅保留：`baseline` 与 `replay_only`。
- `full` 作为负结果消融记录，不再投入调参成本。
- 若后续重启该方向，必须改为单模块逐步叠加（一次只加一个模块）并先过可视化门槛再看均值指标。
