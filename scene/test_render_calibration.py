from __future__ import annotations

from typing import Any


TEST_RENDER_CALIBRATION_OFF = "off"
TEST_RENDER_CALIBRATION_DIAG_AFFINE = "diag_affine_v1"


def calibrate_test_render(
    render: Any,
    target: Any,
    *,
    mode: str | None = TEST_RENDER_CALIBRATION_OFF,
    eps: float = 1e-6,
    min_scale: float = 0.5,
    max_scale: float = 1.6,
    max_bias: float = 0.20,
    mask: Any | None = None,
):
    mode_name = str(mode or TEST_RENDER_CALIBRATION_OFF)
    if mode_name != TEST_RENDER_CALIBRATION_DIAG_AFFINE:
        return render, {"applied": False, "mode": mode_name}

    x_all = render.reshape(render.shape[0], -1).float()
    y_all = target.reshape(target.shape[0], -1).float()
    sample_mask = None
    if mask is not None:
        sample_mask = mask
        if hasattr(sample_mask, "detach"):
            sample_mask = sample_mask.detach()
        if sample_mask.ndim == render.ndim:
            sample_mask = sample_mask.any(dim=0)
        sample_mask = sample_mask.reshape(-1).bool()
        if int(sample_mask.sum().detach().cpu().item()) <= 1:
            return render, {
                "applied": True,
                "accepted": False,
                "mode": mode_name,
                "reason": "insufficient_mask_support",
            }
    x = x_all[:, sample_mask] if sample_mask is not None else x_all
    y = y_all[:, sample_mask] if sample_mask is not None else y_all
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    var = (x_centered * x_centered).mean(dim=1, keepdim=True)
    cov = (x_centered * y_centered).mean(dim=1, keepdim=True)
    scale = (cov / (var + float(eps))).clamp(float(min_scale), float(max_scale))
    bias = (y_mean - scale * x_mean).clamp(-float(max_bias), float(max_bias))
    calibrated = (scale[:, :, None] * render.float() + bias[:, :, None]).clamp(0.0, 1.0)
    calibrated_all = calibrated.reshape(render.shape[0], -1)
    if sample_mask is not None:
        calibrated_eval = calibrated_all[:, sample_mask]
        before_eval = x
        target_eval = y
    else:
        calibrated_eval = calibrated_all
        before_eval = x_all
        target_eval = y_all
    before_mse = ((before_eval - target_eval) ** 2).mean()
    after_mse = ((calibrated_eval - target_eval) ** 2).mean()
    if not bool(after_mse < before_mse):
        return render, {
            "applied": True,
            "accepted": False,
            "mode": mode_name,
            "before_mse": float(before_mse.detach().cpu().item()),
            "after_mse": float(after_mse.detach().cpu().item()),
        }
    return calibrated.to(dtype=render.dtype), {
        "applied": True,
        "accepted": True,
        "mode": mode_name,
        "scale_mean": float(scale.mean().detach().cpu().item()),
        "bias_abs_mean": float(bias.abs().mean().detach().cpu().item()),
        "before_mse": float(before_mse.detach().cpu().item()),
        "after_mse": float(after_mse.detach().cpu().item()),
    }
