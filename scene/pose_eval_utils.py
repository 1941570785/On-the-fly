from __future__ import annotations

import torch


def select_pose_eval_pairs(
    rts: torch.Tensor,
    gt_rts: torch.Tensor,
    gt_rts_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair estimated poses with available GT poses using the per-keyframe GT mask."""
    n_masked_poses = min(int(gt_rts_mask.shape[0]), int(rts.shape[0]))
    if n_masked_poses <= 0 or int(gt_rts.shape[0]) <= 0:
        return rts[:0], gt_rts[:0]

    valid_mask = gt_rts_mask[:n_masked_poses].to(device=rts.device, dtype=torch.bool)
    selected_rts = rts[:n_masked_poses][valid_mask]
    pair_count = min(int(selected_rts.shape[0]), int(gt_rts.shape[0]))
    return selected_rts[:pair_count], gt_rts[:pair_count]
