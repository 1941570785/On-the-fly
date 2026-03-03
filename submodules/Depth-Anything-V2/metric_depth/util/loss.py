# 损失函数实现
# 参考自：https://github.com/depth-anything/Depth-Anything-V2


import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        # 只在有效像素上计算
        valid_mask = valid_mask.detach()
        # 对数差异（尺度不变）
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        # SiLog 损失：方差项 - λ * 均值项
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss