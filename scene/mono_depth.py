# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 单目深度估计器，用于估计图像的深度
# 参考自：https://github.com/depth-anything/Depth-Anything-V2


import torch
import os
import sys
import urllib.request
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from utils import sample

sys.path.append("submodules/Depth-Anything-V2")
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
from depth_anything_v2.dpt import DepthAnythingV2

size = 518
encoder = "vitl"


class MonoDepthInternal(torch.nn.Module):
    def __init__(self):
        super(MonoDepthInternal, self).__init__()
        model_path = f"models/depth_anything_v2_{encoder}.pth"
        if not os.path.exists(model_path):
            print(f"Downloading Depth-Anything-V2 model for {encoder}, may take a few minutes...")
            model_sizes = {
                "vits": "Small",
                "vitb": "Base",
                "vitl": "Large",
                "vitg": "Giant",
            }
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_sizes[encoder]}/resolve/main/depth_anything_v2_{encoder}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model = model.to("cuda").half().eval()
        self.sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )
        self.sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, image: torch.Tensor):
        img = torch.nn.functional.interpolate(
            image[None].half(), (size, size), mode="bilinear", align_corners=True
        )
        depth = self.model(img)[None]
        t, s = get_t_s(depth)
        depth = (depth - t) / s

        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        edges = torch.cat((grad_x, grad_y), dim=0)

        edges_sq_norm = (edges**2).sum(0, keepdim=True)
        var = 0.2
        confidence = torch.exp(-edges_sq_norm / var)
        return depth.float(), confidence.float()


def get_t_s(d):
    t = d.median()
    s = (d - t).abs().median()
    return t, s


def align_samples(tri_idepth: torch.Tensor, mono_idepth: torch.Tensor):
    t_tri, s_tri = get_t_s(tri_idepth)
    t_mono, s_mono = get_t_s(mono_idepth)
    scale = s_tri / s_mono
    offset = t_tri - t_mono * scale
    return mono_idepth * scale + offset, scale, offset


def align_depth(
    mono_depth_map: torch.Tensor, desc_kpts: DescribedKeypoints, width: int, height: int
):
    """
    【场景表示模块】对齐单目深度到三角化深度
    
    通过计算最优的缩放和偏移，使单目深度与三角化深度对齐。
    使用鲁棒估计（去除异常值后重新对齐）确保对齐质量。
    
    Args:
        mono_depth_map: 单目深度图 [1, H, W]
        desc_kpts: 描述的关键点（包含三角化深度）
        width: 图像宽度
        height: 图像高度
    
    Returns:
        对齐后的单目深度图
    """
    valid_matches = desc_kpts.has_pt3d
    num_matches = int(valid_matches.sum().item())
    if num_matches == 0:
        return mono_depth_map, {
            "num_depth_points": 0,
            "depth_align_scale": 1.0,
            "depth_align_offset": 0.0,
            "depth_align_error_mean": float("nan"),
            "depth_align_error_median": float("nan"),
            "depth_align_inlier_ratio": 0.0,
        }

    mono_idepth = sample(
        mono_depth_map,
        desc_kpts.kpts[valid_matches].view(1, 1, -1, 2),
        width,
        height,
    )[0, 0, 0]
    tri_idepth = 1 / desc_kpts.depth[valid_matches]

    mono_idepth_aligned, scale, offset = align_samples(tri_idepth, mono_idepth)
    err = (mono_idepth_aligned - tri_idepth).abs()
    median_err = err.median()
    valid = err < 5 * median_err
    valid_count = int(valid.sum().item())
    if valid_count >= 2:
        mono_idepth_aligned, scale, offset = align_samples(
            tri_idepth[valid], mono_idepth[valid]
        )
        err = (mono_idepth_aligned - tri_idepth[valid]).abs()
    mono_depth_map_aligned = mono_depth_map * scale + offset

    return mono_depth_map_aligned, {
        "num_depth_points": num_matches,
        "depth_align_scale": float(scale.item()),
        "depth_align_offset": float(offset.item()),
        "depth_align_error_mean": float(err.mean().item()),
        "depth_align_error_median": float(err.median().item()),
        "depth_align_inlier_ratio": float(valid.float().mean().item()),
    }


class MonoDepthEstimator:
    """
    【场景表示模块】单目深度估计器
    
    使用Depth-Anything-V2模型（ViT-Large）估计单目深度。
    同时计算深度置信度（基于深度梯度的边缘检测）。
    
    深度估计用于：
    1. 初始化高斯点的深度
    2. 深度损失（与渲染深度对齐）
    3. 深度对齐（与三角化深度对齐）
    """
    @torch.no_grad()
    def __init__(self, width: int, height: int):
        """
        【场景表示模块】初始化单目深度估计器
        
        Args:
            width: 图像宽度
            height: 图像高度
        """
        self.width = width
        self.height = height
        model = MonoDepthInternal()

        dummy = torch.zeros(3, height, width).cuda()
        self.model = torch.cuda.make_graphed_callables(model, [dummy])

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth, conf = self.model(image)
        return depth.clone(), conf.clone()