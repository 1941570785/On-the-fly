# Copyright (C) 2023 - 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 高斯光栅化模块初始化
# 参考：https://github.com/graphdeco-inria/gaussian-splatting/blob/main/diff_gaussian_rasterization/__init__.py


from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    # 将 tuple 内的 tensor 复制到 CPU（用于调试/保存）
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    viewmatrix,
    raster_settings,
):
    # Python 包装：调用自定义 autograd Function
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        viewmatrix,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        viewmatrix,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        # 按 C++/CUDA 接口需要的顺序组织参数
        args = (
            raster_settings.bg, 
            means3D, 
            colors_precomp, 
            opacities, 
            scales, 
            rotations, 
            raster_settings.scale_modifier, 
            cov3Ds_precomp, 
            viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            raster_settings.image_height, 
            raster_settings.image_width, 
            dc, 
            sh, 
            raster_settings.sh_degree, 
            raster_settings.campos, 
            raster_settings.prefiltered, 
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        # 前向渲染：返回颜色、逆深度等
        num_rendered, num_buckets, color, invdepth, mainGaussID, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        # 保存反向所需中间量
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, viewmatrix)
        return color, invdepth, mainGaussID, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_invdepth, _, __):

        # Restore necessary values from context
        # 读取前向缓存用于反向求导
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        num_buckets = ctx.num_buckets
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, viewmatrix = ctx.saved_tensors

        # Restructure args as C++ method expects them
        # 按 C++/CUDA 接口需要的顺序组织反向参数
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                dc, 
                sh, 
                grad_out_invdepth,
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        # 调用 CUDA 反向计算梯度
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, grad_viewmatrix = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D, 
            grad_means2D, 
            grad_dc, 
            grad_sh, 
            grad_colors_precomp, 
            grad_opacities, 
            grad_scales, 
            grad_rotations, 
            grad_cov3Ds_precomp, 
            grad_viewmatrix,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, dc, shs, scales, rotations, viewmatrix):
        
        raster_settings = self.raster_settings

        # if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
        #     raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        # if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
        #     raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # if shs is None:
        #     shs = torch.Tensor([])
        # if colors_precomp is None:
        #     colors_precomp = torch.Tensor([])

        # if scales is None:
        #     scales = torch.Tensor([])
        # if rotations is None:
        #     rotations = torch.Tensor([])
        # if cov3D_precomp is None:
        #     cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        # 仅传 SH 或预计算颜色中的一种
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            torch.Tensor([]),
            opacities,
            scales, 
            rotations,
            torch.Tensor([]),
            viewmatrix,
            raster_settings
        )

def adamUpdate(params, grads, exp_avg, exp_avg_sq, visibility, lr, beta1, beta2, eps, N, M):
    # 稀疏可见性驱动的 Adam 更新
    _C.adamUpdate(params, grads, exp_avg, exp_avg_sq, visibility, lr, beta1, beta2, eps, N, M)

def adamUpdateBasic(params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, eps):
    # 基础 Adam 更新（无稀疏掩码）
    _C.adamUpdateBasic(params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, eps)