"""
【DINOv2模块】DINOv2 Vision Transformer实现

这是Meta（Facebook）开发的DINOv2 Vision Transformer模型的实现。
DINOv2是一个自监督学习的视觉Transformer模型，主要用于提取图像特征。

主要特性：
1. 支持多种模型规模（Small, Base, Large, Giant2）
2. 支持register tokens（额外的分类token，用于增强表示）
3. 支持位置编码插值（用于处理不同分辨率的图像）
4. 支持分块处理（用于大模型的内存优化）
5. 支持mask token（用于自监督学习）

参考：
- https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
- https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0
"""

# DINOv2 Vision Transformer实现
# 参考自：https://github.com/depth-anything/Depth-Anything-V2


from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from .dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


# ========== 工具函数 ==========

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    """
    【工具函数】递归地对模块及其子模块应用函数
    
    类似于`apply`函数，但会在调用时传递模块的名称路径。
    用于初始化权重等需要对每个模块执行操作的任务。
    
    Args:
        fn: 要应用的函数，接收(module, name)参数
        module: 要处理的PyTorch模块
        name: 当前模块的名称路径
        depth_first: 如果为True，先处理子模块再处理当前模块（深度优先）
        include_root: 是否对根模块调用函数
        
    Returns:
        nn.Module: 处理后的模块
    """
    # 如果为广度优先且包含根模块，先处理根模块
    if not depth_first and include_root:
        fn(module=module, name=name)
    # 递归处理所有子模块
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    # 如果为深度优先且包含根模块，最后处理根模块
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    """
    【模块】Transformer块分块包装器
    
    将多个Transformer块包装在一起，用于分块处理。
    这是为了支持大模型的内存优化（FSDP - Fully Sharded Data Parallel）。
    
    继承自nn.ModuleList，前向传播时按顺序执行所有块。
    """
    def forward(self, x):
        """
        【前向传播】按顺序执行所有Transformer块
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 经过所有块处理后的输出
        """
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """
    【核心模块】DINOv2 Vision Transformer模型
    
    这是DINOv2的核心实现，基于Vision Transformer架构，包含以下组件：
    1. Patch Embedding：将图像分割成patches并嵌入
    2. Position Embedding：位置编码（支持插值）
    3. Class Token：分类token
    4. Register Tokens：额外的可学习token（DINOv2的创新）
    5. Transformer Blocks：多层Transformer编码器
    6. Layer Norm：最终的归一化层
    
    支持的特性：
    - 多种FFN层类型（MLP、SwiGLU等）
    - 随机深度（Stochastic Depth）
    - 层缩放（Layer Scale）
    - 分块处理（用于大模型）
    - 位置编码插值（处理不同分辨率）
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        【初始化】创建DINOv2 Vision Transformer模型
        
        Args:
            img_size: 输入图像尺寸（可以是int或tuple）
            patch_size: patch大小（图像被分割成patch_size×patch_size的块）
            in_chans: 输入通道数（RGB图像为3）
            embed_dim: 嵌入维度（每个patch的向量维度）
            depth: Transformer的深度（块的数量）
            num_heads: 多头注意力的头数
            mlp_ratio: MLP隐藏层维度与嵌入维度的比值
            qkv_bias: 是否在QKV投影中使用偏置
            ffn_bias: 是否在FFN中使用偏置
            proj_bias: 是否在注意力输出投影中使用偏置
            drop_path_rate: 随机深度（Stochastic Depth）的丢弃率
            drop_path_uniform: 是否对所有块使用统一的丢弃率
            init_values: 层缩放（Layer Scale）的初始值（None或0表示不使用）
            embed_layer: patch嵌入层的类
            act_layer: MLP激活层
            block_fn: Transformer块类
            ffn_layer: FFN层类型（"mlp", "swiglu", "swiglufused"或"identity"）
            block_chunks: 将块序列分成多少个chunk（用于FSDP内存优化）
            num_register_tokens: register token的数量（DINOv2的创新）
            interpolate_antialias: 插值位置编码时是否使用抗锯齿
            interpolate_offset: 插值时的偏移量（用于避免浮点误差）
        """
        super().__init__()
        # 归一化层（LayerNorm，epsilon=1e-6）
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # ========== 基本参数存储 ==========
        # 【模型配置】存储模型的基本配置参数
        self.num_features = self.embed_dim = embed_dim  # 嵌入维度（用于与其他模型兼容）
        self.num_tokens = 1  # class token的数量（固定为1）
        self.n_blocks = depth  # Transformer块的数量
        self.num_heads = num_heads  # 注意力头数
        self.patch_size = patch_size  # patch大小
        self.num_register_tokens = num_register_tokens  # register token数量
        self.interpolate_antialias = interpolate_antialias  # 插值抗锯齿标志
        self.interpolate_offset = interpolate_offset  # 插值偏移量

        # ========== Patch Embedding ==========
        # 【图像嵌入】将图像分割成patches并转换为嵌入向量
        # 输入：[B, C, H, W] -> 输出：[B, num_patches, embed_dim]
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # patch的总数量

        # ========== Class Token ==========
        # 【分类Token】可学习的分类token，用于聚合全局特征
        # 形状：[1, 1, embed_dim]，会在前向传播时扩展到batch大小
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # ========== Position Embedding ==========
        # 【位置编码】可学习的位置编码，用于为每个patch提供位置信息
        # 形状：[1, num_patches + num_tokens, embed_dim]
        # num_patches + num_tokens：包括所有patch tokens和class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        
        # ========== Register Tokens ==========
        # 【Register Tokens】DINOv2的创新：额外的可学习token，用于增强特征表示
        # 这些token不用于分类，但参与注意力计算，可以学习到有用的特征
        # 如果num_register_tokens > 0，会插入到class token和patch tokens之间
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        # ========== 随机深度（Stochastic Depth） ==========
        # 【正则化】为每个Transformer块设置不同的drop path率
        # 如果uniform=True，所有块使用相同的drop path率
        # 如果uniform=False，使用线性递增的drop path率（浅层较小，深层较大）
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth  # 统一的drop path率
        else:
            # 线性递增：从0到drop_path_rate
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # ========== FFN层类型选择 ==========
        # 【前馈网络】根据配置选择FFN层的类型
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp  # 标准MLP
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused  # SwiGLU激活（更高效的FFN）
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")
            # 恒等映射（用于测试或特殊场景）
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError

        # ========== Transformer Blocks ==========
        # 【Transformer编码器】创建多层Transformer块
        blocks_list = [
            block_fn(
                dim=embed_dim,  # 嵌入维度
                num_heads=num_heads,  # 注意力头数
                mlp_ratio=mlp_ratio,  # MLP比例
                qkv_bias=qkv_bias,  # QKV偏置
                proj_bias=proj_bias,  # 投影偏置
                ffn_bias=ffn_bias,  # FFN偏置
                drop_path=dpr[i],  # 当前块的drop path率
                norm_layer=norm_layer,  # 归一化层
                act_layer=act_layer,  # 激活层
                ffn_layer=ffn_layer,  # FFN层类型
                init_values=init_values,  # 层缩放初始值
            )
            for i in range(depth)
        ]
        
        # ========== 块分块处理 ==========
        # 【内存优化】如果block_chunks > 0，将块分成多个chunk
        # 这样可以用于FSDP（Fully Sharded Data Parallel）等内存优化策略
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks  # 每个chunk的块数
            for i in range(0, depth, chunksize):
                # 为了保持块索引一致，在chunk前面添加Identity层
                # 例如：chunk0=[blocks 0-5], chunk1=[Identity*6, blocks 6-11]
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            # 不使用分块，直接使用所有块
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        # ========== 最终归一化层 ==========
        # 【输出归一化】在Transformer块之后应用LayerNorm
        self.norm = norm_layer(embed_dim)
        
        # ========== 分类头 ==========
        # 【分类头】默认使用Identity（特征提取模式）
        # 如果需要分类，可以替换为Linear层
        self.head = nn.Identity()

        # ========== Mask Token ==========
        # 【掩码Token】用于自监督学习的mask token
        # 当某些patches被mask时，用这个token替换
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # ========== 权重初始化 ==========
        # 【初始化】初始化模型权重
        self.init_weights()

    def init_weights(self):
        """
        【初始化】初始化模型权重
        
        对位置编码、class token、register tokens和所有线性层进行初始化。
        """
        # 位置编码：截断正态分布初始化（std=0.02）
        trunc_normal_(self.pos_embed, std=0.02)
        # Class token：正态分布初始化（std=1e-6，很小的初始值）
        nn.init.normal_(self.cls_token, std=1e-6)
        # Register tokens：正态分布初始化
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        # 对所有子模块应用timm风格的权重初始化
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        """
        【位置编码插值】将位置编码插值到不同的图像分辨率
        
        当输入图像的分辨率与训练时不同时，需要对位置编码进行插值。
        这是Vision Transformer处理不同分辨率图像的关键技术。
        
        Args:
            x: 输入token序列 [B, num_tokens + num_patches, embed_dim]
            w: 输入图像的宽度（像素）
            h: 输入图像的高度（像素）
            
        Returns:
            torch.Tensor: 插值后的位置编码 [1, num_tokens + num_patches, embed_dim]
        """
        previous_dtype = x.dtype  # 保存原始数据类型
        npatch = x.shape[1] - 1  # 当前patch数量（排除class token）
        N = self.pos_embed.shape[1] - 1  # 原始位置编码的patch数量
        
        # ========== 快速路径：无需插值 ==========
        # 【优化】如果patch数量和图像尺寸都匹配，直接返回原始位置编码
        if npatch == N and w == h:
            return self.pos_embed
        
        # ========== 分离class token和patch token的位置编码 ==========
        pos_embed = self.pos_embed.float()  # 转换为float以支持插值
        class_pos_embed = pos_embed[:, 0]  # class token的位置编码 [1, embed_dim]
        patch_pos_embed = pos_embed[:, 1:]  # patch tokens的位置编码 [1, N, embed_dim]
        dim = x.shape[-1]  # 嵌入维度
        
        # ========== 计算目标patch网格尺寸 ==========
        # 【坐标计算】计算目标图像的patch网格大小
        w0 = w // self.patch_size  # 宽度方向的patch数
        h0 = h // self.patch_size  # 高度方向的patch数
        
        # 添加偏移量以避免浮点误差（DINOv2的特殊处理）
        # 参考：https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        
        # ========== 计算插值比例 ==========
        # 【插值参数】计算从原始patch网格到目标patch网格的缩放比例
        sqrt_N = math.sqrt(N)  # 原始patch网格的边长（假设为正方形）
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N  # 水平和垂直缩放比例
        
        # ========== 执行插值 ==========
        # 【双三次插值】将位置编码从[sqrt_N, sqrt_N]插值到[w0, h0]
        # 1. 重塑为2D网格：[1, N, dim] -> [1, sqrt_N, sqrt_N, dim]
        # 2. 转置为CHW格式：[1, sqrt_N, sqrt_N, dim] -> [1, dim, sqrt_N, sqrt_N]
        # 3. 双三次插值（使用scale_factor或antialias）
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),  # 使用缩放因子
            mode="bicubic",  # 双三次插值（保持平滑）
            antialias=self.interpolate_antialias  # 是否使用抗锯齿
        )
        
        # ========== 验证和重塑 ==========
        # 【后处理】验证插值后的尺寸正确，然后重塑回1D序列
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        # 转置回HWC格式并重塑为1D：[1, dim, h0, w0] -> [1, h0*w0, dim]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        # ========== 合并位置编码 ==========
        # 【拼接】将class token和插值后的patch tokens位置编码拼接
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """
        【Token准备】将输入图像转换为token序列并添加位置编码
        
        执行以下步骤：
        1. Patch Embedding：将图像转换为patch tokens
        2. Mask处理：如果提供了masks，将对应位置替换为mask token
        3. 添加Class Token：在序列开头添加class token
        4. 添加位置编码：通过插值添加位置信息
        5. 添加Register Tokens：如果有register tokens，插入到class token和patch tokens之间
        
        Args:
            x: 输入图像 [B, C, H, W]
            masks: 可选，掩码 [B, num_patches]，True表示该patch被mask
            
        Returns:
            torch.Tensor: 准备好的token序列 [B, 1 + num_register_tokens + num_patches, embed_dim]
                顺序：class_token, register_tokens..., patch_tokens...
        """
        B, nc, w, h = x.shape  # Batch大小、通道数、宽度、高度
        
        # ========== Patch Embedding ==========
        # 【图像嵌入】将图像分割成patches并转换为嵌入向量
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # PatchEmbedding 输出 token 序列
        x = self.patch_embed(x)
        
        # ========== Mask处理 ==========
        # 【自监督学习】如果提供了masks，将masked的patches替换为mask token
        # 这是DINO等自监督方法的核心：mask部分patches，让模型学习恢复
        if masks is not None:
            # masks.unsqueeze(-1): [B, num_patches] -> [B, num_patches, 1]
            # mask_token: [1, embed_dim] -> 广播到 [B, num_patches, embed_dim]
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        # ========== 添加Class Token ==========
        # 【分类Token】在序列开头添加class token
        # [B, num_patches, embed_dim] -> [B, 1 + num_patches, embed_dim]
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # ========== 添加位置编码 ==========
        # 【位置信息】通过插值添加位置编码（根据当前图像分辨率）
        x = x + self.interpolate_pos_encoding(x, w, h)

        # ========== 添加Register Tokens ==========
        # 【Register Tokens】如果有register tokens，插入到class token和patch tokens之间
        # 最终序列：[class_token, register_tokens..., patch_tokens...]
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],  # class token
                    self.register_tokens.expand(x.shape[0], -1, -1),  # register tokens
                    x[:, 1:],  # patch tokens
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        """
        【前向传播】处理多个输入图像（批量特征提取）
        
        Args:
            x_list: 图像列表，每个元素 [C, H, W]
            masks_list: 掩码列表，每个元素 [num_patches]
            
        Returns:
            list: 特征字典列表，每个字典包含：
                - x_norm_clstoken: 归一化后的class token
                - x_norm_regtokens: 归一化后的register tokens
                - x_norm_patchtokens: 归一化后的patch tokens
                - x_prenorm: 归一化前的特征
                - masks: 掩码
        """
        # 为每个图像准备tokens
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        # 通过所有Transformer块
        for blk in self.blocks:
            x = blk(x)

        # 对每个输出应用归一化并提取不同部分
        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)  # 应用LayerNorm
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],  # class token [B, embed_dim]
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],  # register tokens [B, num_reg, embed_dim]
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],  # patch tokens [B, num_patches, embed_dim]
                    "x_prenorm": x,  # 归一化前的特征
                    "masks": masks,  # 掩码
                }
            )
        return output

    def forward_features(self, x, masks=None):
        """
        【前向传播】提取图像特征
        
        主要的特征提取函数，处理单个图像或图像列表。
        
        Args:
            x: 输入图像 [B, C, H, W] 或图像列表
            masks: 可选的掩码 [B, num_patches] 或掩码列表
            
        Returns:
            dict: 特征字典，包含：
                - x_norm_clstoken: 归一化后的class token [B, embed_dim]
                - x_norm_regtokens: 归一化后的register tokens [B, num_reg, embed_dim]
                - x_norm_patchtokens: 归一化后的patch tokens [B, num_patches, embed_dim]
                - x_prenorm: 归一化前的特征 [B, num_tokens, embed_dim]
                - masks: 掩码 [B, num_patches]
        """
        # 如果是列表，调用列表处理函数
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        # ========== 准备Tokens ==========
        # 【预处理】将图像转换为token序列并添加位置编码
        x = self.prepare_tokens_with_masks(x, masks)

        # ========== Transformer编码 ==========
        # 【特征提取】通过所有Transformer块提取特征
        # 顺序通过 Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # ========== 归一化和分割 ==========
        # 【后处理】应用LayerNorm并分割不同部分
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],  # class token（用于全局特征）
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],  # register tokens
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],  # patch tokens（用于密集特征）
            "x_prenorm": x,  # 归一化前的完整特征
            "masks": masks,  # 掩码
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        """
        【中间层提取】获取中间层的特征（未分块版本）
        
        用于提取Transformer中间层的特征，常用于特征可视化或多尺度特征。
        
        Args:
            x: 输入图像 [B, C, H, W]
            n: 要提取的层数（int：最后n层，list：指定的层索引）
            
        Returns:
            list: 中间层特征列表，每个元素 [B, num_tokens, embed_dim]
        """
        x = self.prepare_tokens_with_masks(x)
        # 如果n是int，取最后n层；如果是list，取指定的层
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        
        # 逐层前向传播，收集指定层的输出
        # 遍历 blocks 并收集中间层
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        """
        【中间层提取】获取中间层的特征（分块版本）
        
        与未分块版本类似，但处理分块的blocks结构。
        
        Args:
            x: 输入图像 [B, C, H, W]
            n: 要提取的层数（int：最后n层，list：指定的层索引）
            
        Returns:
            list: 中间层特征列表
        """
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        
        # 遍历每个chunk，跳过Identity层（通过切片block_chunk[i:]）
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # 跳过前面的Identity层
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        【中间层提取】获取中间层的patch token特征
        
        这是用于特征提取的主要接口，提取指定层的patch token特征。
        常用于下游任务（如密集预测、特征匹配等）。
        
        Args:
            x: 输入图像 [B, C, H, W]
            n: 要提取的层数（int：最后n层，list：指定的层索引）
            reshape: 是否将输出重塑为空间格式 [B, C, H_patch, W_patch]
            return_class_token: 是否同时返回class token
            norm: 是否对输出应用LayerNorm
            
        Returns:
            tuple: 
                - 如果return_class_token=False: (patch_tokens, ...) 每个元素 [B, num_patches, embed_dim] 或 [B, C, H_p, W_p]
                - 如果return_class_token=True: ((patch_tokens, class_token), ...)
        """
        # 根据是否分块选择相应的函数
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        
        # 应用归一化（如果需要）
        if norm:
            outputs = [self.norm(out) for out in outputs]
        
        # ========== 提取Class Token ==========
        # 【特征分割】分离class token和patch tokens
        class_tokens = [out[:, 0] for out in outputs]  # 提取class token
        # 跳过class token和register tokens，只取patch tokens
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        
        # ========== 重塑为空间格式 ==========
        # 【格式转换】如果需要，将1D序列重塑为2D空间格式
        if reshape:
            B, _, w, h = x.shape
            # [B, num_patches, embed_dim] -> [B, H_patch, W_patch, embed_dim] -> [B, embed_dim, H_patch, W_patch]
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        
        # ========== 返回结果 ==========
        if return_class_token:
            # 返回 (patch_tokens, class_token) 的元组对
            return tuple(zip(outputs, class_tokens))
        # 只返回patch tokens
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        """
        【前向传播】模型的主要前向传播函数
        
        Args:
            *args: 输入参数（通常是图像）
            is_training: 是否为训练模式
            **kwargs: 其他关键字参数（如masks）
            
        Returns:
            如果is_training=True: 返回完整的特征字典
            如果is_training=False: 返回class token的head输出（用于分类）
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            # 训练模式：返回完整特征（用于自监督学习）
            return ret
        else:
            # 推理模式：通过head处理class token（用于分类）
            return self.head(ret["x_norm_clstoken"])


# ========== 权重初始化函数 ==========

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """
    【权重初始化】ViT风格的权重初始化（timm实现）
    
    用于与timm库保持一致的可复现性。
    对线性层使用截断正态分布初始化权重，偏置初始化为0。
    
    Args:
        module: 要初始化的PyTorch模块
        name: 模块名称（可选，用于调试）
    """
    if isinstance(module, nn.Linear):
        # 权重：截断正态分布（std=0.02）
        trunc_normal_(module.weight, std=0.02)
        # 偏置：初始化为0
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ========== 模型工厂函数 ==========

def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    """
    【模型工厂】创建ViT-Small模型
    
    模型配置：
    - embed_dim: 384
    - depth: 12层
    - num_heads: 6头
    - mlp_ratio: 4
    
    Args:
        patch_size: patch大小
        num_register_tokens: register token数量
        **kwargs: 其他参数
        
    Returns:
        DinoVisionTransformer: ViT-Small模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),  # 使用内存高效的注意力
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    """
    【模型工厂】创建ViT-Base模型
    
    模型配置：
    - embed_dim: 768
    - depth: 12层
    - num_heads: 12头
    - mlp_ratio: 4
    
    Args:
        patch_size: patch大小
        num_register_tokens: register token数量
        **kwargs: 其他参数
        
    Returns:
        DinoVisionTransformer: ViT-Base模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    """
    【模型工厂】创建ViT-Large模型
    
    模型配置：
    - embed_dim: 1024
    - depth: 24层
    - num_heads: 16头
    - mlp_ratio: 4
    
    Args:
        patch_size: patch大小
        num_register_tokens: register token数量
        **kwargs: 其他参数
        
    Returns:
        DinoVisionTransformer: ViT-Large模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    【模型工厂】创建ViT-Giant2模型
    
    接近ViT-Giant的配置：
    - embed_dim: 1536
    - depth: 40层
    - num_heads: 24头
    - mlp_ratio: 4
    - 每个头的维度: 1536 / 24 = 64
    
    Args:
        patch_size: patch大小
        num_register_tokens: register token数量
        **kwargs: 其他参数
        
    Returns:
        DinoVisionTransformer: ViT-Giant2模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def DINOv2(model_name):
    """
    【模型工厂】创建DINOv2模型的主函数
    
    根据模型名称创建相应的DINOv2模型，使用标准的DINOv2配置。
    
    Args:
        model_name: 模型名称，可选：
            - "vits": ViT-Small
            - "vitb": ViT-Base
            - "vitl": ViT-Large
            - "vitg": ViT-Giant2
            
    Returns:
        DinoVisionTransformer: DINOv2模型实例
        
    注意：
        - 默认图像尺寸：518x518（DINOv2的训练尺寸）
        - 默认patch大小：14x14
        - ViT-Giant使用SwiGLU FFN，其他使用MLP FFN
        - 使用Layer Scale（init_values=1.0）
    """
    model_zoo = {
        "vits": vit_small,   # ViT-Small
        "vitb": vit_base,    # ViT-Base
        "vitl": vit_large,   # ViT-Large
        "vitg": vit_giant2   # ViT-Giant2
    }
    
    # 使用DINOv2的标准配置创建模型
    return model_zoo[model_name](
        img_size=518,  # DINOv2的训练图像尺寸
        patch_size=14,  # DINOv2的patch大小
        init_values=1.0,  # Layer Scale的初始值
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",  # Giant使用SwiGLU
        block_chunks=0,  # 不使用分块
        num_register_tokens=0,  # 不使用register tokens
        interpolate_antialias=False,  # 不使用抗锯齿插值
        interpolate_offset=0.1  # 插值偏移量
    )
