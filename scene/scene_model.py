#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 场景模型类，用于管理3D高斯点云和关键帧
# 参考自：https://github.com/verlab/accelerated_features


from argparse import Namespace
import gc
import os
import json
import math
import threading
import time
import warnings
import cv2
import torch
import torch.nn.functional as F
import numpy as np

import lpips
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distIndex2
from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.guided_mvs import GuidedMVS
from scene.optimizers import SparseGaussianAdam
from scene.keyframe import Keyframe
from scene.anchor import Anchor
from utils import (
    RGB2SH,
    depth2points,
    focal2fov,
    get_lapla_norm,
    getProjectionMatrix,
    inverse_sigmoid,
    align_poses,
    make_torch_sampler,
    psnr,
    rotation_distance,
)
from dataloaders.read_write_model import write_model


class SceneModel:
    """
    【场景表示模块】场景模型类
    
    这是整个3D重建系统的核心类，负责管理：
    1. 3D高斯点云（Gaussians）：场景的几何和外观表示
    2. 关键帧（Keyframes）：包含图像、位姿、深度等信息
    3. 锚点（Anchors）：用于大尺度场景的分块管理
    4. 渲染和优化：从任意视角渲染场景，并优化高斯参数
    
    主要功能：
    - 添加新关键帧并初始化高斯点
    - 渲染场景（支持多分辨率）
    - 优化高斯参数（位置、颜色、不透明度、尺度、旋转）
    - 管理锚点（用于处理大尺度场景）
    - 保存和加载场景
    """
    def __init__(
        self,
        width: int,
        height: int,
        args: Namespace,
        matcher: Matcher = None,
        inference_mode: bool = False,
    ):
        """
        【场景表示模块】初始化场景模型
        
        Args:
            width: 图像宽度（像素）
            height: 图像高度（像素）
            args: 训练参数配置（包含学习率、损失权重、锚点重叠等）
            matcher: 特征匹配器（用于关键帧匹配，推理模式下可为None）
            inference_mode: 是否为推理模式（True=仅渲染，False=训练模式）
        """
        # ========== 基础参数初始化 ==========
        self.width = width
        self.height = height
        self.matcher = matcher
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device="cuda")  # 图像中心点
        self.anchor_overlap = args.anchor_overlap  # 锚点重叠区域大小（用于平滑融合）
        self.optimization_thread = None  # 异步优化线程（流式模式下使用）

        # ========== LPIPS评估器初始化 ==========
        # 用于评估渲染质量（感知损失）
        try:
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            warnings.filterwarnings("ignore")
            self.lpips = lpips.LPIPS(net="vgg").cuda()
            sys.stdout = original_stdout
        except:
            self.lpips = None

        # ========== 训练模式下的初始化 ==========
        if not inference_mode:
            self.num_prev_keyframes_check = args.num_prev_keyframes_check  # 用于匹配的关键帧搜索窗口
            self.active_sh_degree = args.sh_degree  # 当前使用的球谐函数阶数
            self.max_sh_degree = args.sh_degree  # 最大球谐函数阶数
            self.lambda_dssim = args.lambda_dssim  # DSSIM损失权重
            self.init_proba_scaler = args.init_proba_scaler  # 高斯初始化概率缩放因子
            self.max_active_keyframes = args.max_active_keyframes  # 最大活跃关键帧数（超过后移到CPU）
            self.use_last_frame_proba = args.use_last_frame_proba  # 使用最新帧进行训练的概率
            self.active_frames_cpu = []  # CPU上的关键帧索引
            self.active_frames_gpu = []  # GPU上的关键帧索引
            self.guided_mvs = GuidedMVS(args)  # 【场景表示模块】引导多视图立体匹配（用于深度估计）
            
            # 学习率配置（位置参数使用衰减学习率）
            self.lr_dict = {
                "xyz": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            # ========== 高斯参数初始化 ==========
            # 3D高斯点云的所有可优化参数
            self.gaussian_params = {
                "xyz": {
                    "val": torch.empty(0, 3, device="cuda"),  # 3D位置 [N, 3]
                    "lr": args.position_lr_init,
                },
                "f_dc": {
                    "val": torch.empty(0, 1, 3, device="cuda"),  # 球谐函数DC项（基础颜色）[N, 1, 3]
                    "lr": args.feature_lr,
                },
                "f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device="cuda",
                    ),  # 球谐函数高阶项（视角相关颜色）[N, SH_rest, 3]
                    "lr": args.feature_lr / 20.0,
                },
                "scaling": {
                    "val": torch.empty(0, 3, device="cuda"),  # 尺度（log空间）[N, 3]
                    "lr": args.scaling_lr,
                },
                "rotation": {
                    "val": torch.empty(0, 4, device="cuda"),  # 旋转四元数 [N, 4]
                    "lr": args.rotation_lr,
                },
                "opacity": {
                    "val": torch.empty(0, 1, device="cuda"),  # 不透明度（logit空间）[N, 1]
                    "lr": args.opacity_lr,
                },
            }
            # 创建第一个活跃锚点（包含所有高斯点）
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors = [self.active_anchor]
            # 初始化优化器
            self.reset_optimizer()

        # ========== 关键帧和锚点管理 ==========
        self.keyframes = []  # 所有关键帧列表
        self.anchor_weights = [1.0]  # 锚点混合权重（用于多锚点融合）
        self.f = 0.7 * width  # 初始焦距（像素单位，基于图像宽度的经验值）
        self.init_intrinsics()  # 初始化内参（FoV、投影矩阵等）

        # ========== 位姿管理 ==========
        self.approx_cam_centres = None  # 所有关键帧的近似相机中心（用于距离计算）
        self.gt_Rts = torch.empty(0, 4, 4, device="cuda")  # 真实位姿矩阵（4x4，用于评估）
        self.gt_Rts_mask = torch.empty(0, device="cuda", dtype=bool)  # 真实位姿有效性掩码
        self.gt_f = self.f  # 真实焦距
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")  # 缓存的优化后位姿（避免重复计算）
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)  # 位姿缓存有效性标记
        self.sorted_frame_indices = None  # 按距离排序的关键帧索引（用于匹配搜索）
        self.last_trained_id = 0  # 最后一次训练的关键帧ID
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)  # 关键帧有效性掩码
        self.lock = threading.Lock()  # 线程锁（用于多线程场景下的高斯参数访问）
        self.inference_mode = inference_mode  # 是否为推理模式

        # ========== 高斯初始化辅助工具 ==========
        # 圆盘卷积核：用于计算图像的Laplacian范数（检测纹理/边缘区域）
        radius = 3  # 卷积核半径
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        # 创建圆形掩码（距离中心<=radius+0.5的像素置为1）
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()  # 归一化（平均池化）

        # 像素坐标网格：每个像素的(u,v)坐标，用于深度估计时的坐标查询
        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )

    def reset_optimizer(self):
        """
        【优化模块】重置优化器
        
        重新初始化Adam优化器，确保所有高斯参数都启用梯度计算。
        用于在锚点切换或优化器重置时调用。
        """
        # 确保所有高斯参数都启用梯度计算
        for key in self.gaussian_params:
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        # 创建新的稀疏高斯Adam优化器
        # (0.5, 0.99): Adam的beta参数（动量衰减率）
        self.optimizer = SparseGaussianAdam(
            self.gaussian_params, (0.5, 0.99), lr_dict=self.lr_dict
        )

    @property
    def xyz(self):
        """
        【属性】获取高斯点的3D位置 [N, 3]
        
        Returns:
            torch.Tensor: 高斯点的世界坐标系位置
        """
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        """
        【属性】获取球谐函数DC项（基础颜色）[N, 1, 3]
        
        Returns:
            torch.Tensor: 球谐函数的基础颜色系数
        """
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        """
        【属性】获取球谐函数高阶项（视角相关颜色）[N, SH_rest, 3]
        
        Returns:
            torch.Tensor: 球谐函数的高阶系数（控制视角相关的外观变化）
        """
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        """
        【属性】获取高斯点的尺度 [N, 3]（从log空间转换）
        
        Returns:
            torch.Tensor: 高斯点在三个轴上的尺度（指数空间）
        """
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        """
        【属性】获取高斯点的旋转四元数 [N, 4]（归一化）
        
        Returns:
            torch.Tensor: 归一化的四元数旋转
        """
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        """
        【属性】获取高斯点的不透明度 [N, 1]（从logit空间转换）
        
        Returns:
            torch.Tensor: 不透明度值 [0, 1]
        """
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def n_active_gaussians(self):
        """
        【属性】获取当前活跃的高斯点数量
        
        Returns:
            int: 高斯点数量
        """
        return self.xyz.shape[0]

    @classmethod
    def from_scene(cls, scene_dir: str, args):
        """
        【场景表示模块】从保存的场景目录加载场景模型
        
        从场景目录读取元数据和锚点，重建场景模型。
        用于推理模式下的场景加载。
        
        Args:
            scene_dir: 场景目录路径（包含metadata.json和point_clouds/）
            args: 配置参数
            
        Returns:
            SceneModel: 加载的场景模型实例
        """
        # 读取元数据文件
        with open(os.path.join(scene_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # 从元数据获取图像尺寸
        width = metadata["config"]["width"]
        height = metadata["config"]["height"]
        # 创建场景模型（推理模式）
        scene_model = cls(width, height, args, inference_mode=True)
        scene_model.active_sh_degree = metadata["config"]["sh_degree"]
        scene_model.max_sh_degree = metadata["config"]["sh_degree"]

        # ========== 加载锚点 ==========
        # 从PLY文件加载所有锚点的高斯点云
        scene_model.anchors = []
        for i in range(len(metadata["anchors"])):
            scene_model.anchors.append(
                Anchor.from_ply(
                    os.path.join(scene_dir, "point_clouds", f"anchor_{i}.ply"),
                    torch.tensor(metadata["anchors"][i]["position"]),
                    metadata["config"]["sh_degree"],
                )
            )

        # 设置第一个锚点为活跃锚点
        scene_model.active_anchor = scene_model.anchors[0]

        # ========== 加载关键帧 ==========
        # 从JSON文件加载所有关键帧的位姿和图像信息
        for i in range(len(metadata["keyframes"])):
            keyframe = Keyframe.from_json(metadata["keyframes"][i], i, width, height)
            scene_model.add_keyframe(keyframe)

        return scene_model

    @property
    def first_active_frame(self):
        """
        【属性】获取活跃锚点中第一个关键帧的索引
        
        Returns:
            int: 第一个关键帧的索引
        """
        return self.active_anchor.keyframe_ids[0]

    @property
    def last_active_frame(self):
        """
        【属性】获取活跃锚点中最后一个关键帧的索引
        
        Returns:
            int: 最后一个关键帧的索引
        """
        return self.active_anchor.keyframe_ids[-1]

    @property
    def n_active_keyframes(self):
        """
        【属性】获取活跃关键帧的数量
        
        Returns:
            int: 活跃关键帧数量
        """
        return self.last_active_frame - self.first_active_frame + 1

    def optimization_step(self, finetuning=False):
        """
        【优化模块】执行一步优化
        
        这是训练的核心函数，执行以下步骤：
        1. 选择要优化的关键帧
        2. 从该关键帧视角渲染场景
        3. 计算损失（L1 + DSSIM + 深度损失）
        4. 反向传播并更新参数（高斯参数 + 关键帧位姿）
        
        Args:
            finetuning: 是否为微调模式（微调时随机选择关键帧）
        """
        if len(self.xyz) == 0:
            return
        
        # ========== 关键帧选择策略 ==========
        # 训练策略：以一定概率使用最新关键帧，否则随机选择
        # 这样可以平衡新区域的学习和旧区域的细化
        if (
            np.random.rand() > self.use_last_frame_proba
            or self.last_trained_id == -1
            or finetuning
        ):
            keyframe_id = np.random.choice(self.active_frames_gpu)
        else:
            keyframe_id = -1  # 使用最新关键帧
        keyframe = self.keyframes[keyframe_id]
        lvl = keyframe.pyr_lvl  # 当前使用的金字塔层级

        # ========== 梯度清零 ==========
        keyframe.zero_grad()
        self.optimizer.zero_grad()

        # ========== 渲染 ==========
        # 【渲染模块】从关键帧视角渲染图像和深度
        render_pkg = self.render_from_id(
            keyframe_id, pyr_lvl=lvl, bg=torch.rand(3, device="cuda")
        )
        image = render_pkg["render"]  # 渲染的RGB图像 [3, H, W]
        invdepth = render_pkg["invdepth"]  # 渲染的逆深度 [1, H, W]

        # 获取真实图像和单目深度
        gt_image = keyframe.image_pyr[lvl]
        mono_idepth = keyframe.get_mono_idepth(lvl)

        # ========== 掩码处理 ==========
        # 如果有关键帧掩码，应用掩码（排除无效区域）
        if keyframe.mask_pyr is not None:
            image = image * keyframe.mask_pyr[lvl]
            gt_image = gt_image * keyframe.mask_pyr[lvl]
            invdepth = invdepth * keyframe.mask_pyr[lvl]
            mono_idepth = mono_idepth * keyframe.mask_pyr[lvl]

        # ========== 损失计算 ==========
        # 【损失函数模块】计算多任务损失
        l1_loss = (image - gt_image).abs().mean()  # L1损失（像素级）
        ssim_loss = 1 - fused_ssim(image[None], gt_image[None])  # DSSIM损失（结构相似性）
        depth_loss = (invdepth - mono_idepth).abs().mean()  # 深度损失（与单目深度对齐）
        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
            + keyframe.depth_loss_weight * depth_loss
        )
        loss.backward()

        # ========== 参数更新 ==========
        with torch.no_grad():
            # 【优化模块】更新关键帧位姿（6D表示 + 曝光 + 深度缩放/偏移）
            keyframe.step()

            # 测试关键帧不参与场景优化（仅用于评估）
            if not keyframe.info["is_test"]:
                # 【优化模块】更新高斯参数（位置、颜色、不透明度、尺度、旋转）
                # visibility_filter: 可见性掩码（用于自适应密度控制）
                # radii.shape[0]: 高斯点数量（用于学习率调度）
                self.optimizer.step(
                    render_pkg["visibility_filter"], render_pkg["radii"].shape[0]
                )

            # 保存最新渲染的逆深度（用于后续的三角化）
            keyframe.latest_invdepth = render_pkg["invdepth"].detach()

        # 标记位姿缓存失效（需要重新计算）
        self.valid_Rt_cache[keyframe_id] = False
        self.last_trained_id = keyframe_id

    def optimization_loop(self, n_iters: int, run_until_interupt: bool = False):
        """
        【优化模块】优化循环
        
        执行至少n_iters次优化步骤。
        如果run_until_interupt为True，会持续运行直到join_optimization_thread被调用
        （用于流式模式下持续优化直到添加下一个关键帧）。
        
        Args:
            n_iters: 最小优化迭代次数
            run_until_interupt: 是否持续运行直到中断信号
        """
        # 重置中断标志
        self.interupt_optimization = False
        i = 0
        # 持续优化直到达到最小迭代次数，或收到中断信号
        while i < n_iters or (run_until_interupt and not self.interupt_optimization): 
            self.optimization_step()
            i += 1
        
    def join_optimization_thread(self):
        """
        【优化模块】中断优化循环并等待线程结束
        
        发送中断信号给优化线程，并等待其完成当前迭代后退出。
        用于在添加新关键帧前确保优化线程已停止。
        """
        if self.optimization_thread is not None:
            # 设置中断标志
            self.interupt_optimization = True
            # 等待线程结束
            self.optimization_thread.join()
            self.optimization_thread = None
    
    def optimize_async(self, n_iters: int):
        """
        【优化模块】异步启动优化线程
        
        在后台线程中运行优化循环，至少执行n_iters次优化步骤。
        用于流式模式下在不阻塞主线程的情况下持续优化场景。
        
        Args:
            n_iters: 最小优化迭代次数
        """
        # 先停止之前的优化线程（如果存在）
        self.join_optimization_thread()
        # 创建新的优化线程
        self.optimization_thread = threading.Thread(
            target=self.optimization_loop, args=(n_iters, True)
        )
        self.optimization_thread.start()

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """
        【渲染模块】统一测试关键帧的曝光矩阵
        
        通过平均相邻关键帧的曝光值来统一测试关键帧的曝光。
        这样可以确保测试帧的渲染质量不受曝光差异影响。
        """
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.info["is_test"]:
                # 获取前一个和后一个关键帧的索引（边界处理）
                idxm = index - 1 if index != 0 else 1
                idxp = (
                    index + 1
                    if index != len(self.keyframes) - 1
                    else len(self.keyframes) - 2
                )
                # 使用相邻关键帧的平均曝光
                keyframe.exposure = (
                    self.keyframes[idxm].exposure + self.keyframes[idxp].exposure
                ) / 2

    @torch.no_grad()
    def evaluate(self, eval_poses=False, with_LPIPS=False, all=False):
        """
        【评估模块】评估场景质量
        
        计算渲染质量和位姿误差指标。
        
        Args:
            eval_poses: 是否计算位姿误差
            with_LPIPS: 是否计算LPIPS感知损失
            all: 是否评估所有关键帧（False时只评估活跃锚点的关键帧）
            
        Returns:
            dict: 包含PSNR、SSIM、LPIPS（可选）、位姿误差（可选）的字典
        """
        # ========== 统一测试关键帧曝光 ==========
        # 确保测试关键帧的曝光与相邻帧一致
        self.harmonize_test_exposure()

        # ========== 计算图像质量指标 ==========
        # 初始化指标字典
        metrics = {"PSNR": 0, "SSIM": 0}
        if with_LPIPS:
            metrics["LPIPS"] = 0
        n_test_frames = 0
        # 确定评估的关键帧范围
        start_index = 0 if all else self.active_anchor.keyframe_ids[0]
        
        # 遍历测试关键帧并计算指标
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if keyframe.info["is_test"]:
                # 获取真实图像（全分辨率）
                gt_image = keyframe.image_pyr[0].cuda()
                # 渲染当前视角
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]
                
                # 应用掩码（如果有）
                mask = (
                    keyframe.mask_pyr[0].cuda()
                    if keyframe.mask_pyr is not None
                    else torch.ones_like(image[:1] > 0)
                )
                mask = mask.expand_as(image)
                image = image * mask
                gt_image = gt_image * mask
                
                # 计算PSNR（峰值信噪比）
                metrics["PSNR"] += psnr(image[mask], gt_image[mask])
                # 计算SSIM（结构相似性）
                metrics["SSIM"] += fused_ssim(
                    image[None], gt_image[None], train=False
                ).item()
                # 计算LPIPS（感知损失，如果启用）
                if with_LPIPS and self.lpips is not None:
                    metrics["LPIPS"] += self.lpips(image[None], gt_image[None]).item()
                n_test_frames += 1

        # 计算平均指标
        if n_test_frames > 0:
            for metric in metrics:
                metrics[metric] /= n_test_frames
        else:
            metrics = {}

        # ========== 计算位姿误差 ==========
        if eval_poses:
            # 获取优化后的位姿和真实位姿
            Rts = self.get_Rts()
            gt_Rts = self.get_gt_Rts(align=False)
            if len(Rts) == len(gt_Rts):
                # 对齐位姿（计算相似变换）
                Rts_aligned = torch.linalg.inv(align_poses(Rts, gt_Rts))
                gt_Rts = torch.linalg.inv(gt_Rts)
                # 计算旋转误差（角度）
                R_error = rotation_distance(Rts_aligned[:, :3, :3], gt_Rts[:, :3, :3])
                # 计算平移误差（欧氏距离）
                t_error = (Rts_aligned[:, :3, 3] - gt_Rts[:, :3, 3]).norm(dim=-1)

                # 转换为度数和米
                metrics["R°"] = R_error.mean().item() * 180 / math.pi
                metrics["t"] = t_error.mean().item()

        return metrics

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        """
        【评估模块】保存测试关键帧的渲染图像
        
        为所有测试关键帧渲染图像并保存到指定目录。
        用于生成评估结果的可视化。
        
        Args:
            out_dir: 输出目录路径
        """
        # 统一测试关键帧曝光，确保渲染质量
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)
        
        # 遍历所有关键帧，渲染并保存测试帧
        for keyframe in self.keyframes:
            if keyframe.info["is_test"]:
                # 渲染全分辨率图像
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                # 转换为8位RGB图像（0-255范围）
                image = torch.clamp(render_pkg["render"], 0, 1) * 255
                image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # 转换为BGR格式（OpenCV使用BGR）
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # 根据文件扩展名选择保存格式（JPEG或PNG）
                is_jpeg = os.path.splitext(keyframe.info["name"])[-1].lower() in [
                    ".jpg",
                    ".jpeg",
                ]
                # JPEG质量设置为100（无损）
                write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
                cv2.imwrite(
                    os.path.join(out_dir, keyframe.info["name"]), image, write_flag
                )

    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        scaling_modifier=1,
        bg=torch.zeros(3, device="cuda"),
    ):
        """
        【渲染模块】从指定关键帧ID渲染场景
        
        从给定关键帧视角渲染场景，支持多分辨率（金字塔层级）和曝光校正。
        
        Args:
            keyframe_id: 关键帧索引
            pyr_lvl: 金字塔层级（0=全分辨率，1=半分辨率，...）
            scaling_modifier: 高斯尺度缩放因子（用于可视化）
            bg: 背景颜色 [3]
            
        Returns:
            dict: 包含渲染图像、逆深度、主要高斯ID、半径等信息的字典
        """
        # 获取关键帧和视图矩阵
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt().transpose(0, 1)
        
        # 根据金字塔层级计算分辨率
        scale = 2**pyr_lvl
        width, height = self.width // scale, self.height // scale
        
        # 调用底层渲染函数
        render_pkg = self.render(width, height, view_matrix, scaling_modifier, bg)
        
        # ========== 应用曝光校正 ==========
        # 曝光矩阵：[3x4]，包含颜色变换和平移
        # 将渲染图像从原始颜色空间转换到关键帧的曝光空间
        render_pkg["render"] = (
            keyframe.exposure[:3, :3] @ render_pkg["render"].view(3, -1)
        ) + keyframe.exposure[:3, 3, None]
        # 裁剪到[0,1]范围并恢复形状
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        return render_pkg

    def render(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = torch.zeros(3, device="cuda"),
        top_view: bool = False,
        fov_x: float = None,
        fov_y: float = None,
    ):
        """
        【渲染模块】底层渲染函数
        
        使用3D高斯光栅化渲染图像和深度。支持自定义分辨率和视场角。
        这是所有渲染功能的底层实现。
        
        Args:
            width: 渲染图像宽度
            height: 渲染图像高度
            view_matrix: 视图矩阵（4x4，世界到相机坐标变换）
            scaling_modifier: 高斯尺度缩放因子（1.0=正常，>1.0=放大）
            bg: 背景颜色 [3]
            top_view: 是否为顶视图模式（用于可视化高斯位置）
            fov_x: 水平视场角（弧度，可选，默认使用场景内参）
            fov_y: 垂直视场角（弧度，可选，默认使用场景内参）
            
        Returns:
            dict: 包含渲染图像、逆深度、主要高斯ID、半径等信息的字典
        """
        # 计算相机中心（视图矩阵的逆矩阵的第4列前3行）
        cam_centre = view_matrix.detach().inverse()[3, :3]

        # ========== 内参设置 ==========
        # 如果没有提供自定义FOV，使用场景的默认内参
        if fov_x is None and fov_y is None:
            tanfovx, tanfovy = self.tanfovx, self.tanfovy
            projection_matrix = self.projection_matrix
        # 如果提供了自定义FOV，计算对应的投影参数
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)
                .transpose(0, 1)
                .cuda()
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        # ========== 光栅化设置 ==========
        # 创建高斯光栅化器配置
        raster_settings = GaussianRasterizationSettings(
            height,
            width,
            tanfovx,
            tanfovy,
            bg,
            1 if top_view else scaling_modifier,  # 顶视图模式下不使用缩放
            projection_matrix,
            self.active_sh_degree,  # 当前使用的球谐函数阶数
            cam_centre,
            False,  # 是否使用预计算的颜色
            False,  # 其他选项
        )
        rasterizer = GaussianRasterizer(raster_settings)
        
        # ========== 多线程安全的高斯参数访问 ==========
        with self.lock:
            # 【场景表示模块】推理模式下混合多个锚点的高斯参数
            # 策略：根据相机中心到锚点的距离加权混合（用于大尺度场景）
            if self.inference_mode and not top_view:
                self.gaussian_params, self.anchor_weights = Anchor.blend(
                    cam_centre, self.anchors, self.anchor_overlap
                )
            
            # 屏幕空间点（用于计算2D位置，需要梯度用于优化）
            screenspace_points = torch.zeros_like(self.xyz, requires_grad=True)
            
            if self.xyz.shape[0] > 0:
                # ========== 顶视图模式 ==========
                # 顶视图：使用固定尺度和不透明度，便于可视化高斯点位置
                if top_view:
                    scaling = torch.ones_like(self.scaling) * scaling_modifier
                    opacity = torch.ones_like(self.opacity)
                else:
                    # 正常渲染：使用优化后的高斯参数
                    scaling = self.scaling
                    opacity = self.opacity
                
                # ========== 执行光栅化 ==========
                # 【渲染模块】调用底层CUDA光栅化器进行渲染
                color, invdepth, mainGaussID, radii = rasterizer(
                    self.xyz,  # 3D位置
                    screenspace_points,  # 屏幕空间位置（输出）
                    opacity,  # 不透明度
                    self.f_dc,  # 球谐函数DC项
                    self.f_rest,  # 球谐函数高阶项
                    scaling,  # 尺度
                    self.rotation,  # 旋转四元数
                    view_matrix,  # 视图矩阵
                )
            else:
                # ========== 空场景处理 ==========
                # 如果没有高斯点，返回空张量
                color = torch.zeros(3, height, width, device="cuda")
                invdepth = torch.zeros(1, height, width, device="cuda")
                mainGaussID = torch.zeros(
                    1, height, width, device="cuda", dtype=torch.int32
                )
                radii = torch.zeros(1, height, width, device="cuda")
        
        # 返回渲染结果字典
        return {
            "render": color,  # RGB图像 [3, H, W]
            "invdepth": invdepth,  # 逆深度 [1, H, W]
            "mainGaussID": mainGaussID,  # 每个像素的主要高斯点ID [1, H, W]
            "radii": radii,  # 每个高斯点在屏幕空间的半径 [1, H, W]
            "visibility_filter": radii > 0,  # 可见性掩码（用于自适应密度控制）
            "screenspace_points": screenspace_points,  # 屏幕空间位置（用于优化）
        }

    def get_closest_by_cam(self, cam_centre, k=3):
        """
        【场景表示模块】根据相机中心获取最近的k个锚点
        
        用于推理模式下选择需要混合的锚点。根据相机中心到锚点的距离排序。
        
        Args:
            cam_centre: 相机中心位置 [3]
            k: 返回的锚点数量
            
        Returns:
            tuple: (最近锚点列表, 锚点ID列表)
        """
        closest_anchors = []
        closest_anchors_ids = []
        offset = 0
        # 克隆相机中心列表（用于标记已选中的锚点）
        approx_cam_centres = self.approx_cam_centres.clone()
        
        # 迭代选择k个最近的锚点
        for l in range(min(k, len(self.anchors))):
            if approx_cam_centres.shape[0] == 0:
                break
            # 计算到所有相机中心的距离
            dists = torch.linalg.norm(approx_cam_centres - cam_centre[None], dim=-1)
            min_dist, min_id = torch.min(dists, dim=0)

            # 如果找到有效距离（<1e9表示未被标记）
            if min_dist < 1e9:
                # 查找对应的锚点
                for anchor_id, anchor in enumerate(self.anchors):
                    if min_id in anchor.keyframe_ids:
                        closest_anchors.append(anchor)
                        closest_anchors_ids.append(anchor_id)
                        # 标记该锚点的所有关键帧为已选中（设置为大值）
                        approx_cam_centres[
                            anchor.keyframe_ids[0] : anchor.keyframe_ids[-1] + 1
                        ] = 1e9
                        break

        return closest_anchors, closest_anchors_ids

    @torch.no_grad()
    def get_prev_keyframes(self, n: int, update_3dpts: bool, desc_kpts: DescribedKeypoints = None):
        """
        【场景表示模块】获取最近的n个关键帧
        
        用于深度估计和匹配。如果提供了特征点描述符，会基于特征匹配数量选择关键帧；
        否则基于空间距离选择。
        
        Args:
            n: 要返回的关键帧数量
            update_3dpts: 是否更新关键帧的3D点（重新三角化）
            desc_kpts: 特征点描述符（可选，用于基于匹配选择关键帧）
            
        Returns:
            list[Keyframe]: 最近的n个关键帧列表
        """
        # ========== 确保优化线程已停止 ==========
        # 避免在多线程环境下访问关键帧数据时出现冲突
        self.join_optimization_thread()

        # ========== 关键帧选择策略 ==========
        # 如果提供了特征点描述符，基于特征匹配数量选择关键帧
        if desc_kpts is not None and len(self.keyframes) > n:
            # 在搜索窗口内查找匹配数量最多的关键帧
            n_ckecks = min(self.num_prev_keyframes_check, len(self.keyframes))
            keyframes_indices_to_check = self.sorted_frame_indices[:n_ckecks]
            n_matches = torch.zeros(len(keyframes_indices_to_check), device="cuda")
            # 计算每个候选关键帧的匹配数量
            for i, index in enumerate(keyframes_indices_to_check):
                n_matches[i] = self.matcher.evaluate_match(
                    self.keyframes[index].desc_kpts, desc_kpts
                )
            # 选择匹配数量最多的n个关键帧
            _, top_indices = torch.topk(n_matches, n)
            prev_keyframes_indices = keyframes_indices_to_check[top_indices.cpu()]
        # 如果没有提供特征点描述符，直接选择距离最近的n个关键帧
        else:
            prev_keyframes_indices = self.sorted_frame_indices[:n]
        prev_keyframes = [self.keyframes[i] for i in prev_keyframes_indices]

        # ========== 更新3D点 ==========
        # 如果需要，重新三角化关键帧的3D点（用于深度对齐）
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes)
        return prev_keyframes

    def get_Rts(self):
        """
        【场景表示模块】获取所有关键帧的位姿矩阵（带缓存）
        
        返回缓存的位姿矩阵，如果缓存失效则重新计算。
        用于提高渲染和评估时的性能。
        
        Returns:
            torch.Tensor: 所有关键帧的位姿矩阵 [N, 4, 4]
        """
        # 查找缓存失效的关键帧ID
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            # 重新计算失效的位姿并更新缓存
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    def get_gt_Rts(self, align):
        """
        【评估模块】获取真实位姿矩阵
        
        Args:
            align: 是否对齐到优化后的位姿（用于计算误差）
            
        Returns:
            torch.Tensor: 真实位姿矩阵 [N, 4, 4]
        """
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        # 如果需要对齐，计算相似变换将真实位姿对齐到优化位姿
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses(self.gt_Rts[: len(Rts)], Rts)
        else:
            return self.gt_Rts

    def make_dummy_ext_tensor(self):
        """
        【优化模块】创建空的高斯参数张量字典
        
        用于剪枝操作（只移除高斯点，不添加新点）。
        
        Returns:
            dict: 空的高斯参数字典（所有张量的第一维为0）
        """
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
        }

    def reset(self, keyframe_id: int = -1):
        """
        【优化模块】移除指定关键帧中可见的高斯点
        
        用于重置场景的特定区域（例如，当关键帧位姿发生大幅变化时）。
        
        Args:
            keyframe_id: 关键帧索引（-1表示最新关键帧）
        """
        # 初始掩码：保留不透明度>0.05的高斯点
        valid_mask = self.opacity[:, 0] > 0.05
        # 渲染关键帧，获取可见性掩码
        render_pkg = self.render_from_id(keyframe_id)
        # 将可见的高斯点标记为无效（移除）
        valid_mask[render_pkg["visibility_filter"]] = False
        # 执行剪枝（不添加新点，只移除无效点）
        self.optimizer.add_and_prune(self.make_dummy_ext_tensor(), valid_mask)

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """
        【场景表示模块】为新关键帧初始化3D高斯点
        
        这是高斯点云增长的核心函数，执行以下步骤：
        1. 对齐关键帧的单目深度到三角化深度
        2. 基于Laplacian概率采样候选像素位置
        3. 使用引导MVS估计深度
        4. 初始化高斯参数（位置、颜色、尺度、不透明度等）
        5. 剪枝遮挡和过大的高斯点
        
        Args:
            keyframe_id: 关键帧索引（-1表示最新关键帧）
        """
        keyframe = self.keyframes[keyframe_id]
        
        # ========== 深度对齐 ==========
        # 如果关键点还没有3D点，先进行三角化
        if keyframe.desc_kpts.has_pt3d.sum() == 0:
            keyframe.update_3dpts(self.keyframes)
        # 【场景表示模块】对齐单目深度到三角化深度（通过缩放和偏移）
        keyframe.align_depth()

        # 测试关键帧不添加高斯点（仅用于评估）
        if keyframe.info["is_test"]:
            return

        # ========== 基于Laplacian的概率采样 ==========
        # 【场景表示模块】计算每个像素成为新高斯点的概率
        # 策略：在图像边缘/纹理丰富区域（高Laplacian）更可能添加高斯点
        img = keyframe.image_pyr[0]
        img = F.avg_pool2d(img, 2)  # 轻微下采样以减少噪声
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        init_proba = get_lapla_norm(img, self.disc_kernel)  # 公式1：Laplacian范数作为概率

        # 应用掩码（排除无效区域）
        if keyframe.mask_pyr is not None:
            dilated_mask = (
                F.conv2d(
                    keyframe.mask_pyr[0][None].float(), self.disc_kernel, padding="same"
                )[0, 0]
                >= 0.99
            )
            init_proba *= dilated_mask

        # ========== 渲染惩罚机制 ==========
        # 【场景表示模块】计算惩罚项：如果已有高斯点能很好地渲染该区域，则降低添加新点的概率
        # 这避免了在已有良好表示的区域重复添加高斯点
        penalty = 0
        rendered_depth = None
        if self.xyz.shape[0] > 0:
            render_pkg = self.render_from_id(keyframe_id)
            render = render_pkg["render"]
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
            penalty = get_lapla_norm(render, self.disc_kernel)  # 渲染图像的Laplacian作为惩罚

        # ========== 采样掩码生成 ==========
        # 公式3：最终采样概率 = init_proba - penalty
        # 在纹理丰富且渲染质量差的区域添加新高斯点
        init_proba *= self.init_proba_scaler
        penalty *= self.init_proba_scaler
        sample_mask = torch.rand_like(init_proba) < init_proba - penalty

        # ========== 深度估计 ==========
        sampled_uv = self.uv[sample_mask]  # 采样像素坐标
        
        # 【场景表示模块】使用引导多视图立体匹配（Guided MVS）估计深度
        # 策略：利用历史关键帧的密集特征进行立体匹配
        prev_KFs = self.get_prev_keyframes(
            self.guided_mvs.n_cams + 1, update_3dpts=False
        )
        for i, prev_keyframe in enumerate(prev_KFs):
            if keyframe.index == prev_keyframe.index:
                prev_KFs.pop(i)
                break
        depth, accurate_mask = self.guided_mvs(sampled_uv, keyframe, prev_KFs)
        
        # 过滤：保留置信度高且深度有效的点
        valid_mask = (keyframe.sample_conf(sampled_uv) > 0.5) * (depth > 1e-6)
        sample_mask[sample_mask.clone()] = valid_mask
        depth = depth[valid_mask]
        sampled_uv = sampled_uv[valid_mask]
        accurate_mask = accurate_mask[valid_mask]

        # ========== 剪枝过粗的高斯点 ==========
        # 【场景表示模块】如果新点比现有高斯点更精细，则移除过粗的旧高斯点
        # 策略：统计每个旧高斯点被新点覆盖的次数，如果超过阈值则移除
        if len(self.xyz) > 0:
            main_gaussians_map = render_pkg["mainGaussID"]  # 每个像素对应的主要高斯点ID
            accurate_sample_mask = sample_mask.clone()
            accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
            selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
            ids, counts = torch.unique(
                selected_main_gaussians[selected_main_gaussians >= 0],
                return_counts=True,
            )
            valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
            valid_gs_mask[ids] = counts < 10  # 被覆盖次数少于10次的保留
            with self.lock:
                self.optimizer.add_and_prune(
                    self.make_dummy_ext_tensor(), valid_gs_mask
                )
            render_pkg = self.render_from_id(keyframe_id)
            rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

        # ========== 遮挡检查 ==========
        # 【场景表示模块】移除被现有高斯点遮挡的新点（避免重复表示）
        if rendered_depth is not None:
            valid_mask = depth < rendered_depth[sample_mask]  # 新点深度必须小于渲染深度
            sample_mask[sample_mask.clone()] = valid_mask
            depth = depth[valid_mask]
            sampled_uv = sampled_uv[valid_mask]
            accurate_mask = accurate_mask[valid_mask]

        # ========== 3D位置初始化 ==========
        # 【场景表示模块】将像素坐标+深度转换为世界坐标系3D点
        new_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
        new_pts = (new_pts - keyframe.get_t()) @ keyframe.get_R()  # 转换到世界坐标系
        
        # 添加从特征匹配三角化得到的3D点（这些点通常更准确）
        match_pts = keyframe.desc_kpts.pts3d[keyframe.desc_kpts.has_pt3d]
        new_pts = torch.cat([new_pts, match_pts], dim=0)

        # ========== 颜色初始化 ==========
        # 【场景表示模块】从图像中采样颜色并转换为球谐函数表示
        f_dc = img[:, sample_mask]  # 采样像素的颜色
        match_sampler = keyframe.desc_kpts.kpts[keyframe.desc_kpts.has_pt3d]
        match_sampler = make_torch_sampler(match_sampler, self.width, self.height)
        match_colors = F.grid_sample(
            img[None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(3, -1)
        f_dc = torch.cat([f_dc, match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))  # RGB转球谐函数DC项

        # ========== 尺度初始化 ==========
        # 【场景表示模块】根据初始化概率和到相机距离设置高斯尺度
        # 公式4：尺度与初始化概率的平方根成反比，并考虑距离
        sampled_init_proba = init_proba[sample_mask]
        match_init_proba = F.grid_sample(
            init_proba[None, None],
            match_sampler[None, None],
            mode="bilinear",
            align_corners=True,
        ).view(-1)
        sampled_init_proba = torch.cat([sampled_init_proba, match_init_proba], dim=0)
        # 期望到最近邻的距离（公式4）
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)  # 限制尺度范围
        # 根据到相机中心的距离缩放
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(
            new_pts - keyframe.approx_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)  # log空间

        # ========== 不透明度初始化 ==========
        # 【场景表示模块】根据深度估计精度设置初始不透明度
        opacities = torch.ones(f_dc.shape[0], 1, device="cuda")
        # 不准确的点使用较低的不透明度（0.02），准确的点使用稍高的不透明度（0.07）
        opacities[: sampled_uv.shape[0]] *= (
            0.07 * accurate_mask[..., None] + 0.02 * ~accurate_mask[..., None]
        )
        # 三角化得到的点使用更高的不透明度（0.2）
        opacities[sampled_uv.shape[0] :] *= 0.2
        opacities = inverse_sigmoid(opacities)  # 转换到logit空间

        # ========== 其他参数初始化 ==========
        # 【场景表示模块】球谐函数高阶项初始化为0（视角相关颜色）
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device="cuda",
        )
        # 旋转初始化为单位四元数（无旋转）
        rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
        rots[:, 0] = 1

        # ========== 剪枝策略 ==========
        # 【场景表示模块】确定哪些现有高斯点应该被剪枝
        if self.xyz.shape[0] > 0:
            # 只保留不透明度足够高的高斯点（>0.05）
            valid_gs_mask = self.opacity[:, 0] > 0.05

            # 移除在屏幕上过大的高斯点（可能是异常值）
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], dim=-1
            )
            screen_size = self.f * self.scaling.max(dim=-1)[0] / dist  # 屏幕空间大小
            valid_gs_mask *= screen_size < 0.5 * self.width  # 屏幕大小不能超过图像宽度的一半
        else:
            valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)

        # ========== 添加新高斯点 ==========
        # 【优化模块】将新高斯点添加到优化器中，同时剪枝无效的旧高斯点
        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        with self.lock:
            self.optimizer.add_and_prune(extension_tensors, valid_gs_mask)

    def init_intrinsics(self):
        """
        【渲染模块】初始化相机内参
        
        根据焦距和图像尺寸计算视场角（FoV）和投影矩阵。
        用于光栅化渲染时的坐标变换。
        """
        # 计算水平和垂直视场角（弧度）
        self.FoVx = focal2fov(self.f, self.width)
        self.FoVy = focal2fov(self.f, self.height)
        # 计算半视场角的正切值（用于光栅化）
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        # 计算投影矩阵（OpenGL格式，4x4）
        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)  # 转置为列主序（OpenGL格式）
            .cuda()
        )

    def move_rand_keyframe_to_cpu(self):
        """
        【内存管理模块】将随机关键帧移动到CPU内存
        
        当活跃关键帧数量超过限制时，将部分关键帧移到CPU以节省GPU内存。
        保留最后n_kept_frames个关键帧始终在GPU上。
        """
        # 从GPU关键帧中随机选择一个（排除最后n_kept_frames个）
        frame_id = np.random.choice(self.active_frames_gpu[:-self.n_kept_frames])
        self.keyframes[frame_id].to("cpu")
        self.active_frames_cpu.append(frame_id)
        self.active_frames_gpu.remove(frame_id) 

    def move_rand_keyframe_to_gpu(self):
        """
        【内存管理模块】将随机关键帧移动到GPU内存
        
        当需要更多关键帧参与训练时，从CPU加载关键帧到GPU。
        """
        if len(self.active_frames_cpu) > 0:
            frame_id = np.random.choice(self.active_frames_cpu)
            self.keyframes[frame_id].to("cuda")
            self.active_frames_gpu.insert(0, frame_id)  # 插入到列表开头（优先使用）
            self.active_frames_cpu.remove(frame_id) 

    def add_keyframe(self, keyframe: Keyframe, f=None):
        """
        【场景表示模块】添加新关键帧到场景
        
        这是场景增长的核心函数，执行以下操作：
        1. 将关键帧添加到列表并更新索引
        2. 更新相机内参（如果提供新的焦距）
        3. 更新位姿缓存
        4. 将关键帧添加到活跃锚点
        5. 管理GPU/CPU内存（当关键帧过多时）
        
        Args:
            keyframe: 要添加的关键帧对象
            f: 新的焦距值（可选，如果提供则更新内参）
        """

        # ========== 确保训练线程已停止 ==========
        # 避免在添加关键帧时与优化线程冲突
        self.join_optimization_thread()

        # ========== 添加关键帧并更新索引 ==========
        # 将关键帧添加到列表
        self.keyframes.append(keyframe)
        # 更新近似相机中心列表（用于距离计算和锚点选择）
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        # 计算所有关键帧到最新关键帧的距离，并排序（用于匹配搜索）
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        # ========== 更新内参 ==========
        # 如果提供了新焦距，更新内参（用于处理变焦或焦距估计变化）
        if f is not None:
            self.f = f.item()
            self.init_intrinsics()

        # ========== 更新位姿缓存 ==========
        # 为新关键帧添加位姿缓存条目（初始标记为有效）
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device="cuda", dtype=torch.bool)], dim=0
        )
        # 如果有真实位姿（用于评估），添加到gt_Rts
        gt_pose = keyframe.info.get("Rt", None)
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0,
        )
        self.gt_f = keyframe.info.get("focal", self.f)

        # ========== 训练模式下的额外操作 ==========
        if not self.inference_mode:
            # 将关键帧添加到活跃锚点
            self.active_anchor.add_keyframe(keyframe)
            self.active_frames_gpu.append(keyframe.index)

            # ========== 内存管理 ==========
            # 如果活跃关键帧数量超过限制，将部分关键帧移到CPU
            if len(self.active_frames_gpu) > self.max_active_keyframes:
                self.move_rand_keyframe_to_cpu()
                # 每5个关键帧重排一次（保持GPU/CPU关键帧的平衡）
                if len(self.active_frames_cpu) % 5 == 0:
                    self.move_rand_keyframe_to_cpu()
                    self.move_rand_keyframe_to_gpu()
                    # 清理内存碎片
                    gc.collect()
                    torch.cuda.empty_cache()

    def enable_inference_mode(self):
        """
        【场景表示模块】启用推理模式
        
        切换到推理模式（停止训练），并更新锚点位置为活跃关键帧的平均位置。
        用于完成训练后的场景渲染。
        """
        self.inference_mode = True
        self.update_anchor()

    def update_anchor(self, n_left_frames: int = 0):
        """
        【场景表示模块】更新锚点位置
        
        将锚点位置设置为活跃关键帧相机中心的平均值，并可选地移除最后n_left_frames个关键帧。
        用于锚点固定（在创建新锚点前）。
        
        Args:
            n_left_frames: 要从活跃锚点移除的关键帧数量（从末尾移除）
        """
        # 计算活跃关键帧的相机中心平均值（排除最后n_left_frames个）
        anchor_position = self.approx_cam_centres[
            self.first_active_frame : self.last_active_frame - n_left_frames
        ].mean(dim=0)
        self.active_anchor.position = anchor_position
        # 如果指定了要移除的关键帧数量，从锚点中移除
        if n_left_frames > 0:
            self.active_anchor.keyframes = self.active_anchor.keyframes[:-n_left_frames]
            self.active_anchor.keyframe_ids = self.active_anchor.keyframe_ids[
                :-n_left_frames
            ]

    def place_anchor_if_needed(self):
        """
        【场景表示模块】根据高斯点大小判断是否需要创建新锚点
        
        当大部分高斯点在屏幕上显示很小时（大尺度场景），创建新锚点并合并细小的高斯点。
        这是大尺度场景管理的关键函数。
        
        策略：
        1. 检查屏幕空间大小<1的高斯点比例
        2. 如果超过阈值，固定当前锚点并创建新锚点
        3. 合并细小的高斯点（减少点数，提高效率）
        """
        small_prop_thresh = 0.4  # 细小高斯点比例阈值（超过40%则创建锚点）
        k = 3  # 每个高斯点合并的最近邻数量
        self.n_kept_frames = 20  # 在新锚点中保留的关键帧数量
        if (
            self.xyz.shape[0] > 0
            and self.first_active_frame < len(self.keyframes) - 2 * self.n_kept_frames
        ):
            with torch.no_grad():
                dist = torch.linalg.vector_norm(
                    self.xyz - self.approx_cam_centres[-1][None], dim=-1
                )
                screen_size = self.f * self.scaling.mean(dim=-1) / dist
                small_mask = screen_size < 1
                small_prop = small_mask.float().mean()

            if small_prop > small_prop_thresh:
                with torch.no_grad():
                    # 扩大细小高斯点的掩码（屏幕大小<1.5）
                    small_mask = screen_size < 1.5
                    # 【场景表示模块】更新锚点位置（使用用于优化的相机位姿）
                    # 固定当前锚点，移除最后n_kept_frames个关键帧（这些将用于新锚点）
                    self.update_anchor(self.n_kept_frames)

                    # ========== 合并细小高斯点 ==========
                    # 【优化模块】选择需要合并的高斯点及其最近邻
                    # 提取所有细小高斯点的参数
                    small_gaussians = {
                        name: self.gaussian_params[name]["val"][small_mask]
                        for name in self.gaussian_params
                    }
                    xyz = small_gaussians["xyz"].contiguous()
                    # 使用KNN查找每个细小高斯点的k个最近邻
                    _, nn_idx = distIndex2(xyz, k)
                    nn_idx = nn_idx.view(-1, k)
                    # 随机选择部分高斯点作为合并中心（每个组包含k+1个高斯点）
                    perm = torch.randperm(xyz.shape[0], device=xyz.device)
                    idx = perm[: (xyz.shape[0] // (k + 1))]
                    # 每个合并组包含：1个中心点 + k个最近邻点
                    selected_nn_idx = torch.cat([idx[..., None], nn_idx[idx]], dim=-1)

                    # ========== 计算合并权重 ==========
                    # 【优化模块】基于高斯点对渲染的贡献计算合并权重
                    # 权重 = 不透明度 * 屏幕空间大小的平方（表示渲染贡献）
                    weights = self.gaussian_params["opacity"]["val"][
                        selected_nn_idx, 0
                    ].sigmoid() * (screen_size[selected_nn_idx] ** 2)
                    # 归一化权重（使每个组的权重和为1）
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                    weights.unsqueeze_(-1)

                    # ========== 合并高斯点参数 ==========
                    # 【场景表示模块】通过加权平均合并高斯点参数
                    # 位置：加权平均
                    # 颜色（球谐函数）：加权平均
                    # 不透明度：加权平均（在logit空间）
                    # 尺度：加权平均（在指数空间），并考虑合并后的点数(k+1)
                    # 旋转：加权平均（四元数）
                    merged_gaussians = {
                        "xyz": (self.gaussian_params["xyz"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                        "f_dc": (self.gaussian_params["f_dc"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "f_rest": (self.gaussian_params["f_rest"]['val'][selected_nn_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "opacity": inverse_sigmoid(self.gaussian_params["opacity"]['val'][selected_nn_idx, :].sigmoid() * weights).sum(dim=1),
                        "scaling": torch.log((torch.exp(self.gaussian_params["scaling"]['val'][selected_nn_idx, :]) * weights * (k+1)).sum(dim=1)),
                        "rotation": (self.gaussian_params["rotation"]['val'][selected_nn_idx, :] * weights).sum(dim=1),
                    }

                    # ========== 将旧锚点移到CPU ==========
                    # 【内存管理模块】复制参数字典并移动到CPU（释放GPU内存）
                    self.active_anchor.duplicate_param_dict()
                    self.active_anchor.to("cpu", with_keyframes=True)

                    # ========== 添加合并后的高斯点 ==========
                    # 【优化模块】将合并后的高斯点添加到优化器，同时移除细小高斯点
                    # ~small_mask: 保留非细小高斯点的掩码
                    with self.lock:
                        self.optimizer.add_and_prune(merged_gaussians, ~small_mask)

                    # ========== 创建新活跃锚点 ==========
                    # 【场景表示模块】使用合并后的高斯点创建新锚点
                    # 位置：最新关键帧的相机中心
                    # 关键帧：最近n_kept_frames个关键帧
                    self.active_anchor = Anchor(
                        self.gaussian_params,
                        self.approx_cam_centres[-1],
                        self.keyframes[-self.n_kept_frames :],
                    )
                    self.anchors.append(self.active_anchor)
                    # 更新活跃关键帧列表（新锚点的关键帧都在GPU上）
                    self.active_frames_gpu = [kf.index for kf in self.active_anchor.keyframes]
                    self.active_frames_cpu = []

                    # ========== 可视化权重设置 ==========
                    # 设置锚点混合权重（仅新锚点权重为1，其他为0）
                    self.anchor_weights = np.zeros(len(self.anchors))
                    self.anchor_weights[-1] = 1.0

                gc.collect()
                torch.cuda.empty_cache()

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0):
        """
        【评估模块】保存场景模型到磁盘
        
        将完整的场景模型保存到指定路径，包括：
        1. 所有锚点的高斯点云（PLY格式）
        2. 场景元数据（JSON格式：配置、锚点位置、关键帧信息）
        3. 测试关键帧的渲染图像
        4. COLMAP格式的相机参数和图像信息
        
        Args:
            path: 保存路径（如果为空字符串则跳过保存，仅返回指标）
            reconstruction_time: 重建耗时（秒），用于计算FPS
            n_frames: 处理的关键帧数量，用于计算FPS
            
        Returns:
            dict: 包含场景统计信息（锚点数量、关键帧数量、时间、FPS、质量指标）的字典
        """
        # ========== 计算场景指标 ==========
        # 【评估模块】收集场景统计信息
        metrics = {
            "num anchors": len(self.anchors),  # 锚点数量
            "num keyframes": len(self.keyframes),  # 关键帧数量
        }
        # 如果提供了重建时间，计算FPS
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
        # 计算渲染质量指标（PSNR、SSIM、LPIPS、位姿误差）
        metrics.update(self.evaluate(True, True, True))

        # 如果路径为空，跳过保存，仅返回指标
        if path == "":
            print("No path provided, skipping save")
            return metrics

        # ========== 保存锚点点云 ==========
        # 【场景表示模块】将每个锚点的高斯点云保存为PLY文件
        pcd_path = os.path.join(path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=True)
        for index, anchor in enumerate(self.anchors):
            anchor.save_ply(os.path.join(pcd_path, f"anchor_{index}.ply"))

        # ========== 保存场景元数据 ==========
        # 【场景表示模块】保存场景配置、锚点位置和关键帧信息
        metadata = {
            "config": {
                "width": self.width,  # 图像宽度
                "height": self.height,  # 图像高度
                "sh_degree": self.max_sh_degree,  # 球谐函数阶数
                "f": self.f,  # 焦距
            },
            "anchors": [
                {
                    "position": anchor.position.cpu().numpy().tolist(),  # 锚点位置（世界坐标系）
                }
                for anchor in self.anchors
            ],
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],  # 关键帧信息（位姿、曝光等）
        }
        # 合并指标到元数据
        metadata = {**metrics, **metadata}

        # 将元数据保存为JSON文件
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # ========== 保存测试关键帧渲染图像 ==========
        # 【评估模块】渲染所有测试关键帧并保存图像（用于可视化结果）
        self.save_test_frames(os.path.join(path, "test_images"))

        # ========== 保存COLMAP格式数据 ==========
        # 【场景表示模块】将关键帧转换为COLMAP格式（用于兼容性）
        # COLMAP格式包含相机参数（内参、外参）和图像信息
        images = {}
        cameras = {}
        colmap_save_path = os.path.join(path, "colmap")
        os.makedirs(colmap_save_path, exist_ok=True)
        for index, keyframe in enumerate(self.keyframes):
            camera, image = keyframe.to_colmap(index)
            cameras[index] = camera
            images[index] = image
        # 使用COLMAP的二进制格式保存（.bin文件）
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        return metrics

    def get_closest_keyframe(
        self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        """
        【场景表示模块】根据位置获取最近的关键帧
        
        计算给定位置到所有关键帧相机中心的距离，返回最近的count个关键帧。
        用于基于空间位置的关键帧查询（例如，查找特定区域的关键帧）。
        
        Args:
            position: 查询位置（世界坐标系）[3]
            count: 要返回的关键帧数量（默认为1）
            
        Returns:
            list[Keyframe]: 最近的关键帧列表（按距离从近到远排序）
        """
        # 计算到所有关键帧相机中心的欧氏距离
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        # 选择距离最近的count个关键帧
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]

    def finetune_epoch(self):
        """
        【优化模块】遍历所有锚点并逐个优化
        
        这是微调阶段的核心函数，用于在初始训练完成后进一步细化场景质量。
        逐个加载每个锚点到GPU，对其关键帧进行一轮优化，然后保存并卸载。
        
        策略：
        1. 按顺序处理每个锚点
        2. 将锚点加载到GPU并设置为活跃锚点
        3. 遍历锚点的所有关键帧，对每个关键帧执行一次优化步骤
        4. 更新锚点参数并卸载到CPU（节省内存）
        
        注意：这是微调模式（finetuning=True），优化时会随机选择关键帧，而不是优先选择最新帧。
        """
        # 初始化锚点混合权重（全部设为0，优化时只激活当前锚点）
        self.anchor_weights = np.zeros(len(self.anchors))
        
        # 遍历所有锚点
        for anchor_id, anchor in enumerate(self.anchors):
            # ========== 激活当前锚点 ==========
            # 【场景表示模块】设置当前锚点为活跃锚点
            self.active_anchor = anchor
            # 将锚点及其关键帧加载到GPU
            anchor.to("cuda", with_keyframes=True)
            # 将锚点的高斯参数设置为当前优化参数
            self.gaussian_params = anchor.gaussian_params
            # 激活当前锚点的权重（用于多锚点混合）
            self.anchor_weights[anchor_id] = 1
            # 重置优化器（确保梯度正确计算）
            self.reset_optimizer()

            # 注释：可选的内存优化（将前一个锚点移到CPU）
            # 实际未启用，因为可能导致频繁的GPU-CPU数据传输
            # # Ensure other anchors are on cpu to save memory
            # if anchor_id >= 1:
            #     self.anchors[anchor_id-1].to("cpu", with_keyframes=True)

            # ========== 优化当前锚点 ==========
            # 【优化模块】遍历锚点的所有关键帧，对每个关键帧执行一次优化步骤
            # finetuning=True: 微调模式，随机选择关键帧（不偏向最新帧）
            for _ in range(len(anchor.keyframes)):
                self.optimization_step(finetuning=True)

            # ========== 保存并卸载锚点 ==========
            # 【场景表示模块】将优化后的高斯参数保存到锚点
            anchor.gaussian_params = self.gaussian_params
            # 取消激活当前锚点的权重
            self.anchor_weights[anchor_id] = 0