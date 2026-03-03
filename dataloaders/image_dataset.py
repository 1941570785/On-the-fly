# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 图像数据集类，用于加载图像数据
# 参考自：https://github.com/verlab/accelerated_features


import cv2
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging
from argparse import Namespace

from dataloaders.read_write_model import read_model, qvec2rotmat
from utils import get_image_names


class ImageDataset:
    """
    【数据加载模块】图像数据集类
    
    这是用于从磁盘加载图像数据的主数据集类，支持：
    1. 多线程预加载图像（提高I/O效率）
    2. 加载掩码（如果有）
    3. 加载COLMAP位姿和内参（如果可用）
    4. 自动下采样大分辨率图像
    5. 位姿对齐和缩放
    
    使用方式：
    - 初始化后调用 getnext() 方法顺序获取图像
    - 图像会在后台线程中预加载，提高加载速度
    """
    def __init__(self, args: Namespace):
        """
        【数据加载模块】初始化图像数据集
        
        Args:
            args: 配置参数，包含：
                - source_path: 数据源根目录
                - images_dir: 图像目录名
                - masks_dir: 掩码目录名（可选）
                - downsampling: 下采样因子（<=0表示自动）
                - num_loader_threads: 加载线程数
                - start_at: 起始图像索引（跳过前面的图像）
                - test_hold: 测试帧间隔（每N帧取1帧作为测试）
                - use_colmap_poses: 是否使用COLMAP位姿
                - eval_poses: 是否评估位姿
        """
        # ========== 图像路径收集 ==========
        # 【数据加载模块】收集并排序所有图像路径
        self.images_dir = os.path.join(args.source_path, args.images_dir)
        self.image_name_list = get_image_names(self.images_dir)
        self.image_name_list.sort()  # 按文件名排序
        # 从指定索引开始（支持跳过前面的图像）
        self.image_name_list = self.image_name_list[args.start_at :]
        self.image_paths = [
            os.path.join(self.images_dir, image_name)
            for image_name in self.image_name_list
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        # ========== 掩码路径收集 ==========
        # 【数据加载模块】如果提供了掩码目录，收集掩码路径
        self.mask_dir = (
            os.path.join(args.source_path, args.masks_dir) if args.masks_dir else None
        )
        if self.mask_dir:
            # 掩码文件名：图像名去掉扩展名 + .png
            self.mask_paths = [
                os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")
                for image_name in self.image_name_list
            ]
            # 验证所有掩码文件都存在
            assert all(os.path.exists(mask_path) for mask_path in self.mask_paths), (
                "Not all masks exist."
            )

        # ========== 多线程预加载设置 ==========
        # 【数据加载模块】初始化多线程预加载机制
        self.downsampling = args.downsampling  # 下采样因子
        # 线程数：不超过图像数量和配置线程数的最小值
        self.num_threads = min(args.num_loader_threads, len(self.image_paths))
        self.current_index = 0  # 当前预加载索引
        self.preload_queue = Queue(maxsize=self.num_threads)  # 预加载队列
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)  # 线程池

        # ========== 图像元信息初始化 ==========
        # 【数据加载模块】为每张图像创建元信息字典
        # 包含：是否测试帧、图像名称等（位姿和内参后续从COLMAP加载）
        self.infos = {
            name: {
                "is_test": (args.test_hold > 0) and (i % args.test_hold == 0),  # 测试帧：每隔test_hold帧取一帧
                "name": name,
            }
            for i, name in enumerate(self.image_name_list)
        }

        # ========== 图像尺寸检测和自动下采样 ==========
        # 【数据加载模块】加载首张图像以确定尺寸
        # 如果分辨率过高且未指定下采样因子，自动下采样到1.5M像素
        first_image = self._load_image(self.image_paths[0])
        self.width, self.height = first_image.shape[2], first_image.shape[1]
        res = self.width * self.height  # 总像素数
        max_res = 1_500_000  # 最大分辨率：1.5百万像素（约1224x1224）
        # 如果未指定下采样且分辨率过高，自动计算下采样因子
        if self.downsampling <= 0.0 and res > max_res:
            logging.warning(
                "Large images, downsampling to 1.5 Mpx. "
                "If this is not desired, please use --downsampling=1"
            )
            # 计算下采样因子：使分辨率降至max_res
            # 开平方根是因为宽度和高度都要下采样
            self.downsampling = (res / max_res) ** 0.5
            # 重新加载首张图像（应用下采样）
            first_image = self._load_image(self.image_paths[0])
            self.width, self.height = first_image.shape[2], first_image.shape[1]

        # ========== 加载COLMAP数据 ==========
        # 【数据加载模块】从COLMAP稀疏重建结果加载相机内参和外参
        # 路径：source_path/sparse/0（COLMAP标准目录结构）
        self.load_colmap_data(os.path.join(args.source_path, "sparse/0"))

        # ========== 位姿验证和对齐 ==========
        # 【数据加载模块】检查所有图像是否都有位姿
        has_all_poses = all(
            "Rt" in self.infos[image_name] for image_name in self.image_name_list
        )
        # 如果要求使用COLMAP位姿，确保所有图像都有位姿
        if args.use_colmap_poses:
            assert has_all_poses, (
                "COLMAP poses are required but not all images have poses."
            )
            # 对齐位姿：将第一个位姿设为单位矩阵，并缩放平移
            self.align_colmap_poses()

        # 如果评估位姿但缺少部分位姿，发出警告
        if args.eval_poses and not has_all_poses:
            logging.warning(
                " Not all images have COLMAP poses, pose evaluation will be skipped."
            )

        # ========== 开始预加载 ==========
        # 【数据加载模块】启动多线程预加载（预加载前num_threads张图像）
        self.start_preloading()

    def __len__(self):
        """
        【数据加载模块】返回数据集大小
        
        Returns:
            int: 图像数量
        """
        return len(self.image_paths)

    @torch.no_grad()
    def __getitem__(self, index):
        """
        【数据加载模块】获取指定索引的图像和元信息
        
        这是PyTorch数据集的标准接口，支持：
        1. 加载图像（支持RGBA格式，自动提取Alpha通道作为掩码）
        2. 加载外部掩码（如果提供了掩码目录）
        3. 返回图像（GPU张量）和元信息字典
        
        Args:
            index: 图像索引
            
        Returns:
            tuple: (image, info)
                - image: 图像张量 [3, H, W]，float32，范围[0,1]，在GPU上
                - info: 元信息字典，包含is_test、name、mask（如果有）、Rt（如果有）、focal（如果有）
        """
        # 获取图像路径
        image_path = self.image_paths[index]
        # 加载图像（IMREAD_UNCHANGED：保持原始通道数，包括Alpha通道）
        image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
        # 获取该图像的元信息
        info = self.infos[os.path.basename(image_path)]
        
        # ========== 处理RGBA图像的Alpha通道 ==========
        # 【数据加载模块】如果图像有Alpha通道，提取作为掩码
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].cpu()  # Alpha通道作为掩码 [1, H, W]
            image = image[:3]  # RGB通道 [3, H, W]
        
        # ========== 加载外部掩码 ==========
        # 【数据加载模块】如果提供了掩码目录，从外部文件加载掩码
        # 注意：外部掩码会覆盖RGBA图像的Alpha掩码
        if self.mask_dir:
            mask = self._load_image(self.mask_paths[index])
            info["mask"] = mask[0][None]  # 掩码 [1, H, W]
        
        # 返回GPU上的图像和元信息
        return image.cuda(), info

    def _load_image(self, image_path, mode=cv2.IMREAD_COLOR):
        """
        【数据加载模块】加载并预处理单张图像
        
        执行以下操作：
        1. 从磁盘读取图像（OpenCV）
        2. 应用下采样（如果指定）
        3. 颜色空间转换（BGR -> RGB，或BGRA -> RGBA）
        4. 转换为PyTorch张量（CHW格式，float32，范围[0,1]）
        
        Args:
            image_path: 图像文件路径
            mode: OpenCV读取模式
                - cv2.IMREAD_COLOR: 读取RGB（默认）
                - cv2.IMREAD_UNCHANGED: 读取所有通道（包括Alpha）
                
        Returns:
            torch.Tensor: 图像张量 [C, H, W]，float32，范围[0,1]
                - C=3: RGB图像
                - C=4: RGBA图像
        """
        # 使用OpenCV读取图像
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        
        # ========== 下采样处理 ==========
        # 【数据加载模块】如果指定了下采样因子，对图像进行下采样
        # 使用INTER_AREA插值（适合缩小图像，保持细节）
        if self.downsampling > 0.0 and self.downsampling != 1.0:
            image = cv2.resize(
                image,
                (0, 0),  # 目标尺寸为0表示使用fx/fy缩放
                fx=1 / self.downsampling,  # 水平缩放因子
                fy=1 / self.downsampling,  # 垂直缩放因子
                interpolation=cv2.INTER_AREA,  # 区域插值（适合缩小）
            )
        
        # ========== 颜色空间转换 ==========
        # 【数据加载模块】OpenCV读取的图像是BGR格式，需要转换为RGB
        # 如果有Alpha通道（4通道），转换为RGBA；否则转换为RGB
        image = cv2.cvtColor(
            image, 
            cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB
        )
        
        # ========== 转换为PyTorch张量 ==========
        # 【数据加载模块】转换为PyTorch张量并归一化到[0,1]
        # permute(2,0,1): HWC -> CHW（通道维度放到最前）
        # /255.0: 归一化到[0,1]范围
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def _submit(self):
        """
        【数据加载模块】异步提交下一张图像的预加载任务
        
        将当前索引的图像加载任务提交到线程池，并将Future对象放入队列。
        这是多线程预加载机制的核心函数。
        """
        if self.current_index < len(self):
            # 提交任务到线程池，返回Future对象
            # Future对象包装了异步执行的结果
            self.preload_queue.put(
                self.executor.submit(self.__getitem__, self.current_index)
            )

    def start_preloading(self):
        """
        【数据加载模块】启动图像预加载
        
        在初始化时启动预加载，提前加载前num_threads张图像到队列中。
        这样可以并行加载多张图像，提高后续getnext()的响应速度。
        """
        for self.current_index in range(self.num_threads):
            self._submit()

    def getnext(self):
        """
        【数据加载模块】获取下一张图像（流式访问接口）
        
        这是数据流式处理的主接口，执行以下步骤：
        1. 从预加载队列获取下一张图像（可能阻塞等待加载完成）
        2. 提交下一张图像的预加载任务
        3. 返回当前图像和元信息
        
        使用多线程预加载的好处：
        - 图像加载是I/O密集型操作，多线程可以充分利用磁盘带宽
        - 预加载队列保证了在读取当前图像时，下一张已经在加载中
        - 减少了GPU等待I/O的时间
        
        Returns:
            tuple: (image, info)
                - image: 图像张量 [3, H, W]，float32，范围[0,1]，在GPU上
                - info: 元信息字典
        """
        # 从队列获取已完成的预加载任务结果（阻塞直到完成）
        item = self.preload_queue.get().result()
        # 更新索引
        self.current_index += 1
        # 提交下一张图像的预加载任务（保持队列填充）
        self._submit()
        return item

    def get_image_size(self):
        """
        【数据加载模块】获取图像尺寸
        
        Returns:
            tuple: (height, width) 图像高度和宽度（像素）
        """
        return self.height, self.width

    def load_colmap_data(self, colmap_folder_path):
        """
        【数据加载模块】从COLMAP稀疏重建结果加载相机内参和外参
        
        从COLMAP重建结果（cameras.bin、images.bin、points3D.bin）读取：
        1. 相机内参（焦距、主点等）
        2. 图像外参（位姿：旋转和平移）
        
        将读取的数据存储到self.infos字典中，供后续使用。
        
        Args:
            colmap_folder_path: COLMAP模型文件夹路径（通常为sparse/0）
            
        注意：
            - 仅支持单个相机模型（所有图像使用相同相机）
            - 支持的相机模型：PINHOLE（针孔模型，fx和fy独立）和SIMPLE_PINHOLE（简单针孔，fx=fy）
            - 焦距会根据图像尺寸自动缩放（从COLMAP原始尺寸缩放到当前图像尺寸）
        """
        try:
            # 读取COLMAP模型文件（cameras.bin、images.bin、points3D.bin）
            cameras, images, _ = read_model(colmap_folder_path)
        except Exception as e:
            logging.warning(
                f" Failed to read COLMAP files in {colmap_folder_path}: {e}"
            )
            return
        
        # ========== 相机模型验证 ==========
        # 【数据加载模块】验证仅使用单个相机模型
        if len(cameras) != 1:
            logging.warning(" Only supports one camera")
        # 获取相机模型类型
        model = list(cameras.values())[0].model
        if model != "PINHOLE" and model != "SIMPLE_PINHOLE":
            logging.warning(" Unexpected camera model: " + model)

        # ========== 处理每张图像 ==========
        for image_id, image in images.items():
            # 获取该图像使用的相机参数
            camera = cameras[image.camera_id]

            # ========== 内参计算 ==========
            # 【数据加载模块】计算并缩放焦距
            # COLMAP中焦距的存储方式：
            # - PINHOLE: params = [fx, fy, cx, cy]（fx和fy可能不同）
            # - SIMPLE_PINHOLE: params = [f, cx, cy]（fx = fy = f）
            focal_x = camera.params[0]  # 水平焦距（像素）
            focal_y = camera.params[1] if camera.model == "PINHOLE" else focal_x  # 垂直焦距
            # 使用平均焦距（虽然后续会重新计算）
            focal = (focal_x + focal_y) / 2
            # 根据当前图像尺寸缩放焦距（从COLMAP原始尺寸缩放到当前尺寸）
            # 假设主点在图像中心，焦距按比例缩放
            focal = focal_x * self.width / camera.width

            # ========== 外参计算 ==========
            # 【数据加载模块】构建4x4位姿矩阵（世界到相机的变换）
            Rt = np.eye(4, dtype=np.float32)  # 初始化为单位矩阵
            # 旋转部分：将COLMAP的四元数转换为旋转矩阵
            Rt[:3, :3] = qvec2rotmat(image.qvec)
            # 平移部分：直接使用COLMAP的平移向量
            Rt[:3, 3] = image.tvec

            # ========== 存储到元信息 ==========
            # 【数据加载模块】将位姿和内参存储到对应图像的元信息中
            name = os.path.basename(image.name)  # 提取图像文件名
            # 只有当图像在数据集中时才存储（可能在COLMAP中但不在数据集中）
            if image.name in self.infos:
                # 转换为GPU张量并存储
                self.infos[name]["Rt"] = torch.tensor(Rt, device="cuda")  # 位姿矩阵 [4, 4]
                self.infos[name]["focal"] = torch.tensor([focal], device="cuda").float()  # 焦距 [1]

    def align_colmap_poses(self):
        """
        【数据加载模块】对齐COLMAP位姿
        
        执行以下对齐操作：
        1. 将第一个位姿设为单位矩阵（作为参考坐标系）
        2. 缩放所有位姿的平移部分（使场景尺度合理）
        
        对齐策略：
        - 计算前6张图像的相机中心，估计相对平移的平均尺度
        - 将平均尺度归一化到0.1单位，使场景尺度统一
        - 将第一个位姿逆变换应用到所有位姿，使第一个位姿成为单位矩阵
        
        这样做的好处：
        - 消除COLMAP重建的任意坐标系统
        - 统一场景尺度，便于优化器学习率设置
        - 第一个关键帧作为参考，其他位姿相对第一个位姿定义
        
        注意：
        - 这是不可逆的变换，仅在使用COLMAP位姿作为初始值时调用
        - 对齐后的位姿仍可通过优化进一步调整
        """
        # ========== 估计场景尺度 ==========
        # 【数据加载模块】从前6张图像的相机中心估计相对平移尺度
        centres = []
        for idx in range(6):
            # 计算相机中心：位姿逆矩阵的平移部分
            # Rt是从世界到相机的变换，其逆的平移部分是相机在世界坐标系中的位置
            centres.append(self.infos[self.image_name_list[idx]]["Rt"].inverse()[:3, 3])
        centres = torch.stack(centres)  # [6, 3]
        # 计算相邻图像的相对平移（向量差）
        rel_ts = centres[:-1] - centres[1:]  # [5, 3]
        
        # 计算平均相对平移的尺度，并将其归一化到0.1单位
        # 0.1是经验值，使场景尺度适合优化器
        scale = 0.1 / rel_ts.norm(dim=-1).mean()

        # ========== 应用对齐变换 ==========
        # 【数据加载模块】将第一个位姿设为参考坐标系
        # 计算第一个位姿的逆矩阵（用于对齐变换）
        inv_first_Rt = self.infos[self.image_name_list[0]]["Rt"].inverse()
        
        # 对所有图像的位姿应用对齐变换
        for info in self.infos.values():
            # 对齐位姿：右乘第一个位姿的逆，使第一个位姿变成单位矩阵
            # 结果：所有位姿都是相对于第一个位姿定义的
            info["Rt"] = info["Rt"] @ inv_first_Rt
            # 缩放平移部分：应用估计的尺度因子
            info["Rt"][:3, 3] *= scale
