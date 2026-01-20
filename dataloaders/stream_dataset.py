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

# 流式数据集类，用于从视频流中加载图像
# 参考自：https://github.com/verlab/accelerated_features


import argparse
import queue
import time
from threading import Thread

import cv2
import torch
from torch import Tensor


class StreamDataset:
    """
    【数据加载模块】流式数据集类
    
    用于从视频流（摄像头、RTSP、HTTP流等）实时加载图像数据。
    特点：
    1. 后台线程持续拉流，保证数据的实时性
    2. 使用队列缓存最新帧，丢弃旧帧（只保留最新一帧）
    3. 自动重连机制（连接失败时定期重试）
    4. 支持下采样和格式转换
    
    使用场景：
    - 实时SLAM/3D重建
    - 视频流处理
    - 在线推理
    
    接口：
    - getnext(): 获取下一帧图像
    - get_image_size(): 获取图像尺寸
    - stop(): 停止流并释放资源
    """
    def __init__(self, video_url: str, downsampling: float, retry_delay: float = 1.0):
        """
        【数据加载模块】初始化流式数据集
        
        Args:
            video_url: 视频流URL，支持：
                - 摄像头索引（如"0"表示默认摄像头）
                - RTSP流（如"rtsp://...")
                - HTTP流（如"http://...")
                - 本地视频文件路径
            downsampling: 下采样因子
                - > 1.0: 缩小图像（如1.5表示缩小到原来的2/3）
                - = 1.0: 不下采样
                - <= 0.0: 不下采样
            retry_delay: 重连延迟（秒），连接失败后等待多长时间再重试
        """
        # ========== 基本参数 ==========
        self.video_url = video_url  # 视频流URL
        self.downsampling = downsampling  # 下采样因子

        # ========== 帧队列和线程管理 ==========
        # 【数据加载模块】使用队列缓存最新帧（maxsize=1：只保留最新一帧）
        # 策略：当有新帧到达时，丢弃旧帧，只保留最新帧
        # 这样可以避免队列堆积，保证始终获取最新的帧
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True  # 线程运行标志
        self.retry_delay = retry_delay  # 重连延迟
        self.cap = None  # OpenCV VideoCapture对象（初始为None，延迟连接）

        # ========== 启动后台拉流线程 ==========
        # 【数据加载模块】创建并启动后台线程持续拉流
        # daemon=True: 守护线程，主程序退出时自动终止
        # 后台线程会持续从视频流读取帧并放入队列
        self.capture_thd = Thread(target=self._capture_frames, daemon=True)
        self.capture_thd.start()

        # ========== 帧计数 ==========
        self.num_frames = 0  # 已读取的帧数
    
    def _connect(self):
        """
        【数据加载模块】连接到视频流
        
        初始化OpenCV VideoCapture对象并连接到视频流。
        如果已经连接，则不执行任何操作。
        
        连接失败时不会抛出异常，而是静默返回，
        由_capture_frames在重试逻辑中处理。
        """
        # 如果已经连接，直接返回
        if self.cap is not None:
            return

        # 尝试打开视频流
        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print(f"Failed to open video stream: {self.video_url}")
            return

        print("Connected to camera stream.")
        self.cap = cap

    def _capture_frames(self) -> None:
        """
        【数据加载模块】后台线程函数：持续捕获视频帧
        
        这是运行在后台线程中的主循环，持续执行以下操作：
        1. 检查连接状态，如果未连接则尝试连接
        2. 从视频流读取帧
        3. 将最新帧放入队列（丢弃旧帧）
        
        特性：
        - 自动重连：连接失败或读取失败时释放资源，等待后重连
        - 只保留最新帧：队列中始终只有一帧，新帧到达时丢弃旧帧
        - 非阻塞：读取失败不会影响主线程
        
        这个函数会一直运行，直到self.running被设为False。
        """
        while self.running:
            # ========== 连接检查 ==========
            # 【数据加载模块】如果未连接，尝试连接并等待
            if self.cap is None:
                self._connect()  # 尝试连接
                time.sleep(self.retry_delay)  # 等待一段时间后重试
                continue

            # ========== 读取帧 ==========
            # 【数据加载模块】从视频流读取一帧
            ret, frame = self.cap.read()
            if not ret:
                # 读取失败：释放资源并标记为未连接（下次循环会重连）
                print("Failed to read frame from stream.")
                self.cap.release()
                self.cap = None
                continue

            # ========== 更新队列 ==========
            # 【数据加载模块】将新帧放入队列，丢弃旧帧
            # 策略：只保留最新一帧，这样可以避免队列堆积和内存占用
            # 对于实时应用，旧帧通常不重要，只需要最新帧
            if not self.frame_queue.empty():
                self.frame_queue.get()  # 丢弃旧帧
            self.frame_queue.put(frame)  # 放入新帧

    def getnext(self) -> tuple[Tensor, dict]:
        """
        【数据加载模块】获取下一帧图像（流式访问接口）
        
        这是数据流式处理的主接口，执行以下操作：
        1. 从队列获取最新帧（阻塞直到有帧可用）
        2. 颜色空间转换（BGR -> RGB）
        3. 下采样（如果指定）
        4. 转换为PyTorch张量（CHW格式，float32，范围[0,1]，在GPU上）
        
        Returns:
            tuple: (image, info)
                - image: 图像张量 [3, H, W]，float32，范围[0,1]，在GPU上
                - info: 元信息字典，包含is_test=False（流式数据不使用测试帧）
                
        注意：
            - 这个函数会阻塞，直到队列中有帧可用
            - 返回的总是队列中的最新帧（旧帧已被丢弃）
            - 如果后台线程停止，这个函数会一直阻塞
        """
        # ========== 从队列获取帧 ==========
        # 【数据加载模块】阻塞等待直到队列中有帧可用
        frame = self.frame_queue.get(block=True)
        self.num_frames += 1  # 更新帧计数
        
        # ========== 颜色空间转换 ==========
        # 【数据加载模块】OpenCV读取的图像是BGR格式，需要转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ========== 下采样处理 ==========
        # 【数据加载模块】如果指定了下采样因子，对图像进行下采样
        # 使用INTER_AREA插值（适合缩小图像，保持细节）
        if self.downsampling > 0.0 and self.downsampling != 1.0: 
            frame = cv2.resize(
                frame,
                (0, 0),  # 目标尺寸为0表示使用fx/fy缩放
                fx=1 / self.downsampling,  # 水平缩放因子
                fy=1 / self.downsampling,  # 垂直缩放因子
                interpolation=cv2.INTER_AREA,  # 区域插值（适合缩小）
            )
        
        # ========== 转换为PyTorch张量 ==========
        # 【数据加载模块】转换为PyTorch张量并归一化到[0,1]
        # permute(2,0,1): HWC -> CHW（通道维度放到最前）
        # cuda(): 移到GPU
        # float() / 255.0: 转换为float32并归一化到[0,1]范围
        image = torch.from_numpy(frame).permute(2, 0, 1).cuda().float() / 255.0
        
        # 流式数据不使用测试帧（所有帧都用于训练）
        info = {"is_test": False}
        return image, info

    def get_image_size(self):
        """
        【数据加载模块】获取图像尺寸
        
        通过读取一帧图像来获取分辨率（高度和宽度）。
        读取后会将帧计数回退，因为这只是为了获取尺寸。
        
        Returns:
            tuple: (height, width) 图像高度和宽度（像素）
            
        注意：
            - 这个方法会消耗一帧（从队列中取出）
            - 帧计数会被回退，所以不会影响总的帧计数统计
        """
        # 读取一帧用于获取分辨率
        frame = self.getnext()[0]
        # 回退帧计数（因为这个帧只是用来获取尺寸的）
        self.num_frames -= 1
        # 返回图像尺寸（CHW格式：shape[-2]是高度，shape[-1]是宽度）
        return frame.shape[-2], frame.shape[-1]

    def stop(self) -> None:
        """
        【数据加载模块】停止视频流并释放资源
        
        执行以下操作：
        1. 设置停止标志（使后台线程退出循环）
        2. 释放VideoCapture资源
        3. 等待后台线程结束
        
        注意：
            - 应该在程序退出前调用此方法
            - 如果不调用，后台线程会一直运行（因为是daemon线程）
        """
        # 设置停止标志（使后台线程退出循环）
        self.running = False
        # 释放视频捕获资源
        if self.cap is not None:
            self.cap.release()
        # 等待后台线程结束（阻塞直到线程退出）
        self.capture_thd.join()
    
    def __len__(self):
        """
        【数据加载模块】返回数据集大小
        
        流式数据集的大小是未知的（无限流），因此返回一个任意大的数字。
        这主要是为了兼容需要数据集大小的接口。
        
        Returns:
            int: 一个很大的数字（100,000,000）
        """
        # 返回一个任意大的数字，因为流式数据的长度是未知的
        return 100_000_000  

# ========== 示例用法 ==========

if __name__ == "__main__":
    """
    【示例代码】流式数据集的命令行使用示例
    
    演示如何使用StreamDataset从视频流读取图像并显示。
    
    使用方法：
        python stream_dataset.py --source_path <video_url> --downsampling <factor>
        
    示例：
        # 从摄像头读取
        python stream_dataset.py --source_path 0 --downsampling 1.5
        
        # 从RTSP流读取
        python stream_dataset.py --source_path rtsp://... --downsampling 1.0
    """
    parser = argparse.ArgumentParser(description="Stream Dataset")
    parser.add_argument("-s", "--source_path", type=str, help="video stream URL")
    parser.add_argument("--downsampling", type=float, default=1.5, help="downsampling factor")
    args = parser.parse_args()

    # 创建流式数据集
    stream = StreamDataset(args.source_path, args.downsampling)
    try:
        # 持续读取并显示帧
        while True:
            image, info = stream.getnext()
            # 示例处理：显示图像
            # 转换为HWC格式并转为BGR（OpenCV使用BGR）
            # [..., ::-1]: RGB -> BGR（最后一个维度反转）
            display_image = image.permute(1, 2, 0).cpu().numpy()[..., ::-1]
            cv2.imshow("Stream", display_image)
            cv2.waitKey(1)  # 等待1ms（非阻塞）
    except KeyboardInterrupt:
        # 用户中断（Ctrl+C）时停止流
        print("\nStopping stream...")
        stream.stop()
        print("Stream stopped.")