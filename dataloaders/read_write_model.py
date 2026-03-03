# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# 读写模型类，用于读写COLMAP模型文件
# 参考自：https://github.com/verlab/accelerated_features


import argparse
import collections
import os
import struct

import numpy as np

# ========== 数据结构定义 ==========

# 【数据结构】相机模型定义
# COLMAP支持的相机模型类型，包含模型ID、名称和参数数量
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

# 【数据结构】相机参数
# 存储相机的内参信息
# - id: 相机ID
# - model: 相机模型名称（如"PINHOLE"）
# - width: 图像宽度（像素）
# - height: 图像高度（像素）
# - params: 相机模型参数数组（根据模型类型不同，参数数量和含义不同）
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

# 【数据结构】图像信息（基类）
# 存储图像的位姿和特征点信息
# - id: 图像ID
# - qvec: 四元数表示的旋转（qw, qx, qy, qz），表示世界到相机的旋转
# - tvec: 平移向量（tx, ty, tz），表示相机在世界坐标系中的位置
# - camera_id: 关联的相机ID
# - name: 图像文件名
# - xys: 2D特征点坐标数组 [N, 2]
# - point3D_ids: 每个2D点对应的3D点ID数组 [N]（-1表示未三角化）
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

# 【数据结构】3D点信息
# 存储稀疏重建的3D点云信息
# - id: 3D点ID
# - xyz: 3D坐标（X, Y, Z）
# - rgb: 颜色（R, G, B，0-255）
# - error: 重投影误差
# - image_ids: 观测到该3D点的图像ID数组
# - point2D_idxs: 在每个图像中的2D点索引数组（与image_ids对应）
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    """
    【数据结构】图像信息类（扩展基类）
    
    继承BaseImage，添加四元数到旋转矩阵的转换方法。
    """
    def qvec2rotmat(self):
        """
        【坐标变换】将四元数转换为旋转矩阵
        
        Returns:
            np.ndarray: 3x3旋转矩阵（世界到相机）
        """
        return qvec2rotmat(self.qvec)


# ========== 相机模型定义 ==========

# 【相机模型】COLMAP支持的所有相机模型类型
# 每种模型有不同的参数数量和含义：
# - SIMPLE_PINHOLE (3): f, cx, cy（简单针孔模型，fx=fy）
# - PINHOLE (4): fx, fy, cx, cy（针孔模型，fx和fy独立）
# - SIMPLE_RADIAL (4): f, cx, cy, k（简单径向畸变）
# - RADIAL (5): f, cx, cy, k1, k2（径向畸变）
# - OPENCV (8): fx, fy, cx, cy, k1, k2, p1, p2（OpenCV模型）
# - OPENCV_FISHEYE (8): fx, fy, cx, cy, k1, k2, k3, k4（鱼眼相机）
# - FULL_OPENCV (12): fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6（完整OpenCV）
# - FOV (5): fx, fy, cx, cy, omega（视场角模型）
# - SIMPLE_RADIAL_FISHEYE (4): f, cx, cy, k（简单径向鱼眼）
# - RADIAL_FISHEYE (5): f, cx, cy, k1, k2（径向鱼眼）
# - THIN_PRISM_FISHEYE (12): 薄棱镜鱼眼模型（完整畸变）
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}

# 【查找表】通过模型ID查找相机模型定义
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)

# 【查找表】通过模型名称查找相机模型定义
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


# ========== 二进制文件读写辅助函数 ==========

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """
    【文件IO】从二进制文件读取并解包字节
    
    从打开的二进制文件读取指定数量的字节，并按照格式字符串解包。
    
    Args:
        fid: 文件对象（二进制读取模式）
        num_bytes: 要读取的字节数（必须是2、4、8的组合，如2, 6, 16, 30等）
        format_char_sequence: 格式字符串，支持：
            - c: char (1字节)
            - e: float16 (2字节)
            - f: float32 (4字节)
            - d: float64 (8字节)
            - h: int16 (2字节)
            - H: uint16 (2字节)
            - i: int32 (4字节)
            - I: uint32 (4字节)
            - l: int64 (8字节)
            - L: uint64 (8字节)
            - q: int64 (8字节)
            - Q: uint64 (8字节)
        endian_character: 字节序字符
            - <: 小端序（默认，Intel）
            - >: 大端序（网络序）
            - =: 本地字节序
            - @: 本地字节序和对齐
            
    Returns:
        tuple: 解包后的值元组
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """
    【文件IO】打包数据并写入二进制文件
    
    将数据按照格式字符串打包为字节，然后写入二进制文件。
    
    Args:
        fid: 文件对象（二进制写入模式）
        data: 要写入的数据
            - 如果是单个值：直接打包
            - 如果是列表或元组：解包后打包
        format_char_sequence: 格式字符串（与read_next_bytes相同）
        endian_character: 字节序字符（默认为小端序）
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


# ========== 相机数据读写函数 ==========

def read_cameras_text(path):
    """
    【文件IO】从文本文件读取相机参数
    
    读取COLMAP文本格式的相机文件（cameras.txt）。
    文件格式：
        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        CAMERA_ID MODEL WIDTH HEIGHT PARAM1 PARAM2 ...
        
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        path: 相机文本文件路径（cameras.txt）
        
    Returns:
        dict: 相机字典 {camera_id: Camera对象}
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    【文件IO】从二进制文件读取相机参数
    
    读取COLMAP二进制格式的相机文件（cameras.bin）。
    二进制格式：
        - 8字节：相机数量（uint64）
        - 对每个相机：
            - 24字节：camera_id (int32), model_id (int32), width (uint64), height (uint64)
            - 8*num_params字节：相机参数数组（double，num_params取决于模型）
            
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        path_to_model_file: 相机二进制文件路径（cameras.bin）
        
    Returns:
        dict: 相机字典 {camera_id: Camera对象}
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    【文件IO】将相机参数写入文本文件
    
    将相机参数写入COLMAP文本格式（cameras.txt）。
    
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        cameras: 相机字典 {camera_id: Camera对象}
        path: 输出文件路径
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    【文件IO】将相机参数写入二进制文件
    
    将相机参数写入COLMAP二进制格式（cameras.bin）。
    
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        cameras: 相机字典 {camera_id: Camera对象}
        path_to_model_file: 输出文件路径
        
    Returns:
        dict: 输入的相机字典（用于链式调用）
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


# ========== 图像数据读写函数 ==========

def read_images_text(path):
    """
    【文件IO】从文本文件读取图像信息
    
    读取COLMAP文本格式的图像文件（images.txt）。
    文件格式（每张图像两行）：
        第1行：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        第2行：X1 Y1 POINT3D_ID1 X2 Y2 POINT3D_ID2 ...
        
    QW QX QY QZ: 四元数表示的旋转（世界到相机）
    TX TY TZ: 平移向量（相机在世界坐标系中的位置）
    POINT3D_ID: -1表示未三角化的2D点
        
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        path: 图像文本文件路径（images.txt）
        
    Returns:
        dict: 图像字典 {image_id: Image对象}
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    【文件IO】从二进制文件读取图像信息
    
    读取COLMAP二进制格式的图像文件（images.bin）。
    二进制格式：
        - 8字节：图像数量（uint64）
        - 对每个图像：
            - 64字节：image_id (int32), qvec[4] (double), tvec[3] (double), camera_id (int32)
            - 变长：图像名称（null终止的UTF-8字符串）
            - 8字节：2D点数（uint64）
            - 24*num_points2D字节：每个2D点（x double, y double, point3D_id int64）
            
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        path_to_model_file: 图像二进制文件路径（images.bin）
        
    Returns:
        dict: 图像字典 {image_id: Image对象}
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images, path):
    """
    【文件IO】将图像信息写入文本文件
    
    将图像信息写入COLMAP文本格式（images.txt）。
    
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        images: 图像字典 {image_id: Image对象}
        path: 输出文件路径
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    【文件IO】将图像信息写入二进制文件
    
    将图像信息写入COLMAP二进制格式（images.bin）。
    
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        images: 图像字典 {image_id: Image对象}
        path_to_model_file: 输出文件路径
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


# ========== 3D点数据读写函数 ==========

def read_points3D_text(path):
    """
    【文件IO】从文本文件读取3D点信息
    
    读取COLMAP文本格式的3D点文件（points3D.txt）。
    文件格式（每行一个3D点）：
        POINT3D_ID X Y Z R G B ERROR IMAGE_ID1 POINT2D_IDX1 IMAGE_ID2 POINT2D_IDX2 ...
        
    TRACK: 观测到该3D点的图像ID和2D点索引对列表
        
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        path: 3D点文本文件路径（points3D.txt）
        
    Returns:
        dict: 3D点字典 {point3D_id: Point3D对象}
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    【文件IO】从二进制文件读取3D点信息
    
    读取COLMAP二进制格式的3D点文件（points3D.bin）。
    二进制格式：
        - 8字节：3D点数量（uint64）
        - 对每个3D点：
            - 43字节：point3D_id (uint64), xyz[3] (double), rgb[3] (uint8), error (double)
            - 8字节：track长度（uint64）
            - 8*track_length字节：track数组（每个元素：image_id int32, point2D_idx int32）
            
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        path_to_model_file: 3D点二进制文件路径（points3D.bin）
        
    Returns:
        dict: 3D点字典 {point3D_id: Point3D对象}
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_text(points3D, path):
    """
    【文件IO】将3D点信息写入文本文件
    
    将3D点信息写入COLMAP文本格式（points3D.txt）。
    
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        points3D: 3D点字典 {point3D_id: Point3D对象}
        path: 输出文件路径
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    【文件IO】将3D点信息写入二进制文件
    
    将3D点信息写入COLMAP二进制格式（points3D.bin）。
    
    参考COLMAP源码：src/colmap/scene/reconstruction.cc
    
    Args:
        points3D: 3D点字典 {point3D_id: Point3D对象}
        path_to_model_file: 输出文件路径
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


# ========== 统一模型读写接口 ==========

def detect_model_format(path, ext):
    """
    【文件IO】检测COLMAP模型格式
    
    检查指定路径下是否存在指定格式的COLMAP模型文件。
    
    Args:
        path: 模型文件夹路径
        ext: 文件扩展名（".bin" 或 ".txt"）
        
    Returns:
        bool: 如果存在所有必需文件（cameras, images, points3D）则返回True
    """
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        return True

    return False


def read_model(path, ext=""):
    """
    【文件IO】读取完整的COLMAP模型
    
    从指定路径读取COLMAP稀疏重建模型，包括相机、图像和3D点。
    如果未指定扩展名，会自动检测格式（优先二进制格式）。
    
    Args:
        path: 模型文件夹路径（包含cameras、images、points3D文件）
        ext: 文件扩展名（".bin"、".txt"或空字符串表示自动检测）
        
    Returns:
        tuple: (cameras, images, points3D)
            - cameras: 相机字典 {camera_id: Camera对象}
            - images: 图像字典 {image_id: Image对象}
            - points3D: 3D点字典 {point3D_id: Point3D对象}
            
    Raises:
        SystemExit: 如果无法检测格式且ext为空
    """
    # 如果未指定扩展名，自动检测格式
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"  # 优先使用二进制格式
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    # 根据格式调用相应的读写函数
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    """
    【文件IO】写入完整的COLMAP模型
    
    将相机、图像和3D点写入指定路径的COLMAP格式文件。
    
    Args:
        cameras: 相机字典 {camera_id: Camera对象}
        images: 图像字典 {image_id: Image对象}
        points3D: 3D点字典 {point3D_id: Point3D对象}
        path: 输出文件夹路径
        ext: 文件扩展名（".bin"或".txt"，默认二进制格式）
        
    Returns:
        tuple: 输入的三个字典（用于链式调用）
    """
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


# ========== 坐标变换工具函数 ==========

def qvec2rotmat(qvec):
    """
    【坐标变换】将四元数转换为旋转矩阵
    
    将四元数（COLMAP格式：qw, qx, qy, qz）转换为3x3旋转矩阵。
    使用标准四元数到旋转矩阵的转换公式。
    
    Args:
        qvec: 四元数数组 [qw, qx, qy, qz]
        
    Returns:
        np.ndarray: 3x3旋转矩阵（世界到相机的变换）
        
    注意：
        COLMAP的四元数格式：qvec = [qw, qx, qy, qz]
        这与某些库的格式（如PyTorch的[w,x,y,z]）一致
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    """
    【坐标变换】将旋转矩阵转换为四元数
    
    将3x3旋转矩阵转换为四元数（COLMAP格式：qw, qx, qy, qz）。
    使用Shepperd方法：通过特征值分解求解。
    
    Args:
        R: 3x3旋转矩阵
        
    Returns:
        np.ndarray: 四元数数组 [qw, qx, qy, qz]
        
    注意：
        - 返回的四元数保证qw >= 0（规范化）
        - 如果qw为负，会将整个四元数取反（q和-q表示相同的旋转）
    """
    # 提取旋转矩阵元素
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    # 构建4x4矩阵K（用于特征值分解）
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    # 特征值分解：最大特征值对应的特征向量就是四元数
    eigvals, eigvecs = np.linalg.eigh(K)
    # 提取四元数（重排序：[w, x, y, z]）
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    # 确保qw >= 0（如果为负则取反）
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def main():
    """
    【命令行工具】COLMAP模型格式转换工具的主函数
    
    提供命令行接口，支持：
    1. 读取COLMAP模型（文本或二进制格式）
    2. 转换为另一种格式（文本或二进制）
    3. 打印模型统计信息
    
    使用方法：
        python read_write_model.py --input_model <path> --input_format <.bin|.txt> 
                                   --output_model <path> --output_format <.bin|.txt>
    """
    parser = argparse.ArgumentParser(
        description="Read and write COLMAP binary and text models"
    )
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument(
        "--input_format",
        choices=[".bin", ".txt"],
        help="input model format",
        default="",
    )
    parser.add_argument("--output_model", help="path to output model folder")
    parser.add_argument(
        "--output_format",
        choices=[".bin", ".txt"],
        help="output model format",
        default=".txt",
    )
    args = parser.parse_args()

    # 读取模型
    cameras, images, points3D = read_model(
        path=args.input_model, ext=args.input_format
    )

    # 打印统计信息
    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))

    # 如果指定了输出路径，写入模型
    if args.output_model is not None:
        write_model(
            cameras,
            images,
            points3D,
            path=args.output_model,
            ext=args.output_format,
        )


if __name__ == "__main__":
    main()