# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 简单 KNN 实现
# 参考：https://github.com/graphdeco-inria/gaussian-splatting/blob/main/setup.py


from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    # Windows 下关闭特定警告
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={
                # 编译参数（NVCC + C++）
                "nvcc": [
                    "-allow-unsupported-compiler",
                    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1"
                ],
                "cxx": cxx_compiler_flags
            })
        ],
    cmdclass={
        # 使用 PyTorch build_ext
        'build_ext': BuildExtension
    }
)