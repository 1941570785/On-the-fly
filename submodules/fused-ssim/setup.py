from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


# Default fallback architectures
fallback_archs = [
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
]

nvcc_args = [
    "-O3",
    "--maxrregcount=32",
    "--use_fast_math",
]

detected_arch = None

if torch.cuda.is_available():
    try:
        device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device)
        arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
        
        # Print to multiple outputs
        # 输出检测到的架构，便于调试
        arch_msg = f"Detected GPU architecture: {arch}"
        print(arch_msg)
        print(arch_msg, file=sys.stderr, flush=True)
        
        nvcc_args.append(f"-arch={arch}")
        detected_arch = arch
    except Exception as e:
        # 检测失败时回退到多架构编译
        error_msg = f"Failed to detect GPU architecture: {e}. Falling back to multiple architectures."
        print(error_msg)
        print(error_msg, file=sys.stderr, flush=True)
        nvcc_args.extend(fallback_archs)
else:
    # 无 CUDA 时也使用多架构编译参数
    cuda_msg = "CUDA not available. Falling back to multiple architectures."
    print(cuda_msg)
    print(cuda_msg, file=sys.stderr, flush=True)
    nvcc_args.extend(fallback_archs)

# Create a custom class that prints the architecture information
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        # 构建时打印最终架构信息
        print("\n" + "="*50)
        print(arch_info)
        print("="*50 + "\n")
        super().build_extensions()

setup(
    name="fused_ssim",
    # Python 包注册
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
                "ssim.cu",
                "ext.cpp"],
            extra_compile_args={
                # 编译器优化参数
                "cxx": ["-O3"],
                "nvcc": nvcc_args + [
                    "-allow-unsupported-compiler",
                    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1"
                ]
            }
        )
    ],
    cmdclass={
        # 使用自定义 build_ext 输出架构信息
        'build_ext': CustomBuildExtension
    }
)

# Print again at the end of setup.py execution
final_msg = f"Setup completed. NVCC args: {nvcc_args}"
print(final_msg)