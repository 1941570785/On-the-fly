# 逆向去畸变，用于将训练时使用的中心化内参转换为无黑边内参
# 参考自：https://github.com/verlab/accelerated_features


import argparse
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append('.')

from scripts.rectify_tum import cam_params_dict, get_K_in_K_out
from utils import get_image_names

if __name__ == "__main__":
    """
    Rectify from what is used from training (centred principal point) 
    to something without black borders for visualization
    """
    # 解析输入路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="../fast-end2end-nvs-comparisons/results")
    args = parser.parse_args()

    for scene, cam_params in cam_params_dict.items():
        # 每个场景分别处理各方法输出
        in_scene_dir = os.path.join(args.base_dir, scene)
        out_scene_dir = os.path.join(args.base_dir, "derectified", scene)

        methods = os.listdir(in_scene_dir)
        for method_id, method in enumerate(methods):
            in_folder = os.path.join(in_scene_dir, method)
            out_folder = os.path.join(out_scene_dir, method)

            os.makedirs(out_folder, exist_ok=True)

            image_names = get_image_names(in_folder)
            h, w = cv2.imread(f"{in_folder}/{image_names[0]}").shape[:2]
            if h ==336 and method_id == 0:
                # 特定分辨率的内参缩放修正
                print("Mannually adjusting scale in intrinsics")
                cam_params[0] *= 448 / 640
                cam_params[1] *= 336 / 480
                cam_params[2] *= 448 / 640
                cam_params[3] *= 336 / 480

            # Get the matrix used for optimization
            # 训练时的内参矩阵
            K_in, K_train = get_K_in_K_out(cam_params, h, w)
            # Get a matrix that fills the full image
            # 生成用于可视化的无黑边内参矩阵
            K_out = cv2.getOptimalNewCameraMatrix(
                K_in, np.array(cam_params[4:]), (w, h), 0, (w, h), True
            )[0]

            rectify_map = cv2.initUndistortRectifyMap(
                K_train, np.array(cam_params[4:]), None, K_out, (w, h), cv2.CV_32FC2
            )[0]

            def process_image(image_name):
                # 单张图像的去畸变与裁剪
                image = cv2.imread(f"{in_folder}/{image_name}")
                dst = cv2.remap(
                    image,
                    rectify_map,
                    None,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                pad = 4
                dst = dst[pad:-pad, pad: -pad]
                cv2.imwrite(f"{out_folder}/{image_name}", dst)
                cv2.imwrite(f"{out_folder}/{os.path.splitext(image_name)[0]}.jpg", dst)

            with ThreadPoolExecutor() as executor:
                # 多线程批处理提升速度
                executor.map(process_image, image_names)
