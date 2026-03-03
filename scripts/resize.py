# 图像下采样，用于将图像下采样到指定分辨率
# 参考自：https://github.com/verlab/accelerated_features


import argparse
import cv2 
import os
import concurrent.futures

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_folder', default="../data/Meta/university2")
    parser.add_argument('--downsampling', default=1.5)
    args = parser.parse_args()

    # 输入/输出目录
    in_folder = f"{args.scene_folder}/images"
    out_folder = f"{args.scene_folder}/images_{args.downsampling}"
    os.makedirs(out_folder, exist_ok=True)

    # 收集待处理图像文件名
    image_names = [file for file in os.listdir(in_folder) if file.endswith('.png') or file.endswith('.jpg')]
    
    def process_image(image_name):
        # 读取并按比例缩放，保持清晰度
        image = cv2.imread(f"{in_folder}/{image_name}")
        dst = cv2.resize(image, (0, 0), fx=1/args.downsampling, fy=1/args.downsampling, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out_folder}/{image_name}", dst, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 多线程批处理
        executor.map(process_image, image_names)