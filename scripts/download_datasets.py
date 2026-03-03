# 下载数据集，用于下载数据集
# 参考自：https://github.com/verlab/accelerated_features


import os
import urllib.request
import zipfile
import argparse

def download_and_extract(url, extract_to):
    """
    Download a zip file from a URL and extract its contents to a specified directory.
    
    Args:
        url (str): The URL of the zip file to download.
        extract_to (str): The directory where the contents will be extracted.
    """
    # 创建输出目录
    os.makedirs(extract_to, exist_ok=True)

    # 本地 zip 文件名
    local_filename = os.path.join(extract_to, os.path.basename(url))

    # 下载数据集压缩包
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, local_filename)

    # 解压到目标目录
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        print(f"Extracting {local_filename}...")
        zip_ref.extractall(extract_to)
    # 解压完成后删除压缩包
    os.remove(local_filename)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Download and extract datasets.")
    parser.add_argument("--datasets", nargs='+', default=["TUM", "MipNeRF360", "StaticHikes"], 
                        help="List of datasets to download. Default is TUM, MipNeRF360, StaticHikes.")
    parser.add_argument("--out_dir", type=str, default="data", 
                        help="Output base directory for the datasets.")
    args = parser.parse_args()
    
    # Define the base URL and dataset names
    # 数据集基地址
    base_url = "https://repo-sam.inria.fr/nerphys/on-the-fly-nvs/datasets"    

    # Download and extract each dataset
    # 逐个下载并解压
    for dataset in args.datasets:
        download_and_extract(f"{base_url}/{dataset}.zip", args.out_dir)
    
    print("Done")