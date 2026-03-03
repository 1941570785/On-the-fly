# 工具函数实现
# 参考自：https://github.com/depth-anything/Depth-Anything-V2


import os
import re
import numpy as np
import logging

logs = set()


def init_log(name, level=logging.INFO):
    # 防止重复初始化同名 logger
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        # 分布式环境下仅 rank0 输出日志
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    # 统一日志格式
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger