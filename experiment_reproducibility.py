from __future__ import annotations

import random

import cv2
import numpy as np
import torch


def experiment_cuda_graphs_enabled(deterministic: bool) -> bool:
    return not bool(deterministic)


def configure_experiment_reproducibility(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    experiment_seed = int(seed)
    random.seed(experiment_seed)
    np.random.seed(experiment_seed)
    torch.random.manual_seed(experiment_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(experiment_seed)
    cv2.setRNGSeed(experiment_seed)

    if not deterministic:
        return

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
