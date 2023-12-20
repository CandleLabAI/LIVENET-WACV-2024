# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------
"""
basic helper functions
"""
import numpy as np
import random
import torch
import os


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(prefer='cuda', gpu_index=0):
    if prefer == 'cuda' and torch.cuda.is_available():
        # If CUDA/GPU is preferred and available, return the selected CUDA device
        return torch.device(f'cuda:{gpu_index}')
    else:
        # If CUDA is not preferred or not available, return the CPU device
        return torch.device('cpu')

