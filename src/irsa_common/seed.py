"""Seeding and torch threading helpers."""
import numpy as np
import torch


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def set_torch_single_core():
    torch.set_num_threads(1)
