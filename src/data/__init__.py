import numpy as np
import torch
from .clevr import ClevrDataset
from .coco import CoCoDataset
from .shapestacks import SSDataset

def build_dataset(cfg: dict, mode: str) -> torch.utils.data.Dataset:
    if cfg.dataset == "clevr":
        dataset = ClevrDataset(cfg, mode)
    elif cfg.dataset == "coco":
        dataset = CoCoDataset(cfg, mode)
    elif cfg.dataset == "shapestacks":
        dataset = SSDataset(cfg, mode)
    else:
        raise NotImplementedError
    return dataset
