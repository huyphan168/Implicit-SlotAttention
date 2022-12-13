import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbones import build_backbones

class ImplicitSLATE(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbones(cfg.backbone)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class VannilaSlot(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbones(cfg.backbone)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class SLATE(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbones(cfg.backbone)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def build_model(cfg: dict) -> nn.Module:
    if cfg.model_type == "implicit_slate":
        return ImplicitSLATE()
    elif cfg.model_type == "vanilla_slot":
        return VannilaSlot()
    elif cfg.model_type == "slate":
        return SLATE()
    else:
        raise NotImplementedError