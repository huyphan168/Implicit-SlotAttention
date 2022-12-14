import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbones import build_backbones
from src.tasks import build_task
from src.utils import *

class VannilaSlot(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbones(cfg.backbone)
        self.task_executor = build_task(cfg)
        self.num_iterations = cfg.max_iter_fwd
        self.num_slots = cfg.num_slots
        self.input_size = cfg.backbone.input_slot_size
        self.slot_size = cfg.slot_size
        self.mlp_hidden_size = cfg.mlp_hidden_size
        self.epsilon = cfg.epsilon
        self.num_heads = cfg.num_heads

        self.norm_inputs = nn.LayerNorm(cfg.backbone.input_slot_size)
        self.norm_slots = nn.LayerNorm(cfg.slot_size)
        self.norm_mlp = nn.LayerNorm(cfg.slot_size)
        
        # Linear maps for the attention module.
        self.project_q = linear(cfg.slot_size, cfg.slot_size, bias=False)
        self.project_k = linear(cfg.input_size, cfg.slot_size, bias=False)
        self.project_v = linear(cfg.input_size, cfg.slot_size, bias=False)
        
        # Slot update functions.
        self.gru = gru_cell(cfg.slot_size, cfg.slot_size)
        self.mlp = nn.Sequential(
            linear(cfg.slot_size, cfg.mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(cfg.mlp_hidden_size, cfg.slot_size))
        
        # Slot statistics.
        self.slot_mu = nn.Parameter(torch.zeros(1,1,cfg.slot_size))
        self.slot_logsigma = nn.Parameter(torch.zeros(1,1,cfg.slot_size))
    
    def forward(self, data: dict(torch.Tensor)) -> torch.Tensor:
        inputs, label = self.task_executor.preprocess(data)
        output, attn_vis = self.forward_generic(inputs)
        loss = self.task_executor.loss(output, label)
        return loss, attn_vis

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        output, attn_vis = self.forward_generic(inputs)
        return output, attn_vis
    
    def forward_generic(self, inputs: torch.Tensor) -> torch.Tensor:
        features_encoder = self.backbone(inputs)
        B, _ = features_encoder.size()
        init_slot = self.slot_mu.expand(B, self.num_slots, -1) + torch.exp(self.slot_logsigma.expand(B, self.num_slots, -1)) * torch.randn_like(
                            self.slot_mu, device=features_encoder.device)
        features_encoder = self.task_executor.feature_transform(features_encoder)
        slots, attn_vis = self.forward_slot(features_encoder, init_slot)
        output = self.task_executor.decode(slots)
        return output, attn_vis

    def forward_slot(self, inputs: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].

        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k
        
        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))                             # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)                # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_vis = attn.sum(1)                                                      # Shape: [batch_size, num_inputs, num_slots].
            
            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)                              # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)                          # Shape: [batch_size, num_slots, slot_size].
            
            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis
    

class ImplicitSLATE(nn.Module):
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