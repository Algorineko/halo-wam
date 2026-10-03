"""HaloACT-D：冻结 DINOv2-small 特征骨干 + token 级 memory 的 ACT 策略。

与 HaloACT（TinyViT 从零训、双相机各压成 1 个 mean-pool 向量）的两点差别：
  1. 视觉特征来自冻结 DINOv2-small（21M，ImageNet-1k 蒸馏，patch 14）——LIBERO 级
     demo 量（10 任务 × 50 条）从头训 ViT 学不到的表征先验
  2. 不做 mean-pool：每相机 224px → 16×16=256 个 patch token，双相机 512 token
     + state + task 共同构成 memory，decoder 直接 cross-attend（解除单 token 瓶颈）

可训练参数只有投影/decoder/嵌入 ≈ 4M；DINOv2 冻结不存梯度。
输入仍为 (B,3,H,W) 0-1 float；内部 F.interpolate 到 224 并做 DINOv2 归一化。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


class DINOFeatures(nn.Module):
    """冻结 DINOv2 patch-token 提取器（双相机共享一个实例）。"""

    def __init__(self):
        super().__init__()
        from transformers import Dinov2Model
        self.net = Dinov2Model.from_pretrained("facebook/dinov2-small")
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(DINO_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(DINO_STD).view(1, 3, 1, 1))

    def forward(self, x):  # (B,3,H,W) 0-1
        z = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        z = (z - self.mean) / self.std
        out = self.net(pixel_values=z)
        return out.last_hidden_state[:, 1:]  # 去CLS → (B, 256, 384)


class HaloACTD(nn.Module):
    def __init__(self, dim=256, chunk=13, n_tasks=130, act_dim=7, dec_layers=4):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim
        self.dino = DINOFeatures()
        self.vis_proj = nn.Linear(384, dim)
        self.cam_emb = nn.Embedding(2, dim)  # 区分主/腕相机
        self.state_proj = nn.Linear(8, dim)
        self.task_emb = nn.Embedding(n_tasks, dim)
        self.mem_norm = nn.LayerNorm(dim)
        dec = nn.TransformerDecoderLayer(dim, 8, dim * 2, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers=dec_layers)
        self.q_pos = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.head = nn.Linear(dim, act_dim)

    def encode(self, cam1, cam2, state, task_id):
        B = cam1.shape[0]
        h1 = self.vis_proj(self.dino(cam1)) + self.cam_emb.weight[0]
        h2 = self.vis_proj(self.dino(cam2)) + self.cam_emb.weight[1]
        extra = torch.stack([self.state_proj(state), self.task_emb(task_id)], dim=1)  # (B,2,dim)
        return self.mem_norm(torch.cat([h1, h2, extra], dim=1))  # (B, 256+256+2, dim)

    def forward(self, cam1, cam2, state, task_id):
        """返回 (B, chunk, act_dim) 动作 chunk。"""
        mem = self.encode(cam1, cam2, state, task_id)
        q = self.q_pos.expand(cam1.shape[0], -1, -1)
        return self.head(self.decoder(q, mem))

    @torch.no_grad()
    def select_action(self, cam1, cam2, state, task_id, queue: list):
        if not queue:
            queue.extend(self.forward(cam1, cam2, state, task_id)[0].tolist())
        return torch.tensor(queue.pop(0))
