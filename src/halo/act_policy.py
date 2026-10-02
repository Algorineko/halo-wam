"""极简 ACT (Action Chunking Transformer) 策略——用于 E2 的被验证 policy。

设计：CVAE 变体省略（mlp 压缩 z=cls），最小可用版：
  输入: 双相机帧 (2,3,128,128) + 8 维状态 + 任务嵌入
  backbone: 4 层 ViT 小块 + 平均池化（不加载预训练，128px 输入，轻量可训）
  解码: transformer decoder（queries=chunk_size 个动作 token）→ 7 维动作 chunk
训法: L2 回退 demo 动作 chunk（无 CVAE/无 l1 也够基线用）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TinyViT(nn.Module):
    def __init__(self, dim=256, depth=4, patch=16, img=128, in_ch=3):
        super().__init__()
        n = (img // patch) ** 2
        self.patch = nn.Conv2d(in_ch, dim, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim * 2, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # (B,3,128,128)
        h = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.norm(h)


class HaloACT(nn.Module):
    def __init__(self, dim=256, chunk=50, n_tasks=120, act_dim=7):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim
        self.vis_cam1 = TinyViT(dim)
        self.vis_cam2 = TinyViT(dim)
        self.state_proj = nn.Linear(8, dim)
        self.task_emb = nn.Embedding(n_tasks, dim)
        self.fuse = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mem_norm = nn.LayerNorm(dim)
        dec = nn.TransformerDecoderLayer(dim, 8, dim * 2, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers=4)
        self.q_pos = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.head = nn.Linear(dim, act_dim)

    def encode(self, cam1, cam2, state, task_id):
        h = self.fuse(torch.cat([
            self.vis_cam1(cam1).mean(1),
            self.vis_cam2(cam2).mean(1),
            self.state_proj(state),
        ], dim=-1)) + self.task_emb(task_id)
        return self.mem_norm(h.unsqueeze(1))

    def forward(self, cam1, cam2, state, task_id):
        """返回 (B, chunk, act_dim) 动作 chunk。"""
        mem = self.encode(cam1, cam2, state, task_id)
        q = self.q_pos.expand(cam1.shape[0], -1, -1)
        return self.head(self.decoder(q, mem))

    @torch.no_grad()
    def select_action(self, cam1, cam2, state, task_id, queue: list):
        """队列式推理：空队列时整 chunk 推理，逐个弹出（评测用）。"""
        if not queue:
            queue.extend(self.forward(cam1, cam2, state, task_id)[0].tolist())
        return torch.tensor(queue.pop(0))
