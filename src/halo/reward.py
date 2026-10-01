"""E1 奖励/成功头：预测 (ctx 隐状态, 动作) → 任务成功的 logit。

输入设计（与 E0 世界模型共享冻结编码器）：
  ctx_latent: (B, 256, D) 单帧 tubept latent
  pred_fut:   (B, Nf, D) 世界模型预测的未来 latent（或 rollout 末步）
  task_id:    (B,) 任务索引 → 可学习任务嵌入
输出: (B,) success logit。

训练数据：自产标注（评测农场 rollout + check_success）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RewardHead(nn.Module):
    def __init__(self, dim: int = 1024, n_tasks: int = 200, hidden: int = 512):
        super().__init__()
        self.task_emb = nn.Embedding(n_tasks, dim)
        self.ctx_pool = nn.Sequential(nn.Linear(dim, dim), nn.GELU())       # token 池化前投影
        self.fut_pool = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx_latent: torch.Tensor, pred_fut: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """返回 (B,) success logit。"""
        c = self.ctx_pool(ctx_latent).mean(dim=1)      # (B,D)
        f = self.fut_pool(pred_fut).mean(dim=1)        # (B,D)
        t = self.task_emb(task_id)                     # (B,D)
        return self.trunk(torch.cat([c, f, t], dim=-1)).squeeze(-1)

    @staticmethod
    def bce_loss(logits: torch.Tensor, success: torch.Tensor) -> dict:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, success.float())
        with torch.no_grad():
            acc = ((logits > 0).float() == success.float()).float().mean()
        return {"loss": loss, "acc": acc.detach()}
