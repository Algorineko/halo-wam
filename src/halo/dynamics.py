"""E0 核心：action-conditioned latent 动态头（在冻结 V-JEPA 2 latent 空间预测未来）。

设计（协议见 docs/progress.md）：
  输入: context 窗口 latent tokens (B, Nc, D)（T'c×16×16, 相对位置 0 起）
        + 动作序列 (B, A, 7)（context 末帧 → future 末帧之间的 OSC_POSE 动作）
  输出: future 窗口 latent tokens 预测 (B, Nf, D)（相对位置 0 起，与独立编码的
        future 窗口 latent 对齐）

结构：future 可学习位置 queries 交叉注意到 [ctx tokens; 动作 tokens]，
      K 个独立头组成 deep ensemble（分歧 = 认知不确定性，E1 的幻觉门控用）。

损失（JEPA 风格）：cosine + smooth-L1（对 latent 直接回归，目标来自冻结编码器）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamicsHead(nn.Module):
    """单个动态头。n_ctx_tubelets = T'c（时间步数），每步 256 个空间 token。"""

    def __init__(
        self,
        dim: int = 1024,
        act_dim: int = 7,
        n_fut_tubelets: int = 4,   # T'f（future 8 帧 / tubelet 2）
        n_heads: int = 16,
        n_layers: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_fut_tokens = n_fut_tubelets * 256

        # future 位置 queries（时间×空间可分离：时间嵌入 + 空间嵌入）
        self.t_pos = nn.Parameter(torch.randn(n_fut_tubelets, dim) * 0.02)
        sp = self._space_grid(16, 16)  # (256, 2) 归一化坐标
        self.space_pos = nn.Parameter(torch.randn(1, 256, dim) * 0.02)  # 共享，加坐标投影
        self.space_proj = nn.Linear(2, dim)

        # 动作 token 化
        self.act_proj = nn.Sequential(nn.Linear(act_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.act_pos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # 顺序位置（可换 sinusoidal）

        # 记忆 token = [ctx tokens; act tokens]，无额外投影（同维）
        # queries = future 位置嵌入
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=int(dim * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.out_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    @staticmethod
    def _space_grid(h: int, w: int) -> torch.Tensor:
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        g = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)
        return g  # (h*w, 2)

    def future_queries(self, batch: int, device, dtype) -> torch.Tensor:
        q = self.t_pos[:, None, :] + self.space_proj(self._space_grid(16, 16).to(device))[None] + self.space_pos  # (T'f,256,D)
        q = q.reshape(self.n_fut_tokens, self.dim).unsqueeze(0).expand(batch, -1, -1)
        return q.to(device=device, dtype=dtype)

    def forward(self, ctx_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ctx_tokens: (B, Nc, D)；actions: (B, A, act_dim) padded 对齐。
        返回 (B, Nf, D) future latent 预测。"""
        b = ctx_tokens.shape[0]
        act_tokens = self.act_proj(actions) + self.act_pos.to(actions.dtype)  # (B,A,D)
        memory = torch.cat([ctx_tokens, act_tokens], dim=1)
        q = self.future_queries(b, ctx_tokens.device, ctx_tokens.dtype)
        h = self.decoder(q, memory)
        return self.out_mlp(self.norm(h))


class DynamicsEnsemble(nn.Module):
    """K 个独立 LatentDynamicsHead。分歧（token 级 std）= 认知不确定性。"""

    def __init__(self, k: int = 5, **head_kwargs):
        super().__init__()
        self.heads = nn.ModuleList([LatentDynamicsHead(**head_kwargs) for _ in range(k)])

    def forward(self, ctx_tokens: torch.Tensor, actions: torch.Tensor):
        """返回 (K, B, Nf, D)。"""
        return torch.stack([h(ctx_tokens, actions) for h in self.heads])

    @staticmethod
    def jepa_loss(preds: torch.Tensor, target: torch.Tensor) -> dict:
        """preds: (K,B,Nf,D)，target: (B,Nf,D)（冻结编码器，无梯度）。
        cosine + smooth-L1 组合。"""
        tgt = target.detach().unsqueeze(0).expand_as(preds)
        l_cos = 1 - F.cosine_similarity(preds, tgt, dim=-1).mean()
        l_l1 = F.smooth_l1_loss(preds, tgt)
        return {"loss": l_cos + l_l1, "cos": l_cos.detach(), "l1": l_l1.detach()}

    @torch.no_grad()
    def epistemic_std(self, ctx_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """(B, Nf, D)→返回 (B, Nf) token 级 ensemble 标准差（幻觉检测信号）。"""
        preds = self(ctx_tokens, actions)  # (K,B,Nf,D)
        return preds.std(dim=0).mean(dim=-1)
