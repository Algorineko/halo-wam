"""E2-lite：基于认知不确定性的动作块验证器（无需奖励头）。

原理：给定当前观测（单帧）与候选动作块，E0 世界模型 ensemble 预测未来 latent；
ensemble 分歧（token 级 std）低 = 所有头一致认为该动作的后果合理 → 可信动作。
错配/离谱动作会触发高分歧（E0-v5 已验证：注入 ratio 1.72）。

用法（评测时）:
    scorer = UncertaintyScorer(encoder, ensemble, device="cuda:0")
    scores = scorer.score(obs_frame_uint8, action_chunks)  # (N,) 越低越可信
    best = action_chunks[scores.argmin()]
"""
from __future__ import annotations

import torch

from .dynamics import DynamicsEnsemble
from .encoder import HaloEncoder


class UncertaintyScorer:
    def __init__(self, encoder: HaloEncoder, ensemble: DynamicsEnsemble, device: str = "cuda:0"):
        self.enc = encoder
        self.ens = ensemble.to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(
        self,
        obs_frame: torch.Tensor,          # (H,W,3) uint8 当前帧
        action_chunks: torch.Tensor,      # (N, A, act_dim) 候选动作块
        return_pred_cos: bool = False,
    ) -> torch.Tensor:
        """返回 (N,) 不确定性分数（ensemble std，越低越可信）。"""
        f = obs_frame.to(self.device)
        if f.dim() == 3:
            f = f.unsqueeze(0)  # (1,H,W,3) 单帧 ctx
        # 每个候选共享同一 ctx latent（只编码一次）
        ctx_lat = self.enc.encode(f)                       # (1,1,16,16,D)
        ctx_tok = ctx_lat.reshape(1, -1, 1024).expand(action_chunks.shape[0], -1, -1)
        acts = action_chunks.to(self.device).to(ctx_tok.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            std = self.ens.epistemic_std(ctx_tok, acts)    # (N, Nf)
        return std.mean(dim=-1)
