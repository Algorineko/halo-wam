"""冻结 V-JEPA 2 编码器封装：视频帧 → tubept latent 网格。

预处理（对齐官方 VJEPA2VideoProcessor）：
  最短边 resize 到 292 → 中心裁剪 256×256 → /255 → ImageNet 归一化。
模型：ViT-L，patch 16，tubelet 2 → 每个输出 token 覆盖 2 帧 × 16px × 16px。
输出 token 网格 (T/2, 16, 16)，dim=1024。

注意：token 网格的时间维排序假设为 row-major (T', H', W')，E0 中用
temporal-swap 一致性检验确认（scripts/validate_encoder.py）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import VJEPA2Model

DEFAULT_MODEL_DIR = "/home/tione/notebook/home/arianliu/data/halo-wam/models/vjepa2-vitl-fpc64-256"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HaloEncoder(nn.Module):
    """frozen V-JEPA 2 视频编码器。

    用法:
        enc = HaloEncoder(device="cuda:0")
        grid = enc.encode_frames(frames)   # frames: (T,H,W,3) uint8 tensor/ndarray
        # grid: (T/2, 16, 16, 1024) bf16
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = (
            VJEPA2Model.from_pretrained(model_dir, torch_dtype=dtype, attn_implementation="sdpa")
            .to(device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """(B,T,H,W,3) uint8 → (B,T,3,256,256) 归一化，全在 GPU 上。"""
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        x = frames.to(device=self.device, dtype=self.dtype).permute(0, 4, 1, 2, 3)  # B,C,T,H,W
        b, c, t, h, w = x.shape
        # 最短边 292（等比）
        if h < w:
            nh, nw = 292, round(w * 292 / h / 2) * 2  # 偶数化，中心裁剪无歧义
        else:
            nh, nw = round(h * 292 / w / 2) * 2, 292
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
        # 中心裁剪 256×256
        top, left = (nh - 256) // 2, (nw - 256) // 2
        x = x[:, :, top : top + 256, left : left + 256]
        x = x.reshape(b, c, t, 256, 256) / 255.0
        x = (x - self.mean) / self.std
        return x.permute(0, 2, 1, 3, 4).contiguous()  # B,T,C,H,W（VJEPA2 帧维在前）

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B,T,H,W,3) uint8 → (B,T',16,16,D) latent 网格，T'=T//2。"""
        x = self.preprocess(frames)
        out = self.model(pixel_values_videos=x)
        b, n, d = out.last_hidden_state.shape
        grid = out.last_hidden_state.reshape(b, n // 256, 16, 16, d)  # (T',H',W') row-major 假设
        return grid

    def encode_frames(self, frames) -> torch.Tensor:
        """单 clip 便捷接口：(T,H,W,3) → (T',16,16,D)。"""
        if not torch.is_tensor(frames):
            import numpy as np

            frames = torch.from_numpy(np.ascontiguousarray(frames))
        return self.encode(frames)[0]
