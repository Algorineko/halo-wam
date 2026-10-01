"""V-JEPA 2 latent 特性验证（E0 协议设计依据）。

发现（2026-10-02）: V-JEPA 2 无绝对位置嵌入，用 RoPE + 双向注意力 →
token latent 依赖 clip 上下文与相对时序位置。同帧在不同偏移下 latent 不同
(cos≈0.50)，因此世界模型必须采用【固定窗口相对编码协议】：
  context 窗口与 future 窗口独立编码，各自从相对位置 0 开始。

本脚本验证三件事：
  1. 确定性: 同一 clip 编码两次 → cos = 1.0
  2. 上下文敏感度: [f2..f9] 单独编码 vs 作为 [f0..f9] 后缀编码，
     共享 tubept (帧2-9) 的 latent 偏移量（报告值，协议依据）
  3. 时间局部性信号: 同 clip 内相邻时间步 vs 远距时间步的 latent 相似度差
     （动态头可学的信号强度）

用法: source scripts/env.sh && python scripts/validate_encoder.py [libero_h5_file]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
from halo.encoder import HaloEncoder


def libero_clip(h5_path: str, t: int = 10):
    import h5py

    with h5py.File(h5_path, "r") as f:
        return f["data/demo_0/obs/agentview_rgb"][:t]


def main():
    if len(sys.argv) > 1:
        src = libero_clip(sys.argv[1])
    else:
        raise SystemExit("需要 LIBERO h5 路径作为参数")

    enc = HaloEncoder(device="cuda:0")

    def cos(u, v):
        u, v = u.float().flatten(), v.float().flatten()
        return torch.nn.functional.cosine_similarity(u, v.unsqueeze(0)).item()

    # 1. 确定性
    g1 = enc.encode_frames(torch.from_numpy(src[:8]))
    g2 = enc.encode_frames(torch.from_numpy(src[:8]))
    det = cos(g1, g2)
    print(f"[1] 确定性 cos = {det:.4f}  {'✅' if det > 0.999 else '❌'}")

    # 2. 上下文敏感度: 短 clip [f2..f9] vs 长 clip [f0..f9] 的对应 tubept
    short = enc.encode_frames(torch.from_numpy(src[2:10]))          # T'=4, 相对位置 0..3
    long = enc.encode_frames(torch.from_numpy(src[:10]))            # T'=5, 帧(2,3)起在相对位置 1..4
    shared_short, shared_long = short[0:4], long[1:5]               # 同像素内容的 tubept
    ctx = cos(shared_short, shared_long)
    print(f"[2] 上下文敏感度（同内容跨上下文 cos，越低越敏感）= {ctx:.4f}")

    # 3. 时间局部性: 长 clip 内 grid 相邻 vs 远距时间步相似度
    g = long  # (5,16,16,D)
    near = np.mean([cos(g[i], g[i + 1]) for i in range(4)])
    far = np.mean([cos(g[i], g[j]) for i in range(4) for j in range(i + 2, 5)])
    print(f"[3] 时间局部性: 相邻步 cos={near:.4f} vs 远距 cos={far:.4f}（差 {near-far:.4f}，越大动态头信号越强）")

    # 4.（顺带）旧的对齐检验在固定相对位置协议下的正确形态：
    #    context=[f0..f7] 与 future=[f8..f15] 各自独立编码，tubept 均从相对位置 0 起
    if src.shape[0] >= 16:
        g16 = enc.encode_frames(torch.from_numpy(src[:16]))         # T'=8
        ctx_w, fut_w = g16[:4], g16[4:8]                            # 相对位置各自 0..3 的窗口
        print(f"[4] 16 帧单编码的前/后窗口相似度 = {cos(ctx_w, fut_w):.4f}（后续 E0 用双窗口独立编码协议）")


if __name__ == "__main__":
    main()
