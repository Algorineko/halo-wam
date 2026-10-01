"""V-JEPA 2 在 BW1000 (DCU) 上的冻结推理冒烟测试。

用法: source scripts/env.sh && python scripts/smoke_vjepa2.py [--frames 16|64]
验证: 加载本地权重 → bf16 前向 → 输出形状 + 延迟。SDPA 注意力（DTK 无 flash-attn）。
"""
import argparse
import time

import torch
from transformers import VJEPA2Model

MODEL_DIR = "/home/tione/notebook/home/arianliu/data/halo-wam/models/vjepa2-vitl-fpc64-256"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(f"[smoke] loading {MODEL_DIR} (bf16, sdpa) ...")
    t0 = time.time()
    model = VJEPA2Model.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] loaded in {time.time()-t0:.1f}s, params: {n_params/1e6:.1f}M")

    # 随机视频片段 (B, T, C, H, W) —— VJEPA2 约定帧维在前；256px，tubelet 2×16×16
    x = torch.randn(args.batch, args.frames, 3, 256, 256, device=args.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(pixel_values_videos=x)
    hidden = out.last_hidden_state
    print(f"[smoke] input {tuple(x.shape)} -> hidden {tuple(hidden.shape)}, dtype {hidden.dtype}")

    # 延迟：warmup 3 + 计时 10
    with torch.no_grad():
        for _ in range(3):
            model(pixel_values_videos=x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            model(pixel_values_videos=x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
    print(f"[smoke] latency: {dt*1000:.0f} ms/forward (B={args.batch}, T={args.frames})")
    print("[smoke] PASS ✅")


if __name__ == "__main__":
    main()
