"""E0 训练：action-conditioned 动态头（LIBERO demos，冻结 V-JEPA 2）。

用法:
  source scripts/env.sh && python scripts/train_e0.py --steps 200 --bs 8   # 冒烟
  source scripts/env.sh && python scripts/train_e0.py --steps 4000         # 完整

输出: checkpoints/e0/dyn_ens_k{K}_s{steps}.pt；末尾跑幻觉注入自检
（打乱动作 → ensemble 分歧应上升）。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
from halo.data.libero_h5 import LiberoWindowDataset, collate
from halo.dynamics import DynamicsEnsemble
from halo.encoder import HaloEncoder


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ctx-frames", type=int, default=8)
    p.add_argument("--fut-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/e0")
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = LiberoWindowDataset(None, args.ctx_frames, args.fut_frames, args.stride, samples_per_demo=4)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True, persistent_workers=True)
    print(f"[e0] dataset: {len(ds.files)} files, {len(ds)} samples")

    enc = HaloEncoder(device=args.device)
    n_ctx_tp, n_fut_tp = args.ctx_frames // 2, args.fut_frames // 2
    ens = DynamicsEnsemble(
        k=args.k, dim=1024, act_dim=7, n_fut_tubelets=n_fut_tp, n_heads=16, n_layers=6
    ).to(args.device)
    n_params = sum(p.numel() for p in ens.parameters())
    print(f"[e0] ensemble: K={args.k}, {n_params/1e6:.1f}M params total")

    opt = torch.optim.AdamW(ens.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))))

    it, t0 = 0, time.time()
    log_acc = {"loss": [], "cos": []}
    while it < args.steps:
        for batch in dl:
            if it >= args.steps:
                break
            with torch.no_grad():
                ctx_lat = enc.encode(batch["ctx"])          # (B,T'c,16,16,D) bf16
                fut_lat = enc.encode(batch["fut"])          # (B,T'f,16,16,D)
            b = ctx_lat.shape[0]
            ctx_tok = ctx_lat.reshape(b, -1, 1024)          # 保持 bf16（DTK 上 SDPA 走显式矩阵，bf16 省一半）
            fut_tok = fut_lat.reshape(b, -1, 1024)
            acts = torch.stack(batch["actions"]).to(args.device).to(ctx_tok.dtype)  # stride=0 时等长 (B,A,7)

            # heads 前向/损失在 bf16 autocast（params fp32 master，优化器状态 fp32）
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = ens(ctx_tok, acts)                  # (K,B,Nf,D)
                losses = DynamicsEnsemble.jepa_loss(preds, fut_tok)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(ens.parameters(), 1.0)
            opt.step(); sched.step()

            log_acc["loss"].append(losses["loss"].item())
            log_acc["cos"].append(losses["cos"].item())
            if it % 20 == 0:
                el = time.time() - t0
                print(f"[e0] step {it:5d} | loss {np.mean(log_acc['loss'][-20:]):.4f} "
                      f"| cos {np.mean(log_acc['cos'][-20:]):.4f} | lr {sched.get_last_lr()[0]:.2e} "
                      f"| {el/ max(it+1,1):.2f}s/step", flush=True)
            it += 1

    ckpt_path = os.path.join(args.out, f"dyn_ens_k{args.k}_s{args.steps}.pt")
    torch.save({"model": ens.state_dict(), "args": vars(args), "loss_tail": log_acc["loss"][-50:]}, ckpt_path)
    print(f"[e0] saved {ckpt_path}")

    # ---- 幻觉注入自检：打乱 batch 内动作序列 → ensemble 分歧应上升 ----
    with torch.no_grad():
        ep = []
        for batch in dl:
            ctx_lat = enc.encode(batch["ctx"]); fut_lat = enc.encode(batch["fut"])
            b = ctx_lat.shape[0]
            ctx_tok = ctx_lat.reshape(b, -1, 1024).float()
            acts = torch.stack(batch["actions"]).to(args.device)
            perm = torch.randperm(b)
            std_right = ens.epistemic_std(ctx_tok, acts).mean().item()
            std_wrong = ens.epistemic_std(ctx_tok, acts[perm]).mean().item()  # 他人动作
            ep.append((std_right, std_wrong))
            if len(ep) >= 5:
                break
    sr = np.mean([a for a, _ in ep]); sw = np.mean([w for _, w in ep])
    print(f"[e0] 幻觉注入自检: 正确动作 ensemble-std={sr:.4f} vs 错配动作={sw:.4f} "
          f"(ratio {sw/max(sr,1e-8):.2f}, >1 即方向正确)")


if __name__ == "__main__":
    main()
