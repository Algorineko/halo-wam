"""诊断：动态头到底用不用动作？（E0 验收失败后的根因分析）

条件对比（每个窗口）：
  A. 真实动作        —— 预测精度基线
  B. 错配动作(同batch) —— 软负样本
  C. 随机均匀动作     —— 硬负样本
  D. 零动作          —— 硬负样本
  E. 真实动作×2      —— 幅度敏感性
  F. 丢弃动作 token   —— 动作信息量消融（架构级）

指标：pred vs 真实 future latent 的 cos（越高=预测准）；ensemble std（不确定性）。
若 A 的 cos 不显著高于 B/C/D → 动作被忽略，需架构干预。
用法: source scripts/env.sh && python scripts/diag_action_sensitivity.py [ckpt]
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
from halo.data.libero_h5 import LiberoWindowDataset, collate
from halo.dynamics import DynamicsEnsemble
from halo.encoder import HaloEncoder

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/e0/dyn_ens_k5_s2000.pt"


@torch.no_grad()
def main():
    dev = "cuda:0"
    enc = HaloEncoder(device=dev)
    ck = torch.load(CKPT, map_location=dev)
    ens = DynamicsEnsemble(k=5, dim=1024, act_dim=7, n_fut_tubelets=4, n_heads=16, n_layers=6).to(dev)
    ens.load_state_dict(ck["model"])
    ens.eval()

    ds = LiberoWindowDataset(None, 2, 8, 4, samples_per_demo=2)  # 与 v5/e0_prod 训练协议一致
    if len(sys.argv) > 2:  # 第二参数：只评测留出任务
        with open(sys.argv[2]) as f:
            pats = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        ds.files = [fp for fp in ds.files if any(p in os.path.basename(fp) for p in pats)]
        ds.index = [(fi, d, t) for fi, d, t in ds.index
                    if any(p in os.path.basename(ds.files[fi]) for p in pats)]
        ds.samples = [s for s in ds.samples if any(
            p in os.path.basename(ds.files[s[0]]) for p in pats)]
        print(f"[diag] 留出任务模式: {len(ds.files)} files, {len(ds)} samples")
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

    stats = {k: {"cos": [], "std": []} for k in ["A_true", "B_mismatch", "C_random", "D_zero", "E_double", "F_noact"]}
    n_batches = 0
    for batch in dl:
        ctx_lat = enc.encode(batch["ctx"])
        fut_lat = enc.encode(batch["fut"])
        b = ctx_lat.shape[0]
        ctx_tok = ctx_lat.reshape(b, -1, 1024)
        fut_tok = fut_lat.reshape(b, -1, 1024)
        acts = torch.stack(batch["actions"]).to(dev).to(ctx_tok.dtype)

        perm = torch.randperm(b)
        conds = {
            "A_true": acts,
            "B_mismatch": acts[perm],
            "C_random": (torch.rand_like(acts) * 2 - 1),
            "D_zero": torch.zeros_like(acts),
            "E_double": acts * 2,
        }
        for name, a in conds.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = ens(ctx_tok, a)  # (K,B,Nf,D)
            mean_pred = preds.float().mean(0)
            cos = torch.nn.functional.cosine_similarity(mean_pred, fut_tok.float(), dim=-1).mean().item()
            std = preds.float().std(0).mean().item()
            stats[name]["cos"].append(cos)
            stats[name]["std"].append(std)
        n_batches += 1
        if n_batches >= 8:
            break

    print(f"[diag] ckpt={CKPT}  (12 任务子集, {n_batches} batches)")
    print(f"[diag] {'条件':<14} {'pred-cos↑':>10} {'ens-std':>9}")
    for k, v in stats.items():
        if not v["cos"]:
            continue
        print(f"[diag] {k:<14} {np.mean(v['cos']):>10.4f} {np.mean(v['std']):>9.4f}")
    a = np.mean(stats['A_true']['cos']); b_ = np.mean(stats['B_mismatch']['cos']); c = np.mean(stats['C_random']['cos'])
    print(f"[diag] 动作敏感性: A-B={a-b_:+.4f}  A-C={a-c:+.4f}  （>0.02 才算动作起作用）")


if __name__ == "__main__":
    main()
