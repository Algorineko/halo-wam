"""E1 奖励头训练：demo(成功) vs 策略rollout失败(失败) 的 posthoc 成功标注。

样本构造（固定窗口协议，与 E0 一致）:
  正样本: LIBERO demo 窗口（demo 即成功轨迹）→ label 1
  负样本: rollout h5 窗口（attrs success=0）→ label 0
  输入: ctx 末帧 + 动作 chunk → 冻结 V-JEPA 编码 ctx、动态头预测 fut → RewardHead(ctx_pool, fut_pool, task) → logit
  （动态头用 e0v5 生产权重，预测的 fut latent = "该动作块将导致的世界状态"）

用法: source scripts/env.sh && python scripts/train_e1.py --steps 3000
输出: checkpoints/e1/reward_head.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
from halo.encoder import HaloEncoder
from halo.dynamics import DynamicsEnsemble
from halo.reward import RewardHead

HALO_ROOT = "/home/tione/notebook/home/arianliu/project/halo-wam"
HALO_DATA = "/home/tione/notebook/home/arianliu/data/halo-wam"


def build_samples(max_per_side=1500):
    """返回 [(ctx(T,H,W,3) uint8, actions(A,7), task_name, label)]"""
    import h5py

    samples = []
    # 正：demo（每个任务取 1 条 demo 的 2 个窗口）
    demo_files = sorted(glob.glob(f"{HALO_DATA}/datasets/libero/libero_spatial/*.hdf5"))
    for fp in demo_files:
        task = os.path.basename(fp).replace("_demo.hdf5", "")
        with h5py.File(fp, "r") as f:
            demos = sorted(f["data"].keys())[:1]
            for d in demos:
                g = f[f"data/{d}"]
                T = g["actions"].shape[0]
                if T < 16:
                    continue
                for s in (0, T // 2):
                    ctx = g["obs/agentview_rgb"][s : s + 2]
                    acts = g["actions"][s + 1 : s + 13]
                    if len(acts) < 12:
                        continue
                    samples.append((ctx, acts.astype(np.float32), task, 1))
    n_pos = len(samples)
    # 负：rollout 失败 episode（每个任务取 2 个窗口）
    roll_files = sorted(glob.glob(f"{HALO_DATA}/datasets/rollouts/act_15k/*/*.hdf5"))
    neg_pool = []
    n_bad = 0
    for fp in roll_files:
        try:
            with h5py.File(fp, "r") as f:
                g = f["data/demo_0"]
                if int(g.attrs.get("success", 0)) != 0:
                    continue
                T = g["actions"].shape[0]
                if T < 16:
                    continue
                task_r = g.attrs.get("task", "task")
                for s in (0, T // 2):
                    ctx = g["obs/agentview_rgb"][s : s + 2]
                    acts = g["actions"][s + 1 : s + 13]
                    if len(acts) < 12:
                        continue
                    neg_pool.append((ctx, acts.astype(np.float32), task_r, 0))
        except Exception:
            n_bad += 1
    print(f"[e1] 跳过损坏 rollout 文件: {n_bad}")
    rng = np.random.RandomState(0)
    rng.shuffle(neg_pool)
    samples += neg_pool[: min(max_per_side, len(neg_pool))]
    print(f"[e1] 正样本(demo)={n_pos} 负样本(rollout失败)={len(samples)-n_pos}")
    return samples


class E1Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ctx, acts, task, label = self.samples[i]
        return (torch.from_numpy(np.ascontiguousarray(ctx)), torch.from_numpy(acts), task, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--verifier", type=str, default="checkpoints/e0v5/dyn_ens_k5_s2000.pt")
    ap.add_argument("--out", type=str, default="checkpoints/e1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    samples = build_samples()
    vocab = {t: i for i, t in enumerate(sorted({s[2] for s in samples}))}
    dl = DataLoader(E1Dataset(samples), batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)

    dev = "cuda:1"
    enc = HaloEncoder(device=dev)
    ens = DynamicsEnsemble(k=5, dim=1024, act_dim=7, n_fut_tubelets=4, n_heads=16, n_layers=6).to(dev)
    ck = torch.load(args.verifier, map_location=dev)
    ens.load_state_dict(ck["model"])
    ens.eval()
    for p in ens.parameters():
        p.requires_grad_(False)

    head = RewardHead(dim=1024, n_tasks=len(vocab)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    pos_w = torch.tensor([3.0], device=dev)  # 轻度正类加权

    import time
    it, t0 = 0, time.time()
    log = []
    while it < args.steps:
        for cam, acts, tasks, labels in dl:
            if it >= args.steps:
                break
            with torch.no_grad():
                ctx_lat = enc.encode(cam.to(dev, non_blocking=True))           # (B,1,16,16,D)
                b = ctx_lat.shape[0]
                ctx_tok = ctx_lat.reshape(b, -1, 1024)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_fut = ens(ctx_tok, acts.to(dev).to(ctx_tok.dtype))    # (K,B,Nf,D)
                fut_tok = pred_fut.float().mean(0)                             # ensemble 均值预测
            tid = torch.tensor([vocab[t] for t in tasks], device=dev)
            y = labels.float().to(dev)
            logits = head(ctx_tok.float(), fut_tok, tid)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            log.append(loss.item())
            if it % 100 == 0:
                acc = (((logits > 0).float() == y).float().mean().item())
                print(f"[e1] step {it} | bce {np.mean(log[-100:]):.4f} | acc {acc:.3f} "
                      f"| {(time.time()-t0)/max(it+1,1):.2f}s/step", flush=True)
            it += 1
    torch.save({"model": head.state_dict(), "vocab": vocab}, os.path.join(args.out, "reward_head.pt"))
    print(f"[e1] DONE -> {args.out}/reward_head.pt")


if __name__ == "__main__":
    main()
