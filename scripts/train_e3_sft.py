"""E3-A 蒸馏 SFT：HaloACT 在 HALO 选中的动作块上微调（监督 = verifier-chosen chunk）。

数据: gen_e3_data.py 产出的 train.h5（demo ctx + HALO top chunk + std 门控已滤 10%）
验证: 按 demo episode 留出，报 val MSE（在线评测另跑）
基线: 从 act60k 权重热启（chosen chunk 接近策略自身分布，从头训浪费）

用法: source scripts/env.sh && python scripts/train_e3_sft.py --steps 8000
输出: checkpoints/act_e3/best.pt
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
from halo.act_policy import HaloACT

HALO_DATA = "/home/tione/notebook/home/arianliu/data/halo-wam"


class DistillDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.ctx = f["ctx"][:]
            self.state = f["state"][:]
            self.chosen = f["chosen"][:]
            self.task = [t.decode() if isinstance(t, bytes) else t for t in f["task"][:]]
            self.ep = [t.decode() if isinstance(t, bytes) else t for t in f["ep"][:]]
        self.vocab = {}
        for t in sorted(set(self.task)):
            self.vocab[t] = len(self.vocab)

    def __len__(self):
        return len(self.ctx)

    def __getitem__(self, i):
        return (torch.from_numpy(self.ctx[i]), torch.from_numpy(self.state[i]),
                self.task[i], torch.from_numpy(self.chosen[i]))


def make_collate(vocab):
    def collate(b):
        ctx = torch.stack([x[0] for x in b])
        st = torch.stack([x[1] for x in b])
        ch = torch.stack([x[3] for x in b])
        tid = torch.tensor([vocab[x[2]] for x in b])
        return ctx, st, tid, ch
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=f"{HALO_DATA}/datasets/e3_distill/train.h5")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5, help="热启微调小学习率")
    ap.add_argument("--init", type=str, default="checkpoints/act60k/best.pt")
    ap.add_argument("--out", type=str, default="checkpoints/act_e3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = DistillDataset(args.data)
    eps = sorted(set(ds.ep))
    rng = np.random.RandomState(0)
    rng.shuffle(eps)
    val_eps = set(eps[: max(1, len(eps) // 10)])
    tr_idx = [i for i in range(len(ds)) if ds.ep[i] not in val_eps]
    va_idx = [i for i in range(len(ds)) if ds.ep[i] in val_eps]
    print(f"[e3sft] samples {len(ds)} (train {len(tr_idx)} / val {len(va_idx)}), "
          f"tasks {len(ds.vocab)}, val episodes {len(val_eps)}")

    dev = "cuda:1"
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model = HaloACT(dim=ck["args"].get("dim", 256), chunk=ck["args"]["chunk"],
                    n_tasks=ck["args"].get("n_tasks", 130), act_dim=7).to(dev)
    model.load_state_dict(ck["model"])
    print(f"[e3sft] init from {args.init} (val-mse {ck.get('val', '?')})")

    dl = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=args.bs, shuffle=True,
                    num_workers=4, collate_fn=make_collate(ds.vocab), drop_last=True, persistent_workers=True)
    dl_val = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=args.bs, shuffle=False,
                        num_workers=2, collate_fn=make_collate(ds.vocab))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    def to_dev(ctx, st, tid):
        img = ctx.to(dev).float().permute(0, 1, 4, 2, 3) / 255.0  # (B,2,3,H,W)
        return img[:, 0], img[:, 1], st.to(dev), tid.to(dev)

    it, t0, best = 0, time.time(), 1e9
    while it < args.steps:
        for ctx, st, tid, ch in dl:
            if it >= args.steps:
                break
            c1, c2, stt, tt = to_dev(ctx, st, tid)
            pred = model(c1, c2, stt, tt)
            loss = torch.nn.functional.mse_loss(pred, ch.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if it % 200 == 0:
                print(f"[e3sft] step {it} | mse {loss.item():.5f} | "
                      f"{(time.time()-t0)/max(it+1,1):.2f}s/step", flush=True)
            if it % 2000 == 0 and it > 0:
                model.eval()
                vls = []
                with torch.no_grad():
                    for ctx, st, tid, ch in dl_val:
                        c1, c2, stt, tt = to_dev(ctx, st, tid)
                        vls.append(torch.nn.functional.mse_loss(
                            model(c1, c2, stt, tt), ch.to(dev)).item())
                vl = float(np.mean(vls))
                print(f"[e3sft] step {it} | VAL-mse {vl:.5f}", flush=True)
                if vl < best:
                    best = vl
                    torch.save({"model": model.state_dict(), "vocab": ds.vocab,
                                "args": dict(ck["args"]) | {"chunk": 13, "dim": 256, "n_tasks": model.task_emb.weight.shape[0]}},
                               os.path.join(args.out, "best.pt"))
                model.train()
            it += 1
    torch.save({"model": model.state_dict(), "vocab": ds.vocab,
                "args": dict(ck["args"]) | {"chunk": 13, "dim": 256, "n_tasks": model.task_emb.weight.shape[0]}},
               os.path.join(args.out, "last.pt"))
    print(f"[e3sft] DONE best-val {best:.5f} -> {args.out}")


if __name__ == "__main__":
    main()
