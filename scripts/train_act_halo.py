"""HaloACT 训练：LIBERO demos 直读（自有管线，绕开 lerobot），L2 回退动作 chunk。

用法: source scripts/env.sh && python scripts/train_act_halo.py --steps 15000
输出: checkpoints/act/best.pt（val 最小）+ last.pt；日志含 val MSE。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
from halo.data.libero_h5 import LiberoWindowDataset
from halo.act_policy import HaloACT

HALO_ROOT = "/home/tione/notebook/home/arianliu/project/halo-wam"


def build_task_vocab(files):
    names = sorted({os.path.basename(f).replace("_demo.hdf5", "") for f in files})
    return {n: i for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=13, help="= ctx-1+stride+fut（与数据窗一致）")
    ap.add_argument("--exclude-file", type=str, default="")
    ap.add_argument("--out", type=str, default="checkpoints/act")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    full = LiberoWindowDataset(None, 2, 8, 4, samples_per_demo=4)
    vocab = build_task_vocab(full.files)
    print(f"[act] tasks: {len(vocab)}, samples: {len(full)}")
    n_val = max(1, len(full) // 20)
    ds_train, ds_val = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))

    dl = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=6,
                    collate_fn=__import__("halo.data.libero_h5", fromlist=["collate"]).collate,
                    persistent_workers=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=2,
                        collate_fn=__import__("halo.data.libero_h5", fromlist=["collate"]).collate)

    dev = "cuda:0"
    model = HaloACT(dim=256, chunk=args.chunk, n_tasks=len(vocab), act_dim=7).to(dev)
    print(f"[act] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    def to_dev(b):
        def img(t):
            return t.to(dev, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0  # (B,3,H,W)
        return (img(b["ctx"][:, 0]), img(b["ctx"][:, -1]),
                b["state"].to(dev),
                torch.tensor([vocab[t] for t in b["task"]], device=dev))

    it, t0, best = 0, time.time(), 1e9
    log = []
    while it < args.steps:
        for b in dl:
            if it >= args.steps:
                break
            cam_prev, cam_last, state, tid = to_dev(b)
            target = torch.stack(b["actions"])[:, : args.chunk].to(dev)  # (B,chunk,7)
            pred = model(cam_prev, cam_last, state, tid)
            loss = torch.nn.functional.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            log.append(loss.item())
            if it % 200 == 0:
                print(f"[act] step {it} | train-mse {np.mean(log[-200:]):.4f} | "
                      f"{(time.time()-t0)/max(it+1,1):.2f}s/step", flush=True)
            if it % 2000 == 0 and it > 0:
                model.eval(); vls = []
                with torch.no_grad():
                    for vb in dl_val:
                        p, t = to_dev(vb), torch.stack(vb["actions"])[:, : args.chunk]
                        vls.append(torch.nn.functional.mse_loss(
                            model(*p), t.to(dev)).item())
                vl = float(np.mean(vls))
                print(f"[act] step {it} | VAL-mse {vl:.4f}", flush=True)
                if vl < best:
                    best = vl
                    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args)},
                               os.path.join(args.out, "best.pt"))
                model.train()
            it += 1
    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args)},
               os.path.join(args.out, "last.pt"))
    print(f"[act] DONE best-val {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
