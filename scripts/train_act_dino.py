"""HaloACT-D 训练：冻结 DINOv2-small 特征 + token 级 memory（基线强化路线）。

用法: source scripts/env.sh && python scripts/train_act_dino.py --steps 15000
输出: checkpoints/act_dino/best.pt（val 最小）+ last.pt
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
from halo.act_policy_dino import HaloACTD

HALO_ROOT = "/home/tione/notebook/home/arianliu/project/halo-wam"


def build_task_vocab(files):
    names = sorted({os.path.basename(f).replace("_demo.hdf5", "") for f in files})
    return {n: i for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=13)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--out", type=str, default="checkpoints/act_dino")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    full = LiberoWindowDataset(None, 2, 8, 4, samples_per_demo=4)
    vocab = build_task_vocab(full.files)
    print(f"[act-d] tasks: {len(vocab)}, samples: {len(full)}")
    n_val = max(1, len(full) // 20)
    ds_train, ds_val = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))

    collate = __import__("halo.data.libero_h5", fromlist=["collate"]).collate
    dl = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=6,
                    collate_fn=collate, persistent_workers=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)

    dev = "cuda:1"
    model = HaloACTD(dim=args.dim, chunk=args.chunk, n_tasks=len(vocab), act_dim=7).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[act-d] trainable: {n_train/1e6:.1f}M (+ frozen dinov2-small 21M)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.05)

    def to_dev(b):
        def img(t):
            return t.to(dev, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
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
            target = torch.stack(b["actions"])[:, : args.chunk].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(cam_prev, cam_last, state, tid)
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            log.append(loss.item())
            if it % 200 == 0:
                print(f"[act-d] step {it} | train-mse {np.mean(log[-200:]):.4f} | "
                      f"{(time.time()-t0)/max(it+1,1):.2f}s/step", flush=True)
            if it % 2000 == 0 and it > 0:
                model.eval()
                vls = []
                with torch.no_grad():
                    for vb in dl_val:
                        p, t = to_dev(vb), torch.stack(vb["actions"])[:, : args.chunk]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pv = model(*p)
                        vls.append(torch.nn.functional.mse_loss(pv.float(), t.to(dev)).item())
                vl = float(np.mean(vls))
                print(f"[act-d] step {it} | VAL-mse {vl:.4f}", flush=True)
                if vl < best:
                    best = vl
                    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args)},
                               os.path.join(args.out, "best.pt"))
                model.train()
            it += 1
    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args)},
               os.path.join(args.out, "last.pt"))
    print(f"[act-d] DONE best-val {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
