"""E1-v3 奖励头训练（同分布修复版）：成功 rollout(+) vs 失败 rollout(−)，可选同 ctx 动作错配(−)。

v1/v2 的教训：demo(+)/rollout(−) 跨分布 → acc 1.000 捷径学习，奖励头成了"demo 检测器"，
在线打分时对策略 rollout 全体输出饱和负 logit，无判别力（R2 完整 HALO 臂 12% < 纯验证臂 18%）。

v3 样本（全部来自同一策略的 rollout，视觉/状态分布完全一致）:
  pos : (ctx_s, acts_s)  成功 episode 的窗口
  neg1: (ctx_f, acts_f)  失败 episode 的窗口
  neg2: (ctx_s, acts_f') 同任务失败 episode 的动作块接到成功 ctx —— 教动作条件化
验证:留出整个 episode（不切窗口），报 AUC（主指标）而非 acc。

用法: source scripts/env.sh && python scripts/train_e1_v3.py --steps 1200
输出: checkpoints/e1_v3/reward_head.pt（val AUC 最大）
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


def load_pool(rollout_dir):
    """返回 {task: [(actions(T,7), agentview(T,H,W,3), label)]}"""
    import h5py

    pool = {}
    for fp in sorted(glob.glob(f"{rollout_dir}/*/*.hdf5")):
        try:
            with h5py.File(fp, "r") as f:
                g = f["data/demo_0"]
                T = g["actions"].shape[0]
                if T < 16:
                    continue
                task = str(g.attrs.get("task", "task"))
                pool.setdefault(task, []).append(
                    (g["actions"][:].astype(np.float32),
                     g["obs/agentview_rgb"][:], int(g.attrs.get("success", 0))))
        except Exception:
            pass
    return pool


def build_samples(pool, win_stride=6, mismatch_neg=True, seed=0, val_frac=0.15):
    """窗口化 + 同 ctx 动作错配负样本。按 episode 留出 val（同 episode 窗口不跨集合）。
    返回 (train_samples, val_samples)，元素为 [(ctx(2,H,W,3), acts(12,7), task, label)]"""
    rng = np.random.RandomState(seed)
    train_s, val_s = [], []
    for task, eps in sorted(pool.items()):
        suc = [e for e in eps if e[2] == 1]
        fail = [e for e in eps if e[2] == 0]
        if not suc or not fail:
            continue  # 该任务缺成功或失败样本则跳过（无法构成同分布对比）

        # episode 级留出：每任务留 1 成功 + 1 失败 episode 进 val（若充足）
        rng.shuffle(suc), rng.shuffle(fail)
        n_vs = 1 if len(suc) >= 4 else 0
        n_vf = 1 if len(fail) >= 4 else 0
        suc_v, suc_t = suc[:n_vs], suc[n_vs:]
        fail_v, fail_t = fail[:n_vf], fail[n_vf:]

        def windows(ep):
            acts, obs, _ = ep
            T = acts.shape[0]
            out = []
            for s in range(0, T - 14, win_stride):
                ctx = obs[s:s + 2]
                a = acts[s + 1:s + 13]
                if len(a) == 12:
                    out.append((ctx, a))
            return out

        for into_val, s_ep, f_ep in ((False, suc_t, fail_t), (True, suc_v, fail_v)):
            out = val_s if into_val else train_s
            suc_w = [w for ep in s_ep for w in windows(ep)]
            fail_w = [w for ep in f_ep for w in windows(ep)]
            for ctx, a in suc_w:
                out.append((ctx, a, task, 1))
            for ctx, a in fail_w:
                out.append((ctx, a, task, 0))
            if mismatch_neg and not into_val:
                # 每个 pos 配 1 条同任务失败动作块（ctx 保持成功窗口）——只进 train
                for ctx, _ in suc_w:
                    fa = fail_w[rng.randint(len(fail_w))][1]
                    out.append((ctx, fa, task, 0))
    for lst in (train_s, val_s):
        idx = rng.permutation(len(lst))
        lst[:] = [lst[i] for i in idx]
    return train_s, val_s


class E1Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ctx, acts, task, label = self.samples[i]
        return (torch.from_numpy(np.ascontiguousarray(ctx)), torch.from_numpy(acts), task, label)


@torch.no_grad()
def eval_auc(head, enc, ens, vocab, samples, dev):
    """对预编码样本集合算 AUC（样本已缓存 latent 时用快速路径）"""
    head.eval()
    ys, ss = [], []
    for cam, acts, tasks, labels in DataLoader(E1Dataset(samples), batch_size=32, shuffle=False):
        ctx_lat = enc.encode(cam.to(dev))
        b = ctx_lat.shape[0]
        ctx_tok = ctx_lat.reshape(b, -1, 1024).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_fut = ens(ctx_tok, acts.to(dev).to(ctx_tok.dtype))
        fut_tok = pred_fut.float().mean(0)
        tid = torch.tensor([vocab[t] for t in tasks], device=dev)
        logits = head(ctx_tok, fut_tok, tid).float()
        ys += labels.tolist()
        ss += torch.sigmoid(logits).cpu().tolist()
    ys, ss = np.array(ys), np.array(ss)
    # 秩和 AUC
    order = np.argsort(ss)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ss) + 1)
    n_pos, n_neg = (ys == 1).sum(), (ys == 0).sum()
    auc = (ranks[ys == 1].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
    head.train()
    return float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--verifier", type=str, default="checkpoints/e0v5/dyn_ens_k5_s2000.pt")
    ap.add_argument("--rollouts", type=str, default=f"{HALO_DATA}/datasets/rollouts/act60k")
    ap.add_argument("--win-stride", type=int, default=6)
    ap.add_argument("--no-mismatch", action="store_true")
    ap.add_argument("--out", type=str, default="checkpoints/e1_v3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    pool = load_pool(args.rollouts)
    n_suc = sum(1 for eps in pool.values() for e in eps if e[2] == 1)
    n_eps = sum(len(eps) for eps in pool.values())
    print(f"[e1v3] rollouts: {n_eps} eps ({n_suc} success), tasks: {len(pool)}")
    ds_train, ds_val = build_samples(pool, args.win_stride, not args.no_mismatch)
    print(f"[e1v3] windows: train {len(ds_train)} (pos {sum(1 for s in ds_train if s[3]==1)}) "
          f"| val {len(ds_val)} (pos {sum(1 for s in ds_val if s[3]==1)}) [episode 级留出]")
    vocab = {t: i for i, t in enumerate(sorted(pool.keys()))}
    dl = DataLoader(E1Dataset(ds_train), batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)

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
    n_pos = sum(1 for s in ds_train if s[3] == 1)
    pos_w = torch.tensor([max(1.0, (len(ds_train) - n_pos) / max(n_pos, 1))], device=dev)
    print(f"[e1v3] train {len(ds_train)} (pos {n_pos}), val {n_val}, pos_w {pos_w.item():.2f}")

    import time
    it, t0, best = 0, time.time(), 0.0
    log = []
    while it < args.steps:
        for cam, acts, tasks, labels in dl:
            if it >= args.steps:
                break
            with torch.no_grad():
                ctx_lat = enc.encode(cam.to(dev, non_blocking=True))
                b = ctx_lat.shape[0]
                ctx_tok = ctx_lat.reshape(b, -1, 1024)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_fut = ens(ctx_tok, acts.to(dev).to(ctx_tok.dtype))
                fut_tok = pred_fut.float().mean(0)
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
                acc = ((logits > 0).float() == y).float().mean().item()
                print(f"[e1v3] step {it} | bce {np.mean(log[-100:]):.4f} | acc {acc:.3f} "
                      f"| {(time.time()-t0)/max(it+1,1):.2f}s/step", flush=True)
            if it % 300 == 0 and it > 0:
                auc = eval_auc(head, enc, ens, vocab, ds_val, dev)
                print(f"[e1v3] step {it} | VAL-AUC {auc:.3f}", flush=True)
                if auc > best:
                    best = auc
                    torch.save({"model": head.state_dict(), "vocab": vocab, "args": vars(args)},
                               os.path.join(args.out, "reward_head.pt"))
            it += 1
    auc = eval_auc(head, enc, ens, vocab, ds_val, dev)
    if auc > best:
        best = auc
        torch.save({"model": head.state_dict(), "vocab": vocab, "args": vars(args)},
                   os.path.join(args.out, "reward_head.pt"))
    print(f"[e1v3] DONE best-val-AUC {best:.3f} -> {args.out}/reward_head.pt")


if __name__ == "__main__":
    main()
