"""E1-v3 奖励头训练（demo-ctx 配对版）：同 ctx 下动作对错判别，消除视觉捷径。

v1/v2 教训：demo(+)/失败rollout(−) 跨分布 → acc 1.0 捷径（demo 检测器）。
原 v3 计划（成功 vs 失败 rollout）受限于成功样本稀缺（act60k 240 eps 仅 2 成功且全出 task3）。

v3' 设计：ctx 一律来自 demo（视觉分布单一），pair 内 pos/neg 共享同一 ctx，唯一变量是动作块：
  pos : (demo ctx, demo actions)                  —— 正确动作 → 成功
  neg1: (demo ctx, 同任务失败 rollout 的动作块)    —— 该状态下错误动作
  neg2: (demo ctx, 其他任务的 demo 动作块)         —— 任务错配动作
奖励头输入 (ctx_pool, fut_pool=动态头预测, task_id)，只能靠动作条件化的 fut 预测判别。
留出：demo 按 episode 切、rollout 按文件切，val AUC 选模。

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

HALO_DATA = "/home/tione/notebook/home/arianliu/data/halo-wam"


def load_demos():
    """{task: [(actions(T,7), obs(T,H,W,3), demo_id)]} —— libero_spatial 全部 50 demo/任务"""
    import h5py

    demos = {}
    for fp in sorted(glob.glob(f"{HALO_DATA}/datasets/libero/libero_spatial/*.hdf5")):
        task = os.path.basename(fp).replace("_demo.hdf5", "")
        with h5py.File(fp, "r") as f:
            for d in sorted(f["data"].keys()):
                g = f[f"data/{d}"]
                T = g["actions"].shape[0]
                if T >= 24:
                    demos.setdefault(task, []).append(
                        (g["actions"][:].astype(np.float32), g["obs/agentview_rgb"][:], f"{task}/{d}"))
    return demos


def load_rollout_fails():
    """{task: [actions(T,7)]} —— act_15k + act60k 的失败 episode 动作序列"""
    import h5py

    fails = {}
    for pat in ("act_15k", "act60k"):
        for fp in sorted(glob.glob(f"{HALO_DATA}/datasets/rollouts/{pat}/*/*.hdf5")):
            try:
                with h5py.File(fp, "r") as f:
                    g = f["data/demo_0"]
                    if int(g.attrs.get("success", 0)) == 0 and g["actions"].shape[0] >= 16:
                        task = str(g.attrs.get("task", "task"))
                        fails.setdefault(task, []).append(g["actions"][:].astype(np.float32))
            except Exception:
                pass
    return fails


def demo_windows(actions, obs, n_win=5):
    """均匀取 n_win 个窗口: ctx=obs[s:s+2]（拷贝，解除对大数组的引用）, acts=actions[s+1:s+13]"""
    T = actions.shape[0]
    idx = np.linspace(0, T - 14, n_win).astype(int)
    out = []
    for s in idx:
        a = actions[s + 1:s + 13]
        if len(a) == 12:
            out.append((obs[s:s + 2].copy(), a))
    return out


def rollout_window(actions, rng):
    s = int(rng.randint(0, max(1, actions.shape[0] - 13)))
    a = actions[s + 1:s + 13]
    return a if len(a) == 12 else None


def build_samples(demos, fails, seed=0, val_frac=0.15):
    """按 demo episode + rollout 文件留出 val。返回 (train, val)。"""
    rng = np.random.RandomState(seed)
    tasks = sorted(demos.keys())
    other = {t: [d for d in tasks if d != t] for t in tasks}
    train_s, val_s = [], []
    for task in tasks:
        eps = demos[task]
        rng.shuffle(eps)
        n_val = max(2, int(len(eps) * val_frac))
        fail_acts = fails.get(task, [])
        n_fval = max(1, int(len(fail_acts) * val_frac))
        for into_val, ep_slice, fa_slice in ((False, eps[n_val:], fail_acts[n_fval:]),
                                             (True, eps[:n_val], fail_acts[:n_fval])):
            for acts, obs, _ in ep_slice:
                for ctx, a in demo_windows(acts, obs):
                    out = val_s if into_val else train_s
                    out.append((ctx, a, task, 1))
                    if fa_slice:  # neg1: 同任务失败动作
                        fa = rollout_window(fa_slice[rng.randint(len(fa_slice))], rng)
                        if fa is not None:
                            out.append((ctx, fa, task, 0))
                    # neg2: 其他任务 demo 动作
                    ot = other[task][rng.randint(len(other[task]))]
                    oa, oobs, _ = demos[ot][rng.randint(len(demos[ot]))]
                    na = rollout_window(oa, rng)
                    if na is not None:
                        out.append((ctx, na, task, 0))
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
    ap.add_argument("--out", type=str, default="checkpoints/e1_v3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    print("[e1v3] loading demos (500 h5) ...", flush=True)
    demos = load_demos()
    fails = load_rollout_fails()
    n_fail = sum(len(v) for v in fails.values())
    print(f"[e1v3] demos: {sum(len(v) for v in demos.values())} eps / {len(demos)} tasks | "
          f"rollout fails: {n_fail} eps / {len(fails)} tasks")
    ds_train, ds_val = build_samples(demos, fails)
    print(f"[e1v3] samples: train {len(ds_train)} (pos {sum(1 for s in ds_train if s[3]==1)}) "
          f"| val {len(ds_val)} (pos {sum(1 for s in ds_val if s[3]==1)})")
    del demos  # 释放原始数组，保留样本引用即可（samples 内含 ctx 切片）

    vocab = {t: i for i, t in enumerate(sorted(set(s[2] for s in ds_train + ds_val)))}
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
    pos_w = torch.tensor(1.0, device=dev)  # 1:1:1 构造天然平衡

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
