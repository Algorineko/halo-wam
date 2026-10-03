"""E3-A BoN 蒸馏数据生成器：demo 状态上策略采样 N 候选 → HALO 打分 → 选优落盘。

每条记录:
  ctx (2,H,W,3) uint8   demo 上下文两帧
  state (8,)            本体状态
  task_id int           任务号（与 act vocab 一致）
  chosen (13,7)         HALO 选中的动作块（蒸馏监督信号）
  demo_act (13,7)       demo 原动作（对照/消融用）
  score/std/margin      选中块的 reward、认知 std、与次优的分差
低置信过滤：std > std_p90 的状态整个跳过（门控蒸馏——不在不确定处教）。

用法: source scripts/env.sh && python scripts/gen_e3_data.py --n-cand 8
输出: $HALO_DATA/datasets/e3_distill/train.h5（供 train_e3_sft.py 消费）
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
from halo.encoder import HaloEncoder
from halo.dynamics import DynamicsEnsemble
from halo.reward import RewardHead
from halo.act_policy import HaloACT

HALO_DATA = "/home/tione/notebook/home/arianliu/data/halo-wam"
HALO_ROOT = "/home/tione/notebook/home/arianliu/project/halo-wam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.05, help="候选动作块噪声（比评测 0.02 大，换取多样性）")
    ap.add_argument("--lambda-u", type=float, default=0.25)
    ap.add_argument("--reward-ckpt", type=str, default="checkpoints/e1_v3/reward_head.pt")
    ap.add_argument("--verifier", type=str, default="checkpoints/e0v5/dyn_ens_k5_s2000.pt")
    ap.add_argument("--policy", type=str, default="checkpoints/act60k/best.pt")
    ap.add_argument("--win-per-demo", type=int, default=4)
    ap.add_argument("--std-p90-skip", type=float, default=1e9, help="先收集后过滤，此参数仅作日志参考")
    ap.add_argument("--out", type=str, default=f"{HALO_DATA}/datasets/e3_distill/train.h5")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = "cuda:1"

    pol_ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    vocab = pol_ck["vocab"]
    pol = HaloACT(dim=pol_ck["args"].get("dim", 256), chunk=pol_ck["args"]["chunk"],
                  n_tasks=len(vocab), act_dim=7).to(dev).eval()
    pol.load_state_dict(pol_ck["model"])

    enc = HaloEncoder(device=dev)
    ens = DynamicsEnsemble(k=5, dim=1024, act_dim=7, n_fut_tubelets=4, n_heads=16, n_layers=6).to(dev)
    ck = torch.load(args.verifier, map_location=dev)
    ens.load_state_dict(ck["model"])
    ens.eval()
    rk = torch.load(args.reward_ckpt, map_location=dev, weights_only=False)
    rhead = RewardHead(dim=1024, n_tasks=len(rk["vocab"])).to(dev).eval()
    rhead.load_state_dict(rk["model"])

    @torch.no_grad()
    def score_batch(ctx_frames, acts, task_name):
        """ctx (M,2,H,W,3) uint8, acts (M,N,13,7) → rewards (M,N), stds (M,N)"""
        M, N = acts.shape[:2]
        cam = torch.from_numpy(ctx_frames).to(dev).float().permute(0, 1, 4, 2, 3) / 255.0  # (M,2,3,H,W)
        z = enc.encode(cam)                       # (M,1,16,16,1024) 2帧→1 tubelet
        ctx_tok = z.reshape(M, -1, 1024)          # (M,256,D) 与 e1_v3 训练严格一致
        ctx_rep = ctx_tok.unsqueeze(1).expand(M, N, -1, -1).reshape(M * N, -1, 1024)
        a = torch.from_numpy(acts.reshape(M * N, 13, 7)).to(dev)
        stds = ens.epistemic_std(ctx_rep, a).mean(-1).reshape(M, N)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fut = ens(ctx_rep, a.to(ctx_tok.dtype))
        fut_tok = fut.float().mean(1)
        tid = torch.tensor([vocab[task_name]] * (M * N), device=dev)
        rew = rhead(ctx_rep.float(), fut_tok, tid).reshape(M, N)
        return rew.float().cpu().numpy(), stds.float().cpu().numpy()

    demo_files = sorted(glob.glob(f"{HALO_DATA}/datasets/libero/libero_spatial/*.hdf5"))
    print(f"[e3gen] {len(demo_files)} demo files, n_cand={args.n_cand}, noise={args.noise}")
    recs = {"ctx": [], "state": [], "task": [], "chosen": [], "demo_act": [],
            "score": [], "std": [], "margin": [], "ep": []}
    torch.manual_seed(0)
    batch_ctx, batch_st, batch_ta, batch_demo, recs_ep = [], [], [], [], []

    def flush():
        if not batch_ctx:
            return
        M = len(batch_ctx)
        N = args.n_cand
        acts = np.zeros((M, N, 13, 7), dtype=np.float32)
        for i, (ctx, st, tid, demo_a) in enumerate(zip(batch_ctx, batch_st, batch_ta, batch_demo)):
            cam1 = torch.from_numpy(ctx[0]).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
            cam2 = torch.from_numpy(ctx[1]).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
            stt = torch.from_numpy(st).float().unsqueeze(0).to(dev)
            tt = torch.tensor([tid]).to(dev)
            for n in range(N):
                with torch.inference_mode():
                    chunk = pol(cam1, cam2, stt, tt)[0] + torch.randn(13, 7, device=dev) * args.noise
                acts[i, n] = chunk.clamp(-1, 1).cpu().numpy()
        rew, std = score_batch(np.stack(batch_ctx), acts, batch_ta[0][1])
        best = rew - args.lambda_u * std
        for i in range(M):
            order = np.argsort(-best[i])
            b, s2 = order[0], order[1]
            recs["ctx"].append(batch_ctx[i]); recs["state"].append(batch_st[i])
            recs["task"].append(batch_ta[i][1]); recs["chosen"].append(acts[i, b])
            recs["demo_act"].append(batch_demo[i])
            recs["score"].append(float(rew[i, b])); recs["std"].append(float(std[i, b]))
            recs["margin"].append(float(best[i, b] - best[i, s2]))
            recs["ep"].append(recs_ep[i])
        batch_ctx.clear(); batch_st.clear(); batch_ta.clear(); batch_demo.clear(); recs_ep.clear()

    B = 32
    n_states = 0
    cur_task = None
    for fp in demo_files:
        task = os.path.basename(fp).replace("_demo.hdf5", "")
        if cur_task is not None and task != cur_task:
            flush()  # batch 不跨任务：tid 按批次首任务统一
        cur_task = task
        with h5py.File(fp, "r") as f:
            for d in sorted(f["data"].keys()):
                g = f[f"data/{d}"]
                T = g["actions"].shape[0]
                if T < 24:
                    continue
                obs = g["obs/agentview_rgb"][:]
                acts = g["actions"][:].astype(np.float32)
                joints = g["obs/joint_states"][:]
                grips = g["obs/gripper_states"][:]
                idx = np.linspace(0, T - 14, args.win_per_demo).astype(int)
                for s in idx:
                    ctx = obs[s:s + 2].copy()
                    da = acts[s + 1:s + 13]
                    if len(da) != 12:
                        continue
                    da = np.vstack([da, np.zeros((1, 7), np.float32)])  # 13 帧对齐
                    st = np.concatenate([joints[s], grips[s][:1]]).astype(np.float32)
                    batch_ctx.append(ctx); batch_st.append(st)
                    batch_ta.append((vocab[task], task)); batch_demo.append(da)
                    recs_ep.append(f"{task}/{d}")
                    n_states += 1
                    if len(batch_ctx) >= B:
                        flush()
        print(f"[e3gen] {task} done ({n_states} states)", flush=True)
    flush()

    stds_arr = np.array(recs["std"])
    thr = float(np.quantile(stds_arr, 0.9))
    keep = stds_arr <= thr  # 门控蒸馏：去掉不确定性最高的 10% 状态
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        for k in recs:
            if k in ("ctx", "chosen", "demo_act", "state"):
                arr = np.stack(recs[k])[keep]
            else:
                vals = np.array([s.encode() if isinstance(s, str) else s for s in recs[k]])
                arr = vals[keep]
            f.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
        f.attrs["std_p90_thr"] = thr
        f.attrs["n_total"] = len(stds_arr)
        f.attrs["n_keep"] = int(keep.sum())
        f.attrs["noise"] = args.noise
        f.attrs["lambda_u"] = args.lambda_u
    print(f"[e3gen] DONE {int(keep.sum())}/{len(stds_arr)} states (std_p90={thr:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
