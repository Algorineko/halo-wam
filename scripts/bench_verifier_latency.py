"""验证器打分延迟 microbench（论文卖点数据：隐空间验证器 vs 生成式 rollout 成本）。

测三段：
  A. 策略 chunk 生成（HaloACT tiny, CPU）——best-of-N 的候选来源成本
  B. 验证器打分 N 候选（V-JEPA ctx 编码一次 + 动态头 ensemble 批前向, GPU）
  C. 每候选边际成本（B/N）与每决策总开销（A + B）
输出 eval_results/latency_bench.json。
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")

DEV = "cuda:0"  # 配合 HIP_VISIBLE_DEVICES 选物理卡


def main():
    from halo.act_policy import HaloACT
    from halo.dynamics import DynamicsEnsemble
    from halo.encoder import HaloEncoder

    # ---- 装载（与 eval_libero_parallel 同构）----
    ck = torch.load("checkpoints/act60k/best.pt", map_location="cpu", weights_only=False)
    pol = HaloACT(dim=ck["args"].get("dim", 256), chunk=ck["args"]["chunk"],
                  n_tasks=ck["args"].get("n_tasks", len(ck["vocab"])), act_dim=7).cpu().eval()
    pol.load_state_dict(ck["model"])
    enc = HaloEncoder(device=DEV)
    eck = torch.load("checkpoints/e0v5/dyn_ens_k5_s2000.pt", map_location=DEV)
    ens = DynamicsEnsemble(k=5, dim=1024, act_dim=7, n_fut_tubelets=4, n_heads=16, n_layers=6)
    ens.load_state_dict(eck["model"])
    ens = ens.to(DEV).eval()

    obs = torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8)
    st = torch.randn(8)
    tid = torch.tensor([0])
    res = {}

    # ---- A. 策略 chunk（CPU）----
    a1 = obs.permute(2, 0, 1).float().unsqueeze(0) / 255.0
    a2 = torch.randn(1, 3, 128, 128)
    s = torch.from_numpy(st.numpy()).float().unsqueeze(0)
    for _ in range(3):
        with torch.inference_mode():
            pol(a1, a2, s, tid)
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.inference_mode():
            pol(a1, a2, s, tid)
    res["policy_chunk_ms"] = round((time.perf_counter() - t0) / 20 * 1000, 2)

    # ---- B/C. 验证器打分（GPU, bf16 autocast, 与 eval 同路径）----
    lat = {}
    for n in (1, 3, 5, 10):
        cands = torch.randn(n, 13, 7).clamp(-1, 1)
        # ctx 编码 + ens 前向（打分热身后计时 30 轮）
        with torch.inference_mode():
            for _ in range(5):
                latents = enc.encode(obs.unsqueeze(0).to(DEV))
                ctx = latents.reshape(1, -1, 1024).expand(n, -1, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ens(ctx, cands.to(DEV).to(ctx.dtype))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                latents = enc.encode(obs.unsqueeze(0).to(DEV))
                ctx = latents.reshape(1, -1, 1024).expand(n, -1, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ens(ctx, cands.to(DEV).to(ctx.dtype))
            torch.cuda.synchronize()
            lat[n] = round((time.perf_counter() - t0) / 30 * 1000, 2)
    res["score_ms_by_ncand"] = lat

    # ---- D. 打分分解（n=5）：ctx 编码 vs ens 前向（缓存下界，RoVer 式缓存参照）----
    cands5 = torch.randn(5, 13, 7).clamp(-1, 1)
    with torch.inference_mode():
        latents = enc.encode(obs.unsqueeze(0).to(DEV))
        ctx5 = latents.reshape(1, -1, 1024).expand(5, -1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ens(ctx5, cands5.to(DEV).to(ctx5.dtype))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            enc.encode(obs.unsqueeze(0).to(DEV))
        torch.cuda.synchronize()
        enc_ms = (time.perf_counter() - t0) / 30 * 1000
        t0 = time.perf_counter()
        for _ in range(30):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ens(ctx5, cands5.to(DEV).to(ctx5.dtype))
        torch.cuda.synchronize()
        ens_ms = (time.perf_counter() - t0) / 30 * 1000
    res["score_decomp_n5"] = {"ctx_encode_ms": round(enc_ms, 2),
                              "ens_forward_ms": round(ens_ms, 2),
                              "sum": round(enc_ms + ens_ms, 2)}
    res["per_decision_ms_n5"] = round(res["policy_chunk_ms"] + lat[5], 2)
    res["per_candidate_ms_n5"] = round(lat[5] / 5, 2)
    res["note"] = "score 含 ctx 重编码（保守上界；部署可缓存 ctx，仅 ens 前向更低）"
    res["env"] = {"device": torch.cuda.get_device_name(0), "bf16": True}

    with open("eval_results/latency_bench.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
