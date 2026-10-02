"""LIBERO 并行评测农场（纯 CPU，MuJoCo OSMesa，无 N 卡）。

用法:
  # 随机策略基线（同时测农场吞吐）
  source scripts/env.sh && python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 5 --workers 32

  # 指定策略（后续接 SmolVLA/OpenPI checkpoint）
  python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 50 --workers 64 --policy random

结果: eval_results/<suite>_<policy>_<ts>.json + 控制台汇总表。
worker 内每个任务串行跑 episode，env 创建开销 ~3s 只付一次/worker。
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime

_RESULT_DIR = "/home/tione/notebook/home/arianliu/project/halo-wam/eval_results"


def run_one_task(args):
    """子进程：跑一个任务的 n 个 episode，返回成功率。"""
    task_idx, suite_name, n_episodes, seed, policy, ckpt, verifier, n_cand, dump_dir, reward_ckpt, lambda_u = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/tione/notebook/home/arianliu/data/halo-wam/libero_config")

    import numpy as np
    import torch
    torch.set_num_threads(4)  # 容器 32GB：不限线程时每进程 OpenMP 栈会撑爆内存
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_idx)
    task_str = task.name.replace("_", " ")
    init_states = suite.get_task_init_states(task_idx)
    bddl = suite.get_task_bddl_file_path(task_idx)

    pol = None
    verifier_scorer = None
    if policy in ("haloact", "haloact_v", "haloact_h"):
        import sys

        sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
        from halo.act_policy import HaloACT

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        vocab = ck["vocab"]
        inv_vocab = {v: k for k, v in vocab.items()}
        pol = HaloACT(dim=ck["args"].get("dim", 256), chunk=ck["args"]["chunk"], n_tasks=len(vocab), act_dim=7).to("cpu").eval()
        pol.load_state_dict(ck["model"])
    if policy in ("haloact_v", "haloact_h"):
        import sys

        sys.path.insert(0, "/home/tione/notebook/home/arianliu/project/halo-wam/src")
        from halo.dynamics import DynamicsEnsemble
        from halo.encoder import HaloEncoder
        from halo.verify import UncertaintyScorer

        enc = HaloEncoder(device="cuda:0")
        ens = DynamicsEnsemble(k=5, dim=1024, act_dim=7, n_fut_tubelets=4, n_heads=16, n_layers=6)
        ck = torch.load(verifier, map_location="cuda:0")
        ens.load_state_dict(ck["model"])
        verifier_scorer = UncertaintyScorer(enc, ens, device="cuda:0")
        reward_head = None
        if policy == "haloact_h" and reward_ckpt:
            from halo.reward import RewardHead
            rk = torch.load(reward_ckpt, map_location="cpu", weights_only=False)
            reward_head = RewardHead(dim=1024, n_tasks=len(rk["vocab"])).cuda().eval()
            reward_head.load_state_dict(rk["model"])
            _rvocab = rk["vocab"]

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    n_episodes = min(n_episodes, len(init_states))
    task_dir = None
    if dump_dir:
        task_dir = os.path.join(dump_dir, f"task{task_idx:02d}")
        os.makedirs(task_dir, exist_ok=True)
    successes, steps_total = 0, 0
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for ep in range(n_episodes):
        obs = env.reset()
        env.set_init_state(init_states[ep])
        obs, _, _, _ = env.step(np.zeros(env.env.action_dim))  # 触发一次 obs 生成（0 动作）
        act_queue = []  # 新 episode 清空动作队列
        done, step = False, 0
        while not done and step < 500:  # LIBERO 上限 500 步（robosuite horizon）
            if policy == "random":
                action = rng.uniform(-1, 1, env.env.action_dim)
            elif policy in ("haloact_v", "haloact_h"):
                if not act_queue:
                    state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][:1]])
                    tid = torch.tensor([vocab[task.name]])
                    with torch.inference_mode():
                        cands = []
                        for _ in range(n_cand):
                            noise = torch.randn(pol.chunk * 7) * 0.02  # 轻微扰动打破确定性流
                            a1 = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                            a2 = torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                            st = torch.from_numpy(state).float().unsqueeze(0)
                            chunk = pol(a1, a2, st, tid)[0] + noise.reshape(pol.chunk, 7)
                            cands.append(chunk.clamp(-1, 1))  # (13,7)
                        cand_acts = torch.stack(cands)
                        stds = verifier_scorer.score(
                            torch.from_numpy(obs["agentview_image"]), cand_acts.float()
                        )
                        if reward_head is not None:
                            from halo.encoder import HaloEncoder as _HE
                            ctx_lat = verifier_scorer.enc.encode(torch.from_numpy(obs["agentview_image"]).unsqueeze(0))
                            ctx_tok = ctx_lat.reshape(1, -1, 1024).expand(cand_acts.shape[0], -1, -1)
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                preds = verifier_scorer.ens(ctx_tok, cand_acts.cuda().to(ctx_tok.dtype))
                            fut_tok = preds.float().mean(0)
                            rtid = torch.tensor([_rvocab[task.name]] * cand_acts.shape[0]).cuda()
                            with torch.inference_mode():
                                rewards = reward_head(ctx_tok.float(), fut_tok, rtid)
                            scores = rewards - float(lambda_u) * stds  # 完整 HALO 打分
                        else:
                            scores = stds
                        best = int(scores.argmin())
                    act_queue = cands[best].tolist()
                action = np.array(act_queue.pop(0))[: env.env.action_dim]
                obs, _, done, _ = env.step(action)
                step += 1
                continue
            elif policy == "haloact":
                state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][:1]])
                a1 = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                a2 = torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                st = torch.from_numpy(state).float().unsqueeze(0)
                tid = torch.tensor([vocab[task.name]])
                with torch.inference_mode():
                    if not act_queue:
                        chunk = pol(a1, a2, st, tid)[0].clamp(-1, 1)  # (13,7)
                        act_queue = chunk.tolist()
                action = np.array(act_queue.pop(0))[: env.env.action_dim]
            elif policy == "smolvla_v":
                if not act_queue:
                    batch = {
                        "observation.images.camera1": torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                        "observation.images.camera2": torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                        "observation.images.camera3": torch.zeros(1, 3, 128, 128),
                        "observation.state": torch.from_numpy(
                            np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][:1]])
                        ).float().unsqueeze(0),
                        "task": [task_str],
                    }
                    with torch.inference_mode():
                        cands = [pol.predict_action_chunk(batch).squeeze(0) for _ in range(n_cand)]  # (n_steps,A)
                        cand_acts = torch.stack([c[:13, :7] for c in cands])  # 对齐训练动作窗长（ctx2+stride4+fut8=13）
                        scores = verifier_scorer.score(
                            torch.from_numpy(obs["agentview_image"]), cand_acts.float()
                        )
                        best = int(scores.argmin())
                    act_queue = cands[best].tolist()
                    step_info = f"[v] step {step} 选中 {best}/{n_cand} 分数 {[round(float(s),3) for s in scores.tolist()[:4]]}"
                    print(step_info, flush=True)
                action = np.array(act_queue.pop(0))[: env.env.action_dim]
                obs, _, done, _ = env.step(action)
                step += 1
                continue
            elif policy == "smolvla":
                batch = {
                    "observation.images.camera1": torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.images.camera2": torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.images.camera3": torch.zeros(1, 3, 128, 128),
                    "observation.state": torch.from_numpy(
                        np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][:1]])
                    ).float().unsqueeze(0),
                    "task": [task_str],
                }
                with torch.inference_mode():
                    a = pol.select_action(batch)
                action = a.squeeze(0).numpy()[: env.env.action_dim]
            else:
                raise ValueError(f"unknown policy {policy}")
            obs, _, done, _ = env.step(action)
            step += 1
        successes += int(env.check_success())
        steps_total += step
    env.close()
    dt = time.time() - t0
    return {
        "task": task.name,
        "episodes": n_episodes,
        "success": successes,
        "rate": successes / max(n_episodes, 1),
        "avg_steps": steps_total / max(n_episodes, 1),
        "wall_s": round(dt, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--policy", type=str, default="random", choices=["random", "haloact", "haloact_v", "haloact_h"])
    p.add_argument("--ckpt", type=str, default="", help="smolvla checkpoint 目录")
    p.add_argument("--verifier", type=str, default="", help="E0 动态头 checkpoint（smolvla_v 用）")
    p.add_argument("--n-cand", type=int, default=5, help="best-of-N 候选数")
    p.add_argument("--reward-ckpt", type=str, default="", help="E1 奖励头（haloact_h 用）")
    p.add_argument("--lambda-u", type=float, default=1.0, help="不确定性惩罚权重")
    p.add_argument("--n-tasks", type=int, default=0, help="0=全部任务")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = suite.get_num_tasks()
    n_run = args.n_tasks or n_tasks
    jobs = [(i, args.suite, args.episodes, args.seed + i, args.policy, args.ckpt, args.verifier, args.n_cand, "", args.reward_ckpt, args.lambda_u) for i in range(n_run)]
    print(f"[eval] {args.suite}: {n_tasks} tasks × {args.episodes} eps, {args.workers} workers, policy={args.policy}")

    t0 = time.time()
    # 子进程直评模式：multiprocessing spawn-pool 在本容器会随机挂起，改为每任务独立子进程
    import subprocess
    import sys as _sys
    results, procs = [], []
    job_queue = list(jobs)
    active = {}
    while job_queue or active:
        while job_queue and len(active) < args.workers:
            j = job_queue.pop(0)
            code = (
                "import json,sys;sys.path.insert(0,'/home/tione/notebook/home/arianliu/project/halo-wam/scripts');"
                "from eval_libero_parallel import run_one_task;"
                f"print(json.dumps(run_one_task({j!r})))"
            )
            pr = subprocess.Popen([_sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            active[pr.pid] = (pr, j)
        done_pids = [pid for pid, (pr, _) in active.items() if pr.poll() is not None]
        for pid in done_pids:
            pr, j = active.pop(pid)
            out = pr.stdout.read().strip() if pr.stdout else ""
            err = pr.stderr.read().strip() if pr.stderr else ""
            try:
                results.append(json.loads(out.splitlines()[-1]))
            except Exception:
                results.append({"task": f"task{j[0]}", "episodes": j[2], "success": 0, "rate": 0.0, "avg_steps": -1, "wall_s": -1, "error": True, "stderr": err[-300:]})
        time.sleep(2)
    wall = time.time() - t0

    os.makedirs(_RESULT_DIR, exist_ok=True)
    out = {
        "meta": vars(args) | {"wall_s": round(wall, 1), "ts": datetime.now().isoformat()},
        "results": results,
        "mean_rate": sum(r["rate"] for r in results) / len(results),
    }
    path = os.path.join(_RESULT_DIR, f"{args.suite}_{args.policy}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[eval] {'任务':<60} 成功率  步数")
    for r in sorted(results, key=lambda x: x["rate"], reverse=True):
        print(f"[eval] {r['task'][:58]:<60} {r['rate']:6.1%}  {r['avg_steps']:.0f}")
    print(f"[eval] ===== {args.suite} 平均成功率 {out['mean_rate']:.1%} | 总耗时 {wall:.0f}s | 存 {path}")


if __name__ == "__main__":
    main()
