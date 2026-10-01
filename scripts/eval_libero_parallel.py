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
    task_idx, suite_name, n_episodes, seed, policy = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/tione/notebook/home/arianliu/data/halo-wam/libero_config")

    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_idx)
    init_states = suite.get_task_init_states(task_idx)
    bddl = suite.get_task_bddl_file_path(task_idx)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    n_episodes = min(n_episodes, len(init_states))
    successes, steps_total = 0, 0
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for ep in range(n_episodes):
        env.reset()
        env.set_init_state(init_states[ep])
        done, step = False, 0
        while not done and step < 500:  # LIBERO 上限 500 步（robosuite horizon）
            if policy == "random":
                action = rng.uniform(-1, 1, env.env.action_dim)
            else:
                raise ValueError(f"unknown policy {policy}")
            _, _, done, _ = env.step(action)
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
    p.add_argument("--policy", type=str, default="random")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = suite.get_num_tasks()
    jobs = [(i, args.suite, args.episodes, args.seed + i, args.policy) for i in range(n_tasks)]
    print(f"[eval] {args.suite}: {n_tasks} tasks × {args.episodes} eps, {args.workers} workers, policy={args.policy}")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        results = pool.map(run_one_task, jobs)
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
