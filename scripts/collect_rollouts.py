"""E1 自产标注采集器：SmolVLA policy 在 LIBERO CPU 上的 rollout → LIBERO h5 同构数据。

输出与官方 demo h5 同构（data/demo_i/{actions, obs/*}），可直接喂 LiberoWindowDataset；
episode 成败写入 attrs（success=0/1），失败 episode 即 E1 奖励头负样本。

用法:
  source scripts/env.sh && python scripts/collect_rollouts.py --ckpt runs/smolvla_libero_spatial/checkpoints/last \
      --suite libero_spatial --episodes 5 --workers 10 --out $HALO_DATA/datasets/rollouts/smolvla_5k
"""
import argparse
import multiprocessing as mp
import os
import time


def rollout_one_task(args):
    task_idx, suite_name, n_episodes, ckpt, out_dir = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/tione/notebook/home/arianliu/data/halo-wam/libero_config")

    import h5py
    import numpy as np
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_idx)
    task_str = task.name.replace("_", " ")
    init_states = suite.get_task_init_states(task_idx)

    pol = SmolVLAPolicy.from_pretrained(ckpt)
    pol.eval()
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_idx),
                             camera_heights=128, camera_widths=128)
    n_episodes = min(n_episodes, len(init_states))
    task_dir = os.path.join(out_dir, f"task{task_idx:02d}_{task.name[:40]}")
    os.makedirs(task_dir, exist_ok=True)

    for ep in range(n_episodes):
        obs = env.reset()
        env.set_init_state(init_states[ep])
        obs, _, _, _ = env.step(np.zeros(env.env.action_dim))
        pol.reset()
        frames_a, frames_w, joints, grips, acts = [], [], [], [], []
        done, step = False, 0
        while not done and step < 500:
            batch = {
                "observation.images.camera1": torch.from_numpy(obs["agentview_rgb"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                "observation.images.camera2": torch.from_numpy(obs["eye_in_hand_rgb"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                "observation.images.camera3": torch.zeros(1, 3, 128, 128),
                "observation.state": torch.from_numpy(
                    np.concatenate([obs["joint_states"], obs["gripper_states"][:1]])
                ).float().unsqueeze(0),
                "task": [task_str],
            }
            with torch.inference_mode():
                a = pol.select_action(batch)
            action = a.squeeze(0).numpy()[: env.env.action_dim]
            frames_a.append(obs["agentview_rgb"]); frames_w.append(obs["eye_in_hand_rgb"])
            joints.append(obs["joint_states"]); grips.append(obs["gripper_states"])
            acts.append(action)
            obs, _, done, _ = env.step(action)
            step += 1
        success = int(env.check_success())
        out = os.path.join(task_dir, f"ep{ep:03d}_s{success}.hdf5")
        with h5py.File(out, "w") as f:
            g = f.create_group("data/demo_0")
            g.create_dataset("actions", data=np.stack(acts))
            og = g.create_group("obs")
            og.create_dataset("agentview_rgb", data=np.stack(frames_a), compression="gzip", compression_opts=4)
            og.create_dataset("eye_in_hand_rgb", data=np.stack(frames_w), compression="gzip", compression_opts=4)
            og.create_dataset("joint_states", data=np.stack(joints))
            og.create_dataset("gripper_states", data=np.stack(grips))
            g.attrs["success"] = success
            g.attrs["task"] = task.name
        print(f"[collect] task{task_idx} ep{ep}: success={success} steps={step}", flush=True)
    env.close()
    return task_idx, n_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    jobs = [(i, args.suite, args.episodes, args.ckpt, args.out) for i in range(suite.get_num_tasks())]
    print(f"[collect] {len(jobs)} tasks × {args.episodes} eps, workers={args.workers}")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        results = pool.map(rollout_one_task, jobs)
    print(f"[collect] DONE {sum(n for _, n in results)} episodes in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
