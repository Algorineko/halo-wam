"""LIBERO h5 demos → LeRobot v2.1 数据集（lerobot 0.4.2 create API，含 mp4 编码）。

用法:
  source scripts/env.sh && python scripts/convert_libero_to_lerobot.py --suite libero_spatial
输出: ~/data/halo-wam/datasets/lerobot/{suite}（可直接喂 lerobot-train）
帧内容: agentview_rgb + eye_in_hand_rgb (uint8 HWC)，state = joint(7)+grip(1)，action = OSC_POSE(7)
"""
import argparse
import glob
import os

import h5py
import numpy as np

HALO_DATA = "/home/tione/notebook/home/arianliu/data/halo-wam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default="libero_spatial")
    ap.add_argument("--max-demos-per-task", type=int, default=50)
    ap.add_argument("--demos-per-file", type=int, default=10)  # 分批建库，防单进程内存/时间过长
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.agentview": {"dtype": "video", "shape": (128, 128, 3), "names": None},
        "observation.images.wrist": {"dtype": "video", "shape": (128, 128, 3), "names": None},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    repo_id = f"halo-wam/{args.suite}"
    # 每套件独立 root，避免 flat meta 冲突（spatial 首次转换在 datasets/lerobot/，保持不动）
    root = f"{HALO_DATA}/datasets/lerobot/{args.suite}"
    ds = LeRobotDataset.create(repo_id=repo_id, fps=20, features=features, root=root, use_videos=True)
    print(f"[convert] created dataset {repo_id} at {root}")

    files = sorted(glob.glob(f"{HALO_DATA}/datasets/libero/{args.suite}/*.hdf5"))
    print(f"[convert] {len(files)} task files")
    n_ep = 0
    for fi, fp in enumerate(files):
        task = os.path.basename(fp).replace("_demo.hdf5", "")
        with h5py.File(fp, "r") as f:
            demos = sorted(f["data"].keys())[: args.max_demos_per_task]
            for di, demo in enumerate(demos):
                g = f[f"data/{demo}"]
                agent = g["obs/agentview_rgb"][:]        # (T,128,128,3) uint8
                wrist = g["obs/eye_in_hand_rgb"][:]
                joint = g["obs/joint_states"][:]          # (T,7)
                grip = g["obs/gripper_states"][:]         # (T,2)
                acts = g["actions"][:]                    # (T,7)
                state = np.concatenate([joint, grip[:, :1]], axis=1).astype(np.float32)
                t = min(len(agent), len(acts))
                for k in range(t):
                    ds.add_frame({
                        "observation.images.agentview": agent[k],
                        "observation.images.wrist": wrist[k],
                        "observation.state": state[k],
                        "action": acts[k].astype(np.float32),
                        "task": task.replace("_", " "),
                    })
                ds.save_episode()
                n_ep += 1
                if n_ep % 20 == 0:
                    print(f"[convert] {n_ep} episodes done (file {fi+1}/{len(files)}, {task[:40]} demo {di+1})", flush=True)
    print(f"[convert] DONE: {n_ep} episodes -> {root}/{args.suite}")


if __name__ == "__main__":
    main()
