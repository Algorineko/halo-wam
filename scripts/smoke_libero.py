"""LIBERO MuJoCo CPU 离屏渲染冒烟测试（无 N 卡）。

用法: source scripts/env.sh && python scripts/smoke_libero.py
验证: 创建任务 env → reset 到 init state → 随机 step → agentview 渲染出图 + 计时。
依赖: MUJOCO_GL=egl + LIBGL_ALWAYS_SOFTWARE=1（Mesa llvmpipe 软渲染，env.sh 已设）。
"""
import time

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def main():
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    task_name = task.name
    init_states = suite.get_task_init_states(0)
    bddl_path = suite.get_task_bddl_file_path(0)
    print(f"[smoke] task: {task_name} | init states: {len(init_states)}")

    t0 = time.time()
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    print(f"[smoke] env created in {time.time()-t0:.1f}s")

    t0 = time.time()
    obs = env.reset()
    env.set_init_state(init_states[0])
    import numpy as np
    dim = env.env.action_dim
    obs, _, _, _ = env.step(np.zeros(dim))
    print(f"[smoke] reset+set_init in {time.time()-t0:.1f}s | agentview: {obs['agentview_image'].shape}")

    t0 = time.time()
    for _ in range(10):
        action = np.random.uniform(-1, 1, dim)
        obs, reward, done, info = env.step(action)
    dt = (time.time() - t0) / 10
    img = obs["agentview_image"]
    print(f"[smoke] step+render: {dt*1000:.0f} ms/step | img {img.shape} dtype {img.dtype} range [{img.min()}, {img.max()}]")

    import imageio.v2 as imageio
    imageio.imwrite("/home/tione/notebook/home/arianliu/project/halo-wam/logs/libero_smoke_agentview.png", img[::-1])
    print("[smoke] saved logs/libero_smoke_agentview.png")
    print("[smoke] PASS ✅")


if __name__ == "__main__":
    main()
