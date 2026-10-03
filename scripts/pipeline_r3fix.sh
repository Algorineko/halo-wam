#!/bin/bash
# R3 halo 臂 task7/8 补跑（OOM 死亡重试）：等 act60k 采样 DONE → workers=1 串行补跑
# 内存预算：采样结束后 e1v3 训练(~8GB) + 1 个评测 worker(~6GB) + 常驻(~4GB) ≈ 18GB << 32GB
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

# 采样完成标志（collect_rollouts 打印 DONE）
while ! grep -q "DONE" logs/collect_act60k.log 2>/dev/null; do sleep 120; done
echo "$(date +%H:%M) collection done -> rerun halo task7,8 (workers=1)" >> logs/pipeline_r3fix.log

python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 1 \
  --policy haloact_h --ckpt checkpoints/act60k/best.pt \
  --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 \
  --reward-ckpt checkpoints/e1_v2/reward_head.pt --lambda-u 0.25 --task-ids 7,8 \
  > logs/eval_r3_halo_fix2.log 2>&1
grep "平均成功率" logs/eval_r3_halo_fix2.log >> logs/pipeline_r3fix.log
echo DONE > logs/r3fix.done
