#!/bin/bash
# E1-v3 流水线：等 R3 收尾 → act60k 大规模 rollout 采样（成功/失败同分布）→ 奖励头 v3 训练
# 由 watchdog 或前台 nohup 启动；全程 card 1，CPU workers 保守（sub-work 约束）
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

# 1) 等 R3 三臂全部完成（halo 臂落 e2_r3.done）
while [ ! -f logs/e2_r3.done ]; do sleep 60; done
echo "$(date +%H:%M) R3 done, start rollout collection" >> logs/pipeline_e1v3.log

# 2) act60k 采样：10 任务 × 24 eps，6 workers，策略推理走 card 1
CUDA_VISIBLE_DEVICES=1 python scripts/collect_rollouts.py \
  --ckpt checkpoints/act60k/best.pt --suite libero_spatial \
  --episodes 24 --workers 6 --policy haloact \
  --out /home/tione/notebook/home/arianliu/data/halo-wam/datasets/rollouts/act60k \
  > logs/collect_act60k.log 2>&1

# 3) E1-v3：成功 vs 失败 rollout（同分布）+ 动作错配负样本，episode 级留出，AUC 选模
python scripts/train_e1_v3.py --steps 1200 \
  --rollouts /home/tione/notebook/home/arianliu/data/halo-wam/datasets/rollouts/act60k \
  > logs/train_e1v3.log 2>&1

echo DONE > logs/e1_v3.done
echo "$(date +%H:%M) pipeline_e1v3 complete" >> logs/pipeline_e1v3.log
