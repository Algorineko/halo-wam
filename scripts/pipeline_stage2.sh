#!/bin/bash
# 第二段流水线：等 e1_v3.done → ① haloact_h 用 v3 奖励头复测（核心问题：修好的奖励头能否超过纯验证器）
# → ② act_dino 训练（DINOv2 基线强化）→ ③ act_dino 评测。全部串行，评测 workers 2。
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

while [ ! -f logs/e1_v3.done ]; do sleep 60; done
echo "$(date +%H:%M) e1_v3 done -> halo_h v3 re-eval" >> logs/pipeline2.log

# ① haloact_h 复测：唯一变量是奖励头 v1_v2 -> v3（λ 保持 0.25 与 R2/R3 可比）
python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
  --policy haloact_h --ckpt checkpoints/act60k/best.pt \
  --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 \
  --reward-ckpt checkpoints/e1_v3/reward_head.pt --lambda-u 0.25 \
  > logs/eval_r4_halo_v3.log 2>&1

echo "$(date +%H:%M) halo_h v3 done -> act_dino train" >> logs/pipeline2.log

# ② act_dino 训练（card 1，DINOv2 冻结特征基线）
CUDA_VISIBLE_DEVICES=1 python scripts/train_act_dino.py --steps 15000 \
  > logs/train_act_dino.log 2>&1

echo "$(date +%H:%M) act_dino trained -> act_dino eval" >> logs/pipeline2.log

# ③ act_dino 基线评测（10 eps，与 R3 同协议）
python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
  --policy haloact_d --ckpt checkpoints/act_dino/best.pt \
  > logs/eval_act_dino_base.log 2>&1

echo DONE > logs/pipeline2.done
echo "$(date +%H:%M) pipeline2 complete" >> logs/pipeline2.log
