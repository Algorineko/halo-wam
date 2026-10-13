#!/bin/bash
# pipeline8：等 pipeline7（λ=1.0）完成后，双卡分工低内存并行（合计 ~21GB < 28 红线）
#   卡0：λ=0.5 复跑（1 worker，验证 22% 是否真实）
#   卡1：E3-R3 数据生成（以 act_e3_r2 为候选策略的迭代轮，win16）
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

while [ ! -f logs/pipeline7.done ]; do sleep 60; done
echo "$(date +%F\ %H:%M) pipeline8 start (dual-card)" >> logs/pipeline8.log

HIP_VISIBLE_DEVICES=0 python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 1 \
  --policy haloact_h --ckpt checkpoints/act60k/best.pt \
  --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 \
  --reward-ckpt checkpoints/e1_v3/reward_head.pt --lambda-u 0.5 \
  > logs/eval_lambda05_rep.log 2>&1 &
PID_A=$!
python scripts/gen_e3_data.py --win-per-demo 16 --policy checkpoints/act_e3_r2/best.pt \
  --out /home/tione/notebook/home/arianliu/data/halo-wam/datasets/e3_distill_r3/train.h5 \
  > logs/gen_e3r3.log 2>&1 &
PID_B=$!
wait $PID_A $PID_B
grep "平均成功率" logs/eval_lambda05_rep.log >> logs/pipeline8.log || true
echo DONE > logs/pipeline8.done
echo "$(date +%F\ %H:%M) pipeline8 complete" >> logs/pipeline8.log
