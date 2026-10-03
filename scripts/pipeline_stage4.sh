#!/bin/bash
# 第四段流水线（E3-A BoN 蒸馏）：等 pipeline3.done →
# ① gen_e3_data（demo 状态 × 8 候选 × HALO 选优，card 1）→
# ② train_e3_sft（act60k 热启微调，card 1）→
# ③ act_e3 基线评测（10 eps，与 R3 可比；成功判据：蒸馏策略裸跑 > act60k 基线 11.0%）
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

while [ ! -f logs/pipeline3.done ]; do sleep 120; done
echo "$(date +%H:%M) pipeline3 done -> E3-A gen" >> logs/pipeline4.log

python scripts/gen_e3_data.py --n-cand 8 --reward-ckpt checkpoints/e1_v3/reward_head.pt \
  > logs/gen_e3.log 2>&1
tail -1 logs/gen_e3.log >> logs/pipeline4.log

python scripts/train_e3_sft.py --steps 8000 > logs/train_e3sft.log 2>&1
tail -1 logs/train_e3sft.log >> logs/pipeline4.log

python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
  --policy haloact --ckpt checkpoints/act_e3/best.pt > logs/eval_e3_base.log 2>&1
grep "平均成功率" logs/eval_e3_base.log >> logs/pipeline4.log

echo DONE > logs/pipeline4.done
echo "$(date +%H:%M) pipeline4 complete" >> logs/pipeline4.log
