#!/bin/bash
# pipeline2 续跑（v3 评测已完成 17.0%）：act_dino 训练 → 评测 → 交棒 pipeline3/4
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

echo "$(date +%H:%M) resume: act_dino train" >> logs/pipeline2.log
python scripts/train_act_dino.py --steps 15000 > logs/train_act_dino.log 2>&1
echo "$(date +%H:%M) act_dino trained -> eval" >> logs/pipeline2.log

python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
  --policy haloact_d --ckpt checkpoints/act_dino/best.pt > logs/eval_act_dino_base.log 2>&1
grep "平均成功率" logs/eval_act_dino_base.log >> logs/pipeline2.log

echo DONE > logs/pipeline2.done
echo "$(date +%H:%M) pipeline2 complete (resumed)" >> logs/pipeline2.log
