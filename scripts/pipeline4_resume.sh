#!/bin/bash
# pipeline4 续跑（v2）：SFT（collate bug 已修）→ act_e3 裸跑评测
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

echo "$(date +%H:%M) resume: e3 sft train" >> logs/pipeline4.log
python scripts/train_e3_sft.py --steps 8000 > logs/train_e3sft.log 2>&1
tail -1 logs/train_e3sft.log >> logs/pipeline4.log

python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
  --policy haloact --ckpt checkpoints/act_e3/best.pt > logs/eval_e3_base.log 2>&1
grep "平均成功率" logs/eval_e3_base.log >> logs/pipeline4.log

echo DONE > logs/pipeline4.done
echo "$(date +%H:%M) pipeline4 complete (resumed)" >> logs/pipeline4.log
