#!/bin/bash
# pipeline9：E3-R3 SFT（act_e3_r2 热启，margin>p50）→ spatial 10-eps 评测
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4
echo "$(date +%F\ %H:%M) pipeline9 R3 start" >> logs/pipeline9.log

python scripts/train_e3_sft.py \
  --data /home/tione/notebook/home/arianliu/data/halo-wam/datasets/e3_distill_r3/train.h5 \
  --margin-pmin 0.5 --out checkpoints/act_e3_r3 \
  > logs/sft_e3r3.log 2>&1
tail -2 logs/sft_e3r3.log >> logs/pipeline9.log || true

python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
  --policy haloact --ckpt checkpoints/act_e3_r3/best.pt \
  > logs/eval_e3r3.log 2>&1
grep "平均成功率" logs/eval_e3r3.log >> logs/pipeline9.log || true
echo DONE > logs/pipeline9.done
echo "$(date +%F\ %H:%M) pipeline9 complete" >> logs/pipeline9.log
