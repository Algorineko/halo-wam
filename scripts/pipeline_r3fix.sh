#!/bin/bash
# R3 halo 臂 task7/8 补跑（v2）：等 pipeline2 的 halo_h v3 评测出结果后，workers=1 串行补跑
# 标记：eval_results 下出现比 e1_v3.done 更新的 haloact_h JSON（即 pipeline2 首个评测完成）
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

while [ ! -f logs/e1_v3.done ]; do sleep 120; done
while ! find eval_results -name "*haloact_h*" -newer logs/e1_v3.done | grep -q .; do sleep 120; done
echo "$(date +%H:%M) halo_h v3 eval done -> rerun halo task7,8 (workers=1)" >> logs/pipeline_r3fix.log

python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 1 \
  --policy haloact_h --ckpt checkpoints/act60k/best.pt \
  --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 \
  --reward-ckpt checkpoints/e1_v2/reward_head.pt --lambda-u 0.25 --task-ids 7,8 \
  > logs/eval_r3_halo_fix2.log 2>&1
grep "平均成功率" logs/eval_r3_halo_fix2.log >> logs/pipeline_r3fix.log
echo DONE > logs/r3fix.done
