#!/bin/bash
# λ 敏感性干净补臂：v3 诚实头，λ∈{0.5,1.0}（0.25 已有 130415），10-eps 协议
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4
for LAM in 0.5 1.0; do
  python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
    --policy haloact_h --ckpt checkpoints/act60k/best.pt \
    --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 \
    --reward-ckpt checkpoints/e1_v3/reward_head.pt --lambda-u $LAM \
    > "logs/eval_lambda_$LAM.log" 2>&1
  grep "平均成功率" "logs/eval_lambda_$LAM.log" >> logs/pipeline7.log || true
done
echo DONE > logs/pipeline7.done
