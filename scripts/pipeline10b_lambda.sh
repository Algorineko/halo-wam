#!/bin/bash
# pipeline10b：λ 族 50-eps（card0，等 goal_v50 完成后接续；方向已实证：argmin 低=好）
set +u  # dtk env.sh 有未绑定变量，禁 -u（skill 铁律）
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4
while [ ! -f logs/p10/goal_v50.done ]; do sleep 120; done

LAM() {  # lambda tag
  [ -f "logs/p10/$2.done" ] && return 0
  echo "$(date +%F\ %H:%M) start $2" >> logs/p10/pipeline10.log
  HIP_VISIBLE_DEVICES=0 python scripts/eval_libero_parallel.py \
    --suite libero_spatial --episodes 50 --workers 1 \
    --policy haloact_h --ckpt checkpoints/act60k/best.pt \
    --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 \
    --reward-ckpt checkpoints/e1_v3/reward_head.pt --lambda-u "$1" --log-decisions \
    > "logs/p10/$2.log" 2>&1 \
    && echo DONE > "logs/p10/$2.done" \
    && grep "平均成功率" "logs/p10/$2.log" >> logs/p10/pipeline10.log \
    || echo "FAIL $2" >> logs/p10/pipeline10.log
}
LAM 0.25 lam025_v50
LAM 0.5  lam05_v50
LAM 1.0  lam10_v50
LAM 0    reward_only50
echo DONE > logs/p10/lambda.done
