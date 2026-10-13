#!/bin/bash
# pipeline10：审稿返修 50-eps 统计引擎（Phase 1，配对检验数据采集）
# 双卡各 1 worker 串行队列（保守内存 ~22GB < 28 红线）；每格幂等 done 标记
# λ 族（0.25/0.5/1.0/reward-only）待 reward 头方向验证后由 cron 追加到 card0 队列
set -u
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4
mkdir -p logs/p10

CELL() {  # suite policy ckpt tag card [extra...]
  local suite=$1 pol=$2 ckpt=$3 tag=$4; shift 4
  [ -f "logs/p10/${tag}.done" ] && { echo "skip ${tag}"; return 0; }
  echo "$(date +%F\ %H:%M) start ${tag}" >> logs/p10/pipeline10.log
  HIP_VISIBLE_DEVICES=$card python scripts/eval_libero_parallel.py \
    --suite "$suite" --episodes 50 --workers 1 \
    --policy "$pol" --ckpt "$ckpt" \
    --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 --log-decisions \
    "$@" > "logs/p10/${tag}.log" 2>&1 \
    && echo DONE > "logs/p10/${tag}.done" \
    && grep "平均成功率" "logs/p10/${tag}.log" >> logs/p10/pipeline10.log \
    || echo "$(date +%F\ %H:%M) FAIL ${tag}" >> logs/p10/pipeline10.log
}

card=0
{
  CELL libero_spatial haloact_v checkpoints/act60k/best.pt sp_v50
  CELL libero_goal    haloact_v checkpoints/act60k/best.pt goal_v50
} &
PID0=$!

card=1
{
  CELL libero_object  haloact_v checkpoints/act60k/best.pt obj_v50
  CELL libero_spatial haloact_v checkpoints/act_e3_r2/best.pt r2_bon50
  CELL libero_spatial haloact_v checkpoints/act_e3_r3/best.pt r3_bon50
  CELL libero_spatial haloact_r checkpoints/act60k/best.pt rand50
} &
PID1=$!

wait $PID0 $PID1
echo DONE > logs/p10/all_gpu.done
echo "$(date +%F\ %H:%M) pipeline10 GPU streams complete" >> logs/p10/pipeline10.log
