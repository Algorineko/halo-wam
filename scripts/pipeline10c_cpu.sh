#!/bin/bash
# pipeline10c：50-eps CPU base 臂（纯 CPU 策略 ~2GB/worker，2 并行保守内存）
set +u
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

CELL() {  # suite policy ckpt tag
  [ -f "logs/p10/$4.done" ] && return 0
  echo "$(date +%F\ %H:%M) start $4" >> logs/p10/pipeline10.log
  python scripts/eval_libero_parallel.py --suite "$1" --episodes 50 --workers 1 \
    --policy "$2" --ckpt "$3" > "logs/p10/$4.log" 2>&1 \
    && echo DONE > "logs/p10/$4.done" \
    && grep "平均成功率" "logs/p10/$4.log" >> logs/p10/pipeline10.log \
    || echo "FAIL $4" >> logs/p10/pipeline10.log
}
{
  CELL libero_spatial haloact checkpoints/act60k/best.pt sp_base50
  CELL libero_goal   haloact checkpoints/act60k/best.pt goal_base50
  CELL libero_spatial haloact checkpoints/act_e3_r2/best.pt r2_base50
} &
A=$!
{
  CELL libero_object haloact checkpoints/act60k/best.pt obj_base50
  CELL libero_spatial haloact checkpoints/act_e3_r3/best.pt r3_base50
  CELL libero_spatial haloact_c checkpoints/act60k/best.pt cons50
} &
B=$!
wait $A $B
echo DONE > logs/p10/cpu.done
echo "$(date +%F\ %H:%M) pipeline10c CPU streams complete" >> logs/p10/pipeline10.log
