#!/bin/bash
# 第五段流水线（主结果多 suite 扩展）：libero_object / libero_goal 两 suite ×
# {基线, 纯不确定性验证器} 四臂，10-eps 与 R3 同协议——把 +73% 主张做成三 suite 表
# act60k / e0v5 均为 130 任务全量训练，vocab 直接覆盖
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

EVAL() {  # suite policy tag
  python scripts/eval_libero_parallel.py --suite "$1" --episodes 10 --workers 2 \
    --policy "$2" --ckpt checkpoints/act60k/best.pt \
    ${3:+--verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5} \
    > "logs/eval_ms_$4.log" 2>&1
  grep "平均成功率" "logs/eval_ms_$4.log" >> logs/pipeline5.log
}

echo "$(date +%H:%M) multi-suite start" >> logs/pipeline5.log
EVAL libero_object haloact  ""  obj_base
EVAL libero_object haloact_v v   obj_ver
EVAL libero_goal   haloact  ""  goal_base
EVAL libero_goal   haloact_v v   goal_ver
echo DONE > logs/pipeline5.done
echo "$(date +%H:%M) pipeline5 complete" >> logs/pipeline5.log
