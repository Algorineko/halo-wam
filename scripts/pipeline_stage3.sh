#!/bin/bash
# 第三段流水线（LIBERO-PRO 泛化验证，W13 里程碑）：等 pipeline2.done →
# swap（位置互换）/ object（外观替换）两轴 × {基线, 纯不确定性验证器} 四臂评测。
# 清洁基线数字沿用 R3（10 eps：11.0% / 19.0%），此处只跑扰动轴。
# 扰动协议：configs/libero_pro_spatial_{swap,object}.json（goal 不变、场景几何/外观变化）
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

while [ ! -f logs/pipeline2.done ]; do sleep 120; done
echo "$(date +%H:%M) pipeline2 done -> LIBERO-PRO evals" >> logs/pipeline3.log

EVAL() {  # tag spec policy extra...
  local log="logs/eval_pro_$1.log"
  python scripts/eval_libero_parallel.py --suite "libero_pro_$1" \
    --task-spec "configs/libero_pro_spatial_$2.json" \
    --episodes 10 --workers 2 --policy "$3" --ckpt checkpoints/act60k/best.pt \
    ${4:+--verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5} \
    > "$log" 2>&1
  grep "平均成功率" "$log" >> logs/pipeline3.log
}

# 1) swap 轴：基线
EVAL swap_base swap haloact
# 2) swap 轴：验证器
EVAL swap_ver swap haloact_v "v"
# 3) object 轴：基线
EVAL obj_base object haloact
# 4) object 轴：验证器
EVAL obj_ver object haloact_v "v"

echo DONE > logs/pipeline3.done
echo "$(date +%H:%M) pipeline3 complete" >> logs/pipeline3.log
