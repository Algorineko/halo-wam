#!/bin/bash
# 第六段流水线（E3-R2 迭代蒸馏）：×4 状态（win_per_demo 16）+ 高边际过滤（p50）
# → SFT 热启 → spatial 10-eps 评测。判定"数据规模×选择质量"能否破 R1 平价。
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

echo "$(date +%F\ %H:%M) pipeline6 E3-R2 start" >> logs/pipeline6.log

# ① 数据生成：win_per_demo 16（R1 的 4 倍 → ~7200 状态），卡 1（脚本内 dev=cuda:1）
python scripts/gen_e3_data.py --win-per-demo 16 \
  --out /home/tione/notebook/home/arianliu/data/halo-wam/datasets/e3_distill_r2/train.h5 \
  > logs/gen_e3r2.log 2>&1
grep -E "states|keep" logs/gen_e3r2.log | tail -3 >> logs/pipeline6.log || true

# ② SFT 热启（margin>p50 高边际过滤，8000 步，act60k 热启）
python scripts/train_e3_sft.py \
  --data /home/tione/notebook/home/arianliu/data/halo-wam/datasets/e3_distill_r2/train.h5 \
  --margin-pmin 0.5 --out checkpoints/act_e3_r2 \
  > logs/sft_e3r2.log 2>&1
tail -2 logs/sft_e3r2.log >> logs/pipeline6.log || true

# ③ 评测：spatial 10-eps，纯 CPU（策略 haloact，2 worker）
python scripts/eval_libero_parallel.py --suite libero_spatial --episodes 10 --workers 2 \
  --policy haloact --ckpt checkpoints/act_e3_r2/best.pt \
  > logs/eval_e3r2.log 2>&1
grep "平均成功率" logs/eval_e3r2.log >> logs/pipeline6.log || true

echo DONE > logs/pipeline6.done
echo "$(date +%F\ %H:%M) pipeline6 complete" >> logs/pipeline6.log
