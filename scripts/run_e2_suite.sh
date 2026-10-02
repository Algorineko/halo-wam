#!/bin/bash
# E2 套件一键执行：基线评测 → E2 验证评测 → E1 数据采集（串行，CPU 10 workers）
# 用法: bash scripts/run_e2_suite.sh <smolvla_ckpt_dir> [verifier_ckpt]
# 例:   bash scripts/run_e2_suite.sh runs/smolvla_libero_spatial/checkpoints/002500 checkpoints/e0v5/dyn_ens_k5_s2000.pt
set -e
HALO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HALO_ROOT/scripts/env.sh"
CKPT="${1:?需要 SmolVLA checkpoint 目录}"
VERIFIER="${2:-checkpoints/e0v5/dyn_ens_k5_s2000.pt}"
TAG=$(basename "$CKPT")
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1

echo "[e2-suite] 1/3 基线评测 (smolvla) $TAG"
python "$HALO_ROOT/scripts/eval_libero_parallel.py" --suite libero_spatial \
  --episodes 5 --workers 10 --policy smolvla --ckpt "$CKPT"

echo "[e2-suite] 2/3 E2 验证评测 (smolvla_v, best-of-5 + $VERIFIER)"
python "$HALO_ROOT/scripts/eval_libero_parallel.py" --suite libero_spatial \
  --episodes 5 --workers 5 --policy smolvla_v --ckpt "$CKPT" --verifier "$VERIFIER" --n-cand 5

echo "[e2-suite] 3/3 E1 rollout 采集 -> $HALO_DATA/datasets/rollouts/smolvla_$TAG"
python "$HALO_ROOT/scripts/collect_rollouts.py" --ckpt "$CKPT" --suite libero_spatial \
  --episodes 5 --workers 10 --out "$HALO_DATA/datasets/rollouts/smolvla_$TAG"

echo "[e2-suite] DONE — 结果在 eval_results/ 与 datasets/rollouts/"
