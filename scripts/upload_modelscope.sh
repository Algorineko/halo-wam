#!/bin/bash
# HALO-WAM 模型上传 ModelScope（令牌走 .env.secrets 的 MODELSCOPE_API_TOKEN，严禁入库）
# 用法: bash scripts/upload_modelscope.sh <本地目录> <repo内路径>
# 例: bash scripts/upload_modelscope.sh checkpoints/e0v5 e0v5/dyn_ens_k5_s2000
set -e
HALO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HALO_ROOT/scripts/env.sh"

LOCAL="$1"
REMOTE="${2:-$(basename "$LOCAL")}"
REPO="Algorineko/halo-wam"   # ModelScope 模型仓库（首次上传自动创建需网页建仓，或用 sdk create）

if [ ! -e "$LOCAL" ]; then echo "not found: $LOCAL"; exit 1; fi
modelscope upload "$REPO" "$LOCAL" "$REMOTE" --token "$MODELSCOPE_API_TOKEN"
echo "uploaded $LOCAL -> $REPO/$REMOTE"
