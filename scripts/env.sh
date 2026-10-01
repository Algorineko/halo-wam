#!/bin/bash
# HALO-WAM 项目环境（source 本脚本）
# 用法: source scripts/env.sh
export HALO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HALO_DATA=/home/tione/notebook/home/arianliu/data/halo-wam
export HALO_VENV=/home/tione/notebook/home/arianliu/venvs/halo-wam

# 海光 DCU 运行时
source /home/tione/notebook/home/arianliu/env/hcu_env.sh

# HuggingFace 镜像（勿开 hf_transfer，勿升级 huggingface_hub 1.x）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

# ModelScope 令牌（.env.secrets 已 gitignore）
if [ -f "$HALO_ROOT/.env.secrets" ]; then
  set -a; source "$HALO_ROOT/.env.secrets"; set +a
fi

# LIBERO 隔离配置（预生成，避免交互式询问；datasets 指向项目数据目录）
export LIBERO_CONFIG_PATH=/home/tione/notebook/home/arianliu/data/halo-wam/libero_config

# MuJoCo CPU 离屏渲染（OSMesa 软渲染，无 N 卡/EGL 设备场景；libosmesa6 已 apt 安装）
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1

# 项目 venv（若存在则激活；基于 hcu python3.10 --system-site-packages，隔离于其他项目）
if [ -f "$HALO_VENV/bin/activate" ]; then
  source "$HALO_VENV/bin/activate"
fi

mkdir -p "$HALO_DATA"/{models,datasets,logs} "$HALO_ROOT"/logs
