# HALO-WAM

**HALO: Hallucination-Aware Latent wOrld model — 无仿真 VLA 验证与后训练**

在冻结的 V-JEPA 2 隐空间上构建 action-conditioned 世界模型 + 奖励头 + 认知不确定性门控，用作：
1. **测试时验证器**——VLA 动作 chunk 的 best-of-N 打分选择（免重训练）
2. **想象训练场**——不确定性门控 GRPO 后训练，治 WM-as-Verifier 赛道的 reward hacking / verifier OOD 过自信

全程无 N 卡仿真：训练 2×Hygon BW1000，评测 MuJoCo CPU + 真机合作。

## 研究计划

见 [docs/plan.md](docs/plan.md)（含文献缺口、方法、里程碑 W1-W20、风险与回退）。

## 目录结构（随进展更新）

```
docs/        # 研究计划、实验报告（E0-E3）
scripts/     # 环境、数据、训练、评测脚本
src/halo/    # 核心代码：encoder 封装、动态头、奖励头、不确定性、GRPO
configs/     # 训练/评测配置
```

## 快速开始

```bash
# 海光 DCU 环境（torch 2.4.1+das.opt2.dtk2504）
source /home/tione/notebook/home/arianliu/env/hcu_env.sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

# ModelScope 令牌（严禁提交）
cp .env.secrets.example .env.secrets  # 填入 MODELSCOPE_API_TOKEN
set -a; source .env.secrets; set +a
```

## 状态

- [x] M0 建仓与计划（2026-10-02）
- [ ] M0 V-JEPA 2 BW1000 冒烟
- [ ] M0 LIBERO CPU 评测农场
- [ ] E0-E3 见 docs/plan.md 里程碑表
