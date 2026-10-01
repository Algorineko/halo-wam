# HALO-WAM 进度追踪（自监督 tick 的状态源）

> 本文件是活文档：每个 cron tick / 工作会话后更新。计划见 [plan.md](plan.md)。

## 当前状态（更新于 2026-10-02 00:30）

**阶段：M0 基建（W1-2）**

### ✅ 已完成
- [x] 建仓 github.com/Algorineko/halo-wam，推送初始骨架（commit c24b0d2）
- [x] DCU 环境验证：`source /home/tione/notebook/home/arianliu/env/hcu_env.sh && python3.10`；torch 2.4.1+das.opt2.dtk2504，2×BW1000_H 68.7GB，bf16 matmul ✅ SDPA ✅（无 flash-attn，SDPA 回退可行）
- [x] ModelScope 令牌就位（.env.secrets，gitignored）
- [x] 自监督 cronjob（每小时 :07/:52）

### 🔄 运行中 / 进行中
- [ ] V-JEPA 2 下载（hf-mirror → ~/data/halo-wam/models/，日志 logs/dl_vjepa2.log）

### ⏭️ 下一步（按序）
1. V-JEPA 2 冻结推理冒烟（BW1000，ViT-L 先行）：随机视频张量 → encoder → latent 形状/耗时
2. LIBERO CPU 评测农场：mujoco(osmesa) + robosuite + LIBERO 装进独立 venv（基于 hcu python3.10 --system-site-packages，保持项目隔离），冒烟 1 任务 1 episode
3. 数据下载：Bridge V2、RoboMIND（失败标注）、LIBERO demos
4. M1 E0 开工：action-conditioned 动态头代码骨架

### 📌 备注
- 隔离：venv 用 ~/venvs/halo-wam；数据 ~/data/halo-wam/；不动 safi-wam/tempo-pai 的环境与数据
- HF: HF_ENDPOINT=https://hf-mirror.com，HF_HUB_ENABLE_HF_TRANSFER=0，勿升级 huggingface_hub 1.x
- V-JEPA 2 HF id 候选：facebook/vjepa2-vit-l-32（先）、facebook/vjepa2-aco-vit-l-32（action-conditional 参考）、facebook/vjepa2-vit-g-384（大，后）
