# HALO-WAM 研究计划

**HALO: Hallucination-Aware Latent wOrld model — 无仿真 VLA 验证与后训练**

> 版本：v1.0（2026-10-02）
> 仓库：github.com/Algorineko/halo-wam ｜ 模型：ModelScope 同名
> 投稿目标：**CoRL 2027**（主，约 2027-02/03 截稿）/ IROS 2027（回退，约 2027-03）

---

## 1. 背景与文献缺口（2025-2026 调研结论）

WM-as-Verifier / 无仿真 VLA 后训练是当前最热赛道，但所有已发表工作共同报告两个失败模式：

| 失败模式 | 证据 |
|---|---|
| ① 幻觉 rollout 上的 reward hacking | WMPO（arXiv:2511.09515, ICLR 26）的后续 HaWMPO 明确展示 WM 幻觉直接导致奖励被劫持 |
| ② verifier 在 OOD 上过自信 | VLA-RFT（OpenReview）被审稿人批评 reward hacking 且无真机验证 |

关键现状：**廉价、不确定性感知、隐空间（非生成式）的 verifier 尚不存在**。

同时存在一个 2×2 矩阵空格：{生成式 WM vs JEPA} × {游戏 vs 真实机器人}——
"真实机器人数据上训练的 JEPA 世界模型内做闭环策略学习/后训练"无人完成：
- V-JEPA 2-AC（2506.09985）：62h 机器人数据 action-conditioning，但停在 zero-shot 规划
- LeWM（2603.19312）/ NE-Dreamer（2603.02765）：稳定端到端 JEPA，但只在玩具任务/游戏
- DiWA（2508.03645, CoRL 25）：WM 内离线 RL，但用小像素 WM、任务特定
- CoVer（Stanford）：verifier 带来 45% 真机提升，但 verifier 是另一个 VLM、无动力学、无不确定性
- PLaW-VLA（CoRL 投稿，已精读）：LWM 是 **action-free** 的，无法做反事实 rollout/验证 —— 本课题直接补此缺口

## 2. 方法

HALO = frozen V-JEPA 2 编码器 + 三个轻量头（全隐空间，无像素解码）：

```
观测帧序列 o_{t-h:t} ──frozen V-JEPA 2──▶ latent z_{t-h:t}
                                            │
        ┌───────────────────────────────────┼──────────────────────────┐
        ▼                                   ▼                          ▼
 动态头（action-conditioned）          奖励/成功头               ensemble 分歧
 ẑ_{t+k} = f(z_{≤t}, a_{t:t+k})      R̂ = g(z_{≤t}, ẑ, 指令)      u = Var_k[f_k] （认知不确定性）
        │                                   │                          │
        └─────────────── 门控：u > τ 的 rollout 截断/降权 ──────────────┘
```

- **动态头**：action-conditioned JEPA 预测（补 PLaW-VLA 缺口），训练数据 Bridge V2 + LIBERO demos + RoboMIND
- **奖励头**：IRL-VLA 式（2508.06571），用 **RoboMIND 失败标注** + Bridge rollout（成功/失败段）训练
- **不确定性**：deep ensemble（K 个动态头副本）分歧 → 幻觉检测器

两个用途（论文两层贡献）：
1. **测试时验证（E2）**：VLA 采样 N 个 action chunk → HALO 隐空间 rollout 打分选优（CoVer/RoboMonkey 式 best-of-N，免重训练）
2. **不确定性门控 GRPO 后训练（E3）**：在 HALO 想象中 rollout 供奖励，高不确定性处截断/降权 → 治 WMPO/HaWMPO 的 reward hacking

策略模型：smolvla_base（450M，先行）→ pi05_base（后续）。

## 3. 与在研项目区分（隔离关系）

| 项目 | 主题 | 关系 |
|---|---|---|
| safi-wam | WAM 推理效率（语义辅助+按需未来推理） | 正交；可复用其 LIBERO/LIBERO-plus 评测经验 |
| tempo-pai | 自适应推理频率 | 正交 |
| HG-FM-TTT | 部署时免训练适配 | 正交 |
| **halo-wam** | **世界模型作为验证与训练信号** | 全新代码库，依赖独立（不共享 conda/venv），仅参考文档经验 |

## 4. 算力与数据

- 算力：2×BW1000 64G（own，可双卡 DDP 或单卡分任务）；384 核 CPU（MuJoCo 评测农场）；4090 **不可用**（被占用），全程无 N 卡仿真
- 软件栈：`source /home/tione/notebook/home/arianliu/env/hcu_env.sh` → torch 2.4.1+das.opt2.dtk2504；HF_ENDPOINT=https://hf-mirror.com；HF_HUB_ENABLE_HF_TRANSFER=0；DTK 无 flash-attn → SDPA 回退；≤2-3B 全参 / 7B LoRA / bf16
- 数据（hf-mirror 下载至 ~/data/halo-wam/）：
  - Bridge V2（60k 轨迹，含 rollout）— 动态头 + 奖励头
  - RoboMIND（107k 轨迹，**带失败标注**）— 奖励头金矿
  - LIBERO demos — 下游评测任务训练
  - V-JEPA 2（Meta, 1.2B，frozen）— 编码器

## 5. 评测路径

- **主**：LIBERO / LIBERO-PRO（2510.03827，所有 VLA 从 >0.9 崩到 ~0 的诚实基准）/ LIBERO-plus —— MuJoCo CPU（osmesa 软渲染），384 核 worker 池并行
- **verifier 离线效度**：RoboMIND 失败检测 AUC；幻觉注入检测（故意喂错误动作）；策略成功率排序相关性（与已发表真机数字对比）
- **真机**：合作资源补实验（W17-20）
- 效率报告：wall-clock / FLOPs（测试时验证的开销必须报告）

## 6. 里程碑（W1 = 2026-10-06 当周）

| 阶段 | 周次 | 内容 | 交付物 |
|---|---|---|---|
| M0 基建 | W1-2 | 建仓、V-JEPA 2 在 BW1000 冒烟（SDPA）、LIBERO CPU 评测农场、数据下载 | 冒烟报告 + 评测管线 |
| M1 动态头 (E0) | W3-6 | action-conditioned 动态头训练；离线 rollout 保真度；幻觉注入检测基准 | E0 报告 |
| M2 verifier (E1) | W7-10 | 奖励头 + ensemble 门控；效度验证（AUC、排序相关性） | E1 报告 |
| M3 测试时验证 (E2) | W9-12 | best-of-N on LIBERO 三件套 vs CoVer/RoboMonkey 基线 | E2 报告 |
| M4 想象后训练 (E3) | W13-16 | 不确定性门控 GRPO（smolvla 先行） | E3 报告 |
| M5 真机+写作 | W17-20 | 真机实验、论文撰写 | CoRL 2027 投稿 |

**回退线**：M4 不稳 → "verifier 门控数据引擎"（给 OXE/Bridge 打分/挖掘 near-failure 恢复段/重加权训练）作第二贡献投 IROS 2027。
**备选课题 B**（第二篇）：Anytime Latent Thinking（自适应深度隐空间推理+停机头，LIBERO-PRO 鲁棒性），与本项目共享基建。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| V-JEPA 2 在 DTK 跑不通 | W1 即冒烟；不行则换 DINOv2/自定义 ViT 编码器（思路不变） |
| GRPO 在想象中不稳定 | 回退线（数据引擎）；先冻结策略只做测试时验证 |
| LIBERO CPU 评测慢 | 384 核 worker 池 + 批量并行；参考 safi-wam 经验 |
| 离线指标与在线成功率相关性差 | 以 CPU 在线评测为主，离线指标只作 verifier 效度检验 |

## 8. 工作流约定

- 每完成一个里程碑/子步骤 → git commit + push GitHub 存档
- 模型/权重 → ModelScope（`modelscope upload`，令牌只走环境变量 `MODELSCOPE_API_TOKEN`，见 `.env.secrets.example`，**严禁入库**）
- 数据 → `/home/tione/notebook/home/arianliu/data/halo-wam/`（与 safi-wam 等项目数据隔离）
- 与其他项目隔离：独立目录、不共享虚拟环境、文档互不引用代码
