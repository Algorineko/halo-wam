# HALO-WAM 进度追踪（自监督 tick 的状态源）

> 本文件是活文档：每个 cron tick / 工作会话后更新。计划见 [plan.md](plan.md)。

## 当前状态（更新于 2026-10-02 00:52）

**阶段：M0 基建（W1-2）—— 核心冒烟全部通过 ✅**

### ✅ 已完成
- [x] 建仓 github.com/Algorineko/halo-wam，推送骨架（c24b0d2）
- [x] DCU 验证：torch 2.4.1+das.opt2.dtk2504，2×BW1000_H 68.7GB，bf16 matmul ✅ SDPA ✅
- [x] ModelScope 令牌就位（.env.secrets，gitignored）
- [x] 自监督 cronjob（每小时 :07/:52，durable）
- [x] **V-JEPA 2 ViT-L（326M）BW1000 冒烟通过**：bf16+SDPA；输入 (B,T,3,256,256) 帧维在前；T=16: 103ms/2clips → hidden (2,2048,1024)；T=64: 265ms → (1,8192,1024)。模型 6.0GB 已下到 ~/data/halo-wam/models/vjepa2-vitl-fpc64-256
- [x] **LIBERO CPU 评测管线打通**：OSMesa 软渲染（libosmesa6 已 apt 装；MUJOCO_GL=osmesa + PYOPENGL_PLATFORM=osmesa）；libero_spatial 任务创建+init states(50)+step+渲染全通；env 创建 3.1s，161ms/step(含渲染)；画面统计验证为真实场景
- [x] venv ~/venvs/halo-wam（hcu python3.10 + system-site-packages，numpy 钉 1.26.4）

### 🛠️ 踩坑记录（复用价值）
- LIBERO PyPI wheel(0.1.1) 不含 benchmark 套件，必须用 git 仓库（ghfast.top 代理克隆秒成，SSH 直连慢）；仓库顶层 libero/ 无 __init__.py，pip editable 映射为空 → 本地 touch libero/__init__.py 修复
- LIBERO_CONFIG_PATH 指到 data/halo-wam/libero_config/ 预写 config.yaml，避免交互式询问
- robosuite EGL 在无 N 卡/海光 EGL 缺失时 PLATFORM_DEVICE 报错 → OSMesa 路线
- LIBERO env API：OffScreenRenderEnv(bddl_file_name=..., camera_heights=256, camera_widths=256)；无 action_space/_get_obs，用 env.env.action_dim；obs 从 reset/step 返回
- PyPI 直连慢，用清华镜像 -i https://pypi.tuna.tsinghua.edu.cn/simple；gym 用 0.26.2（0.21 与新 pip 不兼容）

### 🔬 E0 前置发现（2026-10-02，重要协议决策）
- **V-JEPA 2 latent 特性**（scripts/validate_encoder.py 实测）：
  1. 确定性 cos=1.0000（冻结无 dropout）
  2. **强上下文/位置敏感**：同像素内容在不同 clip 偏移下 latent cos 仅 0.546（V-JEPA 2 无绝对位置嵌入、RoPE+双向注意力所致）
  3. 时间局部性信号弱但存在（相邻步 0.470 vs 远距 0.437，Δ0.033）
- **协议决策：E0 动态头采用固定窗口相对编码** —— context 窗口与 future 窗口各自独立编码（tubept 均从相对位置 0 起），与 V-JEPA 2-AC / PLaW-VLA 的 future query 协议一致。禁止把"同帧不同偏移应同 latent"当作假设
- token 排布（源码证实）：conv3d → flatten(2) → (B, T'*16*16, D)，(T',H',W') row-major，reshape (B,T',16,16,D) 正确

### 🔄 运行中 / 进行中（2026-10-02 01:15 更新）
- **E0 K=5 训练**（GPU0, logs/train_e0_k5_s2000_v3.log）：0.97s/step, loss 2.56→1.60↓, cos→0.36↓, ~33min
- LIBERO demos 下载（4 套件慢速推进，logs/dl_libero_all.log）
- RoboMIND 公开包下载（bread_in_basket 3.5G/34files + bread_on_table，LeRobot 格式）

### ✅ E0 代码全链路打通（2026-10-02 01:00）
- `src/halo/data/libero_h5.py`：固定窗口采样管线（13 h5 → 2600 样本，随下载增长）
- `src/halo/dynamics.py`：LatentDynamicsHead（future queries 交叉注意 [ctx;act] tokens）+ DynamicsEnsemble（K 头 / jepa_loss / epistemic_std）
- `scripts/train_e0.py`：完整训练循环 + 幻觉注入自检（错配动作 vs 正确动作的 ensemble 分歧比）
- `scripts/eval_libero_parallel.py`：**384 核评测农场验证通过**（10 tasks×3eps 并行 288s；随机策略 0% 基线；64 workers 下整套 ~12min）

### 🛠️ 新增踩坑
- **DTK 上 SDPA 无融合核**：fp32 下 nn.TransformerDecoder 激活物化注意力矩阵，K=5×6层×2attn ≈ 32GB → OOM。对策：heads 走 bf16 autocast + PYTORCH_HIP_ALLOC_CONF=expandable_segments:True（44% 显存占用稳定）
- RoboMIND 主库（x-humanoid-robomind/RoboMIND）受限；公开替代 = BAAI-DataCube 逐任务包（LeRobot v2 格式：parquet+mp4）；franka_1rgb 仅 2 任务公开，franka_3rgb 有 ~15 任务（后续按需下）
- 数据集字符串 typo 教训：`n_fut_tubelets`（非 tubeplets）

### 🔬 RoboMIND 标注发现（2026-10-02 tick 2）
- BAAI 公开 LeRobot 包 episodes parquet **无 success/failure 列**（bread_in_basket 2975 eps 只有 index/tasks/length/stats）→ RoboMIND 失败标注仅存于受限主库 h5
- **E1 奖励头数据策略调整（默认决策，不停等）**：
  1. 首选：**自产标注** —— 用评测农场跑 policy（早期低成功率 policy 失败样本充足），check_success() 免费生成 success/failure 轨迹；与 LIBERO 域一致，且 CPU 侧无限量产
  2. 并行：等 bread_on_table/3rgb 包下完复查有无标注；真机合作阶段再补真实失败数据
  3. bridge_orig 的 9.7k rollout（含失败）仍是 Bridge 侧候选，找 LeRobot 转换源
- bread_in_basket 单任务 2975 episodes —— 动态头数据广度充足

### 🔄 运行中（tick 2 时点）
- E0 K=5 训练（step ~200+/2000，loss 1.57↓）
- libero_10+libero_90（=libero_100 全量 demos）下载中
- RoboMIND bread_in_basket 24/34 文件 + bread_on_table 排队

### 📐 数据格式版图（tick 2 补充）
- 本机 lerobot 0.4.2（hcu root 继承，v2.x 格式）；RoboMIND BAAI 包 = LeRobot **v3.0**（2975 eps/52万帧，30fps）
- **决策**：E0 动态头继续用自有 h5 加载器（无 lerobot 依赖）；RoboMIND 数据用时自写轻量 v3.0 读取器（parquet+mp4）；SmolVLA（E2）尝试 venv 内 lerobot 1.x --no-deps 安装，失败则自写微调循环
- 现成 LIBERO LeRobot 源：zeromidnight/libero_goal_lerobot_v3.0（仅 goal 套件）、physical-intelligence/libero（PI 格式 41k 下载，转换用）——备选

### 🔬 E0-v1 验收结论（2026-10-02 tick 3）：**动作被完全忽略** ❌→ 修复中
- 训练正常收敛（loss 2.56→1.09，cos 1.01→0.20），但诊断（scripts/diag_action_sensitivity.py）显示：
  **所有动作条件下预测精度完全相同（cos 0.6478）**——真动作/错配/随机/零/×2 全无差异
- 根因（架构性）：1024 ctx tokens vs 9 action tokens + stride=0 下未来可纯外推 → 动作梯度压力≈0
- **这正是论文要解决的核心技术点**（PLaW-VLA 批判的动作条件化缺失，隐空间版）
- **E0-v2 修复包**（已实现并启动训练）：
  1. 残差跳连：ctx 末帧 tubept → skip_mlp 作“惯性基线”，decoder 学动作驱动增量
  2. ctx token dropout 0.3（训练时随机擦除，削弱纯外推捷径）
  3. stride=4（隔 4 帧预测，外推难度上升）
- 验收标准升级：diag 中 A_true 与 C_random 的 pred-cos 差 ≥ 0.05

### 🔄 运行中（tick 3）
- E0-v2 训练（GPU0，logs/train_e0v2_k5_s2000.log，1.06s/step，~35min）
- libero_90 下载（6/100 慢速）；RoboMIND 双包已齐（13G+11G）

### 🏆 里程碑：E0 达成（2026-10-02 tick 5，v5 动作对比协议）

**验收数据**（checkpoints/e0v5/dyn_ens_k5_s2000.pt，已传 ModelScope）：
| 条件 | pred-cos↑ | ens-std↓ |
|---|---|---|
| A 真实动作 | **0.7205** | **0.4336** |
| B 错配动作 | 0.6120 | 0.5577 |
| C 随机动作 | 0.6292 | 0.5618 |
| D 零动作 | 0.6532 | 0.5734 |
| E 动作×2 | 0.6160 | 0.5667 |

- **双通道检测信号成立**：错动作 → 预测精度降 0.09-0.11 cos 且 ensemble 分歧升 +0.13（幻觉注入 ratio 1.72）
- 训练曲线：act_gap 0.0008 → 0.173（margin 0.1 下持续爬升）；JEPA cos 0.26（与 v3 相当，未牺牲预测质量）
- **v5 配方**：单帧 ctx + stride4 + JEPA(cos+smoothL1) + 动作对比项 `clamp(cos_neg−cos_pos+0.1)`，λ=1，负样本=batch 内错配动作

**E0 完整迭代史（可进论文消融表）**
| 版本 | 协议/架构 | A-C 动作敏感 | 结论 |
|---|---|---|---|
| v1 | 8f ctx, stride 0 | 0.0000 | 动作完全被忽略 |
| v2 | +残差跳连+ctx dropout, stride 4 | 0.0039 | 预测更好但仍不敏感 |
| v3 | 单帧 ctx（V-JEPA2-AC 式）| 0.0032 | 单帧仍不够 |
| **v5** | **+动作对比损失** | **0.0913** | ✅ 达标（4.6×目标）|

能量分析副产物：V-JEPA 2 latent 的 Δ 由编码器上下文伪影主导（||Δ||/||f||=0.88 均匀分布，非静态场景问题）——v5 的对比目标绕开了该伪影。

- **能量分析否定了"静态场景主导"假设**：||Δ||/||fut||=0.88 且均匀分布 → 差异主要是编码器上下文敏感性伪影（同内容不同窗口 cos≈0.57），动作相关物理运动是小残差，被 JEPA 主损失淹没
- **v5 核心思想**：训练目标直接对齐部署属性——`clamp(cos_neg − cos_pos + margin)` 把"随机动作的预测必须更差"写进损失（act_gap 指标随训练实时输出）
- 若 v5 达标（A-C≥0.05）：全量重训 + ModelScope；若不达标：加重 Hard negative（反动作/缩放动作）或 inverse-dynamics 辅助头

### 🚀 SmolVLA 基线（tick 4，GPU1 运行中）
- 完整 DTK 配方（已验证到权重加载）：`lerobot-train --dataset.video_backend=pyav --policy.push_to_hub=false --policy.empty_cameras=1 --rename_map={agentview→camera1, wrist→camera2}`
- 官方 lerobot/smolvla_base 是 PI-aloha 谱系（camera1/2/3）；本地 tempo-pai 谱系底座同样不匹配（勿用）
- libero_spatial 已转 LeRobot v2.1（500 eps, mp4, state=joint7+grip1, action=OSC_POSE 7）
- 20k steps bs=32 训练中 → 评测农场测 libero_spatial 成功率（E2 的被验证 policy）

### 🔄 运行中
- E0-v5（GPU0，logs/train_e0v5_k5_s2000.log，1.58s/step 含负样本前向）
- SmolVLA libero_spatial 微调（GPU1，logs/train_smolvla_spatial.log）
- libero_90 下载完成 ✅ LIBERO 全量 130 任务 demos 齐备

### ⏭️ 下一步
1. E0-v5 验收（act_gap 曲线 + diag）→ 达标则全量版 + ModelScope 上传
2. SmolVLA 训练监控 → 5k checkpoint 评测农场跑 libero_spatial
3. 其余 3 套件数据转换（object/goal/libero_90）备全量训练
