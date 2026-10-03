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

### 🏆 E0 生产版完成（2026-10-02 tick 9，130 任务全量 4000 步）

| 指标 | v5（12任务）| **e0_prod（130任务）** |
|---|---|---|
| pred-cos(A_true) | 0.7205 | **0.7547** |
| A-B 动作敏感 | +0.109 | **+0.220** |
| A-C | +0.091 | **+0.173** |
| 错动作 std 升幅 | 0.43→0.56 | **0.45→0.86（近翻倍）** |
| 幻觉注入 ratio | 1.72 | **2.53** |
| act_gap 训练终值 | 0.173 | **0.298** |

checkpoint 已传 ModelScope（e0_prod/dyn_ens_k5_s4000.pt）。生产版世界模型就绪，E2 verifier 直接引用。

### 📉 act-v2 基线结果（2026-10-03 07:54）：2.0%——宽度路线证伪
- act-v2（17.9M，val 0.0409 三者最优）在线仅 **2.0%**，远差于 act60k（8.1M，val 0.0427，**10%**）
- **再次确认：val-mse 与在线成功率完全脱节**；TinyViT 加宽路线证伪，基线锁定 act60k
- E2-R3 启动：10-eps 降噪三臂复测（基线/纯不确定性/λ0.25 完整 HALO）

### 🌍 LIBERO-PRO 泛化验证基建就绪（2026-10-03 10:25，W13 前置）
- 官方 HF 扰动数据落地（hf-mirror）：spatial 的 **swap**（plate/ramekin 位置互换）与
  **object**（black_bowl/yellow_plate 外观替换）两轴，goal 不变、10 任务/轴、配套 pruned_init 50 态
- eval_libero_parallel.py 新增 `--task-spec` 直通模式：扰动任务与原任务同名 →
  HaloACT vocab 与奖励头 vocab 天然兼容，无需重训/注册
- LIBERO 安装打补丁：google_scanned_objects.py 换成 LIBERO-PRO 严格超集（+7 资产类），
  资产树补 7 类；**20/20 扰动 env 冒烟通过**；原 .bak 备份保留
- 设计取舍：HaloACT 是 goal-blind 策略（task_emb 无语言输入）→ lan/task 扰动维无意义，只测
  swap/object 视觉物理轴。主张：**验证器增益（+73% 相对）在 OOD 场景下是否保持**
- pipeline_stage3.sh 挂起：pipeline2 完成后自动跑 swap/object × {基线, 验证器} 4 臂（10 eps）
  清洁基线沿用 R3（11.0% / 19.0%）

### 🎯 E2 定谳（2026-10-03 13:15）：v3 头在线 17.0%——奖励头路线在线证伪，价值转向离线数据引擎
| 臂（10-eps） | 成功率 | 备注 |
|---|---|---|
| 基线 | 11.0% | |
| 纯不确定性 best-of-5 | **19.0%** | **在线赢家，+73% 相对** |
| halo_h（v2 头 λ0.25） | ≥19%（task7/8 补跑中） | R3 原始 |
| halo_h（**v3 头** λ0.25，AUC 0.933） | **17.0%** | R4 |

- **决定性负结果**：v3 头把"头质量不足"这个借口移除了——AUC 0.933 的诚实头在线仍不超纯不确定性。
  三代头（捷径 v1 / 扩充 v2 / 诚实 v3）在线 12-24% 波动，从未显著赢过 19.0%
- **根因假设**：①训练 ctx 全是 demo 状态，在线 ctx 是策略 rollout 状态——分布 gap 三代头共享；
  ②在线 5 候选出自同一策略（noise 0.02 近亲），reward 排序边际 < 噪声
- **价值转向（关键洞察）**：v3 头的舒适区 = **demo ctx 离线打分**（训练分布完全匹配）——
  这正是 E3 蒸馏数据生成的输入分布。奖励头从"在线验证器"转型为"离线数据引擎"
  （= plan.md 的 IROS 回退线：verifier 门控数据重加权）
- 论文叙事成形：在线=纯不确定性赢（+73% 稳健）；奖励头在线失败有 3 代对照的机理分析（诚实）；
  同一头离线 AUC 0.933 → 用于后训练数据选择（E3 实证其价值）
- 运维：pipeline2 因 DTK CVD 坑死在 act_dino 训练步（已修），resume 链重建；
  act_dino 训练 0.22s/step（比预期快 4×，DINOv2 冻结前向便宜），ETA ~14:15

### 🏆 E1-v3 训练完成（2026-10-03 11:52）：VAL-AUC 0.933，首个诚实奖励头
- 曲线 0.827(300)→0.901(600)→**0.933(最终)** 单调爬升，留出为完整 demo episode——
  对比 v1/v2 的 acc=1.0 捷径（无判别意义），这是**动作条件化判别力**首次成立
- 已传 ModelScope（e1_v3/reward_head.pt）；pipeline2 已自动触发 halo_h v3 复测
  （λ=0.25 与 R3 严格可比，唯一变量奖励头 v2→v3，ETA ~12:50）

### 🔄 E1-v3 设计迭代（2026-10-03 11:20）：on-policy 正样本路线证伪 → demo-ctx 配对
- act60k 采样 72 eps 仅 2 成功（4%，且全部集中在 task3 cookie_box）——on-policy 正样本
  不足以支撑跨任务奖励头，**路线证伪**，采样中止
- **新设计（demo-ctx 配对判别）**：ctx 一律 demo，pair 内共享同一 ctx，唯一变量是动作块
  （pos=demo 动作 / neg1=同任务失败 rollout 动作 / neg2=跨任务 demo 动作）——
  视觉捷径被"同 ctx 配对"结构性封死，数据无瓶颈（500 demo + 149 失败 rollout 全在盘上）
- 已开训：6450 训练样本（2150 pos），demo 按 episode / rollout 按文件留出，AUC 选模
  DTK 坑 +1：CUDA_VISIBLE_DEVICES=1 + torch.manual_seed 懒初始化 → default_generators
  IndexError；去掉 CVD（脚本本就写死 cuda:1）即愈
- 链路已重排：e1v3 训练（~11:45）→ pipeline2（halo_h v3 复测 → act_dino → 评测）→
  r3fix（task7/8 补跑）→ pipeline3（LIBERO-PRO 四臂）
- **发现**：e1 v1/v2 训练日志 acc=1.000 / bce=0.0000——demo(+)/失败rollout(−) 是**跨分布二分类**，
  奖励头学成"demo 检测器"而非"动作→成功"预测器；在线打分时策略 rollout 的 ctx 全被饱和判负，
  候选间无判别力 → 解释 R2/λ 扫描中完整 HALO 臂始终逊于纯不确定性臂
- **修复（E1-v3，scripts/train_e1_v3.py）**：正负样本全部来自**同一策略的 rollout**（同分布，唯一差异成败）：
  pos=成功 episode 窗口 / neg1=失败 episode 窗口 / neg2=同 ctx 接失败动作块（教动作条件化）；
  **episode 级留出** + VAL-AUC 选模（窗口级切分会被相邻帧重叠污染）
- 正样本瓶颈：act_15k 的 80 条 rollout 仅 1 条成功 → 已排产 act60k 大规模采样（10 任务×24 eps，
  预期 ~30 条成功，pipeline_e1v3.sh 串行在 R3 之后）
- 同步立项 HaloACT-D（act_policy_dino.py）：冻结 DINOv2-small 特征 + **token 级 memory**
  （512 视觉 token cross-attend，解除现版双相机 mean-pool 成单 token 的瓶颈），可训练 3.3M，
  weights 已下（hf-mirror），冒烟通过；训练排在 halo_h v3 复测后（pipeline_stage2.sh）

### 🚨 OOM 复盘二号（2026-10-03 10:54）：评测与采样不可共存
- **oom_kill 计数 18→27**：halo 臂 task7/8 补跑（2 workers，各加载 V-JEPA ~6GB 峰值）与
  act60k 采样（6 workers）同时跑 → cgroup 32GB 击穿，每对并发子进程恰好死一个（无 traceback，
  stderr 止于 torch.load 警告 = SIGKILL 特征）
- **新铁律：rollout 采集与 GPU 评测（加载编码器）串行**；采集 + card1 训练可共存（~18GB）
- 修复：pipeline_r3fix.sh 等采样 DONE 后以 workers=1 补跑 task7/8（与 e1v3 训练共存，预算 ~18GB）
- 现有 R3 halo 中间值：task0-6,9 有效 = 19/80 eps = **23.8%**；task7/8 待补（区间 [19%, 23%]）
  ——无论补跑结果如何，**完整 HALO 臂（reward−λ·std, λ0.25, v2 头）≥ 纯不确定性臂（19.0%）**，
  与 R2 5-eps 的"融合不如纯不确定性"结论相反——10-eps 降噪后融合价值开始显现，待 task7/8 定谳

### 📊 R3 10-eps 降噪三臂（2026-10-03，halo 臂补跑中）
| 臂 | 5-eps（R2/λ扫描） | 10-eps（R3） |
|---|---|---|
| haloact 基线（act60k） | 10.0% | **11.0%** ✅ 稳 |
| 纯不确定性 best-of-5 | 18.0% | **19.0%** ✅ 稳，+73% 相对 |
| 完整 HALO（λ0.25, e1_v2） | 16.0% | 原始 17.0%⚠️（4 任务因脚本热编辑崩溃记 0）→ task6-9 补跑中 |

- ⚠️ 事故：halo 臂运行中（09:47 起）本会话在 10:00 左右给 eval 脚本加 task_spec 字段（12 元组），
  后派发的 4 个子进程仍用旧 11 元组任务 → ValueError 记 0。**前 6 个任务有效（均值 28.3%）**，
  task6-9 用 `--task-ids 6,7,8,9` 同协议补跑（eval_r3_halo_fix.log）
- 教训入册：**评测运行期间禁止编辑被评测脚本**；长评测脚本改动须等套件 done（已写入运维备忘）

### 📈 λ 扫描完成（2026-10-03 06:54，act60k + e1_v2 奖励头）
| 配置 | 平均成功率 |
|---|---|
| 基线 | 10.0% |
| 纯不确定性 best-of-5 | 18.0% |
| HALO-h λ=0.25 | 16.0% |
| HALO-h λ=0.5 | 12.0% |
| HALO-h λ=1.0（e1_v2 头）| 16.0% |
| HALO-h λ=1.0（e1_v1 头，对照）| 12.0% |

- **结论**：所有验证臂稳超基线；融合变体与纯不确定性在 5-eps 噪声内持平（12-16% vs 18%）——**当前奖励头尚未提供超越不确定性的增益**，非单调曲线提示 5-eps 方差 ±4%
- v1→v2 头同 λ=1 对比：12%→16%（正样本 20→60 有效）；奖励头质量是融合价值的前提
- 下一步排序：① 每任务 eps 5→10 降噪后重测关键点（λ0.25/纯不确定性）② 奖励头正样本扩到全量 demo 窗口 ③ act-v2 基线（评测中）④ LIBERO-PRO 泛化验证

### 🎯 E2 第二轮三臂结果（2026-10-03 03:32，act60k 基线）——**E2 正面信号首次成立**
| 臂 | 平均成功率 | 相对基线 |
|---|---|---|
| haloact 基线（act60k, val 0.0427）| 10.0% | — |
| + 纯不确定性 best-of-5（e0v5）| **18.0%** | **+80%** |
| + 完整 HALO（reward−λ·std, λ=1, e1-v1 奖励头）| 12.0% | +20% |

- **纯不确定性验证器在在线 rollout 中 +8 绝对点**：成功从 2 任务扩散到 5 任务（from_table_center 0→60%、between_plate 0→40%、next_to_cookie_box 0→20%）
- 完整 HALO 臂胜基线但逊于纯不确定性——v1 奖励头（178 样本记忆）信号噪声稀释了不确定性信号；下一步：λ 扫描 {0.25, 0.5} + 扩充正样本（全量 demo 窗口）后重测
- 基线教训：act60k val-mse≈15k 版（0.0427 vs 0.0444）但在线成功率 5×（10% vs 2%）——**验证损失不反映在线成功，评测必须在线**
- 数据文件：eval_results/libero_spatial_haloact{,_v,_h}_20261003*.json

### 📊 E2 首轮完整数据（2026-10-02 20:33，双臂完成）
| 臂 | 平均成功率 | 唯一成功任务 |
|---|---|---|
| haloact 基线 | 2.0% | next_to_ramekin 20% |
| haloact_v（best-of-5 + 纯不确定性 e0v5）| **2.0%** | between_plate 20%（495 步压线）|

- **判定：inconclusive**——极弱基线（2%）下候选动作全是失败轨迹，验证器无信号可选（garbage-in-garbage-selection-out），唯一成功换个任务出现=噪声重排
- **E2 正面主张（verifier 提升 X%）推迟到中等强度基线**（30-70% 成功率区间选择才有意义）；强化路线：60k 步长训 / CVAE 多样性 / 更大 dim
- **科学价值保留**：纯不确定性选不出成功（既不救也不害弱策略）与"不确定性偏好保守动作"假设一致——E1 奖励融合（reward−λ·std）的必要性论据 +1
- 运维沉淀：评测套件 v5 低并行（3/2 workers）稳定跑通全流程 ~2.3h；subprocess 直评模式 + set_num_threads 是容器存活关键

### 🏆 留出任务泛化确认（2026-10-02 10:20）——记忆效应排除
- **e0_heldout**（122 任务训练，8 任务留出）在**从未见过的 8 任务**上：
  A-C=+0.172（对照 e0_prod 见过任务 0.167——**持平**）、注入 ratio 2.68（vs 2.53）
  A 真动作 cos 0.790/std 0.336 vs C 随机 cos 0.618/std 0.724——双通道干净分离
- 结论：动作敏感性与幻觉检测信号是**泛化能力**，非任务记忆——论文核心主张的关键论据
- 消融汇总更新：margin 0.1 优于 0.05（0.173 vs 0.088）；泛化无损（0.172 unseen vs 0.167 seen）

### 🚨 运维事故复盘（2026-10-02 08:00-09:00）：容器 32GB OOM 连环互杀
- **根因**：容器 cgroup memory.max=32GB（`free` 显示宿主机 2.2T 是巨大误导）；SmolVLA 训练启动的内存尖峰触发容器 OOM，killer 选最大 RSS 进程 → 连续误杀当时在训的 e0_heldout（`oom_kill=3`，`max_usage 34.4GB`）
- **假象复盘**：三次"死锁"（16/32/8 workers）实为 worker 被杀后 main 卡 queue.get；关键帧重编码（-g 15）后实测 seek 解码仅 0.12s → 视频 seek 非瓶颈
- **修复**：
  1. 数据集转 **image 格式**（parquet 内嵌帧，绕开视频解码缓冲；spatial 已重转 2.2GB）
  2. **双训练串行化铁律**（看门狗 cron 6e58fd2d 强制）：任意时刻单训练；SmolVLA 排队时降配 num_workers=3 / bs=16
  3. 看门狗每 20 分钟判活（日志 mtime 阈值 10 分钟），死亡自动按队列重启并记 logs/watchdog.log
- **教训**：排查容器问题第一件事看 `/sys/fs/cgroup/memory/`（当前 11.0/32 GB，e0_heldout 独跑健康）

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
