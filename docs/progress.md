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

### 🔄 运行中 / 进行中
- LIBERO demos 下载（4 套件，--use-huggingface，logs/dl_libero_all.log）
- RoboMIND franka_1rgb 下载（68 文件，logs/dl_robomind_franka.log）
- [ ] validate_encoder.py 第 [4] 项（16 帧双窗口检验）未出结果，下 tick 查

### ⏭️ 下一步（按序）
1. 监控两个数据下载完成后校验完整性（h5 可读、任务数对）
2. src/halo/data/libero_h5.py：LIBERO demo → (context 窗口, actions, future 窗口) 训练样本管线
3. src/halo/dynamics.py：action-conditioned 动态头（K=5 ensemble 骨架）+ E0 训练循环 configs/
4. Bridge V2 数据源调研（rail-berkeley/bridge_orig 在镜像不存在，找 OXE/LeRobot 转换版）
5. 评测农场并行化：384 核 worker 池
