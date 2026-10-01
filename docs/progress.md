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

### 🔄 运行中 / 进行中
- [ ] Bridge V2 / RoboMIND 数据源调研与下载（下一个 tick 启动）

### ⏭️ 下一步（按序）
1. Bridge V2 + RoboMIND 数据下载（hf-mirror；RoboMIND 注意失败标注子集）
2. LIBERO demos 下载（policy 微调用，官方 HF 源）
3. src/halo/encoder.py：V-JEPA 2 冻结封装（视频→tubept latents，含预处理）
4. M1 E0 开工：action-conditioned 动态头（bridge 数据先）+ 幻觉注入检测基准设计
5. 评测农场并行化：384 核 worker 池（multiprocessing，参考 lifelong/metric.py 的 use_mp）
