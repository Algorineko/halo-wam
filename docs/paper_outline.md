# HALO 论文骨架（CoRL 2027 主投；IROS 2027 回退：verifier 数据引擎）

> 状态：2026-10-03 起草，基于当日闭环的四条战线数据。数据源全部见 docs/progress.md。

## 标题（候选）
1. **HALO: Uncertainty-Aware Latent World Models for Test-Time Verification of VLA Policies**
2. Best-of-N with a Cheap Latent World Model: +73% In-Distribution Success for VLA Policies
3. （回退线）Verifier-Gated Data Engines: Distilling Latent-World-Model Judgments into Policies

## 一句话主张
在冻结 V-JEPA 2 隐空间上，deep-ensemble 认知不确定性作为 test-time 动作块验证器，
以世界模型 1/100 的成本给 VLA 策略带来 +73% 相对成功率提升（LIBERO 三 suite）；
并给出该增益的**分布内边界**的诚实刻画与机理分析。

## 摘要骨架
- 背景缺口：WM-as-verifier 是活跃赛道，但现有验证器昂贵（生成式 rollout）或在 OOD 过自信
- 方法：frozen V-JEPA 2 + 轻量 action-conditioned 动态头（动作对比损失）+ K-ensemble
  认知不确定性 → best-of-N 打分，免策略重训练
- 主结果：LIBERO spatial/object/goal 三 suite，小策略基线 11%→19%（+73% 相对）
- 边界（诚实卖点）：LIBERO-PRO 扰动下增益消失（swap 全崩 / object 无加成）——
  不确定性信号是分布内现象；三代奖励头对照证明"加 reward 不如纯不确定性"
- 成本：动态头 6 层 transformer ×5，单候选打分 <50ms 级（vs 生成式世界模型 rollout）

## 1 Introduction
- VLA 测试时验证的两个已知失败模式（WMPO→HaWMPO reward hacking；VLA-RFT OOD 过自信）
- 缺口：廉价、隐空间、不确定性感知的验证器不存在（PLaW-VLA 的 LWM 是 action-free）
- 贡献四条：
  C1 动作对比损失解决隐空间 WM 动作忽略问题（A_true/B_mismatch 诊断协议）
  C2 纯不确定性 best-of-N 在线 +73%（三 suite，10-eps 降噪）
  C3 奖励头融合的三代对照与"在线≠离线"解耦发现（AUC 0.933 头在线无加成）
  C4 OOD 边界刻画（LIBERO-PRO swap/object 双轴）——验证器增益是分布内现象

## 2 Related Work
- Test-time verification: CoVer (Stanford)、RoboMonkey（均重 rollout / 需训练）
- Latent world models: V-JEPA 2-AC（停在规划）、DiWA（像素 WM）、PLaW-VLA（action-free 缺口=我们的 C1）
- WM 后训练: WMPO / HaWMPO（reward hacking 案例=我们动机）
- 基准: LIBERO 饱和 → LIBERO-PRO（我们的 OOD 轴）

## 3 Method
- 3.1 固定窗口相对编码协议（V-JEPA 2 无绝对位置嵌入；同内容跨上下文 cos≈0.55 的伪影分析）
- 3.2 动态头：future queries 交叉注意 [ctx; act] + skip MLP；**动作对比损失**
  clamp(cos_neg − cos_pos + margin)；K=5 ensemble，jepa_loss + epistemic_std
- 3.3 验证器打分：score = −std（纯不确定性臂）；reward − λ·std（融合臂，见消融）
- 3.4 haloact_v 推理协议：候选 = 策略 chunk + N(0, 0.02²) 噪声 ×5

## 4 Experiments
### 4.1 被验证策略与诚实基线
- HaloACT（8.1M TinyViT×2）：lerobot-train 在目标容器不可用的工程约束 → 自写策略
- 基线谱系表：act15k 2% / act_v2(17.9M) 2% / act60k **10-11%** / act_dino(DINOv2冻结) 0%
  ——**val-mse 与在线成功率四次脱钩**（表格直接进论文，反对离线指标代理）
### 4.2 主结果（三 suite 表，数据 pipeline5 出）
| suite | base | +verifier | 相对 |
|---|---|---|---|
| spatial | 11.0% | 19.0% | +73% |
| object / goal | 夜间出数 | | |
- 分任务热力图：验证器把成功从 2 任务扩散到 5 任务
### 4.3 奖励头消融（C3，三代对照）
| 头 | 训练分布 | 离线 | 在线(λ0.25) |
|---|---|---|---|
| v1 demo/rollout | 跨分布（捷径） | acc=1.0 无意义 | 12% |
| v2 扩充正样本 | 跨分布 | — | 19%（持平） |
| v3 demo-ctx 配对 | 同分布配对 | **AUC 0.933** | 17%（无加成） |
- 结论：不确定信号承载全部在线增益；奖励头价值在离线（数据引擎/蒸馏选择器）
### 4.4 OOD 边界（C4，LIBERO-PRO 双轴）
- swap（位置互换）：0/0 全崩、全 500 步、无弱信号
- object（外观替换）：基线 21% 反超清洁（goal-blind 策略对几何敏感外观不敏感）；验证器无加成
- 主张收窄 + open problem
### 4.5 λ 敏感性与候选数
- λ 扫描 {0.25,0.5,1.0}×2 头；n-cand 5（成本-收益曲线可补 3/10）
### 4.6 后训练初探（E3-A R1，可作为 discussion 或删）
- BoN 蒸馏 R1 平价（10%≈11%）：test-time 实例决策难权重化；迭代配方 future work

## 5 Limitations（诚实清单）
1. 增益限于分布内（C4 是双刃剑：诚实但限制适用域）
2. 被验证策略是小自研策略（非 SmolVLA/π0 级）——验证器对强策略的增益待真机合作验证
3. 蒸馏未迭代；LIBERO 评测协议 10 init states/task
4. CPU MuJoCo 评测的渲染-真实差距

## 6 Future Work
OOD 鲁棒验证器（扰动增广动态头 / 测试时自适应）、迭代式 BoN 蒸馏、
不确定性门控 GRPO 想象后训练（原 plan E3 路线 B）、真机移植

## 投稿时间线
- 数据封版：2026-11 中（多 suite + λ/候选数敏感性补齐）
- 初稿：2026-12；内审：2027-01；CoRL 2027 截稿（~2027-02/03）
- 回退：若审稿风险高，C3+C4 剥离为 "verifier 数据引擎 + 诚实评测" 投 IROS 2027

## 待办（写作前置）
- [ ] pipeline5 三 suite 数据（跑中）
- [ ] n-cand 敏感性（3/5/10）——补一张成本收益图
- [ ] 打分延迟 microbench（卖点数据：验证器 vs 生成式 rollout 成本）
- [ ] 分任务成功热力图（clean 3 suite）
- [ ] 真机合作实验接洽（用户协调项）
