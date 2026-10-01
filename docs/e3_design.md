# E3 设计：不确定性门控的后训练（想象空间 × 像素策略的桥接）

> 2026-10-02 起草，随实验迭代更新

## 核心矛盾
世界模型在 **latent 空间**想象（V-JEPA 2 tubept tokens），但 SmolVLA 是**像素策略**——
想象出的 latent 无法直接喂回策略。DiWA（扩散策略在小 WM 内 RL）能闭环是因为它的策略吃 WM 状态；
我们不能。三条可行路线：

## 路线 A（先做）：Verifier-Distilled Post-Training（BoN 蒸馏）
1. 状态源 = 真实 demo 轨迹（LIBERO h5 / 自产 rollout）
2. 每个状态：策略采样 N 个 action chunk（flow 噪声多样性）
3. 每个 chunk → E0 世界模型想象未来 → 奖励头(E1) 打分 + ensemble std 门控
   - score = reward_logit − λ·std（认知不确定性惩罚，HALO 核心项）
4. 选 top chunk 作监督 → SFT 微调策略（lerobot-train 复用，监督=选出的 chunk 而非 demo 原动作）
- 优点：今天就能实现（全部组件已在库）；训练稳定；天然消融（去 std 门控 = CoVer 式蒸馏基线）
- 这是 E3 的第一阶段交付

## 路线 B（后续）：Imagined-Advantage Policy Gradient
- 在 A 的采样框架上把"选中"换成 REINFORCE：advantage = normalized(score)
- flow 策略的 log-prob 需从 flow-matching 目标导出（π0 系有先例：RDT 的 FlowPRO 路线）
- 风险：flow log-prob 方差大；留作扩展

## 路线 C（远期/论文讨论段）：latent 观测策略
- 训练吃 V-JEPA latent 的轻量动作头（DiWA 式闭环想象 RL）
- 需重训策略，工程量大，作为 future work / 真机阶段备选

## 门控消融设计（论文表）
| 变体 | score 定义 |
|---|---|
| 无验证（BC 基线）| demo 动作直接 SFT |
| 仅奖励 | reward_logit |
| 仅不确定性 | −std |
| HALO 全量 | reward_logit − λ·std |

λ 扫描 {0.5, 1, 2}；评测 = LIBERO 三件套 CPU 农场 + wall-clock 开销报告
