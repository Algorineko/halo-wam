# 统计预注册（Pre-registered Analysis Plan）

> **锁定时间：2026-10-13（Phase 1 50-eps 重跑之前）**。本文档一经提交不再修改；
> 所有主表结论按此规则判定，不因结果调整。回应审稿 II.1/II.2。

## 1. 实验单元与配对结构

- 每任务 **50 episodes**（LIBERO 官方 50 init states 全量，`init_states[ep]` 跨臂相同
  → 同一初始状态在两臂构成**配对样本**）。
- 主表单元 = suite 级（10 任务 × 50 eps = 500 eps/臂）；辅助单元 = 任务级（50 eps/臂）。

## 2. 单一判定规则（全文唯一，禁止替代）

1. **主检验（suite 级）**：配对 bootstrap——对 500 个配对 episode 的成功指示差
   Δ_i（treat − base）重采样（episode 为重采样单元，B=10,000），报告均值差
   （绝对 pp）与 95% percentile CI。**判定：CI 不含 0 且 Holm 校正后 α=0.05**。
2. **任务级辅助**：McNemar 精确检验（50 配对 eps 的不一致对 b/c），
   Holm 校正跨同族比较（每 suite 内的 arms 集合为一族）。
3. **稳健性**：leave-one-task-out 增益范围必须与主 CI 同号报告；
   报告剔除最大 |Δ| 任务后的 suite 差（审稿指出单任务驱动风险，主动披露）。
4. **相关性检验**（C4）：Pearson r 附 Fisher 95% CI；n=10 时只在 |r|>0.632 才称显著；
   分层（suite 内）优先，pooled 相关不再作为"无普适律"的证据（审稿 III.5）。

## 3. 表述纪律

- 摘要与结论只报**绝对 pp + 配对 95% CI**；相对百分比至多并列且必须带 CI。
- 任何单 run 差距 < 配对 CI 半宽 → 只能表述为 "not distinguishable"，禁止选择性
  解释（对己有利当信号、不利当噪声——审稿 II.2 双重标准封死）。
- "75% of the gain"、"= pure"、"orders of magnitude" 类无检验表述一律删除或改实测。

## 4. 预注册的比较族（Phase 1 数据落地后按此生成主表，不增删）

- **族 A（三 suite 主表）**：{spatial, object, goal} × {BoN-5 vs base} = 3 个比较
- **族 B（spatial 阶梯）**：{BoN-5, R2-distill, R2+BoN-5, R3-distill, ver-on-R3,
  random-pick, consensus-pick} vs base = 7 个比较
- **族 C（λ 扫描）**：{λ=0.25, 0.5, 1.0, ∞(reward-only)} vs 纯 σ = 4 个比较
- **族 D（机制消融）**：{act-weight 0 的 BoN} vs {act-weight 1 的 BoN} = 1 个比较
- **族 E（跨架构）**：SmolVLA {BoN-5 vs base} = 1 个比较（若 Phase 3 落地）

## 5. 已知功效（设计依据）

n=500/臂配对设计下，配对 bootstrap 对 +8pp 真差的功效 ≈80%（配对相关 ρ≥0.3 时）；
n=100 时仅 ~36%（审稿复算）——本重跑的目的即补足功效，而非重试运气。

## 6. 数据完整性

- 每格 JSON 必含逐 episode success 向量（插桩后）；分析脚本
  `scripts/analyze_paired.py` 从向量复算一切，不手工转录。
- 决策级日志（候选/σ_K5/σ_K3/选中）仅用于机制分析（override 率、K 消融、
  consensus 重放、σ-成败判别力），不进入主表判定。
