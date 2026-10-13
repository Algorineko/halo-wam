# Related Work 精读笔记（2026-10-13，写作直接引用）

## 1. RoVer（arXiv:2510.10975，2025-10，Dai et al.）
**机制**：Robot Process Reward Model（PRM，需训练）做 test-time verifier——标量过程奖励 +
**动作空间方向向量**（不只选优，还沿 PRM 方向扩展/细化候选）；感知特征缓存摊销编码成本。
**与 HALO 同：** plug-and-play、不改策略权重、best-of-N 部署形态、把测试时算力换成决策质量。
**与 HALO 异：**
- 信号：习得奖励（需奖励训练数据）vs 我们**免训练 ensemble 认知不确定性**
- 候选生成：PRM 方向引导扩展 vs 我们噪声扰动
- 无不确定性校准/失效边界分析（我们的 C4 正面回答"什么时候该信 verifier"——RoVer 假设永远可信）
- 无蒸馏桥（我们 C5：增益可蒸进权重）
**可借鉴**：感知缓存——我们 microbench 的 ctx 重编码是保守上界，部署加缓存可再降（论文 cost 段提一句即可）。

## 2. AtomVLA（arXiv:2603.08519，2026-03，Sun et al.）——机制最近竞品
**机制**：subtask-aware VLA 后训练管线：LLM 把 demo 分解成原子子任务 → 预训练预测式
隐空间 WM 对候选动作块**相对子任务目标**打分 → 兼作 offline GRPO 的奖励。LIBERO 97.0% /
LIBERO-PRO 48.0%，真机 Galaxea R1 Lite。
**与 HALO 异（写区分度段的核心）：**
- **打分信号**：goal-referenced（需子任务目标参照）vs 我们 **goal-free 不确定性**（无需目标定义，
  适用无 reward/无目标的场景）
- **架构位置**：WM 打分内嵌其专属后训练管线（策略+WM 联合）vs 我们 **policy-agnostic 外挂**
  （任意现成策略即插即用）
- **不确定性**：无认知不确定性概念，无失效边界——我们的 C4（失败模态假说）与其正交且互补
- **后训练**：WM 打分喂 GRPO vs 我们 BoN-SFT 蒸馏（C5，部署零验证器开销的摊销路线）
- **regime 不同**：其策略是大模型级（LIBERO 97% 饱和区），我们刻意用 8M 小策略把验证器
  增益暴露在低成功率区——两文数字不可直接比，但 **其 LIBERO-PRO 48% 说明大策略+子任务
  分解可部分抗 OOD**，引用时注明 regime 差异
**无撞车判定维持**：无 frozen V-JEPA2 + ensemble std + 分布内边界刻画组合。

## 3. FAIL-Detect / Can We Detect Failures Without Failure Data?（arXiv:2503.08558，2025-03/06 v3，Xu et al.）
**机制**：两阶段失败**检测**：策略输入输出蒸馏为标量信号（含认知不确定性，学得信号以
flow-based 密度估计最佳）→ conformal prediction 给统计保证 → 运行时报警（sequential OOD 框架）。
**与 HALO 关系：**
- 它把不确定性用于**被动报警**（detection），我们用于**主动选择**（best-of-N 成功率增益
  +67~73% 实测）——互引时定位为同一不确定性信号的两种消费方式
- 它有统计保证（CP 覆盖率）而我们没有——**诚实引用**，可作 future work（给我们的
  选择规则加 conformal 校准）
- 我们的 C4 失败模态假说直接补充它：**不确定性信号在 capability regime 会误导**——
  对一切"不确定性检测器"都是适用条件警告（引用价值高）

## 写作动作项
- [x] 三篇区分度要点入 paper_outline.md §2
- [ ] Intro 缺口句改写：现有 verifier 要么习得奖励（RoVer）要么 goal-referenced（AtomVLA），
      **免训练、goal-free、不确定性感知**的 verifier 缺位——HALO 补位
- [ ] C4 段引 FAIL-Detect 作"不确定性检测器适用条件"互证
- [ ] cost 段提 RoVer 式感知缓存可进一步压我们已很低的延迟（631ms/决策 → ctx 缓存后更低）
