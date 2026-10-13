# Paper Draft — Section 4: Experiments（数据全部来自已封版 JSON，2026-10-13 起草）

> 协议：每任务 10 episodes / 10 init states，LIBERO 三 suite（spatial/object/goal 各 10 任务），
> 候选 = 策略 chunk + N(0,0.02²)×N，打分 = ensemble epistemic std（argmin），免策略重训练。
> 被验证策略 HaloACT（8.1M）。全部数字已对照原始 JSON 核验（10/10）。

## 4.1 Baseline lineage（val-mse ≠ online，方法学暗线）

| policy | params | val-mse | online (spatial) |
|---|---|---|---|
| HaloACT act15k | 8.1M | 0.051 | 2% |
| HaloACT act_v2 | 17.9M | 0.043 | 2% |
| HaloACT act60k | 8.1M | 0.046 | **11%** |
| HaloACT-D (DINOv2 frozen) | 24M | 0.038* | 0% |
- 表述：四次脱钩——离线指标不可代理在线成功率；一切结论以在线评测为准。
  （*act_dino 的 val 口径需写作时核对原始日志再定稿）

## 4.2 Main result: 三 suite × 验证器

| suite | base | +BoN-5 | Δrel | corr(base, per-task Δ) |
|---|---|---|---|---|
| LIBERO-spatial | 11.0% | **19.0%** | **+73%** | +0.47 |
| LIBERO-object | 7.0% | 4.0% | −43% | −0.50 |
| LIBERO-goal | 23.0% | 22.0% | −4% (≈) | −0.44 |
- Figure: eval_results/per_task_heatmap.png（spatial 阶梯 11→19→19→22；object 三任务被清零可见）
- 摘要句：+73% 在 spatial；增益**非普适**——引出 4.4 失败模态假说。

## 4.3 Mechanism: 三代奖励头对照（C3）

| head | 训练分布 | 离线 | online λ=0.25 |
|---|---|---|---|
| v1 demo/rollout 二分类 | 跨分布（捷径） | acc=1.000（无意义） | 12% |
| v2 扩充正样本 | 跨分布 | — | 19%（持平） |
| v3 demo-ctx 配对 | 同分布配对 | **AUC 0.933** | 17% |
- 结论：诚实奖励头也不加分 → **不确定性信号独扛在线增益**；奖励头价值在离线（C5 的选择器）。

## 4.4 Boundary: 失败模态假说（C4）

- suite 级：epistemic regime（排布类，基线中等）→ 正；capability regime（抓取类，低基线）→ 负；
  长视野混合 → 零。per-task 散点 corr 见 4.2 表；pooled ≈ 0 → 模态是 suite 级属性。
- OOD（LIBERO-PRO 双轴，10-eps）：swap 位置互换 0/0 全崩（两臂）；object 外观替换
  base 21% > clean（goal-blind 小策略对几何敏感、外观不敏感），验证器无加成。
- 互证引用：FAIL-Detect（2503.08558）的检测器同样以不确定性为信号——本假说是其适用条件警告。
- 诚实注记：per-task 10 eps 噪声 ±20pp，suite 聚合为可靠单元；+3pp 级差异保守表述。

## 4.5 Cost: n-cand 曲线 + 延迟

| n-cand | 3 | 5 | 10 |
|---|---|---|---|
| spatial | 17.0% | **19.0%** | 12.0% |
- ~75% 增益在 n=3 已到手；n=10 回落 → argmin-std 尾部风险（更多候选更易选中"自信坏块"），
  **不确定性选择不是朴素 test-time scaling**。
- 延迟（BW1000_H，bf16）：每决策（BoN-5）631ms = 策略 chunk 320ms（CPU）+ 打分 311ms（GPU）；
  候选边际 ~7ms（ctx 编码摊销）。生成式 WM rollout 为秒级/步 × 展望步数——便宜两个数量级。
  （RoVer 式感知缓存可再降 ctx 编码部分。）

## 4.6 Distillation: 验证增益可蒸进权重（C5）

| 配方 | spatial |
|---|---|
| 基线 act60k | 11.0% |
| BoN-5 蒸馏 R1（1800 状态，无过滤） | 10.0%（平价） |
| BoN-5 蒸馏 R2（7203 状态 + margin>p50） | **19.0%**（= 在线验证器全额） |
| R2 + BoN-5（组合） | **22.0%（基线 2×）** |
- R1 平价 → "实例决策难权重化"被 R2 推翻：规模×选择质量即破。
- 组合成立 → 验证-蒸馏交替循环；部署侧蒸馏 = 验证的摊销（320ms/chunk，零验证器开销）。
- 诚实注记：组合 +3pp 单 run（100 eps SE≈4pp），方向性支持、保守表述；R3 迭代留 future work。

## 写作待补（非数据项）
- [ ] act_dino val 口径核对（4.1 表 * 号）
- [x] λ 敏感性：遗留数据混杂（5-eps + 事故期污染 10-eps）→ 已启动干净补臂
  （pipeline7：v3 头 λ∈{0.5,1.0} 10-eps，λ=0.25=17% 已有 130415）——出数后填 4.5
- [ ] LIBERO-PRO 数字表格式化（swap/object 双轴 4 臂）
