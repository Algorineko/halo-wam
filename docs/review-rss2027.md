# 审稿报告（Peer Review Report）

**稿件**：HALO: Calibration-Free Latent World-Model Verification for VLA Policies — and When It Helps（paper/main.pdf，2026-10-13 draft for RSS 2027）
**投稿目标**：RSS 2027（Robotics: Science and Systems；两阶段制：6 页扩展摘要 2026-12-04 截止，8 页全文 ~2027-01 截止，双盲）
**审稿日期**：2026-10-13
**核查方法**：逐行核对 main.tex/main.pdf/图1 热力图；全部统计量独立复算（Wilson CI、两比例检验、相关系数检验、功效分析）；13 条参考文献逐一联网核实；候选批评经独立对抗验证（76 条候选 → 66 条确认、0 条被推翻、全部引文逐字比对无误）。

---

## 致作者 (Comments to the Author)

## I. 综合评估与推荐 (Overall Assessment & Recommendation)

### 1. 核心贡献（审稿人理解的版本）

论文提出 HALO：在 frozen V-JEPA-2 编码器上训练带 action-contrast hinge loss 的轻量 dynamics head，用 K=5 deep-ensemble 的 epistemic 标准差 σ_ens 作为 best-of-N 候选动作选择分数，构成"免校准、免目标参考、免奖励数据"的 VLA 测试时验证器；并在 LIBERO 三 suite 上刻画增益边界（spatial +8pp / object −3pp / goal −1pp、OOD 全失效），提出 epistemic vs capability 失败 regime 假设（C4），并展示 best-of-5 蒸馏一次性吸收增益（R2 19%）、二次蒸馏塌缩（R3 8%）（C5）。

### 2. 针对 RSS 2027 的契合度评估

**不契合，且不是打磨问题，是证据结构问题。** 具体判断：

- **RSS 的评审重心**（novelty、technical quality、significance、experimental validation；8 页正文、双盲）下，本稿当前有**九项各自单独足以拒稿的缺陷**（见 II.1–II.9）：头条结果在自身样本量下不显著且由单一任务驱动；全文零显著性检验并存在系统性的噪声门槛双重标准；标题级卖点 C4 不可证伪且被自家 goal suite 证伪；**被测策略根本不是 VLA**（题名名实不符）；零外部基线且 Q1 的排他性前提为假；核心评分机制是 PETS/Deep Ensembles 的已知技术且全文未引用；机制主张（C1/C3）无直接证据链；延迟优势声明无实测；参考文献存在多处事实性错误。
- **赛道拥挤度**：2025–2026 该方向竞品密集——RoboMonkey（arXiv:2506.17811）、RoVer（2510.10975）、MG-Select（2510.05681）、CoVer/CoVer-VLA（2602.12281，Stanford：contrastive verifier + test-time scaling laws + 分层验证管线）、CheckVLA（2607.26789：frozen action-conditioned world model + conformal 校准 + 执行期监控修复）、ContactGuard（2608.13438：action-conditioned latent world model 执行监控）、BOKBO（2605.30660：calibrated abstention）、HaWMPO（2609.09941：hallucination-aware world model RL 后训练——与 HALO 的 "Hallucination-Aware" 命名直接撞车）。HALO 的差异化空间被压缩到"增益边界的刻画"，而边界刻画本身的定量支柱（三个相关系数）在 n=10 任务下全部不显著。
- **RSS 偏好真机**：本稿 sim-only（MuJoCo CPU 渲染）+ 自研 8.1M 策略，无任何真机或社区可横向比较的策略。
- **真实的资产**（不应丢弃）：诚实的负结果报告框架（把 object 负增益、OOD 失效、蒸馏塌缩写进主表）、逐任务热力图与逐格数据、"扰动候选本身即有探索价值"的对照设计意识。这些是重构后的论文的种子，但在当前统计功效下没有一条立得住。

### 3. 推荐意见 (Recommendation)

**拒稿 (Reject)。**

建设性方向（二选一，均需 2–3 个月量级的补实验，RSS 2027 全文截稿前勉强可行，更现实的是 CoRL 2027）：

- **路线 A（做实"何时有效"）**：把"VLA 验证器适用边界的系统性映射"升格为唯一主线——每任务 ≥50 episodes、≥3 seeds、配对检验；在一个真正的社区 VLA（OpenVLA-OFT/SmolVLA/π0-mini 非饱和设置）+ 一个 sensorimotor 基线上跨架构验证；给出 ex-ante regime 判据并在留出任务上测 precision/recall。负结果格子（object、OOD、迭代蒸馏塌缩）是这篇论文唯一可能被记住的东西。
- **路线 B（做实验证器本身）**：与 RoVer/CoVer/CheckVLA/MG-Select 在同一 benchmark 上正面对比 + "HALO+conformal 校准版自己"消融，若 calibration-free 在对比中仍有帕累托优势（成本-精度），方法论文才成立。

---

## II. 必须解决的核心问题 (Critical Issues Requiring Mandatory Revision)

### II.1 头条主结果统计不显著，且由 10 个任务中的 1 个任务驱动

- **事实**：n=100/臂（10 task × 10 ep）。11% vs 19%：两比例检验 z=1.58，**p=0.113**；Wilson 95% CI 分别为 [6.3%, 18.6%] 与 [12.5%, 27.8%]，大幅重叠；对真实 +8pp 的功效仅 36%，80% 功效的最小可检测差 ≈14pp。主表三行 base vs +BoN **没有任何一行显著**：object 7%→4%（"actively hurts"）p=0.352；goal 23%→22% p=0.866。C5 的 "R2 recovers the full 19%"（19 vs 11）继承同一个 p=0.113。
- **更严重**：从图 1 热力图逐格复算，spatial 净增 +80pp 中单任务（20%→90% 任务）独占 +70pp；**剔除该任务后 base 10.0% → BoN 11.1%（+1.1pp），落入论文自设噪声带**。Table 2 caption 引用的 "20%→90%, 40%→70" 恰是这个单任务极值；同时 spatial 有 5/10 任务 base=0%，与摘要 "epistemic failure regime (tasks the policy sometimes solves)" 的定义直接冲突。
- **必须做**：每任务 ≥50 episodes（500/臂以上）；利用已声明的 "same seeds across arms" 做配对检验（McNemar / 配对 bootstrap）——这是论文手里已握有却完全未用的功效；报告 leave-one-task-out 增益范围与 suite 级 bootstrap CI；在拿到显著结果前，摘要/C2/结论的确定性表述全部降级为带 CI 的点估计。
- 位置：Abstract (L24)、C2 (L37)、Table 2 (L104–109)、Conclusion (L182)。

### II.2 全文零显著性检验，且"±4pp 噪声线"被系统性双重使用

- **事实**：Limitations (3) 自设 "mark single-run gaps <4pp as noise"，但 ±4pp 是**单臂** 1 个 SE；两臂差值的 SE≈5.0pp、95% CI 半宽≈±10pp——该规则把噪声阈值低估约 2.5 倍。在此基础上正文把 **+3pp / +4pp 的差距在对己有利时当信号**（C4(iii) "+3pp"、C5 "stacks to 22%"、§4.5 "the score ... carries the gain" 15→19），**对己不利时当噪声**（C3 把 v3 head λ=0.5 达到的 22% 判为噪声，而 22 vs 19 与 19→22 在统计上完全不可区分，p=0.599）。两个断言不能同时成立：若 +3pp 是噪声，C5 的 stacking 与 §4.5 机制归因同塌；若 +3pp 可为信号，C3 的 "never beats pure uncertainty" 被自家 λ-curve 证伪。
- 讽刺的是，全文在线结果里**统计显著的只有两个负结果**（R3 塌缩 8% vs 19%，p=0.023；lineage 2% vs 11%，p=0.010）。
- **必须做**：预注册单一判定规则（配对 bootstrap 95% CI 或配对置换检验，α=0.05，多重比较 Holm 校正）全文统一执行；所有表加 CI/p 值列；"75% of the gain arrives by N=3"（2pp 差距）、"= pure"（单次运行相等）一类表述删除或降级。
- 位置：Limitations (3) (L179)、C3 (L38)、C4(iii) (L137)、C5 (L40)、§4.5 (L153)、Table 3/4/5 caption。

### II.3 C4 "When It Helps"：不可证伪、事后拟合、且被自家 goal suite 证伪

这是标题的后半段（"— and When It Helps"）与摘要承诺的 "falsifiable account"，当前状态是三重失守：

1. **相关性全部不显著**：corr +0.47 / −0.50 / −0.44 基于 n=10 任务，n=10 需 |r|>0.632 才 p<0.05；+0.47 的 95% CI 为 [−0.23, 0.85]，三条全部横跨零。"Predictions, all confirmed" (L137) 名不副实。
2. **统计量本身有偏**：corr(base, Δ) 中 base 同时出现在两侧（part-whole），在"验证器完全无效"的零假设下期望为负——按热图 base rate 离散度估算，零效应下期望 corr 约 −0.4~−0.6，观测到的 object −0.50 / goal −0.44 **与纯噪声伪影定量一致**，不构成"harms concentrate at low base rates"的证据。且"倒 U 形"假设用线性 Pearson 检验是形状错配；论文自己的最大增益任务出现在观测 base rate 范围顶端，与 "intermediate base rates" 措辞冲突。
3. **内部反例被沉默**：goal suite base 23% 为三 suite 最高、最符合 "sometimes solves" 判据，结果 −1pp、corr=−0.44（与 object 同号）——C4 正文与贡献条目对此只字未提，构成选择性确认。regime 标签只能在看到验证结局后赋予（有帮助=epistemic、有害=capability），是循环操作化；C4(ii) 又自认 "not a function of raw success rate"，则 regime 无任何 ex-ante 判据。另外 (iii) 的"跨策略迁移"三条腿中 +3pp 与 0pp 均在噪声内，且 R3 的 8% 是作者自己蒸馏造成的退化，用它验证 "capability regime" 是循环论证；LIBERO-PRO 的 0/0 地板效应对任何机制都"确认"假设（不可证伪），appearance-swap 的 base 21% 反而高于分布内 11% 这一反常未获解释。
- **必须做**：给出只用验证前可观测量（候选间 ensemble 分歧分布、held-out 通过率等）的 ex-ante 判别规则，在留出任务/suite 上报告 precision/recall；任务数 ≥30 或重复运行使相关检验有功效；把 goal 作为公开反例写进 C4 并给出能同时解释 spatial(+)/object(−)/goal(0) 的扩展判据；在此之前删除 "falsifiable account"、"principled, predictable places"、"Predictions, all confirmed"，并考虑把标题改为 "and Where It Does Not"。
- 位置：Abstract (L24)、C4 (L39)、§4.4 (L137)、Conclusion (L182)、标题 (L16)。

### II.4 题名与实验对象不符：被测策略不是 VLA

- 标题、摘要、Q1 全部以 "VLA Policies" 为对象，但 HaloACT 是 8.1M TinyViT×2、以 130-task ID 词表条件化的自研策略——**没有语言主干、不接收自然语言**。13 条参考文献中无任何 VLA 主线文献（RT-2/OpenVLA/π0/Octo 全部缺席）。摘要只说 "lifts an 8.1M-parameter policy"，读者会默认这是一个小型 VLA。一篇题为 "...Verification for VLA Policies" 却从未在真正 VLA 上实验的稿件，framing 构成误导。
- 同时 "policy-agnostic by construction" / "attaches to any chunk-emitting policy" 是零跨架构证据的普适声明：全部证据来自同一 HaloACT 家族（base、R2、R3 都是其自身蒸馏/退化产物）。"small policies deliberately"（避天花板）的辩护不成立——避天花板只需低成功率设置，不需要自研不可横向比较的策略；LIBERO 官方 BC baseline 或公开 VLA 中间 checkpoint 都能满足，作者自己（R3 从 19% 降到 8%）就证明了低成功率可以人为制造。
- **必须做**：要么在至少一个社区公认的 VLA 上验证（非饱和设置），要么把标题与全文的 "VLA" 系统性改为 "imitation/sensorimotor policies" 并在摘要明确说明；删除普适性声明。
- 位置：标题 (L16)、Abstract (L24)、Q1 (L30)、Policy (L75)、Limitations (1) (L179)。

### II.5 零外部基线，且 Q1 的排他性前提为假

- 全文没有一个已发表 verifier 在同一 benchmark 上的对比（RoVer/CoVer/CheckVLA/MG-Select 均只在 Related Work 出现）；也没有任何轻量替代（MC-dropout、policy-ensemble disagreement、多温度采样+一致性投票、retry-until-timeout）。"calibration-free" 作为标题级卖点，从未与"同一 σ 信号 + 小规模 conformal 校准"的**校准版自己**对比以证明校准是多余的。
- 更致命：论文自己写 MG-Select "removes the verifier via self-consistency"——经联网核实（arXiv:2510.05681），MG-Select 免训练、免外部模块、用内部 KL 置信度做选择，**同样是 calibration-free、goal-free、reward-free**。因此摘要 "existing verifiers either require trained reward models, goal references, or calibration data" 与 "answers Q1 affirmatively where its predecessors do not apply" 为假。同理 "none characterizes when verification helps" 是可被审稿人一眼证伪的 universal negative（CoVer 的 verification-vs-policy-learning scaling 分析即属此类刻画）。
- **必须做**：在 LIBERO-spatial/object 上至少复现 3 个基线（有监督 verifier 上界如 RoVer 式 PRM；MC-dropout 或 policy-ensemble；MG-Select 式内部置信度）+ 校准版消融；重写 Q1 的差异声明为可辩护形式。
- 位置：Abstract (L24)、Intro (L28, L32)、Related Work (L46, L50)、Sec 4 全部表格。

### II.6 方法新颖性不足，关键先例零引用，2026 竞品已把差异空间挤占

- **σ_ens 作候选选择 = 已知技术**：deep ensemble prediction disagreement 作为采样/规划信号是 PETS（Chua et al. 2018）的 acquisition function 与 active learning 标准做法，σ_ens 直接来自 Deep Ensembles（Lakshminarayanan et al. 2017）——**两条源头文献全文未引**（全文仅 13 条引用，RSS 论文常规 40+）。LLM 侧 best-of-N + verifier（Cobbe 2021 等）亦未引。
- C1 的 action-contrast 是标准 triplet/hinge 换名，且其前提观察（frozen 视频编码器动作不敏感）正是 V-JEPA-2-AC 自己讨论过的问题；C5 的"一次性可蒸馏、迭代即塌缩"是 self-training/model-collapse 已知结论的领域移植。
- **联网核实的最新竞品**（均 2026）：CoVer（contrastive verifier——连"对比训练的验证器"这一步都有先例）、CheckVLA（frozen action-conditioned world model + conformal 校准 + 延迟感知修复——HALO 的"最近亲"，差异只剩"用 ensemble σ 而非校准分数、选候选而非监控执行"）、ContactGuard（action-conditioned latent world model 执行监控）、HaWMPO（"Hallucination-Aware World Model"——与 HALO 命名直接冲突，且 HALO 全文从未定义"幻觉"为何物，"Hallucination-Aware" 目前是品牌化修辞）。
- **必须做**：补引 Deep Ensembles/PETS/BoN/PRM/VLA 主线；重构贡献叙事——C1/C2/C3 合并为一个 "assembled from standard parts" 的系统贡献并显式自认，把"边界刻画"提为第一贡献（前提是 II.3 把它做实）；与 CheckVLA、CoVer、MG-Select 逐条对比差异维度。
- 位置：Related Work 全节、C1–C5 (L36–41)。

### II.7 核心机制主张无直接证据链

- **C1 "prerequisite for all downstream gains" 无消融**：全文没有"无 L_ac 训练 + BoN"对照；A-true/B-mismatch 诊断没有一个数字（"barely depend" 无度量、无图）。负样本 a⁻ 取自**其他任务**，跨任务动作在 latent 空间可被任务布局指纹区分，对比损失可以不学真实动作条件化就收敛——缺同任务难负样本消融。
- **Eq.(1) 数学问题**：a⁺≡a 使正项 cos(f(z,a),f(z,a))≡1，损失退化为 [cos(f(z,a⁻),f(z,a))−0.9]₊ 的单侧排斥项，达到阈值后梯度为零；公式如所写与"contrast"名义不符，需给出精确定义。
- **C3 设计测不出结论**：从未测试 reward-only 选择臂（λ→∞ 即 argmin r），λ-sweep 只覆盖弱 reward 权重，就断言 "uncertainty, not reward, carries the gain"；"17/22·20/16%" 记号不可解析（哪两个数是 λ=0.5 的两次运行？）；v2 head 无 offline 数字，v1/v2 无 λ-sweep；v3 的 AUC 0.933 评测集未定义、"shortcut-free" 无探针证据。
- **σ 本身的判别力零测量**：全文没有任何"σ 与候选 chunk 真实成败"的离线 AUC/秩相关（按 suite 对比），也没有在线选择统计（override 率、每 episode 决策数、被选/被拒候选的 σ 分布）。K=5 不消融，无 MC-dropout 等更廉价不确定性对照——"ensemble epistemic std 是有效成分"缺内部证据。
- **必须做**：with/without-L_ac × BoN 消融 + 诊断量化；reward-only 臂 + λ 全程 sweep + 明确记号；离线 σ–成败判别力测量；K 消融与廉价替代对照。
- 位置：Eq.(1) (L64)、Eq.(2) (L71)、§3 (L62–73)、Table 3 (L127–132)。

### II.8 延迟与成本声明无实测，且账目内部矛盾

- "orders of magnitude below generative world-model rollout"（摘要）与 "two orders of magnitude"（结论）——论文**没有在同一硬件实测任何 generative world model**，"seconds per step × horizon" 无出处无引用；按论文自报数字（0.63s vs "seconds per step"）最多支持约 0.8–1.7 个数量级，且两处量级表述互相矛盾。
- 账目矛盾：Intro 称 "The verifier adds 0.63 s per decision (≈87% of which is the dynamics head)"，实验节实测 0.63s = policy chunk 0.32s（base 也要付）+ scoring 0.31s——**verifier 的边际成本是 0.31s**，87% 是 scoring 内部 dynamics head 的占比而非 0.63s 的占比，归属错误。BoN 决策粒度（每 chunk 还是每步？每 episode 几次决策？累计延迟对时限的影响）全文未定义。无训练总算力声明（4 个策略版本 + 5 ensemble 成员 + BoN rollout 收集），对一篇以成本为卖点的论文是明显缺口。
- **必须做**：同一硬件实测至少一个 generative rollout baseline（或引用可比实测）；统一为"边际成本 0.31s/决策"；给出决策粒度与每 episode 累计延迟；补总算力表。
- 位置：Abstract (L24)、Intro (L32)、§4.5 (L155)、Conclusion (L182)。

### II.9 参考文献存在多处事实性错误（联网逐一核实）

| 引用 | 核查结果 |
|---|---|
| liberopro2025 | **题名引错**。真实题名 "LIBERO-PRO: Towards Robust and **Fair** Evaluation of Vision-Language-Action Models **Beyond Memorization**"（arXiv:2510.03827），论文引作 "...robust and honest evaluation of VLA models" |
| diwa2025 | **题名引错**。真实题名 "DiWA: **Diffusion Policy Adaptation with World Models**"（arXiv:2508.03645，CoRL 2025），论文引作 "World model-distilled skills for robotic manipulation" |
| vjepa2_2025 | **一作引错**。真实一作 **Mido Assran**（arXiv:2506.09985），论文引作 "A. Bardes et al." |
| hawmpo2025 | **条目残缺且年份错**。真实为 "HaWMPO: Hallucination-Aware World Model-based Policy Optimization for Generalist Robot Policy"，Zengjue Chen et al.，arXiv:2609.09941，**2026-09**；条目无作者、题名改写、误作 2025 |
| cover2026 | **题名非实际题名且缺 ID**。真实论文 "Scaling Verification Can Be More Effective than **Scaling Policy Learning** for Vision-Language-Action Alignment"（J. Kwok, X. Zhang, M. Xu, Y. Liu, A. Mirhoseini, C. Finn, M. Pavone；arXiv:2602.12281；官方 repo github.com/cover-vla/cover-vla），论文引作 "CoVer-VLA: Scaling verification can be more effective than scaling **data**"，缺 arXiv 号，"ECCV 2026" venue 未能确证 |
| plawvla2025 | **characterization 存疑**。真实题名 "PLaW-VLA: **Predictive** Latent World Modeling for Vision-Language-Action Policies"（arXiv:2610.12285），论文称 "Physics-aware" 且 "the world model is action-free"——摘要显示其以预测未来状态条件动作生成 |
| checkvla2026 | 存在（arXiv:2607.26789）但日期误作 "Oct 2026"（实际 2026-07），题名漏 "for **Long-Horizon Mobile Manipulation**"；"36.1% vs 27.6% on RoboCasa365" 转述属实 |
| rover2025 / robomonkey2025 / mgselect2025 / faildetect2025 / atomvla2026 / libero2023 | 存在、作者正确（arXiv:2510.10975 / 2506.17811 / 2510.05681 / 2503.08558 / 2603.08519；LIBERO NeurIPS 2023） |

另：5 条 bibitem 无作者；natbib+plainnat 却手写 thebibliography 的混搭；13 条总量对 RSS 远远不足（核心机制源头文献零引用，见 II.6）。**引用题名/作者错误在双盲评审中是可信度级问题，必须全条目复核修正。**

### II.10 呈现与可复现性不达投稿标准

- 编译版 PDF 存在 **"Sec. ??"（未定义引用）**；全文 4 表 1 图**无一处被正文交叉引用**（"Table X shows..."式的句子一句没有）。
- 图 1 热力图按 \textwidth 排版后任务标签字号约 2–4pt，印刷不可读；caption 术语（"the spatial ladder"）与图中轴标签不一致；任务名被截断。
- 无附录/supplementary；方法节无训练超参表；"fixed-window relative encoding protocol"、"margin>p50"、N(0, 0.02²) 的量纲基准等关键协议细节缺失，无法复现。
- "We release all checkpoints, evaluation logs, and per-task results" 无匿名链接（双盲下应提供 anonymized repo）；\date 行 "October 2026 (draft for RSS 2027)" 应删；未用 \todo 宏残留；全文未使用 RSS 官方模板（裸 article 11pt + 1in margin），页数合规无法评估（RSS 全文限 8 页）。

---

## III. 其他改进建议 (Other Suggestions for Improvement)

1. **相对数修辞**："+73% relative" 作为摘要与结论的唯一收益数字，在 n=100 下未配对 95% CI 的相对数区间约 [−16%, +162%]——建议统一改报绝对差 + CI，相对数至多并列出现。
2. **λ 曲线与代号系统**："17/22·20/16%" 改为表格；HaloACT-15k/-v2/-D/act60k/R1–R3/e0 系内部代号需一次性定义或统一；"BW1000 DCU" 对国际审稿人不可读（首次出现给出说明），DCU/GPU/CPU 指称前后一致。
3. **"Hallucination-Aware" 命名**：全文未定义幻觉、未测量幻觉——要么给出可操作定义与测量，要么改名（且与 HaWMPO 撞名）。
4. **Table 1**："Four independent decouplings" 名不副实（4 个混杂配置、3 行在地板/噪声内），且本节"offline 指标无效"的教训与 §4.3 用 offline AUC 佐证 v3 head 自相矛盾；建议改为 2×2 干净消融。
5. **pooled corr ≈ 0 的解读**：n=30 混合 suite 间与 suite 内变异源，推不出 "no universal monotone law"；建议分层分析或删除。
6. **蒸馏阶梯混杂**：R1→R2 同时改数据量（4×）与过滤，无法归因于 margin filtering；补 1.8k-filtered 与 7.2k-unfiltered 两个格子；R3 塌缩补 ≥3 seeds 与 candidate diversity 度量（如候选间动作/latent 距离），否则"多样性丧失"只是猜想。
7. **候选生成协议**：说明 0.02 相对动作归一化的量纲、per-step/per-chunk、gripper 维处理、chunk 长度 T；补扰动幅度敏感性。
8. **写作**：摘要 300+ 词单段拆分；标题 " --- and When It Helps" 口语化后缀在 II.3 完成前建议撤下；"goal-blind policy" 等自造词定义。
9. **文献覆盖**：补 Deep Ensembles、PETS（或 model-based RL disagreement 一支）、LLM best-of-N/PRM、VLA 主线（RT-2/OpenVLA/π0）、以及上述 2026 竞品（ContactGuard、BOKBO、2606.20754/2606.18043 两个 VLA 不确定性工作）。
10. **双盲合规**：anonymous 仓库 + 去除草稿痕迹；release 声明与匿名性一致。

---

## 附：审稿证据与来源

- 统计复算：Wilson 95% CI 与两比例检验均为审稿人独立实算（正文所列数值）；相关系数检验 n=10 临界值 |r|=0.632；part-whole 偏差 E[Cov(base,Δ)]=−Var(base)。
- 逐格复算依据：paper/per_task_heatmap.png（spatial base 逐任务 [20,20,0,10,0,40,20,0,0,0] → +BoN-5 [90,20,0,0,0,70,10,0,0,0]）。
- 引用核查来源：arxiv.org 摘要页（2510.10975, 2506.17811, 2506.09985, 2503.08558, 2510.03827, 2603.08519, 2607.26789, 2610.12285, 2510.05681, 2508.03645, 2602.12281）；HaWMPO：arXiv:2609.09941；CoVer-VLA：官方仓库 github.com/cover-vla/cover-vla + arXiv:2602.12281。
- RSS 2027 制式：[RSS Call for Papers](https://roboticsconference.org/information/cfp)（扩展摘要 6 页=5+1，2026-12-04 截止；全文 8 页上限；双盲）。
