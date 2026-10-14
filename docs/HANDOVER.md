# HALO-WAM 交接文档（2026-10-14 服务器重启前停机归档）

> 重启后恢复：**读完本文档即可零上下文续跑**。深度背景见 docs/progress.md（滚动更新）
> 与 ~/.claude/plans/cozy-toasting-puffin.md（返修总计划 Phase 0-5）。

## 一、项目当前阶段（一句话）

对抗审稿（docs/review-rss2027.md 九项）返修中，**论文已转向"预注册配对审计研究"**：
σ-select 验证器家族在 50-eps 配对功效下三 suite 全零（+73% 头条系 1.4SE 噪声），
机制闭环（候选多样性坍缩 + 78-83% 选择空转 + σ 零判别，35,902 决策审计）。
投稿目标 **CoRL 2027（~2027-03）**，系统名 σ-select（弃 HALO，撞名 HaWMPO）。

## 二、停机时点状态（2026-10-14 21:00）

### 已完成（50-eps 统计引擎，8/12 格）
| 格 | 结果 |
|---|---|
| spatial base / BoN | 17.0% / 16.0%，Δ=−1.0pp CI[−4.4,+2.6] **零** |
| object base / BoN | 5.0% / 4.8%，Δ=−0.2pp CI[−2.4,+2.0] **零** |
| goal base / BoN | 25.4% / 27.6%，Δ=+2.2pp CI[−2.0,+6.4] 零（holm 0.405）|
| r2_bon50 | 14.0% vs base 17%——"蒸馏吸收"同塌 |
| r3_base50 | done（数字见 logs/p10/r3_base50.log）|

### 被重启杀掉的进行中格（**恢复后必须重跑，无部分产出**）
- **lam025_v50**（λ 族第 1 格，card0，死于 ~7h 处）
- **r3_bon50、rand50**（card1 队列，r3_bon 死于中段）
- **cons50**（CPU 流，死于此早期）
- 未启动：lam05 / lam10 / reward_only（10b 队列后续）

## 三、恢复操作（按序，全部幂等）

```bash
cd /home/tione/notebook/home/arianliu/project/halo-wam
# 1. 重启三条流水线（done 标记幂等，已完成格自动跳过，只补被杀的格）
nohup bash scripts/pipeline10_50eps.sh  > logs/p10_launch.log  2>&1 & disown   # GPU 主族（补 r3_bon50/rand50）
nohup bash scripts/pipeline10b_lambda.sh > logs/p10b_launch.log 2>&1 & disown  # λ族（补 lam025 + 后续3格）
nohup bash scripts/pipeline10c_cpu.sh   > logs/p10c_launch.log 2>&1 & disown   # CPU（补 cons50）
# 2. 验证：10 分钟后 8 进程、cgroup <28GB
ps aux | grep eval_libero | grep -v grep | wc -l
# 3. 重建 cron（每小时 :21，7 天后到期需再建）
#    用上次 prompt：CoRL 2027 返修期推进+监督（要点：跑批盯 p10 done/收数/analyze_paired/
#    Phase 2 C1消融 --act-weight 0 重训 / Phase 3 SmolVLA 适配器 / 崩溃先诊断）
# 4. 每格落地后：python scripts/analyze_paired.py --treat <json> --base <json> --label X
```

**预计补跑时长**：lam025（~10h）+ r3_bon/rand（~14h，双卡并行）+ cons（CPU ~7h）+ λ 后 3 格（~30h）
≈ 2 天全齐。

## 四、恢复后的工作队列（按优先级）

1. **收齐审计地图**：λ 族 + cons50/rand50/r3_bon50 → 全家族零效应地图完整
   （预言：random/consensus/λ 全部 ≈ base；若某格显著≠base，机制章要改写）
2. **跨 suite 机制审计补 goal 日志**（goal_v50 的 npz 已在盘，多样性测量脚本模式见
   progress.md 2026-10-14 10:20 段——直接对 decisions/libero_goal_*/npz 跑同款）
3. **Phase 2 C1 消融**：`python scripts/train_e0.py --act-weight 0 ...`（GPU 空隙）→
   BoN 50-eps 一格（判定对比损失是否是"必要成分"——在零效应世界里此格大概率也零，
   但审稿 II.7 要求有）
4. **Phase 3 SmolVLA 适配器**（审计跨架构关键项）：本体在
   ~/data/models/smolvla_base/，eval 脚本已有 smolvla/smolvla_v 骨架，缺加载器；
   10-eps 探针定去留
5. **论文重写**（Phase 5）：paper/main.tex 现为旧正向叙事（HALO 名已弃、refs.bib 已重建），
   按"预注册配对审计"重构——标题/贡献/主表全换，stats_prereg.md 的表述纪律执行
6. **投稿前**：ContactGuard/BOKBO 文献作者补全（refs.bib 内注释标记）

## 五、铁律备忘（全部踩过坑）

- `set -u` 撞 dtk env.sh（未绑定 LD_LIBRARY_PATH）——脚本一律 `set +u`
- 双验证器臂并行 2-worker = 40GB 爆 cgroup（32GB 硬墙）——只允许各 1-worker
- AdamW 必 `foreach=False`（已全量补）；pkill 自匹配先 pgrep 人工过目
- 容器 `free` 显示宿主 2.2T 是误导；hy-smi 信 VRAM% 不信 HCU%
- 评测脚本不可在跑批中编辑（worker 子进程读盘）；统计规则见 docs/stats_prereg.md **不得改动**
- ModelScope token（.env.secrets，严禁入库）2026-10-13 起 400 失效，act_e3_r2/r3 待传

## 六、关键文件索引

- 计划：~/.claude/plans/cozy-toasting-puffin.md（Phase 0-5）
- 状态：docs/progress.md（顶部 = 最新）；审稿：docs/review-rss2027.md
- 统计：docs/stats_prereg.md（预注册，唯一判定规则）；分析器 scripts/analyze_paired.py
- 论文：paper/main.tex（旧正向叙事待重构）+ paper/refs.bib（26 条已核）
- 决策日志：eval_results/decisions/*/task*/task*.npz（候选/σ_K5/σ_K3/chosen/逐ep向量）
- GitHub: Algorineko/halo-wam；本次停机前最后 commit 见 git log -1
