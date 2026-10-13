"""配对统计分析器（docs/stats_prereg.md 预注册规则的唯一执行体）。

用法:
  python scripts/analyze_paired.py --treat <json> --base <json> [--label sp_v50_vs_base]
  python scripts/analyze_paired.py --family B   # 族级 Holm 校正（progress.md 登记的对比对）

规则（预注册，不得偏离）:
  - suite 级: 500 配对 episode 的 Δ 成功指示 bootstrap (B=10000, episode 重采样单元),
    报均值差 (pp) + 95% percentile CI; 判定 = CI 不含 0 且 Holm 后 α=0.05
  - 任务级: McNemar 精确检验 (50 配对 eps)
  - 稳健性: leave-one-task-out 范围 + 剔除 |Δ| 最大任务后的 suite 差
"""
import argparse
import glob
import itertools
import json

import numpy as np
from scipy import stats

RESULT_DIR = "/home/tione/notebook/home/arianliu/project/halo-wam/eval_results"


def load_ep_matrix(fp):
    """JSON → (T, E) 0/1 矩阵，任务按名排序（跨臂对齐）。"""
    d = json.load(open(fp))
    rows = {}
    for r in d["results"]:
        if r.get("error") or "ep_success" not in r:
            raise ValueError(f"{r['task']}: 无逐 episode 向量（旧协议 run？）")
        rows[r["task"]] = np.array(r["ep_success"], dtype=np.int8)
    tasks = sorted(rows)
    return np.stack([rows[t] for t in tasks]), tasks


def paired_bootstrap(delta, n_boot=10000, seed=0):
    """delta: (T,E) 配对差. 返回 (mean_pp, lo, hi)."""
    rng = np.random.RandomState(seed)
    flat = delta.reshape(-1)
    idx = rng.randint(0, len(flat), size=(n_boot, len(flat)))
    means = flat[idx].mean(axis=1) * 100
    return flat.mean() * 100, np.percentile(means, 2.5), np.percentile(means, 97.5)


def mcnemar_exact(a, b):
    """a,b: 同任务配对 0/1 向量. 返回 (b01, b10, p)."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = stats.binomtest(min(n01, n10), n, 0.5).pvalue * 1.0
    return n01, n10, p


def compare(treat_fp, base_fp, label=""):
    T, tasks = load_ep_matrix(treat_fp)
    B, tasks_b = load_ep_matrix(base_fp)
    assert tasks == tasks_b and T.shape == B.shape, "任务集或 eps 数不一致，无法配对"
    delta = T - B
    mean_pp, lo, hi = paired_bootstrap(delta)
    # 任务级 McNemar
    task_ps = [mcnemar_exact(T[i], B[i])[2] for i in range(len(tasks))]
    # 稳健性: LOTO + 剔最大 |Δ| 任务
    task_means = delta.mean(axis=1) * 100
    loto = [(tasks[i], np.delete(task_means, i).mean()) for i in range(len(tasks))]
    worst = int(np.argmax(np.abs(task_means)))
    drop_worst = np.delete(delta, worst, axis=0).mean() * 100
    return {
        "label": label or f"{treat_fp.split('/')[-1]}_vs_{base_fp.split('/')[-1]}",
        "treat_rate": T.mean() * 100, "base_rate": B.mean() * 100,
        "delta_pp": mean_pp, "ci95": [lo, hi],
        "significant_raw": bool(lo > 0 or hi < 0),
        "task_mcnemar_p": dict(zip(tasks, task_ps)),
        "loto_range": [min(x[1] for x in loto), max(x[1] for x in loto)],
        "drop_worst_task": {"task": tasks[worst], "delta_pp": drop_worst},
        "n_pairs": int(delta.size),
    }


def holm(results):
    """results: list of dict (含 significant_raw 与最小任务级 p). 原地加 Holm 字段。"""
    ps = [min(min(r["task_mcnemar_p"].values()), 1.0) for r in results]
    order = np.argsort(ps)
    m = len(ps)
    ranked = []
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * ps[i])
        adj = max(adj, prev)  # 单调性
        prev = adj
        ranked.append((i, adj))
    for i, adj in ranked:
        results[i]["p_holm"] = adj
        results[i]["significant_holm"] = bool(adj < 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treat", type=str, default="")
    ap.add_argument("--base", type=str, default="")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--family", type=str, default="", help="A/B/C/D/E → 预注册对比族")
    args = ap.parse_args()

    def latest(pattern):
        hits = sorted(glob.glob(f"{RESULT_DIR}/{pattern}"))
        if not hits:
            raise SystemExit(f"no file: {pattern}")
        return hits[-1]

    if args.treat:
        r = compare(args.treat, args.base, args.label)
        holm([r])
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return

    fams = {
        # 族 A: 三 suite BoN vs base（base 臂落地后填 pattern）
        "A": [("libero_spatial_haloact_v_20261014*.json", "libero_spatial_haloact_2026101[45]*.json", "spatial"),
              ("libero_object_haloact_v_20261014*.json", "libero_object_haloact_2026101[45]*.json", "object"),
              ("libero_goal_haloact_v_20261014*.json", "libero_goal_haloact_2026101[45]*.json", "goal")],
    }
    if args.family in fams:
        rs = []
        for tp, bp, lab in fams[args.family]:
            try:
                rs.append(compare(latest(tp), latest(bp), lab))
            except (SystemExit, ValueError, FileNotFoundError) as e:
                print(f"[skip] {lab}: {e}")
        holm(rs)
        for r in rs:
            print(f"{r['label']:10s} {r['base_rate']:5.1f}% → {r['treat_rate']:5.1f}%  "
                  f"Δ={r['delta_pp']:+5.1f}pp CI[{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]  "
                  f"holm_p={r.get('p_holm', 1):.3f} {'*SIG*' if r.get('significant_holm') else ''}  "
                  f"drop-worst={r['drop_worst_task']['delta_pp']:+.1f}")


if __name__ == "__main__":
    main()
