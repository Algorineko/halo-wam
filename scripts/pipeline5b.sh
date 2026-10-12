#!/bin/bash
# 第五段补跑：① object 验证器臂 task 0/1（停机夜 OOM 殉职，晚到结果 8 任务已干净）
# ② libero_goal 基线 ③ libero_goal 验证器 ④ 合并 object 验证器为干净 10 任务
set -e
P=/home/tione/notebook/home/arianliu/project/halo-wam
cd "$P"
source "$P/scripts/env.sh"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=4

echo "$(date +%F\ %H:%M) pipeline5b resume start" >> logs/pipeline5.log

# ① object verifier 补 task 0,1（与 235157 那份 8 干净任务同协议同种子）
python scripts/eval_libero_parallel.py --suite libero_object --episodes 10 --workers 2 \
  --policy haloact_v --ckpt checkpoints/act60k/best.pt \
  --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 --task-ids 0,1 \
  > logs/eval_ms_obj_ver_fix.log 2>&1
grep "平均成功率" logs/eval_ms_obj_ver_fix.log >> logs/pipeline5.log || true

# ② goal 基线
python scripts/eval_libero_parallel.py --suite libero_goal --episodes 10 --workers 2 \
  --policy haloact --ckpt checkpoints/act60k/best.pt \
  > logs/eval_ms_goal_base.log 2>&1
grep "平均成功率" logs/eval_ms_goal_base.log >> logs/pipeline5.log || true

# ③ goal 验证器
python scripts/eval_libero_parallel.py --suite libero_goal --episodes 10 --workers 2 \
  --policy haloact_v --ckpt checkpoints/act60k/best.pt \
  --verifier checkpoints/e0v5/dyn_ens_k5_s2000.pt --n-cand 5 \
  > logs/eval_ms_goal_ver.log 2>&1
grep "平均成功率" logs/eval_ms_goal_ver.log >> logs/pipeline5.log || true

# ④ 合并 object 验证器：晚到的 8 任务 + 补跑 2 任务 → 干净 10 任务
python - <<'EOF'
import glob, json
partial = "eval_results/libero_object_haloact_v_20261003_235157.json"
rerun = sorted(glob.glob("eval_results/libero_object_haloact_v_*.json"))[-1]
by_task = {}
for f in (partial, rerun):
    d = json.load(open(f))
    for r in d["results"]:
        if not r.get("error"):
            by_task[r["task"]] = r
assert len(by_task) == 10, f"expected 10 clean tasks, got {len(by_task)}"
merged = {"meta": json.load(open(partial))["meta"] | {"merged_from": [partial, rerun], "merged_ts": "pipeline5b"},
          "results": sorted(by_task.values(), key=lambda r: r["task"]),
          "mean_rate": sum(r["rate"] for r in by_task.values()) / 10}
out = "eval_results/libero_object_haloact_v_MERGED10.json"
json.dump(merged, open(out, "w"), indent=2, ensure_ascii=False)
print(f"[merge] object verifier clean 10-task mean = {merged['mean_rate']:.1%} -> {out}")
EOF
echo DONE > logs/pipeline5b.done
echo "$(date +%F\ %H:%M) pipeline5b complete" >> logs/pipeline5.log
