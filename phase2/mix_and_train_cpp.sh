#!/usr/bin/env bash
# The C++ mixed-corpus run, self-synchronized on the full-slice scan marker.
#   wait full cpp_sft (214 shards) → mix (gd 63.5k + cpp ≤45k, GD-dominant)
#   → 14B QLoRA @ lr 3e-5, 1 epoch (~6 h on the 7900 XT)
#   → greedy + draws-3 evals of BOTH suites (gdscript 24 + cpp 16)
#   → merge → K9 k9/k15/k63 containers → CPU ppl pass → report
# Log: phase2/out/mix_and_train.log
ROOT=/home/rose/Work/base9-quantization
LOG=$ROOT/phase2/out/mix_and_train.log
cd "$ROOT/phase2" || exit 1
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
VENV=$ROOT/.venv-rocm/bin/python
BASEVENV=$ROOT/.venv-baselines/bin/python

# GUARD (2026-10-05 evening): the mix trainer fills its card's VRAM (~19-21 GB)
# — on a card that also composites, that starves the desktop compositor (kwin
# ENOMEM storms). Concurrent pipelines stacking ~30 GB of host RAM each caused
# the 18:49 global-OOM cascade. Abort here if a mix train/eval is already live
# anywhere. The canonical launcher is mix_pipeline_unit.sh (systemd user unit:
# waits for a compute-free card, then runs train → evals → merge → K9 → ppl
# fully serialized; setsid leaves the session but NOT the cgroup).
if pgrep -f "train_qlora.py --data out/godot_cpp_mix.jsonl" >/dev/null ||
   pgrep -f "eval_(gdscript|cpp)\.py --model Qwen/Qwen2.5-Coder-14B-Instruct --nf4 --adapter out/qlora_14b_mix" >/dev/null; then
    log "GUARD: mix train/eval already active — chain aborts (see NOTES below)"
    exit 0
fi

log "waiting for the full C++ scan to finish"
while ! grep -aq '"written"' /tmp/cpp_scan_full.log 2>/dev/null; do sleep 120; done
log "full scan done: $(grep -a '"unique"' /tmp/cpp_scan_full.log | tail -1)"

log "building the mixed corpus (gd + cpp≤45k)"
python3 - <<'EOF' >> "$LOG" 2>&1
import json, random
gd = [json.loads(l) for l in open("out/godot4_sft.jsonl")]
cpp = [json.loads(l) for l in open("out/cpp_sft.jsonl")][:45000]
rows = gd + cpp
random.Random(0).shuffle(rows)
with open("out/godot_cpp_mix.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"mixed: {len(gd):,} gd + {len(cpp):,} cpp = {len(rows):,} rows "
      f"→ out/godot_cpp_mix.jsonl")
EOF

log "training 14B on the mix (lr 3e-5, 1 epoch) — detached so it survives the
harness's cgroup-scoped child teardown (setsid leaves the session but NOT the
cgroup; the durable pattern is the systemd-user unit in mix_pipeline_unit.sh)"
setsid nohup "$VENV" train_qlora.py --data out/godot_cpp_mix.jsonl \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --out out/qlora_14b_mix \
    --epochs 1 --lr 3e-5 --save-every 600 --log-every 100 \
    > out/train_14b_mix.log 2>&1 &
TPID=$!
disown $TPID 2>/dev/null
log "train pid $TPID (own session)"
while ! grep -aq "adapter → " out/train_14b_mix.log 2>/dev/null; do
    if ! kill -0 $TPID 2>/dev/null; then
        log "TRAIN PROCESS DIED before finishing (pid $TPID)"
        exit 1
    fi
    sleep 120
done
grep -aE "base val|final_loss|val_loss|minutes" out/train_14b_mix.log >> "$LOG"

log "evals on the 7900 XT (adapter over base, both suites)"
"$VENV" eval_gdscript.py --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --nf4 --adapter out/qlora_14b_mix/adapter --draws 3 \
    --tag mix14b_gd_d3 > out/eval_mix14b_gd_d3.log 2>&1 || log "gd d3 FAILED"
"$VENV" eval_cpp.py --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --nf4 --adapter out/qlora_14b_mix/adapter --draws 3 \
    --tag mix14b_cpp_d3 > out/eval_mix14b_cpp_d3.log 2>&1 || log "cpp d3 FAILED"
cat out/eval_mix14b_*_d3.summary.json >> "$LOG" 2>/dev/null

log "merge + K9 containers + CPU ppl"
"$BASEVENV" merge_adapter.py \
    --base Qwen/Qwen2.5-Coder-14B-Instruct --adapter out/qlora_14b_mix/adapter \
    --out out/merged_14b_mix > out/merge_14b_mix.log 2>&1 \
    && tail -1 out/merge_14b_mix.log >> "$LOG" || log "MERGE FAILED"
for k in 9 15 63; do
    "$VENV" quantize_k9.py --model out/merged_14b_mix \
        --out results/qwen14b_mix_tuned_k${k}_embed99.k9 --k-body $k \
        > out/quant_14b_mix_k$k.log 2>&1 \
        && tail -1 out/quant_14b_mix_k$k.log >> "$LOG"
done
"$VENV" ppl_check_k9.py --model out/merged_14b_mix --cpu --prefix qwen14b_mix_tuned \
    > out/ppl_14b_mix.log 2>&1 && grep -a "bf16:\|k15:\|k9:\|k63:" out/ppl_14b_mix.log >> "$LOG"
log "MIX PIPELINE DONE $(date)"