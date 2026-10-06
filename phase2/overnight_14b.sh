#!/usr/bin/env bash
# Overnight 2026-10-04: the 14B Phase-2 scale run, self-contained pipeline.
#
#   7900 XT free?  <- waits for the 7B robust-eval series marker
#   weights here?  <- waits for the 14B download marker
#   1. train 14B (NF4, batch 1 x accum 16 — the 7B recipe unchanged)  ~9-20 h on RDNA3
#   2. merge on the CPU (bf16 dir)
#   3. K9 containers k9/k15/k63 on the CPU, then one CPU ppl pass over all three
#   4. NF4-hosted draws-3 evals (24 tasks) on the 7900 XT — bf16 14B fits no local card
#   5. restart K9 Chat on the 5060 Ti at morning with the k63 variant live
#
# Everything logs to phase2/out/overnight.log; run:  bash phase2/overnight_14b.sh
ROOT=/home/rose/Work/base9-quantization
LOG=$ROOT/phase2/out/overnight.log
cd "$ROOT/phase2" || exit 1
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
VENV=$ROOT/.venv-rocm/bin/python
BASEVENV=$ROOT/.venv-baselines/bin/python

log "overnight pipeline started"
while ! grep -q "downloaded →" /tmp/hf14b_download.log 2>/dev/null; do sleep 60; done
log "14B weights present"
while ! grep -q "ROBUST SERIES DONE" /tmp/robust_series.log 2>/dev/null; do sleep 60; done
log "7900 XT free — starting 14B QLoRA"

"$VENV" train_qlora.py --data out/godot4_sft.jsonl \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --out out/qlora_14b --epochs 1 \
    --save-every 400 --log-every 50 > out/train_14b.log 2>&1 \
    || { log "TRAIN FAILED — see out/train_14b.log"; exit 1; }
grep -aE "base val|final_loss|val_loss|minutes" out/train_14b.log >> "$LOG"

log "training done — merging on CPU"
"$BASEVENV" merge_adapter.py \
    --base Qwen/Qwen2.5-Coder-14B-Instruct --adapter out/qlora_14b/adapter \
    --out out/merged_14b > out/merge_14b.log 2>&1 \
    || { log "MERGE FAILED — see out/merge_14b.log"; exit 1; }
tail -1 out/merge_14b.log >> "$LOG"

log "writing K9 containers (CPU)"
for k in 9 15 63; do
    "$VENV" quantize_k9.py --model out/merged_14b \
        --out results/qwen14b_tuned_k${k}_embed99.k9 --k-body $k \
        > out/quant_14b_k$k.log 2>&1 && tail -1 out/quant_14b_k$k.log >> "$LOG"
done

log "CPU ppl pass over the 14B containers (runs while the GPU evals go)"
"$VENV" ppl_check_k9.py --model out/merged_14b --cpu --prefix qwen14b_tuned \
    > out/ppl_14b.log 2>&1 && grep -a "bf16:\|k15:\|k9:\|k63:" out/ppl_14b.log >> "$LOG" \
    || log "CPU PPL FAILED"

log "NF4-hosted draws-3 evals on the 7900 XT"
"$VENV" eval_gdscript.py --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --nf4 --draws 3 --tag base14b_nf4_d3 > out/eval_base14b_nf4_d3.log 2>&1 \
    || log "14B base eval FAILED"
"$VENV" eval_gdscript.py --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --nf4 --adapter out/qlora_14b/adapter --draws 3 \
    --tag tuned14b_nf4_d3 > out/eval_tuned14b_nf4_d3.log 2>&1 \
    || log "14B tuned eval FAILED"
cat out/eval_base14b_nf4_d3.summary.json out/eval_tuned14b_nf4_d3.summary.json >> "$LOG" 2>/dev/null

log "restarting K9 Chat on the 5060 Ti with the k63 variant live"
OLD=$(pgrep -f "chat_k9.py --preset 7b-tuned" | head -1)
if [ -n "$OLD" ]; then kill "$OLD" 2>/dev/null; sleep 3; fi
python3 "$ROOT/chat_k9.py" --preset 7b-tuned > /tmp/chat_k9_morning.log 2>&1 &
log "OVERNIGHT PIPELINE DONE $(date)"