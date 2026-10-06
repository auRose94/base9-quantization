#!/usr/bin/env bash
# Night-cap after the lr-1e-4 14B run regressed on the mechanical eval:
# the same recipe at 1/3 the LR (3e-5), then self-evaluates both ways.
# Train ~4 h on the 7900 XT; evals ~30 min. Log: phase2/out/overnight2.log
ROOT=/home/rose/Work/base9-quantization
LOG=$ROOT/phase2/out/overnight2.log
cd "$ROOT/phase2" || exit 1
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
VENV=$ROOT/.venv-rocm/bin/python

log "14B follow-up starting: lr 3e-5, 1 epoch, chunked-CE, batch 1 x accum 16"
"$VENV" train_qlora.py --data out/godot4_sft.jsonl \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --out out/qlora_14b_lr3e5 \
    --epochs 1 --lr 3e-5 --save-every 400 --log-every 50 \
    > out/train_14b_lr3e5.log 2>&1 \
    || { log "TRAIN FAILED"; exit 1; }
grep -aE "base val|final_loss|val_loss|minutes" out/train_14b_lr3e5.log >> "$LOG"

log "greedy eval (ladder protocol)"
"$VENV" eval_gdscript.py --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --nf4 --adapter out/qlora_14b_lr3e5/adapter --draws 1 \
    --tag tuned14b_lr3e5_greedy > out/eval_tuned14b_lr3e5_greedy.log 2>&1 \
    || log "GREEDY EVAL FAILED"
log "draws-3 eval"
"$VENV" eval_gdscript.py --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --nf4 --adapter out/qlora_14b_lr3e5/adapter --draws 3 \
    --tag tuned14b_lr3e5_d3 > out/eval_tuned14b_lr3e5_d3.log 2>&1 \
    || log "D3 EVAL FAILED"
cat out/eval_tuned14b_lr3e5_greedy.summary.json \
    out/eval_tuned14b_lr3e5_d3.summary.json >> "$LOG"
log "NIGHT-CAP DONE $(date)"