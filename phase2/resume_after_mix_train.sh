#!/usr/bin/env bash
# Tail of the mix chain, detached from the killed mix_and_train_cpp.sh run
# (that chain aborted early; the 14B trainer itself was moved to the 5060 Ti
# with a --resume restart). Waits for the resume-train PID, then runs the
# original post-train stages: both eval suites on the freed 7900 XT,
# merge, K9 containers, CPU ppl.
#   usage: resume_after_mix_train.sh <train pid>
ROOT=/home/rose/Work/base9-quantization
LOG=$ROOT/phase2/out/mix_and_train.log
TRAIN_PID=$1
cd "$ROOT/phase2" || exit 1
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
VENV=$ROOT/.venv-rocm/bin/python
BASEVENV=$ROOT/.venv-baselines/bin/python

log "watching resume-train pid $TRAIN_PID (log out/train_14b_mix_resume.log)"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 120; done
if ! grep -aq '"minutes"' out/train_14b_mix_resume.log; then
    # '"minutes"' is the last key of the completion summary; plain "adapter →"
    # would false-match our own resume-verify print (that bug ran the post-train
    # stages twice on the stale adapter on 2026-10-05)
    log "RESUME TRAIN FAILED — post-train stages skipped"
    exit 1
fi
log "resume train finished — running the mix chain's post-train stages"

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