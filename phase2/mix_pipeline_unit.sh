#!/usr/bin/env bash
# The full serialized 14B-mix pipeline as a crash-proof systemd user unit:
#   wait for the 7900 XT to be compute-free (split_active.sh, route-agnostic:
#   BIOS-hybrid iGPU compositing, the kwin-gpu config route, or a logout
#   window) → resume-train the 14B from the step-3600 checkpoint on the full
#   card → the follower chain (draws-3 evals → merge → K9 k9/15/63 → CPU ppl).
# If the machine reboots mid-run, re-arm (systemd refuses a double run;
# the checkpoint makes every interruption resumable):
#   systemd-run --user --unit=mixpipeline \
#     bash /home/rose/Work/base9-quantization/phase2/mix_pipeline_unit.sh
# Stop the whole thing:  systemctl --user stop mixpipeline  (+ pkill -f mix_gpu_guard to also lift the fence)
ROOT=/home/rose/Work/base9-quantization
cd "$ROOT/phase2" || exit 1
LOG=$ROOT/phase2/out/mix_and_train.log
PY="$ROOT/.venv-rocm/bin/python"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

until bash split_active.sh; do
    log "pipeline armed — waiting for the 7900 XT to be compute-free (compositor on iGPU / 5060 Ti)"
    sleep 60
done
log "SPLIT DETECTED — starting the 14B mix train FROM SCRATCH (the resume path's
unexplained +0.6 GiB surplus OOM'd 13 launches on a clean card; from-scratch
verified fit: 25-step probe with backward+opt-step+save completed 09:16).
Prior step-3600/4200 state preserved: out/qlora_14b_mix/adapter_resume3600
and the s4200 backup; interim evals (gd d3 83.3%, cpp d3 43.8-50%) stand."
"$PY" train_qlora.py --data out/godot_cpp_mix.jsonl --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --out out/qlora_14b_mix --epochs 1 --lr 3e-5 --save-every 600 --log-every 100 \
    --max-steps 6668 \
    > out/train_14b_mix_resume.log 2>&1 &
TPID=$!
log "train pid $TPID (log: out/train_14b_mix_resume.log)"
bash resume_after_mix_train.sh "$TPID"