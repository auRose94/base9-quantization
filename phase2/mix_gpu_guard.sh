#!/usr/bin/env bash
# Sentry fence for the mix run: no foreign mix-trainer or mix-adapter eval may
# run while the fence is alive — they killed three launches and, via an 18:49
# RAM cascade, the desktop's compositor. Accepted residents: anything inside
# mixpipeline.service or k9chat.service cgroups (this run's own stages), plus
# the fence itself.
#
#   stop with:  pkill -f mix_gpu_guard
#   status:     tail /tmp/mix_guard.log
PATTERNS=(
    "train_qlora.py --data out/godot_cpp_mix.jsonl"
    "eval_gdscript.py --model Qwen/Qwen2.5-Coder-14B-Instruct --nf4 --adapter out/qlora_14b_mix"
    "eval_cpp.py --model Qwen/Qwen2.5-Coder-14B-Instruct --nf4 --adapter out/qlora_14b_mix"
)
ACCEPT_CGROUPS="mixpipeline.service k9chat.service"
echo "[guard $(date +%H:%M:%S)] sentry armed (accepts: $ACCEPT_CGROUPS)"
while sleep 30; do
    for pat in "${PATTERNS[@]}"; do
        for pid in $(pgrep -f "$pat" 2>/dev/null); do
            [ "$pid" = "$$" ] && continue
            cg=$(cat /proc/$pid/cgroup 2>/dev/null) || continue
            keep=""
            for ac in $ACCEPT_CGROUPS; do
                case "$cg" in *"$ac"*) keep=1; break ;; esac
            done
            [ -n "$keep" ] && continue
            echo "[guard $(date +%H:%M:%S)] killed stray $pat pid $pid"
            kill -9 "$pid" 2>/dev/null
        done
    done
done