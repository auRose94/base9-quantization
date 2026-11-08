#!/usr/bin/env bash
# M3-HIP gate on the A9 (docs/13 §2.6): the cache-bank upload chain, end to end.
#
# Root cause under test (fixed in llama-context.cpp sched_copy_experts):
# the fallback leg of the expert upload chain read K9 paged bytes straight
# from the residency region without decoding. It fires whenever the
# MoE-cache slot-map path is disabled — cache off, budget too small, or
# prefill batches above max_batch (32) — which is EVERY prefill >= 33
# tokens. mul_mat_id over host weights offloads to the GPU at batch >= 32
# (GGML_OP_OFFLOAD_MIN_BATCH), so CPU-only builds never take this path
# (the in-op hook decodes there) — the garbage was HIP-only and
# resident-only by construction.
#
# Gate = greedy outputs token-identical across:
#   eager                     reference (k9_buf decoded at load)
#   resident + moe cache ON   cache leg at decode, fixed fallback at prefill
#   resident + cache OFF      fixed fallback at prefill (decode: batch 1 keeps
#                             mul_mat_id on CPU -> in-op hook)
#   resident + tiny budget    real evictions under both legs
set -u
export GGML_NO_BACKTRACE=1

BIN=/mnt/Matrix/Work/llama.cpp/build-hip-a9/bin/llama-cli
MODEL=/mnt/Matrix/Work/base9-quantization/results/olmoe_k63_v3pages.gguf
OUT=/tmp/k9gate-hip-a9
mkdir -p "$OUT"

# 55 tokens of prefill: past max_batch(32) AND past the op-offload
# threshold(32) — forces the fallback leg at prefill in every resident run.
PROMPT="The quick brown fox jumps over the lazy dog while the river winds through the valley of old stone bridges where merchants once traded silk and spices under lanterns that swayed in the evening breeze near the harbor of forgotten ships."

extract() { # strip the interactive chrome, keep the completion text
    sed -e 's/\x1b\[[0-9;]*m//g' \
        -e '/^available commands:/,/^$/d' \
        -e '/^  \//d' \
        -e '/^\[ Prompt:/,$d' \
        -e 's/^> //' \
    | grep -v '^\s*$'
}

run() { # name, envs, extra-args...
    local name="$1"; shift
    local envs="$1"; shift
    echo "=== $name ==="
    ( export $envs
      "$BIN" -m "$MODEL" -ngl 99 -p "$PROMPT" -n 48 --temp 0 --simple-io \
             -b 512 --no-display-prompt "$@" ) > "$OUT/$name.raw" 2> "$OUT/$name.err"
    echo "  exit=$? bytes_raw=$(wc -c < "$OUT/$name.raw")"
    extract < "$OUT/$name.raw" > "$OUT/$name.txt"
    echo "  bytes: $(wc -c < "$OUT/$name.txt")"
}

run eager       "PATH=$PATH"
run res_cache   "K9_RESIDENT=1"                    --moe-cache-mib 512
run res_nocache "K9_RESIDENT=1"
run res_small   "K9_RESIDENT=1 K9_BUDGET_GIB=0.25" --moe-cache-mib 512

echo; echo "=== token-identity vs eager ==="
for m in res_cache res_nocache res_small; do
    if diff -q "$OUT/eager.txt" "$OUT/$m.txt" >/dev/null; then
        echo "PASS  $m"
    else
        echo "FAIL  $m  (diff: $(diff "$OUT/eager.txt" "$OUT/$m.txt" | head -4 | tr '\n' ' '))"
    fi
done
