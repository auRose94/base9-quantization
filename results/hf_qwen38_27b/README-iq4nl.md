---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B-Base
tags:
- quantization
- gguf
- iq4_nl
---

# Qwen3.8-27B — stock-compatible IQ4_NL GGUF

A plain IQ4_NL quantization of Qwen3.8-27B that loads in any current
llama.cpp / Ollama / LM Studio build — no custom runtime needed. 15.6 GiB
(70% smaller than f16); embedding table + output head kept at Q8_0 for
head fidelity.

Produced by materializing the ninths (k9) stream container
(`Qwen3.8-27B-ninths-k63-e99-GGUF` on this Hub) with a deterministic
per-32-block RTN packer, byte-verified against the fork's runtime
materializer. The container remains the master copy: it can be
re-materialized to exact Q8_0 or f16 at any time.

Measured on an idle Ryzen AI Max+ 395 (Radeon 8060S, 96 GiB unified
memory, full-GPU residency, 16 KiB context):
- load: **~13 s** (no decode-on-load — the file is the materialized model)
- prompt eval: **134 tok/s**
- generation: **13.1 tok/s** plain, **27.0 tok/s with `--spec-type draft-mtp`**
  (2.06×; 64% draft acceptance, 2.93 tokens/draft) — this arch's
  gated-delta-net scan is compute-bound, so speculation helps a lot
- 24-task Godot-4 battery (greedy, mechanical graders): 21/24

Supports upstream speculative decoding via `--spec-type draft-mtp`
(the model ships its MTP layer, materialized).

Apache-2.0 (inherited from the base model).
