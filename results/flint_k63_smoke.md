# Flint (40B-A2.5B MoE) → ninths k63/e99 v2 — first run on a brand-new arch

**2026-10-27, session on the A9 Mega.** Flint = RecursiveSI's 40B-A2.5B MoE coding
model (Apache-2.0, released ~2026-10-21): 49 layers (2 dense + 47 MoE, 6+1 of 256
experts), 35 SWA + 14 full-attention layers, per-head QK-RMSNorm + attention
sinks, RoPE only on non-full layers, sigmoid router, learnable softmax, 262k
ctx, vocab 150,144. Everything below built and evaluated with our k9 fork on
gfx1151 (HIP), no per-arch tuning of the ninths pipeline.

## Pipeline (glm_k9_build.py, unmodified recipes)

| stage | result |
|---|---|
| reference | ggml-org/Flint-GGUF **Flint-BF16.gguf** (DAS ref; sha256 `bddf310f…` verified) |
| pass1 | 486 tensors quantized, **28.39 GB blobs = 5.713 b/param**, 556 s wall |
| pass2 | `results/flint_k63_e99_v2.gguf` **26.54 GiB** (vs Q8_0 39.44 GiB = −33%, vs BF16 74.2 GiB = −64%) |
| gates | `token_embd.weight` (k99), `blk.0.attn_q.weight`, `blk.6.ffn_gate_exps.weight` (256-expert rank-3 flatten), `output.weight` (k99): digits **0 mismatches**, scales **0.000 ulp**, q8_0 materialization **byte-exact vs python** — all PASS |
| stock exports | `results/flint_iq4nl.gguf` 21.21 GiB (iq4_nl body, k99 heads q8_0-exact), `results/flint_k63_as_q8_0.gguf` 39.44 GiB (K9 digits re-packed as stock q8_0 — the CPU-mmap eval twin, weights byte-equal to the K9 GGUF runtime) |

## Fork port (k9-flint branch, local)

Cherry-picked upstream `a37cc46104` (the only flint commit upstream so far) onto
`k9` → `61b433fbd`. Conflict resolutions: arch/model/vocab/test lists unioned
with foreign-arch tokens (VLLM_DECISION2, KOLIBRI1, MUSE_SPARK, K2_HORIZON)
dropped; graph.cpp MoE down-proj kept the fork's non-slots calling convention
(upstream's hadamard/slots MoE-cache feature NOT ported) + the amax rescale
flint needs (`build_moe_ffn`: FLINT added to MISTRAL4/MELLUM; `build_amax_src1_f16`
ported whole, scale 1/1024). `build_lora_mm_id` decl gained the upstream `slots`
param; the definition ignores it (fork callers pass nullptr).

**Runtime bug (fixed):** upstream's flint.cpp lacks the final
`ggml_build_forward_expand(gf, cur)` after `res->t_logits = cur` — this fork's
framework expands graphs explicitly per-model (cf. qwen3moe.cpp), so the final
norm + output matmul stayed orphaned: the graph "ran" but `result_output` was
never scheduled → `llama-context.cpp` `ggml_backend_sched_get_tensor_backend(t_logits)`
returned nullptr on the first decode (warmup or request) and aborted. One-line
fix in `src/models/flint.cpp`. Flagged as an upstream-PR candidate only if
upstream's implicit-expansion path is actually missing it too — not tested there.

Serving notes: flint's chat_template.jinja crashes llama.cpp's jinja lexer on
`"\x00"` (removed into `chat_template_llamacpp.jinja` — 1 transform, negligible);
server = `--no-warmup` not needed after the fix; template file passed via
`--chat-template-file`.

## A/B on gfx1151 (HIP, -ngl 99, plain processes)

People eval: llama-perplexity, corpus = first 300 KB of results/wiki2_raw_test.txt
(150k-token window, session-24 corpus), defaults + `-t 16`.

| arm | file | size | ppl (300k wiki2) | battery 24 GDScript tasks @ greedy 2500-tok |
|---|---|---|---|---|
| Q8_0 (ggml-org) | Flint-Q8_0.gguf | 39.44 GiB | **22.8678 ± 0.511** | 12/24 |
| **K9 k63/e99 v2** | flint_k63_e99_v2.gguf | **26.54 GiB** | **23.6822 ± 0.536** | **19/24** |
| IQ4_NL (ours) | flint_iq4nl.gguf | 21.21 GiB | 27.5552 ± 0.642 | 19/24 |
| Q4_K_M (ggml-org) | Flint-Q4_K_M.gguf | 21.78 GiB | TBD | TBD |

Reading so far:
- **K9 k63/e99 at 26.54 GiB lands ~0.8 ppl (+3.6% rel) above true Q8_0** at −33%
  size. The GLM-4.5-Air k63 equivalence (Δppl ≈ +0.0) did NOT fully transfer to
  flint — a rougher reading than the GLM/Qwen runs, still far below the 4-bit tier.
- The battery is **not** a quality metric between these arms: greedy outputs
  diverge chaotically off rounding differences (task pass/fail sets are
  disjoint); it is a behavior sample. Note flint is reasoning-first — battery
  budget 480 tok → 4/24, 1200 → 9/24, 2500 → 19/24 for the K9 arm.
- Load times (HIP, materialize-on-load): K9 26.54 GiB ≈ 36 s; Q8_0 ≈ 15 s;
  IQ4_NL ≈ 12 s.

## Provenance / housekeeping

- ref kept at `/home/rose/flint_ref/Flint-BF16.gguf` (public, re-downloadable; delete any time)
- eval artifacts on `/mnt/Matrix/Work/base9-quantization/results/`: flint_k63_e99_v2.gguf,
  flint_iq4nl.gguf, flint_k63_as_q8_0.gguf, glm45air_ab_flint_*.jsonl
- ppl logs: /home/rose/flint_ref/ppl_*.log (copied verbatim into results/flint_ppl/)
- NOT published anywhere (HF/fork push) pending rose's go.