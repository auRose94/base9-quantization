# 14. Strata assessment — future engine-support candidate

Date: 2026-11-08. Status: **examination only, no pivot** (rose's framing).
Repo: https://github.com/Niko1221/Strata (MIT, v0.2.9, commit 74680f9).
Investigation: shallow clone at /tmp/strata-investigate (may be cleaned).

## 1. What Strata is

A from-scratch local-inference engine + desktop/terminal apps for running
huge MoE models on consumer hardware. NOT a llama.cpp fork:

- `src/` = own engine (`fnx` namespace): core, plan, program, spec
  (drafting), split (pack streaming), kernels (AVX expert kernels),
  qwen35/ dsv41 (per-model-family modules), ngram (28.8 GB prompt-lookup
  table), prefill.
- `third_party/ggml/` = ONLY `ggml-common.h` (quant/data-layout header,
  pinned at ggml 3cf03257f, 2026-09) + license. The GPU side runs on
  **upstream llama.cpp's ggml-cuda** via a configure-time HEADER OVERLAY
  (`STRATA_MMQ_TQ_OVERLOAD`, cmake/strata_ggml_tq_overlay.cmake — the same
  mechanism they use for their Gyro trellis MMQ kernels).
- Own minimal GGUF reader: `src/split/src/gguf.cpp` (487 lines, mmap,
  closed type table incl. **T_Q2_0 (64,18)** — the lineage our K9 descends
  from), `model.cpp` (1666 lines) for the split/pack layer.
- Docs of note: DISJOINT_EXPERT_CACHE, EXPERTDEC, BATCHED_DMA,
  KV_PREFETCH, EXCHANGE_ROTATION, GDN_CONTRACT (gated DeltaNet — the same
  GDN our triton work covered), KVA (approximate prefill).

## 2. Architecture (HOW_IT_WORKS.md)

The same problem we are solving, solved their way, at 1.75-2.25 bpw:

- GPU: attention + DeltaNet mixers, routers, shared experts, output head,
  MTP draft, KV cache, and an **adaptive expert cache** filling the rest of
  VRAM with most-used experts.
- RAM: ALL experts pinned (24,576 of them for Qwen3.8-Flash-Next); CPU
  computes cache misses in place, in parallel with the GPU. Low-RAM mode:
  keep in RAM only what the GPU does not hold, stream the rest from SSD.
- SSD: the n-gram table + the packs themselves ("what does not fit in
  memory is read from your SSD").
- MTP drafting (2.4-3.2 tok/pass), prompt-lookup drafting, KVA approximate
  prefill; next-layer experts prefetch over PCIe during attention.
- Hardware: RTX 20-50, RX 7700-9070 / AI PRO R9700, **Strix Halo (the A9
  Mega's exact platform — their headline config: DeepSeek V4.1-Flash
  2.00 bpw = 257 GB pack on 128 GB unified, 19-21 t/s code)**, DGX Spark,
  Metal, Intel Arc. CUDA + HIP.

Model families: Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, DeepSeekV4.1-Flash —
all families our fork already has arch support for (qwen35/qwen3.8), and
their packs are ISTA-DASLab / Unsloth GGUFs.

## 3. Fit with ninths

- **Same thesis**: giant MoE on tight hardware. They do it at 1.75-2.25 bpw
  + SSD streaming; we do k63 (5.744 b/p, ppl-free) + M3 expert paging.
  Complementary tiers: k63 = a quality tier ABOVE their IQ2/IQ3 packs; the
  calibrated k5/k3 low-k rungs would compete AT their tier once
  calibration lands.
- **Their expert-cache design IS our M3**: expert-level cache + decode on
  demand + CPU/GPU overlap. Our v3 page-lane decode maps onto their miss
  path naturally.
- **Memory math for 5.3-Flash on the A9**: k63 container 209.86 GiB vs
  their DeepSeek 257 GB pack — same class of fit; both rely on SSD streaming
  for what RAM cannot hold.

## 4. What a ninths port would take (small, concrete)

1. Their GGUF type table (`fnx::type_info`) + vendored `ggml-common.h`:
   add the K9 types + block sizes (we already keep the fork's
   ggml-common.h diff for K9_4/6/7/KV).
2. CPU expert kernel: port the k63/e99 CPU dequantizer (exp28 C coder —
   byte-exact, fast, standalone) as a fnx kernel; their design already
   computes cache misses on CPU.
3. GPU MMQ for K9: via their existing header-overlay mechanism — the same
   overlay path our MMVQ-HIP roadmap item targets.
4. M3 hook (decode selected experts before the gather) = their cache-miss
   handler; our M3 CPU-side logic ports directly.
5. Pack tooling: an exporter that lays out a ninths container as a Strata
   pack (their pack format is just split GGUF shards + metadata).

Licensing: MIT both sides. **The one-fork-per-network rule does not block
this** — Strata is not a llama.cpp fork, so ninths support is a PR to
their repo (or a maintained overlay), not a competing fork. Their
CONTRIBUTING.md is discussion-first with maintainer scope agreement —
the same shape as our llama.cpp upstream policy; AI assistance welcome
with contributor responsibility.

## 5. Risks / open questions

- Per-model-family engine code (src/qwen35, src/dsv41): glm5_next
  (5.3-Flash) needs a family module or generic paths — unverified.
- Their type table is deliberately small; a K9 PR needs maintainer buy-in
  (issue-first).
- The vendored ggml-common.h pin (3cf03257f) may drift from our fork's
  K9 header diffs.
- GPU-side ninths (K9 MMQ) depends on the MMVQ-HIP roadmap item landing
  in our fork first; their overlay rides upstream llama.cpp, which does
  not carry K9 types.
- Project maturity: v0.2.x, single primary maintainer; watch, don't
  depend.

## 6. Recommendation

Watch + prepare, don't pivot. Sequencing:
1. (now) 5.3-Flash paged serve on OUR fork — the current rung.
2. (after) a throwaway port experiment: k63 container → fnx GGUF reader +
   CPU dequant kernel, byte-exact gate vs our decoder (one session's work;
   the C coder and goldens exist).
3. (only if it gates green) issue-first contact with the maintainer —
   bring the ppl-free-at-5.7bpw + M3-paging evidence and the port sketch.
