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
(150k-token window, session-24 corpus), defaults + `-t 16`. All baselines
sha256-verified against ggml-org's published oids; the Q8_0 ppl was re-run on
the sha-verified copy and reproduced **22.8678 identically** (deterministic eval).

| arm | file | size | b/p (quantized part) | ppl (300k wiki2) | battery 24 GDScript tasks @ greedy 2500-tok |
|---|---|---|---|---|---|
| Q8_0 (ggml-org) | Flint-Q8_0.gguf | 39.44 GiB | 8.50 | **22.8678 ± 0.511** | 12/24 |
| **K9 k63/e99 v2** | flint_k63_e99_v2.gguf | **26.54 GiB** | 5.713 | **23.6822 ± 0.536** | **19/24** |
| Q4_K_M (ggml-org) | Flint-Q4_K_M.gguf | 21.78 GiB | ~4.85 | **23.7434 ± 0.536** | 19/24 |
| IQ4_NL (ours) | flint_iq4nl.gguf | 21.21 GiB | ~4.28 | 27.5552 ± 0.642 | 19/24 |

Verdict on flint (honest):
- The K9 k63/e99 v2 container is **byte-exact end-to-end** (gates) and serves
  fine, but the **k63 quality-rung result did NOT match GLM-4.5-Air's ppl-free
  equivalence**: K9 lands +0.81 ppl (+3.6% rel) above true Q8_0 at −33% size.
- **Q4_K_M dominates K9 k63 on this arch**: statistically identical ppl
  (Δ−0.06, within noise) at 21.78 vs 26.54 GiB. On GLM/Qwen the k63 body beat
  q4_k_m; on flint it does not.
- The IQ4_NL rung is clearly worse on flint (+4.7 over Q8_0) — more than the
  k99-protected GLM/Qwen iq4_nl runs; 4-bit hurts flint more.
- Battery is a behavior sample, not a quality metric (greedy thresholds flip
  task outcomes between rounding grids; see per-arm fail sets in the jsonl).
  Flint is reasoning-first: its K9 arm went 4/24 → 19/24 purely on battery
  budget (480 → 1200 → 2500 tokens).

So for flint specifically the research answer is: **the format machinery
(port, packer, gate chain, loader) transfers to a brand-new arch with zero
per-arch tuning — but the quality/size advantage over ggml's own k-quants
does not transfer this time; Q4_K_M is the better serving file at this
quality tier.** Candidate follow-ups if this matters to rose: k15/k9_4 rung
on flint (needs the MMVQ-HIP kernels for serving), KLD-vs-BF16 protocol run,
or an act-order/GPTQ-informed K9 pass (exp18's small-model-only lever).

Load times (HIP, materialize-on-load): K9 26.54 GiB ≈ 36 s; Q8_0 ≈ 15 s;
IQ4_NL ≈ 12 s.

## PUBLISHED (2026-10-27, rose's go)

- fork branch `k9` pushed: github.com/auRose94/llama.cpp @ `2fafd9ccc`
  (flint cherry-pick + expand fix + slots-param fix; branch `k9-flint` merged in via ff)
- base9 repo pushed: eb46e75 → 154045f
- HF (auRose94): `Flint-40B-A2.5B-ninths-k63-e99-GGUF` (model + README + both
  chat templates) and `Flint-40B-A2.5B-iq4nl-GGUF` (same)
- bf16 ref moved to cold storage (DAS verify-then-delete): sha-verified copy at
  `/run/media/rose/Warehouse 1/.flint_ref/Flint-BF16.gguf` (the /mnt/Warehouse*
  mountpoints still hold stale stacked mounts → I/O errors; live drives are at
  /run/media/rose/"Warehouse N" per udisks while the fstab repair settles)

## Budget sensitivity + agentic battery (rose's correction: flint thinks, a lot)

24-task GDScript battery, K9 arm, greedy, thinking mandatory:

| max_tokens | result |
|---|---|
| 480 | 4/24 |
| 1200 | 9/24 |
| 2500 | 19/24 |
| **6000** | **22/24 (91.7%), 723.6 s** |

Do not force short max_tokens with Flint — it plans and self-checks inside the
thinking block and (per rose) can loop there on longer agentic work until
satisfied.

## llama-bench (16 t, gfx1151, full offload)

| arm | pp512 | tg128 |
|---|---|---|
| K9 k63/e99 v2 (runtime = materialized Q8_0) | 1452 ± 31 | 49.86 ± 0.29 |
| Q8_0 | 1430 ± 20 | 49.95 ± 0.60 |
| Q4_K_M | 1589 ± 43 | 56.19 ± 0.15 |
| IQ4_NL (ours) | 1701 ± 77 | 67.43 ± 0.16 |

## exp27 mini-KLD protocol vs BF16 base (60 KB corpus slice)

| candidate | PPL(Q) | PPL(Q)/PPL(base) | ln(PPL(Q)/PPL(base)) ± | Cor |
|---|---|---|---|---|
| Q8_0 | 18.9704 | 1.0395 | 0.0388 ± 0.0082 | 98.6% |
| Q4_K_M | 19.3943 | 1.0628 | 0.0609 ± 0.0119 | 96.9% |
| K9 k63/e99 | 19.7279 | 1.0811 | 0.0779 ± 0.0110 | 97.5% |
| IQ4_NL | 22.3863 | 1.2267 | 0.2044 ± 0.0180 | 93.3% |

(base = Flint-BF16, PPL 18.2487 on this slice; kld-base file kept at
results/flint_ppl/flint_60k.kld for exact mean-KL re-runs — the per-chunk KL
column was filtered out of this run's capture.)

## Fork synced with upstream master (2026-10-27, rose's go)

- `git merge upstream/master` (514 commits, to `d05706b4f`) onto k9 → merge
  `c5029ebd4`, pushed. Only 6 conflicted files: llama-vocab.h (pre-type enum),
  models/flint.cpp add/add (upstream never touched it since the PR → kept ours
  with the expand fix), tests/test-llama-archs.cpp, llama-model.cpp (took
  upstream's VLLM_DECISION2 case — the enum now exists), llama-graph.cpp (kept
  fork's non-slots MoE body at first, then adopted upstream's whole
  build_lora_mm_id/hadamard/moe_cache machinery since the merge brought it in
  — callers pass slots), ggml-cpu/arch-fallback.h (k9 fallback defines kept;
  the q2_0 alias removed where upstream moved the impl to top-level quants.c).
  Two post-merge fixes: duplicate `llama_model_flint` struct in models.h
  (auto-merge double-added it) and 7 stale `q2_0` vec-dot aliases (multiple
  definition at link).
- Upstream's flint.cpp still ends at `res->t_logits = cur;` with no explicit
  expand — their framework reaches the output some other way (unresolved by
  static reading); our explicit call stays, harmless either way. Not reported
  upstream without rose's own words.
- Both trees clean-rebuilt (HIP had stale pre-merge objects — full `ninja -t
  clean` needed; CPU needed the same after a partial-link ABI crash in
  libllama-common). Sanity: flint loads + serves on both trees post-merge.

## Extreme rungs (rose's request: IQ1_S/IQ2_S-targeting K9, "most degraded but accessible")

`--k-body 5` / `--k-body 3` on the same unmodified pipeline (this required a
real fix: pass1/gate ignored --k-body and used the module constants — plan_k
now takes k_body/k_aux):

| rung | GGUF | b/param (486 tensors) | ppl (wiki2 300k) | notes |
|---|---|---|---|---|
| k5/e99 | flint_k5_k99_v2.gguf 9.13 GiB | 1.951 | **149,932 ± 6,550** | Cor 31.0%, ln ratio 8.394 — unmoored |
| k3/e99 | flint_k3_k99_v2.gguf 5.23 GiB | 1.109 | **2.58×10¹⁴** | the far edge |

- Gates: byte-exact at both grids (k5 and k3 pass digits/scales/q8_0-materialize
  exactly like k63; k99 embed/head exact).
- Runtime twin test: a stock materialized q8_0 of the k5 weights gives
  **identical** ppl (149,932.0528 exactly) — the cliff is the grid, not the
  loader. (Diagnostic twin was built without `--ref` first, which flattened
  rank-3 experts — materialize silently emits a 2-D frame without the ref's
  true shapes; --ref is mandatory for MoE archs.)
- Serving sanity: k5 loads ~20 s, generates degenerate text ("and the first,
  and the…"). Mechanically accessible, semantically destroyed.
- Half the Gaussian mass lands exactly on 0 with an RTN 5-level per-64 grid —
  the ~2 b/p regime needs calibration (imatrix/GPTQ/learned codebooks) that
  upstream's IQ types also require. Future rung: calibrated k9 grids.
- Published: `auRose94/Flint-40B-A2.5B-ninths-extreme-GGUF` (both GGUFs + card
  + templates). GLM-4.5-Air extreme rungs = deferred (rose: another task).

## Provenance / housekeeping

- ref kept at `/home/rose/flint_ref/Flint-BF16.gguf` (public, re-downloadable; delete any time)
- eval artifacts on `/mnt/Matrix/Work/base9-quantization/results/`: flint_k63_e99_v2.gguf,
  flint_iq4nl.gguf, flint_k63_as_q8_0.gguf, glm45air_ab_flint_*.jsonl
- ppl logs: /home/rose/flint_ref/ppl_*.log (copied verbatim into results/flint_ppl/)
- NOT published anywhere (HF/fork push) pending rose's go.