# ninths — calibration track + stream-paging v3 plan

Status: **DRAFT (2026-10-29)**. Two coordinated work items: (A) calibrated
low-bit rungs so coarse grids stop destroying perplexity; (B) v3 paging so
tensors can be resident on demand and the runtime can iterate faster. Tracks
are deliberately ordered A-first — B improves load time and memory, not
quality, and A is the blocker for "stable low-k".

## 0. Why this document exists

Measured reality on flint (README, results/hf_flint_extreme): at 1.95 b/param
RTN the model hits 149,932 ppl; at 1.11 it collapses (2.6×10¹⁴). The
container passes byte-identical gates at every level — the cliff is the grid,
not the codec. PPL is the user-visible priority; load time is tolerable. The
side project that was holding VRAM should finish shortly, freeing the card.

## 1. Track A — calibrated low-k rungs

### 1.1 Goal
Produce k5 (≈2 b/p) and k3 (≈1.1 b/p) rungs whose wiki2 ppl is within an
order of magnitude of k63 (23.68 ± 0.54), not 4 orders above. Stretch:
approach the stock IQ2_S/IQ1_S ladder that already builds with imatrix.

### 1.2 Root cause (from the docs)
- Per-64-group symmetric RTN, `s = fp16(amax/H)` with fp16 floor 1e-4; code
  = rounded clamped digit. Half of a Gaussian weight lands on 0 at 5 levels.
- K9's symmetric odd grid is confirmed superior to affine min+scale (exp22
  negative), so we keep symmetric. Asymmetry is not the lever.
- GPTQ-style reconstruction error did not transfer at 7B on EITHER grid
  (exp18: +0.333 ± 0.034 ≈ noise). Do not re-litigate GPTQ for low-k.
- What's missing: per-WEIGHT importance, not per-tensor. Upstream's IQ quants
  refuse to build without an importance matrix for this range for a reason.

### 1.3 Proposed experiment (smallest useful first)
1. Imatrix pass: add `--imatrix <path>` to `glm_k9_build.py`'s quantize
   step. Use llama.cpp's `llama-imatrix` on the flint corpus to produce an
   importance matrix. **First experiment (2026-10-29) tried a weighted amax
   (importance·|w|) as the per-64-group scale and FAILED catastrophically:**
   calibrated k5 on qwen1.5b hit ~821,000 ppl vs 94 for uncalibrated k5
   (baseline uncalibrated k5 already terrible vs f16 ~7-10; calibrated was
   ~10,000× worse). Cause: weighting the amax by sqrt(in_sum²/count)
   inflated scales instead of preserving high-importance weights, because
   the dequantization multiplier is the stored scale — importance changed
   which channels dominate the scale rather than concentrating precision.
   - Where: `experiments/glm_k9_build.py` + helper
     `experiments/k9_imatrix.py`.
   - Scope: body tensors only; k99 embeddings/head and norms untouched.
   - **Correction needed:** scale weights by importance before quantization and
     divide the stored scale back by importance is the analytic form; instead
     of storing inv-importance, run an importance-weighted **MSE scale search**
     over a log-spaced candidate set and let that search choose the stored m.
     - Experiment 2 (2026-10-29): implemented as `rtn_frame_mse` in
       glm_k9_build.py, selected with `--cal-mode weighted-mse --imatrix`.
       On qwen1.5b k5: uncalibrated PPL 94.06 → calibrated 24.50 (3.8×
       better), and weighted MSE per tensor dropped ~1.8×. Promising.
       Integrated into glm_k9_build pass1 (16 candidate scales, 64-row
       chunks); full 1.5B k5 build runs 59s.
- Remaining for Track A: run the same build on the real target (flint body)
  and report against the harder eval battery before considering publish.
2. Per-tensor k allocation from sensitivity: run the mini-KLD probe (exp27
   harness) per tensor on a 1.5B model first (cheap), build a sensitivity
   table (Δppl per tensor at k5 vs k9), allocate k per tensor under a b/p
   budget rather than uniform k5. Reuses the allocator already described in
   the K9 pattern memory.
3. Eval upgrade before any gate claim: extend the 24-task battery (it is
   saturated at 79.2%). Report per-task, not just aggregate.
4. Multi-draw verification: draws-3 minimum before ranking rungs (the
   measured note says single-draw k9==bf16 was draw luck).

### 1.4 Acceptance gates for track A
- k5/e99 on flint: wiki2 ppl ≤ 100 (vs current 149,932); battery ≥ 60%.
- k3: ppl ≤ 1000 (currently 2.6×10¹⁴) — aspirational; if still broken, the
  honest finding is "sub-2-bit needs learned codebooks, not grids."
- Full byte-exact gate chain unchanged (digits/scales/materialized weights = 0
  mismatch) — calibration must not compromise container correctness.
- HF card updated with measured honestly-mixed numbers.

## 2. Track B — v3 stream paging (perf/residency)

### 2.1 Goal
Let the runtime keep only the tensors it is actively using resident, decoded
on demand and evicted when done. Targets MoE expert residency and faster
iteration. NOT a quality change.

### 2.2 Concept
- A **page** is a contiguous, independently decodable frame: one
  (tensor, row-range, lane) slice, sized 4–16 MiB, decodable to the same
  Q8_0 bytes as today's v2 path (because v2 lanes are row-aligned and
  self-contained).
- A **page table** maps `(tensor, row_offset) → page id`; pages carry a
  content-addressed digest for cross-model sharing.
- A **resident cache** (LRU, bounded by VRAM budget) with prefetch for the
  predicted next expert/layer.
- Materialize the page to Q8_0 on decode and hand the runtime the same buffer
  shape as full-tensor decode. Byte-exactness is preserved because paging is a
  residency/orchestration layer on top of v2's decode.

### 2.3 Interfaces that must NOT change
- rANS core (rans_fast.c / ninths-codec): state domain, renorm, entropy
  accounting.
- v2 lane layout and `ninths.directory2` record format — v3 adds a NEW KV
  (`ninths.pages` or similar) alongside directory2, same no-corrupt-old-reader
  discipline as doc 11 §3.5.
- Materialize-to-Q8_0 theorem: a fully resident paged model must be
  byte-identical to today's v2 materialized GGUF, tensor for tensor, row
  for row.
- Loader and KoboldCpp port: both must learn the page-table parser.

### 2.4 Gates for track B
1. Page-level round-trip: encode pages, decode pages, sha256 = original.
2. Full-model equivalence: fully resident paged model byte-equal to v2
   baseline (existing gate chain).
3. Constrained VRAM (budget below full residency): sustained soak; verify no
   eviction corruption, no deadlock, output token-identical to the fully
   resident path.
4. KoboldCpp port updated or it loudly refuses v3 files (document in its
   readiness checklist, do not half-support).

### 2.5 Explicit non-goals for v3
- Not a new container type. Paging is residency.
- Not a quality change — track A owns that.
- Not on-the-fly weight-streaming matmul (decompress during GEMM). That is
  the doc-09 RQ10 "runtime as digits" bet, separate, much larger; do not
  smuggle it into v3.

### 2.6 Shipped container design + gate status (2026-11-06)

**Design decision: a page IS a v2 lane.** No new digit layout, no scale
paging. The v2 lane-interleaved digit area is already a table of
independently decodable row-aligned rANS streams sharing one freq table
(`encode_stream` in glm_k9_build.py; fork k9.cpp:547+ decodes lane-by-lane),
so v3 = (1) export with lanes chosen for page size, (2) a small marker KV,
(3) loader lazy decode. ent8 scales stay whole-tensor and always resident —
they are ~1/100th of the digit bytes (5.3-Flash: ≈0.17 GB vs 117 GB), which
sidesteps the ent8-does-not-split finding from the P1 prototype.

- Exporter: `glm_k9_build.py pass1 --page-mib N` sets lanes ≈
  ceil(uncompressed_digit_bytes / N MiB), cap 255 (u8 field + fork validator;
  big expert tensors hit the cap → ~2-4 MiB pages there — fine). Default 0 =
  legacy docs/11 rule. `pass2 --pages` writes the NEW `ninths.pages` KV:
  u16 ver + u16 count, per paged tensor {u16 name_len, name, u8 lanes,
  u32 page_rows}. directory2 untouched → old readers unaffected (docs/11
  §3.5 discipline).
- Gates run (1.5B blk.0, 5 tensors):
  - P1 prototype row-range page round-trip (experiments/k9_pages.py): PASS.
  - `gate-pages` (per-lane decode == fresh RTN + KV/directory2 agreement):
    PASS.
  - Materialize equivalence: v2-built (lane_plan rule) vs v3-built
    (--page-mib 8) containers materialize byte-identical Q8_0 on all 5
    tensors: PASS.
  - Fork primitive (`llama-k9probe <gguf> ALL pages`): per-page
    k9_decode_page materialization byte-equal to whole-tensor k9_dequantize
    on all paged tensors, poison-fill 0xA5 catches unwritten bytes: PASS
    (also full-model on the 1.5B v3 container: 141 K9 tensors, incl. the
    28-lane token_embd, ALL PAGE-EQUAL).
  - Refactor drift check: C `weights` output == python materialize bytes
    (q8_0) after hoisting pack_row into the shared k9_pack_ctx/k9_pack_row
    helper: BYTE-EQUAL.
  - **Gate 3 soak (2026-11-06): PASS.** qwen1.5b full v3 container,
    llama-simple greedy `-t 1`: eager == resident(full) ==
    resident(0.6 GiB virtual budget, 21 evictions) token-identical at
    48 tokens; sustained 192-token run TOKEN-IDENTICAL with **60
    evictions/re-decodes**. The registry, LRU and re-decode machinery is
    correctness-proven under thrash.
- Residency runtime LANDED (2026-11-06, `src/k9-residency.{h,cpp}`):
  - Env: `K9_RESIDENT=1` + `K9_BUDGET_GIB=f` (0/unset = full). Loader
    registers K9 tensors (name, mmap blob, meta, f_ne, dst) instead of
    decoding; `llama_context::graph_compute` calls
    `k9_residency::ensure_graph(gf)` before the scheduler — weights are
    graph LEAFS (walk node srcs, depth 2), matched BY NAME, entries rebind
    to the graph's object; misses decode into the tensor's k9_buf home
    (in-place) with LRU eviction (victim = materialized && last_use != i).
  - **HARD-WON GOTCHAS (do not re-learn):** (1) K9 tensors have NO buffer at
    load — the scheduler would home them into its COMPUTE buffer and
    lazily-filled bytes would not be what the ops read; the resident branch
    must still `ggml_backend_tensor_alloc` them into k9_buf, skipping only
    the decode+set. (2) The loader releases unused mmap fragments after
    load via mmaps_used/lmlocks — the resident branch MUST register the
    blob range there or blobs die before first use. (3) Eviction policy
    must evict `last_use != i` (both future tensors, re-filled later this
    pass, and passed tensors, re-filled next token); `<i`-only or `>i`-only
    both deadlock or corrupt. (4) K9_BUDGET_GIB is VIRTUAL in v1: k9_buf is
    full-size, so memory is not yet freed — real relief = M3 expert paging
    or a residency buffer type.
- Gate 4 (KoboldCpp): **VERIFIED LIVE (2026-11-06)** — the vendored loader
  (koboldcpp_default, rebuilt) refuses v3 files at model load:
  "this KoboldCpp build does not support v3 paged K9 containers
  (ninths.pages KV present) — see docs/13 gate 4". Its k9_load_meta also
  only parses legacy `k9.directory` (v2 `ninths.directory2` containers
  silently skip the K9 path — separate latent gap for the port, flagged in
  the readiness doc).
- Fork facts: the materializer ALREADY supports the full weight target menu
  — Q8_0 / F16 / IQ4_NL (k9.cpp; iq4_nl = deterministic amax/113 RTN,
  permuted + k99 aux tensors route to Q8_0), switchable at runtime via the
  K9_MATERIALIZE env override or ninths.materialize KV. Lane validator
  raised 64 → 255 (u8 field). k9_decode_page() + k9_parse_pages() landed;
  llama-model-loader parses ninths.pages into k9_pages. KV targets already
  exist (stock cache types + k9_4kv/k9_6kv digit KV, HIP gate green).
- GLM-5.3-Flash target facts: HF tag `glm5_next` (fork src/models/
  glm5-next.cpp came with the upstream sync), **≈325B total params MoE**
  (correction 2026-11-06: FP8 repo 305.8 GiB = 1 B/param; unsloth BF16
  597.6 GiB / 2 = 325B — the earlier 163B was a 2× misread), 288 routed +
  1 shared experts, 8 active, 45 layers (3 dense), KDA linear-attn hybrid +
  sparse indexer + 1 MTP layer. Sizing: k63/e99 container ≈234 GiB, Q8_0
  materialized ≈325 GiB, IQ4_NL ≈183 GiB → **nothing fully materializes
  into the A9's 128 GB**; serving this model on the box REQUIRES real
  memory-relief paging (M2.5 arena re-pointing / M3 expert-granular, or
  REAP-pruning composition — stock only fits Q2 ≈110 GiB). Ref = unsloth
  BF16 GGUF (upstream has NO glm5_next converter; official repo is FP8).

## 3. Sequencing
1. Track A experiment 1.3.1 (imatrix) on 1.5B first; gate on ppl improvement.
2. If ≥2×, scale to the GLM body and re-run the battery.
3. Track B spec → prototype on the freed machine; gate on full residency
   equivalence before enabling paging.
4. V3 lands as a (tiny) perf update once Track A published numbers are
   stable.

## 4. Open questions
- Publish calibrated k5 as its own HF repo, or re-upload the flint k63 repo
 (storage-quota caution from new-machine notes).
- imatrix corpus: wiki2 (ppl target) vs the training mix. Risk = overfitting
  the metric. Recommend training mix + a separate hold-out ppl slice.