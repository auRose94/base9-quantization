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
- **5.3-Flash k63/e99 v3 container BUILT + GATED (2026-11-07)**:
  results/glm53flash_k63_e99_v3pages.gguf = 209.86 GiB, 601 K9 tensors
  (301 paged) + 811 copies; pass1 = 225.12 GB blobs @ 5.616 b/p (1h57m,
  320.66 B params quantized, 98M copied, 40G MemoryMax cap); pass2 368s.
  Gates: fork k9_load_meta parses (v3 paged file, 301 registered),
  `k9probe ALL pages` **PASS, 0 failures** (per-page byte-equal incl.
  76-lane token_embd/output @ 674 MB each). Serving smoke fails EXACTLY
  at the designed boundary — k9_buf needs 332,806,701,056 B (in-place
  Q8_0) > the A9's 128 GB → M2.5 real-budget arena is the next rung.
- **pass2 bf16 gotcha (cost one container rewrite):** the unsloth BF16
  refs store norms/biases/router as BF16; gguf-py's typed writer rejects
  bf16, and an f32-widening workaround produced a writer offset
  inconsistency (C reader: recorded offset > cumulative declared, +1.5 MB
  at the first wide hc_*_fn tensor). Correct fix:
  raw_dtype=GGMLQuantizationType.BF16 + raw bit bytes — lossless, half
  the size, GGML_TYPE_BF16 read natively. Committed a31a2e7.
- **M2.5 real-budget arena LANDED (2026-11-07)**, with two findings that
  correct the plan:
  1. **Dense-model constrained budgets are semantically impossible at
     tensor granularity.** All fills happen pre-execution, so a tensor
     evicted during its own graph's ensure is read (post-fill) at
     execution with no re-fill in between — unless it is re-decoded at a
     LATER node of the same pass. Execution-safe victims are therefore
     `last_use > i` ONLY, and in a dense graph every materialized tensor
     has `last_use < i` → nothing is ever evictable → the arena must be ≥
     the working set (= the full model for dense). The v1 "60-eviction
     soak" is hereby re-interpreted: those evictions were re-decode churn
     over stale-correct bytes (in-place), never real frees — gate 3's
     constrained-soak semantics were weaker than assumed. **Gate 3's
     constrained leg RE-SCOPES to M3 (MoE: selected experts per token →
     working set < full).** The loader now enforces this with a clear
     "below this graph's working set" error instead of corrupting.
  2. **Eviction must free the slot BEFORE poisoning the pointer** — the
     reordered version (poison first, offset read after) computed every
     free offset off a nulled pointer and handed out the poison slab as a
     slot. Symptom: victim slot read as `(2^64 − base)`, decode SEGV in
     pack_row. Cost the bring-up hour; the comment in evict_until warns.
  - What M2.5 ships: budget-sized arena (K9_BUDGET_GIB; 0 = full + slack),
    overlapping dummy homes at a zeroed poison slab (scheduler-invariant
    preserved), derived residency (resident ⇔ `cur->data != poison` — no
    bool that can go stale), contiguity-aware eviction (largest span, not
    byte sums — crumbs can sum past a request while no span fits it),
    first-fit + adjacency-merged free ranges, self-heal on invalidated
    state. Gates: eager == resident(full) token-identical (48 + sustained
    192 tok); peak RSS identical to eager at full budget (arena replaces
    k9_buf 1:1 — no regression, no win yet); constrained budgets fail
    LOUDLY with the working-set explanation. **The actual memory relief
    for 5.3-Flash on the A9 = M3: expert-granular pages hooked at the MoE
    gather (per-token working set ≈9.5 GiB at Q8_0 → 60-90 GiB budget
    viable). M2.5 is the substrate M3 reuses.**
- **M3 LANDED (2026-11-07): expert-granular paging hooked at the MoE
  gather — the actual memory relief, kernel-measured.**
  - Design: ONE `MAP_NORESERVE` region backs every K9 tensor at a fixed
    offset (anonymous zero pages = free poison; no carving, no free lists).
    Pages = the container's v3 lanes. Non-MoE tensors fill wholly at first
    graph use (ensure_graph); **MoE expert tensors are hook-managed: the
    CPU `mul_mat_id` op calls `k9_res_moe_ensure` at EXECUTION time, when
    the router ids are actually known** — thread 0 decodes the selected
    experts' lanes, all threads barrier, then the stock kernel runs.
    Eviction = `madvise(MADV_DONTNEED)`: region pages drop to zero on LRU
    eviction, and each decoded lane's digit-stream range is dropped from
    the blob mapping (re-faults from the file on the next decode).
  - **BUG LEDGER (each cost real time; do not re-learn):**
    1. The hook MUST be thread-0-guarded — `mul_mat_id` executes per
       thread; unguarded concurrent calls race the registry (symptom:
       interleaved eviction counters 1,3,4,2 and "random" bit wipes).
       Uniform decision on all threads (cheap managed-check) + decode on
       thread 0 + `ggml_barrier`.
    2. Per-page budget checks let one op's decode EAT ITS OWN TAIL (each
       new lane evicts the previous ones; the kernel reads zero pages).
       Fix = RESERVATION semantics: collect the whole call's lane set,
       evict only non-set pages until the set fits, then decode.
    3. Exceptions must not cross ggml-cpu.c's C frames ("terminate called
       recursively") — the extern-C hook catches and aborts loudly.
    4. `madvise` REQUIRES page-aligned addresses — raw 64-aligned offsets
       return EINVAL silently and nothing is ever dropped. Absolute-
       address alignment helper: inward rounding for region pages
       (neighbor tensors share 4K pages — a rounded-out drop zeroes a
       live neighbor), outward for file-backed blob pages (safe to
       over-drop; re-fault from the file).
    5. The load-time `MADV_WILLNEED` whole-file prefetch sets peak RSS to
       the whole GGUF before residency does anything — gated off in
       resident mode (blob pages re-fault per decode; resident mode never
       writes into the mapping, so no COW pages exist to lose).
    6. `RUSAGE_CHILDREN.ru_maxrss` is the MAX over all waited children —
       sequential measurements in one parent pollute each other. Sample
       `/proc/PID/status` VmHWM instead.
  - Gates (OLMoE-1B-7B k63/e99 v3 container, 4.61 GiB, 48 hook-managed
    expert tensors, 64 experts each):
    - eager == resident(full) == resident(1.5 GiB budget): **TOKEN-
      IDENTICAL**; sustained 128-token run TOKEN-IDENTICAL with 91
      real evictions.
    - **Peak VmHWM: eager 12021 MiB vs constrained 7971 MiB — Δ4.05 GiB
      (34%) of kernel-measured memory relief with identical output.**
  - 5.3-Flash serving math (same machinery): region 310 GiB virtual
    NORESERVE; non-expert ≈0.5 GiB decoded + expert budget 60-90 GiB →
    peak ≈8-10 GiB/token working set + budget → **fits the A9's 128 GB;
    serve via the HIP unified-memory build next.**
  - Files: src/k9-residency.{h,cpp} (region + pages + hook impl),
    ggml/include/k9-moe-hook.h (C hook API), ggml/src/ggml-cpu/ggml-cpu.c
    (mul_mat_id hook), loader resident branch (region homing), llama-mmap
    (prefetch gate).

+- **M3-HIP hunt continued (2026-11-08, fork 740c2bd28): the residual
+  bug is INDEPENDENT OF WEIGHT DELIVERY — it is the attention output
+  being ~zero in resident-GPU mode.**
+  - Killed hypotheses (each with a measured test): weight-delivery
+    mechanism (direct GPU reads, per-layer sched copies, CPU-backend pin,
+    MoE-cache banks — ppl 66,533.6544 bit-stable across ALL of them);
+    HIP graph capture (gate off → same value); FA kernel (-fa off →
+    66,586, equally broken); fusion (decoration-tolerant gate → same);
+    D2H copy races (HIP cpy_tensor_async refuses CPU dst → sync path);
+    region bytes, device-side weight bytes, window-1 router ids, embd —
+    all verified identical to golden.
+  - **Decode-mode file-dump taps (deterministic, batch-1): resident
+    ffn_inp-0 (residual after attention) |max| = 0.05 ≈ the embedding
+    scale alone, vs eager 1.08 — the attention contributes ~nothing.
+    embd verified identical (|max| 0.0477 both). FA-off equally broken →
+    the fault sits in the Q/K/V → KV-write → attention chain upstream of
+    the FA kernel, only when residency is active.** CPU-vs-GPU numerics
+    make cross-build hash comparison meaningless — the working method is
+    HIP-eager vs HIP-resident file dumps at decode (tiny tensors, no
+    buffer-recycling ambiguity), NOT prefill FNVs (4 MB intermediate
+    nodes read post-split hit RECYCLED compute-buffer slots — the hash
+    taps on mid-split nodes are unusable at prefill).
+  - Harness: K9_TAP (comma-separated filters) dumps raw f32 files to
+    /tmp/k9tap2 per tap; append-mode after the first 64 taps. Dispatch
+    counters (CPU + GPU mul_mat_id) print at calls 1-5/1000. Region
+    tensors are user-pinned to the CPU backend every execution (defeats
+    ggml pass-4 "4.cur" handing unassigned weight leafs to the GPU).
+  - NEXT: (1) tap the KV-cache write path at decode — check whether the
+    K/V rows actually land in the GPU KV (the attn_inp_k_idxs/v_idxs
+    inputs and the SET_ROWS-with-indices path are candidates); (2) check
+    whether Q (copied CPU→GPU) reads back correctly inside the attention
+    split; (3) if both are clean, diff the mask input. The qwen3.5 dense
+    reference is green because its hybrid (mamba-dominant) graph has a
+    different split skeleton — OLMoE is the minimal pure-attention
+    reproducer on this container.

+- **M3-HIP surgical taps (2026-11-08 late, fork 9d95e9365): the mask
+  fill is ELIMINATED — and a tap-methodology lesson landed.**
+  - Verified IDENTICAL eager vs resident: the KQ mask fill
+    (set_input_kq_mask, same ROCm_Host buffer, same sentinel values,
+    correct post-fill bytes), inp_pos/k_idxs/v_idxs fills, and the embd.
+  - **Tap-methodology lessons (do not re-learn): (a) named graph inputs
+    are LEAFS — split views carry no leafs; tap them at
+    llama_context::graph_compute on the full graph via the new public
+    ggml_graph_n_leafs/ggml_graph_leaf accessors; (b) naive
+    ggml_backend_tensor_get on ROCm_Host-buffered tensors passes a host
+    pointer as a device source and returns GARBAGE — the "unfilled
+    mask" reading was this artifact, caught by instrumenting the fill
+    itself; (c) prefill mid-split hash taps read recycled compute
+    buffers; (d) the /tmp tmpfiles cleaner races dump files — dumps now
+    live in probes/k9tap.
+  - Standing signal: resident-GPU attention output ≈ zero at decode
+    (ffn_inp |max| 0.05 = embd-only scale vs eager 1.08), invariant to
+    every weight-delivery mechanism. Remaining suspects: the Q/K/V copy
+    content inside the attention split at decode (backend-aware reads
+    required — ROCm_Host and ROCm0 buffers need different read paths),
+    the KV-write SET_ROWS-with-indices landing, and the mask/Q as seen
+    by the FA kernel itself.

+- **M3-HIP taps round 3 (2026-11-08, fork 40c6297e7): KV-index fill
+  eliminated; the capture-decision map drawn; ONE suspect standing.**
+  - Verified IDENTICAL eager vs resident: attn_inp_k_idxs/v_idxs fills
+    (idx 0 1 -1 -1 both). Mask fill previously verified. FA outputs
+    (K9_FA_PROBE v2, per-execution in/out) identical at sampled points.
+  - **K9_GATE_TRACE (per-graph capture decisions): ZERO captured
+    mul_mat_id graphs — the CPU-pin + gate work as designed; the MoE ops
+    dispatch on the CPU backend with hooks firing every window (an
+    earlier "only 5 GPU dispatches" reading was a counter-print artifact
+    — prints gate at <=5/1000th calls).**
+  - The HIP-captured graphs = the small attention/KV splits (4/10/12/16/19
+    nodes, mmid=n), REPLAYED every execution with no host dispatch (FA
+    probe silent after exec 1). FA outputs at sampled points identical.
+  - **TOP SUSPECT (next session): stale copy-slot sets in replayed
+    attention splits — GGML_SCHED_MAX_COPIES=4 rotates
+    sched->cur_copy; a replayed graph reads capture-time
+    input_cpy addresses while per-execution copies may land in the
+    rotating slot set → stale mask/Q/K/V/idx reads in replayed
+    attention. Verify by checking how compute_splits picks
+    tensor_id_copy(src, backend, cur_copy) for replayed splits vs what
+    the capture recorded; fix = force n_copies=1 for resident runs or
+    re-record on rotation.** All other stages verified-correct: embd,
+    mask fill+copy, idx fills, region bytes, device weight bytes,
+    window-1 router ids, CPU mul_mat_id+hook dispatch, 0 mmid captures.

+- **M3-HIP taps round 4 (fork 9a105808a): the FIT-PASS-CAPTURE theory.**
+  - Methodology corrections that re-opened the case: the GPU dispatch
+    counter prints at <=5/1000th calls — "5 prints" NEVER meant "5
+    dispatches" (the mul_mat_id dispatch + hooks run every window);
+    per-token ppl shows token 1 = 72,467 already broken while window-1
+    router ids are golden — broken before the MoE; FA probe silent after
+    exec 1 = the attention splits REPLAY without host dispatch.
+  - **TOP THEORY: the FIT PASS (LOAD_MODE_NONE) executes each graph key
+    FIRST and HIP-captures it with unfilled inputs; real windows replay
+    those captures.** Attention/KV splits replay (benign IF every
+    captured input re-refreshes per window — mask/Q/K/V copies do; rope
+    freqs / K-V rows captured mid-fill are the unverified remainder).
+    The MoE splits dispatch (gate ✓ hooks ✓). Router-ids golden + token-1
+    broken + topology-invariant ppl all fit: the divergence is inside
+    the first-capture/replay boundary, before the MoE.
+  - ELIMINATED (final list): mask fill, K/V idx fills, region bytes,
+    device weight bytes, window-1 ids, weight-delivery mechanism,
+    FA kernel itself, graph capture of MoE splits, fusion.
+  - NEXT (one session): (1) disable the fit pass (or force capture AFTER
+    real fill) and re-gate — the single decisive test; (2) if green,
+    keep the fit-pass guard permanently (e.g. skip capture until the
+    first REAL execution); (3) re-run the full gate matrix + 5.3-Flash.

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
  the metric. Recommend training mix + a separate hold-out ppl slice.+- **M3-HIP session 2026-11-08: three real fixes landed, the residual bug
+  localized to a precise cell.**
+  - FIX 1 (load crash, HIP-only): the upstream iGPU rule (#28160,
+    llama-model.cpp) dropped `use_mmap` when any device reports
+    `mmap_support=false` (the 8060S does) → the first K9 tensor disabled
+    residency mid-load → the next tensor hit `ggml_backend_tensor_alloc`
+    on its dummy buffer → abort. Exempt K9 stream models from the rule
+    (file mmap is independent of the device host-ptr capability the rule
+    targets; the from_ptr buffer path keeps its own
+    `buffer_from_host_ptr_supported` gate), and `k9_prepare_residency`
+    now disables up-front when mmap is off (no region, no mid-load
+    flip-flop).
+  - FIX 2 (prefill garbage): `sched_copy_experts`'s FALLBACK leg (the
+    expert upload when the MoE-cache slot-map path is disabled — cache
+    off, budget too small, or prefill batches above max_batch 32, i.e.
+    EVERY real prefill) uploaded paged bytes straight from the region
+    without decoding. Now decodes the used experts first.
+  - FIX 3 (generic-copy corruption): when the split starts at another op
+    (layer's first matmul), the sched copies the WHOLE weight tensor
+    generically — in resident mode that swept undecoded (zero) region
+    pages into the device buffer. The callback now whole-decodes managed
+    tensors before returning false (idempotent; budget-safe).
+  - FIX 4 (hardening): HIP graph capture disabled for any mul_mat_id
+    graph while residency is active (`k9_residency_is_active()` gate) —
+    replay skips host-side decode hooks; the name-based gate alone missed
+    it because the sched's input copies hide the K9 identity.
+  - **REMAINING BUG (measured, not yet fixed): resident × MoE × GPU =
+    ppl 66,533.6544 (bit-stable across every topology permutation:
+    ±op-offload env, ±graphs, ±FA, ±moe-cache, direct-read
+    pre-assignment, capture gate).** Green: eager at any ngl; resident at
+    -ngl 0 (pure CPU); the DENSE paged container (qwen 1.5B) is
+    bit-perfect at -ngl 99 resident (5.1814 all three modes). Verified
+    IDENTICAL to golden at real windows: region bytes (fill + page
+    hashes), all weight bytes on the device at kernel time (dense +
+    MoE sampled read-backs dev==reg), router ids at window 1 (21 18 46
+    35 55 28 6 41 both builds), embedding activations. DIVERGES: layer-0
+    `ffn_moe_out` at window 1 (CPU golden 9882e5b4 vs HIP 8b5f6fa8).
+    Earlier "ids/activation divergences" were warmup-vs-real-window
+    comparison artifacts. CAUTION: several debug samplers crash inside
+    live HIP graph capture (capture is now gated off under residency, so
+    they are usable again), and debug_expert_fnv had an offset bug
+    (fixed). NEXT: per-layer output taps through layer 0's attention
+    chain (attn_out → residual → ffn_norm) with capture known-off, to
+    find the first divergent activation; the tap harness is in
+    ggml-backend.cpp compute_splits (K9_TAP=name-substring env).
+  - Gate: probes/gate_m3hip_a9.sh (eager vs resident ×cache ×budget on
+    the OLMoE v3pages container; the script's extra-args must be passed
+    UNQUOTED — a quoted "--flag value" string aborts the run silently
+    and produces vacuous PASSes).
