# base9-quantization

**Ninths-grid / repeating-radix model quantization.** Origin idea by Rosemary Mercury
(2026-10-03, kept verbatim in `docs/00-idea-origin.md`): quantize a model so
it "fits in the repeating numbers," using a base-9 digit system, as a
model-compression mechanism. Developed and tested with ZCode. MIT licensed
(`LICENSE`).

Status: **a measured format with a verified runtime.** The idea was first
tested synthetically and on TinyStories-33M (2026-10-03); it now has real
`.k9` containers, a llama.cpp fork that serves them natively (session 24 in
the log), and a Phase-2 Godot-4 coder fine-tune built on top of the format
(2026-10-04/05). Running log: `RESEARCH_LOG.md`.

One-line pitch: a repeating-radix weight grid that lets entropy coding slide
between the power-of-two quantizations to win quality-per-byte — proven
end-to-end from the number theory to a bit-exact llama.cpp runtime, and
applied to train a verified Godot-4 coder model.

## Why it's useful

Production quantization formats draw from a small menu of power-of-two
alphabets: ternary (1.58 bits/param), 4-bit, 8-bit. The ninths family
occupies the alphabet sizes *between* them (log2(9) ≈ 3.170 raw symbols,
entropy-coded), and the measured outcome of this repo is that it buys real
accuracy-per-byte there — against the deployed GGUF and bitsandbytes
formats, on the same eval windows, with paired noise floors:

- **Near-lossless tier.** At 1.5B, k63+embed99 is fp16-parity in ppl (13.90
  vs 13.88) in a **1.12 GB** file — the same size class as q4_k_m, which
  pays Δ +0.80 in the same harness (llama.cpp end-to-end, session 24). At
  7B and 14B, the same tier reaches **q8_0-parity perplexity in files
  roughly a third smaller** (7B: 5.47 vs 8.10 GB, Δ +0.021 ± 0.065; 14B:
  10.49 vs 15.70 GB, Δ +0.018 ± 0.052).
- **Small-file tier.** K9 points exist that are smaller *and* better than
  GGUF q2_k, q4_0 and bitsandbytes NF4 — e.g. 768.8 MB at Δcode +0.145 vs
  q4_k_m's 1117.3 MB at +0.170 (1.5B, session 12).
- **It is a working format, not a proposal.** Real files round-trip
  bit-exactly over 1.54B weights (0 digit mismatches); the entropy
  accounting behind the tables matched the written file to +0.00%; a C
  rANS coder runs at 278/210M symbols/s (46.8×/27.2× the Python
  prototype); encode is parallel (2.2×, byte-identical output) and decode
  streams tensor-by-tensor; and the llama.cpp fork decodes `.k9` on load —
  measured 109 t/s at 1.5B k63 (RTX 5060 Ti, CUDA) and 35.6 t/s at 14B k63
  (AMD 7900 XT, HIP).
- **The negatives are logged with the same rigor as the wins.** Asymmetric
  (GGUF-style) grids don't pay at matched bytes; GPTQ-style error
  compensation is a small-model effect that vanishes by 7B; repeating-
  decimal *notation* itself adds no compression; fitted codebooks and
  compensation don't stack. Anyone testing non-power-of-two alphabets can
  start from what already failed (sessions 7–8, 16, 20–21).

Scope, stated plainly: the scaling claims are Qwen2.5-Coder at 1.5B/7B/14B,
post-training quantization, on two consumer GPUs — a measured first pass
with error bars, not a survey of model families.

## Applications

1. **Near-lossless local serving on consumer hardware.** q8_0-grade quality
   in files roughly a third smaller moves 7B–14B models inside memory and
   disks they didn't fit before; the fork's `llama-server` already serves
   them on CUDA (RTX 50, sm_120) and HIP (RDNA3, gfx1100). Local, private,
   offline inference at near-full precision is the plain use case.
2. **Shipping fine-tuned models in the same container.** Phase 2
   (`phase2/`) builds verified SFT corpora (every sample passes the real
   Godot 4 parser) and trains a QLoRA coder judged by a mechanical,
   parser-based eval: 12.5% base → 50.0% tuned at 1.5B, **83.3% at 7B**,
   79.2% (draws-3) at 14B — and the tuned models re-quantize into K9
   containers at essentially the base model's cost (tuned 7B: 3.10 GB at
   the same task score).
3. **A reference for quantization research on non-power-of-two alphabets.**
   Every number in this README comes from a script in `experiments/`;
   predictions were pre-registered in `docs/01-hypothesis.md` before the
   experiments ran; null results are logged as verdicts. The reusable
   pieces are the eval discipline (paired windows, noise floors, side-info
   accounting) and the EC-SQ rate-distortion reasoning (exp15).
4. **A public, rebuildable artifact.** The repo ships ~1 MB of source, docs
   and text results; every multi-GB product is rebuildable from the table
   in *What is not in this repository* below.

## The idea in one minute

Two instincts turn out to be two views of one grid:

1. **Repeating numbers.** Every infinitely repeating decimal is a rational
   number, and vice versa. Quantize weights to multiples of one ninth —
   `{0, ±1/9, ±2/9, ±3/9, ±4/9} × scale` — and every weight becomes a
   *single repeating decimal digit*: 4/9 = 0.4444… in base 10 (because
   10 ≡ 1 mod 9) and 0.4 exactly in base 9. The whole model is then a string
   of base-9 digits.
2. **Base 9 = two ternary weights.** Modern ternary LLMs (BitNet b1.58,
   "1.58 bits") store weights in {-1, 0, +1}; two of them have exactly
   3 × 3 = 9 joint states — one base-9 digit. log2(9) = 3.1699 = 2 × log2(3),
   exactly. Base-9 is to balanced ternary what hexadecimal is to binary.

The compression question is then precise: does storing weights as base-9
digits (with entropy coding, since 9 symbols is not a whole number of bits)
beat standard integer-grid quantization in bits or accuracy-per-bit?

## First results (synthetic: 200k Gaussian weights, seed 42)

| scheme | levels | raw index bits | effective (rANS) bits | rel MSE |
|---|---|---|---|---|
| int8 symmetric | 255 | 8.0 | 7.44 | 0.000129 |
| int4 symmetric | 15 | 4.0 | 2.59 | 0.042466 |
| ternary (BitNet-style) | 3 | 2.0 | 1.58 | 0.264413 |
| **9-level ninths grid** | 9 | 4.0 | **1.83** | **0.129681** |
| Lloyd–Max 9 (MSE-optimal 9 levels) | 9 | 3.17 | ~2.98 | 0.028026 |
| uniform 16 (int4-style 16 levels) | 16 | 4.0 | 2.66 | 0.036883 |

Five findings from the first pass (details in `results/`, verdicts in
`RESEARCH_LOG.md`):

1. **The ninths identity is real** — machine-verified: k/9 = 0.kkk… (period
   1) in base 10, and terminates as 0.k in base 9, for k = 1..8; the
   base-9/base-10 tables in `results/exp2_repeating_representations.md` show
   the complementary structure (1/8 terminates in base 10 but repeats digit 1
   in base 9; 1/7 has period 6 in base 10 but only 3 in base 9).
2. **The ternary-pair → base-9 bridge is information-tight.** Two BitNet
   trits stored as one base-9 digit + rANS compress to 1.5836 bits/parameter
   vs the 1.585 alphabet bound — the "1.58 bits" claim survives actual
   lossless coding of the digit stream.
3. **The 9-level ninths grid earns a real spot on the rate-distortion
   curve**: rANS-coded to 1.83 bits/param with MSE 0.130 — it dominates
   raw-coded ternary (2.0 bits, MSE 0.264) and raw int4 (4.0 bits, MSE
   0.042) simultaneously, and fills the empty point between entropy-coded
   ternary (1.58 bits) and entropy-coded int4 (2.59 bits), both of which
   int4+rANS and ternary index-coding leave vacant.
4. **But "repeating numbers" do not add lossless compression by
   themselves** — the pre-registered null-result (P4) held: storing weights
   as minimal rationals (continued fractions, e.g. multiples of 1/9 ≈
   "one repeating digit") costs 6.4–22.8 bits/weight vs 1.58–2.6 for index
   coding, and repeating-decimal *notation* ("0.(4)") is 40 bits of ASCII for
   3.17 bits of content. The repeating structure is a representation
   identity, not extra compression.
5. **A fitted 9-level quantizer beats the uniform 16-level (4-bit) grid on
   both axes**: Lloyd–Max 9 levels reaches MSE 0.0280 at ~3.0–3.17 bits vs
   uniform-16's MSE 0.0369 at 4 raw bits. Alphabet sizes between powers of
   two have real accuracy-per-bit value when the levels are fitted to the
   weight distribution.

Honest caveats: Gaussian stand-in weights; MSE is a proxy, not perplexity;
int8's rANS figure here overstates codec cost (256-symbol frequency-table
granularity); see `docs/01-hypothesis.md` §6.

## Real-model results — TinyStories-33M (exp4 / exp4b / exp5, 2026-10-03)

Setup: every transformer-body weight matrix quantized per output row (exp4)
or per group of 64 input weights (exp4b); embeddings/wpe/lm_head/biases stay
fp32 (side info). Eval: 300 × 1024 streamed TinyStories-validation tokens.
**Effective bits = rANS-coded digits + fp32 side info** — row/group scales,
and for fitted (Lloyd–Max) grids the 9 center floats per row/group, which are
real decoder side info. Baseline perplexity **40.663**; eval noise ±~0.3.

| scheme | eff bits/param | ppl |
|---|---|---|
| int8 (per-row) | 8.02 | 40.63 |
| 27-level (per-row) | 3.97 | 42.93 |
| int4-15 (per-row) | 3.08 | 45.01 |
| 9-level Lloyd–Max (per-row, codebook counted) | 3.15 | 47.16 |
| 16-level (per-row) | 3.18 | 48.04 |
| 8-level (per-row) | 2.12 | 81.00 |
| 9-level ninths (per-row) | 2.30 | 93.33 |
| int4-15 (g64) | 3.98 | 42.33 |
| 9-level ninths (g64, uniform) | 3.20 | 54.68 |
| 8-level (g64) | 3.02 | 58.21 |
| int4-15 (g64) + GPTQ | 3.99 | 42.12 |
| 9-level ninths (g64, uniform) + GPTQ | 3.21 | **43.64** |
| 8-level (g64) + GPTQ | 3.03 | 46.84 |
| 9-level Lloyd–Max (g64, **fp16 codebook, measured**) | 5.02 | **40.62** |
| ternary absmean (PTQ — broken, see below) | 1.62 | 992.3 |

Findings:

1. **Group scales fix the MSE↔perplexity inversion.** With per-row scales
   the uniform 9-level grid *lost* to uniform 8 (93.3 vs 81.0 ppl) despite
   better MSE — outlier channels dominate rows. With g=64 groups the
   ordering heals: 9-level 54.7 ppl beats 8-level 58.2 (and MSE agrees).
2. **Fitted 9-level is the strongest accuracy-per-bit point** in the 3–8 bit
   region: per-row it dominates uniform-16 on both axes *even with codebook
   side info counted*; per-group with fp16 codebooks it reaches fp32-parity
   perplexity (40.5, within noise of 40.66) at ~5.0 b/param vs int8's 8.0.
3. **Codebook side info is real money** — 9 fp32 centers per row/group cost
   0.28–4.5 b/param. The first draft of exp4 under-counted this and made
   fitted-9 look free; all published numbers here include it (logged).
4. **Post-hoc ternary collapses (992 ppl)** while its ternary-pair→base-9
   codec stays information-tight (1.584 b/param). Ternary weights (and hence
   the base-9 pairing's payload) need QAT — RQ8 in docs/03 — post-training
   ternarization of a standard fp32 model is not competitive.
5. **The printed artifact is real** (exp5): 28.3M weights printed as
   repeating decimals — `0.(4)0.(3)0.(5)…`, each token the infinite
   repetition of one base-9 digit — re-parse **bit-exactly** to weights and
   reproduce perplexity identically (93.333 → 93.333). Size ladder for the
   linear weights: fp32 113.2 MB → printed text 142.2 MB (gzip 12.9 MB) →
   compact digits ~57 MB → rANS binary ~8 MB.
6. int4-15 stays the per-row champion at low bits. The ninths grid's honest
   role on a real model: a **viable odd-sized middle alphabet between
   3-bit and 4-bit that only pays off with entropy coding** — raw 4-bit
   indexing of 9 levels wastes ~0.7 bits/param.
7. **"More math on the GPU" pays, precisely on coarse grids** (exp6):
   GPTQ-style Hessian error compensation cuts the 9-level ninths grid from
   54.7 → **43.6 ppl at the same 3.2 bits** (−11.0) and 8-level from 58.2 →
   46.8 (−11.4), while int4 gains only −0.2. But it *hurts* ternary (992 →
   4027) — compensation assumes small errors; gross ternary errors amplify.
   Post-GPTQ frontier: LM9-g64 40.5 ppl @ ~5.0 b/p (fp16 codebook est.) ·
   int4-g64+GPTQ 42.1 @ 3.99 · **9-ninths-g64+GPTQ 43.6 @ 3.21 (low-bit
   champion)** · 8-level-g64+GPTQ 46.8 @ 3.03.
8. **Multi-digit repeating decimals work too** (exp7): the law generalizes
   to grids of b^L−1 levels — decimal 9/99/999, binary 7/15/63/127/255/1023
   (so int4-15 = binary 4-digit repeating grid; dequant
   w = m·((k/H)·0.b̄ − 1), affine in one repeating decimal). On the real
   model: the 99-level "0.(43)" grid is the best point of the 6-bit band
   (40.57 ppl @ 6.75 b/p, fp32 parity) and pays nothing for not being
   binary-aligned (P10 pass); monotonicity flags: 127-level inverts (40.98 @
   7.11) — noise-region MSE↔ppl divergence, under investigation.
9. **Open items closed (exp8/9/9b/10, session 5):** GPTQ+Lloyd do *not*
   stack — compensation hurts fitted codebooks (40.53 → 41.09); the fp16
   codebook is essentially free (+0.09 ppl) and **the measured accuracy king
   is LM9-g64 RTN with fp16 codebook: 40.62 ppl @ 5.02 b/param**, beating
   int8 (40.63 @ 8.02) on both axes. The 127-level inversion is real (all 3
   eval windows) but an endpoint-structure artifact of *uniform* fine grids —
   fitted codebooks shrink it to noise (+0.41 → +0.18) and fitted-99 even
   lands parity-or-better vs fp32 (40.46 @ 6.75). Embeddings: 99-level
   embeddings cost only +0.1 ppl over body-only → **the full model fits in
   45.5 MB (−83%) at 54.8 ppl** (27.7 MB at −90% for 61.0 ppl with 9-level
   embeddings; combined damage is super-additive).
10. **Where this leaves the curve:** 43.6 ppl @ 3.21 b/p (9-ninths+GPTQ,
    low-bit champion) · 40.6 ppl @ 5.02 b/p (fitted-9, fp16 codebook) ·
    40.5 ppl @ 6.75 b/p (fitted-99) — the ninths/decimal repeating family
    is competitive or leading at every bit depth it occupies.
11. **A NEW model was created natively on the grids** (exp11, three
    identical training arms — pretrained weights are not converged, so all
    comparisons are against a matched fp32 control, 4.591 ppl): **QAT-RECIPE
    (body on 9-level ninths + embeddings on 99-level): 4.89 ppl — a 6.4%
    loss for ~5.7× body compression**; **QAT-TERN (ternary body + 99-level
    embeddings): 5.98 ppl (+30%) at body 1.58 b/param — and its body digits
    code to 1.584 b/param in base-9 pairs, the BitNet bound, in a model that
    actually works** (post-hoc ternary was 989.8). All three pre-registered
    predictions (P17/P18/P19) PASS. Digit artifacts:
    `results/qat_digits_{control,recipe,tern}.npz`.
12. **RQ4 closed (null) + the deployable artifact** (session 7): the trained
    models' digit streams are informationally ~iid (lag-1 at iid expectations,
    order-1 Markov gain ≈ 0–0.2%, shuffled controls match) — rANS is
    effectively optimal, finishing the lossless story. The DEPLOYABLE PTQ
    point exists: **embed99 + GPTQ'd body9 = 44.96 ppl @ 45.5 MB total
    (−83% vs fp32's 274.1)**, saved as digits + scales
    (`results/full_model_digits.npz`); with a compensated body, fine
    embeddings cost +1.3 ppl (vs +0.1 with RTN body) — component errors
    compound. Longer-QAT (4000 steps): control reached 3.895 (from 4.591);
    the recipe/tern arms are paused pending GPU driver recovery (session's
    CUDA error 719 killed the device; `experiments/exp14b_arm_runner.py`
    resumes them, one arm per command).

13. **The lever is the grid, not the digits** (exp15, session 8): against an
    entropy-constrained scalar quantizer (EC-SQ, the true ceiling once rANS is
    allowed), the *uniform* ninths grid is only ~11% above optimal at its own
    entropy (1.81 b), and uniform+entropy is essentially rate-optimal for
    k ≥ 17 (eff 0.99–1.03). Fixed-rate Lloyd–Max is **not** the right ceiling —
    at matched entropy it sits 23% (Gaussian) / 80% (heavy tails) above the
    EC-SQ envelope. So the win available is *entropy-constrained codebook
    design* (a shared/parametric codebook approximates it), not the base-9
    digit identity. P25 PASS; P26–P28 FAIL, logged.
14. **The byte story survives an honest audit** (exp16, session 8): per-tensor
    rANS tables cost < 1 KB/model and reproduce the published digit rates
    within ±0.007 b/param; entropy-coded log scales cut the full exp13 model
    **45.5 → 42.1 MB (−7.4%)** with the digits untouched; the ternary body
    codes to 1.578 b/param. The RQ5 container spec is `docs/04-k9-codec-spec.md`.

15. **The grids hold up on a modern coder** (exp17, session 9): K9 PTQ of
    `Qwen2.5-Coder-1.5B-Instruct` (1.54B params, tied 151,936 embedding,
    RMSNorm/RoPE/SwiGLU/GQA) — the 9-level body at 2.67 b/param gives code
    ppl 4.477 vs the 8↔15 interpolation's 5.062 (**frontier-efficient**, the
    TinyStories result replicated); a 99-level embedding is essentially free
    (K9-full body9+embed99 = **638 MB vs fp32 6175 MB, −89.7%** at code ppl
    +20.5% / wiki +35.6%); int4-15 still wins raw quality at ~7% more bytes.
    First pass: 24.5k-token eval, entropy-based bytes, no external baselines yet.

16. **GPTQ works on the grids — once conditioned** (exp18, session 10): paired
    error bars (`eval_harness.py`) show GPTQ with **act-order + 128 calibration
    blocks + damping 0.05** cuts the 9-ninths code penalty from +1.107±0.127 to
    **+0.658±0.106 ppl** at +0.003 b/param (int4-15: +0.364→+0.149). The first
    port (exp6's recipe, 32 blocks, damp 1%) made GPTQ *worse* than RTN — an
    ill-conditioned-Hessian failure at 8960 dims. K9 full (body9-GPTQ +
    embed99) = **641.9 MB** at code ppl +0.657±0.106 / wiki +5.54; the 99-level
    embedding is free for the third time.

17. **K9 beats the smallest shipped format** (exp19, session 11): against GGUF
    q8_0/q5_k_m/q4_k_m/q4_0/q2_k and bitsandbytes NF4, measured on the same
    windows — K9-full (641.9 MB, code +0.657) is *smaller and better* than GGUF
    **q2_k** (752.9 MB, +1.426), and **1.74× smaller than q4_k_m** (1117.3 MB,
    +0.170). It is the smallest point on the deployed frontier, not the best
    quality-per-byte (NF4 999.5 MB/+0.270 and q4_k_m win there). q8_0
    reproduces fp32 within ~2%, validating the dequantization harness.

18. **K9 now dominates the most-used quant** (exp20, session 12): allocating the
    grid per tensor — body **k=15** (entropy-coded int4, the binary member of
    the same repeating-decimal family) + **k=99** embeddings + GPTQ — gives
    **768.8 MB at code ppl +0.145±0.037**, which is *smaller and better* than
    GGUF **q4_k_m** (1117.3 MB, +0.170) and NF4 (999.5 MB, +0.270), and ~10×
    lower Δ than q2_k at the same size (752.9 MB, +1.426). Attention layers are
    the sensitive type; the k∈{9,15} palette saturates at 768.8 MB, so a third
    level (k=27/63) is the next lever. (K9 bytes are entropy-estimated; the
    real file writer is the remaining end-to-end step.)

19. **K9's frontier now lies below the deployed formats'** (exp20 extended,
    session 13): adding body grids k=27/63 (3³ and 2⁶−1 members of the same
    family), body **k=63** + embed99 is **1114.5 MB at Δcode +0.012±0.004** —
    near-lossless, *smaller and ~14× lower Δ than q4_k_m* (1117.3 MB, +0.170),
    and 1.15×/1.70× smaller than q5_k_m/q8_0 while scoring better than both.
    At 809.4 MB (`attn63/mlp15`) K9 dominates q4_k_m, q4_0, NF4 and q2_k, and
    reaches parity with q5_k_m at 1.59× smaller. For every deployed format
    tested, a K9 point is smaller and at least as good. (Bytes still
    entropy-estimated; the real file writer is the last validation.)

20. **The K9 codec is real and round-trips a 1.5B model** (exp21, session 14):
    `experiments/k9.py` writes an actual `K9Q1` container; a 1112.0 MB file
    (5.763 b/param) for Qwen2.5-Coder-1.5B decodes back with **0 digit
    mismatches over 1.54 B weights** and identical perplexity (4.4995 →
    4.4995). The entropy accounting behind exp16–20 matched the real file to
    **+0.00%**, so those tables are measured, not estimated. Encode 5.66 /
    decode 3.74 M sym/s in pure Python (a vectorized coder is the remaining
    engineering). Weights differ by ≤1.4e-2 only because scales are stored in
    the lossy `ent8` mode; fp32 scale mode is bit-exact.

21. **The coder is fast now** (exp21/rans_fast, session 15): a C implementation of
    the *identical* byte-renorm rANS (`rans_fast.c`, ctypes) is **46.8× faster
    to encode / 27.2× to decode** than the pure-Python coder (278 / 210 M sym/s)
    and produces **byte-identical** output. The whole 1.54 B-weight model now
    compresses in **8 s** and decompresses + loads in **25 s** (was 273 s / 413 s).
    Diagnosis: the bottleneck was the CPython inner loop, not the algorithm — a
    single rANS stream is serial by construction, so lane interleaving (not
    faster scalar code) is what would buy the next 2-4× for SIMD/GPU.

22. **Asymmetric grids are a dead end — the symmetric one is right** (exp22,
    session 16): an affine per-group grid `w = lo + d·step` (GGUF Q4_K style)
    improves quality at fixed k but, at **matched bytes**, is parity at best and
    worse at low bits (k=9: 5.336 ppl on 721.3 MB vs **5.112** for the symmetric
    family at the same size). It pays 2 scales/group (0.347 vs 0.096 b/param) and
    a higher digit entropy. Tail clipping is also a non-issue (`symclip` ≈
    `sym`), because the odd grid always places a level exactly at 0 — where
    transformer weights concentrate. This closes the last accuracy-side
    enhancement; the exp18–21 numbers stand.

23. **K9 is fast and streams** (exp23, session 17): encoding is parallel across
    tensors — **2.2× at 24 threads** (8.3 s → 3.8 s for the 1.5 B model) with
    **byte-identical** output (same sha256) — and decoding streams tensor-by-tensor
    in row chunks, so the loader's device allocation is ~25 MB instead of a whole
    tensor. That removes exp21's transient CUDA OOM. Both are implementation-only:
    no format change. The remaining ~2-4× would need lane-interleaved rANS for
    SIMD/GPU, which matters at 7B+ but not for this workflow.

24. **7B: K9 stays below the deployed frontier, but no longer dominates it**
    (exp24/exp25, sessions 18–19). `Qwen2.5-Coder-7B-Instruct` (7.62 B params,
    untied lm_head). Real K9 files: **k=15 + embed99 = 3740.2 MB, code ppl
    +0.117 ± 0.015** vs bf16; **k=9 + embed99 = 3103.7 MB, +0.374 ± 0.043**.
    Measured GGUF baselines on the same windows: q2_k 3015.9 MB/+0.561,
    q4_k_m 4683.1 MB/+0.070. Interpolating the GGUF pair gives +0.348 (at
    3740 MB) and +0.535 (at 3104 MB), so K9 is **0.23 / 0.16 ppl better than the
    deployed frontier at matched bytes** — but it is 20% smaller than q4_k_m
    while 0.047 ppl worse, i.e. a trade rather than the 1.5B-style domination.
    This run is RTN; adding the GPTQ that bought −8% at 1.5B is the obvious next
    lever. (Baseline is bf16, not fp32 — 7B fp32 does not fit a 16 GB card.)

25. **GPTQ does not transfer to 7B — a clean null** (exp26, session 20): GPTQ with
    act-order over **71% of the body parameters** (in ≤ 4096; `down_proj` stayed
    RTN because its 18944² Hessian doesn't fit the card) moved 7B k=15 from
    **+0.117 → +0.115 ppl** at +0.5% bytes — i.e. nothing, inside the ±0.020
    noise. The reason is that the penalty is already small at 7B: k=15 costs
    +8.1% relative at 1.5B but only +3.3% at 7B, leaving compensation little to
    recover. So the 7B result stays a **size-vs-quality trade** (20% smaller than
    q4_k_m, 0.045 ppl worse) rather than the 1.5B-style domination; K9 remains
    ~0.23 ppl below the GGUF frontier at matched bytes.

26. **GPTQ is a null on the coarse grid too — the lever is scale-limited**
    (exp26 k=9 probe, session 21). Running the same 71%-of-body GPTQ at k=9:
    3103.7 MB/+0.374 (RTN) → **3129.5 MB/+0.333 ± 0.034** (GPTQ), a shift of
    −0.041 ppl ≈ 1.2 SE — inside noise, though in the favourable direction.
    At 1.5B the same lever recovered 41% of the k=9 gap (−0.449 of +1.107);
    at 7B it recovers ~11% (−0.041 of +0.374). So the 1.5B "compensation pays
    on coarse grids" result is a **small-model** effect: as the model scales,
    both the RTN penalty and the compensatable share of it shrink. k=9 + GPTQ
    (3129.5 MB/+0.333) does sit **0.195 ppl below the interpolated GGUF frontier
    (+0.528) at matched bytes**, and beats q2_k's quality by 0.228 ppl — but at
    +3.8% bytes, so the 1.5B "dominates q2_k on both axes" win (exp19) does not
    survive to 7B on size. K9's byte-efficiency advantage holds at 7B; only its
    ability to beat q4_k_m on quality does not.

27. **K9 runs inside llama.cpp** (session 24): a fork (branch `k9`, its commit
    `14490cf2d` sitting directly on upstream `c25030496`) added a
    decode-on-load GGUF container — a loader-only sentinel type
    `GGML_TYPE_K9` = 43 plus one `k9.directory` KV carrying rank/shape/k/
    group/scale metadata — and materializes K9 tensors on load into q8_0
    layouts. That materialization is **lossless**: K9's dequant
    `w = m·(d−H)/H` is a scaled integer, so Q8_0's per-32-block scale can be
    set to `m/H` exactly — the only loss is the fp16 rounding of that scale
    (≤ 4.9e-4 relative, finer than the k99 grid spacing). Gates are green:
    digit streams bit-exact vs `k9.py`, 0-ulp scales, byte-exact
    materialized q8_0 — verified on 1.5B k63, tuned 7B k15, and
    GPTQ-act-order k15 (perm tensors take the F16 path in the loader). The
    serving numbers in *Why it's useful* are `llama-server` on the exported
    `.k9`-in-GGUF files; the verified ppl matrix:

    | model | variant | file GB | ppl (wikitext-2 test) |
    |---|---|---|---|
    | 1.5B base | f16 | 3.09 | 13.88 ± 0.110 |
    | 1.5B base | **K9 k63 + embed99** | **1.119** | **13.90 ± 0.110** |
    | 1.5B base | q4_k_m | 1.117 | 14.70 ± 0.118 |
    | 7B tuned | q8_0 | 8.10 | 9.231 ± 0.064 |
    | 7B tuned | **K9 k63 + embed99** | **5.47** | **9.253 ± 0.065** |
    | 14B tuned | q8_0 | 15.70 | 7.805 ± 0.052 |
    | 14B tuned | **K9 k63 + embed99** | **10.49** | **7.823 ± 0.052** |

    (1.5B rows compare our K9 file against upstream GGUF quants of the base
    model; the 7B/14B rows are tuned merges quantized with `llama-quantize`,
    so both members of each comparison came through the same pipeline.)
    Stage 2 is planned: resident container types (K9_4/6/7) with fused MMVQ
    matmul kernels, then a PR — the upstreamable artifact is the container
    type, not the rANS storage layer.

28. **The format earned its first real user** (sessions 22–23, `phase2/`):
    the verified-corpus Godot-4 coder fine-tune ladder — 12.5% base →
    50.0% tuned at 1.5B → **83.3% tuned at 7B** → 79.2% draws-3 at 14B
    (which needed a chunked-CE fix to train at all and an lr recipe change
    to 3e-5) — every score judged by the real Godot 4 parser, never by eye.
    The tuned models re-quantize into K9 at essentially the base model's
    cost (tuned 7B k9+embed99 = **3.10 GB**, task score unchanged), so the
    fine-tuned, verified coder ships in the same container the research
    produces. An execution-verified C++ corpus/eval extension (build, run,
    stdout judging) is the current front line.

## Requirements (what applying this research needs)

- **Hardware:** this experiment ran on one RTX 5060 Ti (~2 min/eval sweep);
  CPU-only works for ≤ 100M-param models (minutes per scheme). Models ≥ 1B
  want more VRAM/time.
- **Disk/network:** ~1.5 GB total (python stack + model + eval text +
  artifacts); one-time network for pip and the HF download (cached under
  `.hf-cache/`).
- **Software:** Python ≥ 3.11 with numpy, torch, transformers, datasets —
  this machine has torch 2.14.0 (CUDA), transformers 5.15.0, datasets 5.0.1,
  numpy 2.5.3 on Python 3.14.7; exact versions recorded in
  `results/versions.txt`.
- **Method fine print (from exp4/exp4b):** per-row or per-group fp32 scales
  are unavoidable side info (0.03–0.5 b/param); fitted codebooks add
  0.28–4.5 b/param and must be counted; embeddings/lm_head left fp32 here
  (on 33M-class models that's over half the bytes — quantizing them is
  orthogonal follow-up); perplexity windows of ~300k tokens carry ±0.3 ppl
  noise — schemes inside that band are "parity", not "wins".

## Repository map

    docs/00-idea-origin.md          the idea, verbatim, and how it unfolded
    docs/01-hypothesis.md           formalization, number theory, pre-registered predictions
    docs/02-related-work.md         web-verified literature (incl. the novelty-gap check)
    docs/03-open-questions.md       roadmap RQ1–RQ8
    docs/04-k9-codec-spec.md        the K9 codec container spec (RQ5)
    experiments/                    self-asserted experiment scripts (see its README)
    results/                        generated CSV + MD outputs of every experiment
    phase2/                         GDScript/Godot corpus + verifier + coder fine-tunes (see its README)
    chat_k9.py                      local chat UI for the K9-quantized models (bf16 A/B)
    RESEARCH_LOG.md                 dated log: predictions, verdicts, surprises

The serving runtime is a separate fork: llama.cpp, branch `k9` (one commit
over upstream `c25030496`); findings 27–28 above summarize it.

## Running

Python 3 + numpy. From `experiments/`:

    python3 exp1_entropy_bits.py
    python3 exp2_repeating_representations.py
    python3 exp3_quantization_error.py
    python3 exp15_alphabet_sweep.py          # ~2 min: EC-SQ ceiling (numpy only)
    python3 exp16_codec_accounting.py        # ~3 min: re-costs stored artifacts (numpy only)

exp15 is numpy-only. exp16 needs the artifacts in `results/`
(`full_model_digits.npz`, `qat_digits_{recipe,tern}.npz`) and is numpy-only;
both write into `results/`.

### Chatting with a K9 model

`chat_k9.py` at the repo root loads one base model and streams K9 files into its
weights on demand, so you can talk to the quantized model and A/B it against bf16:

    python3 chat_k9.py                       # 1.5B: bf16 baseline vs k63/embed99
    python3 chat_k9.py --preset 7b           # 7B: bf16, k9, k15, k9+GPTQ, k15+GPTQ
    python3 chat_k9.py --preset 7b --no-serve   # terminal REPL instead of the web UI

It serves a local web chat with a variant picker (stdlib `http.server`, no
gradio/flask, no network) and prints the URL. Two memory notes: switching between
two K9 variants needs no restore because each file covers *all* decoder tensors,
and switching back to bf16 restores from a CPU cache of the pristine weights
(1.5B ~3 GB, 7B ~15 GB RAM). The 7B needs ~15.3 GB VRAM, so it cannot run
alongside the 1.5B on a 16 GB card.

Use it as a *quality* instrument, not just a demo: paste a reply into
`phase2/gdscript_verify.py --file x.gd` and the real Godot 4 parser tells you
whether the model invented an API. It caught one immediately — asked for a
one-line FPS print, the 1.5B replied `print(GD.get_fps())`, which is **C#-only**
and fails to parse (`Identifier "GD" not declared`); the correct call is
`Engine.get_frames_per_second()`.

## What is not in this repository

Everything here is **source** — code, docs, and small text/CSV results (about
1 MB in total). The multi-GB products are deliberately excluded and can all be
rebuilt:

| excluded | size | rebuild with |
|---|---|---|
| `*.k9` model containers | 1.1–3.8 GB each | `experiments/exp21_k9_codec.py`, `exp24_qwen7b.py` |
| `*.npz` digit/scale dumps | 0.5–1.1 GB | the exp11/exp13 runners |
| `.hf-cache/` model weights | 31 GB | downloaded on first run |
| `.venv-baselines/` | 144 MB | `python -m venv .venv-baselines && pip install torch transformers bitsandbytes peft` |
| `phase2/data/` scraped corpus | 644 MB | `phase2/build_corpus.py` (downloads it) |
| `phase2/out/` SFT data, adapters, merged models | ~4 GB | `phase2/build_sft.py`, `train_qlora.py`, `merge_adapter.py` |

See `.gitignore`. One trap worth knowing: git only treats `#` as a comment at the
**start** of a line, so an inline `# comment` after a pattern becomes part of the
pattern and silently matches nothing — which is how you end up with a 17 GB
`git add`.

## Data provenance

The Phase 2 corpus is **built, not redistributed**. `phase2/build_corpus.py`
downloads `wallstoneai/godot-gdscript-dataset` (**Apache-2.0**) and extracts the
Godot-4 half; the derived JSONL, the QLoRA adapters and the merged models stay
out of git. `phase2/README.md` documents the filtering, and the derived data is a
transformation of Apache-2.0 source, so the licence permits it either way.

## License

MIT — see `LICENSE`.