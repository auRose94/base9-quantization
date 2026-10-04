# experiments

Self-contained Python 3 + numpy scripts (stdlib otherwise). Each is seeded
(42), asserts its own invariants (every rANS stream is roundtrip-verified;
every reconstructed expansion matches `fractions.Fraction` exactly), and
writes CSV + Markdown output into `../results/`.

    python3 exp1_entropy_bits.py                # ~5 s  (numpy only)
    python3 exp2_repeating_representations.py   # ~30 s (numpy only)
    python3 exp3_quantization_error.py          # ~5 s  (numpy only)
    python3 exp4_real_model_ptq.py              # ~10 min: needs torch + transformers + datasets + HF network
    python3 exp5_repeating_digits_roundtrip.py  # ~3 min, same deps as exp4

exp1–3 and `rans.py` are pure numpy (stdlib otherwise). exp4/exp5 additionally
need torch (CPU suffices; CUDA used when present), transformers, and datasets;
they download the model and eval text on first run (cached under `../.hf-cache/`).
`rans.py` is the canonical shared coder (exp1 predates it and carries a local twin).

## What each measures

- **exp1_entropy_bits.py** — bits-per-parameter table for int8, int4,
  BitNet-style ternary, and the 9-level ninths grid: raw index bits, alphabet
  bits, empirical entropy, and *actual* rANS coder output (frequency tables
  shared as metadata, excluded). Includes the ternary-pair stream coded as
  base-9 digits. Tests predictions P1, P2, P5 of `../docs/01-hypothesis.md`.
- **exp2_repeating_representations.py** — number theory: periodic-expansion
  ⇔ rational theorem verified by exact reconstruction on 400 (fraction,
  base) pairs for bases 9 and 10; ninths identity k/9 = 0.k̄ and the
  base-9/base-10 duality table; multiplicative orders; sqrt(2) irrational
  negative control; and the P4 test — continued-fraction (minimal rational)
  encoding vs digit-index coding, plus notation-vs-compression arithmetic.
- **exp3_quantization_error.py** — relative MSE vs alphabet size for uniform
  grids (3..32 levels) with Lloyd–Max (MSE-optimal) rows for 3/4/8/9/16
  levels. Tests prediction P3 and locates the 9-level grid on the
  rate-distortion curve.
- **exp4_real_model_ptq.py** — RQ2: post-training quantization of
  TinyStories-33M (per output row; embeddings/lm_head stay fp32) across
  int8/int4/16/8/ternary/9-uniform-ninths/9-Lloyd–Max/27, with perplexity on
  300×1024 streamed validation tokens and entropy + actual-rANS effective
  bits per scheme. Includes the ternary-pairs → base-9 coder row.
- **exp5_repeating_digits_roundtrip.py** — the printed artifact: quantize to
  ninths, print every weight as its repeating decimal `0.(s)`, re-parse the
  file from disk, and assert bit-exact reconstruction + identical perplexity
  after a fp32-restore/reload cycle. Produces `results/printed_model.txt`,
  its gzip, and a human-readable sample; reports the size ladder (fp32 →
  printed → gzip → compact digits → rANS → alphabet floor).
- **exp4b_group_scales.py** — exp4 with per-group scales (g=64, GGUF-style):
  int4/8/9-uniform/9-Lloyd–Max; shows group scaling heals exp4's uniform-9 vs
  uniform-8 inversion, and honestly counts Lloyd–Max's per-group codebook
  side info (fp32 + fp16 estimates).
- **exp6_gptq_grids.py** — "more math on the GPU": GPTQ-style Hessian
  error compensation (Frantar et al., arXiv:2210.17323) applied to the
  exp4b grids (g=64 int4/8/9-ninths/ternary). Calibration = 64 TRAIN-split
  blocks; per-layer Hessians 2·XᵀX; column-wise compensated rounding with
  static exp4b-identical grids/scales. Comparing ppl vs exp4b isolates the
  compensation effect. Note: the Hessian chain runs on CPU float64 because
  cusolver on this Blackwell GPU (torch 2.14) errors in cusolverDnXpotrs.
- **exp7_multidigit.py** — multi-digit repeating grids (the b^L−1 law):
  decimal 9/99/999 vs binary 7/15/63/127/255/1023 level grids at g=64 with
  wide-alphabet rANS (M=4096), GPTQ on 9/15/99, and the L=2 print→parse
  round trip; pre-registered predictions P6–P10 in its docstring.
- **exp11_qat.py** — RQ8, the "new model" experiment: QAT (weight-swap STE)
  trains new checkpoints natively on the grids — QAT-RECIPE (body on the
  9-level ninths grid + embeddings on 99-level, learnable-free) and QAT-TERN
  (ternary body with learnable per-row gamma). Compares every run against
  fresh same-window PTQ baselines (gain) and fp32 (loss); pre-registered
  P17–P19. Saves digit artifacts (`qat_digits_{recipe,tern}.npz`) including
  the ternary stream's base-9 pair-codec cost.
- **exp12_structure_scan.py** — RQ4, CLOSED: autocorrelation + order-1
  Markov scan of exp11's trained digit artifacts (with shuffled controls);
  verdict: streams are ~iid, rANS is effectively optimal, no exploitable
  structure (P24 pass).
- **exp13_full_model.py** — the deployable full-model PTQ point:
  99-level embeddings + GPTQ'd 9-ninths body, full side-info byte table,
  reloadable digit artifact (`full_model_digits.npz` with digits + scales).
- **exp14_longer_qat.py / exp14b_arm_runner.py** — 4000-step QAT (3 identical
  arms); exp14b runs one arm per fresh CUDA process (resilience after a
  CUDA error 719 driver fault) and finalizes verdicts P20–P22.
- **exp15_alphabet_sweep.py** — RQ3's open half: k=3..32 sweep of uniform,
  Lloyd–Max, and the entropy-constrained scalar quantizer (EC-SQ, min D s.t.
  H≤R) on Gaussian + Student-t(4). λ=0 reproduces exp3's Lloyd–Max exactly.
  Shows uniform ninths is within 11% of EC-SQ at its own entropy and that
  uniform+entropy is asymptotically optimal for k≥17 — i.e. the lever is
  entropy-constrained grid design, not the base-9 identity (P25 PASS,
  P26/P27/P28 FAIL; pre-registered in the docstring).
- **exp16_codec_accounting.py** — re-costs the stored exp13/exp11 artifacts
  with per-tensor rANS (tables counted) and realistic scale coding
  (fp32/fp16/int8/int6/entropy-coded log). Confirms the published digit rates
  survive per-tensor tables and shows fp32→ent-log8 scales cut the full exp13
  model 45.5 → 42.1 MB (digits unchanged). No model/GPU needed.
- **exp17_qwen_k9_ptq.py** — the scaling move: K9 PTQ on
  `Qwen/Qwen2.5-Coder-1.5B-Instruct` (modern arch, tied 151k embedding) vs
  int4-15/8/99-level grids, with WikiText-103 + Python ppl and full-model byte
  accounting. Needs torch + transformers + datasets + HF download (~3 GB) and a
  GPU; ~4 min on an RTX 5060 Ti. Result: the 9-level grid is frontier-efficient
  on a modern coder, a 99-level embedding is ~free, K9-full is 638 MB vs fp32
  6175 MB. `--smoke` for a fast 2-scheme run.
- **exp18_qwen_gptq_noise.py** — GPTQ on the grids at 1.5B with a real noise
  floor. Multi-window eval (via `eval_harness.py`), paired deltas vs fp32,
  Hessians collected once on the fp32 model, act-order + configurable damping
  (env: `K9_NCAL`, `K9_DAMP`, `K9_ACT`). `--gptq-only` for fast iteration.
  Result (~12 min on a 5060 Ti): GPTQ with act-order + 128 cal blocks + damp
  0.05 improves 9-ninths code ppl 5.599 → 5.150 and int4-15 4.855 → 4.641;
  K9 full (body9-GPTQ + embed99) = 641.9 MB. exp18 v1 (32 cal blocks, damp 1%,
  no act-order) made GPTQ *worse* than RTN — an ill-conditioned-Hessian failure
  logged in RESEARCH_LOG (session 10).
- **exp19_external_baselines.py** — Phase 1b: GGUF k-quants (q8_0/q5_k_m/
  q4_k_m/q4_0/q2_k, official Qwen GGUF) and bitsandbytes NF4 evaluated on the
  same windows by dequantizing into the fp32 HF model; bytes are authoritative
  file/packed sizes. Needs `gguf` (system) and `bitsandbytes` (in
  `.venv-baselines`); run it with `../.venv-baselines/bin/python`. `--only q8_0`
  is a validation mode (must match fp32). Result: K9-full (641.9 MB, +0.657)
  is smaller and better than q2_k (752.9 MB, +1.426) and 1.74× smaller than
  q4_k_m (1117.3 MB, +0.170).
- **exp20_k_palette.py** — mixed-precision grid allocation (k per tensor) via
  greedy promotion on Hessian-diagonal-weighted error per coded byte. Body
  GPTQ'd (cached as uint8 digits + scales, rebuilt on demand), embeddings RTN;
  same 8-window harness. Env: `K9_KBODY` (default 9,15), `K9_KEMBED`, `K9_BUDGETS`.
  `--smoke` for a fast two-palette run. Result (~20 min with k=9,15,27,63):
  **body k=63 + embed99 = 1114.5 MB at +0.012 code ppl — smaller and better than
  q4_k_m (1117.3 MB, +0.170), q5_k_m and q8_0; and 809.4 MB (attn63/mlp15)
  dominates q4_k_m/q4_0/NF4/q2_k.** Early assertions guard the digit+scale
  rebuild (act-order permutation) and the embed cache; embeddings are processed
  before the long body pass so storage bugs fail fast.
- **exp26_qwen7b_gptq.py** — 7B + GPTQ. Applies compensation (act-order, damp
  0.05, 64×1024 in-domain calibration, Hessians per transformer block with
  upstream layers already quantized) to every body matrix with in ≤ 4096 — 71% of
  body params — leaving `down_proj` RTN because its 18944² Hessian (1.4 GB fp32)
  does not fit beside a 15.3 GB model. `gptq_apply` row-blocks the column loop so
  the device working set stays ~4096×c floats, and is unit-tested bit-identical to
  exp18's implementation. Result: **a null** — k=15 +0.117 → +0.115 ppl at +0.5%
  bytes; the k=15 penalty is already small at 7B (+3.3% relative vs +8.1% at
  1.5B), so there is little for compensation to recover. `K9_CONFIGS="9,99"`
  reruns the probe on the coarse grid (result files get a `_k9` suffix) — also a
  null: k=9 +0.374 → **+0.333 ± 0.034** at +0.8% bytes, i.e. −0.041 ppl ≈ 1.2 SE.
  GPTQ recovers 41% of the k=9 gap at 1.5B but only 11% at 7B.
- **exp24_qwen7b.py / exp25_gguf7b.py** — the 7B scale test on
  `Qwen/Qwen2.5-Coder-7B-Instruct` (7.62 B, untied lm_head). exp24 quantizes the
  body at k=15/k=9 + embed/lm_head at k=99, writes real K9 files, and evaluates
  against a **bf16** baseline (7B fp32 does not fit 16 GB); exp25 dequantizes the
  official GGUF q4_k_m/q2_k tensor-by-tensor into the same bf16 model on the same
  windows, with paired deltas. Result: K9 **3740.2 MB / +0.117** and **3103.7 MB /
  +0.374** vs GGUF q4_k_m 4683.1 MB / +0.070 and q2_k 3015.9 MB / +0.561 — K9 is
  0.23/0.16 ppl better than the GGUF frontier at matched bytes, but trades size
  for quality against q4_k_m. Both use chunked, in-place quantisation and chunked
  restore so the 15.3 GB model fits the 15.57 GiB card.
- **exp23_k9_perf.py** — performance: parallel encode (`write_k9(threads=)`,
  2.2× at 24 threads, byte-identical output) and streaming decode (`K9File` +
  `load_into` with row-chunked dequant, ~25 MB device allocation, no OOM).
  Result (~1 min): encode 8.3 → 3.8 s, decode+load 22.5 s, 0 digit mismatches,
  ppl unchanged. No format change.
- **exp22_asym_grid.py** — asymmetric (min+scale / affine) grids vs the
  symmetric odd grid, plus a tail-clipped symmetric variant, at k=9/15/27 on
  Qwen2.5-Coder-1.5B. Result (negative, ~2 min): affine grids improve quality at
  fixed k but cost 2 scales/group + higher digit entropy, so at **matched bytes**
  they are parity (k=15) or worse (k=9: 5.336 @721 MB vs 5.112 for symmetric at
  the same size). Tail clipping is a non-issue. K9's symmetric grid is confirmed
  as the right choice.
- **rans_fast.c / rans_fast.py** — the same byte-renorm rANS as `rans.py`,
  compiled (`gcc -O3 -shared -fPIC` → `k9rans.so`) and driven via ctypes.
  **Byte-identical** output to the reference, **46.8× encode / 27.2× decode**
  faster (278 / 210 M sym/s vs 5.95 / 7.72). `k9.py` uses it when available and
  falls back to Python if no compiler is present. Note the `.so` is deliberately
  not named `rans_fast.so` — that would shadow this module on import.
- **k9.py** — reference implementation of the K9 container (docs/04): per-tensor
  rANS digits (M=4096) + serialized frequency tables + scales (fp32/fp16/ent8) +
  optional act-order permutation. `write_k9` / `read_k9` / `decode_tensor` /
  `decode_digits`. Reuses the validated rANS in `rans.py`.
- **exp21_k9_codec.py** — the file writer, end to end. `--selftest` proves
  bit-exact round trips on all grids × scale modes × the permutation path in
  seconds; the full run writes a real file for Qwen2.5-Coder-1.5B (body k=63 +
  embed99) and decodes it back. Result: **1112.0 MB, 0 digit mismatches over
  1.54 B weights, identical ppl, entropy-estimate delta +0.00%**, encode
  5.66 / decode 3.74 M sym/s. ~12 min.
- **eval_harness.py** — shared multi-window perplexity harness: N disjoint
  windows, mean ± SE, and `paired_delta()` for scheme-vs-scheme comparison on
  identical windows (cancels the window effect). Supersedes single-window
  point estimates.
- **rans.py** — shared rANS coder + entropy helper (see module docstring).
  Bench (28.3M symbols, ~2.5-bit entropy): 10.1 enc / 9.1 dec M-sym/s pure
  Python — 6 s round trip for the whole 33M-class model; ~23 min/scheme
  projected at 7B → that is the scale where vectorized/GPU coding (nvCOMP
  ANS, interleaved-state rANS) becomes mandatory. General-M support added in
  session 4: renorm threshold f·2^32/M (byte-identical to the M=256 original,
  regression-tested against exp1's local twin) and the fixed decoder mask.

## Adding a scheme

Drop a new row into `main()` of exp1: define its dequantization `deq` and
its integer symbol stream, then report `rel_mse`, `entropy_bits`, and
`rans_bits`. Keep the roundtrip assert — it is the coder's contract.

## Known codec note

exp1's rANS uses a 256-symbol frequency table; with 255-symbol alphabets
(int8 row) that granularity costs ~10% over entropy — visible in the table.
Fine for the 2/3/9/15/16-symbol alphabets that matter here; if you extend to
wide alphabets, raise the scale_bits.