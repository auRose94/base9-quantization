# 04 — K9 codec specification (draft v0.1)

Status: **implemented and validated**, 2026-10-03 (v0.1 spec; reference
implementation in `../experiments/k9.py`, end-to-end validation in
`../experiments/exp21_k9_codec.py`). Grounded in exp1/4/7/12/13/15/16. This is
the RQ5 deliverable: a standalone compressor/decompressor for weights stored on
the odd `b^L−1` grids (9/15/27/63/99/255/…), entropy-coded with rANS.

Validation (exp21, Qwen2.5-Coder-1.5B, body k=63 + embed99, ent8 scales): a real
**1112.0 MB** file (5.763 b/param) decodes back with **0 digit mismatches over
1,543,714,304 weights** and identical perplexity (4.4995 → 4.4995); the entropy
accounting used in exp16–20 matched the real file to **+0.00%**. Encode 5.66 /
decode 3.74 M sym/s (pure-Python rANS — see §13 for the speed work).

Scope note (what the experiments already settled):

- The digit streams of real trained models are informationally iid
  (exp12), so **order-0 rANS is optimal** — no context modelling.
- **Per-tensor rANS tables are cheap** (≈ 2 bytes/symbol; < 1 KB per model at
  k=9) and reproduce the published global-table bit rates to within
  ±0.007 b/param (exp16). So the format uses per-tensor tables and counts them.
- **Entropy coding is mandatory**: raw 4-bit indexing of 9 levels wastes
  1.3–2.2 b/param (digit entropy is 1.81 on Gaussian, 2.70 on the real
  TinyStories-33M body). A fixed-size block format (GGUF-style) cannot carry
  this; hence a variable-rate container. A fixed-rate fallback is specified
  in §9 for GGUF compatibility only.
- **Scale coding is the free win** (exp16): fp32 → entropy-coded 8-bit log
  scales takes the body scale tax from 0.50 to ~0.125 b/param and the full
  exp13 model from 45.5 to ~42 MB with zero accuracy change.
- Grid *placement* is near-optimal for the uniform family, but an
  entropy-constrained fit buys 14–25% MSE on Gaussian and ~80% on heavy tails
  (exp15); v1 ships the uniform grid, v2 adds a fitted codebook mode (§8).

---

## 1. Data model

A K9-compressed tensor is a tuple

    (k, g, m[], d[], pair_mode, codec)

- **k** — number of grid levels; must be odd, `k = 2H+1`, `H = (k−1)/2`.
  Canonical members: `k = 3` (ternary), `9` (ninths / 2 trits), `15` (int4),
  `99` (decimal L=2), `255` (int8). Any odd `k` is legal.
- **g** — weights per scale (group size). `g ∈ {32, 64, 128, 256}`; default 64.
  Groups run along the **input** dimension (last dim of the `(out, in)`
  canonical layout; see §7 for the Conv1D caveat).
- **m[·]** — one scale per `(row, group)`, shared by `g` weights.
- **d[·]** — the digit stream, values in `{0 … k−1}`.
- **pair_mode** — 0: one digit per weight; 1: two ternary digits packed into
  one base-9 symbol (k must be 3; §6).
- **codec** — 0: rANS (order-0); 1: raw digits (fallback / debugging).

### Dequantization (normative)

    w = m * (d − H) / H                    # H = (k−1)/2, per (row, group)

For `k = 2H+1 = b^L − 1` the digit is an L-digit repeating block in base `b`
(`d/k = 0.\overline{d}`), and the same formula is the affine identity
`w = m·((k/H)·0.\bar{d} − 1)` verified in exp7. The codec only ever evaluates
the left-hand form; the identity is documentation.

Worked values: `k=9, H=4` → `w = m(d−4)/4` (exp5's printed equation);
`k=99, H=49` → `w = m(d−49)/49`; `k=3, H=1` → `w = m(d−1)` ∈ {−m, 0, m}.

Encoding is the exact inverse: `d = clip(round(w·H/m) + H, 0, k−1)`.

---

## 2. rANS parameters (normative)

The coder is Duda's rANS, byte renorm, matching `experiments/rans.py`:

| parameter | value |
|---|---|
| state width | 32 bits |
| state domain | `[2^24, 2^32)` |
| renorm | byte-wise (8-bit), renormalize **before** each encode step |
| condition | push while `x ≥ f·2^32/M` |
| frequency-table size | `M = 2^scale_bits`, `scale_bits = 12` (M = 4096) |
| symbol order | encode in reverse, decode forward |
| stream terminator | final 4 bytes = end state, big-endian |

`M = 4096` resolves alphabets up to 4096 symbols; it reduces byte-identically
to the original `M = 256` coder when `scale_bits = 8` (regression-tested in
exp1/exp4). A decoder MUST reject `scale_bits < 8`.

**Frequency table serialization.** Normalize counts to `f[0..k−1]` summing to
`M` with `f[s] ≥ 1` for every present symbol (the `normalize_freqs` rule), then
store `f` as `k` little-endian `uint16` (max value `M = 4096` fits). Table size
= `2k` bytes. Cumulative table and the `M`-entry inverse table are rebuildable
from `f` and are not stored.

---

## 3. Quantization pipeline (encoder)

    for each tensor W with canonical shape (R, C):
        require C % g == 0
        Wr = W.reshape(R, C//g, g)
        m  = max(|Wr|, axis=2)                      # float32, (R, C/g)
        step = 2*m/(k−1)
        d  = clip(round((Wr + m)/step), 0, k−1)     # uint8 (k ≤ 255) or uint16
        stream_digits = interleave_by_tensor_order(d)
        bytes_digits = rans_encode(stream_digits, f, scale_bits=12)
        bytes_table  = f.tobytes()                  # k×uint16 LE
        bytes_scales = encode_scales(m, scale_mode) # §4

`m` is written as a tensor of shape `(R, C/g)` in row-major order.

---

## 4. Scale coding (normative options)

`scale_mode` per tensor:

| mode | id | layout | cost at g=64 |
|---|---|---|---|
| fp32 | 0 | `R·(C/g)` × float32 LE | 0.50 b/param |
| fp16 | 1 | float16 LE | 0.25 b/param |
| int8 | 2 | per-tensor `scale_max` (fp32) + `uint8` codes (m/max) | 0.125 b/param |
| int6 | 3 | per-tensor max + 6-bit codes packed | ~0.094 b/param |
| ent-log8 | 4 | per-tensor `lo,hi` (2×fp32) + rANS(uint8 of `log2(m)` quantized over `[lo,hi]`, M=4096) | **0.125 b/param measured** (exp16) |

Default for v1: **mode 4 (ent-log8)**. It measured 0.125 b/param on the exp13
body (442,368 scales → 0.352 MB), vs 0.50 fp32, and is the reason exp13's
45.5 MB becomes ~42.1 MB with no change to the digits.

---

## 5. Container layout (normative)

Little-endian throughout. Standalone file (not GGUF).

### 5.1 Header

| offset | size | field | value |
|---|---|---|---|
| 0 | 4 | magic | ASCII `"K9Q1"` |
| 4 | 2 | version | `1` |
| 6 | 2 | header_flags | bit0: has_fitted_codebooks (§8) |
| 8 | 4 | n_tensors | |
| 12 | 4 | scale_bits | `12` |
| 16 | 8 | tensor_dir_off | absolute offset of the directory |
| 24 | 8 | data_off | absolute offset of the stream area |

### 5.2 Tensor directory

`n_tensors` records, each:

| size | field | notes |
|---|---|---|
| 2 | name_len | |
| name_len | name | UTF-8, e.g. `transformer.h.0.attn.q_proj.weight` |
| 1 | rank | |
| 4·rank | dims[] | canonical `(out, in, …)`; K9 quantizes 2-D only |
| 1 | grid_k | odd, ≥ 3 |
| 1 | group | 32/64/128/256 |
| 1 | codec | 0 rANS, 1 raw |
| 1 | scale_mode | §4 |
| 1 | pair_mode | 0 single-digit, 1 ternary-pair |
| 2 | freq_table_bytes | `2k`, or 0 for codec=1 |
| 4 | digits_bytes | |
| 4 | scales_bytes | |
| 8 | digits_off | absolute |
| 8 | scales_off | absolute |

The frequency table (`freq_table_bytes`) is stored immediately **before** the
digit stream (at `digits_off − freq_table_bytes`) or inline at the head of the
tensor's data region; either is conformant as long as the offsets are exact.

### 5.3 Example

A 768×768 body matrix, k=9, g=64, scale_mode=4:

    name         "transformer.h.0.attn.q_proj.weight"  (34 B)
    dims         [768, 768]
    digits       589,824 symbols, H≈2.70 b → ≈ 199 KB
    freq table   2·9 = 18 B
    scales       768·12 = 9,216 scales → ≈ 7.3 KB (ent-log8)
    record       ≈ 206 KB   (fp32 scales would add ~29 KB more)

---

## 6. Ternary pair mode (pair_mode = 1)

For `k = 3` only. Two consecutive ternary digits `t1, t2 ∈ {0,1,2}` are packed
into one base-9 symbol `p = 3·t1 + t2 ∈ {0…8}` and coded with a 9-symbol rANS
table. Decode: `t1 = p // 3`, `t2 = p % 3`.

This is information-tight: exp1/exp11 measured 1.584 b/param for the paired
stream vs 1.585 for the single-stream ternary coder — i.e. **the pairing buys
essentially nothing**; it is included because it makes the base-9 digit
identity literal in the file. The payload's viability comes from QAT
(exp11: 5.98 ppl vs 989.8 for post-hoc ternary), not the packing.

If `pair_mode = 1`, `digits_bytes` counts base-9 symbols and the digit count is
`(R·C)/2`; `R·C` must be even.

---

## 7. Canonical layout caveat (Conv1D)

`(out, in)` is the canonical layout; GPT-Neo `Conv1D` weights are `(in, out)`
and MUST be transposed before quantization, then transposed back on decode
(a shape flag in `header_flags` bit1 records this per tensor: `transposed`).

Groups run along the **input** dimension (`in`), matching exp4b/exp6/exp7 and
GGUF. **Note an inconsistency in the current research code**: exp11's QAT
grids group along the last axis of the raw parameter, i.e. the *output* dim for
`Conv1D`, so its body grids are not byte-identical to exp4b's. K9 fixes the
convention at input-dimension grouping; re-run exp11 under this convention
before quoting QAT bytes against PTQ bytes.

---

## 8. Optional fitted-codebook mode (v2, exp15-motivated)

exp15 measured that a *uniform* odd grid is within ~11% of the
entropy-constrained scalar quantizer (EC-SQ) bound at its own entropy, but a
*fixed-rate* Lloyd–Max fit sits 23% (Gaussian) to 80% (heavy tails) above the
EC-SQ envelope at matched entropy. The lever is therefore **entropy-constrained
grid design**, not the base-9 identity.

v2 adds `header_flags` bit0: each tensor may carry a fitted codebook instead of
a uniform grid.

- **v2a — shared codebook.** One codebook of `k` float16 centers per *layer
  type* (not per group), plus one scale per group. Cost `≈ k·16/(g·rows_per_codebook)`
  → ≈ 0.02–0.1 b/param, vs Lloyd–Max's 2.25 b/param at g=64. This alone could
  put a fitted 9-level grid near fp-parity at ≈ 2.9 b/param.
- **v2b — parametric codebook.** `k` centers generated from 2 floats (shape ν,
  scale) via an EC-SQ family; per-group cost ≈ 64 bits → 0.03 b/param. Fitting
  the 2 parameters per group at encode time recovers most of the 14–25%.

Record extension: `codec = 2` (fitted), plus a codebook section per
codebook-sharing group; dequant becomes `w = c[d]` (centers already carry the
scale) rather than the affine form. Decode is unchanged in shape.

---

## 9. Fixed-rate fallback (GGUF compatibility only)

GGUF quant types are fixed-size blocks; a variable-rate rANS stream does not
fit. For drop-in use, define `K9_0` as **uniform nine-level digits packed
4 bits/weight** (22 base-9 digits fit in a `u64`: `9^22 < 2^70`, 94.1% packing
density — RQ1). This is specifiable as a GGUF type but **gives up the entropy
coding gain**: 4.0 b/param raw vs 2.70 (body) / 1.81 (Gaussian) entropy-coded.
Use `K9_0` only when container compatibility matters more than size.

---

## 10. Decoder

```
read header; require magic == "K9Q1", version == 1, scale_bits >= 8
for each tensor record:
    f        = read uint16[k]                       # normalize check: sum(f) == M
    m        = decode_scales(scale_mode, scales_bytes)   # (R, C/g)
    d        = rans_decode(digits_bytes, f, n_symbols, scale_bits)
    if transposed: d = d.T ; m = m.T
    if pair_mode: d = unpack_trits(d)              # base-9 -> two trits
    Wr       = m[..., None] * (d.reshape(R, C/g, g) - H) / H
    W        = Wr.reshape(R, C)
```

Memory is `O(tensor)` — streams are decoded one tensor at a time; the decoder
never needs the whole model resident.

---

## 11. Conformance test (required)

A conformant implementation MUST pass, for each registered `k` and
`scale_mode`:

1. **Bit-exact digit roundtrip** — encode → file → decode reproduces `d`
   exactly (`np.array_equal`). Every rANS stream is asserted, as in `rans.py`.
2. **Dequant roundtrip** — decoded `W` equals the encoder's `W` bit-for-bit
   (float32), for both `transposed` values and both `pair_mode` values.
3. **Table normalization** — `sum(f) == M` and `f[s] ≥ 1` wherever the symbol
   occurs; reject otherwise.
4. **Size accounting** — reported `digits_bytes + freq_table_bytes +
   scales_bytes` equals the on-disk bytes.
5. **Cross-check** — on the exp13 artifact, decoding and re-coding the stored
   digits reproduces the published body rANS rate (RTN 2.701, GPTQ 2.709
   b/param) and the exp11 ternary pair rate (1.584 b/param).

An existing end-to-end precedent is exp5's printed-artifact round trip
(28.3 M weights → text → exact reload); K9 is the binary analogue.

---

## 12. Reference sizes (measured, exp16)

| component | params | k | digits b/param | scales b/param (ent-log8) | coded MB |
|---|---|---|---|---|---|
| body (RTN) | 28,311,552 | 9 | 2.700 | 0.125 | 10.0 |
| body (GPTQ) | 28,311,552 | 9 | 2.709 | 0.125 | 10.1 |
| embeddings | 40,170,240 | 99 | 6.275 | 0.094 | 32.1 |
| body (QAT-ternary) | 28,311,552 | 3 | 1.578 | 0.125 | 6.1 |

Full exp13 model (body + embeddings + fp32 norms/biases): **45.5 MB published
(fp32 scales) → ~42.1 MB with K9 ent-log8 scales** (−7.4%), digits unchanged.
The current `.npz` artifacts are 1.10 GB because digits are stored as int64 —
~26× larger than the coded model; K9 stores the rANS stream, not the digits.

---

## 13. Open items

- **Speed — addressed.** A C implementation of the identical coder
  (`../experiments/rans_fast.c`, built with `gcc -O3`, driven via ctypes from
  `rans_fast.py`) produces **byte-identical** output and is **46.8× faster to
  encode / 27.2× to decode** than pure Python (278 / 210 M sym/s vs
  5.95 / 7.72 on a k=63 stream). The whole 1.54 B-weight model now encodes in
  8 s and decodes + loads in 25 s. Remaining, optional: **lane-interleaved rANS**
  (N independent states advancing in lockstep) to expose SIMD/GPU parallelism —
  the single-stream serial dependency is inherent, so this is the only way to go
  beyond one core — and a **streaming decoder** that never materialises the full
  fp32 model (the current decoder briefly hit a CUDA OOM on a shared 16 GB card).
- **Mixed precision.** Real deployments keep some tensors at higher k; the
  directory already supports per-tensor `grid_k`, so this is an encoder-policy
  question (per-layer sensitivity), not a format change.
- **Ternary from-scratch QAT at scale.** The 1.578 b/param body payload is only
  attractive if the accuracy gap (exp11: +30% ppl vs control) shrinks with
  scale — the BitNet scaling hypothesis. Unverified here.
- **Streaming / mmap decode.** For >7B models, decode per-layer and free;
  the directory makes random access by tensor name cheap.
- **Fitted codebook v2a/v2b** (§8) needs an encoder-side EC-SQ fit; exp15's
  solver is the reference (λ=0 reproduces Lloyd–Max exactly).
