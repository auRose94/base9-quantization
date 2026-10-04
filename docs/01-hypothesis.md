# 01 — Hypothesis and mathematical foundations

Status: **pre-registered 2026-10-03, before the first experiment was run.**
The predictions in §5 were written down before any results existed; what the
experiments actually found is recorded in `../RESEARCH_LOG.md`.

## 1. The idea, formally

The origin idea (`00-idea-origin.md`) unfolds into three precise claims.

### Formalization 1 — the ninths grid (the "repeating numbers" claim)

Quantize each weight to a 9-level grid of multiples of one ninth:
`{0, ±1/9, ±2/9, ±3/9, ±4/9}` scaled by one per-tensor scale factor `s`
(the weight stored is the digit `k ∈ {-4..4}`, the dequantized value is
`k·s/9`).

Two identities make this grid special:

- **Base 10:** every ninth is a *single repeating decimal digit* —
  1/9 = 0.1111..., 2/9 = 0.2222..., 4/9 = 0.4444... — because **10 ≡ 1
  (mod 9)**, the classical "casting out nines" fact. So after quantizing to
  ninths, every weight in the model is literally *one repeating digit*: the
  model fits in the repeating numbers.
- **Base 9:** the same grid *terminates* — k/9 = 0.k exactly, since 9^-1 = 1/9.

Conversely, any base-9 digit string in {0,...,8} decodes to weights via the
grid. So "quantize to ninths" and "store the model as base-9 digits" are the
same operation, and the repeating-decimal view is the base-10 shadow of it.

### Formalization 2 — the ternary bridge (the "information" claim)

BitNet b1.58-style ternary quantization keeps weights in **{-1, 0, +1}** —
3 states, log2(3) ≈ 1.58496 bits per weight (the "1.58 bits" in the
literature).

Two ternary weights therefore have 3 × 3 = **9** joint states — exactly one
base-9 digit. Concretely, the bijection is:

    pair symbol = 3·(w1 + 1) + (w2 + 1)     w1, w2 ∈ {-1, 0, +1}  →  symbol ∈ {0..8}

and log2(9) = 3.1699 = 2 × log2(3), down to the last decimal. **Base-9 is to
balanced ternary what hexadecimal is to binary:** one base-9 digit is
exactly two trits.

| radix | one digit is... | bits |
|---|---|---|
| 2 | 1 bit | 1.0 |
| 8 | 3 bits | 3.0 |
| 16 | 4 bits | 4.0 |
| 3 (trit) | 1 ternary weight | 1.585 (log2 3) |
| **9** | **2 ternary weights (2 trits)** | **3.170 (log2 9)** |
| 27 | 3 ternary weights (3 trits) | 4.755 (log2 27) |

### Formalization 3 — the rational claim ("repeating numbers" taken literally)

A real number has an eventually periodic base-b expansion **if and only if**
it is rational (§2). Compressing "repeating numbers" is therefore the same
as storing the *rational* (numerator + denominator, or the repeating block).
The falsifiable question: **do quantized/trained weights carry rational
structure — bounded denominators, shared repeating periods — that beats
straight entropy coding of the digit stream?** §5/P4 predicts mostly no;
the experiment is cheap, so it runs.

## 2. Number-theory foundations

All claims in this section are machine-checked in
`../experiments/exp2_repeating_representations.py` (results in
`../results/exp2_repeating_representations.md`).

**Theorem (periodicity ⇔ rationality).** The base-b expansion of the
fractional part of x is eventually periodic (this includes terminating,
period 0) **iff** x is rational.

*Why:* long division of p/q produces remainders r_i in {0, ..., q−1}; either
some remainder hits 0 (terminating) or a remainder repeats (periodic). For
the converse, an eventually periodic expansion is a finite geometric series
and sums to a rational.

**Corollary (period length).** For p/q in lowest terms with gcd(q, b) = 1,
the base-b period of p/q is ord_q(b) — the multiplicative order of b modulo
q. The period divides q − 1 when q is prime.

**Terminating expansions.** A fraction terminates in base b iff its reduced
denominator has **only prime factors that divide b**:

- base 10: denominators of the form 2^a · 5^c terminate.
- base 9: since 9 = 3², only denominators 3^a terminate. Powers of 2 do
  **not** terminate in base 9.

**The base-9 / base-10 duality** (each row verified numerically in exp2):

| fraction | base 10 | base 9 |
|---|---|---|
| 1/2 | 0.5 (terminates) | 0.4444... (1-digit period) |
| 1/8 | 0.125 (terminates) | 0.1111... (1-digit period) |
| 1/3 | 0.3333... (1-digit period) | 0.3 (terminates) |
| 1/9 | 0.1111... (1-digit period) | 0.1 (terminates) |
| 1/7 | 6-digit period (ord_7(10) = 6) | 3-digit period (ord_7(9) = 3) |

Base 10 "likes" powers of 2 and 5 (they terminate); base 9 "likes" powers of
3. The **ninths grid is exactly the largest grid that both terminates in
base 9 and has a single-digit repeating period in base 10** — which is why
the origin idea's two instincts (base 9 + repeating numbers) point at the
same object.

### The b^L−1 law — multi-digit repeating grids

The single-digit story generalizes cleanly. An odd symmetric grid of **k =
2H+1** levels is `w = (b − H)·m/H` with block index `b ∈ {0..k−1}` (per-(row,
group) scale m). Whenever **k = bᴸ − 1**, that block *is* an L-digit repeating
decimal in base b: `b/k = 0.bbbb...` (period L), and the dequant identity is
**affine in one repeating decimal**: `w = m·((k/H)·0.b̄ − 1)` — for L=1,
`w = m·(9/4·0.s̄ − 1) = m·(s−4)/4`, exactly exp5's printed equation.

| base b | L | levels | grid |
|---|---|---|---|
| 10 | 1 | 9 | ninths — 1 digit; also = 2 ternary trits |
| 10 | 2 | 99 | two-digit block "0.(43)" = 0.434343… |
| 10 | 3 | 999 | three-digit block |
| 2 | 4 | 15 | binary 4-digit repeating — **= int4-15, exp4's RTN champion** |
| 2 | 8 | 255 | binary 8-digit — = int8 symmetric |

So the GGUF-style "k-bit minus one level" odd grids (15, 31, 63, 127, 255)
are the **binary members of the same family** the ninths grid belongs to —
multi-digit repeating decimals in base 2. Base-3 multi-digit (3²−1 = 8
levels) is an EVEN grid — no zero level — and loses for symmetric weight
distributions (exp3), which is a property of the family membership, not a
fluke. The printing artifact of exp5 generalizes verbatim to L-digit tokens
("0.(43)"). exp7 (`../experiments/exp7_multidigit.py`) tests the decimal
family (9/99/999) against the binary family on TinyStories-33M, with
predictions P6–P10 pre-registered in its docstring before the run.

## 3. Information accounting

- A 9-symbol alphabet is **not a whole number of bits** wide: fixed-width
  indexing costs ceil(log2 9) = 4 bits/symbol (wasteful), while the alphabet
  content is log2 9 = 3.1699 bits/symbol. Achieving the fractional-bit rate
  requires entropy coding (arithmetic coding / rANS — asymmetric numeral
  systems), not naive bit packing.
- Ternary: log2 3 = 1.585 b/param. Pairing two trits into one base-9 digit
  gives 3.170 bits per *pair* = 1.585 b/param — the pairing is
  information-tight *if* ternary symbols are uniform. Real ternarized
  weights skew toward 0, so the joint entropy of pairs can be lower than
  2 × log2 3; the 9-symbol distribution is then skewed and entropy coding
  gains more.
- int8 ≈ 8 bits/param (nearly uniform for Gaussian-ish weights); int4 = 4
  bits/param (16 levels, also nearly uniform). The 9-level grid sits at
  **3.17 effective bits — awkwardly between 3-bit (8-level) and 4-bit
  (16-level)**. Whether 9 levels is *useful* there is an accuracy-per-bit
  question (exp3), not an information question.

So any genuine win from the base-9 framing must come from exactly one of:

1. **Skew:** the 9-symbol (or ternary, or ternary-pair) distribution is
   skewed → rANS/entropy coding beats 4-bit indexing of the same grid.
   Measured in exp1.
2. **Accuracy per bit:** 9 levels at ~3.17 effective bits vs 8 levels at 3
   and 16 at 4 — does the awkward middle earn its place? Measured in exp3
   (incl. the optimal Lloyd–Max 9-level grid as the fair ceiling).
3. **Engineering:** digit-parallel decode, storage-layer tricks, or a
   GGUF-style codec. Open (docs/03).

## 4. Scope — what this does *not* claim

Established prior work (see docs/02) already covers: ternary/1.58-bit
*training* (BitNet), entropy coding *on top of* quantization (Deep
Compression's Huffman coding of quantized weights, 2015; ANS-based coders
since), and learned codebooks (AQLM/GPTVQ). This project does **not**
preclaim novelty there.

The specific bets that look under-explored:

- **(a)** non-power-of-two alphabets *between* 2^k widths (5, 6, 7, 9, 11,
  12, 13 levels) evaluated at entropy-matched bits — most hardware-driven
  quantization literature jumps 3 → 4 → 8 bits and leaves the middle out.
- **(b)** whether a *rational-grid prior* (weights on small-denominator
  fractions) adds lossless compression beyond symbol entropy — registered
  as P4, expected null.
- **(c)** the two-trits-per-base-9-digit identity itself as a documented,
  testable storage codec for ternary models (a small, citable observation
  even if the accuracy story is neutral).

Explicitly **not** claimed: base-9 hardware arithmetic is faster or
more efficient (it almost certainly is not, see docs/03 RQ1); that
repeating-decimal *notation* reduces storage by itself (notation ≠
compression — exp2 measures the gap); anything about 9-level training
dynamics (QAT is future work).

## 5. Pre-registered predictions

Written 2026-10-03 before any experiment was run. Each names the experiment
that tests it and what would refute it. Verdicts are recorded in
`../RESEARCH_LOG.md`.

- **P1 (near-uniform ternary).** Absmean-ternarization of Gaussian weights
  yields a symbol distribution within 0.1 bits of the 3-symbol maximum:
  H ∈ (log2 3 − 0.1, log2 3] ≈ (1.485, 1.585]. Tested by exp1.
- **P2 (9-level skew).** The 9-level ninths grid on Gaussian weights has
  symbol entropy strictly below log2 9 = 3.1699 (central levels carry more
  mass). Tested by exp1.
- **P3 (awkward middle).** Relative MSE of the 9-level grid lies strictly
  between that of 8-level (3 bits) and 16-level (4 bits) grids; the
  Lloyd–Max (entropy-informed) 9-level grid improves on uniform ninths.
  Tested by exp3.
- **P4 (rational encoding loses).** Minimal-rational (continued-fraction)
  encoding of individual weights never Pareto-dominates digit-index coding
  (with entropy coding) — i.e., there is no compression win hiding in the
  "repeating numbers" beyond entropy coding. Tested by exp2.
- **P5 (codec tightness).** A correct rANS implementation reaches the
  empirical entropy of the 9-symbol and ternary-pair streams within ~2%.
  Tested by exp1 (roundtrip-asserted).

- **P6–P10 (multi-digit family — session 4).** pre-registered 2026-10-03 in
  `../experiments/exp7_multidigit.py`'s docstring before its run: wide-alphabet
  entropy < log2(levels) on every grid; ppl monotone in level count; GPTQ
  gain shrinking with grid fineness; bit-exact multi-digit print round trip;
  99-level below the 63↔127 interpolation. Verdicts in `../RESEARCH_LOG.md`.

## 6. Threats to validity

- Gaussian weights are stand-ins: LLM weight distributions are heavier-tailed
  and group-structured. Synthetic results establish the math and the codec
  floor, not model quality. Downstream validity needs a real model +
  perplexity (docs/03 RQ2).
- Frequency tables for the coder are shared metadata and excluded from
  bit counts (standard practice; ≤ ~40 bytes at n = 200k, negligible but
  noted).
- MSE is an accuracy *proxy*, not accuracy.