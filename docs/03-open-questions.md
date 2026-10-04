# 03 — Open questions and roadmap

Ordered roughly by cost. Each item says what would count as an answer.

## RQ1 — Hardware/software reality of a 9-symbol alphabet
A 9-symbol alphabet has log2(9) ≈ 3.17 bits — not byte-, nibble-, or
bit-aligned. Questions: what does rANS decode *throughput* look like for a
9-symbol alphabet vs plain 4-bit nibble indexing; is there word-aligned
packing of base-9 digits worth having (9^k · bit tricks: 9^22 < 2^70 < 9^23,
e.g. 22 base-9 digits fit in a u64 with 5.6 bits to spare — measurable
packing density ≈ 94.1% of 70 bits).
**Answer form:** a decode-throughput benchmark and a packing-density table.

## RQ2 — Real-model validity (the one that matters)
Post-training-quantize a small open LLM (TinyStories/TinyLlama-class) to:
int8, int4, int3, ternary, and the 9-level ninths grid (with and without
entropy coding), and compare perplexity at matched *effective* bits.
**Answer:** a perplexity-vs-bits-per-weight curve; does 9-level sit on or
above the int3/int4 tradeoff line?
**Status:** first pass done 2026-10-03 on TinyStories-33M —
`../results/exp4_real_model_ptq.md` (int8/int4/8/ternary/9-uniform/9-Lloyd–Max/
27 + ternary-pair base-9 coder). See `../RESEARCH_LOG.md` for the reading.

## RQ3 — Codebook design at 9 levels
Uniform ninths grid vs Lloyd–Max (MSE-optimal for the weight distribution)
vs learned k-means centers vs ternary+scale. exp3 covers the first two;
k-means/learned at matched bits is the follow-up.
**Answer:** MSE/bits (then perplexity/bits) table; is the *uniform*
ninths grid close to optimal, or does optimality move the centers?
**Status 2026-10-03 (exp4b/6/7/8/9/9b):** on TinyStories-33M the fitted
Lloyd–Max codebook wins the accuracy crown (40.62 ppl @ 5.02 b/param with
the MEASURED fp16 codebook — dominates int8 on both axes); **GPTQ + Lloyd do
not stack** (compensation hurts fitted codebooks, +0.6 ppl — clean refutation
in exp8); the 127-level *uniform* anomaly is real (3 eval windows), global
(attn+mlp), and cured within noise by fitted codebooks (+0.41 → +0.18) —
endpoint/step structure (exp9/9b). Remaining: refit-after-compensation,
k-means codebooks at matched bits.

## RQ4 — Is there rational structure in real trained weights?
The P4 hypothesis at scale: take a real quantized checkpoint, histogram the
9-symbol (or ternary) digits, test for periodic/autocorrelated structure in
the digit stream and for denominator concentrations in continued-fraction
encodings of the dequantized weights. FFT/autocorrelation of the digit
sequence is the cheap detector.
**Answer:** a null or a signal; either is publishable as a small note.
**Status 2026-10-03 (exp12):** CLOSED, NULL — on exp11's *trained* grid
models (the best case for finding trained-in structure): lag-1 match rates
sit at their iid expectations (9-level: 0.181 vs 0.178; ternary: 0.346 vs
0.341; embedding 99-level: 0.017 vs 0.015), the order-1 Markov conditional
entropy gains ≈ 0.0–0.2% over the marginal H, and shuffled controls match.
rANS (frequency coding) is effectively optimal for these streams; the
lossless story needs nothing beyond symbol entropy. Interesting side
observation: trained ternary digits are nearly max-entropy (H ≈ 1.57–1.58
= log2 3) — training balanced the codebook.

## RQ5 — A real codec artifact
Implement the ternary-pair → base-9-digit → rANS pipeline as a standalone
compressor/decompressor file format (idea: GGUF-style `K9` quant type) and
measure end-to-end file size vs Q8_0/Q4_K/IQ3-class formats.
**Answer:** bytes on disk for a real model, plus decode speed.

## RQ6 — Model-as-digits (rendering / steganography side quest)
With weights on the ninths grid, a model *is* a string of base-9 digits,
which renders as text/art and can be re-parsed losslessly. Decimal-digit
carriers hold base-9 payload at ≈ log2(100)/2 ≈ 3.32 bits/digit capacity
(two decimal digits carry 9² = 81 of 100 combinations). Purely exploratory:
"the model printed as repeating digits in a PDF that unloads back to a
working model" is a legit, testable artifact if RQ2 shows the grid is
viable at all.
**Answer:** a round-trip print↔parse demo (only worth building if RQ2
passes).

## RQ7 — The 27-level extension
Three trits = one base-27 digit (log2 27 = 4.755 bits) — wedged between
int4 (16 levels, 4 bits) and int5 (32, 5 bits). Same framework, different
point on the curve; part of the general "non-power-of-two alphabets in the
awkward middle" study (docs/01 §4(a)).

## RQ8 — Training dynamics (far out)
Does 9-level QAT behave differently from 4-bit QAT at matched effective
bits? Most expensive question; only worth touching if RQ2 shows the grid
is competitive post-training.
**Status 2026-10-03 (exp11):** FIRST PASS DONE — weight-swap-STE QAT on
TinyStories-33M: QAT-recipe (body 9-ninths + embeds 99) loses only 6.4% ppl
vs a matched fp32 control (4.89 vs 4.59); **QAT-ternary works (5.98 ppl,
vs 989.8 PTQ)** at body 1.58 b/param — the base-9 pair codec payload is
viable, coding at 1.584 b/param in pairs. Open remainder: longer runs,
multi-seed variance, from-scratch pretraining with the recipe.