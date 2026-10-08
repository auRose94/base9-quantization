# weights-as-equations

**Model weights as analytic equations, synthesized on the GPU at run time.**
Companion research to `base9-quantization` (lossless digit-stream compression of
quantized weights). Here the representation is fundamentally different: a
trained model's weights become a *variable table + fixed math* — a small set of
fitted coefficients multiplying analytic basis functions. The GPU evaluates the
equation to fill a tensor grid ("synthesize the weights"), and inference runs
from the synthesized buffers. The weight tensor never exists as carried data;
it is computed.

## The mapping (vision → concrete objects)

| vision term | concrete object |
|---|---|
| "equations describing the AI model" | `W[i,j] = Σ_k c_k · φ_k(i,j)` — a basis expansion per weight matrix |
| "variables the special program supplies" | the fitted coefficient table `c_k` (+ fp16 norms), quantized |
| "mathematical constants (not data)" | the basis `φ_k` — pure formulas (DCT-II cosine terms), built on device |
| "GPU interprets outputs as tensor grid data" | a synthesis pass writing the weight buffers from variables + formulas |
| "model lives on GPU as a program" | weights produced by that synthesizer at load time; variables resident, weights derived |

Facts of life this makes explicit:

- Synthesis trades **bytes for FLOPs** (the same bargain as procedural textures
  vs. textures). The research question is where that trade is worth it.
- With a *fixed analytic basis*, quality hinges on coefficient concentration:
  how few coefficients preserve model quality at what size. That is measurable
  with the same paired multi-window perplexity discipline as base9.
- Losslessness is NOT assumed here. Quality is reported against fp32/fp16
  anchors; if a reconstructed model matches, the equation *is* the weight.

## RQ1 results — stories260K pilot (2026-10-07)

Headline: **the machinery works, the representation doesn't (yet).**

| scheme (per-matrix) | size | val loss (384 paired windows) | vs fp32 anchor 1.9726 |
|---|---|---|---|
| fp32 / fp16 | 32 / 16 b·p | 1.9726 / 1.9725 | 1.00× |
| uniform quant | 8 / 7 / 6 b·p | 1.9797 / 2.0115 / 2.0548 | +0.4% / +2.0% / +4.2% |
| uniform quant | 5 / 4 b·p | 2.4884 / 4.4292 | +26% / ×2.25 |
| svd-r (learned basis) | 0.44…12.6 b·p | 6.2–8.9 | ≥3.1× — never works |
| DCT low-pass (index-free) | 0.05–1.35 b·p | 5.8–10.0 | best 2.9× — fails |
| DCT top-K (indexed) | 0.26–8.65 b·p | 6.0–7.6 | best 3.1× — fails |
| DCT top-K, all coefs, b=8 (machinery control) | 21.6 b·p | 2.0187 | +2.3% ✓ |

- **Structure exists but not where we first looked.** Weight matrices are not
  smooth in index space (DCT concentration ≈ iid control for the body), but
  every matrix has a dominant rank-1 direction (rank@99% energy = 1–3% of
  min-dim; controls 77–99%). Outlier-dimension structure — real, yet keeping
  that energy does NOT preserve function: svd-r fails everywhere (the
  low-energy tail is functionally load-bearing).
- **Truncation, not coefficient precision, is the killer**: at f=0.05, b=23
  vs b=8 coefficients are indistinguishable (7.50 vs 7.52).
- **GPU synthesis (the vision) is validated** (P5): 79.3 KB of variables
  (0.08× fp32) + a basis built on device from cosine formulas → all weight
  buffers synthesized in 80 ms, max|Δ| = 1.5e-5 vs CPU, story sampled
  end-to-end from weights that never existed as data.
- **eq3 scoping**: naive INR/MLP fields (weight-L2 fit) don't beat fixed
  bases at matched bytes either ⇒ the promising next step is *functional*
  fitting of the variable table against val loss (NeRN-style), plus the
  base9/rANS codec for the quantized weight variables (their proven pipeline
  already gives +0.4% @ 8 b·p → sub-byte packing + entropy coding next).

Figures: `results/eq0_spectra.png`, `results/eq1_curve.png`,
`results/eq3_fieldmap.png`; samples: `results/eq2_samples.md`; story from
GPU-synthesized weights in `experiments/eq4_gpu_synth.py` output.

## Session 2 (2026-10-07): base9 fusion + trainable variables — the artifact chain works

The base9 stack became the model representation (RQ4), the K9 scales became
**trainable variables** (RQ3a), and the artifact became a live GPU-synthesized
program (RQ2a):

| K9Q1 artifact (RTN + g=64 group scales) | bytes | b/p | val loss | vs fp32 |
|---|---|---|---|---|
| body k9 + embed k99 | 116 KB | 3.58 | 3.0925 | +57% |
| body k27 + embed k99 | 165 KB | 5.09 | 2.0439 | +3.6% |
| body k63 + embed k99 | 203 KB | 6.26 | 1.9939 | +1.1% |
| ...then 400 steps of scale-only functional fit ||||
| k9 + embed k99, fitted | 116 KB | 3.58 | 1.854 | **−6%** |
| k63 + embed k99, fitted | 203 KB | 6.26 | **1.4397** | **−27%** |

- The working recipe: your exp12 lesson held — rANS digit streams don't
  benefit from a DCT pre-transform (replicated null at 1.001×) — but the
  deployed artifact is smaller AND better than the fixed-basis equations of
  session 1 at every point.
- The functional-fit result (going below the fp32 anchor at 5× less data)
  is the session's headline: the variable table has slack that weight-space
  fidelity measures can't see. For an undertrained tiny model the slack is
  large; how much survives at 33M+ is exactly the next question.
- Deployment loop (`eq7_synth_kernel.py`): K9Q1 file → rANS decode → Triton
  kernel materializes fp16 buffers on device (fp32 gate: 1.2e-7 vs the
  container decoder) → eval/story identical to the fitted numbers.

Next: digit-STE + GPTQ compensation for the k9 point (base9 exp6 transfer),
GPU-parallel rANS decode (rANS kernel = RQ2c), then the 33M scale-up.

### The K-ladder (session 2b): does higher K win?

Both halves of the intuition were tested — `eq8_high_k.py`:

| fitted artifact (embed k99, 400-step scale fit) | bytes | RTN pre-fit | fitted loss |
|---|---|---|---|
| body k9 | 116 KB | 3.0925 | 1.854 |
| body k27 | 165 KB | 2.0439 | 1.4631 |
| body k63 | 203 KB | 1.9939 | 1.4397 |
| body k255 | 274 KB | **1.9730 (= fp32 anchor)** | 1.4233 |
| body k1023 = two 5-bit planes (exp7 multi-digit law) | 329 KB | 1.9751 | 1.4250 |
| fp32 | 1040 KB | — | 1.9726 |

k255 is the exact-fidelity point (RTN within 0.02% of fp32 — the 8-bit class
prediction held). But after the functional fit, quality saturates: k255's
+71 KB over k63 buys 0.016 nats; k1023's +55 KB buys nothing (it even inverts
by 0.0017 — flat noise). **The size-quality frontier post-fit is k27–k63** for
this model class. The "turn into math" idea was validated at the coding layer:
k1023 decomposes into two base-32 digit planes that rANS at 10.435 b/p of body
— the exp7 multi-digit law holds end-to-end on a real artifact.

### QAT on the grid (session 2c) — the scale-fit ceiling was not the grid's ceiling

Porting exp11's digit-swap STE (digits reassignable during training, scales
recomputed per step, matched fp32 CONTROL arm, 2000 × 32k tokens each):

| arm | bytes | val loss | vs matched CONTROL (1.2962) |
|---|---|---|---|
| CONTROL fp32 (same training) | 1040 KB | **1.2962** | — |
| QAT k27 + embed k99 | **165 KB** | 1.3143 | **+1.4%** |
| QAT k9 + embed k99 | **116 KB** | 1.4587 | +12.5% |

The matched control converged to 1.2962 — within 0.0006 of the checkpoint's
published best_val_loss (1.2968), reproducing their training target in this
harness. Every arm ships as a real K9Q1 artifact (reload identity ≤ 5e-5) and
samples stay coherent. Current state of the art for this pilot:
**fp32-training-equivalent quality at 165 KB (6.3× smaller than fp32)** — and
−33% loss vs the original source weights. The session-2b saturation (~1.42)
was a property of the fit recipe (scales-only), not of the grid.

## Session 3 (2026-10-07): converged controls, seeds, and the 33M scale-up

**eq10 (longer training + seeds, stories260K).** CONTROL@4000 = 1.2939 — the
2000-step control was already CONVERGED (opposite regime from exp14), and the
published 1.2968 is reproduced within 0.0006. QAT-k27 holds **+1.6% vs control
at 6.3× compression** across seeds (seed Δgap = 0.0001); QAT-k9 = +13.4%.

**eq11 (33M scale-up, roneneldan/TinyStories-33M — actually a 68.5M-param
tied GPT-Neo; fp32 = 274.1 MB, matching base9's exp13 baseline):**

| arm | paired val loss | artifact |
|---|---|---|
| fp32 source (anchor, our windows/prep) | 1.3550 | 274.1 MB |
| CONTROL fp32 @2000 steps | **1.2817** | — |
| QAT-recipe (body k9-g64 + wte/wpe k99-g64) | 1.4352 | **43.3 MB (−84%)** |

+12.0% paired loss at −84% bytes, against a nearly-converged matched control —
the same compression class as exp13's GPTQ deployable point (45.5 MB @ +10.6%
on their windows), but with **zero calibration machinery** (STE training
compensates instead). Breakdown: the embedding tier (wte+wpe @ k99) is 34 MB
of the 43.3 — k27 embeds under QAT is the biggest next lever (−9 MB).


### Session 4 (eq12): embed-tier sweep — and an honest protocol correction

Cross-checking eq12 against eq11 exposed that **eq11's QAT arm was
warm-started from the trained control** (module mutation leaked across the
sequential arms); eq12's fresh-source-load pipeline is the clean one:

| arm (source-init, 2000 × 32k, seed 42) | val loss | vs CONTROL 1.2817 | artifact |
|---|---|---|---|
| recipe body-k9 + embed-k99 | 1.5227 | +18.8% | 43.2 MB (−84%) |
| recipe + embed-k63 | 1.6483 | +28.6% | 40.0 MB |
| recipe + embed-k27 | 1.7690 | +38.0% | 33.8 MB |

Findings: (1) **the honest 33M paired gap for the recipe point is +18.8% at
−84% bytes** (eq11's +12.0% stands as the warm-start variant); (2) the
embedding tier under QAT is a capacity knob — coarsening it costs ~2× more
quality per MB saved than budgeted, and the fit-pool losses go the *other*
direction (coarse tables adapt the pool while damaging the geometry); (3) the
LR sweep silently never varied (imported schedule captured eq11's 1e-4 — bug
logged, LR question open). Standing fair frontiers: 260K k27-QAT = **+1.6% @
165 KB**; 33M recipe = +18.8% @ 43.2 MB; exp13 GPTQ-class = +10.6% (their
windows) — the 33M QAT gap is not yet at exp13's level: longer budget,
warm-start-as-deliberate-protocol, and the RQ2b/c machinery are next.

## Phases

- **RQ1 — stories260K pilot (this session).** Is there structure for an
  equation to exploit? DCT-II / SVD spectra of every weight matrix with
  shape-matched iid-Gaussian controls; coefficient-truncation sweeps (two
  schemes: index-free low-pass rectangle vs magnitude top-K), quantized
  coefficients, paired val-loss eval, story samples original vs equation-model.
- **RQ2 — GPU synthesis.** Weights materialized on device from the coefficient
  variables via formula-computed basis (no basis data shipped); end-to-end
  model-from-variables; bytes-moved accounting. Triton fused matmul-synthesis
  (compute weights inside the matmul, never materialize) after that.
- **RQ3 — learned equations.** INR/NeRN-style neural fields
  `W[i,j] ≈ MLPθ(ξ,η)` (more expressive than a fixed basis; θ is the variable
  table), joint-across-layers variants, cross-layer shared bases.
- **RQ4 — fusion with base9.** The coefficient stream is exactly the input
  base9+rANS was built for: quantize coefficients on a grid, code the digit
  stream losslessly, compare artifact size vs exp13's 45.5 MB @ 44.96 ppl
  point (TinyStories-33M).
- **RQ5 — scale out.** TinyStories-33M and beyond; budget the FLOPs; decide
  where synthesis beats carrying data.

## Repro

- Model: karpathy/tinyllamas `stories260K` (llm.c llama2-style export; Apache-2.0
  origin stack) + its own SentencePiece tokenizer `tok512` (512-vocab).
- Val data: roneneldan/TinyStories `TinyStories-valid.txt` (19.4 MB, CDLA-Sharing-1.0).
- Python: `/mnt/matrix/Work/base9-quantization/.venv-baselines/bin/python`
  (torch 2.14.0 CUDA, Python 3.14). GPU: RTX 5060 Ti 16 GB.
- Run `experiments/eq0_structure.py` first, then `eq1_fit.py`, `eq2_samples.py`,
  `eq4_gpu_synth.py`. Verdicts vs pre-registered predictions are asserted and
  logged (see `RESEARCH_LOG.md`, newest first).
---

## Where this stands (2026-10-07)

The digit-native thread (shared with
[base9-quantization](https://github.com/auRose94/base9-quantization), whose
`docs/09-digit-native-coding.md` carries the registered questions RQ9-RQ13)
is fully measured. Headlines, all with pre-registered predictions in
`RESEARCH_LOG.md`:

- **RQ10 stage B (the exact grid runtime) is CLOSED**: a 260K-class
  transformer trained (+0.70%), stored as a K9Q1 artifact, and then RUN in
  exact integer arithmetic on the scaling alphabet — one defined regrid per
  layer, machine-word carriers (worst 12.6 / 5.7 base-9 digits), order-exact,
  faithful to fp32 within 0.24% of the logit scale. The one int64 wrap found
  (the ff-product) is fixed by an escalating multiply; eq33 shows the same
  wrap exists on CUDA int64, and torch has no int64 matmul there.
- **RQ12 bits-back (eq25)**: the chain works end to end (one file carries the
  model AND the randomness its first generation uses; 3 posteriors, corruption
  detected 5/5), and its size accounting is exact — on a deterministic
  artifact the pop channel is a *carrier*, not a compressor. P45 FAIL with
  the slack located; P46 PASS.
- **RQ13 companions (eq26/eq27/eq30-32)**: the context posterior recovers
  **50.74% of the static-table bit cost** on the runtime's intermediate digit
  streams (held-out, second-order, concentrated at the first hidden state) and
  **+47.96% realized bytes in the C coder** at a 1k cadence with free,
  prefix-derived tables; the registered throughput bar (>= 50% of the byte
  coder) is closed as **out of reach on this CPU** (measured ceiling 40-47%,
  with two hypotheses refuted and the costs attributed).
- **RQ13 c-e, the amortizer (eq29)**: one diagonal calibration pass plus a
  per-group scale search reaches **ppl 40.753 at 0.11 s** — 146.6% of GPTQ's
  gap reduction, 4.43 ppl BETTER than GPTQ and 347x cheaper — matching the
  fitted Lloyd-Max codebook's quality at 0.5 b/param of side-info instead of
  4.5.
- **RQ13 c-x, the executor A/B (eq28)**: a tiny specialist reaches 99.60%
  exact on unseen renorm tasks (the registered bar), a locally served
  Qwen2.5-Coder-7B reaches 69.3% — and out of distribution they converge,
  because the specialist interpolates the range it trained on while the
  generalist reasons from the rule.

The honest summary: this is a **packing decision, not a silver bullet for
optimization.** It packs weight data more densely, at a real cost in
complexity and compute, and it proves the packing can be done exactly on the
grid. It is published so that someone who wants to pick it up can.

## Credits and acknowledgements

This work stands on other people's research, and the thanks are not a formality:

- **Jarek Duda** created **rANS** (range Asymmetric Numeral Systems) and
  released it **patent-free**. Every coder here — the byte coder, the base-9
  state machine, the context-conditioned engines — is his algorithm. A
  patent-free entropy coder is what makes an unencumbered implementation of
  this entire thread possible at all. Thank you.
- **Geoffrey Hinton & Drew van Camp (1993)** for bits-back coding, and
  **James Townsend, Tom Bird & Julius Kunze (ICLR 2019)** for bb-ANS, the
  practical algorithm RQ12 implements ([arXiv:1901.04866](https://arxiv.org/abs/1901.04866));
  their reference implementation (`bits-back/bits-back`) was used to
  cross-check the experiment's operation order and seed handling.
- **Krichevsky & Trofimov** (the KT prior), **Witten & Bell** and
  **Jelinek & Mercer** (the smoothing and interpolation the context companion
  uses), and **Lloyd & Max** (the fitted 9-level codebook the amortizer
  matched at a third of the side-info).
- **Elias Frantar and colleagues** for **GPTQ**, the optimizer baseline every
  amortization claim here is measured against.
- **Johannes Ballé, David Minnen and colleagues** for the learned-image-
  compression architecture (an analysis transform plus a context model feeding
  a range coder) — the external anchor for RQ13's division of labor between
  the main model and codec companions.
- **Andrej Karpathy** for **llama2.c / tinyllamas**: the reference forward
  pass this repo's from-scratch implementation was validated against, and the
  stories260K pilot model.
- **Ronen Eldan** for **TinyStories** and the TinyStories-33M model — the
  corpora and model every quality number here is measured on.
- **Qwen / Alibaba Cloud** for the Qwen2.5-1.5B and Qwen2.5-Coder-7B models
  used in the corpus work and the executor A/B.
- **Georgi Gerganov and the llama.cpp contributors** for the inference stack
  and tooling behind the k9 gates, the GGUF work, and the local generalist
  serving.

## How this was made — an AI-assisted research project

This research was done by **Rosemary Mercury** with an **AI assistant —
glm/deepseek running inside ZCode**. The assistant wrote and ran the
experiments, measured and logged the results, and drafted these diary
entries; the idea, the priorities, the pre-registrations, and the review came
from the author. Every experiment asserts its own machinery, every prediction
is pre-registered in `RESEARCH_LOG.md` before it is tested, and failures are
logged as findings — several of them, prominently, including two refuted
hypotheses and one registered bar that was never met.

### A note from the author, in her words

> I only had an idea and my credit isn't important. I don't believe I made
> anything, anymore than I had an idea and used [the assistant] to see it.
> I was wondering how an AI would handle me asking (to me, silly) math
> questions and wanted to research into an idea. I would feel too full of
> myself if I did a pull request to anything, so the research sits for
> someone to pick it up. It's not a waste, for now it proves a lot. It just
> comes at a severe cost to make data more packed, and it's not a silver
> bullet for optimization but a packing decision. I'm just some house wife
> that honestly was kinda stunned no one wanted to research this and having
> the novelty be entertained was reason enough to continue for me.

Nothing here is a pull request against anyone's project. It is published so
that whoever wants to pick it up can — take it, measure it, refute it.

**Publication note:** this research lives on the `weights-as-equations`
branch of the [base9-quantization](https://github.com/auRose94/base9-quantization)
repository (the two are one thread). To split it into its own repo:
`gh repo create weights-as-equations --source . --push`.
