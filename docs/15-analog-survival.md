# docs/15 — Analog write-path survival, rung 1 (day 1)

Question: **what does an analog compute-in-memory array do to a model's
weights, and which encoding survives it?** This is rung 1 of the analog
ladder — a pure-software channel simulator ahead of any hardware. Substrate
first: the channel is the physics; encodings (f16, q8_0, q4_k_m, k-containers)
are three candidates riding on it. The ninths connection is a hypothesis, not
a requirement.

## Channel definition (`experiments/analog_channel.py`)

Weight-level write-path model. Per tensor:

- **Range-referenced noise** — programming + read noise is absolute, drawn as
  `N(0, sigma * amax_ref)` in weight units. amax_ref is either the tensor's
  full scale (`scale_ref=tensor`, naive array with one global gain) or the
  per-64-group full scale (`scale_ref=group`, gain-calibrated array: the
  analog-native equivalent of what quantization scales do). The defining
  difference from quantization error: quantization error is level-referenced,
  analog noise is range-referenced, so small weights suffer proportionally
  more.
- Rails at ±amax_ref (differential-pair ends), clip always.
- Retention drift `N(1, sigma_drift)`, multiplicative per cell (not yet swept).
- IR-drop: linear gain ramp across columns, far column loses `ir_alpha`
  (not yet swept).
- Stuck-at faults: fraction `stuck_p` of cells saturate to a rail, sign 50/50
  (not yet swept).
- Optional `levels` grid: the "the array itself is a quantizer" knob.
- `rail_limit`: absolute rail override for pre-amplified inputs (boost demos).

Common random numbers: noise draws seeded per (seed, tensor name), scaled by
sigma — every sweep point sees the same underlying realization, so curves are
smooth in sigma by construction. Selftest in the module; ALL PASS.

## Protocol (`experiments/analog_survival.py`)

- Subject: Qwen2.5-Coder-1.5B (ref `results/ref_qwen_1.5b_f16.gguf`).
- Subjects: **f16** (bound), **q8_0** (RTN on the container's exact 197-tensor
  coverage, fp16 block scales), **q4_km** (dequantized from
  `results/base1p5b_q4_km.gguf`), **k63** (production container decoded
  in-process from `results/qwen_coder_1.5b_k63_embed99.k9` via the python
  codec), **kN** (uniform-RTN onto the K9 symmetric odd grid
  `w' = m(d-H)/H`, m = fp16 per-64 amax; trend measurement only, no
  allocator).
- Per (subject, sigma): apply channel to every tensor → write f16-carried GGUF
  with the reference's KVs → fork `llama-perplexity` (build-cpu) on
  `results/wiki2_raw_test.txt`, `--chunks 32`, `-t 12`,
  `GGML_NO_BACKTRACE=1`.
- **Caveat**: `--chunks 32` samples the FIRST 32 chunks of wikitext, which are
  easier — absolutes are lower than full-corpus runs (f16 12.28 here vs 13.85
  full-corpus in docs/05 exp27). Internally consistent across all rows;
  key points need `--chunks -1` re-runs before quoting next to exp27 numbers.
- Single seed (0) so far. ppl SE ±0.4 at baseline, larger at high ppl.
- Data: `results/analog/survival.csv` (+ `boost_demo.csv`, per-run `logs/`).

## Result A — naive arrays are unusable; gain calibration is the fix

Tensor-ref (global gain) knee: σ ≈ 0.5–1% of full scale for EVERY subject
(f16 12.28 → 14.7 @0.5% → 195 @1% → 1.3e5 @2%; same shape for all). Real
memristor programming noise is 1–3% without write-verify, 0.3–0.5% with.
**Naive global-gain arrays are unusable at any real device spec.**

Group-ref (per-64 gain calibration): every subject clean to σ=1%
(ppl 17.3–21.7). At matched σ=1% the calibrated array is **~12× better** than
naive (q8_0 17.3 vs 216). Beyond the knee the failure is a PLATEAU, not an
explosion (q8_0: 3,985 @2% → 5,700 @4% → 9,762 @8%): the array saturates and
the damage stops growing for a while.

## Result B — coarse quantization is analog noise armor

Group-ref at σ=2%: **q4_km 273.7 ≪ k63 1,767 < f16 2,589 < q8_0 3,985** —
the 4.5-bit container is 14× more usable than the 8-bit one. Ordering is
near-monotone in bit-width AND both quantized subjects beat raw f16.
Mechanism OPEN (candidates: level-snapping as implicit error feedback /
flatter basins; q4_km value sparsity). Per-tensor-class ablation + multi-seed
queued. This result was not predicted by binary-era quantization intuition —
it is the seam the analog model and the container research share.

## Result C — amplification is local or nothing (boost demo)

`results/analog/boost_demo.csv`. Amplify every weight ×10 before storage,
rails pinned at the TRUE full scale, divide by 10 after readout (readback
division cannot undo clipping — it is linear; the rail is not):

| ×10 global boost | σ=0 | σ=0.1% | σ=0.5% | σ=1% |
|---|---|---|---|---|
| f16 ppl | **78,186** | 78,310 | 78,848 | 79,657 |

Clipping alone destroys the model at σ=0; noise adds almost nothing on top of
the wreck (flat plateau). Boost is a per-group budget game, not a global dial
— the working form of amplification IS Result A's gain calibration.

## Result D — the k-family crossover (best-k is a function of noise)

Uniform-RTN k-family, group-ref (baselines: k9 18.00 / k15 13.65 / k27 12.56 /
k63 12.2941 / k99 12.34; **k63-RTN matches the production container's 12.2941
to every printed digit — grid math validated**):

| σ | winner (ppl) | runners-up |
|---|---|---|
| 0–0.5% | f16 / k63r / q8_0 (~12.3–12.5) | everything within noise |
| 1% | **k15r 17.52** (q8_0 17.33 by a hair) | k63r 18.7, k27r 19.6 |
| 2% | **k27r 184.98** | q4_km 273.7, k9r 676 |
| 4% | **k63r 3,828** | k15r 4,829, q4_km 4,930 |
| 8–16% | q8_0 / f16 (~9.8k / ~10k) | baseline order re-emerges in wreckage |

Structure: the optimum is NON-monotone in σ. Mid-coarse k (15–27) owns the
1–2% band (real device territory), k63 takes 4%, and deep saturation reverts
to baseline ordering. Relative degradation at σ=1%: k15 1.28× < q8_0 1.41× <
k63 1.52× < k27 1.56× < q4_km 1.61× < k9 1.67× < f16 1.77× ≪ **k99 6.31×**.

**The k99 anomaly** is the sharpest open clue: k99-RTN has nearly identical
weights to f16 (≈6.6-bit snap) yet is 3.6× more fragile at σ=1% (77.8 vs
21.7). "Level spacing absorbs noise" predicts k99 ≈ f16; the measurement says
grid-snapping changes the solution's noise geometry. Dithered-rounding k99 is
the discriminating ablation. E3 (noise-aware training) tests the same question
from the training side.

## Protection ranking (for the hardware design)

Against programming noise (the term swept here): **per-group gain > exact-
zero/dead-zone > differential pair**. Differential pair wins against
common-mode terms (drift, temperature, aging) at the cost of √2 on independent
noise + 2× area. All three compose. Redundancy rejected: √k payoff for k×
area. Hybrid analog-native format sketch: digital sign outside the array +
per-group gain + analog magnitude + exact-zero handling.

## Honest caveats

32-chunk first-corpus protocol (internally consistent only); single seed;
uniform-RTN kN ≠ production allocator (k63-RTN vs production baseline agrees
to 5 digits); noise model is write-path only (no activation/ADC path yet —
that is a PyTorch-level Phase B); drift/IR/stuck axes implemented but not yet
swept; mechanism of Result B/D open.

## Next rungs

1. **E3 — noise-aware training** at 1.5B: fine-tune with channel noise
   injected; the big lever (moves models from "marginal on verified devices"
   to "comfortable"; may also improve clean quality).
2. Ablations: per-tensor-class noise (where does the cliff start?),
   dithered-rounding k99 (the anomaly), multi-seed at the knees.
3. Dead-zone + common-mode-drift knobs → exact-zero and differential-pair
   protection, measured not argued.
4. `--chunks -1` full-corpus re-runs of baselines and knee points (before
   quoting next to exp27).
5. Phase B: activation/ADC path in PyTorch (partial-sum quantization, input
   DAC) — the read-side channel.

## Addendum (same night) — the paired-draw correction, final verdicts

**Method rule, permanent: single-seed cliff claims are banned.** Cross-subject
comparisons near cliffs require paired draws (`--rng-key tensor`: every subject
at a seed shares one noise realization per tensor) + multi-seed. Measured cost
of ignoring it: f16 @2% = 26.2 (seed 0) vs 436.7 (seed 1) — a 17× swing from
the draw alone; 27× at 3%.

**Rescinded** (unpaired-draw artifacts): the "14× q4_km armor" spread (paired
σ=2%: all subjects 26–36); the k-crossover law (Result D); the k99 fragility
anomaly (paired @1%: 13.45 ≈ k63's 13.40 — never existed); dither's apparent
cliff rescue.

**Standing results:** naive-array death / group-gain 12× (survives: scale_ref
never entered the rng key); attention localization (attn-only ≈ full-model
damage; embed/ffn ≈ clean); dead-zone negative (6–10× WORSE — blunt comparator
deletes load-bearing small weights); boost clip catastrophe (ppl ~78k at σ=0);
k9 baseline tax. Full-corpus baselines validate the pipeline: f16 13.880 vs
exp27's 13.849 (five months apart).

**Dither verdict:** free at k99, ties pre-cliff, costs clean quality at coarse
k (k9d +13.8 ppl), loses every paired cliff cell. Fine-grid tool only.

**Error-feedback verdict (rose's snap-to-source idea):** negative on both
axes. k15e clean 20,557 (partly implementation: cross-group carry needs
scale-limiting); fair k27e test: clean 29.33 vs 12.56, @1% 44.35 vs 15.2 — EF
preserves block sums at the cost of individual weights; transformers need the
weights. Belongs in read-path ADCs (sigma-delta), not write-path snapping.

**E3 — noise-aware training (the headline):** LoRA on a substrate re-noised
every forward step; paired A/B on identical realizations:

| σ | control | noise-trained |
|---|---|---|
| 0 | 45.77 | 30.42 |
| 1% | 57.65 | 30.89 |
| 2% | 293.93 | 37.95 |

Cliff flattened (7.7× at 2%), and the noise-trained model is BETTER CLEAN
(val loss 1.69 vs 2.47 — substrate noise as regularizer).

**External convergence:** Cramer, Billaudelle et al., "Surrogate gradients
for analog neuromorphic computing", PNAS 2022 (BrainScaleS-2) — in-the-loop
training on analog silicon compensates decalibration to σ≈30% where naive
weight transfer collapses, and their analog SNN beat its noiseless software
twin (noise-as-regularizer). Both of our E3 findings, independently, on real
hardware. Their ITL surrogate-gradient loop = the appliance's training
blueprint (code: fmi-basel/brainscales-2-surrogate-gradients).

**Full-corpus final (583 chunks):** clean: f16 13.880 / q8_0 13.890 /
k63 13.888 / q4_km 14.104 / k27 14.177 / k15 15.304. σ=1% (knee — draw-robust,
~1% cross-run spread): k15 19.33 < q8_0 20.07 < k27 22.99 < f16 25.25 —
mid-coarse containers beat f16 AND q8_0 at realistic device noise, confirmed
on two protocols. σ=2% (single-seed, draw-chaotic — hints only, no ranking):
k27 268.8 / q4_km 321.5 alive-ish vs k15 3,047 / f16 3,188 / q8_0 5,048;
paired full-corpus cells queued before any claim.

**High-σ regime (rose's ± observation):** at σ=2% the ppl values explode AND
the ± scale with them — but the RELATIVE error stays ~1% (±30 on 3,047). The
estimates are fine; the model is wrecked multiplicatively: per-chunk damage is
roughly proportional (heavy-tailed across chunks, not random). Design reading:
at 2%+ on an untrained-for-noise model, no encoding saves you — the operating
envelope is set by the knee, and the knee is moved by E3 training, not by
container choice. Encode at the knee, train past it.

## Addendum 2 — night-2: mixed-σ, stuck-aware training, container closure

**Mixed-σ training** (σ ~ U(0,2%) drawn fresh each step) is the best model of
the project: 29.92 clean / 29.67 @1% / 32.98 @2% / 109.8 @4% — beats every
point-trained model at every σ AND on clean quality (val loss 1.552).
Noise curriculum, one model for the whole band: the default recipe for a
substrate whose noise wanders.

**Stuck-aware training** (σ=1% + 1% cells pinned to rails each step, fresh
mask) rescues the lethal failure mode:

| eval condition | control | σ1%-trained | stuck-trained |
|---|---|---|---|
| clean | 45.77 | 30.42 | 46.62 |
| 1% Gaussian | 57.65 | 30.89 | 48.64 |
| 2% Gaussian | 293.93 | 37.95 | 75.87 |
| **1% stuck only** | 56.5M | 21.6M | **324.9** |
| 1% stuck + 1% σ | 176,171 | 9,423 | **386.8** |

1% stuck cells killed every previous model (millions of ppl). Stuck-trained:
alive and usable (≈7× over its clean ppl, vs 10,000×+ for the others).
Software can substitute for defect maps at this scale. Notable: clean val
loss ranking control 2.47 → σ 1.69 → mixed 1.55 → stuck 1.46 — harsher noise
monotonically better in-domain; the stuck model's wiki ppl (46.6) stays at
control level though — the heaviest training preserved in-domain learning
but not cross-domain polish. Single seed, 32-chunk; within-pair comparisons.

**Container-robustness closure** (paired, full-corpus, 2 seeds, σ=1%):
f16 15.07/15.34, q8_0 15.08/15.36, k63 15.08/15.38, q4_km 15.42/15.71,
k27 15.45/15.82, k15 17.01/17.47 — relative degradation ≈1.09× for every
subject, no differentiation. All earlier container-robustness claims
(armor, knee advantage, k-orderings) are rescinded as methodology artifacts.
Standing rule: cross-subject claims need paired draws AND full corpus AND
≥2 seeds. Containers are a clean-quality/size trade; robustness lives in
training.
