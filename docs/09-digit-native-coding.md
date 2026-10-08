# 09 — Digit-native coding: the grid as runtime, output channel, and its own codec

**Origin (2026-10-07).** After session 24/25 and the weights-as-equations
fusion, the program so far is: *train as digits* (QAT on the grids — exp11,
wae eq9) and *store as digits* (K9Q1 + rANS, docs/04). The missing third leg
is **run as digits** — inference and output that never leave the grid. The
questions that opened this thread, kept verbatim from the conversation:

> "can we make models exclusively use a scaling alphabet internally without
> quantization? Could a model output rANS encoded tokens?"

> "If the model knew how rANS encodes, it could decode as more data in the
> end as a side effect. Or that's the hope."

> "Decode and encode on the grid might be worth investigating, because if
> you can do one, perhaps you could do the other."

Priorities as set 2026-10-07: **RQ9** is the smoke test (cheapest),
**RQ10/RQ11** are the core, **RQ12** is the ambition. The "learn the codec
as a curriculum" idea (rung 2) survives only as the opportunistic P44 probe
inside RQ11 and is NOT expected to pass.

## 0. The closure facts (why the grid can be a runtime at all)

Written out once; RQ9–RQ12 all stand on them.

- Finite base-9 expansions are exactly the rationals m/9ⁿ, and they are
  **closed under multiply and add**: (m/9ᵃ)(k/9ᵇ) = mk/9ᵃ⁺ᵇ, and sums share
  a denominator. Multiply-accumulate against grid values therefore never
  leaves the grid — it costs mantissa digits, not fidelity.
- The exact-accumulation cost is bounded, not exploding: summing d terms
  with mantissa span n digits and exponent spread s needs span ≤
  **s + n + ⌈log₉ d⌉** digits after a power-of-9 renormalization — and
  renormalizing by 9^k is an *exact digit shift* (this is what the
  "scaling" in scaling alphabet buys; a plain finite alphabet cannot
  renormalize losslessly).
- The base-9 mirror of the exp2 identity: **9 ≡ 1 (mod 8)**, so ÷2, ÷4, ÷8
  have period-1 repeating expansions in base 9 (1/2 = 0.(4), 1/4 = 0.(2),
  1/8 = 0.(1)) — same law as 10 ≡ 1 (mod 9) ⇒ 1/9 = 0.111… in base 10.
  Divisions by powers of 3 terminate; reciprocals of powers of 2 repeat
  with period 1.
- What genuinely escapes every base: **sqrt and exp**. RMSNorm's sqrt and
  softmax's exp cannot be rounded onto the grid — they must be *replaced*.
  Replacement is an architecture choice (the model is defined this way),
  not a representation loss: power-of-9 scale norm instead of RMS,
  rational attention kernels instead of softmax (rational functions are
  closed under +,−,×,÷; learned rational activations are established art —
  Padé activation units, Molina et al. 2019), clamp/ReLU gates are already
  closures.
- Honest asymmetry to keep in view: closure is not a free lunch. The runtime
  win is *exactness* — integer add/multiply are exactly associative, so the
  forward pass becomes order-independent and bit-identical across devices —
  plus digit-uniformity of state (activations are .k9-able streams like
  weights). Quality is NOT expected to beat fp compute; it must merely
  survive the op substitutions, measured in RQ10.

Two design choices are what make exactness fit in u128-class hardware
instead of bignum:

1. **Norm = power-of-9 renorm**, keyed on the max exponent bucket, with no
   division at all. (Mean-abs / RMS norms carry a ÷d or a sqrt; only
   power-of-9 scaling is division-free and exact.)
2. **Rational attention over a FIXED shared denominator per layer** —
   scores map to integer numerators n₁…n_d over one constant D, so the
   output (Σ nᵢ vᵢ)/D is exact with a bounded, constant denominator.
   (Free denominators would lcm-grow across layers; the fixed D is the
   definitional choice that keeps them constant.)

## RQ9 — Decode-synchrony output: generation as decompression (rung 1, smoke test)

The cheapest test of "a model outputs rANS tokens". Training-free,
inference-only, on any served model with logit access (the llama.cpp fork's
server, or the HF eval harness). Three demonstrations, one artifact each:

1. **Seeded regeneration.** Bits file + seed → entropy decode over per-step
   token distributions → a story. Re-run: identical story, bit-exact.
2. **Scripted generation.** The reverse: entropy-encode a *chosen* text
   under the model's own per-step distributions; the resulting bytes decode
   to that text exactly. Byte cost = that text's cross-entropy under the
   model — the chosen continuation costs exactly what it is worth.
3. **The rate dial.** bits/token vs temperature: T→0 should collapse toward
   0 bits/token (argmax carries no information); T→∞ toward log₂(vocab)
   scale. Determinism and compression are two ends of one dial.

**Answer form:** three artifacts plus the rate/temperature curve.

**Status (first pass 2026-10-07, wae eq15 — P34/P35/P36 all PASS):** on the
eq5_C_k63emb99 artifact, CPU: scripted generation = a **1500-token story in
264 bytes** (1.408 b/tok; coder overhead +0.011 nats/token vs masked CE);
regeneration ×2 bit-exact, one flipped bit diverges the decode at token 747;
rate dial monotone over τ ∈ [0.1, 50]: 0.10 → 8.95 b/tok (τ→∞ limit =
log₂ 512 = 9 ✓). The τ=0.1 residual of 0.10 b/tok is the artifact's own
near-tie entropy (masked CE there = 0.113 bits/token; coder overhead
−0.013 bits) — the registered "≤ 0.05 b/tok ties aside" bar under-modeled
tie entropy; measured and noted, not coder waste.

Pre-registered: Entropy-coded rate ≤ the directly computed
  cross-entropy on the same windows + 0.05 nats/token (quantized-CDF
  overhead).
- **P35 (determinism).** 0 token mismatches on regeneration for ≥ 3 stories
  × 2 temperatures; flipping any single input bit diverges detectably.
- **P36 (rate dial).** bits/token is monotone in temperature with the two
  limits above; T→0 residual ≤ 0.05 bits/token (ties aside).

Landscape note: this is standard arithmetic-coding ground (LLM-as-compressor,
Delétang et al. 2023; production demos include ts_zip, LLMZip). The point is
*our* decode-synchrony chain in our pipeline as the substrate RQ10–RQ12 plug
into — not novelty of the duality itself.

## RQ10 — The grid-native runtime (rung 3, core)

**Stage A — closure census** (CPU, cheap). Walk a stories260K fitted/QAT
artifact (wae eq7/eq9) under its real activations; measure per-layer
exponent spread s and mantissa span; compare against the bound
s + n + ⌈log₉ d⌉ of §0.

**Stage B — exact-grid forward.** Weights fixed from the QAT-k27 artifact;
runtime runs as digits: integer digit products, exact accumulation,
per-layer power-of-9 renorm, §0's power-of-9 norm and fixed-denominator
rational attention, clamp gates.

**Answer form:** (a) measured span table + bound check; (b) bit-exactness
matrix (torch-CPU vs torch-CUDA vs permuted batch order → 0 logit delta);
(c) paired-val-loss vs the fp-activation arms; (d) activation-digit entropy
study with the exp12 tooling.

**Status (2026-10-07): stage A done — wae eq14, P37-census PASS.** Worst
per-layer exact-accumulate span = **11.2–14.2 base-9 digits** over all four
subjects (eq5 RTN k9/k63 + eq9 QAT k9/k27) at n_act ∈ {1,2,3} — inside the
20-digit bar, u64-class; bignum refuted for this artifact class. The session
decisive number: **residual-stream spread is FLAT across layers** (p99
4.39–4.54 digits at every layer; SwiGLU products flat ~6.2) — the exact
runtime's precision budget is per-layer constant, NOT depth-growing (the §0
linear-growth bound is real but unused: norms reset absolute scale and no op
accumulates relative spread). Op inventory measured for the substitutions:
silu spread map mild (6.6 → 6.9 digits); attention softmax underflows to
exact zero in the median row (the rational attention DEFINES its floor,
~2⁻²⁹ relative); rmsnorm eps 2e-6–2e-4 relative (stage B: exact
zero-vector guard); RoPE angles transcendental (stage B: rational
rotation). Stage B (P37/P38/P39/P40 — the runtime itself) queued behind a
GPU slot. Data-integrity flag: eq9/eq12/eq13 artifacts persist only 2-D
grid tensors (trained 1-D norms never written to disk); census QAT rows
declared as fp16-source-norm proxies.

Pre-registered:

- **P37 (span bound).** Measured per-layer exact-precision need ≤ 20
  base-9 digits (u128-class accumulate, no bignum) on this artifact class.
  A failure pivots the design to block-float renorm and is logged.
- **P38 (exactness).** 0 logit delta across devices and batch orders. This
  is near-theorem for integer paths (exact associativity); the experiment
  verifies no op in the implemented path leaks float.
- **P39 (quality survives substitutions).** With weights fixed at the eq9
  QAT-k27 arm, the substituted-ops runtime lands within +2% of that arm's
  matched-control gap (+1.6%). Registered ablations: **P39a** scale-norm
  (power-of-9) vs RMSNorm swapped alone; **P39b** rational attention vs
  softmax swapped alone. P39 passes only if each ablation stays ≤ +1% and
  the joint stays ≤ +2%.
- **P40 (activation digits carry context).** Activation digit streams are
  NOT iid — an order-1 Markov context model gains ≥ 5% bits over static
  frequency coding. Direct contrast with exp12's weight-stream null
  (0.0–0.2%). This decides whether digit-native memory / .k9-able
  activation checkpointing has any headroom at all.

**Stage-B status (2026-10-07, wae eq16/eq17): P40 PASS · P38 PASS · P37
measured-and-reframed · P39 post-hoc-FAIL with mechanism + P39d registered.**
P40's measured gain: **7.4%** order-1 (bar 5%; exp12's null 0–0.2% — a 40×
gap: activation digit streams genuinely carry context at this artifact
class). P38: the exact-integer runtime (state = (num, den, E, J) per row,
exact gcd-reduction + 2/9-track extraction after every site) is
**accumulation-order-exact end to end** — split-sum reversal leaves
bit-identical logits and width tracks in both int64 and big-int regimes.
P37's registered ≤20-digit bar is refuted for FULL forward passes with
measured numbers: bit-exact fp32 constants percolate ~+27 bits/layer
(logits ≈ 237 base-9 digits ≈ 790-bit carriers); uniform-power grid-rounded
constants (10-bit / 6-bit rests) still land ~55 / ~43 digits at the head —
u128 (40.4) just missed. **The constants' digit width is the dominant
lever on exact-state growth** (the K9 lesson extended from storage to
compute); the compliant bounded form is the regrid-per-layer design
(census: 11.2–14.2 digits), which is a defined rounding step per layer —
so the honest statement of RQ10's question is: *a bounded-state transformer
cannot avoid at least one defined rounding per layer; exactness without it
costs linearly-growing carriers, measured*. P39's post-hoc swaps: norm
+4.4% (with train-slice λ-calibration), attention +23–50% by kernel,
silu +5.7%, joint +210% — the exp11 ternary lesson transfers (post-hoc
collapses, trained-in works): **P39d = substitution-aware QAT** is the
registered next step, plus the exact-vs-fp32ref drift pin (open, not gated;
L0 verified faithful at 1e-3–1e-4).

**Correction (eq18 drift walk, same day):** the drift was CAUGHT — the exact
forward had the residual-rebind bug (normed state reused as the residual;
Rose's own eq2 lesson predicts it verbatim). With the residual carriers
restored and verified faithful at every L0 site (add brute-checked
value-exact: −0.3455 + −0.2002 = −0.5457 = post), the P37 width law is
RE-measured as **exponential, not linear**: with mean-abs norms in the
exact track, the norm's 1/Σ|x| is a coprime denominator factor per
position/layer and the adds lcm-multiply the odd parts (2/9-parts are
track-extractable; odd parts are not) — measured ~+130–160 base-9
digits/layer: logits at ~694–790 digits (2.2–2.5 kbit carriers) *regardless
of the constants' class* (the constants' digit width: a secondary lever —
790/718/694 for fp32-exact/10-bit/6-bit grid rests). The remaining
bounded-exact lever is the norm's algebra: **a scale-norm (pure powers-of-9
resets, carried in the (E,J) tracks alone)** removes 1/Σ|x| entirely and
should hold the track at the matmul-compounding floor (~+2–3 digits/layer,
the census's 21–33-digit class, u64/u128-class carriers) — registered as
the bounded-exact design, with its quality gate folded into P39d's QAT.
So the honest shape of RQ10's answer after stage B: **exactness through a
residual network with arbitrary-rational norms needs exponentially-growing
carriers (measured); bounded carriers require either the scale-norm class
(no denominators) or one defined rounding per layer (the regrid).**

**eq19 (2026-10-07, same day): the bounded-exact design is built, measured,
and it wins by ~130×.** Carrier tournament on the 260K artifact (2×512,
CPU): mean-abs-norm track → 694–790 digits (multiplicative lcm law);
(E,J)-full-extraction → 138 digits (the negative-J 2-drift inflates the
num); plain (num, den) → 228 digits (the 9^K multipliers + odd-lcm
accumulate in the den); **(num, den, E) hybrid with the scale-norm:
5.4 / 5.7 digits — u32-class, no widening events, order-identical logits,
and fp-twin fidelity at every site (logits |Δ| 1.8e-3 on a scale of 68;
zero attention flips)** — which also CLOSES the drift item: the old
21–38 drift was entirely the rebind bug. The number-theory reading: divisors
that are powers of two are free as integers in the den; divisors that are
powers of three are free as shifts on E; every OTHER divisor (Σ|x|, the
constants' odd mantissas) latches a repeating-period cost into every
downstream value — measured at ~+130–160 digits/layer. **A 260K-class
transformer runs its entire linear path exactly on the scaling alphabet
inside ~19-bit carriers.** Remaining: P39d's substitution-aware QAT
(scale-norm + rational attention + grid rests trained in; GPUs free at
session end).

**P39d/e/f (2026-10-07, wae eq20/eq21) — the trained-in decomposition,
measured.** The joint substitution-QAT: 2.0649 (+59% vs eq9's control;
P39d FAIL at its registered +2% bar), artifact identity 3.8e-4 ✓, and the
trained rests now SHIP with the artifact (`eq20_*_rest.npz` — the eq9–13
integrity gap closed going forward). Decomposed by matched arms:

| arm | trained loss | vs the original-ops k27 arm | reading |
|---|---|---|---|
| silu-r (rational gate) | 1.3168 | **+0.19%** | FREE trained-in (post-hoc: +5.7%) |
| quadratic rational attention (c=1) | 1.3509 | +2.78% | nearly recovers exp trained-in (post-hoc: +23–50%) |
| scale-norm (the 9-bucket reset) | 2.2885 | **+74%** | the killer: no magnitude normalization |

**The stage-B tension is now quantified in both directions:** bounded
exactness (scale-norm: u32-class exact carriers) costs +74% trained-in;
RMS-class quality (+0.2–4%) costs exponentially-growing exact carriers
(694–790 digits). Registered next designs: **P39e** — scale-norm + a
learned per-tensor *power-of-nine* gain (free on the E-track; restores a
global magnitude per tensor at zero carrier cost); **P39f** — the
carrier-cheap middle class: how much of RMS's normalization a {9,2}-power
algebra reproduces (trained (m, 2-exp) gains + per-channel rests); and the
exactness-maximal branch: the mean-abs track at python-bigint carriers
(~800-digit state), value exact to the bit.

**eq22 (that same day): the norm-design matrix is measured and complete.**

| norm | exact-carrier law | trained-in (QAT) loss |
|---|---|---|
| RMS (the control class) | exponential | 1.2962 (matched control) |
| **mean-abs + frozen λ** | **exponential (same law)** | **1.3262 = +0.91% — PASS** (post-hoc was +4.4%) |
| free-norm (9-reset + 2-band rms) | **u32-class (5.4 digits)** | 3.3050 = +25% (early derail; bucket switching suspected) |
| scale-norm (9-reset only) | **u32-class (5.4 digits)** | 2.2885 = +74% |

The one-sentence law: **divisors in {2ᵃ, 9ᵇ} are free (den-integers and
E-shifts respectively); every divisor with an odd part latches repeating-
period digits multiplicatively; the mean-abs's odd Σ is the only reason its
carriers blow, and it is also the only norm the model actually wants.**
Registered **P39g — the shipping design**: mean-abs trained in (+0.91%)
with the per-layer regrid onto the K9-native scaled alphabet (n_act mantissa
digits — one DEFINED rounding per layer, not a float artifact): predicted
quality ≤ +2% with bounded 11–14-digit carriers — the K9-native runtime that
lets digit-training, digit-storage, and digit-running hold together with one
disciplined rounding per layer.

**eq23 (2026-10-07, same day): P39g PASSES — measured.** composed arm
(mean-abs + per-site 9-gains frozen at the calibrated init + activation-
regrid n=4 + 10-bit grid rests + digit-STE k27): **trained 1.3235 = +0.70%
vs the original-ops arm (bar +2%)**, artifact identity ~1e-4. The six
decomposing arms nailed the attribution: the loss is INSENSITIVE to the
regrid's n (3→6: +6.86%→+6.67%), insensitive to the rest bits (6→10), and
insensitive to the gains' frozen/trained state — the +6-7% was a BUG in the
equation-draft (the FFN's silu gate missing: a raw GLU), not a property of
the design. So: **the activation-regrid is free; the mean-abs norm trains
in at ≈ +0.7–0.9%; the K9-native runtime is real, quality-gated, and now
has its training recipe.** eq24 (registered): the (mant, e)-state runtime —
the regrid-disciplined exact track as code, carriers predicted ~8–14
digits, value-exact between regrids.

**eq24 RAN (2026-10-07, same day): the regrid-disciplined exact runtime is
faithful end to end — STAGE B CLOSED.** The L0.res_out divergence was an
**int64 wrap at the ff-product** (the fixed-point silu × the w3
mantissa-form reaches ~1.07e19 > int64's max; numpy wraps silently) —
fixed by the escalating integer multiply. End state: every value-site's
|Δ| in fp32-noise class (embed 2.9e-8 → logits 5.0e-2 on a scale of 20.7 =
0.24%, one entry at the flip threshold); **carriers: num 12.6 / den 5.7
digits, u64-class, zero widening events** (vs 694–790 under the
undisciplined norm). The stage-B verdict, final: *train in, store in, run
in — digits everywhere, one defined rounding per layer, order-exact,
machine-word carriers, verified to fp noise.* Registered leftovers: the
int64-GPU device axis; a fp64-twin check to split the score-wobble's
noise-vs-chaos share; then RQ12 (bits-back) and RQ13 (companions).

## RQ11 — The base-9 codec: decode AND encode on the grid

The conversation's design principle, made concrete. The existing coder is
rANS with byte (base-256) renormalization over a stream *sourced* from
base-9 digits — the state arithmetic itself is binary. A grid-native codec
keeps the whole state machine in base 9:

- Frequency table M = 9ⁿ (per-symbol values sum exactly to a power of 9);
  state window L = 9^m. Renormalization emits **base-9 digits**
  (emit x mod 9, x /= 9); the symbol slot is x mod 9ⁿ = the low n base-9
  digits of x — **symbol lookup becomes digit inspection**.
- Consequences:
  1. Codec, model, and artifact speak the same digit algebra — RQ10's
     runtime can consume/emit codec digits with no representation
     conversion.
  2. **Decode ⟹ encode for free**: rANS encode/decode are exact inverses
     of one state-update op — the "if you can do one, you can do the
     other" intuition, guaranteed by construction (the repo already
     round-tripped 1.54B weights at 0 mismatches with the byte variant,
     exp21).
  3. Ties to docs/03 RQ1's packing idea (22 base-9 digits fit a u64).
- Costs to measure: renorm emits log₂9 ≈ 3.17 bits/step, so the inner loop
  runs more renorm steps than byte renorm; frequency tables quantize onto
  M = 9ⁿ (per-symbol error ≤ 1/(2·9ⁿ), negligible for n ≥ 24); C throughput
  vs the measured 278/210 M sym/s byte coder.

**Answer form:** a base-9 rANS coder (C), rate table vs the byte coder on
identical digit streams, throughput, round-trip proof.

**Status (first pass 2026-10-07, exp28 — P41/P42/P43 all PASS).** C core +
Python reference, output-identical parity (the rans_fast house bar).
Constants: M = 9⁴ (6561), L = 9¹², u64 state domain [9¹², 9¹³), renorm
threshold fs·9⁹; symbol slot = the low 4 base-9 digits of the state.
P41 PASS: worst rate delta **+0.0032 b/digit** vs the byte coder over 62
real digit streams (260K QAT + 33M QAT artifacts, including k99 embedding
streams at 6.27 b/digit); M ∈ {9⁴,9⁵,9⁶} rate-identical — granularity
doesn't pay (exp15's EC-SQ lesson re-confirmed). P42 PASS: 190/159 M sym/s
(enc/dec, 33M-real digits) vs byte coder 300/195 = **1.1–1.6×, inside the
2× bar**. P43 PASS: **10⁹ symbols, 0 mismatches**. Container packing: 169
digits → 67 bytes (9¹⁶⁹ < 2⁵³⁶; 0.285 bits/block), pack/unpack asserted.
Two recorded design lessons: (1) the first M = 9⁶ design FAILED P42 at
2.3× decode — the inverse-lookup table fell out of cache (16 KB at M=9⁴ vs
2.1 MB at M=9⁶): **M is a cache knob as much as a granularity knob**;
(2) rate accounting = exact output length (whole digit-emits + block slack
+ 6-byte end state — no byte-granularity discretization anywhere). The
artifact chain is now base-9 end to end: digits in, base-9 digit stream
out, and "decode ⟹ encode on the grid" is exact by construction.

Pre-registered:

- **P41 (rate parity).** ≤ +0.01 b/param vs the byte-renorm coder on the
  same digit streams (entropy accounting per docs/04's discipline).
- **P42 (throughput).** ≤ 2× slower than the byte coder on 9-symbol
  sources; lane interleaving applies to base-9 emits the same way.
- **P43 (round-trip).** 0 mismatches over ≥ 10⁹ symbols — the exp21 bar.
- **P44 (opportunistic rung-2 probe — registered, not expected to pass).**
  A tiny model trained on in-grid codec traces executes one base-9 codec
  step (digit-lookup → state update → renorm emit) with ≥ 95% exact
  renorm-digit accuracy zero-shot on an UNSEEN frequency table. Explicitly
  exploratory per the 2026-10-07 decision ("don't plan on Rung 2 working
  but could").

## RQ12 — The bits-back chain: decoding ends with MORE data (rung 4)

The strongest form of the origin hope — "decode as more data in the end as
a side effect" — is a theorem-anchored mechanism: **bits-back coding**
(Hinton & van Camp 1993; practical as bb-ANS, Rizzo/Townsend et al. 2019;
the relative-entropy-coding literature around it). Requirement: latent-
variable structure, which is exactly what weights-as-equations provides. A
plain autoregressive LM has no posterior/prior gap and gains nothing — the
surplus randomness comes from the posteriors over coefficient-table latents.

Design: give the K9Q1 variable table a coarse diagonal variational
posterior q over coefficient/scale digits (prior p per docs/04's entropy
model). bb-ANS-encode the table under q; decode under p. Two measurements:

1. **Bytes.** Artifact size vs the static-rANS RQ5 baseline. bb-ANS rate →
   log p(x|z) + KL(q‖p); with a coarse q the win may be small — that is a
   measurement, not a failure.
2. **The self-seeding chain (the demo that matters).** The decoder returns
   K recovered random bits *alongside* the weights. Chain: bits file →
   decode (weights + K seed bits) → GPU synthesis (wae eq7) → story seeded
   from the recovered bits. One file carries the model AND the randomness
   its first generation uses; the whole path stays deterministic and
   provenance-perfect.

**Answer form:** bytes table + the reproducible chain demo.

Pre-registered:

- **P45 (size).** bb-ANS artifact ≤ static-K9 baseline + its measured KL
  overhead; strictly smaller only if the posterior is sharper than the
  prior. Registered expectation is parity-to-small-win for a coarse
  diagonal q; a null is informative about where the slack lives.
- **P46 (self-seeding).** The end-to-end chain reproduces the reference
  story for ≥ 3 distinct posteriors; single-bit file corruption always
  diverges detectably downstream.

**Status (2026-10-07, eq25 — RAN).** The bb-ANS chain was built and measured
on this machine's own base-9 state machine (M = 9⁴, L = 9¹²), op order
verbatim from the paper, cross-checked against the reference implementation
(bits-back/bits-back). Subject eq5_C_k63emb99.k9 (265,728 digits); prior p =
the tensors' own M=9⁴ tables; q = p^β, β ∈ {1, 1.5, 2}. Two pop-absorb
instantiations: **clean** (fresh RNG digits, carried by the file) and
**recycle** (the paper's own chaining — the prior appends' leftovers).

- **P45 FAIL — the registered null, with the slack located.** A machine
  identity holds exactly at every β: file Δtrits = the pops' absorb count +
  the realized KL in trits (419,458 + 0; 332,615 + 41,459; 302,728 + 71,479).
  Clean chains: bb = static(186,591 B) + carrier + KL → 352,885 / 334,893 /
  334,945 B. Recycle chains: **parity** (186,593 / 186,626 / 186,459 B,
  Δ = +4 / +89 / −334 trits) with the recycling residue equal to the KL in
  trits (80 / 41,452 / 71,856) — **but their decode diverges** (all three βs):
  the encoder-side recycling leaves the KL-worth of digits in an off-stream
  buffer the decoder cannot see. Net: on a deterministic artifact nothing is
  both static-sized and exactly decodable; "strictly smaller iff sharper" is
  falsified in direction — sharper q only re-splits (fewer pop-absorbs, more
  KL). The slack = the absent conditional structure (the RQ13 c-d posterior /
  a y-conditional likelihood is the upgrade path this ledger now prices).
- **P46 PASS.** Three posteriors: bb roundtrip exact, the mirror returns the
  encoder's x₀, the draws roundtrip, weights value-identical, chain story ==
  reference story (seeded from the recovered draws); one self-contained demo
  file (368,083 B) rebuilds weights + draws; byte-flip corruption 5/5
  detected. Results: weights-as-equations `results/eq25_bits_back.{json,md}`,
  `results/eq25_bb_demo.k9bb`; diary: that repo's RESEARCH_LOG session 15.

## RQ13 — Companion models for codec duty (added 2026-10-07, same conversation)

**Origin line, verbatim:** "What if we got companion models that helped
decode and encode?" The principle: the MAIN model stays a pure digit-flow
transformer (rung 2 unburdened from it per the priority decision); codec-
shaped work goes to small dedicated companions. This is also exactly the
division of labor every modern learned codec already uses (analysis network
+ entropy/context model feeding the range coder — the learned-image-
compression hyperprior shape), so the architecture is externally validated;
what's ours is the digit-native instantiation and the three assignments.

### c-d — decode-side context model

A small net predicts **conditional frequency tables at chunk granularity**
(per group of symbols / per context step); the rANS state machine consumes
them at symbol speed. Targets, by the measured evidence: generation token
streams (obviously context-dependent — the strongest case); activation
digit streams (gated on P40); weight digit streams — **expected gain ≈ 0**,
exp12's iid null extended; registering that expected null keeps a surprise
loud if it comes.

**Status (2026-10-07, eq26 — RAN). P48 PASS at 50.74% held-out (bar 20%).**
Streams: the trained artifacts' weight digits, eq16/P40's own activation
digit representation (fp + substituted-norm variants), and the generation
recipe's token stream. Companions: order-1/2 Markov with Jelinek-Mercer
interpolation to a KT order-0 base (chunk-table shape), plus a small GRU;
strictly held-out (tables from the first 80%, each λ fitted on a validation
slice, rate on the last 20%). Machinery cross-check: P40's estimator on the
identical streams reproduces eq16 exactly (7.41/7.35). Findings: the
activation gain is **second-order** and concentrated at layer 0's
pre-attention norm output (the token-identity-bearing stream): static 5.372
→ order-1 5.021 → order-2 2.646 b/sym (λ = [0.95, 0.95]); all other sites
0.4-3.5%; GRU 42.2%. Weight digits: aggregate 0.50% (exp12's null extended
through order-2), one flagged surprise (`tok_embeddings.weight`, 2.75% > the
2% loud bar — flagged, not claimed). Tokens: 7.042 → 4.450 b/tok (+36.8%)
vs the LM's own 1.392 b/tok (80.2% ceiling). Consequence for RQ12: the c-d
posterior halves the runtime's intermediate digit streams, while the weight
payload route stays closed — eq25's ledger is where that leverage is spent.
Results: weights-as-equations `results/eq26_context_companion.{json,md}`;
diary: that repo's RESEARCH_LOG session 16.

**P47 fix-path status (2026-10-07, eq30 — RAN; bar still unmet, floor
quantified).** Both eq27-identified costs were removed in C
(`eq30_ctx_fast.c`): the per-context inverse tables are gone (a branchless
binary search over the alpha-sized cumulative array: ~1 KB per context
instead of 26 KB) and the companion lives in C with counts updated
O(1)/symbol and every seen context's table rebuilt eagerly per refresh
boundary (a tagged lazy cache evicted and broke the roundtrip at R > 1k —
isolated on synthetic streams and traced to traversal-order dependence; making
the table a function of (ctx, generation) only fixed it, all sweeps digit-
exact). The refresh period R is a free parameter (causality needs a prefix),
so it was swept. Measured frontier (bytes | M sym/s | % of the byte coder,
237/231 M sym/s): activation stream R=1024 92,024 | 31.2 | 13% → R=65536
126,983 | 41.3 | 17%; embed stream R=1024 25,306 | 37.9 | 16% → R=65536
27,352 | 68.7 | 30%. **P47 FAIL at best 17.4%/29.7%.** Findings: the build
cost was real and amortizes with R (31 → 41 M/s) but the rate pays for large
R (the knee is R=4096: +46.8% bits at 36 M/s); and the residual floor is the
per-symbol context path itself — base9's plain single-table coder reaches
161-188 M/s (68-81% of the byte coder), so the context fold + table selection
+ slot search cost 3-4× the state machine, and the binary search did not beat
eq27's inverse tables. The bar needs a cheaper context mechanism: context
clustering into a few hot tables, a position-class context, or a specialized
datapath. Files: weights-as-equations `results/eq30_p47_fix.{json,md}`,
`experiments/eq30_ctx_fast.c`, `experiments/eq30_p47_fix.py`; diary: session
20.

**P47 clustering frontier (2026-10-07, eq31 — RAN; bar still unmet at
37-46%, residual identified by ablation).** eq30's engine + context
clustering (ctx→cluster array, K pooled tables, deterministic assignment from
the prefix counts, refreshed at R=4096). Two decisive results: (1) the
working-set hypothesis is REFUTED — with the branchy search, K=1 (one ~1 KB
table, perfectly L1-resident) ran at 32.5 M sym/s, SLOWER than base9's plain
coder with a 26 KB inverse table (162-188); a tiny table cannot be
cache-bound. (2) Swapping in a per-cluster branchless inverse table for the
binary search nearly tripled the throughput (activation 32.5→90.3 M/s, embed
41.8→105.3 = **37.5%/46.0% of the byte coder**), so the data-dependent search
branches were the residual (~20-30 cycles/symbol). The frontier trades rate
against throughput in the wrong direction for the cheap top-1-argmax
clusterer: useful rate wins need K≥64 (K=16: −11.2%, K=64: −23.3% vs the
static table) by which point K×26 KB spills L2 and the throughput falls back
to ~30%. Named next: a distribution-aware clusterer (k-means) for the
frontier's middle. Engineering reframing: these engines decode at 33-105 M
sym/s, so a 1.5B-class k9 artifact (~1.5G digits) loads in ~15-45 s with
context coding on — the registered 50%-of-byte-coder bar is stricter than the
load-time application needs. Files: `results/eq31_cluster.{json,md}`,
`experiments/eq31_ctx_cluster.c`, `experiments/eq31_cluster.py`; diary:
session 21.

**P47 closes (2026-10-07, eq32 — RAN; bar out of reach on CPU, measured
ceiling 21–47%).** Three deterministic context keys × K ∈ {4,8,16,32} at
R=4096 with the branchless inverse-table decode: top-1 argmax rank (eq31's
key), k-means on the context distributions, and a p2-bucket key (one
L1-sized lookup). All roundtrip digit-exact; the control reproduced eq31 to
the byte. Frontier (activation stream: static 175,975 B, byte coder 240
M/s; embed: 24,749 B, 225 M/s): the best throughput ratio is k-means K=4 at
**40.2% / 47.1%** (with ~no rate win), and the best useful rate
(−11.2% at K=16) sits at 31.3%. eq31's K=1 control bounded the family at
37.5%/46%, so the 50% bar is unreachable for a per-symbol context coder on
this CPU regardless of clusterer. Two lessons recorded: clustering for
compression must use the CODING cost (CE/KL) as the distance — my L2-on-raw-
probabilities k-means codes WORSE than the cheap key at the same K
(activation K=16: 166,880 vs 156,280 B) — and the weight-digit null
reproduces in the coder (every key/K codes the embed 5.4% worse than the
stored static table: the collapsed λs buy nothing and pay the prefix tax).
Framing: 33–106 M sym/s = a 1.5B k9 artifact loads in ~15–45 s with context
coding on; the bar is stricter than the load-time path needs. Files:
`results/eq32_kmeans.{json,md}`, `experiments/eq32_ctx_kmeans.c`,
`experiments/eq32_kmeans.py`; diary: session 22.

**eq27's P47 run (the diagnosis this fix-path started from; session 17):**
codec = per-chunk independent base-9 rANS segments (a
prefix-derived companion forces stream-order decoding, which a single LIFO
stack forbids; the 6-byte per-chunk end state costs 0.047 b/sym at 1k).
Rate half lands the eq26 win in the real coder: activation stream 175,975 B
(stored static table) → **91,586 B (+47.96%)**, companion chosen in 251/256
chunks, tables free (prefix-derived); the embed digits +10.65% by
prefix-adaptivity only (their order-2 lambdas collapse, per eq26's null).
Throughput: byte coder 243 M sym/s, base9 plain 163, ctx@1k **38 (16% of
byte, bar 50%)**, ctx@4k 36, whole-stream single call 43 — the cadence is
not the bottleneck; the 62 MB context-table working set (26 KB inverse per
context) is, plus the companion build at 134 us/symbol (~5,100x the loop).
Fix path named: cache-resident context decode structure + a C-side
incremental companion build. Files: weights-as-equations
`results/eq27_chunk_throughput.{json,md}`, `experiments/eq27_ctx_coder.c`,
`experiments/eq27_chunk_throughput.py`; diary: session 17.

### c-e — encode-side amortizer

A hypernetwork predicts in one forward pass what today costs optimization:
scale-fit values (eq6's 400 Adam steps per artifact) and GPTQ-quality digit
placement (exp6/exp18's calibration). Amortization converts per-load fits
and per-encode calibration into inference, making "fitted artifact" a load-
time property rather than a training-time one.

### c-x — the codec executor (rung 2, specialized)

The dedicated tiny model executes base-9 codec steps bit-exactly on-grid
(the P44 question) — but as a *specialist*: no general reasoning demanded,
main model untouched, and the same core runs inverted for encode because
rANS encode/decode share the state-update op. **P44 vs P50 is itself an
experiment**: identical question, executor differs only in identity
(generalist main model vs dedicated companion) — the specialization contrast
is a finding either way.

### The load-bearing engineering constraint

Companions must NEVER sit in the per-symbol inner loop: the loader decodes
at 210 M sym/s (exp21/23, streaming tensor-by-tensor — chunk-granularity
companion tables slot into that loop naturally; a neural forward pass per
symbol would strangle it). Companions predict distributions in chunks; the
state machine consumes them at symbol rate. Corollary for RQ12: the
companion posterior (a learned q) is the stronger upgrade over "coarse
diagonal q" — same recovered-bits mechanism, likely more bits.

Packaging note: companions are trained models too, so they ship as K9Q1
artifacts like everything else — the chain stays all-digits end to end.

**Answer form:** throughput table (companion-augmented decode vs the byte
coder), a paired amortized-encode comparison against optimizer GPTQ/scale-
fit, the executor-accuracy A/B (P44 vs P50).

Pre-registered:

- **P47 (granularity keeps the rate).** Companion-augmented decode
  maintains ≥ 50% of the byte coder's throughput with companions predicting
  per-group/per-block tables (chunk ≥ 1k symbols), not per-symbol.
- **P48 (context pays where context exists).** On generation-adjacent
  activation digit streams (IF P40 passes), the companion context model
  recovers ≥ 20% of the static-table bit cost. On weight digit streams the
  registered expectation is ≈ 0 (exp12 null class).
- **P49 (amortized encode).** Companion-predicted placement/scales recover
  ≥ 80% of GPTQ's gap reduction on the 1.5B k9 point (exp6: 54.7→43.6) at
  ≥ 100× less compute than the optimizer run, within +0.05 ppl of
  optimizer-GPTQ.

**Status (2026-10-07, eq29 — RAN). PASS, with a flagged surprise.** Subject
correction first: exp6's 54.679→43.644 are **TinyStories-33M** (the doc's
"1.5B k9 point" attribution is a slip; the 1.5B's own exp18 delta is −0.45
code-ppl, not re-measured). Both sides re-run under one harness: exp6's GPTQ
fresh (45.181; the recorded 43.644 carried alongside) and the amortizer = ONE
diagonal calibration pass + a per-group scale search on a 24-point ladder by
the activation-weighted group error (+ a learned companion MLP). Anchors
reproduce exactly: fp32 40.663, RTN 54.679. The best amortized point (K=4
calibration blocks, 0.11 s) reaches **ppl 40.753 — 146.6% of the in-harness
gap (126.2% of the recorded), 4.43 ppl BETTER than GPTQ, at 347× (2,500× vs
the recorded wall)**; all three clauses pass (the +0.05 clause read
one-sided: no worse than GPTQ). Cross-check: exp4b's FITTED Lloyd-Max
9-level point sits at 40.534 — within the noise floor of the amortizer, but
Lloyd-Max pays 4.5 b/param for its stored codebook vs the amortizer's 0.5
(eff 7.27 vs 3.20 b/param): the codebook's quality is reachable by deriving
the per-group scale in one pass, nothing stored (heavy-tailed groups: finer
step + clipping spends the nine levels on the bulk). The learned MLP form
gives 43.111 (121.8%) after a 1.5 s one-off training (unseeded init: ±~0.6
ppl spread between runs). Files: weights-as-equations
`results/eq29_amortizer.{json,md}`, `experiments/eq29_amortizer.py`; diary:
session 19.
- **P50 (specialist executor).** The dedicated companion reaches ≥ 95%
  exact renorm-digit accuracy zero-shot on an UNSEEN frequency table —
  P44's threshold with a dedicated model; the contrast pair (P44 generalist
  vs P50 specialist) is the registered specialization experiment.

**Status (2026-10-07, eq28 — RAN).** Task (identical for both executors): one
base-9 rANS push+renorm step — given x (13 base-9 digits) and the symbol's
frequency f (4 base-9 digits), emit the renorm digits (0–4, MSD-first);
metric = the EXACT step. P50: a tiny transformer (2 layers, d=96, trained 6k
steps on 24 train tables) reaches **99.60% exact on unseen tables** (bar 95%
PASS; count 99.70%, digits 98.73%) — but **38.36%** on a strict regime with a
DISJOINT f-range (801–1500 vs the trained 100–800): the net interpolates the
f-range instead of computing the comparison as a function of f. P44: Qwen2.5-
Coder-7B-Instruct f16 (local llama-server, greedy, 8 in-context train-table
examples) reaches 69.3% exact on the same cases — its emitted COUNT is right
95.0% while the digits fail (MSD/LSD confusion) — and 41.7% in the strict
regime. Contrast (identical 300 cases): 99.3% vs 69.3% in-distribution;
43.3% vs 41.7% out-of-distribution: the specialist wins where it was trained,
the generalist is the more OOD-robust, neither is codec-ready — and P50's
PASS is operationalization-sensitive (the registered "unseen table" reading
vs a genuinely unseen f-range). Files: weights-as-equations
`results/eq28_executor_ab.{json,md}`, `experiments/eq28_executor_ab.py`;
diary: session 18.

## Status and sequencing

All RQs in this doc are PROPOSED — nothing has run. Sequenced against the
live queue:

- CPU-safe, ready when a slot opens: **RQ10 stage A** (census),
  **RQ11** (C coder), **RQ9** (inference-only on any served model).
- RQ10 stage B wants short GPU blocks (260K-class, minutes).
- **RQ12** RAN 2026-10-07 (eq25, weights-as-equations): **P45 FAIL** (the
  null with the slack located: the pop channel is a carrier on a
  deterministic artifact; parity is reachable but not decodable — see the
  RQ12 status block), **P46 PASS** (the self-seeding chain over 3
  posteriors). The companion posterior (RQ13 c-d/c-e) is its registered
  upgraded q, now priced by eq25's ledger.
- **RQ13** companions pair up: c-d/c-e gates on P40 and existing fit
  machinery respectively; c-x rides RQ11's base-9 state machine (companions
  must emit tables summing to M = 9ⁿ — a small projection step).

Landscape honesty: Delétang et al. 2023 (LLM = compressor); ts_zip
(Bellard); LLMZip (2024); the bb-ANS/bits-back line; learned-codec range
coders (Ballé/Minnen) — whose **context-model architecture is the external
anchor for RQ13's division of labor**; BitNet and integer-only inference
(Q8BERT line); rational activations (Padé activation units). Per the
docs/02 convention, a web-verified novelty pass must confirm the gap BEFORE
any "nobody has done this" claim. Working hypothesis of the new ground: the
digit-native closure runtime with exact base-9 renormalization, the base-9
state-machine codec, the self-seeding bits-back chain on weight-table
latents, and companion models carrying codec duty at chunk granularity.