# 02 — Related work (web-verified 2026-10-03)

Bibliographic details below were verified against arXiv/collector pages by a
web-search pass on 2026-10-03; two entries it could not fully confirm are
marked [UNVERIFIED]. This file is the citation backbone for the project;
`docs/01-hypothesis.md` §4 says what is *already established* by this list
and what this project can still claim.

## Ternary / 1-bit weight training

- **BitNet b1.58** — "The Era of 1-bit LLMs: All Large Language Models are in
  1.58 Bits" — Shuming Ma et al. — 2024 — arXiv:2402.17764. Ternary
  {-1, 0, +1} LLM weights "at 1.58 bits" (= log2 3); the direct precedent for
  ternary LLM quantization, and the training-side anchor of this project's
  ternary-pair → base-9-digit bridge.
- **BitNet** — "BitNet: Scaling 1-bit Transformers for Large Language Models"
  — Hongyu Wang et al. — 2023 — arXiv:2310.11453. BitLinear layer; base
  architecture.
- **Ternary Weight Networks** — Fengfu Li et al. — 2016 — arXiv:1605.04711.
  Scaled {-a, 0, a} ternarization via threshold minimization; classical
  ternary-weights baseline at the root of BitNet.

## Entropy coding of (quantized) weights

- **Deep Compression** — "Deep Compression: Compressing Deep Neural Networks
  with Pruning, Trained Quantization and Huffman Coding" — Song Han, Huizi
  Mao, William J. Dally — 2015, ICLR 2016 (oral) — arXiv:1510.00149.
  Establishes quantize-then-entropy-code as standard practice — the reason
  this project's compression claims must be scoped beyond plain Huffman.
- **Asymmetric Numeral Systems** — Jarek Duda — 2009 — arXiv:0902.0271; the
  widely-cited sequel "Asymmetric numeral systems: entropy coding combining
  speed of Huffman coding with compression rate of arithmetic coding" — 2013
  — arXiv:1311.2540. The fractional-bit coder family this project's rANS
  experiments (exp1) are built on.
- **ZipNN** — "ZipNN: Lossless Compression for AI Models" — Moshik
  Hershcovitch et al. — 2024 — arXiv:2411.05239 (IEEE Cloud). 33–50%
  lossless savings on model weights; successor "Lossless Compression of
  Neural Network Components: Weights, Checkpoints, and K/V Caches in
  Low-Precision Formats" — Anat Heilper et al. — 2025 — arXiv:2508.19263.
- **DeepCABAC** — "DeepCABAC: A Universal Compression Algorithm for Deep
  Neural Networks" — Simon Wiedemann et al. — 2019/2020 — arXiv:1907.11900,
  IEEE JSTSP (DOI 10.1109/JSTSP.2020.2969554).
- **Approaching Shannon Bound with Lossless LLM Weight Compression** —
  Hongshi Tan et al. — ISCA 2026 — arXiv:2606.15789. Applies ANS explicitly
  to LLM weight tensors — closest current systems work to this project's
  coder layer.

## Codebook / vector quantization at low bits

- **AQLM** — "Extreme Compression of Large Language Models via Additive
  Quantization" — Vage Egiazarian et al. — 2024 — arXiv:2401.06118 (ICML
  2024). Multi-codebook additive VQ for 2–3-bit LLM weights.
- **GPTVQ** — "GPTVQ: The Blessing of Dimensionality for LLM Quantization" —
  Mart van Baalen et al. — 2024 — arXiv:2402.15319. Hessian-guided
  column-wise VQ.
- **QuIP#** — "QuIP#: Even Better LLM Quantization with Hadamard Incoherence
  and Lattice Codebooks" — Albert Tseng et al. — 2024 — arXiv:2402.04396
  (ICML 2024). E8-lattice codebooks ≤ 4-bit. [UNVERIFIED detail: Huffman
  entropy coding of residuals appears in the paper body, not the abstract.]
- **APoT** — "Additive Powers-of-Two Quantization: An Efficient Non-uniform
  Discretization for Neural Networks" — Yuhang Li et al. — 2020 —
  arXiv:1909.13144. Non-uniform (dyadic-sum) level sets — the closest prior
  to "grids that are not plain uniform".

## Arbitrary / non-power-of-two bit-widths

- **eXmY** — "eXmY: A Data Type and Technique for Arbitrary Bit Precision
  Quantization" — Aditya Agrawal et al. — 2024 — arXiv:2405.13938.
  Non-power-of-two widths/encodings incl. 5/6/7-bit and small alphabets.
- **Any-Precision LLM** — Yeonhong Park et al. — 2024 — arXiv:2402.10517
  (ICML 2024). Stacked post-training quantization at arbitrary bit-widths.
- **FracBits** — "FracBits: Mixed Precision Quantization via Fractional
  Bit-Widths" — Linjie Yang, Qing Jin — 2020 — arXiv:2007.02017 (AAAI 2021).
  Fractionalizes bit *widths*, not weight *values* — adjacent but distinct
  from rational-value grids.

## Post-training quantization with error compensation

- **GPTQ** — "GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers" — Elias Frantar, Saleh Ashkboos, Torsten Hoefler,
  Dan Alistarh — 2022, ICLR 2023 — arXiv:2210.17323 (verified on arXiv
  2026-10-03). Column-wise quantization with Hessian-based error propagation;
  the established "more math for better PTQ" baseline. exp6 applies its
  machinery (CPU-float64 dampered Cholesky path) to this project's grids.

## Shift-based and alternative numerics

- **DeepShift** — "DeepShift: Towards Multiplication-Less Neural Networks" —
  Mostafa Elhoushi et al. — 2019/2021 — arXiv:1905.13298 (CVPR Workshops
  2021). Shift/sign (power-of-two) weights.
- **Posits** — "Beating Floating Point at its Own Game: Posit Arithmetic" —
  John L. Gustafson, Isaac Yonemoto — 2017 — Supercomputing Frontiers and
  Innovations 4(2), DOI 10.14529/jsfi170206. The also-cited "Posits: the
  alternative to IEEE 754 floating-point standard" (CoNGA 2017) title is
  [UNVERIFIED].
- **Balanced ternary (classical)** — Donald E. Knuth, The Art of Computer
  Programming Vol. 2: Seminumerical Algorithms, 3rd ed., Addison-Wesley,
  1997, pp. 195–213; Brian Hayes, "Third Base", American Scientist 89(6):
  490–494, 2001.

## The gap this project targets (verified by search, 2026-10-03)

arXiv phrase searches for "rational quantization" / "rational weights"
surface only *theory* papers that assume rational weights for
computability/verification results — none train or quantize neural-network
weights on small-denominator rational grids such as multiples of 1/9.
Concretely, three under-explored angles:

1. **Alphabet size as a free design dimension** — codebooks/alphabets whose
   size is not a power of two (exactly 9 levels, 27 levels, ...) evaluated
   at entropy-matched bits. Hardware-motivated literature steps 2→4→8→16
   levels and skips the middle.
2. **Rational-value grids** (multiples of 1/9) as a *quantization target* —
   apparently untouched by the training/PTQ literature.
3. **Joint grid + coder design** — choosing the value grid and the entropy
   coder's alphabet together so code length is optimized inside the
   quantization objective, rather than applying generic entropy coding
   after the fact (the Deep Compression pattern).