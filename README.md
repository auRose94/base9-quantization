# base9-quantization

**Ninths-grid model quantization, its runtimes and published models — and the
analog-computing seam it grew into.** Origin idea by Rosemary Mercury
(2026-10-03, kept verbatim in `docs/00-idea-origin.md`): quantize a model so
it "fits in the repeating numbers," using a base-9 digit system, as a
model-compression mechanism. Developed and tested with ZCode. MIT licensed
(`LICENSE`).

**Naming:** the format is **ninths** (primary name); **K9** is the secondary
shorthand and the GGML type prefix of the native resident containers
(`K9_4``/`K9_6`/`K9_7`); `.k9` stays the extension of the entropy-coded
(rANS) research container. The fork's loader KV is `ninths.directory`
(legacy `k9.directory` still loads).

One-line pitch, current: a repeating-radix weight grid whose entropy-coded
containers reach q8_0-parity quality at roughly two-thirds the size — proven
end-to-end from number theory to bit-exact runtimes to published models on
three architectures — now extended to the question *which model survives an
analog computer*, where its quantization and noise-aware training turn out
to be the enabling software (`docs/15`).

## What this project is (2026-11)

Six connected pieces, each with a measured result:

1. **The container format.** `.k9` rANS containers (spec `docs/04`, stream-v2
   `docs/11`, native-resident spec `docs/12`): per-tensor k-family grids with
   exact dequantization. The headline tier — body **k63 + embed99** — is
   ppl-parity with f16/q8_0 at a third to a half smaller, verified
   end-to-end in llama.cpp:

   | model | variant | file GB | ppl (wikitext-2 test) |
   |---|---|---|---|
   | Qwen2.5-Coder-1.5B | f16 | 3.09 | 13.88 ± 0.110 |
   | Qwen2.5-Coder-1.5B | **K9 k63 + e99** | **1.12** | **13.90 ± 0.110** |
   | 7B tuned | q8_0 | 8.10 | 9.231 ± 0.064 |
   | 7B tuned | **K9 k63 + e99** | **5.47** | **9.253 ± 0.065** |
   | 14B tuned | q8_0 | 15.70 | 7.805 ± 0.052 |
   | 14B tuned | **K9 k63 + e99** | **10.49** | **7.823 ± 0.052** |

2. **Native runtimes.** A public llama.cpp fork (`k9` branch at
   [auRose94/llama.cpp](https://github.com/auRose94/llama.cpp), upstream
   discussion #31293) decodes `.k9` on load with bit-exact gates; a
   byte-identical KoboldCpp port exists (`docs/07`). Serving works on CUDA
   and HIP with MTP speculative decode tuned per model (27–30 t/s on the
   published 27B; `glm_k9_serve.sh` carries the defaults).

3. **Published models** (Hugging Face, `auRose94`):
   [GLM-4.5-Air k63/e99](https://huggingface.co/auRose94/GLM-4.5-Air-ninths-k63-e99-GGUF)
   (73.45 GiB) plus a **stock-compatible iq4_nl export** that loads in
   unmodified llama.cpp/Ollama (59.1 GiB);
   [Qwen3.8-27B](https://huggingface.co/auRose94) in both container and
   stock-compatible forms (18.31 / 15.55 GiB, hybrid SSM+attention arch);
   flint-40B as the portability test, including published extreme rungs down
   to ~1.1 b/param. Three architectures now build with zero pipeline
   changes.

4. **The Godot-4 coder** (`phase2/`): verified SFT corpora (every sample
   passes the real Godot 4 parser), a fine-tune ladder judged mechanically
   (12.5% base → 83.3% at 7B → 79.2% draws-3 at 14B), tuned models shipping
   in K9 containers at the base model's size, and an agent rung
   (`experiments/agent_k9.py`) that edits multi-file GDScript projects over
   llama-server with self-checks.

5. **FPGA bit-exact validation** (`fpga/`): Tang Nano 20K as a silicon
   validator for digit-native decode — gates green on real hardware, with a
   sim-vs-silicon bug ledger and an SDRAM controller for the memory tier.

6. **The analog survival rung** (`docs/15`, newest): what happens to a model
   when its weights live on analog compute-in-memory hardware. A write-path
   channel simulator (range-referenced noise, per-group gain calibration,
   drift/IR/stuck/levels axes) plus **noise-aware training** produced the
   project's current best results: mixed-σ training gives the best model on
   every axis tested, stuck-aware training rescues the one failure mode that
   killed every encoding, and the methodology discipline (paired draws +
   full corpus + multi-seed) rescinded four of the rung's own early claims.
   Hardware characterization rung 2.5 (digipot/OTA/ADC, mock-gated harness)
   is built and waiting on parts. This direction came from the author's
   observation that a quantization scale is an analog gain cell in disguise.

## The idea in one minute

Two instincts are two views of one grid:

1. **Repeating numbers.** Every infinitely repeating decimal is rational.
   Quantize weights to multiples of one ninth and every weight becomes a
   *single repeating decimal digit* — 4/9 = 0.4444… in base 10, 0.4 exactly
   in base 9. The whole model is a string of base-9 digits.
2. **Base 9 = two ternary weights.** Two BitNet-style trits have exactly
   3 × 3 = 9 joint states; log2(9) = 3.1699 = 2 × log2(3). Base-9 is to
   balanced ternary what hexadecimal is to binary.

The grids generalize (b^L−1 levels: 9, 15, 27, 63, 99…), entropy coding
pays for non-power-of-two alphabets, and the measured outcome is the table
above. The full derivation, pre-registered predictions and per-session
verdicts live in `docs/01`, `RESEARCH_LOG.md`, and `results/`.

## History

The first five weeks — synthetic grids, TinyStories-33M, the Qwen 1.5B
frontier against every deployed format, the C rANS codec, the 7B/14B
scaling story, GPTQ's small-model-only effect, the asymmetric-grid null —
are recorded as 26 numbered findings with tables in this README's git
history and, permanently, in `RESEARCH_LOG.md` and `results/`. Headlines
that still stand: k63/e99 is near-lossless at q4_k_m's size (beats it by
+0.8 ppl at 1.5B); the entropy-coded low-bit points dominated q2_k/NF4 at
1.5B but become a size-vs-quality trade at 7B; GPTQ compensation does not
transfer to 7B; asymmetric grids don't pay at matched bytes; repeating-decimal
*notation* adds no compression — the identity is representational, the win
is the odd-sized alphabet plus rANS.

The negatives are logged with the same rigor as the wins — including, in
`docs/15`, the rescission of four of that rung's own early results when a
better methodology showed them to be noise-draw artifacts.

## Repository map

    docs/00–15                     idea origin → hypothesis → specs → rung fact
                                  sheets (05 fork, 07 koboldcpp, 11 stream-v2,
                                  12 container spec, 13 calibration/v3, 14 strata,
                                  15 analog survival)
    experiments/                   self-asserted scripts (see its README); analog_*
                                  + train_noiseaware.py + synapse_char.py = the
                                  analog rung
    phase2/                        Godot-4 corpus, verifier, coder fine-tunes
    fpga/                          Tang Nano 20K validator + SDRAM work
    rs/                            Rust codec workspace (byte-exact vs python)
    probes/                        one-off measurement probes
    results/                       CSV/MD outputs of every experiment
    chat_k9.py                     local chat UI, K9 vs bf16 A/B
    agent via experiments/agent_k9.py + lean-ctx MCP
    RESEARCH_LOG.md                dated log: predictions, verdicts, surprises

## Running

Python 3 + numpy gets you the pure-research experiments; see `experiments/`:

    python3 experiments/exp15_alphabet_sweep.py     # EC-SQ ceiling (numpy only)

The analog rung's no-hardware gate (mock firmware, recovers known ground
truth):

    .venv-glm/bin/python experiments/synapse_char.py --mock --gate

Chat with the quantized models (`chat_k9.py --preset 7b`), or serve the
published containers with the fork's `llama-server` (see `docs/05`).

## What is not in this repository

Everything in git is source plus small text/CSV results. The multi-GB
products are excluded and either rebuildable (see `.gitignore` and
`experiments/`) or **published on Hugging Face** (`auRose94` — see piece 3
above). One `.gitignore` trap documented for posterity: git only treats `#`
as a comment at the start of a line, so an inline comment silently becomes
part of the pattern — which is how you get a 17 GB `git add`.

## Data provenance

The Phase 2 corpus is **built, not redistributed**: `phase2/build_corpus.py`
downloads `wallstoneai/godot-gdscript-dataset` (Apache-2.0) and extracts the
Godot-4 half; derived data stays out of git (`phase2/README.md`).

## License

MIT — see `LICENSE`.

## Credits and acknowledgements

The thanks here are not a formality — this work is built on other people's
research:

- **Jarek Duda** created **rANS** and released it **patent-free**. The K9
  codec and every entropy coder in this repository are his algorithm.
- **Cramer, Billaudelle, Schemmel, Zenke et al.** — "Surrogate gradients
  for analog neuromorphic computing" (PNAS 2022): the in-the-loop training
  architecture `docs/15`'s noise-aware results independently converge with,
  on real analog silicon.
- **Frantar et al.** for GPTQ; **Karpathy** for llama2.c; **Eldan** for
  TinyStories; **Qwen/Alibaba**, **Zhipu (Z.ai)** and the flint team for the
  base models; **Georgi Gerganov and the llama.cpp contributors** for the
  inference stack behind the gates; **Carver Mead** for the analog-VLSI
  lineage the appliance direction stands on.

The related weights-as-equations branch lives at
[auRose94/base9-quantization @ `weights-as-equations`](https://github.com/auRose94/base9-quantization/tree/weights-as-equations).

## How this was made — an AI-assisted research project

This research was done by **Rosemary Mercury** with an **AI assistant —
GLM running inside ZCode**. The assistant wrote and ran experiments,
measured and logged results, and drafted docs; the ideas, priorities,
pre-registrations, and review came from the author. Predictions are
pre-registered before they are tested, and failures — including the
rescinded ones — are logged as findings rather than removed.

> "I only had an idea and my credit isn't important... It's not a waste,
> for now it proves a lot. It just comes at a severe cost to make data more
> packed, and it's not a silver bullet for optimization but a packing
> decision." — the author, in her own words (full note in the
> weights-as-equations README)
