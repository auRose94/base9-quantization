---
license: apache-2.0
base_model: RecursiveSI/flint
tags:
- quantization
- moe
- gguf
- ninths
- k9
- experimental
---

# Flint 40B-A2.5B — ninths extreme rungs (k5 ≈ IQ2_S-class, k3 ≈ IQ1_S-class)

Two deliberately extreme ninths (k9) stream containers of
[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), built on the same
byte-exact pipeline as [`auRose94/Flint-40B-A2.5B-ninths-k63-e99-GGUF`](https://huggingface.co/auRose94/Flint-40B-A2.5B-ninths-k63-e99-GGUF).
They exist for size/quality variety and statistics: both go through the identical
container format (k99-exact embeddings/head, ranks/norms/sinks copied verbatim),
differing only in the body grid. **No calibration is applied on this rung — this
is a pure RTN grid**, and the measurement below is the honest result of that.

These are *not* usable chat models. They load, serve, and answer (the container
machinery is arch-agnostic and byte-verified at every level), but the uncalibrated
coarse grids destroy flint's language modeling. The full quality ladder we
measured on the same 300k-token wiki2 window:

| rung | file | bits/param (quantized) | ppl (wiki2 300k) | KL protocol PPL(Q)/PPL(base) |
|---|---|---|---|---|
| bf16 (ggml-org, ref) | 74.2 GiB | 16 | (base) | 1.0 |
| Q8_0 stock | 39.44 GiB | 8.50 | 22.87 ± 0.51 | 1.0395 |
| **ninths k63/e99** (see its repo) | 26.54 GiB | 5.71 | 23.68 ± 0.54 | 1.0811 |
| Q4_K_M stock | 21.78 GiB | 4.85 | 23.74 ± 0.54 | 1.0628 |
| IQ4_NL re-quant | 21.21 GiB | 4.28 | 27.56 ± 0.64 | 1.2267 |
| **ninths k5/e99** (this repo) | **9.13 GiB** | **1.95** | 149,932 | ~4400× (ln 8.394 ± 0.091, Cor 31.0%) |
| **ninths k3/e99** (this repo) | **5.23 GiB** | **1.11** | 2.58×10¹⁴ | — |

What the two rungs demonstrate:

- The container format and its gate chain are grid-agnostic: k5 and k3 pass the
  same byte-exact digits/scales/materialization gates as k63 (0 mismatches every
  time), and both arms reproduce identical perplexity whether loaded through the
  ninths decode-on-load path or through a stock materialized Q8_0 twin of the
  same weights.
- **The cliff is the grid, not the container**: somewhere between 5.7 bits and
  ~2 bits per weight, a per-64-group RTN grid on this model transitions from
  "minor degradation" to "language model destroyed" — an RTN k5 grid sends about
  half of a Gaussian weight distribution to 0. Recovering this range needs a
  calibration-aware pass (act-order/GPTQ, learned codebooks like the IQ
  ladder's, or the imatrix machinery upstream requires for its IQ2_S/IQ1_S
  types) — that is future work on our side; upstream's own IQ quants refuse to
  build without an importance matrix for good reason.

## What's inside

- `flint_k5_k99_v2.gguf` — 9.13 GiB (k5 body: 5 symmetric levels, k99 exact
  embeddings + head)
- `flint_k3_k99_v2.gguf` — 5.23 GiB (k3 body: 3 levels — for completeness;
  the far edge)
- `chat_template.jinja` / `chat_template_llamacpp.jinja` — as in the k63 repo

## Runtime requirement

Ninths fork of llama.cpp, branch `k9` (≥ `c5029ebd4`, which also syncs the fork
with upstream master): `github.com/auRose94/llama.cpp`. Build with
`-DGGML_HIP=ON` for AMD (tested on gfx1151) or CPU. Decode-on-load materializes
to Q8_0 (~39.5 GB of resident weights for both rungs — the size saving is in
the file distribution, not in runtime RAM).

## License

Apache-2.0 (inherited from the base model). Pipeline and format:
https://github.com/auRose94/base9-quantization.