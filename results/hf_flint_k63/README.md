---
license: apache-2.0
base_model: RecursiveSI/flint
tags:
- quantization
- custom-quantization
- moe
- gguf
- ninths
- k9
---

# Flint 40B-A2.5B — ninths (k9) stream quantization, k63 body / k99 embeddings

A 26.54 GiB containerized quantization of [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint)
in the **ninths** (secondary name: k9) stream format — 5.713 bits/param over the
486 quantized weight tensors (28.39 GB of rANS blobs at build; 33 % smaller than
stock Q8_0 at 39.44 GiB, 64 % smaller than the 74.2 GiB bf16 upstream GGUF).
Weights live as entropy-coded byte streams with an in-file directory; the
runtime materializes them to exact Q8_0 at load time (decode-on-load,
byte-verified). Flint is a 40B-total / 2.5B-active MoE (49 layers: 2 dense +
47 MoE with 6+1 of 256 experts, 2.5B active), so the materialized Q8_0 model
runs at full Q8_0 speed.

Base weights for this container were ggml-org's `Flint-BF16.gguf`
(sha256 `bddf310fc9f33ad394674a6d94f1ba3fe4d83c0ad57bb69f72cc0d443146ff2e`).

## What's inside

- `flint_k63_e99_v2.gguf` — the model (486 stream tensors + 293 copied small
  tensors: norms, per-head QK-RMSNorms, attention sinks, routers, biases).
  Tokenizer is embedded. The GGUF carries no chat-template KV by design; the
  template files are in this repo (see below).
- `chat_template.jinja` — RecursiveSI's upstream template. Note: it contains
  a literal `"\x00"` string transform that llama.cpp's built-in jinja lexer
  rejects; use the next file with `--chat-template-file`.
- `chat_template_llamacpp.jinja` — the same template with that one transform
  removed (NUL-strip; negligible). Use this one with llama.cpp servers.

## Runtime requirement

This format needs the ninths fork of llama.cpp — stock llama.cpp, Ollama, LM
Studio cannot read it:

- fork: `github.com/auRose94/llama.cpp`, branch `k9` (≥ commit 2fafd9ccc, which
  adds Flint support to the fork)
- build: `-DGGML_HIP=ON` for AMD (tested on gfx1151 / Radeon 8060S), or CPU.

## Byte-exactness

Every tensor class was gated against an independent decoder through the fork's
own probe:

- digits (rANS-decoded, v2 lane-interleaved streams): 0 mismatches
- scales: 0.000 max ulp vs reference
- materialized Q8_0 payload: 0 byte mismatches vs the reference packer

Gate examples: `token_embd.weight` (k99, 307M digits), `blk.0.attn_q.weight`,
`blk.6.ffn_gate_exps.weight` (256-expert rank-3 flatten), `output.weight` (k99).

## Measured (A9 Mega — Ryzen AI Max+ 395 / Radeon 8060S, ROCm/HIP)

| arm | file | ppl (wiki2 300k) | pp512 | tg128 | load |
|---|---|---|---|---|---|
| stock Q8_0 (ggml-org) | 39.44 GiB | 22.8678 ± 0.511 | 1430 | 50.0 | ~15 s |
| **this container (k63/e99)** | **26.54 GiB** | **23.6822 ± 0.536** | 1452 | 49.9 | **~36 s** |
| stock Q4_K_M (ggml-org) | 21.78 GiB | 23.7434 ± 0.536 | 1589 | 56.2 | ~13 s |
| IQ4_NL re-quant (`Flint-40B-A2.5B-iq4nl-GGUF`) | 21.21 GiB | 27.5552 ± 0.642 | 1701 | 67.4 | ~12 s |

pp/tg = llama-bench, 16 threads, full GPU offload. Generation speed is
identical to stock Q8_0 (the runtime is the materialized Q8_0).

Honest sizing note: on Flint, stock **Q4_K_M reaches the same perplexity as
this container at a smaller file** — the k63-beats-q4_k_m result from our
GLM-4.5-Air / Qwen3.8 containers did not transfer to this architecture. What
the container does deliver here: −33 % vs Q8_0 for +0.8 ppl (3.6 % rel),
byte-exact build-to-runtime chain, and the ninths container ladder for this
arch family.

## KL protocol vs the bf16 base (exp27 mini-KLD, 60 KB corpus slice)

| candidate | PPL(Q)/PPL(base) | ln(PPL(Q)/PPL(base)) ± | Cor |
|---|---|---|---|
| stock Q8_0 | 1.0395 | 0.0388 ± 0.0082 | 98.6% |
| stock Q4_K_M | 1.0628 | 0.0609 ± 0.0119 | 96.9% |
| **this container (k63/e99)** | **1.0811** | **0.0779 ± 0.0110** | 97.5% |
| IQ4_NL re-quant | 1.2267 | 0.2044 ± 0.0180 | 93.3% |

## Full size/quality ladder

For the measured extreme rungs — **k5 (9.13 GiB, 1.95 b/param, ppl 149,932)** and
**k3 (5.23 GiB, 1.11 b/param)** — see
[`auRose94/Flint-40B-A2.5B-ninths-extreme-GGUF`](https://huggingface.co/auRose94/Flint-40B-A2.5B-ninths-extreme-GGUF)
(uncalibrated coarse grids: the container machinery is byte-exact at every rung,
but beyond ~4 bits the pure-RTN grid itself destroys the model; see that card
for the full ladder and cliff discussion).

## Reasoning model — give it budget

Flint is reasoning-first and the thinking is mandatory; do not force short
`max_tokens`. On a 24-task GDScript coding battery (greedy):

- 480-token budget: 4/24 · 1200: 9/24 · 2500: 19/24 · **6000: 22/24**

The model plans and self-checks extensively inside its thinking block before
writing code.

## Running (tested)

```sh
llama-server \
  -m flint_k63_e99_v2.gguf \
  -ngl 99 -c 8192 -t 16 \
  --chat-template-file chat_template_llamacpp.jinja
```

The materialized model is ~39.5 GB of Q8_0 weights — smaller cards can keep
late MoE blocks in system RAM with
`--override-tensor 'blk\.4[0-9]\..*_exps\.weight=CPU'` as usual.

## License

Apache-2.0 (inherited from the base model). Quantization pipeline and format:
ninths (https://github.com/auRose94/llama.cpp, branch k9;
https://github.com/auRose94/base9-quantization).