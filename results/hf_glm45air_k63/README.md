---
license: mit
base_model: zai-org/GLM-4.5-Air
tags:
- quantization
- custom-quantization
- moe
- gguf
- ninths
---

# GLM-4.5-Air — ninths (k9) stream quantization, k63 body / k99 embeddings

A 73.45 GiB containerized quantization of [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air)
in the **ninths** (secondary name: k9) stream format — 5.706 bits/param over the
quantized weight tensors, 35.7 % of the f16 reference size. This is not a
GGUF k-quant: weights live as entropy-coded base-9 digit streams with an
in-file directory, and the runtime materializes them to exact Q8_0 at load
time (a decode-on-load theorem, byte-verified — see below). The output of a
successful load is bit-identical to the Q8_0 quantization that this container
encodes, and every tensor was gate-checked zero-mismatch against an
independent reference decoder.

## What's inside

- `GLM-4.5-Air-ninths-k63-e99.gguf` — the model (472 stream tensors + 331
  copied small tensors: norms/biases/router etc.). Tokenizer is embedded.
- `chat_template.jinja` — the upstream chat template; the GGUF carries no
  template KV by design (see run notes), pass this file to the server.

## Runtime requirement

This format needs the ninths fork of llama.cpp — stock llama.cpp, Ollama,
LM Studio and friends cannot read it:

- fork: `github.com/auRose94/llama.cpp`, branch `k9`
- build: `-DGGML_HIP=ON` for AMD (tested on gfx1151 / Radeon 8060S), or CPU.

## Running (tested)

Full GPU offload with six late blocks' routed experts kept in system RAM —
the split that fits a 96 GB-carve unified-memory part:

```sh
llama-server \
  -m GLM-4.5-Air-ninths-k63-e99.gguf \
  -ngl 99 -c 8192 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -t 16 \
  --override-tensor 'blk\.4[0-5]\..*_exps\.weight=CPU' \
  --jinja --chat-template-file chat_template.jinja
```

- GPUs with ≥ 128 GiB of device memory can drop the `--override-tensor` line.
- The model materializes to ~113 GB at load; plan memory accordingly
  (the split above holds ~14 GiB of it in host RAM).
- Load time ≈ 9.4 min from this container (single-lane streams);
  measured 119 tok/s prompt eval, 13.4 tok/s generation on a 16-core Zen5
  iGPU with the split above.

## Byte-exactness

Every tensor class was gated against an independent decoder:

- digits (rANS-decoded): 0 mismatches
- scales (ent8 256-level grid, float64 exponent path): ≤ 2 ulp vs reference
- materialized Q8_0 payload: 0 byte mismatches vs the reference packer

The loader decodes on load; there is no runtime quantization step on top.

## Measured (A9 Mega — Ryzen AI Max+ 395 / Radeon 8060S)

| build | file size | load to serving | prompt eval | generation |
|---|---|---|---|---|
| v1 streams | 73.45 GiB | ~9.4 min | 119 tok/s | 13.4 tok/s |
| v2 16-lane | 73.46 GiB | ~2.5 min | same | same |
| stock q8_0 (reference) | 109.4 GiB | ~1.5 min | same | same |

Runtime speed is identical across all three — the format changes only the
load path: K9 decodes-on-load into the same materialized Q8_0 weights. The
v2 load figure was measured while a concurrent upload to this Hub was still
draining (an isolated reload typically lands lower); v1's ~9.4 min was
measured on an otherwise idle machine and is the honest pre-v2 number.

## Byte-verified quality parity vs stock q8_0

24-task Godot 4 coding battery (mechanical graders: required/forbidden API
names per task), greedy decoding, thinking disabled, identical server
configuration and context — only the weight container differs:

| arm | result | notes |
|---|---|---|
| ninths K9 k63/e99 (v2) | **22/24** | load 150 s, decode-on-load exact-Q8_0 |
| stock q8_0 | **22/24** | identical pass/fail pattern |

Per-task failure sets are identical between arms and per-task token counts
match within noise — the k63/k99 grid costs no measurable task-level quality
at ~67% of the q8_0 file size. Both arms also run the same speed, since the
runtime tensor layout is the same.

## Stock-compatible IQ4_NL export

The stream container is source; the load-time target is a choice. For
runtimes WITHOUT the ninths fork, `glm_k9_build.py materialize --dst iq4_nl`
writes a stock GGUF (`GLM-4.5-Air-iq4nl-GGUF` on this Hub) that any
llama.cpp build loads directly — no custom code. Measured on the same box:
loads in ~30 s, 25.8 tok/s generation (142 tok/s prompt), 59 GB device
footprint, zero host-side expert buffers. The fork route reaches the same
quality from the 73.45 GiB container via `K9_MATERIALIZE=iq4_nl` (see
variants below).

## Variants

| file | stream format | needs |
|---|---|---|
| `GLM-4.5-Air-ninths-k63-e99.gguf` | v1 single-lane rANS | fork `k9` at `14490cf2d` or later |
| `GLM-4.5-Air-ninths-k63-e99-v2.gguf` | v2 lane-interleaved (16 lanes on large tensors) | fork `k9` at `4c99facb9` or later; loads in ~2.5 min |
| `GLM-4.5-Air-ninths-k63-e99-v2.gguf` (+`K9_MATERIALIZE=iq4_nl`) | v2, rematerialized to IQ4_NL at load | same build (`d39bc1d55`+); ~62 GB device-resident, 25 t/s gen |
| (separate repo) `GLM-4.5-Air-iq4nl-GGUF` | fully materialized stock IQ4_NL | ANY llama.cpp build; no fork needed |

v1 files keep loading on every build; v2 files are rejected loudly by older
builds (they look for a directory KV that v1 readers ignore).

## Notes

- The GGUF metadata keys are `ninths.*` (legacy files with `k9.*` keys still
  load — the loader accepts both).
- Materialization target is selectable via the `ninths.materialize` metadata
  key (`q8_0` default; `f16` for act-order-permuted tensors).
- License: MIT (inherited from the base model weights).