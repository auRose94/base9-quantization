---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B-Base
tags:
- quantization
- moe
- gguf
- ninths
---

# Qwen3.8-27B — ninths (k9) stream quantization, k63 body / k99 embeddings

An 18.3 GiB ninths-container quantization of Qwen3.8-27B (qwen35
architecture: hybrid gated-delta-net + attention blocks, dense FFNs, 65
blocks + MTP layer). 410 stream tensors at 5.744 bits/param, 36% of the f16
reference size, with norms/biases/copied tensors unchanged.

This is not a GGUF k-quant: weights live as entropy-coded base-9 digit
streams with an in-file directory; the runtime materializes them to exact
Q8_0 at load time (byte-exact, gate-verified per tensor class) — or to
IQ4_NL / f16 via the `K9_MATERIALIZE` override / `ninths.materialize` KV.

## Runtime requirement

Needs the ninths fork of llama.cpp — stock runtimes should use the
stock-compatible IQ4_NL export instead (`Qwen3.8-27B-iq4nl-GGUF`):

- fork: `github.com/auRose94/llama.cpp`, branch `k9` (v2-capable at
  commit `d39bc1d55` or later)
- works on CPU, CUDA, and HIP/ROCm (tested on gfx1151 / Radeon 8060S).

Measured (idle A9 Mega, 96 GiB carve, 16 KiB ctx): K9 load with lane-parallel
decode ~40 s at 16 lanes; generation identical to the stock export (the
materialized weights are the same object): 13.1 tok/s alone, 27.0 tok/s with
`--spec-type draft-mtp` (64% acceptance). 24-task Godot-4 battery: 21/24.

## Running

```sh
llama-server \
  -m Qwen3.8-27B-ninths-k63-e99.gguf \
  -ngl 99 -c 16384 -fa on -t 16 \
  --jinja
```

Add `K9_MATERIALIZE=iq4_nl` to the environment for the smaller/faster
IQ4_NL materialization (~15.6 GiB device footprint, full-GPU residency;
the embedding table and output head stay exact-Q8_0).

The model ships a `nextn` (MTP) layer and can draft with `--spec-type
draft-mtp`.

## Byte-exactness

Gated against an independent decoder, 0 mismatches, on: the k99 embedding
(1.27B-symbol stream), a dense FFN matmul, and the fused attention QKV —
lane-interleaved (16-lane) streams, thread-count-invariant output.

## Notes

- Metadata keys: `ninths.directory2` + `ninths.materialize` (spec:
  docs/12 in the base9-quantization repo).
- Hybrid-attention models are first-class: SSM-side matmuls quantize
  through the same stream path.
- License Apache-2.0 (inherited from the base model).
