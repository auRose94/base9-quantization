---
license: mit
base_model: zai-org/GLM-4.5-Air
tags:
- quantization
- gguf
- iq4_nl
---

# GLM-4.5-Air — stock IQ4_NL GGUF (materialized from the ninths K9 build)

A plain, stock-compatible IQ4_NL quantization of GLM-4.5-Air: standard GGUF
tensor types, loads in any current llama.cpp / Ollama / LM Studio build —
no custom runtime needed.

Produced from `auRose94/GLM-4.5-Air-ninths-k63-e99-GGUF` (a 73.45 GiB
channel-coded container) by materializing its base-9 digit streams to IQ4_NL
with a deterministic per-32-block RTN packer, gate-verified byte-exact
against the fork's runtime materializer. Embedding and output-head tensors
stay at Q8_0.

Measured on an AMD Ryzen AI Max+ 395 (Radeon 8060S, 96 GB unified):
~30 s load, ~25.8 tok/s generation with full GPU residency (59 GiB device),
142 tok/s prompt eval. On a 24-task Godot-4 coding battery (mechanical
graders, greedy decoding) it scores 22/24 — the same as the exact-Q8_0
materialization of the parent container, at 54% of a stock q8_0 file's size.

MIT license (inherited from the base model).
