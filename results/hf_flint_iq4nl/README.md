---
license: apache-2.0
base_model: RecursiveSI/flint
tags:
- quantization
- moe
- gguf
- ninths
- k9
---

# Flint 40B-A2.5B — stock-compatible IQ4_NL re-quant (k99 embeddings exact)

A 21.21 GiB deterministic IQ4_NL re-materialization of [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint)
that **loads in stock llama.cpp, Ollama, LM Studio** — no fork needed. Built by
decoding our ninths k63/e99 container (`auRose94/Flint-40B-A2.5B-ninths-k63-e99-GGUF`)
and re-packing the body tensors as IQ4_NL (4.25 bpw class, d = |amax|/113 RTN
grid); the token-embedding and output-head tensors stay on the exact k99 grid
and are stored as Q8_0 so the head fidelity is preserved.

- body: 481 tensors as IQ4_NL; 293 small tensors (norms, per-head QK-RMSNorms,
  attention sinks, routers, biases) copied verbatim as fp16/fp32
- embeddings + lm_head: Q8_0 (k99-exact grid materialization)

## Measured (A9 Mega — Ryzen AI Max+ 395 / Radeon 8060S, ROCm/HIP)

| arm | file | ppl (wiki2 300k) | pp512 | tg128 | load |
|---|---|---|---|---|---|
| stock Q8_0 (ggml-org) | 39.44 GiB | 22.8678 ± 0.511 | 1430 | 50.0 | ~15 s |
| ninths k63/e99 container | 26.54 GiB | 23.6822 ± 0.536 | 1452 | 49.9 | ~36 s |
| stock Q4_K_M (ggml-org) | 21.78 GiB | 23.7434 ± 0.536 | 1589 | 56.2 | ~13 s |
| **this re-quant** | **21.21 GiB** | 27.5552 ± 0.642 | **1701** | **67.4** | ~12 s |

## KL protocol vs the bf16 base (exp27 mini-KLD, 60 KB corpus slice)

| candidate | PPL(Q)/PPL(base) |
|---|---|
| stock Q8_0 | 1.0395 |
| stock Q4_K_M | 1.0628 |
| ninths k63/e99 container | 1.0811 |
| this re-quant | 1.2267 (ln 0.2044 ± 0.0180, Cor 93.3%) |

Honest note: at this bpw the 4-bit grid costs Flint noticeably more perplexity
than the GLM/Qwen exports did (+4.7 pp over Q8_0 here); it is the
fastest-serving and smallest option in the table, not the quality option.

## Chat template

The GGUF carries no chat-template KV by design.

- `chat_template.jinja` — RecursiveSI's upstream template
- `chat_template_llamacpp.jinja` — same template with its literal `"\x00"`
  transform removed; llama.cpp's built-in jinja lexer cannot parse the
  original, so pass this file:

```sh
llama-server -m flint_iq4nl.gguf -ngl 99 -c 8192 \
  --chat-template-file chat_template_llamacpp.jinja
```

## Reasoning model — give it budget

Flint is reasoning-first and the thinking is mandatory; do not force short
`max_tokens` (our coding battery moved 4/24 → 22/24 between 480 and 6000
token budgets).

## License

Apache-2.0 (inherited from the base model). Pipeline:
https://github.com/auRose94/base9-quantization (experiments/glm_k9_build.py
materialize path).