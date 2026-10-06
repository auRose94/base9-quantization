# exp27 — KL divergence vs the f16 reference (upstream checklist item)

`llama-perplexity` (llama.cpp fork build-cpu), corpus `results/wiki2_raw_test.txt`
(the session-24 ppl corpus), defaults except `-t 8`, CPU backend.
Reference pass: `ref_qwen_1.5b_f16.gguf` -> base file `1p5b_ref_f16.kld` (45.2 GB, regenerable); its final estimate: PPL == 13.849002 (CPU).

| candidate | file GB | Mean KLD ± | PPL(Q) ± | PPL(Q)/PPL(base) | PPL(Q)−PPL(base) |
|---|---|---|---|---|---|
| `K9_k63_embed99` | 1.119 | 0.006401 ± 0.000023 | 13.899698 ± 0.110152 | 1.003661 ± 0.000406 | 0.050696 |
| `q8_0` | 1.647 | 0.000968 ± 0.000003 | 13.898381 ± 0.110249 | 1.003565 ± 0.000256 | 0.049379 |
| `q4_k_m` | 0.986 | 0.036364 ± 0.000137 | 14.134115 ± 0.111613 | 1.020587 ± 0.000877 | 0.285113 |

Notes:
- F16 reference = f16 GGUF storing Qwen2.5's native bf16 weights exactly
  (bf16's 8-bit mantissa ⊂ f16's 10-bit for in-range values).
- K9_k63_embed99 materializes to Q8_0 on load (exact-Q8_0 theorem); the
  runtime therefore runs the materialized Q8_0/F16 tensors, stock kernels.
- Session-24 ppl table used GPU backends; CPU values shift slightly.
- Per-chunk tables and the full llama-perplexity stdout: results/kld/*.pass.log
- Quantize logs: results/kld/quantize_*.log (baselines = llama-quantize from
  the reference, `Q8_0` / `Q4_K_M`).
