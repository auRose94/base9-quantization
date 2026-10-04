# exp17 — K9 PTQ on Qwen2.5-Coder-1.5B-Instruct

Modern coder LLM (1,543,714,304 dedup params; body 1,310,195,712 Linear,
tied embedding 233,373,696, fp32 rest 144,896). Eval: 24x1024
tokens of WikiText-103 test and a held-out Python slice.
Rate = per-tensor digit entropy + entropy-coded log scales + fp32 rest
(RMSNorm weights + Qwen2 attention biases). rANS == entropy on this model to 0.0001 b/param (3-tensor check).

| scheme | k body | k embed | digits b/p | model MB | ppl wiki | ppl code |
|---|---|---|---|---|---|---|
| fp32 baseline | — | fp32 | 0.000 | 6174.9 | 14.041 | 3.716 |
| 8-level g64 (body) | 8 | fp32 | 2.488 | 1357.2 | 27.914 | 5.323 |
| 9-ninths g64 (K9 body) | 9 | fp32 | 2.670 | 1387.0 | 19.007 | 4.477 |
| int4-15 g64 (body) | 15 | fp32 | 3.452 | 1515.1 | 15.168 | 3.940 |
| 99-level g64 (body) | 99 | fp32 | 6.218 | 1968.0 | 14.094 | 3.720 |
| 9-level full (body9+embed9) | 9 | 9 | 2.672 | 534.7 | 20.629 | 4.718 |
| K9 full (body9+embed99) | 9 | 99 | 3.209 | 638.2 | 19.035 | 4.479 |
| int4 full (body15+embed15) | 15 | 15 | 3.454 | 685.6 | 15.594 | 3.986 |

**Frontier check (body, code ppl):** 9-ninths at 2.670 b/p gives 4.477; the 8<->15 linear interpolation at the same rate gives 5.062 — the odd middle alphabet is BELOW it (frontier-efficient (P10-style)).

fp32 footprint 6174.9 MB. Body-only schemes leave the
tied embedding fp32 (its 0.23B params dominate the MB column — the exp10
lesson at a new scale); the full rows quantize it too.

wall 238s
