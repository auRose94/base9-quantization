# exp10 — quantizing the embeddings (roneneldan/TinyStories-33M)

fp32 baseline 40.663 | fp32 model 274.1 MB | body-quant rows
use the 9-level ninths grid (g=64 RTN; cross-check vs exp7: 54.679);
embeddings quantized per-row (g=64) on 9- and 99-level grids.

| scheme | ppl | total model MB | quantized MB | fp32 rest MB |
|---|---|---|---|---|
| fp32 embeds + body9 | 54.679 | 172.1 | 11.3 | 160.8 |
| embed9 + fp32 body | 44.642 | 129.6 | 16.2 | 113.4 |
| embed9 + body9 | 60.990 | 27.7 | 27.5 | 0.1 |
| embed99 + body9 | 54.781 | 45.5 | 45.3 | 0.1 |

P14 (embedding quant ≤ +10% relative ppl): **PASS** (embed9+fp32body: 44.64 vs 40.66)
P15 (combined saves ≥ 40% bytes): **PASS** (embed9 + body9: 27.7 vs 274.1 MB)

wall 92s
