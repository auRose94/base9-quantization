# exp13 — the deployable full-model point: embed99 + body9(+GPTQ)

fp32 baseline 40.663 | fp32 model 274.1 MB | eval
window A, batch 8; body = 9-ninths g=64 (GPTQ where labeled),
embeddings = 99-level grid; ALL side info (scales) counted.

| variant | ppl | total MB | body MB | embed MB | fp32 rest | body digits rANS | body rel MSE |
|---|---|---|---|---|---|---|---|
| RTN | 54.781 | 45.5 | 11.3 | 34.0 | 0.13 | 2.701 | 0.03690 |
| GPTQ | 44.961 | 45.5 | 11.4 | 34.0 | 0.13 | 2.709 | 0.05053 |

P23 (GPTQ full-model ≤ 50 ppl): **PASS** (44.961)

Deployable artifact: results/full_model_digits.npz — per-matrix digits
+ per-(row,group) scales for both variants (inventory inside).
Context: exp11's TRAINED arms sit at 4.89/5.98 ppl (they continued
training); the PTQ routes here keep the pretrained weights.

wall 960s
