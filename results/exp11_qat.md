# exp11 — QAT: creating a new model ON the repeating-decimal grids

Eval: val window A (same as exp4–10; fp32-pretrained 40.663,
TRAIN-format diagnostic 36.416 — the format confound v1 found).
Arms (identical 2000-step schedule, 8 M train tokens,
batch 4×512, AdamW + cosine, weight-swap STE): CONTROL fp32 →
QAT-RECIPE (body 9-level ninths + embeds 99) → QAT-TERN (ternary body
with learnable gamma + embeds 99).

| variant | ppl (window A) | vs CONTROL |
|---|---|---|
| fp32 pretrained | 40.663 | +36.07 |
| PTQ body9 (PTQ) | 54.679 | +50.09 |
| PTQ embed99+body9 (PTQ) | 54.781 | +50.19 |
| PTQ ternary-g64+embed99 (PTQ) | 989.767 | +985.18 |
| CONTROL (fp32 adapted) | 4.591 | — |
| QAT-RECIPE (body9 + embed99) | 4.886 | +0.29 |
| QAT-TERN (ternary + embed99) | 5.980 | +1.39 |

Ternary body payload: rANS 1.585 b/param, base-9 pairs 1.584 b/param.

## Pre-registered verdicts (v2 — P19 re-anchored to the adapted
control after v1 exposed the format confound)

- P17 (QAT-RECIPE ≥2 ppl better than its PTQ): **PASS** (54.78 → 4.89)
- P18 (QAT-TERN ≤ 200 → base-9 pair payload viable): **PASS** (989.77 → 5.98)
- P19 (QAT-RECIPE ≤ CONTROL×1.25 = 5.74): **PASS**

Gain = QAT vs its PTQ and (the honest one) QAT vs the trained control;
loss = QAT vs the trained control. Digit artifacts:
results/qat_digits_{control,recipe,tern}.npz.

wall 1646s
