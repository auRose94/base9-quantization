# exp9 — the 127-level anomaly: independent windows + scope ablation

fp32 baseline window A 40.663; window A values reproduce exp7 RTN.

## Test A — three disjoint eval windows (g=64 RTN, all layers)

| grid | window | ppl |
|---|---|---|
| 63 | A | 40.646 |
| 63 | B | 39.129 |
| 63 | C | 42.775 |
| 99 | A | 40.565 |
| 99 | B | 39.236 |
| 99 | C | 42.764 |
| 127 | A | 40.978 |
| 127 | B | 39.612 |
| 127 | C | 43.130 |

## Test B — scope ablation on window A

| grid | scope | ppl | rel MSE (scoped) |
|---|---|---|---|
| 99 | attn-only | 40.513 | 0.00026 |
| 99 | mlp-only | 40.717 | 0.00024 |
| 127 | attn-only | 40.815 | 0.00016 |
| 127 | mlp-only | 40.821 | 0.00015 |

## Verdicts

- P12 (inversion reproduces in A, B and C): **PASS** — {'A': 'A:inverted', 'B': 'B:inverted', 'C': 'C:inverted'}
- P13 (localization): see Test B — if 127 loses to 99 in one scope
  only, the anomaly is structural; if both, it is global.

wall 101s
