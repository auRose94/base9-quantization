# exp9b — cure test: fitted (Lloyd–Max) codebooks on the 127-level anomaly

fp32 baseline window A 40.663; uniform-RTN references (exp7/exp9): 99 → 40.565, 127 → 40.978.

| grid | fitted ppl | fitted rel MSE |
|---|---|---|
| 99 | 40.457 | 0.00008 |
| 127 | 40.635 | 0.00004 |

P16 (fitted-127 ≤ fitted-99 + 0.3): **PASS**
- If PASS: the anomaly is a uniform-grid structural artifact that fitted
  codebooks cure (outlier/endpoint placement of the ±m-matched step).
- If FAIL: the 127 loss is not endpoint-driven; flag stays open.

wall 16s
