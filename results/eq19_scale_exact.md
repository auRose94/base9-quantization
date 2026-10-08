# eq19 — the (num, den, E) hybrid carrier with the scale-norm

subject: eq9_QAT_k27emb99.k9, 2×512; order-identical: True; widen events: ['layers.0.attention.wo.weight', 'layers.1.attention.wo.weight', 'layers.2.attention.wo.weight', 'layers.3.attention.wo.weight', 'layers.4.attention.wo.weight', 'layers.0.attention.wo.weight', 'layers.1.attention.wo.weight', 'layers.2.attention.wo.weight', 'layers.3.attention.wo.weight', 'layers.4.attention.wo.weight']

BOUNDEDNESS: worst num 5.4 / den 5.7 digits (u128 = 40.4) — BOUNDED.

| site | |Δ| vs twin | scale | flips>0.05 |
|---|---|---|---|
| embed | 2.919e-08 | 1.31 | 0 |
| L0.attnorm | 1.362e-06 | 14.9 | 0 |
| L0.scores | 2.594e-04 | 1.7e+308 | 0 |
| L0.res_mid | 1.175e-05 | 3.23 | 0 |
| L0.ffnorm | 2.393e-05 | 5.62 | 0 |
| L0.res_out | 8.673e-05 | 25.3 | 0 |
| L1.attnorm | 1.473e-04 | 15.3 | 0 |
| L1.scores | 1.770e-03 | 1.7e+308 | 0 |
| L1.res_mid | 1.245e-04 | 26 | 0 |
| L1.ffnorm | 7.387e-05 | 7.38 | 0 |
| L1.res_out | 1.635e-04 | 25.9 | 0 |
| L2.attnorm | 1.608e-04 | 21 | 0 |
| L2.scores | 2.197e-03 | 1.7e+308 | 0 |
| L2.res_mid | 1.824e-04 | 25.9 | 0 |
| L2.ffnorm | 1.232e-04 | 6.76 | 0 |
| L2.res_out | 2.081e-04 | 26.2 | 0 |
| L3.attnorm | 2.495e-04 | 12.1 | 0 |
| L3.scores | 2.926e-03 | 1.7e+308 | 0 |
| L3.res_mid | 2.340e-04 | 26.7 | 0 |
| L3.ffnorm | 1.432e-04 | 7.04 | 0 |
| L3.res_out | 2.965e-04 | 27.7 | 0 |
| L4.attnorm | 3.335e-04 | 11.2 | 0 |
| L4.scores | 2.495e-03 | 1.7e+308 | 0 |
| L4.res_mid | 4.060e-04 | 27.9 | 0 |
| L4.ffnorm | 3.490e-04 | 8.71 | 0 |
| L4.res_out | 1.201e-03 | 39.1 | 0 |
| logits | 1.826e-03 | 68 | 0 |
