# eq17 — RQ10 stage B leg 2: the exact runtime (docs/09)

subject: eq9_QAT_k27emb99.k9, 2×512 tokens; state = (num, den, E, J)
per row with (E, J) exponent tracks absorbing scale; exact per-row
reduction/renorm after every site; fp-mixed RoPE/attention/silu
with declared 2^30 fixed-point intakes; λ dyadic 2^-10.

- P38 (PASS): order-identical logits=True; width tracks identical=True; widening events identical=True; max|Δlogits−fp32ref| = 3.684e+01.
- P37 (FAIL): worst num 694.3 digits, worst den 693.7 digits vs the ≤20-digit bar; int64 widened at ['layers.0.w1', 'layers.0.w3'].

| site | num digits | den digits |
|---|---|---|
| embed | 5.4 | 1.9 |
| L0.attnorm | 7.3 | 3.5 |
| L0.qkv | 13.2 | 11.0 |
| L0.res_mid | 17.4 | 3.2 |
| L0.norm | 17.7 | 16.7 |
| L0.w1w3 | 23.7 | 24.3 |
| L0.res_out | 41.6 | 21.1 |
| L1.attnorm | 46.7 | 42.9 |
| L1.qkv | 52.7 | 50.2 |
| L1.res_mid | 41.6 | 21.1 |
| L1.norm | 46.4 | 42.9 |
| L1.w1w3 | 51.7 | 50.5 |
| L1.res_out | 84.5 | 63.7 |
| L2.attnorm | 89.6 | 85.8 |
| L2.qkv | 95.9 | 93.4 |
| L2.res_mid | 84.5 | 63.7 |
| L2.norm | 89.3 | 85.8 |
| L2.w1w3 | 94.3 | 93.4 |
| L2.res_out | 170.4 | 149.2 |
| L3.attnorm | 175.4 | 171.6 |
| L3.qkv | 181.4 | 179.2 |
| L3.res_mid | 170.4 | 149.2 |
| L3.norm | 175.4 | 171.6 |
| L3.w1w3 | 180.8 | 179.2 |
| L3.res_out | 342.0 | 320.5 |
| L4.attnorm | 347.0 | 343.2 |
| L4.qkv | 353.3 | 350.8 |
| L4.res_mid | 342.0 | 320.5 |
| L4.norm | 346.7 | 343.2 |
| L4.w1w3 | 352.1 | 350.8 |
| L4.res_out | 684.9 | 662.8 |
| logits | 694.3 | 693.7 |
