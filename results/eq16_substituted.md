# eq16 — RQ10 stage B leg 1: substituted semantics + P40 (docs/09)

subject: eq9_QAT_k27emb99.k9 (eq9: ctrl 1.2962 / k27 1.3143 / source 1.9726)

CPU re-eval of the artifact in this harness: 1.3179

λ sites (train-slice calibrated): [0.8135, 0.7947, 0.7709, 0.772, 0.7608, 0.7639, 0.7556, 0.7612, 0.7544, 0.763, 0.7496]

| variant | loss | vs k27 arm |
|---|---|---|
| norm_only | 1.3717 | +4.36% |
| att_only_c4 | 1.9672 | +49.67% |
| att_only_c1 | 1.6224 | +23.44% |
| silu_only | 1.3885 | +5.65% |
| joint_c4 | 2.2759 | +73.16% |
| joint_c1 | 1.8658 | +41.96% |
| joint_uncal | 2.4923 | +89.63% |

- P39a (FAIL): mean-abs norm (calibrated) alone 1.3717 (+4.36%, bar +1%)
- P39b (FAIL): quadratic rational attention — c=4: 1.9672 (+49.67%), c=1: 1.6224 (+23.44%), bar +1% (formal row = the better c; c was not pinned at registration)
- P39c (reported): silu-r alone 1.3885 (+5.65%)
- P39 (FAIL): joint, the better c: 1.8658 (+41.96%, bar +2%)
- joint without λ calibration: 2.4923 (+89.63%) — the calibration is doing real work

- P40 (PASS): best order-1 Markov gain on activation digit streams: 7.41% (bar 5%; exp12's weight-digit null: 0.0–0.2%)
