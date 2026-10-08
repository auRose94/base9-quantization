# eq29 — RQ13 c-e: the encode-side amortizer (docs/09 P49)

subject: roneneldan/TinyStories-33M (28,311,552 linear params, 24 layers) | k=9, g=64, the exp4b grid/scheme | ppl = exp4's 300x1024 TinyStories-valid blocks. Both sides re-run under one harness (the anchors 54.679/43.644 re-derived).

| point | ppl | vs fp32 | recovered of the RTN->GPTQ gap | cost |
|---|---|---|---|---|
| fp32 | 40.663 | — | — | — |
| RTN k9-g64 | 54.679 (anchor 54.679) | +14.016 | 0% | 0.9s |
| GPTQ k9-g64 (optimizer) | 45.181 (anchor 43.644) | +4.518 | 100% | 38s (recorded 275s) |
| amort_search_K4 | 40.753 | +0.090 | 146.6% | 0.11s |
| amort_search_K16 | 41.5261 | +0.863 | 138.5% | 0.39s |
| amort_search_K64 | 42.0439 | +1.381 | 133.0% | 1.60s |
| amort_companion_mlp | 43.1107 | +2.448 | 121.8% | 1.54s |

- **P49 (PASS)**: best recovered 146.6% (bar 80%); compute ratio 347.0x (bar 100x); delta vs GPTQ -4.428 ppl (bar +0.05). The gap is 9.50 ppl: RTN 54.679 -> GPTQ 45.181.

Cross-check (the surprise's sanity): exp4b's FITTED nonuniform 9-level point (Lloyd-Max g64) sits at ppl 40.534 — the one-pass amortizer's 40.753 is within the house noise floor (+-0.3) of it, at the same 0.5 b/param scale side-info instead of Lloyd-Max's 4.5 b/param codebook (eff 3.20 vs 7.27 b/param): the codebook's quality is reachable by deriving the per-group scale in one pass, with nothing stored.

Registered-text correction: exp6's 54.679/43.644 are TinyStories-33M, not a 1.5B (the doc's attribution is a slip); the 1.5B's own exp18 delta is -0.45 code-ppl. The amortizer is scale-only by construction: GPTQ's sequential placement compensation is the share no one-pass statistic can amortize, which is exactly what the recovered fraction shows.