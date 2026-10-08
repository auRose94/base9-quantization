# eq32 — the P47 frontier with a real clusterer (docs/09 RQ13 c-d)

Deterministic context keys from the prefix counts, swept over K, with the branchless inverse-table decode at R=4096 (eq31's finding: the search's branches cost 2-3x).


| stream | key | K | bytes | M/s | ratio of byte coder |
|---|---|---|---|---|---|
| activation_site0 | top1_argmax | 4 | 173,944 | 89.1 | 37% |
| activation_site0 | top1_argmax | 8 | 168,102 | 84.0 | 35% |
| activation_site0 | top1_argmax | 16 | 156,280 | 75.1 | 31% |
| activation_site0 | top1_argmax | 32 | 141,588 | 61.6 | 26% |
| activation_site0 | kmeans | 4 | 174,302 | 96.5 | 40% |
| activation_site0 | kmeans | 8 | 170,586 | 90.7 | 38% |
| activation_site0 | kmeans | 16 | 166,880 | 80.3 | 33% |
| activation_site0 | kmeans | 32 | 160,627 | 65.2 | 27% |
| activation_site0 | p2_bucket | 4 | 178,040 | 67.7 | 28% |
| activation_site0 | p2_bucket | 8 | 177,215 | 64.9 | 27% |
| activation_site0 | p2_bucket | 16 | 175,371 | 59.5 | 25% |
| activation_site0 | p2_bucket | 32 | 172,130 | 50.6 | 21% |
| activation_site0 | **best ratio** | 4 | 174,302 | 96.5 | 40% | *(best rate at >=30%: top1_argmax K=16, 156,280 B, 31%)*
| embed_weight | top1_argmax | 4 | 26,088 | 101.2 | 45% |
| embed_weight | top1_argmax | 8 | 26,099 | 96.1 | 43% |
| embed_weight | top1_argmax | 16 | 26,097 | 87.5 | 39% |
| embed_weight | top1_argmax | 32 | 26,160 | 73.0 | 32% |
| embed_weight | kmeans | 4 | 26,092 | 106.2 | 47% |
| embed_weight | kmeans | 8 | 26,092 | 100.6 | 45% |
| embed_weight | kmeans | 16 | 26,092 | 91.1 | 40% |
| embed_weight | kmeans | 32 | 26,092 | 75.6 | 34% |
| embed_weight | p2_bucket | 4 | 26,097 | 83.5 | 37% |
| embed_weight | p2_bucket | 8 | 26,100 | 80.1 | 36% |
| embed_weight | p2_bucket | 16 | 26,120 | 74.1 | 33% |
| embed_weight | p2_bucket | 32 | 26,134 | 63.3 | 28% |
| embed_weight | **best ratio** | 4 | 26,092 | 106.2 | 47% | *(best rate at >=30%: top1_argmax K=4, 26,088 B, 45%)*

- **P47 (FAIL)** (bar 50%): best ratios {'activation_site0': 0.402, 'embed_weight': 0.471}. eq31's K=1 controls (a single table, no rate win) reached 37.5% / 46.0%, so the activation stream could not clear the bar in this datapath family even with a perfect clusterer; the embed stream is bounded at 46%.

Reference points: the static single-table rate is 175,975 B (activation) / 24,749 B (embed); eq30's unclustered full-context engine coded 93,592 / 26,108 B at 15-19% of the byte coder.

Framing (kept from eq31): 90-105 M sym/s means a 1.5B-class k9 artifact (~1.5G digits) loads in ~15-45 s with context coding on, which is what the companion design is for; the registered bar is stricter than a load-time path needs.