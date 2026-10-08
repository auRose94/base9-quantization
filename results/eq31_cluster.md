# eq31 — P47 clustering frontier: K hot tables (docs/09 RQ13 c-d)

streams = eq27's (eq26's winner + weight null); segments of 1024; the companion refreshes at R=4096 and clusters contexts into K tables (assignment = the argmax continuation's rank among the K-1 most probable symbols; each cluster's table = its members' pooled counts through the same estimator chain).


| stream | n | byte M/s | base9 M/s | K | engine M/s | ratio vs byte | bytes | roundtrip |
|---|---|---|---|---|---|---|---|---|
| activation_site0 | 262144 | 242 | 162 | 1 | 90.3 | 37% | 178,702 | exact |
| activation_site0 | 262144 | 242 | 162 | 2 | 90.8 | 38% | 176,752 | exact |
| activation_site0 | 262144 | 242 | 162 | 4 | 88.3 | 36% | 173,944 | exact |
| activation_site0 | 262144 | 242 | 162 | 8 | 83.5 | 34% | 168,102 | exact |
| activation_site0 | 262144 | 242 | 162 | 16 | 75.1 | 31% | 156,280 | exact |
| embed_weight | 32768 | 229 | 187 | 1 | 105.3 | 46% | 26,092 | exact |
| embed_weight | 32768 | 229 | 187 | 2 | 103.6 | 45% | 26,068 | exact |
| embed_weight | 32768 | 229 | 187 | 4 | 100.6 | 44% | 26,088 | exact |
| embed_weight | 32768 | 229 | 187 | 8 | 95.8 | 42% | 26,099 | exact |
| embed_weight | 32768 | 229 | 187 | 16 | 87.0 | 38% | 26,097 | exact |

- **P47 (FAIL)** (bar: >= 50% of the byte coder's decode throughput): best ratios {'activation_site0': 0.375, 'embed_weight': 0.46} at K [2, 1]; bytes at those points [176752.4, 26092.3] (eq27's static baselines: 175,975 / 24,749 B; eq30's unclustered full-context engines: 93,592 / 26,108 B at 15-19%).

eq30's residual, for contrast: its per-context tables are a 4.7 MB spread, so every symbol's freq/cum/search touched L3 -- the reason its binary search could not beat eq27's inverse tables (both L3-bound) and its loop plateaued at 41-69 M sym/s while the plain single-table machine does 161-188. Clustering is aimed exactly there: K tables are L1-resident.