# eq30 — P47 fix-path: cache-resident, incrementally-built context coder (docs/09 RQ13 c-d)

streams = eq27's (eq26's winner + weight null); segments of 1024 (each its own 6-byte state, exactly eq27's file shape); the companion's tables refresh every R symbols (R swept).


| stream | n | byte M/s | base9 M/s | R | engine M/s | ratio vs byte | bytes | roundtrip |
|---|---|---|---|---|---|---|---|---|
| activation_site0 | 262144 | 237 | 161 | 1024 | 31.2 | 13% | 92,024 | exact |
| activation_site0 | 262144 | 237 | 161 | 4096 | 36.1 | 15% | 93,592 | exact |
| activation_site0 | 262144 | 237 | 161 | 16384 | 37.6 | 16% | 99,600 | exact |
| activation_site0 | 262144 | 237 | 161 | 65536 | 41.3 | 17% | 126,983 | exact |
| embed_weight | 32768 | 231 | 188 | 1024 | 37.9 | 16% | 25,306 | exact |
| embed_weight | 32768 | 231 | 188 | 4096 | 44.0 | 19% | 26,108 | exact |
| embed_weight | 32768 | 231 | 188 | 16384 | 62.6 | 27% | 28,510 | exact |
| embed_weight | 32768 | 231 | 188 | 65536 | 68.7 | 30% | 27,352 | exact |

- **P47 (FAIL)** (bar: >= 50% of the byte coder's decode throughput): best ratios {'activation_site0': 0.174, 'embed_weight': 0.297} at R [65536, 65536].

eq27's diagnosis, for contrast: 16%/20% of the byte coder with 26 KB per-context inverse tables and a 134 us/symbol numpy build; the cadence was never the binding cost (38/36/43 M sym/s across 1k/4k/whole).