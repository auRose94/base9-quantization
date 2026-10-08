# eq27 — RQ13 c-d engineering half: chunk-granularity throughput (docs/09 P47)

codec: per-chunk independent rANS segments (each with its own 6-byte end state = 0.047 b/sym at 1k chunks); companion = the order-2 interpolated count model refreshing the context-conditioned table set once per chunk from the decoded prefix; the state machine does per-symbol table lookups only.

- **P47 (FAIL)** (bar: >= 50% of the byte coder's decode throughput):

| stream | n | byte M/s | base9 M/s | ctx@1k M/s | ratio | ctx@4k | ctx whole |
|---|---|---|---|---|---|---|---|
| activation_site0 | 262144 | 243 | 163 | 38 | 16% | 36 | 43 |
| embed_weight | 32768 | 228 | 187 | 46 | 20% | 58 | 64 |

| stream | static B | companion B | bits win | build us/chunk | ns/sym amortized | per-symbol torch fwd us |
|---|---|---|---|---|---|---|
| activation_site0 | 175,975 | 91,586 | +47.95% | 137209 | 133993 | 5.9 |
| embed_weight | 24,749 | 22,114 | +10.65% | 33079 | 32304 | 5.7 |