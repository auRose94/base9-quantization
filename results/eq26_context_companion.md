# eq26 — RQ13 c-d: the context companion (docs/09 P48)

held-out discipline: tables/weights from the first 80% of each stream, rate accumulated on the last 20% with the stream's own causal context; companions = order-1/-2 Witten-Bell backoff + a small GRU.

- **P48 (PASS)**: best activation-stream recovery 50.74% (order2), GRU 42.24% (bar 20.0%). P40 cross-check reproduced exactly: fp 7.41% / sub_norm 7.35% vs the recorded 7.41% / 7.35%
  (in-sample order-1 Markov on the identical streams; the held-out order-1 numbers are lower, as they must be).

- **weight-digit null (SURPRISE(flagged))**: a codec picks the better model per stream, so the registered null is the codec-choice recovery: aggregate 0.502%, max single tensor 2.748% (bar 2%). The RAW companion numbers show why: aggregate 0.502% (the 2nd-order context is too sparse on near-iid digits, and the fitted lampda collapses it toward the static table), max single 2.748% (tok_embeddings.weight), GRU -10.011% — the exp12 null holds.

- **token streams (the strongest case, recorded)**: 8000 generated tokens; static 7.042 b/tok -> order-k 36.8% | GRU 8.463 b/tok (0.0%) | the LM's own per-step coding 1.392 b/tok = 80.2% recovery as the ceiling.
  story head: `there was a little girl named Lily. She lived in a big hive with her mommy. They loved to play with the icy flag were always happy to find a new salad.
O`

| variant | site | n | static b/sym | order-1 | order-2 | held-out recovery | codec-choice | lampdas | in-sample order-1 |
|---|---|---|---|---|---|---|---|---|---|
| fp | 0 | 262144 | 5.3716 | 5.0214 | 2.6462 | 50.74% | 50.74% | [0.95, 0.95] | 7.26% |
| fp | 1 | 262144 | 4.3682 | 4.3598 | 4.3277 | 0.93% | 0.93% | [0.75, 0.45] | 0.568% |
| fp | 2 | 262144 | 5.5326 | 5.5028 | 5.3507 | 3.29% | 3.29% | [0.55, 0.35] | 1.549% |
| fp | 3 | 262144 | 4.7224 | 4.7189 | 4.6963 | 0.55% | 0.55% | [0.55, 0.3] | 0.321% |
| fp | 4 | 262144 | 5.4797 | 5.4737 | 5.4464 | 0.61% | 0.61% | [0.3, 0.15] | 0.844% |
| fp | 5 | 262144 | 4.8393 | 4.8374 | 4.8341 | 0.11% | 0.11% | [0.4, 0.1] | 0.266% |
| fp | 6 | 262144 | 5.4902 | 5.4884 | 5.4851 | 0.09% | 0.09% | [0.2, 0.05] | 0.676% |
| fp | 7 | 262144 | 5.0117 | 5.0111 | 5.0098 | 0.04% | 0.04% | [0.25, 0.05] | 0.251% |
| fp | 8 | 262144 | 5.4519 | 5.4511 | 5.4497 | 0.04% | 0.04% | [0.15, 0.05] | 0.493% |
| fp | 9 | 262144 | 5.1932 | 5.1929 | 5.1929 | 0.01% | 0.01% | [0.1, 0.0] | 0.3% |
| fp | 10 | 262144 | 5.4816 | 5.3239 | 5.3193 | 2.96% | 2.96% | [0.75, 0.05] | 4.637% |
| sub_norm | 0 | 262144 | 5.3662 | 5.0183 | 2.6532 | 50.56% | 50.56% | [0.95, 0.95] | 7.195% |
| sub_norm | 1 | 262144 | 4.3862 | 4.3784 | 4.3454 | 0.93% | 0.93% | [0.5, 0.45] | 1.249% |
| sub_norm | 2 | 262144 | 5.4834 | 5.4458 | 5.289 | 3.55% | 3.55% | [0.6, 0.4] | 1.667% |
| sub_norm | 3 | 262144 | 4.7313 | 4.7262 | 4.7045 | 0.57% | 0.57% | [0.55, 0.3] | 0.356% |
| sub_norm | 4 | 262144 | 5.4727 | 5.4667 | 5.4449 | 0.51% | 0.51% | [0.3, 0.15] | 0.885% |
| sub_norm | 5 | 262144 | 4.8387 | 4.8367 | 4.8338 | 0.10% | 0.10% | [0.35, 0.1] | 0.288% |
| sub_norm | 6 | 262144 | 5.4902 | 5.4886 | 5.4856 | 0.08% | 0.08% | [0.15, 0.05] | 0.684% |
| sub_norm | 7 | 262144 | 5.0115 | 5.0105 | 5.0094 | 0.04% | 0.04% | [0.25, 0.05] | 0.278% |
| sub_norm | 8 | 262144 | 5.444 | 5.4435 | 5.4435 | 0.01% | 0.01% | [0.15, 0.0] | 0.519% |
| sub_norm | 9 | 262144 | 5.1821 | 5.1817 | 5.1817 | 0.01% | 0.01% | [0.15, 0.0] | 0.321% |
| sub_norm | 10 | 262144 | 5.4782 | 5.3256 | 5.3211 | 2.87% | 2.87% | [0.75, 0.05] | 4.67% |