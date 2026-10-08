# eq15 — decode-synchrony output (RQ9, docs/09)

subject: eq5_C_k63emb99.k9, M=2^16 per-step tables, CPU.

- P34 (PASS): rate 1.408 b/tok (marginal 1.3867) vs masked CE 0.964900016784668 nats/tok -> overhead 0.011 nats (bar 0.05).
- P35 (PASS): regeneration identical=True; single-bit flip diverges at token 747.
- P36 (PASS): dial monotone=True; tau=0.1 marginal = 0.1 b/tok (bar 0.05).

| temp | rate b/tok | marginal b/tok | masked CE nats/tok |
|---|---|---|---|
| 0.1 | 0.3 | 0.1 | 0.07850000262260437 |
| 0.3 | 0.6 | 0.4 | 0.29089999198913574 |
| 0.5 | 0.95 | 0.75 | 0.5268999934196472 |
| 0.8 | 1.55 | 1.35 | 0.9664999842643738 |
| 1.0 | 1.95 | 1.75 | 1.2217999696731567 |
| 1.5 | 3.4 | 3.2 | 2.2399001121520996 |
| 2.0 | 5.2 | 5.0 | 3.4749999046325684 |
| 4.0 | 7.25 | 7.05 | 4.910600185394287 |
| 10.0 | 8.65 | 8.45 | 5.881899833679199 |
| 50.0 | 9.15 | 8.95 | 6.2195000648498535 |

scripted story (1500 tokens, 264 bytes): `'there was a little girl named Lily. She lived in a big hive with her mommy. They loved to play with the icy flag were always happy to find a new salad.\nOne day, Lily asked her mommy if she could show her mommy, her mommy'`
