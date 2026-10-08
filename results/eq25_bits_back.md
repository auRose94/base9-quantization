# eq25 — bits-back chain (RQ12, docs/09 P45/P46)

subject: eq5_C_k63emb99.k9 (36 tensors, 265728 digits); machine: base-9 M=9^4/L=9^12; one latent draw per digit; posteriors q = p^beta, beta in {1, 1.5, 2}; CPU.

- **P45 (FAIL)**: static-b9 (same machine, no pops) = 470639 trits = 186591 B; container file on disk = 202081 B. The bb file = static + the pops' absorb stream + the realized KL (identity measured exact at all three betas): on a deterministic artifact the pop channel carries its randomness, it does not save.

| kind | beta | bb bytes | delta trits | realized KL bits | E[KL] bits | pops | decode | P45 bar |
|---|---|---|---|---|---|---|---|---|
| clean | 1.0 | 352885 | +419458 | 0.0 | 0.0 | 419458 | ok | FAIL |
| clean | 1.5 | 334893 | +374074 | 131422.79 | 19744.93 | 332615 | ok | FAIL |
| clean | 2.0 | 334945 | +374207 | 226582.45 | 69141.56 | 302728 | ok | FAIL |
| recycle | 1.0 | 186593 | +4 | 0.0 | 0.0 | 380601 | FAILED | n/a |
| recycle | 1.5 | 186626 | +89 | 131415.03 | 19744.93 | 332812 | FAILED | n/a |
| recycle | 2.0 | 186459 | -334 | 226452.99 | 69141.56 | 302335 | FAILED | n/a |

- **P46 (PASS)**: 3 posteriors; every clean chain roundtrips (payload exact, mirror returns x0, draws roundtrip, weights identity, chain story = reference story); the file leg rebuilds weights + recovered draws from the single self-contained demo file; byte-flip corruption 5/5.

Recorded negative control (the paper's own chaining, pops absorb = the prior appends' leftovers): the encoder-side recycling is not mirrorable -- the decoder's pops need more absorb digits than the recycle file carries, because that stream is exactly what the file must contain for the mirror to invert. Reason per beta:

- recycle/beta=1.0: clean_served 0, residue 80, 186593 B -- `tensor 35 digits diverge`
- recycle/beta=1.5: clean_served 0, residue 41452, 186626 B -- `tensor 35 digits diverge`
- recycle/beta=2.0: clean_served 0, residue 71856, 186459 B -- `tensor 35 digits diverge`

**clean/beta=1.0 seed=1844293402179826231** (mirror: reversed, 419458 popped; draws 265728): `there was a little girl named Lily. She loved to play outside in the sun and go seek with her friends. One day, her mommy asked her to clean her room. The lady said, "Lily, you can find a toy toy and …`

**clean/beta=1.5 seed=1954182715680064933** (mirror: reversed, 332615 popped; draws 265728): `there was a little girl named Lily. She loved to play outside in the park with her plate. One day, she saw something empty and decided to jump on the door. She ran to her house and saw a puppy stuck o…`

**clean/beta=2.0 seed=5232366278238683222** (mirror: reversed, 302728 popped; draws 265728): `there was a little girl named Lily. She loved to play outside in her garden. One day, she saw a big, scary goat's legs and wanted to go to the goal.
Lily's mom said, "Don't worry, Lily. I don't want t…`
