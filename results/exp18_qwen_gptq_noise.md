# exp18 — GPTQ on the grids at 1.5B, with a noise floor

Qwen2.5-Coder-1.5B-Instruct; 8 disjoint windows x 1024 tok
per corpus (WikiText-103 test, codeparrot-clean-valid); calibration
128 x 1024 in-domain Python (train); GPTQ damping 0.05,
act_order=True. Mean +/- SE over windows;
deltas are PAIRED against fp32 on identical windows. Hessians are
collected once on the fp32 model (see docstring).

| scheme | k | GPTQ | digits b/p | MB | wiki ppl | code ppl | Δwiki | Δcode |
|---|---|---|---|---|---|---|---|---|
| fp32 baseline | — | — | 0.000 | 6174.9 | 14.064 ± 2.336 | 4.492 ± 0.366 | +0.000 ± 0.000 | +0.000 ± 0.000 |
| 8-level g64 RTN | 8 | — | 2.488 | 1357.2 | 28.120 ± 5.134 | 6.952 ± 0.619 | +14.057 ± 2.833 | +2.460 ± 0.325 |
| 9-ninths g64 RTN | 9 | — | 2.670 | 1387.0 | 19.408 ± 3.634 | 5.599 ± 0.445 | +5.345 ± 1.313 | +1.107 ± 0.127 |
| int4-15 g64 RTN | 15 | — | 3.452 | 1515.1 | 15.121 ± 2.528 | 4.855 ± 0.396 | +1.057 ± 0.213 | +0.364 ± 0.066 |
| 99-level g64 RTN | 99 | — | 6.218 | 1968.0 | 14.110 ± 2.338 | 4.493 ± 0.364 | +0.047 ± 0.011 | +0.001 ± 0.004 |
| 9-ninths g64 GPTQ | 9 | yes | 2.693 | 1390.7 | 19.582 ± 3.629 | 5.150 ± 0.403 | +5.518 ± 1.329 | +0.658 ± 0.106 |
| int4-15 g64 GPTQ | 15 | yes | 3.468 | 1517.7 | 15.247 ± 2.588 | 4.641 ± 0.364 | +1.183 ± 0.264 | +0.149 ± 0.038 |
| K9 full (body9-GPTQ + embed99) | 9 | yes | 3.228 | 641.9 | 19.603 ± 3.642 | 5.149 ± 0.402 | +5.539 ± 1.342 | +0.657 ± 0.106 |

A delta whose magnitude is within ~2 SE of zero is parity, not a win.
The GPTQ rows are the test of whether compensation closes the gap to
int4 at matched bits on a real code model.

wall 740s
