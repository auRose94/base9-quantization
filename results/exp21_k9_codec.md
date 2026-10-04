# exp21 — the K9 file writer: real bytes, bit-exact decode

Qwen2.5-Coder-1.5B-Instruct, body k=63 + embedding k=99,
g=64, scales ent8 (spec §4 mode 4). Real container written
by `k9.py` (magic K9Q1), decoded back into the model.

- file size: **1112.0 MB** = 5.763 b/param (fp32 reference 6175 MB)
- entropy estimate: 1112.0 MB (delta +0.00%) — validates the accounting used
  in exp16-20
- encode 8s (185.32 M sym/s), decode+load 25s (61.11 M sym/s), coder: C (k9rans.so, byte-identical to rans.py)
- round-trip: **0 digit mismatches** over 1,543,714,304 weights (rANS is lossless); max weight difference 1.42e-02 comes only from the lossy `ent8` scale mode (fp32 scales are bit-exact — see `--selftest`)
- perplexity: in-memory 4.4995 → decoded 4.4995 code (max per-window Δ 3.70e-03)

Compare to the deployed formats measured in exp19 (same windows):
q4_k_m 1117.3 MB / +0.170 code Δ; q8_0 1894.5 MB / +0.086; NF4 999.5 MB
/ +0.270. K9 is RTN here (no GPTQ), so quality is the RTN point; the
GPTQ variant of the same grid added only ~0.5% bytes in exp20.

wall 62s
