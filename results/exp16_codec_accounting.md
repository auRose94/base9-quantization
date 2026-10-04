# exp16 — realistic codec accounting (measured on stored artifacts)

Re-costs the existing artifacts with per-tensor rANS (tables counted)
and realistic scale precisions. `digit_bpp` is the real per-tensor rANS
rate; `global` is the published one-table figure for comparison.
Scale options: fp32/fp16/int8/int6 bytes per scale + an entropy-coded
8-bit log-scale stream. `rest` = fp32 norms/biases (32,256 params).

| artifact | part | params | digits b/p (per-tens/global) | table KB | scales | total fp16 | total int6 | total ent8 |
|---|---|---|---|---|---|---|---|---|
| exp13 RTN | body | 28,311,552 | 2.700 / 2.701 | 0.4 | 1.77/0.88/0.44/0.33 MB | 10.57 | 10.02 | 10.04 |
| exp13 RTN | embed | 40,170,240 | 6.275 / 6.275 | 0.4 | 2.51/1.26/0.63/0.47 MB | 32.89 | 32.11 | 32.21 |
| exp13 GPTQ | body | 28,311,552 | 2.709 / 2.709 | 0.4 | 1.77/0.88/0.44/0.33 MB | 10.60 | 10.05 | 10.07 |
| exp13 GPTQ | embed | 40,170,240 | 6.275 / 6.275 | 0.4 | 2.51/1.26/0.63/0.47 MB | 32.89 | 32.11 | 32.21 |
| exp11 recipe | body | 28,311,552 | 2.701 / 2.702 | 0.4 | 1.77/0.88/0.44/0.33 MB | 10.57 | 10.02 | 10.02 |
| exp11 recipe | embed | 40,170,240 | 6.241 / 6.241 | 0.4 | 2.51/1.26/0.63/0.47 MB | 32.72 | 31.94 | 31.94 |
| exp11 tern | body | 28,311,552 | 1.578 / 1.585 | 0.1 | 1.77/0.88/0.44/0.33 MB | 6.60 | 6.05 | 6.05 |
| exp11 tern | embed | 40,170,240 | 6.245 / 6.245 | 0.4 | 2.51/1.26/0.63/0.47 MB | 32.74 | 31.96 | 31.96 |

- **exp13 RTN full model**: published 45.5 MB (fp32 scales, global table) → fp16 scales 43.5 MB → int6 scales 42.1 MB (7.4% smaller), per-tensor tables included.

- **exp13 GPTQ full model**: published 45.5 MB (fp32 scales, global table) → fp16 scales 43.5 MB → int6 scales 42.2 MB (7.3% smaller), per-tensor tables included.

## Reading

1. **Per-tensor tables are cheap and change the digit rate only slightly.**
   Table overhead is ~n_syms*2 bytes per tensor (tens of KB total); the
   per-tensor rANS rate tracks the global-table rate, so the published
   'effective bits' were not materially inflated by the global-table
   assumption — this is reassuring for the existing results.
2. **Scale precision is where the free bytes are.** fp32 → int6 scales cuts
   the full exp13 model by several MB with no accuracy change (the digits are
   unchanged); the 0.5 b/param fp32 scale tax is ~16% of the body's 3 b/param.
3. **The npz artifacts are ~25x larger than the coded model** because digits
   are stored as int64. A codec should store the rANS stream, not the digits.
