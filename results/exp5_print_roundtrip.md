# exp5 — the model, printed as repeating numbers (round trip)

Model roneneldan/TinyStories-33M: 28,311,552 weights across 24 body weight
matrices quantized to the ninths grid (per-row scale m, w = (s−4)·m/4,
s ∈ {0..8}). Eval: 300 blocks × 1024 tokens, TinyStories validation.

- fp32 baseline perplexity: **40.663**
- after ninths quantization: **93.333** (linear rel MSE 0.07102)
- after printing to printed_model.txt and re-parsing: **93.333**

## Round-trip assertions (all passed)

- all digits + scales + dequantized weights bit-exact across 24 matrices / 28,311,552 weights
- perplexity after fp32-restore + printed-file reload identical: 93.333475

## Size ladder

| representation | size (linear weights) | lossless? |
|---|---|---|
| fp32 tensors | 113.2 MB | reference |
| printed repeating decimals (this file) | 142.2 MB | yes (round trip above) |
| same, gzip'd | 12.9 MB | yes |
| compact digits (1 char/digit + scales) | 57.3 MB | yes |
| rANS binary (digits) + fp32 scales | 8.1 MB | yes |
| uniform-9 alphabet bound (log2 9 + scales) | 11.3 MB | information bound under uniform symbols; skew lets rANS beat it |

The printed file is the model: it reloads bit-exactly and reproduces
perplexity exactly. Its text size is large (notation overhead D1 in
docs/01) but gzip + digit-frequency coding recover the content; the
rANS row is the honest compressed-size comparison.
