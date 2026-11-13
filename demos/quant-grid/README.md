# Quantization, visually

The classic "what does quantization do" grid, rebuilt for the ninths (K9)
project: the same image is passed through real model-quantization recipes —
the standard formats everyone ships, and the ninths digit containers from
`docs/12-k9-container-spec.md` — with the visible differences kept intact.

| file | contents |
|---|---|
| `grid_classic.png` | FP64, FP16, FP8 (E4M3), INT8, FP4 (E2M1), INT4, 1.58-bit, 1-bit |
| `grid_ninths.png` | source, k99, k63, k15, k9, k5*, k3*, k9 digit map |
| `grid_error.png` | all of the above as \|error\| × 64 in linear light |
| `source_r2head.png` | source crop, Robonaut 2 (NASA, public domain) |
| `make_grid.py` | reproduces everything: `python3 make_grid.py` |

## Pipeline

- **Linear light.** The image is decoded from sRGB to linear and quantized
  there — the honest analogue of weight quantization. Images usually get a
  perceptual encoding (gamma) for free; model weights do not, and the
  shadow-crushing that INT formats show here is exactly what happens to
  small-magnitude weights under a per-tensor scale.
- **Round-to-nearest, no dithering.** Artifacts are the format's own.
- **Classic formats use one scale per image (per-tensor)** — the naive
  baseline the classic grid demonstrates. FP formats need no scale: the
  exponent self-adjusts, which is the entire float-vs-int story.
- **ninths panels use the spec §4.1 grid**: per-64-value group
  max-magnitude scale `m`, digits `d ∈ [0, k−1]`, dequantized as
  `W = m·(d−H)/H`, `H = (k−1)/2`. One disclosed addition: pixels are
  non-negative while weights are signed, so each group is mean-centered
  before gridding.
- Quantization happens at display resolution (768²), so every shown pixel
  is one quantized value — no downscale smoothing.
- `rANS ≈ … b/v` is the measured order-0 entropy of the digit stream (one
  shared frequency table per tensor, as in the container). The same
  histogram is what the real codec compresses; on real weight tensors this
  is where K9's margin over `log2(k)` comes from.

## Format recipes

| format | recipe | b/v |
|---|---|---|
| FP64 | identity (reference) | 64 |
| FP16 | IEEE half round-trip | 16 |
| FP8 | E4M3 (`fn` variant, ±448 max), nearest in the 126-value table, scale-free | 8 |
| INT8 | per-tensor asymmetric [0, 255], `s = amax/255` | 8 |
| FP4 | E2M1 (±{0,.5,1,1.5,2,3,4,6}) with one per-tensor scale `amax/6` (MXFP4-style) | 4 |
| INT4 | per-tensor asymmetric [0, 15], `s = amax/15` | 4 |
| 1.58-bit | BitNet b1.58-faithful: `γ = mean(x)`, digits {−1, 0, +1}·γ | 1.58 |
| 1-bit | centered sign, `c = mean(x)`, `s = mean(|x−c|)` | 1 |
| ninths k | §4.1 grid, per-64-group, mean-centered | log2 k |

k5/k3 are extreme rungs outside spec v1.0 (k ≥ 9) — published in the flint
rung at 1.95/1.11 b/p after entropy coding, included here to show the low-k
cliff. `1.58 b/v = log2(3)`: the classic ternary panel and ninths k3 are the
same bit-width, which makes their difference pure scale-and-grouping policy.

## Reading guide

- **FP64 and FP16 are display-identical here** — the source is an 8-bit
  photo and FP16's 11-bit significand sits below 8-bit visibility (measured:
  max relative error 4.6e-4 ≈ 2⁻¹¹). The error sheet exists to make
  sub-visible residuals comparable: FP64 is exactly black, FP16 is 3.8/255
  at ×64 gain, FP8/INT8 banding and the ternary/binary collapse are obvious.
- **INT8 has the higher PSNR (46.7 vs FP8's 42.4) but the wrong error
  character**: its absolute step is uniform, so shadows crush toward flat
  black while midtones stay clean. FP8's error is *relative* — chunky in
  the shadows but proportional everywhere. PSNR alone doesn't rank formats;
  the structure of the residual does.
- **1.58-bit vs k3\* is the ninths pitch in one glance**: same 1.58 b/v,
  but a per-group grid chosen to cover the data range stays readable where
  a naive absmean scale collapses the image to a few tones.
- **The k-ladder shows the group mechanism**: k99/k63 are near-clean at
  6.6/6.0 b/v, and the error sheet makes the 64-group blockiness visible —
  the same structure that calibrated scales + entropy coding exploit on
  real weights.
- The **k9 digit map** is the actual encoder input: digits d ∈ [0, 8] of
  the luma plane (2.93 b/v after coding). Everything the container stores
  beyond the digit stream is scales, a shared 9×u16 frequency table, and
  the rANS state.

## Source

Robonaut 2 aboard the ISS, NASA `iss026e034306` (public domain), cropped
1800² and resized to 768². The pipeline is float64 end to end; the source
photo is 8-bit, so FP16/FP64 cannot differ visibly — a true 16-bit source
(render or scan) would separate them in the main grid too.
