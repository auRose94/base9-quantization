# exp22 — asymmetric (min+scale) grids vs symmetric absmax

Qwen2.5-Coder-1.5B-Instruct, body k and embedding k=99, group=64.
`sym` = current K9 (symmetric absmax, 1 scale/group); `symclip` = symmetric
with the range clipped at the 0.9995 quantile of |w|; `asym` = affine
min..max (2 scales/group: lo fp16 + step ent8). Digits on the same odd
alphabets. Bytes measured with the K9 ent8 scale codec.

| variant | k | digits b/p | scales b/p | MB | code ppl | Δcode vs fp32 |
|---|---|---|---|---|---|---|
| sym | 9 | 3.208 | 0.096 | 637.7 | 5.598 | +1.107 ± 0.127 |
| symclip | 9 | 3.213 | 0.097 | 638.7 | 5.599 | +1.107 ± 0.134 |
| asym | 9 | 3.391 | 0.347 | 721.3 | 5.336 | +0.844 ± 0.124 |
| sym | 15 | 3.872 | 0.096 | 765.8 | 4.853 | +0.362 ± 0.065 |
| symclip | 15 | 3.877 | 0.097 | 766.8 | 4.855 | +0.363 ± 0.061 |
| asym | 15 | 4.051 | 0.347 | 848.7 | 4.683 | +0.191 ± 0.042 |
| sym | 27 | 4.617 | 0.096 | 909.5 | 4.571 | +0.079 ± 0.014 |
| symclip | 27 | 4.622 | 0.097 | 910.5 | 4.563 | +0.071 ± 0.014 |
| asym | 27 | 4.789 | 0.347 | 991.0 | 4.568 | +0.076 ± 0.018 |

## Reading

- Compare variants **at the same k** (digits ~equal): the difference is
  the grid shape. Then compare at matched **MB** (asym pays 2 scales/group).

wall 113s
