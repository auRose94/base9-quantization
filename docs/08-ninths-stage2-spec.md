# 08 — ninths native containers: stage-2 spec (FROZEN 2026-10-06)

Naming (2026-10-06, rose's call): the format is **ninths**; **K9** is the
secondary shorthand and the GGML type prefix; `.k9` stays the filename
extension of the entropy-coded research container. This doc freezes the
stage-2 resident container: the `GGML_TYPE_K9_4/K9_6/K9_7` types that serve
ninths weights with zero decode-on-load, plus the v1 kernel scope.

## 1. Numerics (the frozen core)

Reference dequant is normative in `experiments/k9.py`: `w = m·(d−H)/H`,
`H = (k−1)//2`, scale arithmetic f64→f32, digits uint8, one scale per
(row, 64-column group).

The native container bakes `H` and `/H` out of the runtime path at export:

```
code = d − H          signed integer, |code| ≤ 49 (2.6× headroom inside int8)
s    = fp16₁₆(m / H)  one fp16 scale per 64-group, round-nearest-even
w    = s · code       exact in fp32 (11-bit mantissa × int ≤ 49 < 24 bits)
```

Runtime kernels see plain (code, fp16-scale) operands — Q8_0 semantics, no
ninths math, and the type carries **no k at all** (k decides only the bit
width). Exactness carries over from the stage-1 theorem (docs/05 §2): fp16
scale rounding ≤ 4.9e-4 relative, finer than k99's grid spacing.

**Stage-1 equivalence (proved, Gate A):** the stage-1 loader materialized Q8_0
with the same `fp16(m/H)` scale and the same int8 codes, so
`unpack(native_bytes)` is bitwise identical to the C materializer's Q8_0
blocks. All docs/05 ppl rows therefore apply to the native containers with
zero additional measurement noise: 1.5B k63 → 13.90 @ 1.119 GB (rANS) ≡ native
blob bytes; 7B k63 9.253; 14B k63 7.823. Gate A proves the bitwise identity
end-to-end; §5 has the results.

## 2. Frozen type layouts

Superblock = **256 codes** along the stored row (= GGUF `ne[0]`, = QK_K),
holding 4 × 64-groups. Tensor blob = rows × (cols/256) superblocks, row-major,
matching GGML's data order (ne[0] contiguous). Tensor requirements:
`ne[0] % 256 == 0` (same as q6_K), group = 64, non-permuted (RTN) tensors only
— act-order permutations stay in the research pipeline, never in a native type.

| type | enum¹ | fields | bytes/block | b/param | k coverage |
|---|---|---|---|---|---|
| k9_4 | 44 | `sc[8] ql[128]` | 136 | 4.25 | k ≤ 15 |
| k9_6 | 45 | `sc[8] ql[128] qh[64]` | 200 | 6.25 | k ≤ 63 |
| k9_7 | 46 | `sc[8] cs[256]` | 264 | 8.25 | k ≤ 99 |

* `sc` — 4 × little-endian fp16 scales, one per consecutive 64-group.
* `ql` — two codes per byte, first in the **low nibble**, second in the high
  nibble, sequential per superblock. k9_4: nibble = `code + 8`. k9_6: low
  half of the unsigned value `u6 = code + 32` (so k9_6 codes span [−31…31] →
  u6 ∈ [1…63]).
* `qh` — k9_6 only: byte j holds the 2-bit high halves of u6 for codes
  (4j…4j+3) in bit pairs 0-1 / 2-3 / 4-5 / 6-7.
* `cs` — k9_7: signed int8 = code.

¹ Fork-provisional ids (written into the fork's gguf-py alongside the 2026-10-05
sentinel `K9 = 43`). **Upstream will renumber at merge time** — the PR must not
hard-claim these ids.

Deliberate choices to defend in review:
- Sequential nibble packing, **not** the k-quant SIMD interleave: kernels are
  ours; numpy packing stays vectorizable (the packer is ~30 lines/type and
  bit-exact).
- Single exact fp16 scale per 64-group, **not** q6_K's two-level fp16×int8:
  the exactness theorem needs the bare fp16 scale; q6_K's scale encoding is
  lossy and would break the bitwise-equal-to-stage-1 property.
- K9_7 exists so k99 embeddings ride the native family instead of being
  re-stuck on Q8_0; it is a `vec_dot_q8_0`-style kernel with wider groups.

## 3. Why: what changes vs stage 1 (the serving story)

Stage 1 (decode-on-load) is a **disk-format** win: runtime memory = Q8_0
(8.5 b/param) and llama-bench = Q8_0's. Native containers make the win
resident:

| | Q8_0 resident | K9_6 native | file (rANS, stage 1) |
|---|---|---|---|
| rate (body) | 8.5 b/param | 6.25 b/param | ~5.7 b/param |
| 14B weights | ~15.0 GB | ~11.0 GB | 11.2 GB |

Two kernels: **CPU vec_dot** (upstream's CPU-first policy makes it mandatory)
and **CUDA/HIP MMVQ** (the batch-1 serving path; memory-bound on weights →
8.5/6.25 ≈ 1.36× traffic ceiling; measured upside expected 1.2–1.3× since the
1.5B Q8_0 path sits at ~46% of the 5060 Ti's bandwidth). The dot is dp4a- and
mma-friendly: `Σ aᵢ·(dᵢ−H) = Σ aᵢ·dᵢ − H·Σ aᵢ` — a bias-corrected int8 dot,
one fp16 scale per 64.

**Honest trade (measured, not negotiable):** fixed-width native containers
give up rANS's compression on the small-k tier. 7B tuned k15: rANS stream
3.5 GiB vs K9_4 native 3.23 GiB body + 1.05 GiB K9_7 embed/lm-head ≈ 4.28 GiB
(+31%): k15 digits are concentrated near the grid center, and order-0 rANS
reaches ~2.9–3.8 b/param there. The sub-4-bit/low-k tier stays a `.k9`
research contribution (q2_k-beating small files); stage 2's upstreamable tier
is the near-lossless one (k63/k15-class quality at 4.25/6.25 b/param), where
native costs only ~4% over rANS at k63.

## 4. v1 scope / deferred

**v1 ships:** ggml.h/ggml.c type table rows; CPU `vec_dot` (+ quantize/dequant
row fns for llama-quantize); CUDA MMVQ (`vecdotq.cuh` + `mmvq.cu` dispatch +
`getrows.cu`/`convert.cu` glue for embedding lookup and the prefill F16
fallback); gguf-py constants + `general.file_type` entries; `llama-quant.cpp`
RTN; `test-backend-ops` type lists.

**Deferred (separate PRs/never):** CUDA MMQ (prefill, ten per-arch
`mmq-config-*.cuh` files); Vulkan/Metal/SYCL/OpenCL/OpenVINO/Hexagon/WebGPU/
AMX/repack (upstream tolerates missing backends for new types — recent
ternary-quant additions landed CPU+CUDA first); GPTQ-permuted tensors in a
native type (rejected at export); sub-4-bit variants.

**Explicitly unchanged:** KV cache/attention/sampling (never weight-type
aware); the loader sentinel path (`GGML_TYPE_K9 = 43`, `ninths.directory`,
decode-on-load) stays for the rANS research layer side-by-side.

## 5. Gate A results (2026-10-06, this session)

- Packer selftest (`experiments/ninths_native.py`): 24/24 cases (k ∈
  {9,11,13,15→K9_4; 17,27,63→K9_6; 99→K9_7}) — integer roundtrip exact,
  fp16-scale dequant exact vs reference, **bitwise equal** to the stage-1
  numpy Q8_0 materialization.
- **1.5B real model**: all 197 k9 tensors (k63 linears ×196 + k99 embed)
  packed native and dequantized **bitwise equal to the C materializer's**
  Q8_0 blocks (`base1p5b_k63_materialized_q8_0.gguf`), including the ent8
  scale-mode path → docs/05 ppl rows carry over verbatim. Exported
  `results/base1p5b_native_k63.gguf` = **1.18 GiB** payload
  (k9_6 0.953 + k9_7 0.224), vs rANS stage-1 1.04 GiB (+13%), q4_k_m 1.117 GB,
  q8_0 1.895 GB.
- **Tuned-7B k15 real model**: all 198 tensors (k15 linears ×196 + k99 embed +
  untied k99 lm_head) packed K9_4/K9_7, dequantized **bitwise equal to the
  C-chain Q8_0 arm** (`tuned7b_k15_materialized_q8_0.gguf`, python
  materializer whose byte layout was C-verified in session 24 → docs/05 9.623
  ppl row carries over). Exported `results/tuned7b_k15_native.gguf` =
  **4.28 GiB** payload (k9_4 3.228 + k9_7 1.047) vs rANS stage-1 3.5 GiB
  (+31% — the small-k-tier trade of §3, measured).
- **Tuned-14B k63 real model**: all 338 tensors exported native —
  `results/tuned14b_k63_native.gguf` = **11.11 GiB** payload (k9_6 9.613 GiB =
  6.26 b/param body incl. norms, k9_7 1.496 GiB embed + untied lm_head) vs
  rANS stage-1 9.76 GiB (+14%) and q8_0 file 14.62 GiB (−24%). Numerics follow
  from the k63/K9_6 proof above (same k, same code path).
- Fork loader KV rename verified via `k9_llamacpp_gate.py` + `llama-k9probe`:
  legacy `k9.directory` GGUF **PASS**, re-exported `ninths.directory` GGUF
  **PASS** (digits bit-exact, scales 0 ulp, Q8_0 byte-exact).

Exporter: `export_k9_gguf.py --native` (this file's layouts; gguf-py entries
`K9_4/6/7 = 44/45/46` with sizes 136/200/264 @ block 256). Gate script:
`ninths_gate_a.py`; packer: `ninths_native.py`.

## 6. Step 2 — fork CPU-side kernels (2026-10-06, same session)

Fork touch-points (Q1_0/Q2_0 = the newest-type template; vec_dot pairs with
**Q8_0 activations**, the same convention those types use — our per-64 weight
scale maps to two 32-code Q8_0 activation blocks):

- `ggml-common.h`: `block_k9_4/6/7` + `QK9_SUPER 256`/`QK9_GROUP 64`
  (own constants, deliberately NOT `QK_K` — that varies by platform).
- `ggml.h`: enum 44/45/46 + `GGML_TYPE_COUNT 47` + `GGML_FTYPE_MOSTLY_K9_4/6/7`
  = 29/30/31.
- `ggml.c`: type-table rows; `ggml_quantize_chunk` cases; ftype→wtype map.
- `ggml-quants.{h,c}`: `dequantize_row_k9_*`, `quantize_row_k9_*_ref` (RTN:
  `s = fp16(max(amax/H, 1e-4f))` per 64-group, `code = clamp(round(x/s))` —
  fp16 floor 1e-4 instead of GROUP_MAX_EPS, since our scale is fp16 and a
  1e-15 eps would underflow to 0 and divide-zero), chunk `quantize_k9_*`
  (imatrix ignored, v1 RTN).
- `ggml-cpu/ggml-cpu.c` traits rows (`.vec_dot_type = GGML_TYPE_Q8_0`,
  nrows 1); `ggml-cpu/quants.{h,c}`: row wrappers + `ggml_vec_dot_k9_*_q8_0`
  (generic only); `arch-fallback.h` aliases in every arch block; `ops.cpp`
  adds the types to the 7 canonical quantized-type switch groups (add, add1,
  acc, out_prod, set, get_rows, clamp — bodies are generic via `to_float`,
  and `get_rows_q` drives embedding lookup from the type table alone).
- llama-side: `LLAMA_FTYPE_MOSTLY_K9_4/6/7` = 42/43/44 (`llama.h`), ftype
  name strings + GGML→FTYPE mapping (`llama-model-loader.cpp`),
  `tensor_type_fallback` (ncols % 256 ≠ 0 → **Q8_0**) + ftype→qtype cases
  (`llama-quant.cpp`).

**Gate B results (all PASS, this session):** fork CPU build **PASS**; upstream
`test-quantize-fns` **PASS** for k9/k9_4/k9_6/k9_7 (default bounds suffice,
types auto-enumerated); `test-backend-ops -b CPU` full suite **PASS** (1/1
backends, 0 fail, k9 types in `all_types`); `llama-cli` smokes on the 1.5B and
tuned-7B-native GGUFs: load and generate coherently through the K9_6/K9_7/K9_4
CPU vec_dots (**ftype label shows F16** — the exporter copies the reference's
`general.file_type` KV; cosmetic, same as the stage-1 caveat); end-to-end ppl
identity on 16 identical chunks: native **16.7452 ± 0.825** vs materialized
Q8_0 **16.7439 ± 0.825** — agreement to 5 significant digits, consistent with
f32 summation-order noise given bitwise-equal weights (expectation note: the
dot implementations differ, so bit-equality of ppl output is NOT required —
only weights are bitwise).
**Known v1 cost (llama-bench, CPU 8 t, 1.5B):** scalar generic dots →
**pp512 9.8 t/s vs q8_0 materialized 483** (~49×, VNNI-accelerated q8_0 path),
**tg128 8.4 vs 15.95** (~1.9×). Correctness-first as intended; the fix list:
AVX2/VNNI CPU dots (structure-clone of the q8_0 kernel: unpack codes to int8
lanes, then the same dp4a/VNNI reduction) and the real speed target, CUDA/HIP
MMVQ (step 3, up to the 1.36× traffic ceiling). CUDA note: native types have
no GPU kernels yet (MMVQ = step 3); do not GPU-offload native GGUFs until
then.