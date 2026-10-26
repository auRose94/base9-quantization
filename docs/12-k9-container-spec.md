# ninths (K9) stream container — specification v1.0 (draft)

Status: DRAFT-1.0, 2026-10-26, session 32. Supersedes docs/04 §5.2 (stale).
Normative references: `experiments/k9.py` (codec + container),
`experiments/rans_fast.c` (fast coder + iq4_nl packer), fork `src/k9.cpp`
(loader-side decode) and `examples/k9probe/k9probe.cpp` (gate tool). Where
this text and the reference code disagree, the code wins and this document
must be corrected.

The container packages a quantized model as **byte streams**, one blob per
weight tensor, with a per-tensor directory carried in GGUF metadata. The
runtime materializes blobs to a concrete GGML quantization at load time; the
materialization for Q8_0 is exact (see §4).

## 1. GGUF layout

- K9 tensors are stored as **1-D blobs**: GGUF type id 43 (`GGML_TYPE_K9`,
  block_size 1, type_size 1), declared `ne[0] == blob_len`. The REAL shape
  lives in the directory record; loaders must build the tensor at the model's
  true rank (rank-3 expert tensors included) and hand the decoder the
  flattened 2-D view `(rows = out·n_expert, cols = in)`.
- All other tensors (norms, biases, routers, small tensors) are ordinary
  GGUF tensors, copied through unchanged.
- Metadata keys:
  - `ninths.directory2` (v2) / `ninths.directory` (v1) / `k9.directory`
    (legacy alias) — u8-array KV, the directory (§2).
  - `ninths.materialize` / `k9.materialize` — string KV, default
    materialization target: `q8_0` | `f16` | `iq4_nl`. An environment
    override (`K9_MATERIALIZE`) takes precedence over the KV; the KV takes
    precedence over the default.

## 2. Directory

Little-endian byte array:

```
v1: u16 n_records
v2: u16 version (=2), u16 n_records
then per record:
    u16 name_len, utf8 name
    u8  rank (== 2), u32 dims[rank]        # stored-frame (rows, cols)
    u8  k, u8 group, u8 scale_mode, u8 perm_flag
    v2 only: u8 lanes                       # 1 = v1 layout
    u32 tbl, u32 dig, u32 sca, u32 plen     # section byte-lengths

Blob placement is the GGUF tensor entry itself (declared byte length);
records carry no offset field. The standalone .k9 file format keeps its own
container header (see k9.py) and is not part of the GGUF contract.
```

Constraints: `k` odd, k ≥ 9, k ≤ 255; `group` = 64 (normative for v1/v2);
`permitted scale_mode` ∈ {0 = fp32, 1 = fp16, 4 = ent8}; `tbl + dig + sca +
plen == blob_len`.

## 3. Blob layout

```
[freq table: k × u16 LE, Σ == M = 4096]
[digit stream]          # v1: one lane; v2: lane table + L streams
[scales blob]           # format per scale_mode (§4.2)
[perm: plen bytes]      # optional: u32 LE per stored column, full width
```

**Digit stream, v1** (single lane): a rANS coder with 32-bit state in
[2^24, 2^32), M = 2^12 = 4096, byte-renorm. The stream's final 4 bytes are
the initial state, big-endian; renormalization bytes are consumed backwards
from `slen−5`. Decoder emits digits in stored (row-major) order.

**Digit stream, v2** (lane-interleaved): `u32 lane_off[L+1]` (relative to the
sub-stream area, ascending, `lane_off[L] == dig − 4·(L+1)`), then L
single-lane v1 sub-streams back-to-back, each with its own 4-byte state
trailer. Lane `i` owns stored rows `[i·ceil(rows/L), (i+1)·ceil(rows/L))`.
All lanes share ONE frequency table; per-symbol semantics are identical to
v1, so a v2 stream decodes bit-identically to the v1 stream of the same
digits. Encoder-side rule: `L = max(1, min(16, digit_bytes >> 21, rows))`.

## 4. Quantization and materialization

### 4.1 Grid (normative)

Per stored row and 64-column group, with group max-magnitude `m` and
`H = (k−1)/2`:
`W = m · (d − H) / H`, digits `d ∈ [0, k−1]`. The grid is symmetric and odd,
so `d == H` reproduces 0 exactly.

### 4.2 Scale encoding

- mode 0: raw f32; mode 1: f16.
- mode 4 (`ent8`): 10-byte header (f32 lo, f32 hi, u16 table_len = 256) +
  256×u16 rANS table + byte-stream of quantized level indices; the exponent
  path computes `exp2(lo + (hi−lo)·code/255)` in **float64**, then casts.
  Encoder normalizes 256 histogram buckets onto M = 2^12 (mismatch fixup
  walks the largest buckets first, subtracting, then adding).

### 4.3 Materialization targets

- **Q8_0 (exact).** Per 64-group, two Q8_0 blocks of 32 weights sharing
  `d = fp16(m/H)` (round-half-even), codes `= digit − H` packed as i8.
  Bit-exact: no arithmetic beyond representation changes.
- **F16 (exact).** Required for permuted tensors (perm_flag ≠ 0): value =
  `m·(d−H)/H` in fp32, converted per element with the permutation applied.
  Permuted tensors must NOT materialize to Q8_0 (blocks scatter).
- **IQ4_NL (lossy, deterministic).** One RTN pass onto the 16-value codebook
  `kvalues_iq4nl`, per 32-weight block: `d = amax/113` stored as f16 (RNE);
  codes = first strict minimum of `|w/d − v|` over the table; nibble packing
  `qs[j] = L[j] | (L[j+16] << 4)`; amax < 1e-8 blocks emit `d = 0` and
  code-index 8 in every nibble. The K9→IQ4_NL step recomputes `w` from the
  exact grid, so quality is bounded by the coarser target, not the stream.
  **k ≥ 99 aux tensors (token_embd, output head) stay exact-Q8_0** when the
  global target is IQ4_NL.
- No scale-search optimization at materialization: implementations must stay
  byte-deterministic (this is testable; an optimizer is a future, separately
  versioned feature).

## 5. Conformance suite (gate ladder, in run order)

1. **Codec round-trip** — random + real digit streams over k ∈ {9, 63, 99},
   skewed histograms with zero-count symbols, non-64-divisible row counts,
   L ∈ {1, 2, 3, 16, 64}: decode(encode(x)) == x and per-lane decode ==
   single-lane decode.
2. **Encoder identity** — the streaming path (chunked histogram + incremental
   renormalization) is byte-identical to the monolithic one.
3. **Loader byte-exactness** — per tensor class (2-D attn, rank-3 expert,
   k99 embed): probe-extracted digits / scales / materialized bytes match the
   reference implementation bit-for-bit (0 mismatches, ≤ 2 ulp on ent8
   scales).
4. **Threading determinism** — 1-worker vs N-worker materializations are
   byte-identical (covers both lane-parallel decode and row-parallel pack).
5. **Loud-fail contract** — a v2 file presented to a v1-only reader MUST
   error, never misdecode; malformed directory records likewise.

## 6. Versioning policy

- New capabilities that change byte layout get a NEW metadata key (vN
  suffix) or a new record field with a version header; old readers must fail
  loudly, never misread.
- Legacy aliases (`k9.*`) are accepted forever; new writers never emit them.
- Materialization targets are additive (q8_0, f16, iq4_nl today).

## 7. Implementation register

| component | path | role |
|---|---|---|
| k9.py | base9-quantization/experiments | normative codec + directory |
| rans_fast.c/.py | base9-quantization/experiments | fast rANS + iq4_nl packer (ctypes) |
| src/k9.cpp, k9.h | llama.cpp fork, branch k9 | loader-side decode + materialize |
| llama-k9probe | fork examples/ | gate tool: dump digits/scales/weights |
| glm_k9_build.py | base9-quantization/experiments | streaming 2-pass builder + stock exporter |

Open items: KoboldCpp port parity for v2 (currently loud-fails on v2 files —
acceptable, noted for publish readiness); per-tensor materialize maps;
GPTQ act-order support is F16-only by design.