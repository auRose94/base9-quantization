# K9 stream format v2 — lane-interleaved rANS (design note)

Status: **IMPLEMENTED & GATED** (2026-10-25, session 32) — fork commit
`4c99facb9` (decoder + loader + probe) and `25d995288` (stage-2 types, landed
same push); encoder in `experiments/k9.py` + `glm_k9_build.py`. Verified:
60-config codec round-trip, mini-GGUF and full-GGUF byte-exact gates at 16
lanes; `glm45air_k63_e99_v2.gguf` published-ready. Originally proposed
as the next work item and built the same evening. Normative v1 reference stays `experiments/k9.py` + the fork's
`src/k9.cpp` (see docs/04 for lineage; its §5.2 byte layout is superseded by
that code).

## 1. Motivation (measured, not hypothetical)

GLM-4.5-Air k63/e99 on the A9 Mega loads in ~9.4 min. Instrumentation
(results/glm45air_k63_smoke.md, improvement round) shows the load is
**decode-bound**: the per-tensor digit stream is a single sequential rANS lane
(backward byte-renorm, resumable forward), running at ~130–165 MB/s, and
~74 GB of digit streams must pass through that one state. The row-packing
stage is already embarrassingly parallel (session 32 threaded it — gates
byte-exact) but saves only ~1.6 % wall because decode dominates.

Splitting one tensor's digits across N independent rANS lanes decodes N×
faster with no format-level entropy cost: lanes are independent states over
the SAME frequency table, so bit-exactness per symbol is preserved by
construction and every existing invariant (ent8 scales, perm-flag F16
fallback, materialize=q8_0 theorem) carries over untouched.

Target: GLM-4.5-Air load 9.4 min → ~1.5–2.5 min at 16 lanes (decode ~31 s +
pack ~45 s overlapped + GGUF IO), with encode-side pass1 gaining similarly
(exp23 measured 2.2× at 64 threads for tensor-level parallelism; lane-level
applies the same trick *within* the 500 MB blobs, which is where the minutes
really are).

## 2. v1 recap (what must not change)

Per-tensor blob = `[freq table: k×u16 LE, Σ == M = 4096]` + `[digit rANS
stream]` + `[scales blob]` + `[perm]`. Single lane: initial state in the final
4 bytes (BE), renorm bytes consumed backwards from the end, decode emits
digits forward in stored row-major order. Decompress identity:
`W[row, col] = m[row, col/64] · (d − H)/H`, H = (k−1)/2. The materialized
destination is the GGUF tensor's normal layout; the loader builds the tensor
at the model's true rank and hands the decoder the flattened 2-D frame.

## 3. v2 binary layout

Per quantized tensor (record + blob changes only; scales/perm/freq-table
formats unchanged):

```
directory record (v2) adds one byte after mode/perm:
    u8 lanes                         1..64; 1 == v1-compatible byte layout

digit stream when lanes > 1:
    [ u32 lane_off[L+1]              sub-stream offsets relative to the start
                                     of the sub-stream area, ascending,
                                     lane_off[L] == dig - 4*(L+1) ]
    [ L contiguous sub-streams; each is exactly one v1 stream = renorm bytes
      followed by its own 4-byte BE initial-state trailer ]
```

Rules:

1. **Row-aligned lanes.** Lane `i` owns stored rows
   `[i·rows_per_lane, (i+1)·rows_per_lane)` with
   `rows_per_lane = ceil(rows / lanes)`. Contiguous chunks (not symbol
   round-robin) keep the producer-side encoder vectorizable and let the
   decoder pack each lane's rows directly into the destination — the v2
   decode path then needs *no* cross-thread synchronization beyond range
   ownership. Last lane may be short.
2. **One frequency table, L states.** The table is computed from the full
   histogram exactly as today (normalize over all digits) — every lane encodes
   against that same table, so per-lane streams decode independently. This is
   the property that makes the change bit-exact rather than approximate.
3. **Lane count choice** (encoder): `L = max(1, min(16, digit_bytes >> 21, rows))`
   where `digit_bytes` = rows·cols evaluated pre-encode (the entropy-coded
   blob is smaller, so this is the conservative bound) — the ~525 MB expert
   blobs get 16 lanes, 31 MB attn tensors get 8–16, ~3 MB tensors stay
   single-lane (L=1 keeps the exact v1 layout).
4. **Overhead** = 4·(L+1) offset bytes + L×~5 state/flush bytes ≈ 150 B per
   tensor at L=16 — noise against 500 MB blobs.
5. **Versioning is a NEW KV, not a bumped record**: `ninths.directory2` with a
   leading `u16 version == 2`, then the v1 record layout plus the `u8 lanes`
   field. Reasons: (a) v1 parsers scan for `ninths.directory`/`k9.directory`
   by name and skip unknown KV keys, so a v2 writer cannot corrupt an old
   reader — old readers loudly reject the model ("no k9.directory") instead of
   misdecoding; (b) no dual-directory trap: v2 files carry ONLY directory2.
   Legacy aliasing rule continues to apply (accept `k9.directory2` if the
   rename lands inconsistently).

## 4. Implementation plan

Fork (`~/Work/llama.cpp`, branch k9):
- `src/k9.h`: `k9_tensor_meta` gains `uint8_t lanes`; header comment gains the
  v2 layout; `k9_parse_directory` parses both KV spellings and both versions.
- `src/k9.cpp`: new decode path — if `lanes > 1`, verify `lane_off`
  monotonicity, then spawn `min(lanes, hw/2)` workers; worker decodes its lane
  (decode loop verbatim from v1, per-lane state) and packs its row range with
  the session-32 row-packer. Single-lane path = today's code, untouched.
- `examples/k9probe/k9probe.cpp`: accept v2 records; keep the digits/scales/
  weights dump interface (dump per-lane or whole-canvas — whole-canvas, same
  signature).
- Rebuild + full gate suite (below) before any re-encode.

base9 (`experiments/`):
- `rans_fast.c/.py`: expose `encode_lanes(digits, f, k, lanes) -> (offsets,
  bytes)` and the matching `decode_lanes`; the C core is already resumable, so
  a lane encoder is just L short encode calls with per-lane final states.
- `k9.py`: `encode_tensor(..., lanes=L)` writing the v2 digit area;
  `decode_digits` accepts L.
- `glm_k9_build.py`: pass1 `--lanes N` (default: rule in §3.3) + v2 directory
  emission; pass2 unchanged (it copies the blob verbatim). `--mini` builds
  v2 too (gates run on it).

## 5. Validation gates (all must be green before the GLM re-encode)

1. Random-data round-trip: `encode_lanes`/`decode_lanes` at L ∈ {1, 2, 3, 16,
   64} bit-exact vs the single-lane reference on synthetic digit streams
   (skewed histograms incl. zero-count tails — the normalize fixup path).
2. Real-tensor cross-check: one k63 expert + one k99 embed encoded BOTH ways;
   materialized q8_0 canvases byte-identical to the v1 result.
3. Full-GGUF gates: the three session-32 tensors (attn_q, rank-3 expert,
   k99 token_embd) through the rebuilt probe — 0 mismatches, 0 ulp.
4. `--mini` GGUF loads through the fork end-to-end before pass1 rerun.

## 6. Re-Encoding and rollout

- GLM-4.5-Air: re-run pass1 (`--lanes`, ~15–20 min under cap) + pass2 →
  replace `glm45air_k63_e99.gguf`; re-run gates; reload server; measure load.
- Legacy artifacts (qwen 1.5B/7B/14B `.k9` GGUFs, the K9 KV runs) stay v1 and
  keep loading forever — the v1 path is frozen, not removed.
- KoboldCpp port parity: the vendored decode must gain the same v2 parser or
  it will refuse v2 files (loudly — acceptable, but note it in the port's
  readiness checklist before any publish).
- HF upload (independent work item): do not gate on v2 — upload the current
  v1 build; v2 lands as a re-upload/side-by-side if it proves out.

## 7. Risks / open questions

- Per-lane state divergence: rANS carries ~3.27 bits/state entropy; 16 lanes
  cost ≤ ~60 B/tensor total overhead, so rate is unaffected — no reason to
  expect a compression change, but gate 1 must confirm byte counts match
  exactly.
- MTP interplay: none (orthogonal), but if the BIOS carve is ever raised for
  MTP, the faster load makes iteration cheaper — sequencing note, not a
  dependency.
- The 1<<24 chunked decode loop inside each lane keeps peak staging memory at
  one lane-chunk + one row (≤ ~46 MiB/lane) — actually *lower* than today's
  full-tensor digits buffer; net RAM at load should DROP by ~700 MB.