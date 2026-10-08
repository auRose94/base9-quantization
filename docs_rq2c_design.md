# RQ2c design note — GPU-parallel rANS decode (the last CPU step in the synthesis loop)

Status: design only (eq12 was running on the GPU at the time of writing).
The eq7 loop currently does rANS decode on CPU (the validated base9 path) and
ships decoded digits+small tables to the GPU. The endgame removes that step:
the artifact's rANS byte stream decodes **on device**.

## Why it matters

- The synthesis loop becomes: K9Q1 bytes → GPU → buffers. Zero CPU decode;
  load time bounded by PCIe + kernel time; scales to 7B+ models where CPU
  digit decoding is minutes.
- rANS decode is inherently sequential per state, so GPU-parallelism comes
  from **interleaved streams** (the standard rANS-in-GPU trick, and the same
  family as the interleaved-stream variants used in LZ4/Zstd GPU decoders):
  split each tensor's digit stream into I independent sub-streams (during
  encoding), decode I states concurrently, one CUDA block per stream-chunk.

## Design

1. **Encoding side (CPU, one-time):** extend the k9 writer with
   `interleave=I` — per tensor, slice the flat digit stream into I runs,
   rANS-encode each run independently (the validated rans.py/rans_fast.c is
   unchanged; the container gains a stream-count field). Byte overhead ≈
   I × (renorm flush tails) + per-stream freq table shared (0 extra).
   Encode a k9-selftest that round-trips at I = 1/32/128.
2. **Kernel side (Triton first, then raw CUDA):** per decode block:
   reload the rANS state from the stream tail (32-bit state × I), then a
   fixed-point decode loop: while symbols remain: symbol = f(state & mask),
   renorm (the M=4069/4096 law from rand.py — reuse the validated tables),
   state = state·M + extra... The k9 coder's exact renorm/mask semantics
   (memory: "renorm f·2^32/M, mask fix, byte-identical at M=256") are the
   spec; the kernel is its transcription — assert bit-identity vs
   k9.decode_digits on real artifacts (the P19-style gate).
3. **Fusion path:** with digits on device, the eq7 synth kernel consumes them
   directly; the *fused* endgame (synthesis inside the matmul, never
   materializing) builds on the same digit-in-registers layout.

## Acceptance gates (pre-registered when implemented, P30+)

- P30: interleaved encoding is byte-competitive (≤ +2% vs single-stream) at
  I = 128 on the 33M body digits.
- P31: Triton/CUDA decode == k9.decode_digits bitwise on real artifacts.
- P32: end-to-end artifact→story with zero CPU digit decode; load-time
  accounting vs the eq7 baseline.