/* rans_fast.c — byte-renorm rANS (Duda), exactly mirroring experiments/rans.py.
 *
 * Same semantics as the reference Python coder: 32-bit state domain
 * [2^24, 2^32), renorm-before-write while x >= f * 2^(32-k), symbols processed
 * in reverse for encode, 4-byte big-endian final state appended. Generic
 * alphabet via M = 2^k. Built to be byte-identical to rans.py's output.
 *
 * Compiled by rans_fast.py (gcc -O3 -shared -fPIC) and called through ctypes.
 */
#include <stdint.h>
#include <stddef.h>

/* Encode `n` symbols (natural order) into out; returns bytes written
 * (including the 4-byte end state), or 0 if out_cap is too small.
 * freq/cumul are the normalized frequency and cumulative tables (size n_syms),
 * cumul[0] == 0. Worst case one symbol emits 2 bytes. */
size_t k9_rans_encode(const uint8_t *sym, size_t n,
                      const uint64_t *freq, const uint64_t *cumul,
                      int k, uint8_t *out, size_t out_cap)
{
    const int kshift = 32 - k;
    uint64_t x = (uint64_t)1 << 24;
    size_t oi = 0;

    for (size_t i = n; i-- > 0; ) {
        const uint32_t s = sym[i];
        const uint64_t fs = freq[s];
        const uint64_t cs = cumul[s];
        const uint64_t thresh = fs << kshift;
        while (x >= thresh) {
            if (oi >= out_cap) return 0;
            out[oi++] = (uint8_t)(x & 0xFF);
            x >>= 8;
        }
        x = ((x / fs) << k) + (x % fs) + cs;
    }
    if (oi + 4 > out_cap) return 0;
    out[oi++] = (uint8_t)((x >> 24) & 0xFF);
    out[oi++] = (uint8_t)((x >> 16) & 0xFF);
    out[oi++] = (uint8_t)((x >> 8) & 0xFF);
    out[oi++] = (uint8_t)(x & 0xFF);
    return oi;
}

/* Decode `n` symbols from `stream` into out (uint8). freq/cumul as above,
 * inv is the M-entry symbol lookup (size 1<<k), cumul[0] == 0. */
void k9_rans_decode(const uint8_t *stream, size_t slen, size_t n,
                    const uint64_t *freq, const uint64_t *cumul,
                    const uint32_t *inv, int k, uint8_t *out)
{
    const uint64_t mask = ((uint64_t)1 << k) - 1;
    long pos = (long)slen - 5;
    uint64_t x = ((uint64_t)stream[slen - 4] << 24)
               | ((uint64_t)stream[slen - 3] << 16)
               | ((uint64_t)stream[slen - 2] << 8)
               | (uint64_t)stream[slen - 1];

    for (size_t i = 0; i < n; i++) {
        const uint32_t s = inv[x & mask];
        x = freq[s] * (x >> k) + (x & mask) - cumul[s];
        while (x < ((uint64_t)1 << 24)) {
            x = (x << 8) | stream[pos--];
        }
        out[i] = (uint8_t)s;
    }
}
