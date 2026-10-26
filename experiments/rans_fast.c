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
#include <string.h>

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

/* ---------------------------------------------------------------------------
 * K9 -> IQ4_NL materialization packing (mirrors the fork's src/k9.cpp branch
 * byte-for-byte): per 32-weight block, d = amax/113 (f16, round-half-even),
 * zero-blocks emit d=0 + code index 8 everywhere, codes = first strict
 * minimum of |w/d - value| over kvalues_iq4nl[16], qs[j] = L[j] | L[j+16]<<4.
 * scales: rows*(cols/64) fp32 (ent8-decoded); digits: rows*cols u8. */
static int8_t k9_iq4nl_tab[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static uint16_t k9_f32_to_f16_rne(float f) {
    /* round-to-nearest-even fp32 -> fp16, matching ggml_fp32_to_fp16 */
    union { float f; uint32_t u; } v; v.f = f;
    uint32_t x = v.u;
    uint32_t sign = x & 0x80000000u;
    uint32_t mant = x & 0x007fffffu;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    if (((x >> 23) & 0xff) == 0xff) {  /* inf/nan */
        return (uint16_t)(sign >> 16 | 0x7c00u | (mant ? 0x200u | (mant >> 13) : 0));
    }
    if (exp >= 0x1f) {  /* overflow -> inf */
        return (uint16_t)((sign >> 16) | 0x7c00u);
    }
    if (exp <= 0) {     /* subnormal or zero */
        if (exp < -10) return (uint16_t)(sign >> 16);
        mant |= 0x00800000u;
        int shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        if ((rem > (1u << (shift - 1))) ||
            ((rem == (1u << (shift - 1))) && (half & 1))) {
            half++;
        }
        return (uint16_t)((sign >> 16) | half);
    }
    uint32_t half = (uint32_t)exp << 10 | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) {
        half++;
    }
    return (uint16_t)((sign >> 16) | half);
}

size_t k9_pack_iq4nl(const uint8_t *digits, const float *scales,
                     int64_t rows, int64_t cols, int k, uint8_t *out) {
    const int kH = (k - 1) / 2;
    const int64_t ng = cols / 64;             /* 64-column scale groups */
    const int64_t bpr = (int64_t)cols / 32;   /* iq4_nl blocks per row */
    float w[32];
    uint8_t L[32];
    uint64_t oi = 0;
    for (int64_t i1 = 0; i1 < rows; i1++) {   /* note: rows-1..0 not needed, output order matches stored rows */
        const uint8_t * rd = digits + i1 * (uint64_t)cols;
        const float * ms = scales + i1 * ng;
        for (int64_t jg = 0; jg < ng; jg++) {
            const float m = ms[jg];
            for (int sb = 0; sb < 2; sb++) {
                float amax = 0.f;
                for (int t = 0; t < 32; t++) {
                    float val = m * ((float) rd[jg * 64 + sb * 32 + t] - (float) kH) / (float) kH;
                    w[t] = val;
                    float av = val < 0.f ? -val : val;
                    if (av > amax) amax = av;
                }
                uint8_t * blk = out + oi; oi += 18;
                float d = 0.f;
                if (amax >= 1e-8f) d = amax / 113.0f;
                const uint16_t d16 = k9_f32_to_f16_rne(d);
                blk[0] = (uint8_t)(d16 & 0xff);
                blk[1] = (uint8_t)(d16 >> 8);
                if (d == 0.f) {
                    memset(blk + 2, 0x88, 16);
                    continue;
                }
                const float id = 1.0f / d;
                for (int t = 0; t < 32; t++) {
                    float target = id * w[t];
                    uint8_t best = 0;
                    float bestdiff = 1e30f;
                    for (int e = 0; e < 16; e++) {
                        float dd = target - (float) k9_iq4nl_tab[e];
                        if (dd < 0.f) dd = -dd;
                        if (dd < bestdiff) { bestdiff = dd; best = (uint8_t) e; }
                    }
                    L[t] = best;
                }
                for (int t = 0; t < 16; t++) {
                    blk[2 + t] = L[t] | (L[16 + t] << 4);
                }
            }
        }
    }
    return (size_t) oi;
}
