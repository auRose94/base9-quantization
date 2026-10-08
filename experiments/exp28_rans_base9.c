/* exp28_rans_base9.c — grid-native rANS: base-9 renormalization (RQ11, docs/09).
 *
 * The byte coder (rans_fast.c) renormalizes in base 256 while its stream is
 * base-9 digits; this coder renormalizes IN BASE 9, so every renorm op is a
 * digit operation of the grid (x % 9, x / 9) and the symbol slot is the low
 * n base-9 digits of the state (x % M with M = 9^n) — codec arithmetic is
 * grid arithmetic.
 *
 * Mirror of rans.py / rans_fast.c semantics with (L, M, B):
 *   byte:  L = 2^24,  M = 2^k,        B = 2^8, state [2^24, 2^32)
 *   base9: L = 9^12,  M = 9^4 (6561), B = 9,   state [9^12, 9^13) — u64-safe.
 * M is the frequency-table size and the inverse-lookup size at once: 6561
 * entries (26 KB) keep the decode table in cache, where the byte coder's
 * M=2^12 table is 16 KB. Renorm-before-write: while x >= fs * (9*L)/M =
 * fs * 9^9: emit x % 9, x /= 9. Encode step: x = (x/fs)*M + (x%fs) + cs.
 * Invariant x in [L, 9L) after every step (same integer-division argument as
 * the byte coder: no-push case has x >= L = 9^12 -> x/fs >= 9^8 -> x' >=
 * M*9^8 = L; push case ends x >= fs*9^9/9 -> x/fs >= 9^8 likewise).
 * End state stored in 6 bytes (x < 9^13 needs 41.25 bits).
 *
 * Digits emit into a stack the caller supplies (worst case 8 per symbol); C
 * returns raw emitted digits in push order and the end state. The 169-digit/
 * 67-byte block packing (9^169 < 2^536, waste 0.285 bits/block) is a container
 * choice measured separately on the Python side; the state machine here is
 * the throughput-measured core, same basis as the byte coder's byte emitter.
 */
#include <stdint.h>
#include <stddef.h>

#define B9_M     6561ULL
#define B9_L     282429536481ULL        /* 9^12 */
#define B9_9LPM  387420489ULL           /* (9*L)/M = 9^13 / 9^4 = 9^9 */

size_t b9_rans_encode(const uint8_t *sym, size_t n,
                      const uint64_t *freq, const uint64_t *cumul,
                      uint8_t *digits_out, size_t dig_cap, uint64_t *end_state)
{
    uint64_t x = B9_L;
    size_t oi = 0;
    for (size_t i = n; i-- > 0; ) {
        const uint64_t fs = freq[sym[i]];
        const uint64_t cs = cumul[sym[i]];
        const uint64_t thresh = fs * B9_9LPM;
        while (x >= thresh) {
            if (oi >= dig_cap) return (size_t)-1;
            digits_out[oi++] = (uint8_t)(x % 9);
            x /= 9;
        }
        x = (x / fs) * B9_M + (x % fs) + cs;
    }
    *end_state = x;
    return oi;
}

void b9_rans_decode(const uint8_t *digits, size_t ndig, uint64_t end_state, size_t n,
                    const uint64_t *freq, const uint64_t *cumul, const uint32_t *inv,
                    uint8_t *out)
{
    long pos = (long)ndig - 1;
    uint64_t x = end_state;
    for (size_t i = 0; i < n; i++) {
        const uint64_t slot = x % B9_M;          /* x mod 9^4 = low base-9 digits */
        const uint32_t s = inv[slot];
        x = freq[s] * (x / B9_M) + slot - cumul[s];
        while (x < B9_L) {
            x = x * 9 + digits[pos--];
        }
        out[i] = (uint8_t)s;
    }
}