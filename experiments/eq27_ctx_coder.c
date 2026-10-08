/* eq27_ctx_coder.c — context-conditioned base-9 rANS (RQ13 P47, docs/09).
 *
 * The exp28_rans_base9 state algebra verbatim (M = 9^4, L = 9^12, base-9
 * renorm, symbol slot = x % M). The only addition is a per-symbol table
 * SELECTION: the coding distribution is looked up by the context — the two
 * previously decoded symbols — so a companion that refreshes the
 * context-conditioned tables every >= 1k symbols slots into this loop
 * without touching the state machine (the spec's load-bearing constraint:
 * companions predict tables in chunks, the state machine consumes them at
 * symbol rate; a model forward per symbol would strangle it).
 *
 * ctx_id maps the folded context (p1*ctxmul + p2) to a compact table id;
 * CTX_NONE falls back to table 0 (the static distribution). Segments are
 * decoded with a caller-refreshed table set; pos/state/prev thread across
 * calls so the caller controls the refresh cadence exactly.
 */
#include <stdint.h>
#include <stddef.h>

#define B9_M     6561ULL
#define B9_L     282429536481ULL        /* 9^12 */
#define B9_9LPM  387420489ULL           /* 9^9 */
#define CTX_NONE 0xFFFFFFFFu

void b9x_decode(const uint8_t *digits, size_t ndig,
                long *pos_io, uint64_t *x_io,
                size_t n,
                const uint64_t *freq, const uint64_t *cumul,
                const uint32_t *inv, const uint32_t *ctx_id,
                size_t alpha, size_t ctxmul,
                uint32_t *p1_io, uint32_t *p2_io, uint8_t *out)
{
    long pos = *pos_io;
    uint64_t x = *x_io;
    uint32_t p1 = *p1_io, p2 = *p2_io;
    for (size_t i = 0; i < n; i++) {
        const uint32_t ctx = p2 * ctxmul + p1;   /* (two-before)*A + (one-before): the companion's key order */
        uint32_t cid = ctx_id[ctx];
        if (cid == CTX_NONE) cid = 0;
        const uint64_t *fr = freq + (size_t)cid * alpha;
        const uint64_t *cu = cumul + (size_t)cid * alpha;
        const uint32_t *iv = inv + (size_t)cid * B9_M;
        const uint64_t slot = x % B9_M;        /* low base-9 digits */
        const uint32_t s = iv[slot];
        x = fr[s] * (x / B9_M) + slot - cu[s];
        while (x < B9_L) x = x * 9 + digits[pos--];
        out[i] = (uint8_t)s;
        p2 = p1;
        p1 = s;
    }
    *pos_io = pos;
    *x_io = x;
    *p1_io = p1;
    *p2_io = p2;
}

size_t b9x_encode(const uint8_t *sym, size_t n,
                  const uint64_t *freq, const uint64_t *cumul,
                  const uint32_t *ctx_id, size_t alpha, size_t ctxmul,
                  uint32_t prev1_in, uint32_t prev2_in,
                  uint8_t *digits_out, size_t dig_cap, uint64_t *x_io,
                  uint32_t *prev1_out, uint32_t *prev2_out)
{
    /* contexts depend on the PRECEDING symbols (forward order), the rANS
       pushes run in reverse: resolve each symbol's context up front. */
    uint64_t x = *x_io;
    size_t oi = 0;
    for (size_t i = n; i-- > 0; ) {
        const uint32_t p1 = (i >= 1) ? sym[i - 1] : prev1_in;
        const uint32_t p2 = (i >= 2) ? sym[i - 2] : prev2_in;
        const uint32_t ctx = p2 * ctxmul + p1;   /* (two-before)*A + (one-before): the companion's key order */
        uint32_t cid = ctx_id[ctx];
        if (cid == CTX_NONE) cid = 0;
        const uint64_t *fr = freq + (size_t)cid * alpha;
        const uint64_t *cu = cumul + (size_t)cid * alpha;
        const uint64_t fs = fr[sym[i]];
        const uint64_t cs = cu[sym[i]];
        const uint64_t thresh = fs * B9_9LPM;
        while (x >= thresh) {
            if (oi >= dig_cap) return (size_t)-1;
            digits_out[oi++] = (uint8_t)(x % 9);
            x /= 9;
        }
        x = (x / fs) * B9_M + (x % fs) + cs;
    }
    *x_io = x;
    *prev1_out = (n >= 1) ? sym[n - 1] : prev1_in;
    *prev2_out = (n >= 2) ? sym[n - 2] : prev2_in;
    return oi;
}
