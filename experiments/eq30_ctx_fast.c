/* eq30_ctx_fast.c — the P47 fix-path: a cache-resident, incrementally-built
 * context-conditioned base-9 rANS (docs/09 RQ13 c-d / P47).
 *
 * eq27's diagnosis: the loop was fine, the WORKING SET and the companion's
 * build were not. Two changes, both aimed exactly there:
 *
 *   1. no per-context inverse table. The slot->symbol lookup is a branchless
 *      binary search over the cumulative frequencies instead: a context's
 *      structure drops from 26 KB (M = 9^4 u32 entries) to ~1 KB (alpha-sized
 *      freq + cum), so a chunk's touched-context set stays L2-resident where
 *      eq27's 62 MB table set could not.
 *   2. the companion lives here. Counts (pair / ctx_tot / glob / order-1
 *      marginals) update in O(1) per symbol; a context's table is built
 *      LAZILY on first use in a chunk and invalidated wholesale at the chunk
 *      boundary by a generation counter, so the per-chunk refresh that cost
 *      134 us/symbol in the numpy prototype costs a few hundred ns here.
 *
 * The estimator is the eq26/eq27 one, mirrored exactly (order-2 ML
 * interpolated with the order-1 conditional given by the two-before symbol,
 * down to a KT order-0 base; largest-remainder quantization to sum M = 9^4),
 * so the byte stream is cross-checkable against the Python reference.
 *
 * Segments are independent chunks (each with its own end state), exactly as
 * eq27: the tables are prefix-derived, so decoding must run in stream order.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define M_M      6561u
#define MAXA     256u
#define CTX_CACHE 8192u

typedef struct {
    uint32_t gen;
    uint32_t ctx;
    uint32_t tot;
    uint32_t freq[MAXA];
    uint32_t cum[MAXA];
} Tab;

typedef struct {
    uint32_t alpha, ctxmul;
    double l1, l2;
    uint32_t gen;
    uint32_t *pair;        /* [alpha*alpha][alpha] */
    uint32_t *ctx_tot;     /* [alpha*alpha] */
    uint32_t *glob;        /* [alpha] */
    uint32_t *m1;          /* [alpha*alpha] order-1 marginals (over the first symbol) */
    uint32_t *m1_tot;      /* [alpha] */
    uint64_t ntot;
    /* Deterministic table storage: tables are a function of (ctx, generation)
       ONLY -- rebuilt eagerly for every seen context at each generation bump,
       so an eviction (or a different traversal order) can never make the two
       sides disagree. Slots are assigned on first sight of a context. */
    int32_t *ctx_slot;     /* [alpha*alpha], -1 = unseen */
    Tab *tab_seen;         /* the seen contexts' tables */
    uint32_t n_seen;
    Tab tab0;              /* the KT fallback for unseen contexts */
} Comp;

void *comp_new(uint32_t alpha, double l1, double l2)
{
    Comp *c = (Comp *)calloc(1, sizeof(Comp));
    c->alpha = alpha;
    c->ctxmul = alpha;
    c->l1 = l1;
    c->l2 = l2;
    c->pair = (uint32_t *)calloc((size_t)alpha * alpha * alpha, sizeof(uint32_t));
    c->ctx_tot = (uint32_t *)calloc((size_t)alpha * alpha, sizeof(uint32_t));
    c->glob = (uint32_t *)calloc(alpha, sizeof(uint32_t));
    c->m1 = (uint32_t *)calloc((size_t)alpha * alpha, sizeof(uint32_t));
    c->m1_tot = (uint32_t *)calloc(alpha, sizeof(uint32_t));
    c->ctx_slot = (int32_t *)malloc((size_t)alpha * alpha * sizeof(int32_t));
    for (size_t i = 0; i < (size_t)alpha * alpha; i++) c->ctx_slot[i] = -1;
    c->tab_seen = (Tab *)calloc((size_t)alpha * alpha, sizeof(Tab));
    c->n_seen = 0;
    return c;
}

void comp_free(void *p)
{
    Comp *c = (Comp *)p;
    free(c->pair); free(c->ctx_tot); free(c->glob);
    free(c->m1); free(c->m1_tot); free(c->ctx_slot); free(c->tab_seen);
    free(c);
}

static void build_tab(Comp *c, Tab *t, uint32_t ctx);

void comp_new_chunk(void *p)
{
    Comp *c = (Comp *)p;
    c->gen++;
    build_tab(c, &c->tab0, 0xFFFFFFFFu);          /* the unseen-context fallback */
    for (uint32_t ctx = 0; ctx < c->alpha * c->alpha; ctx++) {
        if (c->ctx_tot[ctx] == 0) continue;
        int32_t sl = c->ctx_slot[ctx];
        if (sl < 0) {
            sl = (int32_t)c->n_seen++;
            c->ctx_slot[ctx] = sl;
        }
        build_tab(c, &c->tab_seen[sl], ctx);
    }
}

/* -- the estimator: largest-remainder quantization, house tie-breaking ----
 * sorted once per build (O(alpha log alpha)); ties resolve to the lower index,
 * matching the Python reference's stable argsort. */
static void quantize(const double *pp, uint32_t *f, uint32_t alpha)
{
    double frac[MAXA];
    uint32_t idx[MAXA];
    int32_t sum = 0;
    for (uint32_t s = 0; s < alpha; s++) {
        double fl = floor(pp[s] * (double)M_M);
        int32_t v = (int32_t)fl;
        if (v < 1) v = 1;
        f[s] = (uint32_t)v;
        frac[s] = pp[s] * (double)M_M - fl;
        idx[s] = s;
        sum += v;
    }
    int32_t rem = (int32_t)M_M - sum;
    if (rem != 0) {
        for (uint32_t i = 1; i < alpha; i++) {        /* stable insertion sort */
            uint32_t j = i, v = idx[i];
            while (j > 0 && (rem > 0 ? frac[idx[j - 1]] < frac[v]
                                     : f[idx[j - 1]] < f[v])) {
                idx[j] = idx[j - 1]; j--;
            }
            idx[j] = v;
        }
        while (rem > 0) {
            f[idx[rem % (int32_t)alpha]]++;
            rem--;
        }
        while (rem < 0) {
            uint32_t k = 0;
            while (k < alpha && f[idx[k]] <= 1) k++;
            if (k >= alpha) break;
            f[idx[k]]--; rem++;
        }
    }
}

static void build_tab(Comp *c, Tab *t, uint32_t ctx)
{
    const uint32_t a = c->alpha;
    double kt[MAXA], p1[MAXA], p[MAXA];
    double nn = (double)c->ntot;
    for (uint32_t s = 0; s < a; s++)
        kt[s] = ((double)c->glob[s] + 0.5) / (nn + 0.5 * (double)a);
    if (ctx == 0xFFFFFFFFu) {                /* the fallback: KT only */
        quantize(kt, t->freq, a);
        uint32_t acc0 = 0;
        for (uint32_t s = 0; s < a; s++) { t->cum[s] = acc0; acc0 += t->freq[s]; }
        t->ctx = ctx; t->gen = c->gen;
        return;
    }
    const uint32_t c1 = ctx / a;             /* the two-before symbol (eq26's fold) */
    uint32_t t1 = c->m1_tot[c1];
    for (uint32_t s = 0; s < a; s++)
        p1[s] = (t1 > 0) ? ((double)c->m1[(size_t)c1 * a + s] / (double)t1) : kt[s];
    uint32_t tot = c->ctx_tot[ctx];
    if (tot == 0) {                          /* unseen: the KT fallback */
        for (uint32_t s = 0; s < a; s++) p[s] = kt[s];
    } else {
        const uint32_t *cnt = c->pair + (size_t)ctx * a;
        for (uint32_t s = 0; s < a; s++)
            p[s] = c->l2 * ((double)cnt[s] / (double)tot)
                 + (1.0 - c->l2) * (c->l1 * p1[s] + (1.0 - c->l1) * kt[s]);
    }
    quantize(p, t->freq, a);
    uint32_t acc = 0;
    for (uint32_t s = 0; s < a; s++) { t->cum[s] = acc; acc += t->freq[s]; }
    t->tot = tot;
    t->ctx = ctx;
    t->gen = c->gen;
}

static inline Tab *get_tab(Comp *c, uint32_t ctx)
{
    int32_t sl = c->ctx_slot[ctx];
    return (sl < 0) ? &c->tab0 : &c->tab_seen[sl];
}

/* branchless-ish binary search: the symbol whose [cum, cum+freq) holds slot */
static inline uint32_t sym_of(const Tab *t, uint32_t alpha, uint32_t slot)
{
    uint32_t lo = 0, hi = alpha;
    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) >> 1;
        lo = (t->cum[mid] <= slot) ? mid : lo;
        hi = (t->cum[mid] <= slot) ? hi : mid;
    }
    return lo;
}

void comp_update(void *p, const uint8_t *syms, size_t n)
{
    Comp *c = (Comp *)p;
    const uint32_t a = c->alpha;
    for (size_t i = 0; i < n; i++) {
        uint32_t s = syms[i];
        if (i >= 1) {
            uint32_t p1 = syms[i - 1];
            uint32_t p2 = (i >= 2) ? syms[i - 2] : 0u;
            uint32_t ctx = p2 * a + p1;
            c->pair[(size_t)ctx * a + s]++;
            c->ctx_tot[ctx]++;
            c->m1[(size_t)p2 * a + p1]++;      /* eq26's row = the two-before symbol */
            c->m1_tot[p2]++;
        }
        c->glob[s]++;
        c->ntot++;
    }
}

size_t comp_encode_seg(void *p, const uint8_t *syms, size_t n,
                       uint8_t *out, size_t cap, uint64_t *x_io)
{
    Comp *c = (Comp *)p;
    uint64_t x = *x_io;
    size_t oi = 0;
    for (size_t i = n; i-- > 0; ) {
        uint32_t p1 = (i >= 1) ? syms[i - 1] : 0u;
        uint32_t p2 = (i >= 2) ? syms[i - 2] : 0u;
        Tab *t = get_tab(c, p2 * c->alpha + p1);
        uint64_t fs = t->freq[syms[i]];
        uint64_t cs = t->cum[syms[i]];
        uint64_t th = fs * 387420489ULL;        /* fs * 9^9 */
        while (x >= th) {
            if (oi >= cap) return (size_t)-1;
            out[oi++] = (uint8_t)(x % 9);
            x /= 9;
        }
        x = (x / fs) * M_M + (x % fs) + cs;
    }
    *x_io = x;
    return oi;
}

void comp_decode_seg(void *p, const uint8_t *digits, size_t ndig,
                     uint64_t *x_io, size_t n, uint8_t *out)
{
    Comp *c = (Comp *)p;
    long pos = (long)ndig - 1;
    uint64_t x = *x_io;
    const uint32_t a = c->alpha;
    uint32_t p1 = 0, p2 = 0;
    for (size_t i = 0; i < n; i++) {
        Tab *t = get_tab(c, p2 * a + p1);
        uint32_t slot = (uint32_t)(x % M_M);
        uint32_t s = sym_of(t, a, slot);
        x = (uint64_t)t->freq[s] * (x / M_M) + slot - t->cum[s];
        while (x < 282429536481ULL)             /* 9^12 */
            x = x * 9 + digits[pos--];
        out[i] = (uint8_t)s;
        p2 = p1;
        p1 = s;
    }
    *x_io = x;
}
