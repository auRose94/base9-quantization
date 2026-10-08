/* eq32_ctx_kmeans.c — the P47 frontier with a real clusterer (docs/09 RQ13).
 *
 * eq31 measured the frontier with a cheap top-1-argmax clusterer and settled
 * two things: the branchless inverse-table lookup is required (it nearly
 * tripled the loop), and the cheap key leaves half the rate win behind. This
 * engine adds the distribution-aware alternative and one cheaper key:
 *
 *   CLUSTER_MODE 0 = top-1 argmax rank (eq31's key, kept for the A/B)
 *   CLUSTER_MODE 1 = k-means on the context distributions (L2 on the raw ML
 *                    probabilities, mass-weighted centroids, deterministic
 *                    init from the most massive contexts; recomputed every
 *                    32 generations and assigned cheaply in between)
 *   CLUSTER_MODE 2 = p2-bucket: the cluster is the two-before symbol's
 *                    mass-rank bucket -- a single L1-sized array lookup per
 *                    symbol, the cheapest possible context key
 *
 * Original header follows.
 *
 * eq30 removed the inverse tables and the build cost but left the per-symbol
 * lookup landing in a ~4.7 MB spread of per-context tables: every symbol's
 * freq/cum/search touched L3, which is why the binary search did not beat the
 * old inverse tables (both were L3-bound) and why the loop plateaued at
 * 41-69 M sym/s while the plain single-table base-9 machine does 161-188.
 *
 * This engine clusters the contexts into K tables: a ctx -> cluster array
 * (alpha^2 u32, L2) selects among K tables (K x ~1 KB, L1-resident), so the
 * whole coding distribution fits the L1 where eq30's did not. The clustering
 * is deterministic and derived from the same prefix counts on both sides
 * (causality), recomputed at each refresh boundary: each seen context is
 * assigned by its most probable continuation symbol's rank among the K-1
 * globally most probable symbols (cheap, O(alpha) per context, and the
 * assignment is a function of the counts alone, so encode and decode agree
 * by construction). Clusters pool their members' pair counts and run the same
 * estimator chain (order-2 ML over the pooled counts, interpolated with the
 * two-before conditional and a KT base; largest-remainder quantization).
 *
 * K is the frontier's single parameter: K = 1 is the pooled/order-1 end,
 * large K approaches eq30's full per-context tables. Rate and throughput are
 * both measured.
 *
 * Original eq30 header for the invariants that still hold:
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
#ifndef EQ31_USE_INV
#define EQ31_USE_INV 1
#endif
#ifndef CLUSTER_MODE
#define CLUSTER_MODE 1
#endif
#define MAXA     256u
#define CTX_CACHE 8192u

typedef struct {
    uint32_t gen;
    uint32_t ctx;
    uint32_t tot;
    uint32_t freq[MAXA];
    uint32_t cum[MAXA];
    uint32_t inv[M_M];          /* the branchless slot->symbol lookup */
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
    /* Clustered deterministic table storage: ctx -> cluster (a function of
       the counts at the generation bump only, so the two sides agree by
       construction), and K pooled tables rebuilt eagerly at each bump. */
    uint32_t nclust;       /* K */
    uint32_t *ctx_clust;   /* [alpha*alpha] -> cluster id (K-1 = the pool) */
    Tab *clusters;         /* [K] */
    uint32_t *cl_pair;     /* [K][alpha] pooled counts */
    uint32_t *cl_tot;      /* [K] */
    uint32_t *rank_of;     /* [alpha] symbol -> rank among the top K-1, else K-1 */
} Comp;

void *comp_new(uint32_t alpha, double l1, double l2, uint32_t nclust)
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
    c->nclust = (nclust < 1) ? 1 : nclust;
    c->ctx_clust = (uint32_t *)malloc((size_t)alpha * alpha * sizeof(uint32_t));
    c->clusters = (Tab *)calloc(c->nclust, sizeof(Tab));
    c->cl_pair = (uint32_t *)calloc((size_t)c->nclust * alpha, sizeof(uint32_t));
    c->cl_tot = (uint32_t *)calloc(c->nclust, sizeof(uint32_t));
    c->rank_of = (uint32_t *)calloc(alpha, sizeof(uint32_t));
    for (size_t i = 0; i < (size_t)alpha * alpha; i++)
        c->ctx_clust[i] = c->nclust - 1;      /* the pool until a pass assigns */
    return c;
}

void comp_free(void *p)
{
    Comp *c = (Comp *)p;
    free(c->pair); free(c->ctx_tot); free(c->glob);
    free(c->m1); free(c->m1_tot); free(c->ctx_clust); free(c->clusters);
    free(c->cl_pair); free(c->cl_tot); free(c->rank_of);
    free(c);
}



static void quantize(const double *pp, uint32_t *f, uint32_t alpha);

/* Deterministic, mass-weighted k-means over the seen contexts' ML
   distributions: the L2 distance on the raw probabilities, the centroids =
   the weighted mean; the init = the K most massive contexts, in mass order. */
static void kmeans_assign(Comp *c)
{
    const uint32_t a = c->alpha, K = c->nclust;
    static uint32_t seen[65536];
    uint32_t ns = 0;
    for (uint32_t ctx = 0; ctx < a * a; ctx++)
        if (c->ctx_tot[ctx] > 0) seen[ns++] = ctx;
    if (ns == 0) return;
    if (K < 2) { for (uint32_t i = 0; i < ns; i++) c->ctx_clust[seen[i]] = 0; return; }
    /* init: the K most massive contexts (a partial selection, deterministic) */
    uint32_t cent[256];
    uint32_t taken[256];
    for (uint32_t k = 0; k < K; k++) {
        uint32_t best = 0xFFFFFFFFu, bv = 0;
        for (uint32_t i = 0; i < ns; i++) {
            uint32_t ctx = seen[i];
            uint32_t ok = 1;
            for (uint32_t j = 0; j < k; j++) if (taken[j] == ctx) { ok = 0; break; }
            if (ok && c->ctx_tot[ctx] > bv) { bv = c->ctx_tot[ctx]; best = ctx; }
        }
        if (best == 0xFFFFFFFFu) { cent[k] = seen[0]; } else { cent[k] = best; }
        taken[k] = cent[k];
    }
    static double cen[256][MAXA];
    static double cen_new[256][MAXA];
    static double mass[256];
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t s = 0; s < a; s++)
            cen[k][s] = (double)c->pair[(size_t)cent[k] * a + s] /
                        (double)c->ctx_tot[cent[k]];
    for (uint32_t it = 0; it < 8; it++) {
        for (uint32_t k = 0; k < K; k++) {
            mass[k] = 0.0;
            for (uint32_t s = 0; s < a; s++) cen_new[k][s] = 0.0;
        }
        for (uint32_t i = 0; i < ns; i++) {
            uint32_t ctx = seen[i];
            const uint32_t *cnt = c->pair + (size_t)ctx * a;
            double tot = (double)c->ctx_tot[ctx];
            double bestd = 1e300; uint32_t bk = 0;
            for (uint32_t k = 0; k < K; k++) {
                double d = 0.0;
                for (uint32_t s = 0; s < a; s++) {
                    double p = (double)cnt[s] / tot - cen[k][s];
                    d += p * p;
                }
                if (d < bestd) { bestd = d; bk = k; }
            }
            c->ctx_clust[ctx] = bk;
            mass[bk] += tot;
            for (uint32_t s = 0; s < a; s++)
                cen_new[bk][s] += (double)cnt[s];
        }
        for (uint32_t k = 0; k < K; k++) {
            if (mass[k] <= 0.0) continue;
            for (uint32_t s = 0; s < a; s++) cen[k][s] = cen_new[k][s] / mass[k];
        }
    }
}

/* Build one table from a pooled count row (the same estimator chain). */
static void build_pool_tab(Comp *c, Tab *t, const uint32_t *cnt, uint32_t tot,
                           uint32_t c1, uint32_t c1tot)
{
    const uint32_t a = c->alpha;
    double kt[MAXA], p1[MAXA], p[MAXA];
    double nn = (double)c->ntot;
    for (uint32_t s = 0; s < a; s++)
        kt[s] = ((double)c->glob[s] + 0.5) / (nn + 0.5 * (double)a);
    for (uint32_t s = 0; s < a; s++)
        p1[s] = (c1tot > 0) ? ((double)c->m1[(size_t)c1 * a + s] /
                               (double)c1tot) : kt[s];
    if (tot == 0) {
        for (uint32_t s = 0; s < a; s++) p[s] = kt[s];
    } else {
        for (uint32_t s = 0; s < a; s++)
            p[s] = c->l2 * ((double)cnt[s] / (double)tot)
                 + (1.0 - c->l2) * (c->l1 * p1[s] + (1.0 - c->l1) * kt[s]);
    }
    quantize(p, t->freq, a);
    uint32_t acc = 0;
    for (uint32_t s = 0; s < a; s++) { t->cum[s] = acc; acc += t->freq[s]; }
    for (uint32_t s = 0; s < a; s++)
        for (uint32_t k = t->cum[s]; k < t->cum[s] + t->freq[s]; k++)
            t->inv[k] = s;
    t->tot = tot;
    t->gen = c->gen;
}

void comp_new_chunk(void *p)
{
    Comp *c = (Comp *)p;
    const uint32_t a = c->alpha;
    c->gen++;
    /* 1. the K-1 globally most probable symbols, by mass (rank_of: else pool) */
    for (uint32_t s = 0; s < a; s++) c->rank_of[s] = (c->nclust > 1)
        ? (c->nclust - 1) : 0u;
    if (c->nclust > 1) {
        for (uint32_t r = 0; r + 1 < c->nclust; r++) {
            uint32_t best = 0xFFFFFFFFu, bv = 0;
            for (uint32_t s = 0; s < a; s++) {
                if (c->rank_of[s] != c->nclust - 1) continue;
                if (c->glob[s] > bv) { bv = c->glob[s]; best = s; }
            }
            if (best == 0xFFFFFFFFu) break;
            c->rank_of[best] = r;
        }
    }
#if CLUSTER_MODE == 2
    /* 2a. p2-bucket: the cluster is the two-before symbol's mass-rank bucket */
    for (uint32_t ctx = 0; ctx < a * a; ctx++)
        c->ctx_clust[ctx] = c->rank_of[ctx / a];
#elif CLUSTER_MODE == 1
    /* 2b. k-means on the context distributions (recomputed every 32 gens) */
    if ((c->gen & 31) == 1 || c->nclust < 2) {
        kmeans_assign(c);
    }
#else
    /* 2c. assign every seen context by its argmax continuation symbol */
    for (uint32_t ctx = 0; ctx < a * a; ctx++) {
        if (c->ctx_tot[ctx] == 0) { c->ctx_clust[ctx] = c->nclust - 1; continue; }
        const uint32_t *cnt = c->pair + (size_t)ctx * a;
        uint32_t best = 0, bv = 0;
        for (uint32_t s = 0; s < a; s++)
            if (cnt[s] > bv) { bv = cnt[s]; best = s; }
        c->ctx_clust[ctx] = c->rank_of[best];
    }
#endif
    /* 3. pool each cluster's counts and build its table */
    memset(c->cl_pair, 0, (size_t)c->nclust * a * sizeof(uint32_t));
    memset(c->cl_tot, 0, (size_t)c->nclust * sizeof(uint32_t));
    for (uint32_t ctx = 0; ctx < a * a; ctx++) {
        uint32_t tot = c->ctx_tot[ctx];
        if (tot == 0) continue;
        uint32_t cid = c->ctx_clust[ctx];
        const uint32_t *cnt = c->pair + (size_t)ctx * a;
        uint32_t *dst = c->cl_pair + (size_t)cid * a;
        for (uint32_t s = 0; s < a; s++) dst[s] += cnt[s];
        c->cl_tot[cid] += tot;
    }
    for (uint32_t cid = 0; cid < c->nclust; cid++) {
        const uint32_t *cnt = c->cl_pair + (size_t)cid * a;
        /* the two-before conditional: the cluster's own dominant row */
        uint32_t c1 = 0, bv = 0;
        for (uint32_t s = 0; s < a; s++)
            if (cnt[s] > bv) { bv = cnt[s]; c1 = s; }
        build_pool_tab(c, &c->clusters[cid], cnt, c->cl_tot[cid], c1,
                       c->m1_tot[c1]);
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
    return &c->clusters[c->ctx_clust[ctx]];
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
#if EQ31_USE_INV
        uint32_t s = t->inv[slot];
#else
        uint32_t s = sym_of(t, a, slot);
#endif
        x = (uint64_t)t->freq[s] * (x / M_M) + slot - t->cum[s];
        while (x < 282429536481ULL)             /* 9^12 */
            x = x * 9 + digits[pos--];
        out[i] = (uint8_t)s;
        p2 = p1;
        p1 = s;
    }
    *x_io = x;
}
