/* RTL<->C cross-check: the fork's K9 vec_dot kernels compiled as-is and fed
 * the exact stimulus the cocotb Gate-0 bench and the on-silicon gate use.
 *
 * Kernels ggml_vec_dot_k9_{4,6,7}_q8_0_generic are copied verbatim from
 * ~/Work/llama.cpp (branch k9, ggml/src/ggml-cpu/quants.c); block structs from
 * ggml/src/ggml-common.h (QK9_SUPER = 256). The only deviation: FP16_TO_FP32
 * is a hand-rolled IEEE conversion (same result as the fork's F16C path for
 * every finite/inf input, and NaN never occurs in this data path).
 *
 * Build: gcc -O2 -ffp-contract=off -o k9_c_harness k9_c_harness.c
 * (the -ffp-contract=off is the load-bearing flag: the kernel's contract is
 * separate mul+add, no FMA contraction.)
 *
 * I/O:  ./k9_c_harness <in.bin> <out.bin>
 * in.bin:  u32 type (4|6|7)   u32 N
 *          N superblocks (136 B for type 4, 200 B for 6, 264 B for 7)
 *          N * 272 B of Q8_0 activations (8 blocks x 34 B, fp16 scale LE + 32 int8)
 * out.bin: N u32 bit patterns of the running fp32 sumf after each superblock
 * (C accumulates sumf across the chain, exactly matching docs golden cums[],)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

typedef uint16_t ggml_half;

#define QK9_SUPER 256

typedef struct {
    ggml_half ds[4];
    uint8_t  qs[128];
    uint8_t  hh[64];
} block_k9_6;            /* 200 B */

typedef struct {
    ggml_half ds[4];
    uint8_t  qs[128];    /* nibbles, bias -8 */
} block_k9_4;            /* 136 B */

typedef struct {
    ggml_half ds[4];
    int8_t   qs[256];    /* d - H, H = 49 */
} block_k9_7;            /* 264 B */

typedef struct {
    ggml_half d;
    int8_t    qs[32];
} block_q8_0;            /* 34 B */

_Static_assert(sizeof(block_k9_6) == 200, "k9_6 size");
_Static_assert(sizeof(block_k9_4) == 136, "k9_4 size");
_Static_assert(sizeof(block_k9_7) == 264, "k9_7 size");
_Static_assert(sizeof(block_q8_0) == 34,  "q8_0 size");

/* fp16 -> fp32, IEEE exact (matches RTL f16_to_f32 and numpy bit-for-bit on
 * all finite/inf inputs; subnormal h renormalized, inf preserved, nan quiet). */
static float h2f(ggml_half h) {
    uint32_t s = (uint32_t)(h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t bits;
    if (e == 0x1F) {
        bits = m ? (s | 0x7FC00000u | (m << 13)) : (s | 0x7F800000u);
    } else if (!e) {
        if (!m) {
            bits = s;
        } else {
            /* IEEE renormalization exactly as f32_pkg f16_to_f32: value = m * 2**-24,
             * MSB of m at bit p -> biased exponent 103 + p, mantissa m << (23-p). */
            int p = 0;
            for (int i = 9; i >= 0; i--) if (m & (1u << i)) { p = i; break; }
            bits = s | ((uint32_t)(103 + p) << 23) | ((m << (23 - p)) & 0x7FFFFFu);
        }
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

static uint32_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

void ggml_vec_dot_k9_6_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const int qk = QK9_SUPER;
    const int nb = n / qk;

    const block_k9_6 * x = vx;
    const block_q8_0 * y = vy;

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_k9_6 * xb = &x[i];

        for (int g = 0; g < 4; g++) {
            const float d0 = h2f(xb->ds[g]);
            const uint8_t * qs = xb->qs + g*32;
            const uint8_t * hh = xb->hh + g*16;

            float sumi = 0.0f;

            for (int k = 0; k < 2; k++) {
                const block_q8_0 * yb = &y[i*8 + g*2 + k];
                const float d1 = h2f(yb->d);
                const int8_t * qy = yb->qs;

                int sumi_block = 0;
                for (int b = 0; b < 16; ++b) {
                    const int c0 = 32*k + 2*b;
                    const int c1 = c0 + 1;
                    const uint8_t lb = qs[16*k + b];
                    const int h0 = (hh[c0 >> 2] >> ((c0 & 3) * 2)) & 0x03;
                    const int h1 = (hh[c1 >> 2] >> ((c1 & 3) * 2)) & 0x03;
                    sumi_block += ((int)((lb & 0x0F) | (h0 << 4)) - 32) * qy[2*b    ]
                               +  ((int)((lb >> 4)    | (h1 << 4)) - 32) * qy[2*b + 1];
                }
                sumi += d1 * sumi_block;
            }

            sumf += d0 * sumi;
        }
    }

    *s = sumf;
}

void ggml_vec_dot_k9_4_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const int qk = QK9_SUPER;
    const int nb = n / qk;

    const block_k9_4 * x = vx;
    const block_q8_0 * y = vy;

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_k9_4 * xb = &x[i];

        for (int g = 0; g < 4; g++) {
            const float d0 = h2f(xb->ds[g]);
            const uint8_t * qs = xb->qs + g*32;

            float sumi = 0.0f;

            for (int k = 0; k < 2; k++) {
                const block_q8_0 * yb = &y[i*8 + g*2 + k];
                const float d1 = h2f(yb->d);
                const int8_t * qy = yb->qs;

                int sumi_block = 0;
                for (int b = 0; b < 16; ++b) {
                    const uint8_t byte = qs[16*k + b];
                    sumi_block += ((int)(byte & 0x0F) - 8) * qy[2*b    ]
                               +  ((int)(byte >>   4) - 8) * qy[2*b + 1];
                }
                sumi += d1 * sumi_block;
            }

            sumf += d0 * sumi;
        }
    }

    *s = sumf;
}

void ggml_vec_dot_k9_7_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const int qk = QK9_SUPER;
    const int nb = n / qk;

    const block_k9_7 * x = vx;
    const block_q8_0 * y = vy;

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_k9_7 * xb = &x[i];

        for (int g = 0; g < 4; g++) {
            const float d0 = h2f(xb->ds[g]);
            const int8_t * csp = xb->qs + g*64;

            float sumi = 0.0f;

            for (int k = 0; k < 2; k++) {
                const block_q8_0 * yb = &y[i*8 + g*2 + k];
                const float d1 = h2f(yb->d);
                const int8_t * qy = yb->qs;

                int sumi_block = 0;
                for (int b = 0; b < 32; ++b) {
                    sumi_block += (int) csp[32*k + b] * qy[b];
                }
                sumi += d1 * sumi_block;
            }

            sumf += d0 * sumi;
        }
    }

    *s = sumf;
}

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]); return 1; }
    FILE * fi = fopen(argv[1], "rb");
    FILE * fo = fopen(argv[2], "wb");
    if (!fi || !fo) { perror("open"); return 1; }

    uint32_t type, n_sbs;
    if (fread(&type, 4, 1, fi) != 1 || fread(&n_sbs, 4, 1, fi) != 1) { perror("hdr"); return 1; }

    size_t sb_size = type == 4 ? 136 : type == 6 ? 200 : 264;
    void * wbuf = malloc(sb_size * n_sbs);
    void * abuf = malloc(272u * n_sbs);
    if (fread(wbuf, 1, sb_size * n_sbs, fi) != sb_size * n_sbs) { perror("wblk"); return 1; }
    if (fread(abuf, 1, 272u * n_sbs, fi) != 272u * n_sbs) { perror("act"); return 1; }

    float final_s;
    int n = (int)(QK9_SUPER * n_sbs);
    if (type == 4)      ggml_vec_dot_k9_4_q8_0_generic(n, &final_s, 1, wbuf, 0, abuf, 0, 1);
    else if (type == 6) ggml_vec_dot_k9_6_q8_0_generic(n, &final_s, 1, wbuf, 0, abuf, 0, 1);
    else                ggml_vec_dot_k9_7_q8_0_generic(n, &final_s, 1, wbuf, 0, abuf, 0, 1);

    /* Reproduce the per-superblock cumulative state by running the same kernel
     * over prefixes 1..N (cheap, and gives the C-side analog of golden cums[]). */
    for (uint32_t i = 1; i <= n_sbs; i++) {
        float s;
        int ni = (int)(QK9_SUPER * i);
        if (type == 4)      ggml_vec_dot_k9_4_q8_0_generic(ni, &s, 1, wbuf, 0, abuf, 0, 1);
        else if (type == 6) ggml_vec_dot_k9_6_q8_0_generic(ni, &s, 1, wbuf, 0, abuf, 0, 1);
        else                ggml_vec_dot_k9_7_q8_0_generic(ni, &s, 1, wbuf, 0, abuf, 0, 1);
        uint32_t bits = f2b(s);
        if (fwrite(&bits, 4, 1, fo) != 1) { perror("write"); return 1; }
    }
    (void)final_s;

    fclose(fi); fclose(fo);
    return 0;
}