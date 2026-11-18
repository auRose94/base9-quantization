#include <hip/hip_runtime.h>

// k9_dequant_f32 — kennel M1 gate kernel.
//
// One thread per weight element. Digits are the rANS-decoded u8 codes,
// scales are the ent8/f32-decoded per-(row, 64-col-group) multipliers.
// Normative math (ninths-codec dequant_level, docs/12 sec 4.1):
//     w = (m * (float)(d - H)) / H,   H = (k-1)/2
// The multiply-then-divide sequence has no add, so no FMA contraction is
// possible and the kernel is bit-exact vs the CPU reference by construction.

extern "C" __global__ void k9_dequant_f32(const unsigned char* __restrict__ digits,
                                          const float* __restrict__ scales,
                                          float* __restrict__ out,
                                          unsigned long long n,
                                          unsigned int cols,
                                          unsigned int n_groups,
                                          int h_int,
                                          float h_flt) {
    unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    float x = (float)((int)digits[i] - h_int);
    out[i] = (scales[r * n_groups + (c >> 6)] * x) / h_flt;
}
