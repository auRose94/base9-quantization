// HMM divergence test for gfx1151 (Strix Halo iGPU) — the standalone
// decisive experiment for docs/13 §2.6.
//
// Reproduces the k9 residency access pattern against an anonymous
// MAP_NORESERVE region under an initialized HIP context:
//   1. CPU writes pattern A (1.0)
//   2. GPU kernel reads the host pointer directly (zero-fault fault-in)
//   3. CPU rewrites pattern B (2.0)  — the residency re-decode
//   4. GPU kernel reads again — coherent or stale?
//   5. same after MADV_DONTNEED (the eviction path)
//   6. same after hipHostRegister (the retired design)
//
// Build: /opt/rocm/bin/hipcc -O2 -o hmm_divergence_test hmm_divergence_test.cu
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sys/mman.h>

__global__ void k9read_kernel(const float * p, float * out, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        s += p[i];
    }
    *out = s;
}

static float gpu_read(const float * host_ptr, int n, float * dout) {
    hipLaunchKernelGGL(k9read_kernel, dim3(1), dim3(1), 0, 0, host_ptr, dout, n);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("    kernel launch/sync error: %s\n", hipGetErrorString(e));
        return -9999.0f;
    }
    float r = -9999.0f;
    e = hipMemcpy(&r, dout, sizeof(float), hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        printf("    memcpy error: %s\n", hipGetErrorString(e));
        return -9998.0f;
    }
    return r;
}

int main() {
    const int N = 1024;
    const size_t SZ = 64ULL * 1024 * 1024;

    void * region = mmap(nullptr, SZ, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (region == MAP_FAILED) { printf("mmap failed\n"); return 1; }
    float * host = (float *) region;

    hipError_t e = hipInit(0);
    if (e != hipSuccess) { printf("hipInit failed: %s\n", hipGetErrorString(e)); return 1; }
    printf("HIP initialized\n");

    float * dout = nullptr;
    e = hipMalloc(&dout, sizeof(float));
    if (e != hipSuccess) { printf("hipMalloc failed\n"); return 1; }

    int skip_unregistered = getenv("SKIP_UNREG") != nullptr;
    // ---- test 1: plain anonymous region (faults without HMM; SKIP_UNREG=1 skips) ----
    printf("\n[1] anonymous MAP_NORESERVE region\n");
    float r1 = -1, r2 = -1, r3 = -1;
    if (!skip_unregistered) {
    for (int i = 0; i < N; i++) host[i] = 1.0f;
    r1 = gpu_read(host, N, dout);
    printf("    A->GPU read: %.1f (expect %d.0)\n", r1, N);

    for (int i = 0; i < N; i++) host[i] = 2.0f;
    r2 = gpu_read(host, N, dout);
    printf("    B->GPU read: %.1f (expect %d.0)  %s\n", r2, 2*N,
           r2 == 2.0f*N ? "COHERENT" : "DIVERGENT/STALE");

    printf("\n[2] same, after madvise(MADV_DONTNEED) on the page range\n");
    for (int i = 0; i < N; i++) host[i] = 3.0f;
    madvise(region, 65536, MADV_DONTNEED);
    r3 = gpu_read(host, N, dout);
    printf("    C->GPU read: %.1f (expect %d.0)  %s\n", r3, 3*N,
           r3 == 3.0f*N ? "COHERENT" : "DIVERGENT/STALE");
    }

    // ---- test 3: hipHostRegister the region (the retired design) -------
    printf("\n[3] same, after hipHostRegister(region, 64MB)\n");
    e = hipHostRegister(region, SZ, hipHostRegisterDefault);
    printf("    hipHostRegister: %s\n", e == hipSuccess ? "ok" : hipGetErrorString(e));
    if (e == hipSuccess) {
        for (int i = 0; i < N; i++) host[i] = 4.0f;
        float r4 = gpu_read(host, N, dout);
        printf("    D->GPU read: %.1f (expect %d.0)  %s\n", r4, 4*N,
               r4 == 4.0f*N ? "COHERENT" : "DIVERGENT/STALE");

        for (int i = 0; i < N; i++) host[i] = 5.0f;
        float r5 = gpu_read(host, N, dout);
        printf("    E->GPU read: %.1f (expect %d.0)  %s\n", r5, 5*N,
               r5 == 5.0f*N ? "COHERENT" : "DIVERGENT/STALE");
        hipHostUnregister(region);
    }

    // ---- test 4: GPU fault FIRST, then CPU rewrites the same page ------
    // (the exact ordering suspected for the k9 residual decay)
    printf("\n[4] ordering test: GPU fault first, then CPU write\n");
    for (int i = 0; i < N; i++) host[i] = 7.0f;
    float r6 = gpu_read(host + 4096, N, dout);   // fault a LATER page region first
    (void) r6;
    for (int i = 0; i < N; i++) host[i] = 8.0f;  // CPU rewrite of the ALREADY-faulted page
    float r7 = gpu_read(host, N, dout);
    printf("    F->GPU read: %.1f (expect %d.0)  %s\n", r7, 8*N,
           r7 == 8.0f*N ? "COHERENT" : "DIVERGENT/STALE");

    printf("\nhw: dev#0 props: ");
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
        printf("%s, HMM/managed=%d, integrated=%d\n", prop.name, (int) prop.managedMemory, (int) prop.integrated);
    }
    return 0;
}
