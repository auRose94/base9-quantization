# 16. kennel — experimental ninths engine for Strix Halo

Date: 2026-11-17. Status: **research + scaffolding** (crate live at
`rs/crates/kennel`, device probe green on silicon).

kennel is an experimental LLM inference engine, CLI-first ("llmhamster
levels of limited"), targeting exactly one machine: the A9 Mega. Its
experimental thesis: **co-design the engine with the container format.**
K9/ninths can MATERIALIZE any per-tensor layout, so kennel demands the
layout its kernels want, per model family, instead of inheriting
one-size-fits-all GGUF shapes the way llama.cpp/vLLM-style engines must.

## 1. The machine, as kennel sees it

`kennel devices` (source of truth: ROCm 10.1.0 at /opt/rocm, run
2026-11-17):

```
HIP runtime version: 71600000
[0] AMD Radeon 8060S Graphics (gfx1151)
  compute capability 11.5  |  20 CUs  |  warp 32  |  2048 threads/CU
  clocks: gpu 2900 MHz, mem 1000 MHz  |  bus 256 bit  |  L2 2048 KiB
  memory: 58.8 GiB unified (integrated: true)
  LDS opt-in: 64 KiB/block
  pageableMemoryAccess: false  |  hostRegisterSupported: true
```

Decoded, with the M3-HIP rung's lessons folded in:

| Fact | Engine consequence |
|---|---|
| 20 "CUs" = 20 WGPs = 40 physical CUs (hipDeviceProp header note: WGP mode halves the count) | tile/scheduling math must use WGPs |
| 58.8 GiB `totalGlobalMem` = **GTT pool** (checked: `mem_info_gtt_total` = 63.1 GB; default ≈ half of RAM), BIOS UMA carve-out is 8 GiB (`mem_info_vram_total`) | stock config caps `hipMalloc` at ~59 GiB; kernel params (below) lift GTT to ~124 GiB |
| `pageableMemoryAccess: false` | confirms M3-HIP: anonymous malloc memory is NOT coherently kernel-accessible; registration is mandatory for host-resident weights |
| `hostRegisterSupported: true` | the coherent path exists; the fork's lazy region-registration at first graph compute (fork c5936d4c8) is the proven pattern |
| warp 32, 2048 threads/multiProcessor, 64 KiB LDS opt-in, L2 reported 2 MiB (32 MiB L3/Infinity Cache is the RDNA3.5-iGPU differentiator) | kernel shape constraints: wave32-friendly tiles, LDS budget 64 KiB |
| bus 256 bit, LPDDR5X-8000 → 256 GB/s theoretical, **~170–200 GB/s achievable** (measured by every engine; ~85 W package power bounds prefill before FLOPs do) | decode ceiling rule of thumb: t/s ≈ 0.65–0.75 × 256 / active_bytes_per_token |
| `ulimit -l` = 8 MiB for rose's user (realtime group has unlimited; she's not in it) | every Strix Halo engine runs `LimitMEMLOCK=infinity` — config rung, see §4.3 |

The bandwidth line is the whole game on this chip: decode is a memory
wheel. Every design decision below serves either (a) fewer bytes per
token (K9 rungs, per-tensor k), (b) overlapping the wheel with compute
(M3-style resident experts + CPU/GPU split), or (c) prefill throughput
(bigger tiles, WMMA paths).

## 2. Local assets this builds on

- **k9 fork** (C++, tip c5936d4c8): all silicon lessons — M3 paging,
  M3-HIP `hipHostRegister` region registration, the anonymous-memory
  landmine, MoE-cache decode hook. kennel is NOT a port of it; it is the
  second consumer of those lessons.
- **rs/ninths-codec**: byte-exact Rust rANS + ent8 + dequant primitives
  (`dequant_level`, `lane_plan` from docs/11). The reference side of
  every byte-exact gate kennel will ever run.
- **rs/ninths-cli**: codec self-tests only — kennel is the engine CLI,
  sibling crate.
- **ratchet**: orchestration/OpenAI-compat layer over llama-server;
  kennel eventually offers it a native engine instead.
- **docs/14 Strata assessment**: the closest prior art — per-model-family
  modules (src/qwen35, src/dsv41), expert cache = our M3, SSD packs at
  1.75–2.25 bpw, MIT, and their headline config is this exact machine.
  kennel takes the per-model thesis further (container-level
  specialization, not just code paths) with the quality tier above
  (k63 ppl-free) instead of IQ2/IQ3.

## 3. Milestone ladder

| Rung | Gate |
|---|---|
| M0 device probe | ✅ green (output above) |
| M1 GGUF reader + GPU dequant | Rust mmap GGUF (type table incl. K9) → tensor to GPU (registered) → dequant kernel → **byte-exact vs ninths-codec** — reader ✅ green on stock q8_0 AND K9-typed flint (containers already carry `ninths.materialize` + `ninths.directory2`, the materialization plan hook) |
| M2 GEMV/GEMM on dequantized K9 | greedy forward on the 1.5B, tokens to stdout |
| M3 full forward + tokenizer | `kennel run` end-to-end, batch 1 |
| M4 serve | OpenAI-compat HTTP, ratchet-compatible |
| M5 per-model specialization | first "materialization plan": exporter emits layout + kernel plan per (model, GPU) — layout-as-ABI (§5.3), the unified bet no engine has shipped |

## 4. The Strix Halo software landscape (researched 2026-11-17)

The single-silicon engine thesis is **proven and crowded**: an entire
engine wave landed Aug–Nov 2026. kennel's lane is still open — none of
them co-design the container; they all inherit GGUF layouts and tune
kernels around them.

### 4.1 The engine wave

| Engine | License | Numbers on this chip (Qwen3.8-Flash-Next class) | Notes |
|---|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) v0.1.40 | MIT, 45k★ | 53.8 t/s decode @8K, 1293 t/s prefill (maintainer's identical box) | per-arch hipBLASLt tables, expert cache, `STRATA_UMA_REGISTER` arena pinning = our M3-HIP pattern |
| [gufo](https://github.com/gufo-org/gufo) | MIT | 28.3 t/s AR decode flat to 131K depth, MTP 62–116 t/s, pp 1.7K | **Strix-Halo-only**; checkpoint cache for agents (203K-token turn resume 0.7 s); measured 96 GB/s RAM restore path, 3.3 GB/s NVMe |
| [halogen](https://github.com/peonist-ai/halogen-flash-server) | proprietary EULA, container-only | 66.7 t/s, pp 1.7–1.9K, TTFT 1.2 s, 74.2 GiB resident | "every kernel written for this one GPU"; build rejects other arches; AGPL clean-room exists (chlorine-server) |
| llama.cpp master | MIT | decode falls to 8 t/s @131K (KV re-walk) | but now carries `mmq-config-rdna3-5.cuh`, MMVQ tuned on 8060S, WMMA gated-delta-net kernels — all fork-relevant |
| [Maya-XS](https://github.com/mw00/project-maya), moe-cache, ds4-ssd | mixed | SSD expert paging plugins | the tiering ecosystem |

Benchmark cross-check ([strix-halo-llm-bench](https://github.com/Johnny-dai-git/strix-halo-llm-bench), ~84 W): Halogen 47.6 / gufo 49.0 / Strata 34.0 t/s on the same GGUF — engine deltas of 1.4× at identical bytes.

### 4.2 Stack facts that shape kennel

- **ROCm for gfx1151**: natively supported (no HSA_OVERRIDE since the TheRock builds); official matrix lists 8060S; but the doc trains disagree (10.1.0 landing vs 7.13/7.14 install vs 7.2.1 radeon-ryzen) — six-week cadence, Windows HIP SDK folded into "ROCm Core SDK" at 10.0. Local truth on the A9: install labeled 10.1.0, HIP runtime self-reports 71600000. TheRock ships per-arch `gfx1151` tarballs (what Strata pins).
- **Kernel ISA**: WMMA yes (fp16/bf16/i8/i4, 16×16×16), **MFMA no**; wave32 primary, wave64 per-kernel opt-in; 64 KiB LDS/workgroup (doubled vs RDNA3 — matches our probe); 32 MiB L3/Infinity Cache unique among RDNA3.5 iGPUs; hipBLASLt works (Strata ships gfx1151 tuning tables) but gufo/halogen write their own fused kernels entirely.
- **Hybrid GDN MoE is the native workload**: llama.cpp has WMMA `gated-delta-net-mma` kernels + HIP bf16 hi/lo split-K trick; depth-flat decode comes from recurrent state + MTP verification, not raw bandwidth (gufo 28 t/s flat at 131K vs llama.cpp 8).
- **Memory failure modes**: `amdgpu.noretry=0` converts lost-page bugs into silent hangs; kernel compaction stalls under pressure; unclean exit can leave GPU memory driver-held until reboot. Watchdogs must distinguish D-state stalls from wedges (cf. our GGML_NO_BACKTRACE lesson).
- **NPU = sidecar only**: rerank/ASR/small models on XDNA beside the GPU LLM; GPU+NPU concurrency needs fabric clock pinned (TheRock#8709) — consistent with our NPU-paperweight verdict.

### 4.3 Machine-config rung (rose's call, needs sudo + reboot)

Current A9 state vs the gufo/Strata recipe: GTT 59 GiB (stock), carve-out
8 GiB, memlock 8 MiB, no amdgpu kernel params.

1. Kernel cmdline: `amdgpu.gttsize=126976 ttm.pages_limit=32505856` → GTT ≈ 124 GiB (Strata variant: 112 GiB + `amd_iommu=off`, but IOMMU off breaks the NPU — skip it, the NPU path stays available).
2. `LimitMEMLOCK=infinity` in systemd user session (limits.d).
3. BIOS: UMA carve-out 8 GiB → minimum (512 MiB–1 GiB); GTT supersedes it; frees ~7 GiB for host.

## 5. Per-model specialization: precedents and evidence (researched 2026-11-17)

### 5.1 Proven wins (measured, sourced)

| Case | Win | Caveat |
|---|---|---|
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) JIT FP8 GEMM, DeepSeek's exact 1×128/128×128 scale scheme | 1350→1550 TFLOPS H800, ~2× vs then-best CUTLASS on some shapes | the scheme had NO stock library equivalent — quant/kernel co-design was the whole win |
| [vLLM × HPC-Ops](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops/) fully fused FP8 MoE (Hunyuan Hy3) | 1.59–2.5× over best Triton/CUTLASS MoE kernel | attention backend supports only Hy3 — explicit per-family specialization, upstreamed as a plugin |
| [bitnet.cpp](https://arxiv.org/html/2502.11880v1) ternary LUT runtime | 2.37–6.17×, −82% energy | runtime built around ONE weight format — the "layout = ABI" extreme |
| MTP/draft heads in serving (SGLang/AMD blogs) | +60% throughput, 1.25–2.11× | acceptance is temperature-bound: ~2× at temp 0 collapses to **4–9%** at realistic sampling; can slow prefill |
| [FLA](https://github.com/fla-org/flash-linear-attention) chunk_gdn vs FlashAttention | ~3× at B=1 T=8192 | **loses** at short sequences (0.157 vs 0.631 ms) — shape-conditional, ship both paths |
| Fused Mamba2 (conv+SiLU+scan+norm); NVIDIA cuTile vs Triton on B300 | 1.50–2.51×; 1.14–1.82× | kernels needed per-GPU-generation rewrites (B200, B300 passes) |
| [MLX](https://github.com/hoodini/apple-silicon-arena) vs llama.cpp decode (Apple) | 10–50% at 14B+ | "specialize to the memory system" pays independent of model — llama.cpp wins prefill |
| [ktransformers](https://github.com/kvcache-ai/ktransformers) prefill vs llama.cpp | "27.79×" | weak-baseline artifact: **~1.23× decode vs a serious stack**, and AMX-tuned kernels flop on EPYC |

### 5.2 Counter-evidence (what erodes specialization)

- **Generic engines absorb winners in ~6–18 months**: fused MoE went Mixtral-offloading → vLLM upstream; Hunyuan kernels became vLLM backends; llama.cpp absorbed Qwen3-Next's gated DeltaNet (months + a 20k-line refactor — the architecture-dispatch layer, not the kernel math, is the friction).
- **Specialization rots**: FlashMLA's 2026 releases dropped Hopper AND DeepSeek V3/V3.2/V4.0 (users told to pin old commits); Mamba kernels need per-generation rewrites; ktransformers' ISA coupling.
- **Wins are workload-conditional**: FLA vs FA by shape; MTP by temperature; generic-but-tuned beats specialized-but-naive (+46% from merely tuning Triton MoE configs; generic dequant-then-GEMM measured at 19–24% of peak — layout and kernel NOT co-designed wastes most of the win).

### 5.3 Design response — what this means for kennel specifically

1. **Layout = kernel ABI** (DeepGEMM/bitnet lesson): the exporter ships the layout the fastest kernel wants; this is exactly what K9 materialization buys and no GGUF-bound competitor can do.
2. **Per-shape JIT, baked at export** (DeepGEMM/FlashInfer lesson): carry tuned tile configs in the artifact, compile at first run — never precompile all variants (FlashInfer's 97-target explosion).
3. **Fused MoE keyed to exact topology** (HPC-Ops lesson): expert count/top-k/shared structure already live in GGUF KV; materialization extends it to the kernel shape itself.
4. **MTP as optional attachment**: keep the draft head in the artifact; engine disables it at hot sampling or prefill-heavy workloads (temperature-collapse evidence).
5. **Ship measured per-model stats, not just weights**: expert-activation locality, MTP acceptance profiles, shape-threshold tables — per-model data the engine cannot guess; a container carrying it turns engine tuning into artifact lookup.
6. **Specialize per (model × hardware-generation)**: the container should record target-aware materialization variants (gfx1151 first — kennel's single-silicon scope makes this cheap now).
7. **Stay plugin-shaped over a competent generic core, and version-pin kernels in the artifact** (the FlashMLA pin-a-commit failure mode is what a materialization plan prevents).
8. **Claim wins against the strongest generic baseline**, split prefill/decode/temperature (ktransformers lesson).

**The open bet, confirmed open**: tinygrad/MLX/torch.compile prove per-model graph compilation works; DeepGEMM/FlashInfer/HPC-Ops prove per-model/per-shape kernels pay 1.5–2.5×; **no one has unified them in the artifact** — container materializes the layout AND the kernel plan per (model, GPU). K9 can; that is kennel's reason to exist beyond the wheel.
