# exp23 — K9 performance: parallel encode + streaming decode

Qwen2.5-Coder-1.5B-Instruct, body k=63 + embed k=99, ent8
scales, C coder. No format change.

| threads | encode s | M sym/s | file MB |
|---|---|---|---|
| 1 | 8.34 | 185.1 | 1112.0 |
| 4 | 3.98 | 388.3 | 1112.0 |
| 24 | 3.81 | 405.3 | 1112.0 |

- parallel speedup at 24 threads: **2.2×** vs single-threaded
- output bytes identical across thread counts: size=True, sha256=True
- streaming decode + load: **22.51s** (68.6 M sym/s), **0 digit mismatches**, peak GPU alloc 9.14 GB, peak process RSS 15.3 GB
  (the loader's own device allocation is one row-chunk — 4096x1536 floats,
  ~25 MB — rather than the whole tensor, so the embed no longer triggers
  the transient CUDA OOM seen in exp21's full-fp32 path; the reported GPU
  peak is the process high-water mark and is dominated by the resident
  fp32 model, and the RSS figure includes the fp32 master copies and digit
  arrays kept for verification, not the loader)
- perplexity after the streaming reload: code 4.4995
  (identical to the in-memory quantized model)

wall 69s
