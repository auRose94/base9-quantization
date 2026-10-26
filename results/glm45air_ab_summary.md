# GLM-4.5-Air: ninths K9 k63/e99-v2 vs stock q8_0 — paired battery (2026-10-26)

Protocol: same server build, same flags (the -ot CPU-expert split, KV q8_0,
8 KiB ctx), same 24-task pool from phase2/eval_gdscript.py, greedy decoding,
thinking disabled, max_tokens 480. Only the weight container differed.
NOTE: both legs ran without the system prompt attached (harness bug, fixed
afterward) — conditions identical across arms, so the paired comparison is
valid; absolute numbers may shift with the system prompt present.

| arm | file | GiB | correct | wall | notes |
|---|---|---|---|---|---|
| K9-v2 | glm45air_k63_e99_v2.gguf | 73.46 | **22/24 (91.7%)** | 169.7 s | load ~150 s (lane-parallel decode) |
| stock q8_0 | glm45air_q8_0.gguf | 109.4 | **22/24 (91.7%)** | 171.0 s | load ~90 s |

Task-level parity: identical PASS/FAIL pattern — the same two tasks fail
(`export_var`, `resource_class`), both arms with near-identical token counts
per task (largest delta 51 tok on enum_match). Per-token outputs diverge as
expected between grids, but every mechanical grader lands the same way.

Failures (identical in both arms): chatty preamble ("Of course! Here is ...")
instead of code-first answering on two prompt shapes — a prompt-style issue,
not a quantization artifact.

Conclusions:
1. **K9 materialized = q8_0-class behavioral parity** on this battery — the
   k63/k99 grid costs no measurable task-level quality here, at ~67% of the
   file size (73.5 vs 109.4 GiB) and 40 MB/s-class upload/deploy cost.
2. Load cost of compression: K9-v2 loads in ~150 s vs ~90 s for q8_0 on this
   box (16 workers lane-decoding 74 GB of streams). v2 closes most of v1's
   gap (554 s).
3. Next for the REAP criteria: this pool measures API-precision; trimming
   decisions still need a difficulty gradient (larger pools / harder splits)
   per rose's call.
