# 07 — KoboldCpp: K9 support exploration (2026-10-06)

Second runtime for the K9 disk format (GGUF sentinel type 43, decode-on-load),
explored end-to-end: fork-style repo, port, build, bit-exactness gate.
Companion to docs/05 (llama.cpp fork facts); assumes its design and theorem.

## 1. TL;DR

- **K9 runs in KoboldCpp.** The vendored llama.cpp loader takes the byte-identical
  patch from the llama.cpp fork (same sentinel id 43 happened to be free in the
  vendored ggml too), the unity build compiles it, and the ported loader says
  `K9 model with 197 stream tensors (materialize: q8_0)` for the 1.5B k63 file.
  Decoded-weights gate K9-vs-materialized-q8_0: **PASS** (§5).
- The port is small because KoboldCpp is not a wrapper around llama.cpp — it is
  llama.cpp. The repo root is a full llama.cpp tree (vendored, recent snapshot)
  with KoboldCpp's own adapter merged over it; the K9 patch surface is identical.

## 2. Repository facts

- Upstream: github.com/LostRuins/koboldcpp, default branch `concedo`, HEAD
  `f9a32456e` (2026-10-03), v1.122.1 (`KcppVersion` in koboldcpp.py).
- On GitHub the repo **is a fork of ggml-org/llama.cpp** (API: `fork: true`,
  `parent: ggml-org/llama.cpp`) — full llama.cpp history (repo ≈ 630 MB with
  `.git`; the llama.cpp tree including `convert_hf_to_gguf.py` sits at root), with
  KoboldCpp's C++ (`gpttype_adapter.cpp`, `expose.cpp`, `kcpp_*`) and the Python
  launcher (`koboldcpp.py`) merged on top. The `upstream` branch mirrors ggml.
- **One-fork-per-network rule consequence:** `auRose94/llama.cpp` already occupies
  the llama.cpp fork network, so `auRose94/koboldcpp` (a network fork) is
  impossible — both raw API and `gh` return the existing llama.cpp fork. The K9
  port therefore lives in a **standalone repo**: github.com/auRose94/koboldcpp
  (branches `concedo` = stock snapshot `f9a32456e`, `k9` = the port, commit
  `6357ae7ad`). Standalone ≠ network fork: GitHub-PRs *into* LostRuins/koboldcpp
  are also blocked for this account as long as the llama.cpp fork exists, so any
  KoboldCpp-side upstream contribution must come from another account (or after
  deleting the llama.cpp fork — not planned; the llama.cpp PR is the priority).
- Vendored llama.cpp vintage: the loader struct/diff matches our fork's base
  (`c25030496`, Oct 2026) within a ~37-line semantic drift — commented-out debug
  logs, a 64 MB direct-IO buffer tweak, an added `layer_name_to_number` helper.
  ggml type enum in the vendored `ggml.h`: ids 41/42 = `Q1_0`/`Q2_0` (upstream
  modern pair), `GGML_TYPE_COUNT = 43` → **43 free**, same as the fork's base
  sentinel placement.
- Unity build: `src/llama.cpp` `#include`s the whole `src/*.cpp` set
  (`llama-model-loader.cpp` included; `llama.o` is one TU). KoboldCpp Linux
  binaries are `.so` files loaded by `koboldcpp.py` (`koboldcpp_default.so`
  built by `make koboldcpp_default`, plain CPU/AVX2 build without BLAS).

## 3. The port (branch `k9`, standalone repo, commit `6357ae7ad`)

| file | change |
|---|---|
| `src/k9.h`, `src/k9.cpp` | copied verbatim from the llama.cpp fork (rANS + directory + dequant); local q8_0 block struct renamed `k9_block_q8_0` (unity-TU collision safety) |
| `ggml/include/ggml.h` | `GGML_TYPE_K9 = 43`, `GGML_TYPE_COUNT 43 → 44` (id free in this snapshot as in the fork's base) |
| `ggml/src/ggml.c` | K9 type-traits entry (blck_size 1, type_size 1 → `ggml_nbytes()` = blob length; ops reject the type) |
| `src/llama-model-loader.{h,cpp}` | same hooks as the fork: `k9.directory` KV parse (`k9_load_meta()`, called from the two file-based constructors), virtual/meta-device sizing remap in `create_tensor`'s `files.empty()` path, materialization block before `check_tensor_dims` (shape/divisibility checks, buffer per materialized type), K9 branch in `load_all_data` (mmap-alias-free decoding into a host Q8_0 buffer) |
| `src/llama.cpp` | one line: `#include "k9.cpp"` inside the unity TU |
| `Makefile` | `src/k9.cpp` added to the `llama.o` dependency line |
| `gguf-py/gguf/constants.py` | vendored copy: `K9 = 43` + `GGML_QUANT_SIZES` entry |

Adaptation notes (everything else is the fork's code verbatim):

- KoboldCpp has a third `llm_kv = LLM_KV(...)` constructor path (external gguf
  metadata; no files loaded) — intentionally *not* hooked, matching the
  llama.cpp fork's behavior: a no-file loader cannot decode K9 data and its
  `load_all_data` early-return covers it.
- `kcpp_backend_default`/`gpttype_adapter.cpp`/`expose.cpp` unchanged: everything
  below the loader is stock Q8_0/F16, exactly as in llama.cpp.
- `usemmap=False` is KoboldCpp's default for CPU builds → gates exercise the
  `file->seek/read_raw` (non-mmap) branch of the K9 `load_all_data` hook, plus
  the mmap branch in llama-server; both were green in session 24.

## 4. Reproduce

Linux CPU (this session, ~5 min at `-j24`):

```sh
cd ~/Work/koboldcpp && git checkout k9 && make koboldcpp_default -j
# serve a K9 file: (see experiments/k9_koboldcpp_gate.py for the exact flags)
python koboldcpp.py --model results/qwen_1.5b_k63.gguf --usecpu \
  --threads 8 --host 127.0.0.1 --port 5055 --contextsize 2048
```

CUDA/hipblas builds of the `k9` branch are not yet built here (the patch is
loader-side and backend-agnostic; per docs/05 CUDA/HIP builds are green in the
llama.cpp fork, and KoboldCpp's CUDA recipes live in `.github/workflows/` /
`scripts/`). sm_120 note: use the fork's approach (CUDA 13.x, architectures 120)
when building `koboldcpp_cublas`.

## 5. Bit-exactness gate (K9 sentinel vs materialized q8_0, same binary)

`experiments/k9_koboldcpp_gate.py` (base9-quantization). The reference arm is
`results/base1p5b_k63_materialized_q8_0.gguf`, produced by
`experiments/materialize_k9_gguf.py`: pure-Python decode of the same `.k9`
artifact (`experiments/k9.py`) into the exact Q8_0 block canvas verified
byte-for-byte against the C decoder in session 24 (`k9_llamacpp_gate.py`),
non-K9 tensors copied from the f16 skeleton, `k9.*` KVs dropped → an ordinary
GGUF that stock loaders accept. 197 q8_0-packed tensors (28 layers × 7 linears
+ k99 embedding) + 141 copied tensors, 1.53 GiB.

- Both models: 338 tensors, arch qwen2, GGUF V3; both arms reach serving state
  in ≈15-20 s CPU end-to-end (KoboldCpp readiness poll counts 7-8 × 2 s each
  arm; K9-decode overhead indistinguishable at 1.5B scale).
- Greedy generation (temperature 0, top_k 1, top_p 1, rep_pen 1, seed 42,
  6 prompts × 24 tokens, `/api/v1/generate`): **token-identical K9 vs
  materialized — PASS** (6/6 prompts byte-equal; GATE: PASS, exit 0).
- Materialized arm prints `(guessed) unknown` file type (cosmetic;
  `general.file_type` is f16-skeleton metadata — same cosmetic caveat as the
  K9 files, docs/05 §2).
- **7B k63 also PASS** (second data point, same binary+flags, 16-token greedy):
  `merged7b_k63_materialized_q8_0.gguf` = 198 q8_0-packed tensors (196 linears
  + embedding + untied lm_head, all k63/embed99 RTN — no GPTQ-perm records) +
  141 copies, 7.54 GiB; 6/6 prompts token-identical, exit 0.

## 6. Runtimes that could in theory support K9 (survey)

Design property from the theorem (docs/05 §2): K9 is a *loader-only container*;
everything downstream executes as stock Q8_0/F16. Support therefore means one
thing per runtime: decode blobs where weights enter memory.

| runtime | path | status |
|---|---|---|
| **llama.cpp** | `llama-model-loader` hook (this is the canonical patch) | done (session 24, branch `k9`) |
| **KoboldCpp** | vendored llama.cpp loader, byte-identical patch, unity include | **done this session** |
| Jan, GPT4All, text-generation-webui, llama-cpp-python | they ship llama.cpp binaries/libs → get K9 by patching their pinned llama.cpp; get it free once the upstream container-type lands | pattern-verified, not tested |
| llamafile | vendors llama.cpp at a pin inside a Cosmopolitan build; same loader patch, cosmopolitan quirks aside | pattern-verified, not tested |
| Ollama | vendors llama.cpp under `llama/` (Go binding); same loader patch + heavy Go/CMake build | plausible, medium effort |
| LM Studio | closed, llama.cpp-based engine: nothing to patch locally; picks K9 up automatically once the upstream PR ships in their runtime updates | automatic-after-upstream |
| vLLM | native GGUF reader (gguf-py + dequant-in-python); K9 = python rANS decoder + one dispatch hook in the gguf model loader | straightforward shape, not tested |
| HF transformers (GGUF) | same GGUF-dequant-in-python shape | straightforward shape, not tested |
| whisper.cpp / stable-diffusion.cpp / ggml-family tools | ggml loader patterns as in llama.cpp; low relevance | pattern applies |

Not applicable: non-GGUF runtimes (EXL2, TensorRT-LLM, v1-legacy GGML readers).
For runtimes that only *read* GGUF metadata (file listings, model cards,
`gguf-dump`-style tools), a K9 file already reads cleanly given the type-43
entry — only tensor-materialization paths need the decoder.

## 7. Caveats & open items

- Load-time/serial-decode caveats carry over verbatim from docs/05 §2
  (single-threaded rANS decode on load; runtime memory = materialized size — a
  disk-format win, not a resident-memory win, until stage 2's resident
  containers).
- CUDA/hipblas/Vulkan koboldcpp builds of the `k9` branch: patch is
  backend-agnostic loader-side (docs/05 evidence), but not built here.
- `nimths`/K9 naming will change with the base9→ninths rename ([[memory]] plan);
  keep sentinel value 43 and the K9 KV names until the rename lands, then
  re-export + re-port in one sweep.
- The standalone-repo constraint above (network fork rule) is worth re-checking
  if the llama.cpp fork is ever deleted.