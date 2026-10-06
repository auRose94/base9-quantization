# 05 — llama.cpp fork: upstream-PR facts & constraints (2026-10-06)

Source material for writing an upstream llama.cpp PR/discussion — **facts only**.
The description text itself must be written in rose's own words (see §5: the
project's AI policy explicitly prohibits AI-written PR descriptions; "assistance"
here means accurate source material, checklists, and tooling support).

## 1. What is published

- Repo: https://github.com/auRose94/llama.cpp — branch `k9` at commit `14490cf2d`
  (pushed 2026-10-06 from `/home/rose/Work/llama.cpp`; remote `fork`. `origin`
  still points at ggml-org/llama.cpp, kept for future rebases).
- Single commit over upstream `master` `c25030496` (2026-10-05): 11 files,
  +797/−1. Contents:
  - `src/k9.h` (75) / `src/k9.cpp` (370): loader-side `.k9` stream decoder
    (rANS, vendored), shared by CPU/CUDA/HIP builds via decode-on-load.
  - `src/llama-model-loader.{h,cpp}` (+11/+162): sentinel-type hook in
    `create_tensor` + `load_all_data`; `k9.directory` KV parsing;
    `k9.materialize` option (memory: q8_0 default, f16 optional).
  - `examples/k9probe` (157 + 7 + CMake hookup): dumps digits, scales, or
    materialized weights for bit-exactness checking against
    `experiments/k9.py` / `k9_llamacpp_gate.py` (base9-quantization repo).
  - `ggml/include/ggml.h` (3) / `ggml/src/ggml.c` (9): `GGML_TYPE_K9 = 43`
    sentinel (count was 43 → 44; `blck_size=1, type_size=1` so `ggml_nbytes`
    = ne[0] = variable blob length; ops reject the type).
  - `gguf-py/gguf/constants.py` (2): K9 quantization-type entry.
- Fork's `master` is an older upstream snapshot than the commit's base — cosmetic
  only; the branch is self-contained.

## 2. Verbatim-fact sheet (all from RESEARCH_LOG.md, session 24, 2026-10-05)

**Design (stage 1, disk layer).** GGUF v3 derives tensor byte length from
type×shape, so a variable-length rANS stream cannot be a plain tensor. The fix:
loader-only sentinel type; per-tensor metadata in one KV `k9.directory` (name,
rank+shape, k, group, scale_mode, perm_flag, tbl/dig/sca/plen); exporter
`experiments/export_k9_gguf.py` re-embeds `.k9` blobs verbatim over a
`convert_hf_to_gguf.py` f16 skeleton (norms/biases/tokenizer). GPU-offloaded
tensors decode straight into VRAM; CPU tensors get one writable host buffer
(never aliasing the read-only mmap). Decode-on-load is backend-agnostic —
no kernels were added (stock Q8_0/F16 kernels execute).

**Exactness theorem (the load-bearing claim).** K9 dequant is `w = m·(d−H)/H`
with `H = (k−1)//2` — a scaled integer. Q8_0's per-32 block scale can be `m/H`
exactly and its int8 code `d−H ∈ [−H, H] ⊂ [−49, 49]` (2.6× headroom), so Q8_0
materialization is lossless modulo fp16 rounding of the block scale
(≤ 4.9e-4 relative, finer than k99's grid spacing). GGUF's transposed storage
aligns exactly with K9's per-out-row 64-group scales — an index permutation of
digits only. GPTQ full-width-perm tensors (plen = 4·c) scatter their scales →
those materialize as F16 instead.

**Bit-exactness gates (all PASS).** `examples/k9probe` + `k9_llamacpp_gate.py`,
C++ decode vs the Python reference:
- digits bit-exact; scales 0 ulp (f64 via exp2 identical); materialized q8_0
  byte-exact vs a numpy pack replica.
- Models checked: 1.5B k63 (linears + k99 embedding); tuned-7B k15 (linears +
  untied embed/head k99); GPTQ act-order k15 (F16 path, max|Δ| = 9.6e-5 ≤ fp16
  bound).

**Perplexity** (wikitext-2-raw test, full set, same corpus; 5060 Ti unless noted):

| model | variant | file GB | ppl |
|---|---|---|---|
| 1.5B base | f16 | 3.09 | 13.88 ± 0.110 |
| 1.5B base | K9 k63 + embed99 | 1.119 | 13.90 ± 0.110 |
| 1.5B base | q4_k_m | 1.117 | 14.70 ± 0.118 |
| 1.5B base | q8_0 | 1.895 | 14.29 ± 0.114 |
| 7B tuned | q8_0 | 8.10 | 9.231 ± 0.064 |
| 7B tuned | K9 k63 + embed99 | 5.47 | 9.253 ± 0.065 |
| 7B tuned | q4_k_m | 4.68 | 9.365 ± 0.066 |
| 7B tuned | K9 k15 + embed99 | 3.75 | 9.623 ± 0.068 |
| 14B tuned | q8_0 | 15.70 | 7.805 ± 0.052 |
| 14B tuned | K9 k63 + embed99 | 10.49 | 7.823 ± 0.052 (7900 XT) |

- K9-k63 vs q8_0: +0.021 ± 0.065 (7B) and +0.018 ± 0.052 (14B) — within noise
  (q8_0 parity) at 33–37% smaller files. At 1.5B: ≈ f16 quality at exactly
  q4_k_m's file size. 1.5B f16/q8_0/q4_k_m rows are cached HF quants of the
  base model; 7B/14B q8_0/q4_k_m are `llama-quantize` on our merged skeletons.
- **Framing caution (facts, for honesty):** decode-on-load trades load time for
  file size; runtime memory = materialized Q8_0/F16 size, i.e. this stage 1 is
  a *disk-format* win, not a resident-memory win. Resident container types
  (K9_4/6/7 + MMVQ kernels) are planned stage 2.

**Serving measured:** CPU 16.4 t/s (1.5B, 8 threads); CUDA 109.4 t/s (1.5B k63,
5060 Ti); HIP 35.6 t/s (14B k63, 7900 XT, gfx1100); llama-server OpenAI chat
completions served both cards. Builds: CPU, CUDA 13.4 (sm_120), HIP gfx1100.

**Measured caveats:**
- Decode-on-load is serial per tensor (rANS byte-renorm reads backwards; the
  whole blob must be contiguous): ~150 s total for tuned-7B k15
  (≈ 7e9 digits incl. table build; 210 M sym/s single-thread). Per-tensor
  threaded decode (exp23 pattern) is the obvious optimization.
- GPTQ-permuted tensors materialize as F16 (fatter files); production tuned
  artifacts are RTN and unaffected.
- `general.file_type` labels ignore K9 (cosmetic).
- llama.cpp runtime never sees compressed weights post-load → serving speed and
  memory equal Q8_0/F16 (llama-bench numbers would be Q8_0's).
- Python-side `k9.load_into` (chat_k9.py path) crashes on full-width perms
  (base9-repo bug, not the fork; C++ loader implements the correct semantics).

## 3. Upstream CONTRIBUTING.md requirements map (new quantization types)

Their "additional criteria, at minimum" for extending the `ggml_type` enum:

| requirement | status |
|---|---|
| small model converted to the type, GGUF **uploaded to HuggingFace** | ✗ not done (artifacts local in `results/`) |
| perplexity vs FP16/BF16 **and** similar-size types | ✓ covered by §2 table |
| **KL-divergence** data vs FP16/BF16 and similar-size types | ✓ for the 1.5B trio — exp27, §6 |
| llama-bench perf vs similar-size types, **pure CPU** | design-N/A: runtime ships as Q8_0/F16 → must be *stated* in the PR, plus the load-time cost |
| new feature **begins with an issue/discussion, not a PR** | n/a — process rule, changes the sequence |
| new-contributor limit: 1 open PR at a time | process rule |

Their module-pull-request conventions: squash-merge `<module> : <title> (#N)`;
CPU-first for new model/feature PRs (decode-on-load is backend-agnostic → no
backend-specific code shipped).

## 4. llama.cpp AI policy (verbatim, with locations)

`CONTRIBUTING.md` § "AI Usage Policy":
> AI-generated code is allowed. You are 100% responsible for every line, however
> it was produced.
> Undisclosed AI usage may result in your account being permanently banned.
> 1. Explicitly disclose the manner in which AI was employed. [...]
> 5. It is strictly prohibited to use AI to write your posts for you (bug
>    reports, feature requests, pull request descriptions, Github discussions,
>    responding to humans, ...).

`AGENTS.md` § "Prohibited AI Usage (results in immediate PR closure)":
> - AI-written PR descriptions, commit messages, or reviewer responses
> - Implementing features without understanding the codebase
> - Automated commits or PR submissions (may result in contributor ban)
> - Do NOT run `git push` or create a PR (`gh pr create`) on the user's behalf —
>   if asked, PAUSE and require the user to explicitly acknowledge that
>   automated PR submissions can result in a contributor ban from the project

When a commit is AI-assisted at the user's request, their convention is an
`Assisted-by: <assistant name>` trailer (explicitly NOT `Co-authored-by:`).

`.github/pull_request_template.md` requires (do-not-delete section):
- contributing-guidelines agreement checkbox;
- **"AI usage disclosure: YES / NO - if yes, describe how AI was used"**;
- template note addressed to AI agents: remind the user they are responsible
  for all submitted changes (this doc, §5, is that reminder).

## 5. Status & implications (facts → decisions)

- Current commit `14490cf2d` carried no `Assisted-by:` trailer; the code was
  co-developed with AI assistance in session 24. Upstream disclosure in the PR
  (template field above) is mandatory if this is ever submitted; the local
  commit could additionally be re-worded/amended with the trailer (rewrites the
  published branch — rose's call, not done).
- Per their rules: no AI ghost-writing of the description; the account holder
  clicks "Create PR" personally; discussion/issue-first for features.
- Per the session-24 plan: rANS disk layer stays fork-only research; the
  intended up-streamable artifact is the stage-2 resident container type
  (K9_4/K9_6/K9_7, superblock 256 = 4×64, fp16 m/H scales, CPU vec_dot +
  CUDA/HIP MMVQ, benchmark matrix vs q6_K/q4_K_M).
- Gaps blocking a standards-compliant new-type PR: HF-uploaded GGUF model,
  KL-divergence data. Both are ordinary machine-runnable work items (results
  artifacts, not posts) that can be produced with tooling as usual.
- `docs/04-k9-codec-spec.md` §5.2 remains stale vs `k9.py`'s record layout
  (known from session 24; a `k9.directory` note to docs/04 is a noted follow-up).

## 6. exp27 addendum (2026-10-06): KL-divergence data — checklist item filled

`experiments/exp27_llamacpp_kl.py`; llama.cpp's own two-step protocol
(`--kl-divergence-base` save on the reference, `--kl-divergence` compare on
candidates) with the fork's build-cpu `llama-perplexity`, defaults except
`-t 8`, corpus = `results/wiki2_raw_test.txt` (the exact session-24 ppl file).
Reference `ref_qwen_1.5b_f16.gguf` measured 13.849 ppl in the same framework
(CPU; session-24 GPU row: 13.88 — backend shift only). Raw per-pass logs and
the ~45 GB uint16 base-log-prob file live in `results/kld/` (gitignored,
regenerable by the script).

| candidate (1.5B base) | file GB | Mean KLD ± unc | PPL(Q) ± | PPL(Q)/PPL(base) | 99.9% KLD | max KLD |
|---|---|---|---|---|---|---|
| K9 k63 + embed99 | 1.119 | **0.006401 ± 0.000023** | 13.900 ± 0.110 | 1.00366 | 0.0895 | 0.379 |
| q8_0 (from same ref) | 1.647 | 0.000968 ± 0.000003 | 13.898 ± 0.110 | 1.00357 | 0.0131 | 0.061 |
| q4_k_m (from same ref) | 0.986 | 0.036364 ± 0.000137 | 14.134 ± 0.112 | 1.02059 | 0.562 | 3.483 |

- Reading: at 1.119 GB, the K9 logit distribution sits ~5.7× closer to the f16
  reference than same-reference q4_k_m's at 0.986 GB; q8_0 at 1.647 GB is the
  closest overall (0.00097). The K9 people row reproduces session 24 exactly
  (13.90) while being 33% smaller than the q8_0 file measured here.
- **Provenance flag — reconcile before quoting baseline numbers in the PR:**
  the two baselines regenerated by exp27 from the same f16 reference (`q8_0`
  1.647 GB / ppl 13.898; `q4_k_m` 0.986 GB / ppl 14.134) do NOT match the
  session-24 table's cached-HF baseline rows (q8_0 1.895 GB / 14.29; q4_k_m
  1.117 GB / 14.70). The K9 and f16 rows agree between both runs (13.90 ==
  13.90; 13.88 ≈ 13.849 CPU), so the discrepancy is in the baseline files'
  provenance, not the K9 artifacts — the cached files aren't on disk to
  identify further. exp27's set is self-consistent by construction (everything
  quantized/converted from one reference skeleton).