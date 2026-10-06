# 06 — HuggingFace upload prep (upstream checklist item 1)

llama.cpp CONTRIBUTING.md, "additional criteria at minimum" for adding a new
quantization type: *"convert a small model to GGUF using the new type and
upload it to HuggingFace."* The uploaded model is what reviewers reproduce the
ppl/KL rows from, so it ships together with (a) the fp16 reference and (b) the
similar-size comparison types.

## Model identity

- Base model: **Qwen/Qwen2.5-Coder-1.5B-Instruct** (bf16 native; the f16 GGUF
  reference stores those weights exactly — bf16's 8-bit mantissa ⊂ f16's
  10-bit for in-range exponents).
- New-type artifact: `results/qwen_1.5b_k63.gguf` — K9 k63 + embed99, 1.119 GB
  (GGUF v3 carrying `GGML_TYPE_K9 = 43` streams + `k9.directory` /
  `k9.materialize` KV).

## Timing: upload AFTER the ninths rename executes

The rename is decided but not executed (format "ninths", ext `.nth`, tag NTH /
sentinel 43 unchanged, magic NTHS, KV `k9.*` → `ninths.*`): fork loader →
exporter → **re-export all models** → repo rename last. Uploading the current
k9-named artifacts today would publish KV names and magic that are about to
change, so this staging stays files-and-checklist only. The ppl/KL *numbers*
are name-independent; re-running exp27 after the re-export only updates
provenance strings.

## Staging table

| source (results/) | role | size | note |
|---|---|---|---|
| `qwen_1.5b_k63.gguf` | new-type model (the checklist's "small model") | 1.119 GB | upload the post-rename re-export |
| `ref_qwen_1.5b_f16.gguf` | f16/bf16 reference for ppl/KL | 3.09 GB | rename-independent, upload anytime |
| `base1p5b_q8_0.gguf` | comparison type (exp27-generated) | see exp27 | llama-quantize from the reference |
| `base1p5b_q4_km.gguf` | similar-size comparison type (exp27-generated) | see exp27 | llama-quantize from the reference |

Destination repo id proposal (rose's call): `auRose94/<format>-qwen2.5-coder-1.5b-instruct`
under the post-rename format name; MIT; files flat in the repo root or a
subfolder — match llama.cpp PR convention of linking one model repo per type.

## Mechanics (for later)

- Tooling: `huggingface_hub` 1.33.0 inside `.venv-rocm`; auth token present at
  `~/.cache/huggingface/token` (+ `stored_tokens`). Verify with
  `.venv-rocm/bin/hf auth whoami`.
- Upload is a publishing action — execute manually or under rose's explicit
  instruction, after the re-export.

## Model card contents (facts for rose to write from — not card text)

- Base model + license of base (Qwen2.5-Coder-1.5B-Instruct — Apache-2.0 base
  weights; derived-artifact redistribution terms summarized on the card).
- One-line format description pointing to the format spec
  (`docs/04-k9-codec-spec.md`, post-rename naming).
- The measured comparison table (session-24 ppl + exp27 KL, same corpus file).
- Repro recipe: fork branch + `examples/k9probe` + `experiments/export_k9_gguf.py`,
  `llama-perplexity -f wiki2_raw_test.txt` defaults (`-t N`, CPU or GPU noted).
- AI-usage disclosure line listing the assistants used across the project
  (see the llama.cpp PR-template disclosure requirement; identical wording
  goes in the PR template's "AI usage disclosure" field).
- Sizes/links to the reference + comparison GGUFs.