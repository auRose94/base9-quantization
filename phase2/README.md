# Phase 2 — GDScript / Godot corpus and verification harness

Phase 1 (the quantization research) is closed: the K9 frontier at 1.5B and 7B is
established, and both remaining quality levers (asymmetric grids, error
compensation) came back negative. Phase 2 is the actual goal — an **agentic
coder model for video-game development** — and it needs two things Phase 1 never
did: a *corpus* and a *verifier*. This directory has the first pass at both.

It establishes what data exists, how much of it is actually valid under Godot 4,
and provides the machinery to check that mechanically instead of by eye. It now
also **trains and measures a first fine-tune** — see *Fine-tuning* below.

## Tools

| file | what it does |
|---|---|
| `gdscript_verify.py` | Headless Godot parse + runtime check. `--selftest` proves it on 7 known cases. |
| `build_corpus.py` | Parses the per-project dumps into `{project, version, path, code}`; `--stats`, `--emit`, `--reconstruct`. |
| `token_count.py` | Exact token counts (Qwen tokenizer) per Godot major + file-length percentiles. |
| `measure_corpus.py` | Godot-4 parse rate of the flat parquet (no project context). |
| `classify_corpus.py` | Splits those failures into version/syntax vs unresolved-context. |
| `measure_context.py` | Parse rate before/after reconstructing whole projects + `godot --import`. |
| `godot_scan.gd` | Loads every `.gd` in a project inside ONE Godot run and reports pass/fail each. |
| `build_sft.py` | Dumps → verified SFT corpus (recon → import → scan → classify → prompt). |
| `eval_gdscript.py` | The mechanical Godot-4 eval: 24 tasks, judged by the real parser. |
| `train_qlora.py` | 4-bit QLoRA SFT with prompt masking and a project-level holdout. |
| `respec.py` | Rewrite the SFT prompts from stored targets (no Godot rescan). |
| `merge_adapter.py` | Merge a LoRA adapter into bf16 weights → standalone model dir. |

### The verifier

Two layers, both driven by `/usr/bin/godot-mono` (**Godot 4.7.2 stable mono**):

- `check_parse` — `godot --headless --check-only --script res://x.gd`. This is the
  real GDScript parser, so it catches undeclared identifiers
  (`Identifier "foo" not declared in the current scope`), not just syntax.
- `check_runtime` — instantiates the script in a throwaway `SceneTree` and runs
  `--quit-after 3`, catching null derefs and bad method calls.

**Exit codes are not the signal.** Godot returns 0 even when a script raises
`SCRIPT ERROR` at runtime, and 1 only for parse failures — so the verdict is
derived from parsing the messages. Errors come back structured with `file`,
`line`, `func`, `kind`.

Cost: **~0.3 s per check**, single process. `--jsonl` batches, so the corpus pass
is embarrassingly parallel across processes.

`phase2/corpus.jsonl`-style records feed both uses: filtering training data, and
scoring a model's generated GDScript against the real compiler.

## The corpus

Source: `wallstoneai/godot-gdscript-dataset`. It ships two things, and the
difference matters enormously:

- **`*.parquet`** (1.5 MB, 100,199 rows) — the *line-flattened* text. One row per
  line, so file boundaries, project grouping and the declared Godot version are
  all gone.
- **`files/<project>.txt`** (5,172 files, 644 MB) — the *per-project dumps*. Each
  is a complete dump of one Godot project:

      ### Name: <project>
      ### Godot version: N
      ### Directory structure:
        <tree>
      ### Files:
      File name: <relpath>
      ```gdscript
      <content>
      ```

  That `File name:` grouping is what makes verification possible at all.

Counts from `build_corpus.py --stats` + `token_count.py` (complete download,
Qwen2.5-Coder tokenizer, `add_special_tokens=False`):

| Godot | projects | `.gd` files | chars | tokens | ≥256 tok files |
|---|---|---|---|---|---|
| 4 | 3,231 | 114,250 | 349,220,713 | **90,967,469** | 63,265 |
| 3 | 1,940 | 55,488 | 157,614,648 | 43,064,622 | 28,510 |
| 1 | 1 | 734 | 2,734,519 | 667,055 | 466 |

File length (Godot 4): p50 305 tok, p90 1,802, p99 7,217, max 569,447. Token
density is 0.260 tok/char, so the `chars/3.5` heuristic in `--stats` runs ~10%
high.

So the per-project dumps are worth **~100× more usable data than the parquet**
(349 MB / 91 M tokens vs 3.4 MB of Godot-4 GDScript), and they carry the version
label that makes the Godot-3 half separable rather than a source of silent
breakage.

The dumps contain only `.gd` and `.md` files — **no `.tscn`/`.tres`/`.import`** —
which sets a hard ceiling on what can ever resolve (see below).

## Measured: how much of it is valid under Godot 4?

**1. The flat parquet, no project context** (`measure_corpus.py` /
`classify_corpus.py`, n = 200 unique blocks):

| verdict | share |
|---|---|
| parses clean | 28.0% |
| **fails only on unresolved context** | **63.5%** |
| genuine Godot-3 syntax (`onready var`, `export(`, …) | 6.0% |
| other | 2.5% |

Only 6% is actually *old syntax*. The 63.5% looks broken only because the sample
has no siblings on disk — `extends Actor`, `preload("res://scenes/x.tscn")`.
**28% is therefore a lower bound, not the corpus's quality.**

**2. Whole projects reconstructed, with and without `godot --import`**
(`measure_context.py`, 10 projects / 952 `.gd` files):

| stage | parses clean | unresolved context | Godot-3 syntax | other |
|---|---|---|---|---|
| siblings on disk, no import pass | 20.9% | 76.7% | 0.1% | 2.3% |
| after `godot --import` | **37.6%** | 26.6% | 0.1% | 35.7% |

The import pass is required: `extends Foo` cannot resolve until the editor has
registered `class_name Foo` globals in
`.godot/global_script_class_cache.cfg`, even when `Foo.gd` is sitting right
there. It fixes most of the context failures (76.7% → 26.6%) and nearly doubles
the clean rate.

Per project the effect is large and unequal:

| project | `.gd` files | clean before | clean after import |
|---|---|---|---|
| RPGCreator | 235 | 19.1% | 46.4% |
| RDS-Game | 96 | 43.8% | 56.2% |
| BehaviourToolkit | 65 | 4.6% | 6.2% |

(The two populations in 1 and 2 aren't directly comparable — 200 random
self-contained blocks vs whole projects including vendored `addons/`. Whole
projects score *worse* per file because they drag in their dependencies.)

### Why the rest fails — and what's recoverable

| residual cause | recoverable? |
|---|---|
| `preload()`/`load()` of `.tscn`, `.tres`, plugin `*.tscn` the dump omits | **No** — the dumps carry only `.gd`/`.md`. 26.6% of files. |
| vendored `addons/` written for Godot 3 (`Too many arguments for get_as_text()`, `new()` arity) | Only by dropping the addon. One project alone had 107 such files. |
| native class shadowing (`Class "Logger" hides a native class`) | No — genuinely broken under 4.x. |
| real Godot-4 API misuse | No — these are exactly the samples we *want* filtered out. |

Practical consequence: **the corpus is usable, but it must be filtered by the
verifier, not trusted.** The clean-parse pass is the only honest way to size it,
and the ceiling for this source is roughly `ok + part of other` — call it **~40–55%
of Godot-4-labelled files**, before adding anything else.

## Gaps this leaves for a real Phase 2

1. **Not enough data for a robust coder.** 91 M tokens of *labelled* Godot-4
   GDScript, of which under half survives verification, is SFT smoke-test scale
   (63 k files are long enough to be real samples). The bigger sources are
   untapped: `godotengine/godot-demo-projects`, the Godot docs
   (`godotengine/godot-docs`, `.rst`), and GDScript from The Stack v2. The dump
   format above is the template for ingesting them.
2. **`.tscn`/`.tres` context is missing**, which is why a quarter of files can
   never resolve. Scraping whole projects (not text dumps) fixes this and is the
   single highest-value corpus upgrade.
3. **Synthetic, verified data is the differentiator.** The verifier closes a loop
   the corpus can't: generate a GDScript task → run it in Godot → keep only what
   passes. That produces *execution-verified* samples, which no scraped corpus
   has — and the same harness doubles as the eval.
4. **Training stack is not installed.** `.venv-baselines` has bitsandbytes
   0.50.2 but **not `peft`/`trl`**, which QLoRA needs.
5. **The agentic runtime is ready.** The `godot-mcp` server is configured in
   ZCode (`GODOT_PATH=/usr/bin/godot-mono`) and Godot 4.7.2 mono is installed and
   verified, so the MCP can drive real projects as the Phase 2 environment.

## Fine-tuning (Phase 2's actual objective)

### The corpus

`build_sft.py` turns the dumps into training data. Per project: reconstruct the
files, `godot --recovery-mode --import`, then one `godot_scan.gd` run that
validates every `.gd` in the project. Two things make that worth doing:

- **`load()` is not a validator.** It returns a non-null `GDScript` resource even
  for a script that failed to compile, printing only a `SCRIPT ERROR`.
  `reload()` returns `ERR_PARSE_ERROR` (43) instead — that is the signal.
- **The verdict is three-way, not two-way.** A file that fails *only* on
  unresolved externals — an autoload singleton (`UI`, `Enums`, `PS`), a sibling
  class, or a `.tscn` the dump omits — is still valid Godot 4 code; the failure is
  about our reconstruction, not the code. Only genuine syntax/API errors are
  dropped.

Measured over all Godot-4 projects (16 workers, 15 minutes):

| | count |
|---|---|
| projects processed | 3,231 |
| `.gd` files scanned | 108,469 |
| parsed outright (`clean`) | 73,970 |
| context-only failures (kept) | 24,491 |
| genuinely broken (dropped) | 10,008 |
| too small (<200 chars) | 9,852 |
| **kept after dedup** | **63,565 samples / 166 MB / ~43 M tokens** |
| duplicates removed | 24,869 (**28% of the volume**) |

Dedup by content hash matters more than it sounds: the same vendored addon ships
in dozens of project dumps, so without it `addons/dialogue_manager` alone would
appear hundreds of times. 54 projects (1.7%) hang the scan inside a `@tool` /
editor-plugin script under `--headless`; they are killed by process-group
timeout and skipped.

### The eval

`eval_gdscript.py` asks 24 GDScript tasks, decodes greedily, and judges each
reply three ways: does it compile under the real Godot 4 parser, does it contain
every required API name, and does it contain any Godot-3 name. `correct` is all
three. This is deliberately a *mechanical* bar — it catches `KinematicBody2D`,
`yield(`, the removed `export` keyword, and invented APIs like `GD.get_fps()`,
none of which a perplexity number sees.

Two metric bugs were found and fixed while using it, both of which had been
scoring correct Godot 4 as wrong:

- `"onready var"` matches inside **`@onready var`** → needs `(?<!@)`.
- `.connect("name", callable)` is **valid Godot 4** (connecting by signal name);
  only the Godot-3 three-argument form `.connect("name", self, "method")` is
  removed.

### Results (Qwen2.5-Coder-1.5B-Instruct, 4-bit QLoRA, r=16 on all projections)

| run | parses | api | no Godot 3 | **correct** | val loss |
|---|---|---|---|---|---|
| bf16 base | 12.5% | 50.0% | 70.8% | 12.5% | 1.1065 |
| K9 k63+embed99 | 16.7% | 45.8% | 75.0% | 12.5% | — |
| QLoRA, 1500 steps (24% of an epoch) | 54.2% | 62.5% | 79.2% | 37.5% | 0.7684 |
| QLoRA, 1500 steps, aligned prompt | 41.7% | 70.8% | 95.8% | 33.3% | 0.7663 |
| **QLoRA, 1 full epoch (3,904 steps)** | **54.2%** | **87.5%** | **100.0%** | **50.0%** | **0.7310** |

Held-out project val loss: **1.1065 → 0.7310** (−34%). Training took 30.6 minutes for
the full epoch and 12.3 for the quarter-epoch, all on one 16 GB card **while the
chat server kept running** (8 GB total VRAM).

Reading it honestly:

- **The base model is close to useless for Godot 4.** Only 3 of 24 replies
  compile, and every failure is genuine: `KinematicBody2D`, the Godot-3
  `connect("sig", self, "method")` signature, the removed `export`/`yield`
  keywords, invented APIs (`set_node()`, `Callable.new()`, `GD.get_fps()`), and
  once even `def` from Python.
- **Fine-tuning fixes the version confusion completely.** After one epoch, 24 of
  24 replies are free of Godot-3 API and `correct` is 4× the base. That is the
  headline.
- **Training duration was the lever, not prompt wording.** The two 1500-step runs
  differ by one task (9 vs 8 correct) — noise at n=24 — while the full epoch is
  clearly ahead. A quarter of an epoch had already learned the API names; the
  remaining epochs were needed to stop emitting them in wrong shapes.
- **The remaining 11 failures are now semantic, not historical**: declaring
  `var velocity` on a `CharacterBody2D` (which redefines a built-in member),
  calling `move_and_slide(Vector2.UP)` (Godot 4's takes no arguments),
  `get_frames_per_second()` without the `Engine.` prefix, an undeclared `level`,
  and 4 replies that are still fragments with top-level statements rather than a
  complete script.

So the ordering of remaining work is: (1) the fragment/complete-script shape,
(2) argument-count and built-in-member semantics, (3) more data. None of these is
a Godot-3 problem any more.

### Scaled: the 7B (same corpus, same recipe — see RESEARCH_LOG session 22)

`train_qlora.py` is model-agnostic; the full 7.6B Qwen2.5-Coder-7B-Instruct epoch
on the identical corpus took 94.3 min (~1.3 s/step) on a 16 GB CUDA card
(batch 1 × accum 16 — bigger batches OOM on the loss, not the weights: Qwen's
152k-vocab logits + float32 CE upcast need ~15 GB at batch 8 × 2048).

| 7B run | parses | api | no Godot 3 | **correct** | val loss |
|---|---|---|---|---|---|
| bf16 base | 45.8% | 58.3% | 70.8% | 25.0% | 0.9039 |
| **QLoRA 1 epoch (3,904 steps)** | **87.5%** | **91.7%** | **100.0%** | **83.3%** | **0.6604** |
| tuned, K9 k9+embed99 (see below) | 87.5% | 87.5% | 95.8% | 83.3% | — |
| tuned, K15+embed99 (eval-draw noise, see log) | 66.7% | 87.5% | 100.0% | 62.5% | — |

The 7B passes the 1.5B's final score before 20% of its epoch; the residual is
4 semantic failures (wrong constant, `get_process_fps` without `Engine.`,
api-miss, top-level fragment). Quantizing the merged tuned model
(`quantize_k9.py`) ships it as **3.10 GB** (k9+embed99, 3.261 b/param) with the
same 83.3% task score; paired ppl deltas vs tuned bf16 on the exp24 windows
match the base model's exp24 rows (+0.117 code at k15 / +0.380 at k9), so
fine-tuning changed neither the container sizes nor the quantization cost.

    ./.venv-baselines/bin/python train_qlora.py \
        --data out/godot4_sft.jsonl --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --out out/qlora_7b --epochs 1
    ./.venv-baselines/bin/python merge_adapter.py \
        --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter out/qlora_7b/adapter \
        --out out/merged_7b
    ./.venv-rocm/bin/python quantize_k9.py --model out/merged_7b \
        --out results/qwen7b_tuned_k15_embed99.k9 --k-body 15 --k-embed 99

(`.venv-rocm` is the second-GPU env: torch 2.14.0+rocm7.2 with the same pinned
deps — bnb NF4, training smoke, eval, and K9 load/generation all run on the
RX 7900 XT; the CUDA card trains while the RDNA3 card evaluates.)

### The 14B (overnight 2026-10-04 → 05 — see RESEARCH_LOG session 23)

Same recipe at lr 1e-4 needed a **chunked cross-entropy** (`chunked_loss()` in
`train_qlora.py`: vocab logits built per 256-position chunk inside a
checkpoint — HF's full-vocab fp32 CE spike OOM'd even batch 1 on 20 GB;
validated identical to HF's loss at batch 1 × seq 2048). Epoch: 234.8 min on
the 7900 XT. Containers (merged dir → `quantize_k9.py`, CPU): k9+embed99
5728 MB / k15+embed99 7004 MB / **k63+embed99 10480 MB at Δcode +0.005 — the
ppl-free corner holds at 14B** (bf16 hosts: code 2.947 / wiki 7.524).

The mechanical eval **regressed** (the ladder's first negative result): tuned
14B greedy = 45.8% correct (parses 58.3%) vs the tuned 7B's 83.3% — while API
precision is the best in the repo (draws-3 api 95.8%, no_godot3 100%) and val
loss is the *lowest* (0.6171). Signature: short fragment replies re-learning
the corpus's terse style at lr 1e-4. **Follow-up at lr 3e-5 (same night,
212.9 min) recovered it: greedy 70.8% / draws-3 79.2% — now above the 7B's
75.0% at the same draws-3 protocol.** Recipe rule: lr 1e-4 fits 1.5B and 7B;
a 14B-body LoRA wants ~3e-5 — val loss at 1e-4 reached 0.6171 (lowest ever)
while scoring worst, so **held-out corpus loss stopped predicting the
mechanical score at 14B**.

| run (grep tag in out/) | parses | api | no Godot 3 | correct | val loss |
|---|---|---|---|---|---|
| 14B base (NF4 host, draws-3) | 66.7% | 75.0% | 91.7% | 54.2% | 0.8510 |
| 14B tuned lr1e-4 (greedy) | 58.3% | 87.5% | 100% | 45.8% | 0.6171 |
| **14B tuned lr3e-5 (greedy)** | 79.2% | 91.7% | 100% | **70.8%** | 0.6309 |
| **14B tuned lr3e-5 (draws-3)** | **87.5%** | 91.7% | 100% | **79.2%** | 0.6309 |

Run it:

    python3 build_sft.py --workers 16 --out out/godot4_sft.jsonl
    python3 eval_gdscript.py --tag base                      # 12.5% correct
    ./.venv-baselines/bin/python train_qlora.py --data out/godot4_sft.jsonl --epochs 1
    python3 eval_gdscript.py --tag tuned --adapter out/qlora_1.5b/adapter
    ./.venv-baselines/bin/python merge_adapter.py --adapter out/qlora_1.5b/adapter \
        --out out/merged_1.5b

`train_qlora.py` needs the venv with `bitsandbytes`/`peft`
(`../.venv-baselines`); the corpus builder and evaluator only need torch +
transformers, though the evaluator needs `peft` to load an adapter.

## Running

    cd phase2
    python3 gdscript_verify.py --selftest                  # ~3 s, 7 cases, all must pass
    python3 build_corpus.py --stats                        # corpus counts by Godot version
    python3 token_count.py                                 # exact tokens (needs transformers)
    python3 build_corpus.py --emit godot4.jsonl --min-version 4 --min-block 200
    python3 measure_corpus.py 200                          # flat-parquet parse rate
    python3 measure_context.py 10                          # before/after import pass

`GODOT_PATH` overrides the binary; it must be an **editor** build (needs
`--check-only` and `--import`). Data lives under `phase2/data/` (the dumps are
644 MB and are not meant to be committed).
