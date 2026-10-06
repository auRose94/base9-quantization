# Research log

Newest first. Every new experiment or claim goes here; docs/01's predictions
were pre-registered (written before any experiment ran) so verdicts below are
honest.

---

## 2026-10-06, session 25 — C++ corpus extension, the GD+C++ 14B mix, and the ops/split lessons

**C++ extension of the Godot-4 coder phase.** A g++-based verifier
(`phase2/cpp_verify.py`, selftest 7/7) with phase 2's three-way verdict
carried over: ok / parse_error / **context** (unresolved #include kept-but-
unverified), plus an execution mode (real build + run + stdout capture;
tasks judged this way are execution-verified). Engine-source probe: 74% of
Godot's non-thirdparty .cpp parses in place once (a) the include path pins
the repo root + platform/linuxbsd, and (b) the ~180 build-time `.gen` headers
are materialized (a 4-min `scons platform=linuxbsd target=editor` run —
generation, not git history, is what produced them).

**Corpus: the-stack-v1 → full scan.** Content lives in v1 parquets; the-stack-
v2 is index+blobs needing its own grant (see the acceptance lessons in
phase2/README). 214-shard × sample-rate-0.08 scan → 83.5k candidates →
18-thread g++ fan-out → **75,368 unique verified files** (12.7% of candidates
pass `-fsyntax-only`; license-filtered quality windows keep only what a
2048-token window can teach). Result mixed with the GDScript corpus:
**63,565 verified GDScript + 45,000 C++ = 108,565 rows** (seed-0 shuffle,
GDScript-dominant on purpose).

**The 14B mix run — interim scoreboard.** Qwen2.5-Coder-14B NF4 QLoRA, lr
3e-5, 1 epoch = 6,668 optimizer steps. Mid-run adapter (~step 4200, NF4 host,
draws-3): **GDScript 83.3% (20/24 — parity with the tuned 7B, above the
gd-only 14B's 79.2% — the C++ share did not cost Godot skill)**, C++ **43.8-50%
(7-8/16; parses 87.5%, builds 87.5%, output 56.2%, no-bad-isms 100%)** —
+17-21 pts over the 1.5B smoke; profile = compiles-but-semantics-lag, the
expected opening curve.

**The memory mystery that forced a restart.** bitsandbytes 0.50.2 (newest on
this machine's index) ships **no fused NF4 kernels for sm_120 (5060 Ti) or
gfx1100 (7900 XT)**: every 4-bit linear takes `_dequant_linear_fallback`
(materializes the dequantized weight), and the first forward's live set lands
at **18.6-19.4 GiB**. The from-scratch config survived 4,100 batches on a
20 GB card at exactly that edge. The **resume path carried a stubborn ~+0.6
GiB surplus** no flag moved: 13 OOM launches on a completely clean card, each
at a different op (the fp32 embed upcast, the 136 MiB gate/up dequant, an SDPA
buffer). Decision: restart from step 0 — same corpus, same seed, same shuffle,
so the redone segment is a replay of the same trajectory; the step-4200
adapter is preserved (out/qlora_14b_mix/adapter_resume3600 + an earlier
backup) and its interim scores stand as results. train_qlora.py gains:
`--resume/--start-step` (adapter-continue; OneCycleLR replay spans
start-step // grad-accum steps because the schedule advances per *optimizer*
step), `--ce-chunk` (token-chunk of the checkpointed CE — loss value is
chunk-invariant, transients scale with it), `--vram-cap` (memory-fraction cap
for shared cards), `--shrink-frozen` (recast peft's blanket fp32 upcast of the
frozen embed/lm_head back to bf16 — measured ≈0 here because the spike is
dequant-hold-dominated, kept for 16 GB cards).

**Ops: the hardware split and crash-proof long jobs.** New machine layout:
the Ryzen 9 9900X's BIOS-disabled RDNA2 iGPU enabled (4 GiB carve) now
composites KDE, the 7900 XT is compute-only (its full 21.5 GB serves the run),
and the 5060 Ti hosts the chat server. Process lessons from the 18:49 cascade
(62 GB RAM + 39 GB swap → systemd-oomd killed the whole agent app-cgroup):
**setsid detaches the session, not the cgroup — long jobs must run under
`systemd-run --user` units with linger;** CPU merge/quantize passes (~30 GB
host each) must never stack with a model load; a 30-s stray-fence
(`mix_gpu_guard.sh`, cgroup-aware) blocks uncoordinated relaunches from other
sessions. The canonical pipeline is `phase2/mix_pipeline_unit.sh` (waits for
the compute-free-card gate, then train → evals → merge → K9 k9/15/63 → CPU
ppl, strictly serialized). Superseded and removed: `run_mix_e2e.sh`,
`post_train_ops.sh` (the two scripts that codified the disproven "harness
reaping" theory).

**Open items.** Fused 4-bit dequant kernels (a bitsandbytes source build or a
torchao backend) = the real memory+speed lever, scoped for a focused session;
profile the resume-path's +0.6 GiB surplus before the next mid-run resume.

---

## 2026-10-05, session 24 — K9 runs in llama.cpp: `.k9`-in-GGUF, decode-on-load, exact Q8_0 materialization, both GPUs

**What this is.** K9's "missing runtime piece" — a serving stack. Fork of upstream
llama.cpp at `/home/rose/Work/llama.cpp` (branch `k9`, single commit `14490cf2d`
over upstream `c25030496`), built for CPU / CUDA 13.4 (5060 Ti) / HIP gfx1100
(7900 XT). llama-cli and llama-server serve our `.k9` models through an
OpenAI-completions endpoint; perplexity harness runs on the same binaries.

**Design, stage 1 (disk layer).** GGUF v3 derives every tensor's byte length from
type×shape, so a variable-length rANS stream cannot be a plain tensor. Fix in the
fork: a loader-only sentinel type `GGML_TYPE_K9 = 43` (`blck_size=1, type_size=1`
→ `ggml_nbytes()` = ne[0] = blob length); per-tensor metadata rides in one KV key
`k9.directory` (name, rank+shape, k, group, scale_mode, perm_flag, segment sizes).
The exporter `experiments/export_k9_gguf.py` re-embeds `.k9` blobs **verbatim**
(no requantization) on top of a `convert_hf_to_gguf.py` f16 skeleton that supplies
norms/biases/tokenizer. `llama-model-loader` decodes on load into the destination
type; tensors that offload to GPU decode straight into VRAM, CPU tensors get one
writable host buffer (they must never alias the read-only mmap).

**The exactness theorem (why quality claims survive).** K9 dequant is
`w = m·(d−H)/H`, `H=(k−1)//2` — a *scaled integer*. Q8_0's per-32-block scale can
be set to `m/H` exactly and its int8 code to `d−H ∈ [−H, H] ⊂ [−49, 49]` (2.6×
headroom), so Q8_0 materialization of every k family is **lossless modulo the
fp16 rounding of the block scale** (≤ 4.9e-4 relative), which is finer-grained
than k99's own grid spacing. This also resolves the GGUF transpose trap
*for free*: GGUF stores all linears/embeddings transposed vs torch, and the
Q8_0 block structure (per out-row, chunks along in-dim) aligns exactly with K9's
per-out-row 64-group scales — only an index permutation of digits.

**Gates (all green).**
1. `examples/k9probe` + `experiments/k9_llamacpp_gate.py`, C++ decode vs `k9.py`:
   digits **bit-exact**, scales **0 ulp** (`exp2` f64 identical), materialized
   q8_0 **byte-exact** vs a numpy replica — on 1.5B k63 (linears + k99 embed),
   tuned 7B k15 (linears + untied embed/head k99), and GPTQ-act-order k15
   (perm tensors: F16 path, max|Δ| = 9.6e-5 ≤ fp16 bound).
2. llama-perplexity, wikitext-2-raw test (full set, same corpus; 5060 Ti
   otherwise noted):

| model | variant | file GB | ppl |
|---|---|---|---|
| 1.5B base | f16 | 3.09 | 13.88 ± 0.110 |
| 1.5B base | **K9 k63 + embed99** | **1.119** | **13.90 ± 0.110** |
| 1.5B base | q4_k_m | 1.117 | 14.70 ± 0.118 |
| 1.5B base | q8_0 | 1.895 | 14.29 ± 0.114 |
| 7B tuned | q8_0 | 8.10 | 9.231 ± 0.064 |
| 7B tuned | **K9 k63 + embed99** | **5.47** | **9.253 ± 0.065** |
| 7B tuned | q4_k_m | 4.68 | 9.365 ± 0.066 |
| 7B tuned | K9 k15 + embed99 | 3.75 | 9.623 ± 0.068 |
| 14B tuned | q8_0 | 15.70 | 7.805 ± 0.052 |
| 14B tuned | **K9 k63 + embed99** | **10.49** | **7.823 ± 0.052** (7900 XT) |

The exp20 1.5B claim reproduces inside llama.cpp end-to-end: **K9 k63
≈ f16 in ppl at exactly q4_k_m's size; q4_k_m pays +0.80**. At 7B/14B the
materialization-exactness prediction holds (K9-k63 vs q8_0: +0.021 ± 0.065 and
+0.018 ± 0.052 — within noise). Note the 1.5B q8_0/q4_k_m rows are the *base*
model (cached HF quants), 7B/14B rows use `llama-quantize` on our merged skeletons.

3. Serving: CPU 16.4 t/s (1.5B, 8 thr); 5060 Ti CUDA 109.4 t/s (1.5B k63);
   7900 XT HIP 35.6 t/s (14B k63); `llama-server` OpenAI chat completions work
   for both cards.

**Bugs found in our own code (pre-existing, worth fixing upstream in the repo):**
`k9.load_into` (chat_k9's path) **crashes on full-width-perm GPTQ records**
(`q[:, argsort(perm)]` on a 3-D grouped view; exp26's perms are full column
width, plen = 4·c — empirically confirmed on `qwen7b_gptq_k15_embed99.k9`);
`k9.decode_tensor` is correct and is the semantics the C++ loader implements.
The 14B overnight run (session 23's log) finished ~06:23 and the 7900 XT is idle.

**Caveats.** Decode-on-load is serial per tensor (byte renorm reads backwards from
stream end; whole blob must be contiguous): measured ~150 s total load for the
7B tuned k15 GGUF (≈ 7e9 digits incl. table build; 210 M sym/s single-thread).
Per-tensor-thread decode (exp23's pattern) is the obvious optimization. GPTQ
permuted tensors materialize as F16 (fatter; the production tuned artifacts are
RTN and unaffected). `general.file_type`/ftype labels ignore K9 (cosmetic).
Docs/04 §5.2 remains stale vs `k9.py`'s record layout (now also in the fork's
`src/k9.h`); add a `k9.directory` note to docs/04 next session.

**Next (stage 2 of the plan).** Resident digit-container types (K9_4/K9_6/K9_7,
superblock 256 = 4×64 groups, fp16 `m/H` scales → 4.1/6.1/7.1 b/param) with CPU
vec_dot + CUDA/HIP MMVQ kernels (the chat-batch fused dequant+GEMM), then the
tok/s benchmark matrix vs q6_K/q4_K_M on both cards; PR packaging behind that
(container type is the upstreamable artifact; the rANS KV layer stays fork-only
research). 14B at a K9_4 container ≈ 6.9 GiB would finally fit the 5060 Ti.


---

## 2026-10-05, session 23 — the 14B night run: K9 ppl-free at k63 again; the lr-1e-4 recipe regresses the mechanical eval; chunked-CE fix

**What ran** (phase2/overnight_14b.sh, done 02:31): Qwen2.5-Coder-14B-Instruct
(14.84 B; 48 layers, hidden 5120, GQA 40q/8kv, untied head, 152k vocab) 1-epoch
QLoRA on the 7900 XT (RDNA3/ROCm 7.2), batch 1 × accum 16, seq 2048, r=16 — the
7B recipe unchanged, after a chunked-CE fix (below). 3,904 steps in 234.8 min
(cold start ~13.8 s/step, ~3.6 s/step average).

**The chunked-CE fix (recipe-preserving, now in train_qlora.py).** The 14B
OOM'd at batch 1 on the 20 GB card **in backward**: HF's ForCausalLMLoss upcasts
the full [B,S,152064] logits to fp32, spiking ~2.5–3 GB past a base already at
~17 GiB. `chunked_loss()` builds logits per 256-position chunk inside
`torch.utils.checkpoint` (transient ~0.4 GB) — the same mean-over-masked CE;
validated **identical to HF's loss to 5 decimals** at batch 1 × seq 2048
(0.9293 vs 0.9293, gradients flow). This is the same wall that killed batch-8
7B training in the afternoon; the fix unblocks both.

**K9 containers at 14B** (merged tuned bf16 dir, CPU, ~4 min each) and CPU ppl
(one pass, paired windows, same process):

| tuned 14B | MB | b/param |
|---|---|---|
| bf16 host | 29518 | 16.0 |
| k9 + embed99 | 5728.0 (3.103) | +0.302 ± 0.037 code ppl |
| k15 + embed99 | 7004.4 (3.794) | +0.099 ± 0.019 |
| **k63 + embed99** | **10480.4 (5.677)** | **+0.005 ± 0.006 (wiki +0.039 ± 0.017)** |

bf16 refs: code 2.947 / wiki 7.524. Every grid's penalty **shrank again with
scale** (k15 7B +0.117 → 14B +0.099; k9 +0.380 → +0.302), and **k63 is
ppl-measurement-free at 1.5B, 7B and 14B** — the frontier's "essentially loses
nothing" corner is now replicated three times.

**The mechanical eval regressed at 14B — the ladder's first negative result.**
Draws-3 (NF4-hosted: a 14B bf16 host fits no local card): base 54.2% vs tuned
50.0% correct; greedy (the 1.5B/7B ladder protocol): tuned 14B = **45.8%**
(parses 58.3%) vs the tuned 7B's 83.3%. The failure profile is a **style**
regression, not knowledge: api 95.8% (draws) / 87.5% (greedy) and no_godot3
100% — more Godot-4 API precision than any other run in the repo — but replies
got TERSE (median 174 chars vs base 314) and 5 of 13 failures are
top-level-statement fragments ("Unexpected 'match'/'for'/'print' in class
body") — the 1.5B-era shape bug the 7B had largely repaired. Prime suspect:
**lr 1e-4 is too hot for a 14.7 B-body LoRA over 62k short-corpus samples in
one epoch** — val loss 0.6171 is the *lowest* of the three scales
(1.5B 0.7310 / 7B 0.6604 / 14B 0.6171) while `correct` goes 50.0 / 83.3 / 45.8:
**held-out corpus loss stopped predicting the mechanical score** once the
model's base knowledge was high and its output style dominated what fails.

**Night-cap result (done 06:23): 14B @ lr 3e-5 — the fix holds.** Same
everything (1 epoch, r=16, seq 2048, 212.9 min): val 0.8510 → 0.6309; **greedy
(ladder protocol) 70.8% correct (parses 79.2%, api 91.7%, no_godot3 100%)** —
+25 points over the lr-1e-4 run's 45.8% — and **draws-3 79.2% (parses 87.5%)**
vs the lr-1e-4 draws-3 50.0%. Same-protocol ladder (draws-3): 1.5B 50.0
(bf16) / 7B 75.0 (bf16) / 14B **79.2 (NF4)** — the 14B tops the ladder, and the
LR rule is now: **1.5B and 7B trained well at lr 1e-4; 14B wants ~3e-5** (the
1e-4 run's low val loss was over-styling, exactly as diagnosed). Open for next:
is there an intermediate-LR optimum (5e-5) at 14B, and does an lr-scaled run
lift 7B past 83.3? Both are single-evening runs.

**Ops lessons (don't re-discover):** (a) a `&`-child started inside a finishing
Bash-call task gets reaped — long-lived servers must be their own
background task; (b) **chat_k9's preset variant paths were cwd-relative** —
launching from phase2/ silently dropped every K9 variant (a "1-variant" bf16
chat that looks like a success; two chat deaths traced to it) — fixed by
anchoring preset paths to the file's directory in main(); (c) the harness task
shell surviving/killing orphans behaved inconsistently across launches — verify
by PID + `/state`, not by task-exit codes.

**Artifacts:** `out/qlora_14b{,_lr3e5}/`, `out/merged_14b/` (29.5 GB),
`results/qwen14b_tuned_{k9,k15,k63}_embed99.k9`, `out/ppl_14b.log`,
`out/eval_*14b*/{summary.json,jsonl}`, `phase2/{overnight_14b,overnight_14b_lr3e5}.sh`,
`phase2/ppl_check_k9.py` (now `--model/--cpu/--prefix`).

---

## 2026-10-04, session 22 — Phase 2 at 7B: the verified-corpus QLoRA scales — 83.3% correct in 94 min, shipped as a 3.1 GB K9 file

**What was asked.** "Can we make a larger model with this technology?" — scale the
Phase-2 fine-tune from the 1.5B (50.0% correct, see `phase2/README.md`) to
Qwen2.5-Coder-7B-Instruct, on the same 63,565-sample verified corpus, the same
24-task mechanical eval, r=16 LoRA on every projection, 1 epoch, lr 1e-4 OneCycle —
the full recipe unchanged.

**A second GPU entered service this session.** Rosemary asked whether the RX 7900 XT
(20 GB, RDNA3, ROCm 7.2.4 stack already installed) helps and whether both cards
can be used. Answers established by measurement (`.venv-rocm`: torch 2.14.0+rocm7.2
+ bitsandbytes 0.50.2 + peft — all pinned to the versions `.venv-baselines` uses):

- **bitsandbytes NF4 works on gfx1100.** The 3-step `train_qlora.py` smoke ran
  end-to-end on the 7900 XT (losses, val eval, adapter written). Eval and K9-load
  generation run fine on ROCm too (greedy, same eval flow).
- **But the 7B train OOM'd there — on the loss, not the weights.** Qwen2.5-Coder-7B's
  untied 152,064-way head at batch 8 × 2048 creates a ~5 GB bf16 logits tensor and
  the CE path upcasts it (~10 GB in float32): `CrossEntropyLoss` tried to allocate
  7.99 GiB with 15.56 GiB already held. That is an HF-API wall independent of the
  backend (a chunked-CE forward would fix it; not attempted this session — keeping
  the recipe identical was worth more than the batch gain).
- **So the roles split, and both cards ran the whole session without conflict:**
  the 5060 Ti (CUDA) trained (batch 1 × accum 16 — the exact 1.5B recipe shape);
  the 7900 XT evaluated checkpoints and ran the K9 work; the CPU did the merge and
  the K9 re-quantization (7.6 B weights quantized + entropy-rANS-encoded in
  **~100 s per config on 24 cores**). LM Studio's idle loaded model was unloaded
  to free both cards (restore = one click in the app).

**Training result (94.3 min, 1.24–1.35 s/step):**

| 7B run | parses | api | no Godot 3 | **correct** | val loss |
|---|---|---|---|---|---|
| bf16 base | 45.8% | 58.3% | 70.8% | 25.0% | 0.9039 |
| QLoRA step 400 | 54.2% | 75.0% | 95.8% | 45.8% | — |
| QLoRA step 800 | 58.3% | 91.7% | 100.0% | 58.3% | — |
| QLoRA step 1200 | 75.0% | 91.7% | 100.0% | 75.0% | — |
| QLoRA step 2000 | 62.5% | 87.5% | 100.0% | 58.3% | — |
| QLoRA step 2400 | 83.3% | 83.3% | 100.0% | 70.8% | — |
| **QLoRA 1 epoch (3,904 steps)** | **87.5%** | **91.7%** | **100.0%** | **83.3%** | **0.6604** |

The 7B passes the 1.5B's final score (50.0%) **before 20% of its own epoch**
(step 800: 58.3%) and tops out at **+33 points over the same-corpus 1.5B tune**.
Held-out val loss 0.9039 → 0.6604 (−27%; the 1.5B went 1.1065 → 0.7310). The
pre-fine-tune base is already twice as capable (25.0% vs 12.5%), so part of the
gain is the better base — but the *training* still converted it into mechanically
correct Godot 4 code (24/24 replies now free of Godot-3 API; the 1.5B failed
12/24 at its end, the 7B fails 4/24).

**The remaining failures are the same residual family, smaller:** 
`PhysicsServer2D.RAYCAST_MODE_CLOSEST` (wrong constant), `get_process_fps()`
without `Engine.`, one api-miss on `class_name`, and one top-level-statement
fragment (`dictionary_iter`). Version confusion is gone; what is left is argument
and member semantics plus script completeness — the same ordering the 1.5B
analysis gave (semantics, then more data), now at an 83% mechanical pass.

**K9 on the tuned model — quantization cost is unchanged by fine-tuning.**
`phase2/quantize_k9.py` (exp24's verified writer, pointed at the merged dir,
CPU-only) rebuilt the containers from the tuned bf16 weights: k9+embed99 =
**3103.9 MB (3.261 b/param)** and k15+embed99 = 3740.1 MB — within 0.2% of the
base-model files (LoRA deltas are tiny and barely shift digit entropy).
Perplexity on the exp24 windows, loaded back with `k9.load_into` and measured
in the same process (`phase2/ppl_check_k9.py`):

| tuned 7B | code ppl | Δcode (paired) | Δwiki (paired) |
|---|---|---|---|
| bf16 | 3.410 | — | — |
| k15+embed99 | 3.527 | +0.117 ± 0.020 | +0.468 ± 0.102 |
| k9+embed99 | 3.790 | +0.380 ± 0.055 | +1.436 ± 0.266 |

These reproduce exp24's base-model deltas (+0.117 / +0.374) almost digit for
digit — fine-tuning neither amplified nor reduced the quantization penalty.
On the 24-task eval the k9-quantized tuned model scored **the same 83.3% as
bf16** (same 20/24 tasks), while k15 drew 62.5% — the per-task flips scatter in
both directions and the metric's SE at n=24 is ±9 points, so the reliable
comparator is the ppl table above, not single-eval draws.

**Artifacts:** `phase2/out/qlora_7b/` (adapter + train logs), `out/merged_7b/`
(bf16, 15.2 GB), `results/qwen7b_tuned_{k9,k15}_embed99.k9`,
`phase2/{quantize_k9,ppl_check_k9}.py`, eval jsonl/summaries per checkpoint
(`out/eval_tuned7b_*`), and `.venv-rocm/` as the ROCm training/eval env.
**Next:** the two open levers at this quality level are (a) semantic-error data
(the verifier can mint argument-correct samples) and (b) serving — the tuned
model already runs through `chat_k9.py` at 3.1 GB; a fused dequant+GEMM kernel
remains the missing runtime piece for speed. Phase 3 (full QAT / >7B) still
wants the cloud card.

---

**Hypothesis being tested.** Session 20 explained the 7B GPTQ null as "the penalty
is already small at k=15". The load-bearing counter-case: at 1.5B the *largest*
GPTQ gain was on the coarsest grid (k=9: 5.599 → 5.150, −0.449), and k=9 is where
the error is largest, so k=9 is the one setting where compensation might still pay
at 7B. Same script, `K9_CONFIGS=9,99`.

**Result: also a null.**

| 7B k=9 config | MB | code ppl | Δcode vs bf16 (paired) |
|---|---|---|---|
| bf16 reference | — | 3.493 | — |
| K9 k=9 RTN (exp24) | 3103.7 | 3.867 | +0.374 ± 0.043 |
| **K9 k=9 + GPTQ (exp26)** | **3129.5** | **3.826** | **+0.333 ± 0.034** |
| GGUF q2_k | 3015.9 | 4.054 | +0.561 |
| GGUF q4_k_m | 4683.1 | 3.563 | +0.070 |

The shift is **−0.041 ppl at +0.8% bytes**; the two runs' deltas differ by 1.2× the
combined SE (√(0.043² + 0.034²) = 0.055) and share the same windows and the same
re-measured bf16 reference (3.493 in both), so this is inside noise — directionally
favourable, statistically nothing.

**The scale story, now with two points per grid.** GPTQ's recovered share of the
RTN gap falls with model size:

| grid | 1.5B gap → recovered | 7B gap → recovered |
|---|---|---|
| k=15 | +0.364 → −0.215 (59%) | +0.117 → −0.002 (2%) |
| k=9 | +1.107 → −0.449 (41%) | +0.374 → −0.041 (11%) |

So exp6's "compensation pays on coarse grids" is really a **small-model** result:
compensation recovers a *fraction* of the RTN penalty, and both the penalty and
that fraction shrink as the model grows.

**Where this leaves K9 at 7B.** Not a domination result, but a consistent
byte-efficiency one: k=9+GPTQ at 3129.5 MB sits ~0.195 ppl below the GGUF
interpolation of that byte budget (+0.528), and beats q2_k's quality by
0.228 ppl at +3.8% bytes. Note the 1.5B "dominates q2_k on both axes" result
(exp19: 641.9 MB vs 752.9 MB) **does not survive to 7B on size** — here K9 is
the larger file with the better quality. k=15 RTN at 3740.2 MB sits ~0.231 ppl
below the same interpolation (+0.348; the k=15+GPTQ point at 3760.2 MB is 0.227
below +0.342). Both axes of the quantization research are now closed at
7B: **the frontier position is established, and neither of the two remaining
quality levers (asymmetric grids, error compensation) transfers from 1.5B.**

wall 1255s. Result files: `results/exp26_qwen7b_gptq_k9.{md,csv}`
(`K9_CONFIGS` env var; the k=15 record is preserved as `..._k15.{md,csv}`).

---

## 2026-10-03, session 20 — 7B + GPTQ (exp26): the lever does not transfer — a null

**Hypothesis being tested.** At 1.5B, GPTQ with act-order bought −8% code ppl on
the coarse grid, so applying it at 7B should move K9 k=15 from +0.117 to ≤+0.070
and beat GGUF q4_k_m on both axes.

**Result: it did not.**

| 7B config | MB | code ppl | Δcode vs bf16 (paired) |
|---|---|---|---|
| bf16 reference | — | 3.493 | — |
| K9 k=15 RTN (exp24) | 3740.2 | 3.610 | +0.117 ± 0.015 |
| **K9 k=15 + GPTQ (exp26)** | **3760.2** | **3.608** | **+0.115 ± 0.020** |
| GGUF q4_k_m | 4683.1 | 3.563 | +0.070 ± 0.013 |

GPTQ changed the result by **−0.002 ppl (well inside the ±0.020 noise) at +0.5%
bytes** (3760.2 vs 3740.2 MB — slightly higher digit entropy plus the stored
act-order permutations).

**Scope of the run.** GPTQ covered every body matrix with in ≤ 4096 (q/k/v/o,
gate/up = 4.62 B params, **71% of the body**), act-order, damp 0.05, 64×1024
in-domain calibration, Hessians per transformer block with upstream layers
already quantized (the sequential method). `down_proj` (in = 18944) stayed RTN
because its Hessian is 1.4 GB fp32 and does not fit beside a 15.3 GB model.
`gptq_apply` was unit-tested bit-identical to exp18's implementation.

**Why it plausibly does not transfer.** The penalty GPTQ is supposed to recover
is much smaller at 7B to begin with: at 1.5B the k=15 body cost +0.364 code ppl
on a 4.492 baseline (+8.1% relative); at 7B it costs +0.117 on a 3.493 baseline
(**+3.3% relative**). Compensation has less error to fix, so its absolute gain
shrinks toward the noise floor. This is consistent with exp6's rule of thumb
("compensation pays on coarse grids") read across scales rather than across grids.

**What survives.** K9 k=15 is still **below the deployed frontier at matched
bytes**: interpolating q2_k→q4_k_m gives +0.342 at 3760 MB, against K9's +0.115 —
a 0.227 ppl advantage. What does not survive is the *domination* of q4_k_m: K9 is
20% smaller but 0.045 ppl worse. The 7B headline is a size-vs-quality trade, and
this experiment closes the obvious route to converting it.

**Remaining cheap probes** (not run): GPTQ at 7B **k=9** — at 1.5B the largest
GPTQ gain was on the coarse grid (−0.449), and k=9 is where error is largest, so
this is the one place compensation might still pay at 7B; and a full-body variant
including down_proj on a larger card.

**Deliverables.** `experiments/exp26_qwen7b_gptq.py` (with `gptq_apply`,
`collect_hessians`), `results/exp26_qwen7b_gptq.{csv,md}`,
`results/qwen7b_gptq_k15_embed99.k9` (3760 MB).

---

## 2026-10-03, session 19 — 7B: K9 vs the deployed frontier (exp25)

GGUF baselines dequantized tensor-by-tensor into the bf16 model on the **same
windows** as exp24, with **paired** deltas (bf16 reference evaluated in the same
process: code 3.493, wiki 9.914).

| format | MB | b/param | code ppl | Δcode vs bf16 (paired) |
|---|---|---|---|---|
| GGUF q2_k | 3015.9 | 3.17 | 4.054 | +0.561 ± 0.076 |
| **K9 k=9 + embed99** | **3103.7** | 3.260 | 3.867 | **+0.374 ± 0.043** |
| **K9 k=15 + embed99** | **3740.2** | 3.929 | 3.610 | **+0.117 ± 0.015** |
| GGUF q4_k_m | 4683.1 | 4.92 | 3.563 | +0.070 ± 0.013 |

(other published GGUF sizes for reference: q3_k_m 3808.4 · q4_0 4431.4 ·
q5_k_m 5444.8 · q6_k 6254.2 · q8_0 8098.5 MB)

**Findings.**

1. **K9 sits well below the deployed frontier at its operating points.** Linear
   interpolation between q2_k and q4_k_m gives +0.348 at 3740 MB and +0.535 at
   3104 MB; K9 measures **+0.117** and **+0.374** — i.e. **0.23 / 0.16 ppl better
   than the GGUF frontier at matched bytes**, at 7B as at 1.5B.
2. **But it no longer strictly dominates q4_k_m.** At 1.5B the GPTQ-enabled
   k=63 point beat q4_k_m on both axes; here K9's best point (k=15, 3740 MB,
   +0.117) is 20% smaller than q4_k_m (4683 MB) but 0.047 ppl worse. K9 wins on
   bytes, loses on quality — a trade, not domination.
3. **This run is RTN.** exp18 showed GPTQ with act-order buys **−0.45 code ppl
   (8%)** on the 9-level grid at 1.5B. Applying it to the 7B body is the obvious
   next step and would likely push k=15 from +0.117 toward parity with bf16 —
   which *would* beat q4_k_m on both axes.
4. **The accounting is exact at 7B**: estimated and real file sizes agree to the
   tenth of a MB for both configs, and the GGUF loader reported 339/339 tensors
   loaded with none skipped.

**Caveats.** Baseline is bf16, not fp32 (7B fp32 is 30 GB and does not fit the
16 GB card), so these are marginal cost-of-K9-on-top-of-bf16 numbers. 8×1024
tokens per corpus; RTN only; GGUF quants are dequantized into the HF model, not
run through the llama.cpp kernels.

**Deliverables.** `experiments/exp25_gguf7b.py`,
`results/exp25_gguf7b.{csv,md}`.

---

## 2026-10-03, session 18 — K9 at 7B (exp24): the advantage holds with scale

**Setup.** `Qwen/Qwen2.5-Coder-7B-Instruct`, 7,615,616,512 params — body 196
Linear (6,525,288,448), embed + **untied** lm_head (1,089,994,752), 28 layers,
GQA 28q/4kv. Real K9 files written by `k9.py`, 8×1024-token windows per corpus.

**Baseline is bf16, not fp32** — 7B fp32 (30 GB) does not fit the 16 GB card, so
these are the *marginal* cost of K9 on top of bf16 (bf16 reference: code 3.493,
wiki 9.914).

| config | MB | b/param | code ppl | Δcode vs bf16 |
|---|---|---|---|---|
| **k=15 body + k=99 embed/lm_head** | **3740.2** | 3.929 | 3.610 | **+0.117 ± 0.015** |
| **k=9 body + k=99 embed/lm_head** | **3103.7** | 3.260 | 3.867 | +0.374 ± 0.043 |

**Findings.**

1. **The entropy accounting is exact at 7B too** — estimated and real file sizes
   agree to the tenth of a MB on both configs (3740.2 vs 3740.2; 3103.7 vs
   3103.7). Three model scales, same accounting.
2. **K9's 7B points sit below the deployed GGUF sizes.** Published GGUF sizes
   for this model: q2_k 3015.9 MB · q3_k_m 3808.4 · q4_0 4431.4 · q4_k_m 4683.1 ·
   q5_k_m 5444.8 · q6_k 6254.2 · q8_0 8098.5. So K9 k=15 (3740 MB) is **20% smaller
   than q4_k_m** and just under q3_k_m's size, at +0.117 code ppl; k=9 (3104 MB)
   is essentially q2_k's size at +0.374.
3. **The quantisation penalty is not smaller at 7B** in this measurement
   (+0.117 at k=15 vs +0.145 for the same grid at 1.5B) — but the baselines
   differ (bf16 vs fp32), and 7B *does* start from a much lower baseline (code
   3.493 vs 4.492), so in relative terms the penalty is larger. Whether the
   absolute K9-vs-GGUF *advantage* grows with scale needs exp25's measured GGUF
   quality (below).

**Engineering (memory).** 7B bf16 is 15.3 GB of a 15.57 GiB card, ~0.6 GB spare.
Two changes made it fit: the quantiser computes in row chunks and writes back
**in place** (the fp32 working set of a 7B tensor is 272 MB, and the embed's
digits alone would be 545 MB on the device), and restoring the bf16 reference
copies in 1024-row slices rather than allocating a full-size temporary. Both are
now standard in the 7B path.

**Deliverables.** `experiments/exp24_qwen7b.py`,
`results/exp24_qwen7b.{csv,md}`, `results/qwen7b_k15_embed99.k9` (3740 MB),
`results/qwen7b_k9_embed99.k9` (3104 MB).

---

## 2026-10-03, session 17 — K9 perf: parallel encode (2.2×) + streaming decode (no OOM)

Two implementation wins, **no format change** — the bytes are identical.

| encode threads | s | M sym/s |
|---|---|---|
| 1 | 8.3 | 185 |
| 4 | 4.0 | 388 |
| 24 | **3.8** | **405** |

- **Parallel encode** (`write_k9(..., threads=N)`): tensors are independent, so
  encoding is embarrassingly parallel and the ctypes call into the C coder
  releases the GIL. 2.2× at 24 threads; output **byte-identical** (same size and
  the same sha256 across all thread counts). Scaling saturates because the
  per-tensor table build (`normalize_freqs`/`_tables`, numpy, GIL-held) and the
  single-threaded file write become the bottleneck.
- **Streaming decode** (`k9.K9File` + `k9.load_into`): the directory is parsed
  once and one tensor's blob is read on demand; `load_into` dequantizes in **row
  chunks** (4096 rows), so the loader's own device allocation is ~25 MB instead
  of the whole tensor. This removes the transient CUDA OOM that exp21's
  full-fp32 path hit on the 151936×1536 embed. Measured: 22.5 s end-to-end
  (68.6 M sym/s) with **0 digit mismatches**, GPU high-water 9.14 GB (dominated
  by the resident 6.2 GB fp32 model), perplexity unchanged (4.4995).

**What is left, and whether it is worth it.** The remaining ~2-4× would come
from **lane-interleaved rANS** — N independent states advancing in lockstep so
the *inside* of a single tensor's stream becomes SIMD/GPU-friendly. That matters
for very large models (7B+ single tensors) or streaming inference, not for the
1.5B workflow, which now encodes in 3.8 s and decodes in 22 s.

**Deliverables.** `k9.py` (`write_k9(threads=)`, `K9File`, `load_into`),
`experiments/exp23_k9_perf.py`, `results/exp23_k9_perf.{csv,md}`.

---

## 2026-10-03, session 16 — asymmetric grids: a clean negative (exp22)

**Question.** Does an affine per-group grid `w = lo + d·step` (GGUF Q4_K style,
min+scale, 2 scales/group) beat K9's symmetric odd grid `w = m(d−H)/H`
(1 scale/group) once the extra scale cost is paid?

| variant | k | digits b/p | scales b/p | MB | code ppl |
|---|---|---|---|---|---|
| sym (K9) | 9 | 3.208 | 0.096 | 637.7 | 5.598 |
| symclip | 9 | 3.213 | 0.097 | 638.7 | 5.599 |
| **asym** | 9 | 3.391 | 0.347 | **721.3** | **5.336** |
| sym | 15 | 3.872 | 0.096 | 765.8 | 4.853 |
| symclip | 15 | 3.877 | 0.097 | 766.8 | 4.855 |
| **asym** | 15 | 4.051 | 0.347 | **848.7** | **4.683** |
| sym | 27 | 4.617 | 0.096 | 909.5 | 4.571 |
| symclip | 27 | 4.622 | 0.097 | 910.5 | 4.563 |
| asym | 27 | 4.789 | 0.347 | 991.0 | 4.568 |

(fp32 reference: code 4.492. Same 8-window harness as exp17–21.)

**Findings.**

1. **The asymmetric grid does not pay for itself.** It improves quality at
   fixed k (k=9: 5.598 → 5.336; k=15: 4.853 → 4.683), but at **matched bytes**
   it is at best parity and usually worse:
   - k=9: asym costs 721.3 MB; the symmetric family interpolated between k=9
     and k=15 gives **5.112** at that size — so asym is **+0.22 ppl worse**
     than the symmetric grid at equal bytes.
   - k=15: asym 4.683 at 848.7 MB vs symmetric interpolation 4.690 — exact parity.
   - k=27: 4.568 vs 4.571 — no gain.
   The cost comes from both sides: two scales/group (0.347 vs 0.096 b/param) *and*
   higher digit entropy (the wider affine range uses more of the alphabet).
2. **Tail clipping is a non-issue.** `symclip` (range at the 99.95% quantile of
   |w| instead of absmax) changes nothing (5.599 vs 5.598 at k=9). At g=64 the
   absmax rule is not outlier-limited on this model — so the K9 grid has no
   clipping headroom to reclaim.
3. **Likely mechanism (hypothesis).** The affine grid generally does *not* place
   a level exactly at 0, while the odd symmetric grid always does — and
   transformer weights carry a lot of mass near zero. That single level, plus
   the smaller scale side-info, is why the symmetric family stays ahead
   per byte.

**Consequence.** This closes the "asymmetric grid" enhancement: K9's symmetric
odd grid is the right choice, and the existing exp18–21 numbers stand. A
refinement that could still be tried is a *zero-anchored* affine grid (levels
forced to include 0, different spacing on each side) and/or double-quantized
super-block scales to cut the 2-scale cost — but the measured ceiling here is
~parity, so it is low priority.

**Deliverables.** `experiments/exp22_asym_grid.py`,
`results/exp22_asym_grid.{csv,md}`.

**Remaining roadmap.** Lane-interleaved/streaming coder (optional perf); a 7B
run; Phase 2 (QLoRA on GDScript + Godot MCP).

---

## 2026-10-03, session 15 — the coder is now 47×/27× faster, byte-identical

**Diagnosis (the user's question: language or implementation?).** Both, but
predominantly **language/implementation**. Measured baseline on a realistic
k=63 stream (10 M symbols, entropy 4.430 b/sym): Python **5.95 M sym/s encode,
7.72 M sym/s decode**. The cost is the CPython interpreter in the inner loop
(~10-30 bytecodes plus two arbitrary-precision div/mods *per symbol*) — the same
algorithm in C is ~2-5 ns/symbol. There is also a genuine *algorithmic*
constraint: a single rANS stream is a strict serial dependency (each symbol's
state update needs the previous state), so it cannot be vectorized within a
stream. But that constraint is **not** what makes it slow today — a scalar C
implementation of the same stream already runs at 200-280 M sym/s. Lane
interleaving (N independent states advancing in lockstep) is the lever that
would enable SIMD/GPU on top of that; it is optional, not required.

**Fix.** `experiments/rans_fast.c` — the identical byte-renorm rANS
(32-bit state, renorm while `x >= f·2^(32-k)`, 4-byte big-endian end state) —
compiled with `gcc -O3 -shared -fPIC` and called through ctypes from
`rans_fast.py`. `k9.py` uses it when available and falls back to pure Python
otherwise. The shared object is named `k9rans.so`, not `rans_fast.so`, because a
`.so` matching the module name shadows the `.py` on the next import (a real
trap hit during development).

| | Python | C | speedup |
|---|---|---|---|
| encode (10 M sym, k=63) | 5.95 M/s | **278.4 M/s** | 46.8× |
| decode | 7.72 M/s | **210.4 M/s** | 27.2× |

**Correctness.** The C output is **byte-identical** to `rans.py` for
k = 9/15/27/63/99 (same algorithm, same rounding), and the full-model round trip
is unchanged: 1112.0 MB, entropy-estimate delta +0.00%, **0 digit mismatches**,
perplexity 4.4995 → 4.4995.

**End-to-end effect (exp21).**

| step | pure Python | C coder |
|---|---|---|
| encode whole 1.54 B-weight model | 273 s (5.66 M/s) | **8 s (185 M/s)** |
| decode + rebuild + load into model | 413 s (3.74 M/s) | **25 s (61 M/s)** |
| whole experiment wall | 719 s | **~180 s** |

(The end-to-end decode figure is lower than the 210 M/s microbenchmark because
it includes per-tensor table building, dequant reconstruction into fp32, the
bit-exactness comparison, and GPU copies.)

**What remains.** Lane-interleaved rANS (N states) for SIMD/GPU — worth roughly
another 2-4× and needed only for very large models or for streaming inference;
and a streaming decoder that never materializes the whole fp32 model.

**Deliverables.** `experiments/rans_fast.c`, `experiments/rans_fast.py`,
`experiments/k9rans.so` (built artifact), updated `k9.py`, refreshed
`results/exp21_k9_codec.{csv,md}`.

---

## 2026-10-03, session 14 — the K9 codec exists and round-trips a 1.5B model

**RQ5 closed.** `experiments/k9.py` implements the container from
`docs/04` (magic `K9Q1`): per-tensor rANS digit streams (M=4096) with serialized
frequency tables, per-(row,group) scales in fp32/fp16/ent8 modes, and an
optional uint32 act-order permutation. `exp21_k9_codec.py` writes a real file
for Qwen2.5-Coder-1.5B-Instruct (body k=63 + embedding k=99, g=64, ent8 scales)
and decodes it back into the model.

**Results (real file, real decode).**

| quantity | value |
|---|---|
| file size | **1112.0 MB** = **5.763 b/param** (fp32 model 6175 MB) |
| entropy-estimate delta | **+0.00%** — the exp16–20 accounting is exact |
| encode | 273 s = 5.66 M sym/s (pure-Python rANS) |
| decode + load | 413 s = 3.74 M sym/s |
| **digit round-trip** | **0 mismatches over 1,543,714,304 weights** |
| weight difference | 1.42e-2 max — from the *lossy* `ent8` scale mode only |
| perplexity | 4.4995 → 4.4995 code (max per-window Δ 3.7e-3) |

`--selftest` additionally proves bit-exact weights for fp32 scales, on every
grid (9/15/27/63/99), every scale mode, and with the act-order permutation path.

**Three real bugs, all caught by tests rather than by results.** (1) The first
blob offset was set to the header size instead of after the directory, so every
tensor slice read the wrong bytes (NaN weights) — fixed by a two-pass layout.
(2) The ent8 scale header is 10 bytes (`struct.calcsize("<ffH")`), not 12 as
assumed; the 2-byte error corrupted every decoded scale (rel err 0.91) — caught
by the self-test's scale-mode matrix. (3) scale-mode was a string at the call
site but compared as an int. The self-test matrix (grids × modes × perm) is what
made these cheap to find.

**Ops note.** Decoding hit a CUDA OOM allocating 935 MB (the 151936×1536 embed
as float32) alongside the resident model on the shared 16 GB card; the allocator
retried and the round trip still returned 0 mismatches, but a production decoder
should stream tensor-by-tensor without holding fp32 references.

**What this changes.** Every size in exp18–20 is now a *measured file size*, not
an entropy estimate. The accounting matched to +0.00%, so those tables stand
as-is. The codec is functionally validated; what remains is engineering (a
vectorized/native rANS — pure Python is ~15× too slow for production — plus
streaming decode).

**Caveats.** `ent8` scales are lossy (weights differ by ≤1.4e-2; fp32 scale mode
is bit-exact); this run is RTN (the GPTQ variant adds only permutation arrays,
~0.5% of bytes, per exp20); one model, 8 windows, ppl not capability.

**Next.** Asymmetric (min+scale) grids to lift the low-bit points; a 7B run;
then Phase 2 (QLoRA on GDScript + Godot MCP).

**Deliverables.** `experiments/k9.py`, `experiments/exp21_k9_codec.py`,
`results/exp21_k9_codec.{csv,md}`, `results/qwen_coder_1.5b_k63_embed99.k9`
(a real 1112 MB K9 file).

---

## 2026-10-03, session 13 — exp20 extended: K9's frontier now lies below the entire deployed curve

**What changed.** Adding a third and fourth body grid (k=27, 63 — the 3³ and
2⁶−1 members of the b^L−1 family) lets the allocator spend bytes past the
768 MB saturation point of session 12. Same harness and windows as exp17–19.

| palette | MB | k=max tensors | embed k | code ppl | Δcode vs fp32 (paired) |
|---|---|---|---|---|---|
| fp32 | 6174.9 | — | — | 4.492 | — |
| all-k9 (base) | 538.3 | 0 | 9 | 5.447 | +0.956 ± 0.137 |
| all-k9 + embed99 *(= exp18)* | 641.8 | 0 | 99 | 5.149 | +0.657 ± 0.106 |
| greedy ≤700 MB | 700.6 | 19 | 9 | 4.972 | +0.481 ± 0.058 |
| greedy ≤850 MB | 850.5 | 126 | 9 | 4.835 | +0.343 ± 0.048 |
| greedy ≤1000 MB | 1000.4 | 191 | 9 | 4.745 | +0.253 ± 0.043 |
| attn63/mlp15 + embed99 | 809.4 | 112 | 99 | 4.608 | +0.116 ± 0.031 |
| **all-k63 + embed99** | **1114.5** | 196 | 99 | **4.504** | **+0.012 ± 0.004** |

Deployed reference points (exp19, same windows): q8_0 1894.5 MB/+0.086 ·
q5_k_m 1285.5/+0.110 · q4_k_m 1117.3/+0.170 · q4_0 1066.2/+0.243 ·
NF4 999.5/+0.270 · q2_k 752.9/+1.426.

**Findings.**

1. **K9 dominates the deployed formats.** Comparing like-for-like, K9's
   frontier is *below* theirs:
   - **1114.5 MB at +0.012 is smaller AND ~14× lower Δ than q4_k_m**
     (1117.3 MB, +0.170), and also beats q5_k_m (+0.110) and q8_0 (+0.086)
     while being 1.15× / 1.70× smaller.
   - **809.4 MB at +0.116 dominates q4_k_m, q4_0, NF4 and q2_k** (all larger
     and worse), and reaches parity with q5_k_m (Δ 0.006 ± 0.035) at 1.59×
     smaller and with q8_0 at 2.34× smaller.
   - For every deployed format tested, at least one K9 point is smaller and
     at least as good.
2. **The near-lossless point is cheap.** k=63 body (5.68 b/param entropy-coded,
   GPTQ, act-order) + 99-level embedding is +0.012 ppl — indistinguishable from
   fp32 — at 1114.5 MB, versus q8_0's 1894.5 MB for +0.086. That is a ~4× better
   quality-per-byte than GGUF q8_0.
3. **The allocator prefers the top grid outright.** Greedy promotion chose
   the largest grid for all 196 body tensors before touching the embedding
   (≤1000 MB rows), i.e. with a smooth multi-level palette the best use of
   bytes is uniform-highest + fine embedding. The type split still matters at
   low budgets: `attn63/mlp15` at 809 MB beats the greedy ≤850 MB point
   (850.5 MB, +0.343) — attention layers carry more sensitivity per byte.
4. **Exact cross-check again**: `all-k9 + embed99` = 641.8 MB / 5.149 matches
   exp18's K9-full.

**Debugging detour (logged for the record).** Two real bugs, both caught by the
new assertions rather than by the results: (a) with `act_order=True`, GPTQ
builds its groups in the *permuted* column order, so caching digits+scales in
the original order and rebuilding produced garbage (ppl ≈ 4×10⁴) — fixed by
storing the permutation and undoing it at rebuild time; (b) the embed cache
stored a 2-tuple where rebuild unpacked 3. Both are now guarded by early
assertions, and embeddings are processed *before* the ~16 min body pass so such
bugs fail in seconds. Also: the box is shared with a ~30 GB `dotnet` process, so
caching float32 weights per level caused heavy zram swapping — caching uint8
digits + scales and freeing each Hessian as it is consumed fixed it (peak CPU
RSS ~16 GB instead of ~37 GB).

**Caveats (important).** K9 bytes are entropy-estimated, validated to
±0.01 b/param against a real rANS stream on the TinyStories artifact (exp16) —
a real K9 file writer is still the missing end-to-end step. External formats
were dequantized into fp32 HF and run on our windows, not through the llama.cpp
kernels (q8_0 reproduces fp32 within ~2%, validating the mapping, but kernel
precision is not captured). One model, one calibration corpus (Python code, so
GPTQ is in-domain), 8 windows, ppl not capability.

**Next.** (a) implement the K9 writer and re-measure end-to-end bytes + decode
speed (the last validation before this is publishable); (b) an asymmetric
(min+scale) grid, which may lift the 809 MB point toward q8_0 parity; (c) a 7B
run; then Phase 2 (QLoRA on GDScript + Godot MCP).

**Deliverable.** `experiments/exp20_k_palette.py` (env: `K9_KBODY`, `K9_KEMBED`,
`K9_BUDGETS`), `results/exp20_k_palette.{csv,md}`.

---

## 2026-10-03, session 12 — exp20: the k-palette result — **K9 now dominates Q4_K_M**

**Headline.** Allocating the grid per tensor (body k=15, embedding k=99, both
entropy-coded, body GPTQ'd) gives **768.8 MB at code ppl +0.145 ± 0.037** —
*smaller and better than the most-used deployed quant*, GGUF **q4_k_m**
(1117.3 MB, +0.170 ± 0.027), and smaller and better than **NF4** (999.5 MB,
+0.270). It is the first row that dominates on both axes.

| palette | MB | k=15 tensors | embed k | code ppl | Δcode vs fp32 (paired) |
|---|---|---|---|---|---|
| fp32 | 6174.9 | — | — | 4.492 | — |
| all-k9 (base) | 538.4 | 0 | 9 | 5.447 | +0.956 ± 0.137 |
| all-k9 + embed99 *(= exp18 K9-full)* | 641.9 | 0 | 99 | 5.149 | +0.657 ± 0.106 |
| greedy ≤700 MB | 665.3 | 196 | 9 | 4.891 | +0.400 ± 0.049 |
| attn15/mlp9 + embed99 | 656.8 | 112 | 99 | 4.982 | +0.490 ± 0.076 |
| **all-k15 + embed99** | **768.8** | 196 | 99 | **4.637** | **+0.145 ± 0.037** |
| *(reference)* GGUF q4_k_m | 1117.3 | — | — | 4.662 | +0.170 ± 0.027 |
| *(reference)* NF4 | 999.5 | — | — | 4.761 | +0.270 ± 0.053 |
| *(reference)* q2_k | 752.9 | — | — | 5.918 | +1.426 ± 0.178 |

**Findings.**

1. **Domination of q4_k_m and NF4.** 768.8 MB beats q4_k_m's 1117.3 MB on size
   (1.45×) *and* on quality (+0.145 vs +0.170); it also beats NF4 on both axes.
   Against q2_k (752.9 MB, +1.426) it is the same size at ~10× lower Δ.
2. **Exact cross-check.** `all-k9 + embed99` = 641.9 MB / 5.149 reproduces
   exp18's K9-full to the decimal, so the palette machinery composes correctly.
3. **Attention layers are the sensitive type.** `attn15/mlp9 + embed99`
   (656.8 MB, +0.490) beats `all-k9 + embed99` (641.9 MB, +0.657) by −0.167
   for +15 MB — promoting attention first is an efficient move.
4. **The greedy allocator's answer at ≤700 MB was "all body → k15, embedding
   stays k9"** (665.3 MB, +0.400) — it spent on the body before the embedding,
   and that beat the attention-only mix at comparable bytes.
5. **The k∈{9,15} palette space is exhausted at 768.8 MB**: budgets of 850 /
   1000 / 1150 MB all collapse to the same point. To spend more bytes
   productively the palette needs a third level — k=27 or 63 on the body, and/or
   an asymmetric (min+scale) grid.

**Why this beats q4_k_m at smaller size.** Three compounding advantages:
entropy coding of the 15-level digits (3.47 vs 4.0 raw b/param), GPTQ with
act-order, and a fine 99-level (not 4/6-bit) embedding grid. The grid family
k=2^4−1=15 is the binary member of this project's b^L−1 family — so "int4" here
is the repeating-decimal grid, entropy-coded.

**Caveats.** K9 bytes are entropy-estimated (validated to ±0.01 b/param against
the real rANS stream on the TinyStories artifact, exp16) while the GGUF numbers
are actual file sizes — a real K9 file writer is still the missing end-to-end
step. External formats were dequantized into fp32 HF (exp19 caveat: not the
llama.cpp kernels, though q8_0 validates the mapping). One model, one
calibration corpus, 8 windows.

**Next.** (a) add a third palette level (k=27/63) and/or asymmetric grids to
push the frontier toward q5_k_m/q8_0 quality at ≤1 GB; (b) implement the K9
writer and re-measure end-to-end bytes + decode speed; (c) 7B-scale run; then
Phase 2 (QLoRA on GDScript + Godot MCP).

**Deliverable.** `experiments/exp20_k_palette.py`,
`results/exp20_k_palette.{csv,md}`.

---

## 2026-10-03, session 11 — Phase 1b: K9 vs deployed formats (exp19, GGUF + NF4)

**The comparison the project had never made.** External quantizers evaluated on
the *same* 8 windows and through the same `eval_harness.py` as exp17/exp18, by
dequantizing each format into the fp32 HF model (isolates the quantization
effect; bytes are authoritative file/packed sizes). GGUF via the already-present
`gguf` 0.17.1 lib (`gguf.quants.dequantize`); NF4 via `bitsandbytes` 0.50.2 in a
PEP-668 venv (`.venv-baselines`, system-site-packages so it inherits torch).

| format | MB | b/param | code ppl | Δcode vs fp32 (paired) |
|---|---|---|---|---|
| fp32 | 6174.9 | 32.00 | 4.492 ± 0.366 | — |
| GGUF q8_0 | 1894.5 | 9.82 | 4.578 ± 0.377 | +0.086 ± 0.014 |
| GGUF q5_k_m | 1285.5 | 6.66 | 4.602 ± 0.375 | +0.110 ± 0.017 |
| GGUF q4_k_m | 1117.3 | 5.79 | 4.662 ± 0.381 | +0.170 ± 0.027 |
| GGUF q4_0 | 1066.2 | 5.53 | 4.735 ± 0.368 | +0.243 ± 0.037 |
| bitsandbytes NF4 | 999.5 | 5.18 | 4.761 ± 0.392 | +0.270 ± 0.053 |
| GGUF q2_k | 752.9 | 3.90 | 5.918 ± 0.458 | +1.426 ± 0.178 |
| **K9 full (body9-GPTQ + embed99)** *(exp18)* | **641.9** | **3.23** | **5.149 ± 0.402** | **+0.657 ± 0.106** |

**Findings.**

1. **K9 dominates the smallest deployed quant.** q2_k costs 752.9 MB for
   +1.426 code ppl; K9-full is *smaller* (641.9 MB) **and better** (+0.657).
   First head-to-head win over a shipped format.
2. **K9 is 1.74× smaller than q4_k_m** (641.9 vs 1117.3 MB) at +0.657 vs
   +0.170 — it is the smallest point on the curve, not the best quality-per-byte.
3. **Quality-per-byte (Δcode/MB):** q4_k_m 1.5e-4 < NF4 2.7e-4 < **K9 1.0e-3**
   < q2_k 1.9e-3 — K9 sits between NF4 and q2_k, at the low-bit end of the
   deployed frontier.
4. **q8_0 validates the harness**: dequantized q8_0 reproduces fp32 within ~2%
   (Δcode +0.086), so the GGUF name/shape mapping (and the `(out,in)` layout,
   and the lm_head untie for GGUF's separate Q6_K `output.weight`) is correct.

**Caveats.** External formats are dequantized into fp32 HF and run on our
windows, not through the llama.cpp/bnb kernels — the mapping is validated by
q8_0's near-parity, but kernel-level precision differences are not captured.
`gptqmodel` has no py3.14 wheel, so an *external* GPTQ-int4 row is still
missing (our own int4-15 + GPTQ from exp18 is the stand-in: 1517.7 MB,
code 4.641, Δcode +0.149 — bigger than q4_k_m but comparable quality).
AWQ was skipped (no wheel confidence); `torchao` is available if wanted.

**Deliverable.** `experiments/exp19_external_baselines.py` (+ `--only q8_0`
validation mode), `results/exp19_external_baselines.{csv,md}`.

**Next.** The result reframes the pitch: K9's value is not "better than Q4_K_M"
but "**a smaller operating point than anything shipped, and it beats q2_k**".
Two ways to strengthen it: (a) mixed-precision k-palette to move K9 up the
curve toward NF4 quality at ~800-900 MB, and (b) a 7B-scale run where quant
quality matters more. Then Phase 2 (QLoRA on GDScript + Godot MCP).

---

## 2026-10-03, session 10 — a real noise floor + GPTQ at 1.5B (exp18; eval_harness)

**Measurement first.** exp17's single-window point estimates are replaced by
`experiments/eval_harness.py`: N disjoint windows per corpus, mean ± SE, and
`paired_delta()` comparing schemes on identical windows. On Qwen2.5-Coder-1.5B
with 8×1024-token windows, per-scheme code-ppl SE is 0.36–0.62, but the
**paired** Δ vs fp32 has SE 0.04–0.33 — so a code Δ of ~0.2 is now significant,
and exp17's smaller claims (e.g. "+0.002 embedding") are explicitly inside the
noise band.

**exp18 v1 was a failure worth keeping.** Porting exp6's GPTQ (damp 1%, 32
calibration blocks, no act-order) to 1.5B made it *worse* than RTN on every
grid (9-ninths code 5.703 vs RTN 5.599; wiki 22.24 vs 19.41). Diagnosis: the
8960-dim layers have ~3.6 calibration samples per dimension, so H is badly
conditioned and the compensation amplifies noise. **Fix: act-order + 128
calibration blocks (131k tokens, ~15 samples/dim) + damping 0.05.** GPTQ then
helps:

| scheme | k | GPTQ | digits b/p | MB | code ppl | Δcode vs fp32 (paired) |
|---|---|---|---|---|---|---|
| fp32 | — | — | 0 | 6174.9 | 4.492 ± 0.366 | — |
| 8-level RTN | 8 | — | 2.488 | 1357.2 | 6.952 ± 0.619 | +2.460 ± 0.325 |
| 9-ninths RTN | 9 | — | 2.670 | 1387.0 | 5.599 ± 0.445 | +1.107 ± 0.127 |
| **9-ninths GPTQ** | 9 | yes | 2.693 | 1390.7 | **5.150 ± 0.403** | **+0.658 ± 0.106** |
| int4-15 RTN | 15 | — | 3.452 | 1515.1 | 4.855 ± 0.396 | +0.364 ± 0.066 |
| int4-15 GPTQ | 15 | yes | 3.468 | 1517.7 | 4.641 ± 0.364 | +0.149 ± 0.038 |
| 99-level RTN | 99 | — | 6.218 | 1968.0 | 4.493 ± 0.364 | +0.001 ± 0.004 |
| **K9 full (body9-GPTQ + embed99)** | 9 | yes | 3.228 | **641.9** | 5.149 ± 0.402 | +0.657 ± 0.106 |

(Wiki ppl: fp32 14.064; 9-ninths RTN 19.408; 9-ninths GPTQ 19.582; K9 full
19.603. Full table in `results/exp18_qwen_gptq_noise.md`.)

**Findings.**

1. **GPTQ works on the grids at 1.5B — once conditioned.** 9-ninths improves
   5.599 → 5.150 code ppl (−8.0%, paired Δ 1.107→0.658, ~2.7σ); int4-15
   4.855 → 4.641 (−4.4%). The gain is larger on the coarser grid, as exp6 found.
2. **GPTQ is in-domain.** Code ppl improves, wiki ppl is flat-to-slightly-worse
   (+0.17 for 9-ninths) — calibration is Python code. Compensation tunes to the
   calibration distribution; state that in any write-up.
3. **The 9-level grid stays frontier-efficient.** At 2.693 b/p, 9-ninths+GPTQ
   gives code 5.150 vs the 8↔15-GPTQ linear interpolation 6.469 — far below.
4. **The 99-level embedding is still free**: K9 full (641.9 MB) matches
   body9-GPTQ (1390.7 MB) to within +0.001 code ppl while deleting 749 MB of
   fp32 embedding. Third replication of the exp10 result, now with CIs.
5. **int4-15 remains ahead on raw quality at equal-ish bits** (4.641 vs 5.150,
   at 3.468 vs 2.693 b/p) — the ninths grid is the low-bit point, not the
   accuracy leader. Unchanged conclusion.

**Method lesson (portable).** exp6's GPTQ recipe (damping 1%, no act-order,
64×1024 calibration on ≤3072-dim layers ≈ 21 samples/dim) does not transfer to
8960-dim layers. Rule: calibration ≳ 10× the layer's input dim, always
act-order, and expect to tune damping per model scale. The `exp6.gptq_hinv`
CPU-float64 path itself was fine (2.8 s for an 8960² inverse).

**Caveats.** Hessians collected once on the fp32 model (not sequentially on the
partially quantized model, exp6's exact method) — a documented simplification;
one model, one calibration corpus; 8 windows (SEs are usable but a 3rd decimal
is not).

**Next.** Phase 1b — bytes-matched external baselines (GGUF Q4_K_M / GPTQ-int4 /
AWQ / NF4) so K9's 641.9 MB can be placed against what the ecosystem ships;
then the asymmetric/act-order grid and a k-palette mixed-precision policy.

---

## 2026-10-03, session 9 — K9 PTQ on a modern coder LLM (exp17, Qwen2.5-Coder-1.5B)

**The scaling move.** First grid quantization on a real modern coder instead of
TinyStories-33M: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Qwen2ForCausalLM — 28
layers, hidden 1536, GQA 12q/2kv, SwiGLU, RMSNorm, RoPE, **tied** embedding,
vocab 151,936). 1,543,714,304 dedup params: body 1,310,195,712 (196 Linear),
embedding 233,373,696, fp32 rest 144,896 (RMSNorm weights + Qwen2 attention
biases). Grids g=64 on the input dim; eval 24×1024 WikiText-103 test + 24×1024
Python (`codeparrot/codeparrot-clean-valid`). Rate = per-tensor digit entropy
(rANS verified on a layer subset: entropy 2.6514 vs rANS 2.6515 b/param) +
entropy-coded log scales + fp32 rest.

| scheme | digits b/p | model MB | ppl wiki | ppl code |
|---|---|---|---|---|
| fp32 | — | 6174.9 | 14.041 | 3.716 |
| 8-level g64 (body) | 2.488 | 1357.2 | 27.914 | 5.323 |
| 9-ninths g64 (K9 body) | 2.670 | 1387.0 | 19.007 | 4.477 |
| int4-15 g64 (body) | 3.452 | 1515.1 | 15.168 | 3.940 |
| 99-level g64 (body) | 6.218 | 1968.0 | 14.094 | 3.720 |
| 9-level full (body9+embed9) | 2.672 | 534.7 | 20.629 | 4.718 |
| **K9 full (body9+embed99)** | **3.209** | **638.2** | **19.035** | **4.479** |
| int4 full (body15+embed15) | 3.454 | 685.6 | 15.594 | 3.986 |

(Full rows quantize the tied embedding too; body-only rows leave it fp32, which
is 934 MB of their MB column.)

**Findings.**

1. **The odd middle alphabet is frontier-efficient on a modern coder too**
   (P10-style): the K9 body at 2.670 b/p gives code ppl 4.477, while the
   8↔15 linear interpolation at the same rate gives 5.062. The 9-level grid is
   *below* its neighbours' interpolation — the TinyStories result replicates on
   a real 1.5B code model.
2. **int4-15 still wins on raw quality at ~7% more bytes**: full-model
   int4 (685.6 MB) gives code 3.986 / wiki 15.594 vs K9-full (638.2 MB) code
   4.479 / wiki 19.035. The ninths grid's honest role is *low-bit* operation,
   not beating the binary 4-digit grid — same conclusion as exp4/exp7.
3. **A fine embedding grid is ~free** — 10× scale confirmation of exp10:
   body9-only (embed fp32, 1387.0 MB) → K9-full (embed99, 638.2 MB) costs only
   +0.028 ppl wiki / +0.002 ppl code, while removing 934 MB of fp32 embedding.
4. **Quantization hurts general text more than code** (code +20.5%, wiki
   +35.6% for K9-full): the code-specialized model's *weaker* capability
   degrades more — relevant to the game-dev plan, where general reasoning is
   the fragile part.
5. 99-level body is near-lossless: code 3.720 vs fp32 3.716 at 6.2 b/p.

**Caveats.** 24.5k-token eval per set (noise not yet characterised); bytes are
entropy-based (rANS verified on a 3-tensor subset, not the full model); no
external baselines yet — Q4_K_M needs `llama.cpp`, GPTQ needs `auto-gptq`/
`gptqmodel`, AWQ/NF4 need `llms-compressor`/`bitsandbytes`, none installed;
one model, one seed.

**Ops note.** The first eval slice read the wrong column (`codeparrot-clean`
stores code under `content`, not `text`) and reported nonsense ppl (738) — a
reminder to sanity-check in-domain ppl against the fp32 baseline before
trusting deltas.

**Next.** Bytes-matched external baselines (GGUF Q4_K_M / GPTQ-int4 / NF4) on
the same eval; then GPTQ-on-the-grids (exp6 machinery at 1.5B); then Phase 2 —
QLoRA on GDScript/Godot data with the Godot MCP as the agentic runtime.

---

## 2026-10-03, session 8 — the EC-SQ ceiling (exp15) + real codec accounting (exp16); K9 spec

**exp15 — alphabet sweep + the entropy-constrained quantizer (EC-SQ).** Closes
the open half of RQ3 / docs/01 §4a: does the 9-level grid matter, and how far
is the *uniform* ninths grid from the best scalar quantizer at its own
entropy? New comparator: EC-SQ (min D s.t. H ≤ R; Chou–Lookabaugh–Gray
alternation — λ=0 reproduces exp3's Lloyd–Max bit-exactly on k=3/8/9/16).
Sources: Gaussian (exp3's weights, seed 42) + standardized Student-t(4).
Verdicts: P25 **PASS** (odd-k zero-level sawtooth at coarse k, overtaken by
step refinement at fine k); **P26/P27/P28 FAIL**, informatively:

1. **Uniform ninths is within 11% of EC-SQ at its own entropy** (0.1297 vs
   0.1171 at H=1.81 b). P27's ">2× waste" fails: placement is not what limits
   the 9-level point — its low rate is.
2. **Uniform + entropy coding is asymptotically rate-optimal**: eff 0.99–1.03
   for k ≥ 17 (classical result, now on a measured entropy axis).
3. **Fixed-rate Lloyd–Max is not the right ceiling.** At matched entropy it is
   23% above the EC-SQ envelope at k=9 on Gaussian, 80% on the heavy tail — at
   a given entropy the best grid is more non-uniform than the MSE-optimal
   fixed-rate one. The lever is *entropy-constrained codebook design* (a
   shared/parametric codebook approximates it cheaply), not the base-9 identity.
4. **Linear-uniform grids are not heavy-tail-robust**: coarse even-k grids
   collapse (k=4 rel MSE 88.96 under absmax, 10.26 with a p99.99 clip) — why
   per-(row,group) scales are load-bearing on real weights.

**exp16 — realistic codec accounting, measured on the stored artifacts.**
Re-costs exp13/exp11 with per-tensor rANS (tables counted) and realistic scale
precision. Findings:

1. **The published "effective bits" survive per-tensor tables.** Tables cost
   ~2 B/symbol (< 1 KB/model at k=9) and per-tensor rates reproduce the
   global-table figures within ±0.007 b/param (body RTN 2.700 vs 2.701
   published; embed 6.275 = 6.275).
2. **Scale precision is the free win.** fp32 → entropy-coded 8-bit log scales
   takes the body scale tax from 0.50 → 0.125 b/param; the full exp13 model
   goes 45.5 → **42.1 MB (−7.4%)** with the digits untouched. int6 ≈ ent-log8.
3. **The npz artifacts are ~26× the coded size** (1.10 GB vs ~42 MB): digits
   stored int64. A codec stores the rANS stream, not the digits.
4. Ternary body codes to **1.578 b/param** per-tensor (1.585 global; pair codec
   1.584, the BitNet bound); the full QAT-ternary model codes to ~38 MB.

**Code-reading finding (methodological).** exp11's QAT `quant_deq` groups along
the last axis of the raw parameter — the *output* dim for GPT-Neo `Conv1D` —
whereas exp4b/6/7/10/13 group along the *input* dim (after transpose). For the
non-square c_fc/c_proj matrices the QAT body grids are not byte-identical in
convention to the PTQ grids they are compared against. Byte *counts* are
unaffected (r·c/g either way); quality comparisons across the two should be
re-run under one convention.

**Deliverable.** `docs/04-k9-codec-spec.md` (RQ5): the container — odd-grid
digits + per-(row,group) scales + order-0 rANS (M=4096) + serialized per-tensor
tables + scale modes + ternary pair mode + optional fitted-codebook mode (v2,
exp15-motivated) + a bit-exact conformance test. exp16 is the reference byte
table.

**Next.** Implement K9 (vectorized rANS) and roundtrip the exp13 artifact
end-to-end; then the 1B-model scaling run (see the enhancement review).

---

## 2026-10-03, session 7 — RQ4 closed (null); deployable full-model artifact; longer-QAT paused by a dead GPU

**exp12 — RQ4 CLOSED, clean null.** Scanned exp11's *trained* digit artifacts
(the best possible case for trained-in structure): lag-1 match rates sit at
their iid expectations (9-level: 0.181 vs 0.178 expected; ternary: 0.346 vs
0.341; embedding 99-level: 0.017 vs 0.015); order-1 Markov conditional
entropy gains ≈ 0.0–0.2% over the marginal H (P24 PASS at <10%); shuffled
controls match. Conclusion: **frequency coding (rANS) is effectively optimal
for these streams — the lossless story is closed with symbol entropy** (this
completes the arc: P4's null → exp12's null at the trained-model level).
Side observation: trained ternary digits are nearly max-entropy
(1.57–1.58 ≈ log2 3) — QAT balanced the codebook. Artifact note: exp11's
npz streams are stored FLAT; exp12 recovers (row, col) shapes from the
known inventory (`shape_for_key`) — keep 2-D shapes in future artifacts.

**exp13 — the DEPLOYABLE PTQ artifact exists.** embed99 + body9-GPTQ =
**44.96 ppl @ 45.5 MB total** (fp32 was 274.1 MB → −83%; P23 ≤ 50 PASS);
the RTN variant: 54.78 (cross-check vs exp10 EXACT). The whole model is
saved as digits + per-(row, group) scales: `results/full_model_digits.npz`
(reload recipe in the file header). One nuance for the write-up: with a
GPTQ'd body, fine embeddings cost +1.3 ppl (43.64 → 44.96) vs only +0.1
with an RTN'd body — quantization-error interactions across components
compound; budget ~1.3 ppl for the embedding stage when the body is
compensated. exp13 ran 16 min on CPU.

**exp14 — longer QAT: PARTIAL, paused by a hardware event.** Control @4000
steps completed: **3.895 ppl (from 4.591 @2000)** — more tokens help fp32
too (P22's data point). Then a CUDA driver fault (ERROR_LAUNCH_FAILED, 719,
~13:42) and **the GPU disappeared from the system** (nvidia-smi: "No
devices were found"; torch: zero devices). exp14b's arm runs silently fell
back to CPU training (~6 h/step-pace) and were stopped; the recipe@4k and
tern@4k arms are pending. Resume (one command per arm, fresh CUDA context
each, after the driver recovers / a reboot):

    python3 experiments/exp14b_arm_runner.py recipe
    python3 experiments/exp14b_arm_runner.py tern

(the tern invocation also writes exp14's merged verdicts P20/P21/P22 +
the 4000-step ternary payload; P22 is already decidable: PASS, 4.591 → 3.895.)

**Ops notes.** (1) Error 719 on a shared GPU killed the device system-wide —
check dmesg/journal for the nvidia driver's story after recovery; the
per-arm-process design (exp14b) is the standing resilience pattern. (2)
device_setup's CPU fallback is SILENT — before next session, make it print
a loud "GPU unavailable, falling back to CPU (SLOW)" (a fix queued).
(3) Training-rate numbers for planning: ~0.28 s/step at 33M/batch-8 on the
GPU; CPU is ~20-50× slower — CPU is for evals and scans, not training.

**Next.** GPU back → finish exp14b arms → refresh the frontier table;
then from-scratch pretraining with the recipe (the strongest "new model"
claim), vectorized rANS, and the write-up.

---

## 2026-10-03, session 6 — RQ8: a NEW model, created on the grids (exp11, 3 arms)

**Question.** Can we create a new model using this research — and what are
the gains and losses?

**v1 → v2 (honest iteration).** v1 trained only the QAT arms; its log showed
ALL runs dropping people ~40.7 → ~4, far below the pretrained baseline —
v1 pre-registered a "format confound" hypothesis. v2 added the CONTROL arm
(identical schedule, fp32 weights) + a train-format diagnostic; the
diagnostic REFUTED the format hypothesis (pretrained people on
training-format windows: 36.4 ≈ val 40.7): the checkpoint is simply not
converged, and continued training is the true confound. The matched control
prices that gain separately — which is what makes the v2 numbers honest.
(v1 also crashed on a digit-key parse bug before publishing the confounded
comparison — asserts and the control caught it.)

**Design.** Three arms, identical 2000-step schedule (batch ad 8×512 ≈ 8.2 M
train tokens each from a 40M-token train pool, AdamW + cosine, weight-swap
STE; batch fell to 8 with GPU free at 7.1 GB): CONTROL (fp32) · QAT-RECIPE
(body → 9-level ninths g64; wte/wpe → 99-level g64) · QAT-TERN (ternary body,
learnable per-row gamma; embeds → 99). Baselines assert-anchored to earlier
sessions (fp32 40.663, PTQ body9 54.679 — both exact).

**Results (window A people):**

| variant | ppl | vs CONTROL (4.591) |
|---|---|---|
| fp32 pretrained | 40.663 | +36.07 |
| PTQ body9 | 54.679 | +50.09 |
| PTQ embed99+body9 | 54.781 | +50.19 |
| PTQ ternary-g64+embed99 | 989.767 | +985.18 |
| **CONTROL (fp32, trained)** | **4.591** | — |
| **QAT-RECIPE** | **4.886** | **+0.30 (+6.4%)** |
| **QAT-TERN** | **5.980** | **+1.39 (+30%)** |

**Verdicts: P17 PASS · P18 PASS · P19 PASS** (all three). Ternary payload:
single-stream rANS **1.585** b/param; **base-9 pairs 1.584 b/param** —
exactly log2(3), the BitNet bound, now carried by a model that WORKS.

**Reading.**

1. **The ninths recipe is essentially free with native training:** +6.4%
   people over the matched control, with the body at ~3.2 b/param and
   embeddings at ~6.7 — a trained, grid-native model, not a damaged one.
2. **The ternary payload is real:** 1.39 people of loss at ~1.6 b/param
   body (~9.8× vs fp16 body bytes for the matrix weights). PTQ-ternary was
   989.8 people; QAT brings ternary into working range — RQ8 (the payload
   question) is closed in the affirmative. With longer training the ternary
   gap should shrink further (BitNet trains near-fp32 at scale); 2000 steps
   is a lower bound.
3. **Methodological record:** PTQ→QAT comparisons (P17) look like 11×/165×
   gains but are confounded by the extra training; the matched control is
   the metric that matters. Also: ternary gamma gradient in the swap form is
   applied exactly via dL/dgamma_r = Σ_j dL/ddeq·t (recomputed from the
   step's frozen latent/gamma pair).
4. **Artifacts are real deliverables:** results/qat_digits_{control,recipe,
   tern}.npz hold the new models in digit form (control's fp32 latents are
   not digit-coded); the ternary model's body digits + base-9 pairing are
   the compression endgame of the original idea.

**Caveats.** One seed per arm; window-A people; lr 2e-4 tuned for fast
adaptation, not final quality; gamma learned per row (BitNet uses learned
per-parameter scaling); "gain" vs pretrained-fp32 is 9× training gain
available to any arm — do not confuse with quantization gain.

**Next.** Longer QAT + multi-seed variance; from-scratch pretraining with
the recipe; the GPTQ'd full-model point (embed99 + body9-GPTQ); vectorized
rANS → a bigger host model; RQ4 structure scan; then write-up + publish.

---

## 2026-10-03, session 5 — closing the open items (exp8, exp9, exp9b, exp10)

**exp8 — GPTQ + Lloyd combined: REFUTED (a clean null).** Compensation on
static fitted codebooks *hurts*: RTN fp32-codebook 40.534 ppl (exact
cross-check vs exp4b) vs GPTQ fp32-codebook 41.089 (+0.56; MSE also up
0.0247→0.0343). Mechanism: RTN on fitted codebooks is already per-weight
near-optimal; compensation shifts columns so weights land in wrong cells,
and a static codebook can't adapt. The fp16 codebook was measured properly
for the first time: RTN fp16 40.622 @ 5.02 b/param — only +0.09 ppl vs fp32
codebook (P11 PASS on the RTN phase; FAIL on the GPTQ phase where the gap
hits 0.33). **The accuracy king stands, now measured honestly:
LM9-g64 RTN with fp16 codebook = 40.622 ppl @ 5.019 b/param** — int8's
40.63 ppl cost 8.02 b/param, so this dominates int8 on both axes.

**exp9 — the 127-level anomaly is REAL (P12 PASS).** Three disjoint eval
windows all reproduce it (63/99/127 people: 40.65/40.57/40.98 ·
39.13/39.24/39.61 · 42.78/42.76/43.13). Not localized either (P13 FAIL as
"localized"): both attn-only (−0.30) and mlp-only (−0.10) invert. Note:
baseline ppl varies hugely by window (39.1 vs 42.8) — cross-window
comparisons must be paired (within-window), which they were.

**exp9b — cure test: fitted codebooks largely cure the anomaly (P16
PASS).** Fitted-99 40.457 ppl (parity-or-better vs fp32!) and fitted-127
40.635: the inversion shrinks from +0.41 (uniform) to +0.18 (fitted, ~noise
band). So the uniform-127 pathology is endpoint/step structure — the ±m-
matched step of an ultra-fine uniform group interacts badly with outlier
channels; fitted center placement fixes it. Cheap cure: fit codebooks when
going finer than ~64 levels per group.

**exp10 — embeddings quantized (tied lm_head confirmed).** Embed9 alone
costs +3.98 ppl (44.64 on an fp32 body; P14 PASS at the 10% wire, 9.8%).
The combined story: embed9+body9 = 60.99 ppl @ 27.7 MB total (−90% bytes,
**super-additive** ppl penalty — both damage sources compound);
**embed99+body9 = 54.78 ppl @ 45.5 MB (−83% bytes)** — the 99-level
embedding grid costs only +0.10 ppl over body9-only, so the right recipe is
*finer grids for the embedding table* (heavier per-row tails than body
matrices). P15 PASS. Body9 row cross-checked exactly against exp7 (54.679).

**Frontier after session 5** (body-weight b/param incl. all side info):
- 43.64 ppl @ 3.21 b/p — 9-ninths-g64 + GPTQ (low-bit champion)
- 40.62 ppl @ 5.02 b/p — LM9-g64 RTN, fp16 codebook (accuracy king; beats int8 both axes)
- full-model bytes: 45.5 MB (−83%) @ 54.78 ppl (embed99+body9) · 27.7 MB (−90%) @ 60.99 (embed9+body9)

**Next.** (1) refit-the-codebook-after-compensation (the one untried GPTQ+Lloyd
combo); (2) full-model point: embed99 + body9-GPTQ together; (3) scale to a
bigger model once the rANS coder is vectorized (23 min/scheme at 7B in pure
Python); (4) RQ4 (digit-stream structure scan) and RQ8 (ternary QAT) remain.

**Ops note.** The GPU is shared: another process held 8.6 GB mid-session and
cross-entropy's 1.65 GB logits slab OOM'd at batch 8; exp10 now evaluates at
batch 2 with cache clearing. Worth remembering for bigger evals on this box.

---

## 2026-10-03, session 4 — multi-digit repeating grids: the b^L−1 law (exp7)

**Question.** Can the single-digit ninths story ("each weight = one repeating
decimal digit") extend to *multi-digit* repeating decimals?

**Law (now in docs/01 §2 with one correction).** An odd symmetric grid of
k = 2H+1 levels is w = (b−H)·m/H with block b ∈ {0..k−1}; whenever k = bᴸ−1,
the block IS an L-digit repeating decimal in base b, and the weight is
**affine in one repeating decimal: w = m·((k/H)·0.b̄ − 1)**. The first
draft asserted the cruder form "w = 2·0.b̄ − 1" — the round-trip assert
caught it (off by a quantum: w = m·(2·0.b̄ − 1) + m/k), and the corrected
form checks exactly against exp5's verified printed equation at L=1.
Notable consequence: exp4's champion int4-15 **is** the base-2 L=4 member
of the same family (15 = 2⁴−1, binary 4-digit repeating blocks); the
decimal family is 9/99/999 levels. Base-3 multi-digit (3²−1 = 8 levels) is
an even grid — no zero level — and is ruled out by exp3's odd-grid result.

**Coder upgrade for wide alphabets (prerequisite).** rANS frequency tables
generalized to M = 2^12 (4096) — renorm condition f·2^32/M (reduces
byte-identically to the old f·2^24 at M=256; regression-tested against
exp1's local twin) — and the decoder mask hardened from hardcoded `& 255`
to `& (M−1)` (a latent bug that only bites beyond 256 symbols). 999-symbol
rANS now codes within 0.04% of entropy.

**exp7 setup.** Same model/eval as exp4–6. Odd symmetric grids at g=64,
fp32 scales counted (+0.5 b/p), M=4096 coding; GPTQ on 9/15/99; multi-digit
print round trip on the 99-level grid.

**Results (RTN, ppl @ eff b/param):** 7: 76.28 @ 2.81 · 9: 54.68 @ 3.20 ·
15: 42.33 @ 3.98 · 63: 40.65 @ 6.10 · **99: 40.57 @ 6.75** · 127: 40.98 @
7.11 · 255: 40.45 @ 8.11 · 999: 40.70 @ 10.06 · 1023: 40.66 @ 10.10.
GPTQ deltas: 9: −11.04 · 15: −0.22 · 99: +0.16.

**Pre-registered verdicts:** P6 (entropy < log2 levels) **PASS** ·
P7 (ppl strictly decreasing in levels) **FAIL** — 127-level (40.98) is worse
than 99-level (40.57), breaking monotonicity in the noise region; MSE stays
monotone throughout, so the MSE↔ppl divergence extends even to 6–8 bit
grids · P8 (GPTQ gain shrinks with fineness, stays negative) **FAIL** on the
sign clause (Δ99 = +0.16, within noise) while the magnitude ordering held ·
P9 (multi-digit print round trip bit-exact + affine identity) **PASS** ·
P10 (99-g64 below the 63↔127 interpolation: 40.57 vs 40.86) **PASS**.

**Reading.**

1. **The multi-digit decimal family works and is frontier-efficient** (P10):
   the 99-level "0.(43)" grid is the best point in the 6-bit band and pays
   nothing for not being binary-aligned.
2. **Above ~6 effective bits everything is fp32-parity** (all within ±0.3 of
   40.66) — alphabet size stops mattering there; the interesting regime
   remains ≤ 4 b/param, where the ordering is strict: 7 → 9 → 15 people
   76.3 → 54.7 → 42.3, and GPTQ'd 9-ninths (43.6 @ 3.21) sits within 1.3 ppl
   of 15-level RTN at 0.77 fewer bits.
3. **Open flags from the failed predictions:** the 127-level inversion
   (worse ppl at more bits, better MSE) is unexplained — plausibly
   outlier-channel interaction with group scales; worth an act-order or
   per-layer ablation before publishing any monotonicity claim. And GPTQ on
   fine grids is neutral-to-noise, refining session-3's rule: compensation
   buys ppl only on coarse grids.

**Print artifact scale.** L=2 notation ("0.(43)", 6 chars/weight +
scales) extrapolates to ~181 MB of text for all 28.3M linear weights;
compact and rANS forms behave as in exp5 (mechanism already proven there).

**Caveats.** Same model/eval window as exp4–6; noise-region verdicts are
within ±0.3 ppl; 999/1023 GPTQ not run (pointless at parity).

**Next.** GPTQ+Lloyd combined (compensated fitted codebooks); act-order
ablation for the 127 anomaly; embeddings/lm_head quantization; RQ4/RQ8.

---

## 2026-10-03, session 3 — "more math on the GPU": GPTQ compensation (exp6) + speed profile

**Question.** Can we increase performance by running more math on the GPU?
Two answers: (a) quality — activation/Hessian-aware compensation (GPTQ,
Frantar et al. 2022, arXiv:2210.17323) is the GPU math that buys perplexity;
(b) speed — profile where wall time actually goes and where GPU coding would
matter.

**exp6 setup.** Calibration = 64 TRAIN-split blocks × 1024 tokens (never the
eval window). Sequential per-layer Hessians H = 2·XᵀX collected from
calibration forwards through the partially-quantized model; damper = 1% mean
diag; Hinv = upper Cholesky of H⁻¹ — computed on **CPU in float64** because
cusolver on this Blackwell GPU (GB206, torch 2.14) errors inside
cusolverDnXpotrs regardless of damping (backend bug, not conditioning; H is
≤ 3072² so CPU-f64 costs milliseconds). Column loop over the input dim with
static per-(row, group-of-64) grids/scales IDENTICAL to exp4b — every scheme
 ran RTN first (compensation off) as an internal baseline.

**Harness cross-check.** RTN phase reproduced exp4b exactly: 8-level
58.213 = 58.213, 9-ninths 54.679 = 54.679, int4 42.334 = 42.334 (the int4
row's cross-check label mis-keyed on "(15)" vs "-15", but the values are
bit-identical — harness validated on all three grid schemes).

**Results (ppl, effective b/param = rANS digits + fp32 side info):**

| scheme | RTN | GPTQ | Δ ppl | eff b/p |
|---|---|---|---|---|
| int4-15 g64 | 42.334 | 42.119 | −0.22 | 3.98–3.99 |
| 8-level g64 | 58.213 | 46.843 | **−11.37** | 3.02–3.03 |
| 9-level ninths g64 | 54.679 | 43.644 | **−11.04** | 3.20–3.21 |
| ternary absmean g64 | 1029.694 | 4027.256 | **+2997.6** | 2.08 |

**Reading.**

1. **Compensation pays overwhelmingly where the grid is coarse.** Uniform
   ninths at 3.21 b/param goes from 54.7 → 43.6 ppl (within 7% of fp32)
   with GPTQ — now ahead of per-row int4 (45.0) and in the same band as
   LM9-row (47.2). int4-15 gains almost nothing (RTN already fine on a
   15-level grid). The rule of thumb: error-compensation machinery is worth
   ~11 ppl on coarse grids and ~0 on fine ones — at this model scale.
2. **GPTQ cannot rescue PTQ ternary — it makes it worse (10×).** The
   propagation update assumes small rounding errors; ternary's gross errors,
   propagated through Hinv, blow later columns beyond their thresholds.
   Ternary-g64-RTN (1029.7) ≈ ternary per-row (992.3), so group scales don't
   rescue ternary either. QAT (RQ8) remains the only ternary path.
3. **Honest frontier after exp6** (all side info counted): LM9-g64 fitted
   40.5 ppl @ ~5.0 b/p (fp16 codebook est.) — accuracy king; int4-g64+GPTQ
   42.1 @ 3.99; **9-ninths-g64+GPTQ 43.6 @ 3.21 — low-bit champion**;
   8-level-g64+GPTQ 46.8 @ 3.03. The base-9 grid's viability on a real model
   survives and strengthens: with compensation it's competitive at its bit
   depth.
4. **Speed profile (the hardware half of the question).** All quantize +
   eval math already runs on GPU (torch). Pure-Python rANS: 10.1 M sym/s
   encode, 9.1 M sym/s decode → 6 s round trip for this whole model; a 7B
   model projects to ~23 min/scheme → the scale where coding must move to
   vectorized interleaved rANS (numpy, multiple states) or nvCOMP's GPU ANS.
   Small linear algebra already CPU-f64 on this driver (cusolver bug) — no
   accuracy impact, ms cost.

**Requirements added by this session:** calibration data from the train
split (65k tokens; must NOT be the eval window); CPU-f64 fallback for the
small linear algebra on this driver; ~8 min/scheme wall on the RTX 5060 Ti.

**Caveats.** One 33M-class model, one eval window (±0.3 ppl noise); act-order
reordering not used; damper fixed at 1%; fp16-codebook LM9-g64 ppl not
re-measured (estimate only).

**Next.** Compensated *fitted* codebooks (GPTQ + Lloyd on the same pass) —
likely the best point on the curve; re-measure LM9-g64 with fp16 codebooks;
quantize embeddings/lm_head; structure scan (RQ4).

---

## 2026-10-03, session 2 — real model (RQ2 + the printed artifact, RQ6)

**Setup.** TinyStories-33M (roneneldan, GPT-Neo: 4 layers, hidden 768; ~68M
dedup params despite the "33M" name — its ~38.6M-row embedding table + tied
lm_head dominate; see `results/versions.txt`). Body weight matrices (24,
28,311,552 params) quantized per output row (exp4) and per group of 64 input
weights (exp4b); embeddings/wpe/lm_head/biases/norms fp32. Eval: 300×1024
tokens, TinyStories validation, streamed. fp32 baseline **40.663** (exp5
reproduces it exactly — cross-check). exp4 ran without new pre-registration;
claims below are restricted to observed numbers, noise floor ±~0.3 ppl.

**Accounting correction (research finding by itself).** The first exp4 draft
counted only digit-stream bits and omitted the fp32 side info a decoder
needs: row scales (0.03 b/param) and — for fitted schemes — the Lloyd–Max
codebook (9 fp32 centers per row: 0.28 b/param per-row; per group of 64:
4.5 b/param). The corrected "Lloyd–Max 9 levels wins at 3.27 bits" headline
from the first draft collapsed: honest per-row cost is 3.15 b/param (wins vs
uniform-16 but not int4), and honest per-group fp32 cost is 7.3 b/param.
All numbers below include side info; fp16-codebook variants reported as
estimates (2.25 b/param for g64 codebooks) without re-eval.

**exp4/exp4b perplexity vs effective bits (rANS digits + fp32 side info):**

| scheme | eff b/param | ppl |
|---|---|---|
| ternary absmean (PTQ) | 1.61 | 992.3 |
| 8-level, per-row | 2.12 | 81.00 |
| 9-level ninths, per-row | 2.30 | 93.33 |
| 8-level, g64 | 3.02 | 58.21 |
| int4-15, per-row | 3.08 | 45.01 |
| 9-level Lloyd–Max, per-row (codebook counted) | 3.15 | 47.16 |
| 16-level, per-row | 3.18 | 48.04 |
| 9-level ninths, g64 | 3.20 | 54.68 |
| int4-15, g64 | 3.98 | 42.33 |
| 27-level, per-row | 3.97 | 42.93 |
| 9-level Lloyd–Max, g64 (fp16 centers est.) | ~5.02 | 40.53 |
| int8, per-row | 8.02 | 40.63 |

**Reading.**

1. **Group scales heal the MSE↔ppl inversion.** Per-row, uniform 9 *lost* to
   uniform 8 (93.3 vs 81.0 ppl) despite better MSE — transformer outlier
   channels make per-row max-scaling behave badly. With g=64 groups both
   orderings heal: 9-level 54.7 beats 8-level 58.2, and MSE now agrees
   (0.0369 < 0.0484). Methodological takeaway: alphabet-size comparisons
   MUST control the scaling granularity, else MSE misleads.
2. **Fitted 9-level is the standout of the honest table.** Per-row: 3.15
   b/param @ 47.2 ppl dominates uniform-16 (3.18 @ 48.0) on both axes with
   codebook counted. Per-group (fp16 centers, est.): ~5.0 b/param @ 40.53
   ppl — fp32/int8-parity perplexity at ~62% of int8's bits. (Point
   estimates; differences ≤ 0.3 ppl are parity.)
3. **Uniform ninths' honest real-model role:** viable odd-sized middle
   alphabet between 3 and 4 bits, but only with entropy coding (raw 4-bit
   indexing wastes ~0.7+ b/param); per-row it underperforms (outlier
   pathology, fixed by groups).
4. **Post-hoc ternary collapses (992 ppl)** — consistent with BitNet's
   require-training story: the payload of the (information-tight, 1.584
   b/param) ternary-pair → base-9 codec needs QAT (RQ8) to exist. The codec
   layer of this project is ready; the ternary *source material* is not,
   from a standard fp32 model.
5. **exp5 — the printed artifact is real.** 28,311,552 weights printed as
   `0.(4)0.(3)0.(5)…` repeating decimals → re-parsed from disk → **bit-exact
   digits, scales, and dequantized weights across all 24 matrices**, and
   perplexity after fp32-restore + printed-reload identical (93.333 →
   93.333). Size ladder (linear weights): fp32 113.2 MB → printed text
   142.2 MB (gzip 12.9 MB) → compact digits ~57 MB → rANS ~8 MB (2.27
   digit-bits/param + scales). Two numpy pitfalls worth remembering:
   `array_equal` broadcasts mismatched shapes ((768,) vs (768,1) "compared"
   as 768×768) and int64×float32 promotes to float64 — both were caught by
   the round-trip asserts, which is exactly why they exist.
6. **The uniform-9/g64 digit entropy is higher than per-row (2.70 vs 2.27)**
   — group scales make digits use the full alphabet (including extremes),
   improving accuracy at slightly higher digit entropy. Information content
   of the stream and its usefulness are again seen to be different axes.

**Caveats.** One 33M-class model, one eval window; MSE≠ppl (noted twice now);
fp16-codebook ppl not re-measured (estimates); embeddings/lm_head fp32 = over
half the model bytes at this scale — quantizing them is the obvious next
size win and orthogonal to the grid question.

**Next.** (a) re-measure LM9-g64 with actual fp16 codebook + perplexity; (b)
quantize embeddings/lm_head too (the size story is dominated by them at this
scale); (c) Hessian/GPTQ-style ordering as the fitted-competitor check; (d)
RQ4 structure scan (is the real model's digit stream autocorrelated?);
(e) RQ8 ternary QAT so the base-9 pair codec has a competitive payload.

---

## 2026-10-03 — project created, first experiments run

**Setup.** Idea from Rosemary Mercury (`docs/00-idea-origin.md`), formalized into three
claims (`docs/01-hypothesis.md` §1). Predictions P1–P5 pre-registered in
`docs/01 §5` before any experiment ran. Environment: Python 3.14.7, numpy
2.5.3, n = 200,000 Gaussian weights (seed 42). Literature verification pass
completed → `docs/02-related-work.md`; headline of that pass: no prior paper
quantizes/trains on rational grids like multiples of 1/9, and non-power-of-two
alphabets are under-explored.

### Prediction verdicts

| # | prediction | result | verdict |
|---|---|---|---|
| P1 | absmean-ternary of Gaussian is near-uniform (H within 0.1 of log2 3 ≈ 1.585) | H = 1.5833 | **confirmed** (within 0.0016) |
| P2 | 9-level ninths symbols are skewed: H < 3.1699 | H = 1.814 | **confirmed** (far below the bound) |
| P3 | MSE(9-level) strictly between MSE(8-level) and MSE(16-level) | 0.0369 (u16) < 0.1297 (u9) < 0.1698 (u8) | **confirmed** |
| P4 | minimal-rational encoding never Pareto-dominates index coding in the quantization regime | CF/rational: 6.4–22.8 bits/param; index coding: 1.58–4.0 | **confirmed** (no rational win in the ≤ 4 b/param regime) |
| P5 | rANS reaches empirical entropy within ~2% on 9-symbol / ternary streams | ternary 1.584 vs 1.583 (+0.05%); nine 1.831 vs 1.814 (+0.9%) | **confirmed** for the alphabets that matter; int8 (+10.8%) exposes 256-symbol frequency-table granularity — codec TODO |

### Key numbers (results/exp1_entropy_bits.md, exp3_quantization_mse.md)

| scheme | levels | raw bits | entropy | rANS | rel MSE |
|---|---|---|---|---|---|
| int8 symmetric | 255 | 8.000 | 6.715 | 7.440 | 0.000129 |
| int4 symmetric (15) | 15 | 4.000 | 2.564 | 2.594 | 0.042466 |
| ternary absmean | 3 | 2.000 | 1.583 | 1.584 | 0.264413 |
| 9-level ninths grid | 9 | 4.000 | 1.814 | 1.831 | 0.129681 |
| uniform 8 | 8 | 3.000 | 1.648 | — | 0.169834 |
| uniform 12 | 12 | 3.585 | 2.233 | — | 0.068778 |
| uniform 16 | 16 | 4.000 | 2.661 | — | 0.036883 |
| Lloyd–Max 3 | 3 | 1.585 | 1.535 | — | 0.190561 |
| Lloyd–Max 8 | 8 | 3.000 | 2.821 | — | 0.034807 |
| Lloyd–Max 9 | 9 | 3.170 | 2.978 | — | 0.028026 |
| Lloyd–Max 16 | 16 | 4.000 | 3.762 | — | 0.009611 |

(Ternary pair stream decoded as base-9 digits: joint entropy 1.5833 b/param,
rANS 1.5836 b/param — the 2-trits-per-base-9-digit bridge is tight end to
end.)

### Reading

- The **rANS-coded 9-level ninths grid** (1.83 b/param, MSE 0.130) dominates
  raw-coded ternary (2.0 b, MSE 0.264) and raw int4 (4.0 b, MSE 0.042) and
  fills the empty rate-distortion point between entropy-coded ternary and
  entropy-coded int4. On this synthetic curve, "9 levels + entropy coder" is
  a legitimate member of the frontier — the awkward-middle bet (docs/01 §4a)
  has its first evidence.
- **Lloyd–Max 9 levels (MSE 0.0280 @ ~3.0–3.17 bits) beats the uniform
  16-level 4-bit grid (MSE 0.0369 @ 4.0 raw / 2.66 entropy bits)** — fitted
  9 levels buy what 16 uniform levels provide, at fewer raw bits. The uniform
  ninths grid is NOT close to optimal (0.1297 vs 0.0280) — grid *placement*
  matters as much as grid *size*; RQ3 (learned codebooks) is now clearly the
  right next accuracy question.
- **The rational/repeating-number null (P4) held.** Continued-fraction
  encodings cost 6.4 (B=9) to 22.8 (B=4096) bits/param; repeating-decimal
  notation "0.(4)" is 40 bits of ASCII for 3.17 bits of content. The ninths
  structure is a representation identity (verified: k/9 = 0.k̄ in base 10,
  0.k in base 9), not hidden compression. This is a *useful* null: it moves
  the project's value from "lossless trick" to "alphabet-size + grid-design +
  codec joint optimization".

### Surprises

1. **Even-level grids have no zero level** and lose badly for symmetric
   weights: uniform 3 (MSE 0.959) beats uniform 4 (MSE 1.115) with fewer
   levels. Odd alphabet sizes (3, 9, 27) include 0 naturally; 9 = odd =
   ternary-pair = ninths grid stacks three advantages in one size.
2. **Absmean ternary is not MSE-optimal**: 0.264 vs Lloyd–Max 3's 0.1906 —
   BitNet's threshold rule trades quantization MSE for trainability/scaling
   simplicity. Any "9-level beats ternary" accuracy claims must be made
   against *fitted* 3-level, not raw absmean.
3. **1/7 has period 6 in base 10 but period 3 in base 9** (ord tables in
   exp2) — base 9 genuinely shortens some odd-denominator periods; 1/5
   goes 0-terminating → period 2. Period structure depends on the base in
   interesting ways, worth a paragraph in any eventual write-up.
4. int8 entropy 6.715 b/param: even 8-bit streams are skewed enough that
   entropy coding saves ~1.3 b/param (consistent with ZipNN-style results).
5. exp2's theorem check: 400 (fraction, base) reconstructions via the
   geometric-series identity all exact; sqrt(2) control shows no period
   (≤ 2000) with preperiod ≤ 30 in 20,000 digits, base 10 and base 9.

### Caveats

Gaussian stand-in weights (LLM weights are heavier-tailed, group-scaled) —
these are math + codec-floor results, not model-quality claims. MSE is an
accuracy proxy. Frequency tables are shared metadata. rANS state/logic is
roundtrip-asserted but the coder uses a single 256-symbol frequency table
(granularity visible on the int8 row).

### Next (docs/03)

RQ2 real-model PTQ + perplexity vs bits (the claim that matters), RQ3
learned 9-level codebooks vs uniform ninths at matched bits, RQ5 a real
rANS-on-base-9 codec artifact (GGUF-style K9), RQ4 structure scan of a real
checkpoint's digit stream.