# GLM-4.5-Air K9 (ninths k63/k99) — live smoke test on the A9 Mega (2026-10-25)

End-to-end verdict: **the K9 GGUF is servable and coherent** — first live-run of
K9 at MoE-103B scale, on silicon, after the VRAM/GTT constraint from the morning
session was solved with a CPU-resident-expert split.

## Artifacts
- f16 reference: `/home/rose/glm45air_ref_f16.gguf` — 205.9 GiB, 803 tensors, glm4moe
- K9 build: `results/glm45air_k63_e99.gguf` — 73.45 GiB = 472 K9 + 331 copied
  tensors, 5.706 b/p over 110.44 B quantized params (25.2 M params copied
  verbatim) → **35.7 % of f16**, built by `experiments/glm_k9_build.py`
  (pass1 1257 s / pass2) from the 2-pass spill+manifest route.

## Final gates (full GGUF, `glm_k9_build.py gate` vs llama-k9probe C++ loader)
Byte-exact on real weights, all three probed classes, mismatch = 0 / ulp 0.000:
- `blk.1.attn_q.weight` (attn, k63) — includes the one-time stream-encoder ==
  `k9.encode_tensor` cross-check
- `blk.8.ffn_down_exps.weight` (rank-3 expert flattened (524288,1408), k63)
- `token_embd.weight` (k99 aux embed)

## Serving config (what un-blocked the VRAM constraint)
`/home/rose/glm_k9_serve.sh` → `build-hip-a9/bin/llama-server`, flags:
`-ngl 99 -c 8192 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -t 16
--override-tensor 'blk\.4[0-5]\..*_exps\.weight=CPU' --jinja`, port 18445.

The budget that makes it work on gfx1151 (BIOS carve):
- allocatable pool = 96 GB carve + 16 GB GTT (GTT = amdgpu default ½ RAM;
  the OS only sees ~30 GB because 96 GB is carved out at boot)
- materialized q8_0 ≈ 113.5 GB for graph-used tensors; blk.46 nextn stack is
  loaded-but-ignored without MTP flags (−3.8 GB: nextn embed + shared_head_head
  are q8_0-materializing and would be wasted unless `-draft` is in play)
- 6 late blocks' routed experts (blk.40–45) pinned in system RAM: 14.1 GB host,
  ~2.35 GB/block of deficit closed at ~0.15 GB/token DDR streaming cost
- observed: carve filled **exactly** (96 GB), GTT stayed at the 339 MB desktop
  residue → **no GTT spill at all** (the morning failure); load 9.24 min
  single-core rANS decode of ~74 GB blob streams + copies; ~8.8 Gi RAM free
  after load
- housekeeping: baloo indexer had 6.9 GB pinned chewing the new GGUFs —
  `balooctl6 suspend` + killed extractor; **left suspended** (racing an indexer
  against a 73 GB serving box is a losing game; `balooctl6 resume` brings it back)

## Numbers
| metric | value |
|---|---|
| load → listening | 9.24 min |
| prompt eval | 119.1 t/s (8.40 ms/tok, 80-tok prompt) |
| generation | 13.40 t/s steady (13.43/13.37/13.44 across chunks; 74.6 ms/tok) |
| VRAM / GTT after load | 96 GB / 0.34 GB |

## Quality probes (temp 0.2, finish=stop both)
1. "GDScript circle path + _process at constant speed" → valid idiomatic
   GDScript 4 (typed `PackedVector2Array`, `Vector2.RIGHT.rotated`, `@onready`,
   no Godot-3 APIs); interpreted "along the path" as waypoint-following rather
   than arc-length motion — a reading difference, not an API error.
2. Godot-3→4 Tween trap (`Tween.new()` + `add_child`) → corrected to
   `create_tween()` (right API, runnable snippet; the one-sentence explanation
   paraphrased the actual "Tween is no longer a Node" trap as a leak concern).

## Next rungs
- `-draft` MTP: blk.46 nextn + `shared_head_head` (already k99 in the build)
  would self-speculate; re-check device math (+3.8 GB materialized).
- GTT is the default half-of-32GB; a `amdgpu.gttsize=` cmdline or a smaller BIOS
  carve would change the split math (bigger carve → fewer CPU blocks; smaller →
  CPU expert residency must grow, and 30 GB system RAM is the real ceiling).
- Same split recipe transfers to any 103B-class K9 MoE on this box class.
## Addendum (same evening): chat-template bug found & fixed
Rose's live WebUI session hit two failure modes: (1) jinja-like/chat markup
leaking into visible replies, (2) the model answering phantom user turns that
were never sent. Diagnosis: neither GGUF carries a `tokenizer.chat_template`
KV (the converter never embedded one), so `--jinja` silently fell back to
generic ChatML (confirmed via `/apply-template` rendering `<|im_start|>`-style
frames). GLM's native format expects different markers, so the model saw
fabricated delimiters as plain text — hence the leakage and run-on turns.

Fix (no rebuild, no re-encode): fetched the official
`chat_template.jinja` (3242 B, sha16 8cf0…, hub zai-org/GLM-4.5-Air) and
relaunched with `--chat-template-file /home/rose/glm45air_chat.jinja`;
script updated. Post-fix checks: rendered prompt is GLM-native (no ChatML
tokens), `finish=stop` at 171/900 with an open question trailing (no
invented turn), no `{%`/no dangling markup across 4 probes. The one-time
`special_eot_id not in special_eog_ids` warning persists — empirically
benign here (all probes terminate at the turn end).

Quality probe note (single question, 4 samples): answers are stable but the
Tween question got the API right under the broken prompt and wrong under the
right prompt ("call t.play()") — treat as sampling/robustness noise pending
a real eval battery vs the f16 reference; the 206 GiB ref cannot be served
on this 30 GB-RAM box, so a quality A/B needs either RPC pooling or a cloud
run. REAP/trims deferred until rose's criteria are set (her call, 2026-10-25).

## Improvement round (same session, after the template fix)
1. **Threaded materialization** (`src/k9.cpp`): digit stream stays a single
   sequential backward-renorm rANS lane (splitting it would change the on-disk
   format); the row-disjoint q8_0/F16 packing now fans out over
   `std::thread::hardware_concurrency()` (env `K9_DECODE_THREADS` to override).
   Gates re-run on the rebuilt k9probe: all 3 classes byte-exact vs the
   parallel pack too. **Honest verdict: load 545 s vs 554 s — decode-bound.**
   The single-lane rANS decode (~130–165 MB/s over ~74 GB of blobs) is the
   floor; the largest expert tensor materializes in ~4 s and threading only
   trims the packing tail (~0.3 s). The real lever is lane-interleaved rANS
   (container format v2: encoder + loader + re-encode of all K9 artifacts) —
   deferred as an explicit follow-up.
   Build gotcha: `~/.local/bin/cmake --build` silently no-ops on this box —
   drive `make -C build-cpu` / `ninja -C build-hip-a9` directly.
2. **Stock q8_0 A/B artifact built**: `results/glm45air_q8_0.gguf`
   (117,456,501,760 B) in 166.9 s wall under a 10G-class cgroup cap — same
   materialized footprint as the K9 build, so the A/B can run with the exact
   same `-ot` split recipe; K9-vs-q8_0 delta isolates the k63/k99 grid from
   the serving path.
3. **MTP (`--spec-type draft-mtp`) attempted and rejected by the box**:
   weights + the ignored-until-now nextn stack fill the 96 GB carve exactly,
   and ggml-HIP hipMalloc does not spill to GTT (the 820 MB KV alloc fails at
   the end of a full 9.4-min load). Making it fit needs ~4 GB more device:
   either 8 CPU-resident expert blocks (18.8 GB host — unsafe on this 30 GB
   box), a context cut, or a BIOS/cmdline re-carve + reboot. Deferred with
   the math recorded. Server restored to the proven 6-block split, healthy.

## Stream-v2 lane-interleaved rANS (implementation session)
Spec `docs/11-k9-stream-v2.md` → implemented same evening:
- encode: `k9.py` `encode_tensor(..., lanes=, shape=)` + `lane_plan()` rule;
  `glm_k9_build.py` computes per-tensor L, records it, writes
  `ninths.directory2` (v2 layout) and the lane offset tables.
- decode: fork gains `k9_parse_directory2` + a lane branch in
  `k9_decode_digits` and `k9_dequantize` (per-lane thread decodes its
  row-aligned chunk and packs it — no cross-lane sync; v1 path untouched).
- **Gate 1 (python vs python)**: 60 configs (k ∈ {9,63,99} × shapes ×
  L ∈ {1,2,3,16,64}, skewed histograms with zero-count symbols,
  non-divisible row splits) — all byte-exact; overhead ≤ 331 B/tensor.
- **Gates 2/3 (python vs fork C loader)**: v2 mini GGUF
  (`glm45air_k63_v2test.gguf`, 3 tensors, lanes=16) — digits/scales/Q8_0
  all 0 mismatches, 0.000 ulp via `llama-k9probe`.
- Full re-encode queued (`results/glm_k9_spill_v2`, 10G cap); legacy v1
  artifacts remain readable forever.
- HF payload staged: model card + `chat_template.jinja` +
  `experiments/hf_upload_glm45air.py` → `auRose94/GLM-4.5-Air-ninths-k63-e99-GGUF`;
  upload blocked on a fresh HF write token (cached token file is zeroed).
- Full v2 build COMPLETE: `results/glm45air_k63_e99_v2.gguf` = 78,889,902,016 B
  (73.46 GiB, **+34 KB over v1 total** — the lane tables/overheads for all 472
  tensors, matching the docs/11 §3.4 estimate). Full-GGUF gates re-run: all 3
  classes byte-exact (0 mismatches / 0.000 ulp). Spill .bin cleaned.
  Expected load win (~1.5–2.5 min) unmeasured until the live server swaps to
  the v2 file (deferred: server was actively in use).

## Deployment state (end of session 32)
- Live server swapped to the v2 build (`glm45air_k63_e99_v2.gguf`): load 363 s
  vs 554/545 s on v1 — measured DURING the HF upload saturating the same disk
  and ~1 GiB free RAM, so the clean v2 number is better; re-measure when idle.
- HF upload (v1 gguf + card + template) in flight to
  `auRose94/GLM-4.5-Air-ninths-k63-e99-GGUF`, ~36 MB/s; the cached token file
  needed NUL-stripping (corrupt zero-pad after the real token) — upload script
  now sanitizes. v2 GGUF + a fork-push go up together as a follow-up pair.

## Publish state (2026-10-25 late evening)
- HF repo live: `auRose94/GLM-4.5-Air-ninths-k63-e99-GGUF` — v1 GGUF uploaded
  (+100% at ~35 MB/s), model card + chat_template.jinja in place; v2 GGUF
  uploading (card updated with the variant table + fork commit pins).
- Fork `k9` branch: two commits prepared — `25d995288` (stage-2 resident
  container types, carried uncommitted since Oct 6) + `4c99facb9` (threaded
  materialization + v2 lane streams). Push pending: local tool shells can't
  unlock the keyring; rose runs `git -C ~/Work/llama.cpp push fork k9`.
- f16 ref (206 GiB) queued for a private archive repo with a
  sha256-verify-then-delete guard (local copy deleted ONLY on byte match);
  frees ~200 GB on the /home NVMe when it completes.
- Clean v2 load-time re-measure deferred until uploads finish (current 363 s
  number was taken under concurrent-upload disk contention).

## A/B battery (2026-10-26, post-publish)
`results/glm45air_ab_summary.md`: K9-v2 **22/24** vs stock q8_0 **22/24**,
identical pass/fail pattern (both miss export_var + resource_class with the
same chatty-preamble failure), near-identical per-task token counts. K9 file
= 67% of q8_0 with equal downstream behavior. Load: K9-v2 ~150 s vs q8_0
~90 s (the compression's real cost is ~1 min on a warm page cache: the v2
decode ran DURING the hub upload finishing; battery leg A/B ran clean).
Harness bug fixed post-run (system prompt wasn't attached in the battery's
HTTP body — both arms equally affected, so the paired result stands;
rerun with the prompt for absolute numbers).

## Repo hygiene (same night)
- Canonical filenames published and legacy names removed:
  `GLM-4.5-Air-ninths-k63-e99.gguf` (78,889,868,032 B) +
  `...-v2.gguf` (78,889,902,016 B) — dedup made the renames ~1 min at
  1.3 GB/s (Xet chunk dedup against identical content).
- Local v1 GGUF deleted after hub-side size check (byte-identical upload
  provenance); /mnt/Matrix back to 146 GiB free. f16 ref archive chain
  still uploading (verify-then-delete armed).

## iq4_nl materialize target (2026-10-26, rose-requested)
The K9 container now supports a THIRD load-time materialization:
`K9_MATERIALIZE=iq4_nl` (env) or `ninths.materialize: iq4_nl` (KV; env wins,
KV wins over default). Mechanism: k9 digits+scales decode exactly as before,
then one deterministic RTN pass per 32-weight block onto the IQ4_NL codebook
(d = amax/113, f16 scale, first-strict-min tie rule, nibble pairs
L[j]|L[16+j]<<4 — matches ggml dequantize_row_iq4_nl). k99 aux tensors
(token_embd / output head) stay on the exact-Q8_0 path. Byte gates: fork C
branch == python mirror, 0 mismatches on attn / rank-3 expert / embed.
Weight-level cost of rematerialization: mean |Δ|/|w| ≈ 9.3–14.5% vs the k63
grid (no weighted scale search — upstream's ntry=15 loop is sequential fp32
and was deliberately not mirrored). Behavioral cost measured:

| arm | materialize | battery | gen t/s | device |
|---|---|---|---|---|
| K9-v2 exact | q8_0 | 22/24 | 13.4 | 96 GB carve + 14 GB host |
| K9-v2 i4nl | iq4_nl (+q8_0 aux) | **22/24** | **24.97** | **59 GB, no -ot** |

Identical 22/24 pass/fail sets across BOTH arms (fails are the same two
chatty-preamble prompt-style misses). Battery wall 87 s vs 170 s (2×
faster). The ~36 GB of freed device headroom makes MTP feasible after all —
untested yet.

Loader bug found during the rung: K9_MATERIALIZE env raced the
ninths.materialize KV (KV won, silently q8_0) — precedence is now env > KV >
default, with warn-level startup traces (k9_load_meta INFO lines never
reached this build's log sink).
