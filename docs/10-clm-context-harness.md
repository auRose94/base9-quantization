# 10 — CLM context harness (long-context transfer)

Fact sheet for the context-management transfer into the agentic coder.
Sources: **arXiv:2609.37725 "Context Language Models"** (Shao…Koh, 2026-09-29;
code `facebookresearch/context-language-models`) and
**Xnhyacinth/Awesome-LLM-Long-Context-Modeling**. Written 2026-10-15.

## The idea being transferred

CLMs treat the model's context as an editable *file* it owns (append or
rewrite — the paper's version uses Bash), instead of an append-only
transcript. Zero-shot (no training), that beats hand-engineered context
managers: +11.4% on BrowseComp-Plus at −21.5% FLOPs, +65% improvement at
equal compute on a 24-hour multi-repo agent task. Their cost law: a
mid-context edit invalidates KV from the edit onward (edit-at-start ≈ 7.7× an
append-only turn) ⇒ steer tail edits and boundaary compaction.

## What exists here now

`phase2/clm_agent.py` — the transfer core:

- `ContextStore`: a fixed-section markdown context file (`plan`, `tasks`,
  `code`, `notes`); validated ops `set / append / replace / delete /
  delete_section / file_code`; failed ops are reported back to the model
  ("management feedback"), never silently dropped.
- `FileAgent`: one loop over a `chat_k9.Engine` — each turn the model sees
  the current file + the task; its reply may end with ONE ```edits block
  (JSON array; lenient — raw newlines allowed in `text` values). Code is
  never carried inside JSON: `file_code` files the reply's own last
  ```gdscript block (the honest minimal analog of the paper's Bash edits).
- Steering (the paper's skill class, adapted): answer first, file the
  artifact, keep references/conventions in `## notes`, prune scratch at task
  boundaries. Compaction pressure is a system note once the file exceeds
  ~6k tokens.
- Per-turn metric ledger: prompt/reply tokens, cumulative prompt tokens
  (the FLOPs proxy), file token trajectory, edit ops, edit tail-distance —
  the paper's edit-economy quantities, on our models.

`phase2/clm_eval.py` — the paired A/B harness: the `append-only` arm is the
status-quo chat behavior (history appends every turn); the `clm` arm owns the
file. Same 5-step chain, greedy, verified mechanically by the real Godot 4
parser + api/godot-3 must-lists. Depth comes from a real GDScript dump
(`phase2/data/gds_repo/files`) seeded to N tokens with a binding PROJECT
CONVENTIONS tail; step 3 and 4 can only be done right when that seeded
context survived — so the eval measures *quality at depth* (the repo's
white-box agent study warns ppl parity ≠ agent parity) and *token economy*
together. Artifacts: `results/clm_transfer_<tag>.csv` (turn ledger) +
`.jsonl` (replies + verdicts + totals).

Selftests are offline and GPU-free: `phase2/clm_agent.py --selftest` (11
checks, mock engine) and `phase2/clm_eval.py --selftest` (5 checks through
the real Godot verifier).

## Honest limits

- This rung re-renders the file every turn; no cross-turn KV cache is
  claimed. llama.cpp-level prefix caching (and a Suffix-Cache-Reuse-class
  patch, as in the paper) is the fork-side follow-up — and compressed
  (digit-KV) suffixes would make such reuse cheaper.
- The needle convention check is substring-based, not semantic.
- `DAMAGE_UNIT`-style conventions are synthetic; the dump slice is real
  corpus text.
- One chain, greedy, no multi-draw yet — verdicts at this stage are smoke
  signals, not conclusions (the established noise-floor discipline applies:
  see `experiments/eval_harness.py` + the draws-3 eval history).

## Smoke verdict (2026-10-15, tuned 1.5B coder, seed 3072)

Mechanism green: the file evolves through the protocol or the auto-file
assist, junk filings are refused by the Godot verdict, the ledger records
the file trajectory (3184→3351 tokens) and edit positions. Two model-level
findings: the tuned 1.5B never emits the ops block zero-shot (the generic
1.5B does — the paper's RL rung is the real fix), and its greedy
trajectories on chain prompts flip between good code and degenerate loops
between runs (bf16/sdpa nondeterminism) — so quality claims need
multi-draw. Token economy is within ~8% at 5 steps while compaction barely
ran; the file arm's advantage is expected to open up at real compaction or
8k/16k seeds.

## Next rungs from these sources

1. FA dispatch (already NEXT in the KV-quant sweep): K9_4/K9_6 KV types into
   ggml-hip's flash-attention quant table.
2. This harness at 7B/14B tuned coders, multi-draw stats, deeper seeds
   (8k/16k), multi-draw paired stats.
3. OctoLong-style cross-repo packed long-context mid-train (corpus dumps
   already exist).
4. rANS-KV cold tier behind a sparse-attention rung (OasisKV / hierarchical
   sparse attention are the design references).

## Agent merge verdict (2026-10-26, GLM-4.5-Air K9-v2 live)

The harness's context file moved into the real agent loop:
`experiments/agent_k9.py --context-policy clm` (default). Tool payloads over
600 B are FILED into `## data` with a locator stub left in the history
instead of being truncated; the current file re-renders as the tail message
every turn (the history prefix stays byte-stable — the old stub policy
mutated old tool messages in place, invalidating llama-server's LCP prefix
cache at every turn); the model edits the file itself via a local `ctx_edit`
tool carrying the same ops (no fenced-JSON fragility — this model size
tool-calls natively, the 1.5B edits-block protocol is not needed). File
pressure note at ~6k chars; hard budget tombstones oldest `## data`
entries. `--context-policy stub` keeps the pre-merge behavior as the
baseline arm. Battery: `experiments/agent_k9_battery.py` — seeded repo,
binding-rules needle at the BOTTOM of docs/conventions.md, agent writes
enemy/boss.gd, harness verifies the file from disk (6 checks, no model
self-report). 2 draws/arm on the live K9-v2+MTP server, 16-turn budget:

| arm          | success | steps     | cum prompt tok | resident tool payload | file chars |
|--------------|---------|-----------|----------------|-----------------------|------------|
| stub (pre)   | 2/2     | 16.0 (cap)| 62.7k avg      | 3.8–14.3 kB           | —          |
| clm (filed)  | 2/2     | 11.0      | 78.9k avg      | 2.7–3.5 kB            | 4.6–5.2k   |

Small-scale verdict, honestly: **quality parity** — the needle survives both
arms because the workspace is small and the stub arm simply re-reads
conventions.md before writing. **Step economy favors clm** (11 vs 16 turns:
filed data is already in view; no re-read loop). **Token economy favors stub
at this scale** — the per-turn tail re-render plus the model's un-pruned
filing cost more than in-place truncation saved at ~14 kB payload depth, and
curation never engaged (ctx_edit used once across the battery; the file
never crossed the pressure threshold). This matches the tuned-rung finding:
economy crossover needs payloads big enough that stubbing must either
truncate the needle away or re-read under lean-ctx's repeated-read cache
stub. Next: a deep-payload battery variant (needle past 50 kB of decoy
reads) before any real-session claims. Pre-fix smoke rows (lean-ctx path
sandbox burned turns on /tmp workspaces) are in clm_agent_smoke1.csv for
the record; harness-debug only.