#!/usr/bin/env python3
"""Paired A/B eval: append-only chat vs CLM file-managed context (arXiv:2609.37725
transfer to the K9 coder).

One chain of steps builds a small Godot 4 scene; step 3 depends on a binding
convention injected at seeding time only ("the needle"), step 4 reaches back to
the step-1 artifact across the horizon, and step 5 asks for every artifact
restated. The last ```gdscript block of each reply is parse-checked by the real
Godot parser (gdscript_verify) and scored against must/must_not/godot-3 lists.

Arms (same steps, greedy):
  append-only  the current chat harness behavior — history grows every turn
  clm          the agent owns a context file (clm_agent.FileAgent)

    python3 clm_eval.py --selftest            # scoring layer only, no GPU
    python3 clm_eval.py --tag smoke           # 1.5B K9 coder, both arms
    python3 clm_eval.py --seed-tokens 6000    # deeper horizon
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from gdscript_verify import GodotVerifier, OK      # noqa: E402
from clm_agent import (ContextStore, FileAgent,    # noqa: E402
                       build_system_prompt, last_code_block, CODE_FENCE,
                       extract_code_candidates)


def shim_torchaudio() -> None:
    """Unblock the transformers import on this box: torch is the cu13.4 build
    but torchaudio is cu13.3, and torch raises a version-mismatch RuntimeError
    at import, which transformers turns into a fatal model-import failure.
    Our path never touches audio — a stub module (with raising functions for
    the lazily-used entry points) keeps the audio optional import honest."""
    import importlib.machinery
    import types
    try:
        import torchaudio                              # noqa: F401
        return
    except Exception:
        pass
    stub = types.ModuleType("torchaudio")
    # a bare ModuleType has __spec__ = None, and transformers'
    # _is_package_available() raises ValueError on exactly that — give the
    # stub a real package spec so availability checks simply proceed.
    stub.__spec__ = importlib.machinery.ModuleSpec(
        "torchaudio", None, is_package=True)

    def _need(*a, **k):                                # noqa: ARG001
        raise RuntimeError("torchaudio is stubbed (CUDA version mismatch)")

    stub.load = _need
    functional = types.ModuleType("torchaudio.functional")
    functional.resample = _need
    stub.functional = functional
    sys.modules["torchaudio"] = stub

CONVENTIONS = """\
--- PROJECT CONVENTIONS (binding for later tasks) ---
Damage numbers route through `DAMAGE_UNIT = 7.0`.
Every hurtbox class implements `func apply_damage(v: int) -> void` and
multiplies the incoming v by DAMAGE_UNIT internally.
Speeds are specified in px/s. Signals use snake_case past tense
(e.g. coin_collected)."""


# per-step: id must must_not (lowercase substring checks on the code text)
CHAIN = [
    dict(id="player.gd",
         prompt="Write the complete player.gd: a Godot 4 CharacterBody2D that "
                "moves on the horizontal axis with the ui_left/ui_right "
                "actions, applies gravity, and jumps on ui_accept.",
         must=["characterbody2d", "velocity", "move_and_slide"],
         must_not=["kinematicbody2d"]),
    dict(id="coin.gd",
         prompt="Now write coin.gd: an Area2D that detects the player entering "
                "and emits a coin_collected signal when it does.",
         must=["area2d", "coin_collected"],
         must_not=[]),
    dict(id="hurt.gd",
         prompt="Now write hurt.gd: a class_name HurtBox extends Area2D that "
                "handles damage exactly as the project conventions from the "
                "reference requires.",
         must=["hurtbox", "apply_damage", "damage_unit"],
         must_not=[]),
    dict(id="player_take_hit",
         prompt="Modify the player from earlier so it gains `take_hit(v)` that "
                "handles the damage per the project conventions, while keeping "
                "its movement intact. Return the complete updated player.gd.",
         must=["take_hit", "damage_unit", "move_and_slide"],
         must_not=["kinematicbody2d"]),
    dict(id="recall",
         prompt="Restate every artifact you have produced so far (player.gd, "
                "coin.gd, hurt.gd) as three complete ```gdscript blocks, "
                "nothing else.",
         must=[], must_not=[], kind="recall"),
]

G3 = [
    ("kinematic_body", r"\bKinematicBody(2D|3D)?\b"),
    ("spatial", r"\bSpatial\b"),
    ("yield(", r"\byield\s*\("),
    ("bare onready var", r"(?<!@)\bonready\s+var"),
    ("bare export(", r"(?<!@)\bexport\s*\("),
    ("connect 3-arg", r'\.connect\s*\(\s*"[^"]*"\s*,\s*[^,)]+\s*,'),
    ("instance()", r"\.instance\s*\(\s*\)"),
    ("GD.get_fps", r"\bGD\.get_fps\b"),
    ("OS.get_ticks", r"\bOS\.get_ticks\b"),
    ("rand_range(", r"\brand_range\s*\("),
]

import re                                           # noqa: E402  (after __doc__)


def g3_hits(code: str) -> list:
    return [lab for lab, rx in G3 if re.search(rx, code)]


# --------------------------------------------------------------- scoring ----
def score_code(verifier: GodotVerifier, candidates: list[str], step: dict,
               name: str) -> dict:
    """Winner = the parse-clean candidate with the most must-hits (tie: longest).
    This stops a trivia-parse like a bare `extends` line from winning while
    the full, must-satisfying code fails on one declaration. No candidate
    parses = fail, with must-checks reported against the whole reply."""
    if not candidates:
        return dict(step=name, status="no_code", ok=False, detail="")
    musts = step.get("must", [])
    best = None
    best_rank = None
    whole_err = ""
    for code in candidates:
        r = verifier.check_parse(code, name=name)
        if r.status != OK:
            if not whole_err:
                whole_err = "; ".join(e["message"][:60] for e in r.errors[:3])
            continue
        low = code.lower()
        hits = sum(1 for m in musts if m in low)
        rank = (hits, len(code))
        if best is None or rank > best_rank:
            best, best_rank = code, rank
    if best is None:
        whole = candidates[-1] if len(candidates) == 1 else ""
        low = (max(candidates, key=len) if whole == "" else whole).lower()
        missing = [m for m in musts if m not in low]
        return dict(step=name, status="parse_error", ok=False,
                    detail=whole_err
                    + ("; missing " + ",".join(missing) if missing else ""))
    low = best.lower()
    missing = [m for m in musts if m not in low]
    banned = [m for m in step.get("must_not", []) if m in low]
    hits = g3_hits(best)
    if missing or banned or hits:
        detail = ""
        if missing:
            detail += "missing " + ",".join(missing) + "; "
        if banned:
            detail += "banned " + ",".join(banned) + "; "
        if hits:
            detail += "g3 " + ",".join(hits)
        return dict(step=name, status="parse_ok_apis_fail", ok=False,
                    detail=detail.rstrip("; "))
    return dict(step=name, status=OK, ok=True, detail="")


def score_reply(verifier: GodotVerifier, reply: str, step: dict) -> dict:
    if step.get("kind") == "recall":
        blocks = CODE_FENCE.findall(reply)
        if len(blocks) >= 3:
            for b in blocks:
                if not verifier.check_parse(b, name=step["id"]).ok:
                    return dict(step=step["id"], status="recall_block_fail",
                                ok=False, detail="")
            if "move_and_slide" not in blocks[0].lower():
                return dict(step=step["id"], status="recall_player_missing",
                            ok=False, detail="")
            return dict(step=step["id"], status=OK, ok=True, detail="")
        # fenced-less format: markdown "## name" sections; the heading line
        # itself is a name, not GDScript, so parse only what follows it
        segs = [s.strip() for s in re.split(r"(?m)^## ", reply) if s.strip()]
        clean = []
        for s in segs:
            body = s.split("\n", 1)[1].strip("\n`") if "\n" in s else ""
            if body and verifier.check_parse(body, name="seg").ok:
                clean.append(body)
        if len(clean) >= 2 and any("move_and_slide" in s.lower() for s in clean):
            return dict(step=step["id"], status="ok_bare_sections", ok=True,
                        detail="")
        return dict(step=step["id"], status="recall_fail", ok=False,
                    detail=f"fenced={len(blocks)} clean_sects={len(clean)}")
    return score_code(verifier, extract_code_candidates(reply), step,
                      step["id"])


# ---------------------------------------------------------------- filler ----
def load_filler(n_tokens: int, tok, seed: int = 0) -> str:
    """A deterministic real-GDScript slice ~n_tokens long, + the conventions."""
    files = sorted((ROOT / "phase2/data/gds_repo/files").glob("*.txt"))
    files = files[seed:] + files[:seed]
    text = ""
    for f in files:
        t = f.read_text(encoding="utf-8", errors="replace")
        if "class_name" in t:
            text = t
            break
    if not text:
        text = "# (no dump found)\n"
    if tok is not None:
        enc = tok(text, add_special_tokens=False)["input_ids"]
        text = tok.decode(enc[:n_tokens])
    return text + "\n\n" + CONVENTIONS


# ----------------------------------------------------------------- run ------
CSV_COLS = ["arm", "seed_tokens", "step_id", "step_pass", "step_status",
            "prompt_tokens", "reply_tokens", "cum_prompt_tokens",
            "file_tokens", "edits", "edit_ops", "tail_dist", "apply_error",
            "seconds", "tok_per_s"]


def run_eval(preset: str, variant: str | None, arms: list[str], seed_tokens: int,
             max_new_tokens: int, tag: str, seed: int, verbose: bool,
             tuned: list[str], k9: list[str]) -> int:
    shim_torchaudio()
    from chat_k9 import PRESETS, Engine                  # torch lands here
    pre = PRESETS[preset]
    variants = {n: (None if p is None else str(ROOT / p))
                for n, p in pre["variants"].items()}
    for p in tuned:                                      # merged HF dirs
        dd = Path(p).resolve()
        variants[f"fine-tuned ({dd.name})"] = str(dd)
    for p in k9:                                         # extra K9 containers
        variants[Path(p).name.replace(".k9", "")] = str(Path(p).resolve())
    eng_variants = {n: p for n, p in variants.items()
                    if p is None or Path(p).exists()}
    eng = Engine(pre["base"], eng_variants, "cuda", verbose=verbose)
    start = (variant or next((n for n in eng_variants if "fine-tuned" in n), None)
             or next((n for n, p in eng_variants.items() if p is not None), None))
    if start:
        print(f"loading {start} ...", flush=True)
        eng.use(start)
    print("warming up ...", flush=True)
    for _ in eng.stream([{"role": "user", "content": "hi"}], max_new_tokens=1):
        pass

    filler = load_filler(seed_tokens, eng.tok, seed=seed)
    print(f"reference seeded at ~{seed_tokens} tokens", flush=True)

    out_csv = ROOT / f"results/clm_transfer_{tag}.csv"
    out_jsonl = ROOT / f"results/clm_transfer_{tag}.jsonl"
    rows, details = [], []
    with GodotVerifier(workdir=ROOT / "phase2/out/gdverify_clm") as verifier:
        for arm in arms:
            clm = arm.startswith("clm")
            if clm:
                store = ContextStore({"notes": filler,
                                      "plan": "Build a small player/coin/"
                                              "hurtbox scene per the user's "
                                              "steps, one at a time."})
                agent = FileAgent(eng, store, clm=True,
                                  max_new_tokens=max_new_tokens,
                                  auto_file=(arm == "clm-auto"),
                                  verifier=verifier)
                agent.step("Reply with a one-line acknowledgment only.")
            else:
                agent = FileAgent(eng, ContextStore(), clm=False,
                                  max_new_tokens=max_new_tokens)
                agent.step("PROJECT REFERENCE (existing codebase — you will be "
                           "asked to build on it, and its PROJECT CONVENTIONS "
                           "bind later tasks):\n\n" + filler
                           + "\n\nReply with a one-line acknowledgment only.")
            print(f"[{arm}] seeded", flush=True)
            npass = 0
            for step in CHAIN:
                rec = agent.step(step["prompt"], label=step["id"])
                s = score_reply(verifier, rec["reply"], step)
                npass += s["ok"]
                row = {c: "" for c in CSV_COLS}
                row.update(arm=arm, seed_tokens=seed_tokens,
                           step_id=step["id"], step_pass=s["ok"],
                           step_status=s["status"],
                           prompt_tokens=rec["prompt_tokens"],
                           reply_tokens=rec["reply_tokens"],
                           cum_prompt_tokens=rec["cum_prompt_tokens"],
                           file_tokens=rec["file_tokens"],
                           edits=rec["edits"], edit_ops=rec["edit_ops"],
                           tail_dist=rec["tail_dist"],
                           apply_error=bool(rec["apply_error"]),
                           seconds=rec["seconds"], tok_per_s=rec["tok_per_s"])
                rows.append(row)
                tailnote = (f" file={rec['file_tokens']}tok edits={rec['edits']}"
                            f" ({rec['edit_ops']})" if clm else "")
                print(f"  [{arm}] {step['id']:16s} pass={s['ok']}"
                      f" prompt={rec['prompt_tokens']} reply={rec['reply_tokens']}"
                      f"{tailnote} {s['detail']}", flush=True)
                details.append(dict(arm=arm, seed_tokens=seed_tokens, **s,
                                    reply=rec["reply"]))
            tot_prompt = sum(r["prompt_tokens"] for r in rows if r["arm"] == arm)
            print(f"[{arm}] STEPS PASSED {npass}/{len(CHAIN)} · "
                  f"total prompt tokens = {tot_prompt}", flush=True)
            details.append(dict(arm=arm, seed_tokens=seed_tokens,
                                total_prompt_tokens=tot_prompt,
                                steps_passed=npass, steps_total=len(CHAIN)))

    with open(out_csv, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    with open(out_jsonl, "w") as fo:
        for d in details:
            fo.write(json.dumps(d) + "\n")

    print(f"→ {out_csv}\n→ {out_jsonl}")
    return 0


# ------------------------------------------------------------- selftest -----
def eval_selftest() -> int:
    good_player = ("extends CharacterBody2D\n\nvar speed := 300.0\n\n"
                   "func _physics_process(delta: float) -> void:\n"
                   "\tvelocity = Vector2(speed, 0)\n"
                   "\tmove_and_slide()\n")
    bad_player = "extends KinematicBody2D\n\nfunc _ready() -> void:\n\tmove()\n"
    needle = ('class_name HurtBox extends Area2D\n\nconst DAMAGE_UNIT := 7.0\n\n'
              "func apply_damage(v: int) -> void:\n\tvar x = v * DAMAGE_UNIT\n")
    miss_needle = ("class_name HurtBox extends Area2D\n\nvar hp := 10\n\n"
                   "func hit(v: int) -> void:\n\thp -= v\n")
    prose = ("This script defines a Player. It moves with the arrow keys.\n\n"
             "It also handles gravity properly for game feel.\n")
    bare = good_player + "\n\n" + prose          # tuned-model format: no fence
    recall_bare = ("## player.gd\n" + good_player
                   + "\n## coin.gd\nextends Area2D\n\nvar c := 1\n"
                   + "\n## hurt.gd\n" + needle)
    bad = []
    with GodotVerifier(workdir=ROOT / "phase2/out/gdverify_clm_selftest") as v:
        s = score_code(v, [good_player], CHAIN[0], "fenced")
        print(f"  [{'ok ' if s['ok'] else 'BAD'}] fenced player parses: {s['status']}")
        if not s["ok"]:
            bad.append("fenced")
        s = score_code(v, extract_code_candidates(bare), CHAIN[0], "bare+prose")
        print(f"  [{'ok ' if s['ok'] else 'BAD'}] bare code + trailing prose: {s['status']} {s['detail']}")
        if not s["ok"]:
            bad.append("bare+prose")
        s = score_code(v, [bad_player], CHAIN[0], "g3")
        print(f"  [{'ok ' if not s['ok'] else 'BAD'}] kinematicbody2d flagged: {s['detail'][:60]}")
        if s["ok"]:
            bad.append("g3")
        s = score_code(v, [needle], CHAIN[2], "needle")
        print(f"  [{'ok ' if s['ok'] else 'BAD'}] needle conventions met")
        if not s["ok"]:
            bad.append("needle")
        s = score_code(v, [miss_needle], CHAIN[2], "no-needle")
        print(f"  [{'ok ' if not s['ok'] else 'BAD'}] dropped-needle flagged: {s['detail'][:60]}")
        if s["ok"]:
            bad.append("no-needle")
        s = score_reply(v, "```gdscript\n" + good_player + "```\n```gdscript\n"
                        + needle + "```\n```gdscript\n" + needle + "```",
                        CHAIN[4])
        print(f"  [{'ok ' if s['ok'] else 'BAD'}] recall of 3 fenced blocks: {s['status']}")
        if not s["ok"]:
            bad.append("recall")
        s = score_reply(v, recall_bare, CHAIN[4])
        print(f"  [{'ok ' if s['ok'] else 'BAD'}] recall of bare ## sections: {s['status']}")
        if not s["ok"]:
            bad.append("recall-bare")
    print("PASS" if not bad else f"FAIL ({len(bad)}): {bad}")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preset", choices=["1.5b", "7b-tuned"], default="1.5b")
    ap.add_argument("--variant", help="variant name to test (default: the fine-tuned one)")
    ap.add_argument("--tuned", action="append", default=[], metavar="DIR",
                    help="merged HF model dir to add as a variant (repeatable)")
    ap.add_argument("--k9", action="append", default=[], metavar="PATH",
                    help="extra K9 container to add as a variant (repeatable)")
    ap.add_argument("--arms", nargs="+",
                    choices=["both", "append-only", "clm", "clm-auto"],
                    default=["both"])
    ap.add_argument("--seed-tokens", type=int, default=3072)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    if a.selftest:
        return eval_selftest()
    arms = (["append-only", "clm", "clm-auto"] if a.arms == ["both"]
            else a.arms)
    return run_eval(a.preset, a.variant, arms, a.seed_tokens,
                    a.max_new_tokens, a.tag, a.seed, verbose=True,
                    tuned=a.tuned, k9=a.k9)


if __name__ == "__main__":
    sys.exit(main())