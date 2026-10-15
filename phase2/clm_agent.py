#!/usr/bin/env python3
"""CLM-style context-file management for the K9 coder agent.

A transfer of arXiv:2609.37725 ("Context Language Models", 2026-09-29) at
zero-shot: the model's context is an editable FILE it owns, not an
append-only transcript. The old harness (chat_k9.py) appends every turn to
the message history, so late turns re-prefill everything ever said. Here the
model FOLDS its history into a small structured file between steps — the
paper's claims (similar accuracy, fewer prefilled tokens) become measurable
on the K9 coder stack.

That is the whole protocol: 1.5B-class models do not have strong native
tool-calling, so the file editor is a JSON op list in a fenced block, with
code payloads carried by file_code (no escaping):

    answer: ```gdscript block```
    ```edits
    [{"op": "file_code", "section": "code", "header": "player.gd"},
     {"op": "set", "section": "tasks", "text": "[x] step 1 verified"}]
    ```

    python3 clm_agent.py --selftest     # store/parser/loop, mock engine, no GPU
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

MAX_OPS = 16
CODE_FENCE = re.compile(r"```(?:gdscript|gd)?\s*\n(.*?)```", re.S)
# Greedy: the edits block may FILE code containing its own ``` fences, so the
# close is the reply's last fence (the skill asks for the answer first, edits
# last). Accepts the ```json spelling too — small models fence their JSON by
# habit; the ops array itself is unambiguous.
EDIT_FENCE = re.compile(r"```(?:edits|json)\s*\n(.*)```", re.S)


# ---------------------------------------------------------------- store ----
class ContextStore:
    """A fixed-section markdown file the model edits.

    Sections render in fixed order; `append`/`replace`/`delete` target text
    inside a section, `set`/`delete_section` act on the section itself.
    `file_code` is filled in by FileAgent (it appends the reply's own last
    ```gdscript block — code never travels inside JSON with escaping). If it
    arrives without text it is a genuine error.
    Everything is validated and applied exactly — a failed op is returned as
    an error string, never a silent no-op.
    """

    def __init__(self, sections: dict | None = None):
        base = dict(plan="", tasks="", code="", notes="")
        over = sections or {}
        self.sections: dict[str, str] = {**base, **over}

    def render(self) -> str:
        out = "# CONTEXT\n"
        for name, text in self.sections.items():
            out += f"\n## {name}\n"
            if text:
                out += text.rstrip() + "\n"
        return out

    def _pos_before(self, section: str, prev: str) -> int:
        i = prev.find(f"\n## {section}\n")
        return -1 if i < 0 else i

    def apply(self, op: dict, prev: str) -> tuple[bool, str]:
        if not isinstance(op, dict):
            return False, f"op is not a dict: {op!r}"
        kind = op.get("op")
        sec = op.get("section")
        if kind not in ("file_code", "set", "append", "replace", "delete",
                        "delete_section"):
            return False, (f"op={kind!r} not in file_code/set/append/replace/"
                           "delete/delete_section")
        if kind == "file_code":
            kind = "append"
        elif kind == "delete_section":
            if not isinstance(sec, str) or not sec:
                return False, "missing 'section'"
            self.sections[sec] = ""
            return True, ""
        if not isinstance(sec, str) or not sec:
            return False, "missing 'section'"
        if kind == "set":
            text = op.get("text")
            if not isinstance(text, str):
                return False, "'set' needs a 'text' string"
            self.sections[sec] = text
            return True, ""
        if kind == "append":
            text = op.get("text")
            if not isinstance(text, str):
                return False, "'append' needs a 'text' string"
            self.sections[sec] = (self.sections.get(sec, "") + "\n\n" + text).strip()
            return True, ""
        if kind == "replace":
            find, text = op.get("find"), op.get("text", "")
            if find is None or not isinstance(find, str):
                return False, "'replace' needs a 'find' string"
            if sec not in self.sections:
                return False, f"no section {sec!r} to replace in"
            if find not in self.sections[sec]:
                return False, f"'find' {find[:48]!r} not present in section {sec!r}"
            self.sections[sec] = self.sections[sec].replace(find, text, 1)
            return True, ""
        if kind == "delete":
            find = op.get("find")
            if not isinstance(find, str) or not find:
                return False, "'delete' needs a 'find' string"
            if sec not in self.sections or find not in self.sections[sec]:
                return False, f"'find' {find[:48]!r} not present in section {sec!r}"
            self.sections[sec] = self.sections[sec].replace(find, "", 1)
            return True, ""
        # delete_section: clears but keeps the heading order stable
        self.sections[sec] = ""
        return True, ""


def parse_edits(reply: str) -> tuple[list | None, str]:
    """Extract the last ```edits block; None + error on absence/malformation."""
    blocks = EDIT_FENCE.findall(reply)
    if not blocks:
        return None, "no ```edits block in the reply"
    try:
        # strict=False: the file's text values may contain raw newlines —
        # the exact leniency a small model needs when writing prose into ops.
        ops = json.loads(blocks[-1].strip(), strict=False)
    except json.JSONDecodeError as e:
        return None, f"```edits block is not valid JSON: {e}"
    if not isinstance(ops, list):
        return None, "```edits must be a JSON array of op objects"
    if len(ops) > MAX_OPS:
        return None, f"{len(ops)} ops exceeds the {MAX_OPS} limit"
    if len(ops) > MAX_OPS:
        return None, f"{len(ops)} ops exceeds the {MAX_OPS} limit"
    return ops, ""


def last_code_block(text: str) -> str | None:
    """The last ```gdscript/```gd fenced block in a reply (None if no code)."""
    blocks = CODE_FENCE.findall(text)
    return blocks[-1] if blocks else None


def extract_code_candidates(reply: str) -> list[str]:
    """Code candidates for scoring, best-likelihood first.

    Small models fence inconsistently: the tuned coder emits bare GDScript
    (often followed by a prose paragraph), generic ones fence mid-reply.
    Rather than guess, produce ordered candidates and let the real parser
    decide: every fenced block, then the whole reply, then cumulative
    blank-line prefixes AND suffixes (code-before-prose and repeated-fixed-
    copy recovery, both seen in smoke replies)."""
    cands = []
    for b in CODE_FENCE.findall(reply):
        cands.append(b.strip("\n"))
    body = reply.strip()
    if body:
        parts = body.split("\n\n")
        for k in range(len(parts), 0, -1):
            cands.append("\n\n".join(parts[:k]))
        for k in range(len(parts), 0, -1):
            cands.append("\n\n".join(parts[-k:]))
    seen, out = set(), []
    for c in cands:
        c = c.strip("\n")
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ------------------------------------------------------------ steering ----
CODE_RULES = ("You are an expert Godot 4 game developer. You write correct, "
              "idiomatic GDScript 4 — never Godot 3 APIs. Every code answer "
              "is one complete ```gdscript block.")


def build_system_prompt(clm: bool) -> str:
    """Shared coding rules; the CLM arm adds only the file-management skill."""
    if not clm:
        return CODE_RULES
    return CODE_RULES + """

You also manage your own persistent context file (the current version is given
to you before each task). The file has sections: plan, tasks, code, notes.

- Every artifact you produce must ALSO be filed into the file (## code) with a
  short header, and its status into ## tasks — the file is your only memory
  between tasks. Keep only the CURRENT version of each artifact.
- Keep the file small and current: fold finished steps into their entries,
  drop anything superseded. New facts that later tasks depend on (references,
  conventions, requirements) go into ## notes and stay there.
- The file's head (## plan, ## tasks) carries the decisions you will need
  later; scratch you no longer need should be pruned at task boundaries.

Answer the user's task first (the full ```gdscript code), then apply your file
updates in ONE ```edits block right at the end — a JSON array of ops:

    ```edits
    [
      {"op": "file_code", "section": "code", "header": "player.gd"},
      {"op": "set",     "section": "tasks", "text": "[x] step 1 verified"},
      {"op": "append",  "section": "notes", "text": "SPEED=500.0 px/s"},
      {"op": "replace", "section": "notes", "find": "SPEED=400", "text": "..."},
      {"op": "delete",  "section": "notes", "find": "..."},
      {"op": "delete_section", "section": "plan"}
    ]
    ```

`file_code` files the LAST ```gdscript block from THIS reply — never copy code
inside a "text" string (escaping is error-prone); code always travels via
file_code. If a previous edits block failed, fix it on the next turn. After
each task is done, prune step-by-step scratch from plan/notes so only
decisions and current artifacts remain."""


# --------------------------------------------------------------- agent ----
class FileAgent:
    """Runs the model against the current file, applies its edits, keeps the
    per-turn metric ledger (tokens, edits, edit tail-distance).

    auto_file=True adds the harness-assist arm: when the model fails the
    edits protocol entirely BUT delivered code in its reply, the harness
    files that code itself (the "delivery-not-storage" working-memory
    pattern — management starts as a runtime service, not a model skill)."""

    def __init__(self, engine, store: ContextStore, clm: bool = True,
                 max_new_tokens: int = 700, auto_file: bool = False,
                 verifier=None):
        self.eng = engine
        self.store = store
        self.clm = clm
        self.auto_file = auto_file
        self.verifier = verifier
        self.max_new_tokens = max_new_tokens
        self.system = build_system_prompt(clm)
        self.history: list = []          # baseline arm only
        self.feedback = ""               # shown to the model on its next turn
        self.turns: list[dict] = []
        self.cum_prompt = 0

    def _auto_file(self, reply: str, label: str, prev: str) -> bool:
        """File the artifact the model delivered, without protocol help.

        Fenced code files as-is; if none and a verifier is available the
        parse-clean candidates are ranked longest-first (the tuned model's
        bare-code + trailing-prose format). Junk never files."""
        if not label or f"### {label}\n" in self.store.sections["code"]:
            return False
        code = last_code_block(reply)
        if code is None and self.verifier is not None:
            cands = extract_code_candidates(reply)
            ok_c = [c for c in cands
                    if self.verifier.check_parse(c, name="autofile").ok]
            code = max(ok_c, key=len) if ok_c else None
        if code is None:
            return False
        ok, _ = self.store.apply(
            {"op": "file_code", "section": "code",
             "text": f"### {label}\n```gdscript\n{code}\n```"}, prev)
        return ok

    def _file_tokens(self, text: str) -> int:
        if self.eng.tok is None:
            return -1
        return len(self.eng.tok(text, add_special_tokens=False)["input_ids"])

    def step(self, user_text: str, label: str = "") -> dict:
        if self.clm:
            body = ("# CONTEXT FILE (your persistent state)\n" + self.store.render()
                    + ("\n\n# NOTE FROM MANAGEMENT\n" + self.feedback
                       if self.feedback else "")
                    + "\n\n# TASK\n" + user_text)
            msgs = [{"role": "system", "content": self.system},
                    {"role": "user", "content": body}]
        else:
            msgs = [{"role": "system", "content": self.system}] + self.history \
                + [{"role": "user", "content": user_text}]
        file_before = self.store.render() if self.clm else ""
        t0 = time.time()
        reply, stats = "", None
        for piece, st in self.eng.stream(msgs, temperature=0,
                                         max_new_tokens=self.max_new_tokens):
            if piece is None:
                stats = st
            else:
                reply += piece
        dt = time.time() - t0
        prompt_tokens = stats["prompt_tokens"]
        if not self.clm:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})

        rec = dict(clm=self.clm, prompt_tokens=prompt_tokens,
                   reply_tokens=stats["reply_tokens"], seconds=stats["seconds"],
                   tok_per_s=stats["tok_per_s"], edits=0, edit_ops="",
                   tail_dist=-1, apply_error="",
                   file_tokens=self._file_tokens(file_before) if self.clm else -1,
                   reply=reply)
        self.cum_prompt += prompt_tokens
        rec["cum_prompt_tokens"] = self.cum_prompt

        if self.clm:
            ops, err = parse_edits(reply)
            if ops is None:
                if self.auto_file and self._auto_file(reply, label, file_before):
                    rec["edits"] = 1
                    rec["edit_ops"] = "auto_file_ok"
                    rec["tail_dist"] = 0
                else:
                    self.feedback = ("Your last ```edits block failed: " + err
                                     + " — re-emit valid JSON ops this turn.")
                    rec["apply_error"] = err
            else:
                prev = file_before
                kinds, tail = [], MAX_OPS * 10**6
                for op in ops:
                    orig_kind = op.get("op", "?") if isinstance(op, dict) else "?"
                    if (isinstance(op, dict) and orig_kind == "file_code"
                            and not isinstance(op.get("text"), str)):
                        code = last_code_block(reply)
                        if code is not None:
                            header = op.get("header") or "artifact"
                            op["op"] = "append"
                            op["text"] = f"### {header}\n```gdscript\n{code}\n```"
                    ok, err = self.store.apply(op, prev)
                    rec["edits"] += 1
                    kinds.append(orig_kind + ("" if ok else "!err"))
                    if ok and isinstance(op, dict):
                        if op.get("op") == "append":
                            tail = min(tail, 0)
                        elif isinstance(op.get("find"), str):
                            i = prev.find(op["find"])
                            if i >= 0:
                                tail = min(tail, len(prev) - i)
                        elif isinstance(op.get("section"), str):
                            i = prev.find("\n## " + op["section"])
                            if i >= 0:
                                tail = min(tail, len(prev) - i)
                    if not ok:
                        rec["apply_error"] = err
                rec["edit_ops"] = ",".join(kinds)
                rec["tail_dist"] = -1 if tail == MAX_OPS * 10**6 else tail
                if rec["apply_error"]:
                    self.feedback = ("Your edits block had an error: "
                                     + rec["apply_error"] + " — fix it next turn.")
                else:
                    self.feedback = ""
                # auto-file assist: ops existed but filed nothing code-bearing
                # (e.g. a well-formed but empty array) while code was delivered
                if self.auto_file and self._auto_file(reply, label, prev):
                    rec["edits"] += 1
                    rec["edit_ops"] = (rec["edit_ops"] + "," if rec["edit_ops"]
                                       else "") + "auto_file_ok"
                    rec["tail_dist"] = 0
            # shared post-apply tail (protocol or auto-file): refresh the file
            # size metric + compaction pressure for a runaway file
            if rec["edits"] or rec["edit_ops"]:
                ntok = self._file_tokens(self.store.render())
                if ntok > 0:
                    rec["file_tokens"] = ntok
                if ntok > 6000:
                    self.feedback += ("Your context file is now " + str(ntok)
                                      + " tokens — compress it (prune superseded "
                                      "notes, keep artifacts).\n")
        self.turns.append(rec)
        return rec


# ------------------------------------------------------------- selftest ----
class MockEngine:
    """Engine-shaped stub: canned replies + fake tok, for offline tests."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.tok = None
        self.i = 0

    def stream(self, msgs, temperature=0.7, max_new_tokens=512):
        txt = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        yield txt, None
        yield None, dict(prompt_tokens=500 + self.i * 100, reply_tokens=42,
                         seconds=0.5, tok_per_s=10.0)


def selftest() -> int:
    bad = []

    def check(name, cond, detail=""):
        print(f"  [{'ok ' if cond else 'BAD'}] {name} {detail}")
        if not cond:
            bad.append(name)

    s = ContextStore()
    prev = s.render()
    ok, _ = s.apply({"op": "append", "section": "code", "text": "var a = 1"}, prev)
    ok2, _ = s.apply({"op": "replace", "section": "code", "find": "var a = 1",
                      "text": "var a = 2"}, prev)
    check("store apply", ok and ok2 and "var a = 2" in s.render())
    ok, err = s.apply({"op": "replace", "section": "code", "find": "nope", "text": ""},
                      prev)
    check("store rejects missing find", not ok and err != "")
    r1 = ("```gdscript\nvar x = 1\n```\n```edits\n"
          '[{"op": "append", "section": "tasks", "text": "[x] done"}]\n```')
    ops, err = parse_edits(r1)
    check("parse valid edits", ops is not None and err == "" and len(ops) == 1)
    # the ops JSON carries a code fence inside a text value (greedy close handles it)
    r2 = ("```edits\n"
          '[{"op": "append", "section": "code",\n'
          '  "text": "```gdscript\\nvar y=1\\n```"},\n'
          ' {"op": "set", "section": "tasks", "text": "[x] ok"}]\n```\n')
    ops, err = parse_edits(r2)
    check("parse edits whose payload has fences",
          ops is not None and len(ops) == 2, err)
    ops, err = parse_edits("```edits\n[{},\n```")
    check("parse flags bad json", ops is None and err != "")
    ops, err = parse_edits("no block at all")
    check("parse flags missing block", ops is None and err != "")

    reply = ("Answer.\n```gdscript\nvar c = 1\n```\n```edits\n"
             '[{"op": "file_code", "section": "code", "header": "thing.gd"},\n'
             ' {"op": "set", "section": "tasks", "text": "[x] done"}]\n```')
    store = ContextStore()
    ag = FileAgent(MockEngine([reply]), store, clm=True)
    rec = ag.step("make a thing")
    check("agent executed file_code",
          "### thing.gd" in store.sections["code"]
          and "var c = 1" in store.sections["code"], ag.feedback)
    check("turn record filled",
          rec["edits"] == 2 and rec["edit_ops"] == "file_code,set"
          and rec["tail_dist"] >= 0)
    bad_reply = "x\n```edits\n[not json]\n```"
    ag2 = FileAgent(MockEngine([bad_reply]), ContextStore(), clm=True)
    rec2 = ag2.step("t")
    check("agent surfaces edit errors as feedback",
          ag2.feedback != "" and rec2["apply_error"] != "")
    base = FileAgent(MockEngine(["hello"]), ContextStore(), clm=False)
    rec3 = base.step("t")
    check("baseline arm accumulates history", len(base.history) == 2
          and rec3["prompt_tokens"] > 0)
    auto = FileAgent(MockEngine(["```gdscript\nvar z = 9\n```"]),
                     ContextStore(), clm=True, auto_file=True)
    rec4 = auto.step("t", label="z.gd")
    check("auto-file arm files the delivered code",
          "z.gd" in auto.store.sections["code"]
          and rec4["edit_ops"] == "auto_file_ok", auto.feedback)
    # lenient JSON: raw newlines inside string values survive
    ops, err = parse_edits('```edits\n[{"op": "set", "section": "notes",\n'
                           '  "text": "line one\nline two"}]\n```')
    check("raw newlines in text values",
          ops is not None and ops[0]["text"] == "line one\nline two", err)
    print("PASS" if not bad else f"FAIL ({len(bad)}): {bad}")
    return 0 if not bad else 1


def main() -> int:
    return selftest()


if __name__ == "__main__":
    sys.exit(main())