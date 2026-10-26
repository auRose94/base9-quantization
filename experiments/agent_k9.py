#!/usr/bin/env python3
"""agent_k9 — tool-using agent over the local ninths llama-server.

Speaks OpenAI chat-completions to http://127.0.0.1:<port> (llama-server with
--jinja), executes tool calls through the MCP client (mcp_client.py), and
feeds results back until the model answers in prose or hits --max-turns.
MCP errors are returned to the model as tool output so it can steer — the
session-31 rule: on tool errors change args or switch tool kind.

Context policy — the session-32 CLM merge (arXiv:2609.37725 transferred to a
tool-calling agent, replacing the phase2 fenced-edits protocol):

  clm (default)  The model owns a persistent context file (phase2 ContextStore,
                 ops set/append/replace/delete/delete_section). The harness
                 FILES every tool payload over ~600 B into `## data` and puts
                 a locator stub in the message history; the current file is
                 re-rendered as the LAST message each turn. History prefix
                 bytes never change mid-list (stubbing mutated old tool
                 messages in place, invalidating llama-server's LCP prefix
                 cache every time) — only the tail file message changes. The
                 model curates the file with the local `ctx_edit` tool and
                 gets management feedback on file pressure.
  stub           The pre-merge behavior: tool payloads older than the last
                 few rounds truncated to a 400-char head in place. Baseline
                 for A/B.

    agent_k9.py --task "..." [--tools ctx_read,ctx_shell,...]
               [--port 18445] [--max-turns 12] [--context-policy clm|stub]

Library use: run_agent(...) -> AgentK9 (exposes .msgs, .ledger, .store).
    python3 agent_k9.py --selftest     # offline, mock server/MCP, no GPU
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2"))
from clm_agent import ContextStore, MAX_OPS  # noqa: E402


# ------------------------------------------------------------- settings ----
FILE_PAYLOAD_MIN = 600       # bytes: tool outputs below this stay in history
STUB_KEEP_RECENT = 3         # stub arm: tool messages this recent stay intact
STUB_HEAD = 400
PRESSURE_TOKENS = 6000       # steering note when the file grows past this
FILE_BUDGET_TOKENS = 14000   # hard cap: oldest ## data entries get tombstoned
EST_CHARS_PER_TOKEN = 4      # chars->token estimate for ledger/file budget

FILE_PREAMBLE = "# CONTEXT FILE (current state of your persistent memory)\n"
STUB_NOTE = ("\n[elided: filed to your context file as data entry "
             "`{eid}` — {nb} B; consult the file, do not re-run the tool]")
TOMBSTONE = "[pruned {nb} B — was {eid}: {head}]"


def chat(port: int, messages: list, tools: list | None, thinking: bool,
         timeout: float = 900):
    body = {"messages": messages, "temperature": 0.2, "stream": False}
    if tools:
        body["tools"] = tools
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


# ------------------------------------------------------------ ctx_edit ----
CTX_EDIT_SPEC = {"type": "function", "function": {
    "name": "ctx_edit",
    "description": (
        "Edit your persistent context file (the '# CONTEXT FILE' message). "
        "sections: plan, tasks, code, notes, data. op: set | append | "
        "replace | delete | delete_section. replace/delete target the first "
        "occurrence of find inside section. Failed ops come back as errors "
        "— fix them on a later turn."),
    "parameters": {"type": "object", "properties": {
        "ops": {"type": "array", "description": "edit ops, applied in order",
                "items": {"type": "object",
                          "properties": {"op": {"type": "string"},
                                         "section": {"type": "string"},
                                         "text": {"type": "string"},
                                         "find": {"type": "string"}}}}}}}}


class CtxEditor:
    """Local (non-MCP) tool: applies the model's ops to the ContextStore.

    Mirrors clm_agent's op semantics exactly; failed ops are returned as
    tool output (management feedback), never applied silently."""

    def __init__(self, store: ContextStore):
        self.store = store

    def call(self, arguments: dict) -> str:
        ops = arguments.get("ops")
        if isinstance(ops, str):
            try:
                ops = json.loads(ops)
            except json.JSONDecodeError:
                return "ctx_edit error: 'ops' string is not valid JSON"
        if not isinstance(ops, list) or not ops:
            return ("ctx_edit error: 'ops' must be a non-empty list of op "
                    "objects like {\"op\":\"append\",\"section\":\"notes\","
                    "\"text\":\"...\"}")
        if len(ops) > MAX_OPS:
            return f"ctx_edit error: {len(ops)} ops exceeds the {MAX_OPS} limit"
        out, applied = [], 0
        prev = self.store.render()
        for op in ops:
            ok, err = self.store.apply(op, prev)
            if ok:
                applied += 1
                out.append(f"ok: {op.get('op')}({op.get('section')})")
            else:
                out.append(f"ERROR on {op.get('op')}({op.get('section')}): {err}")
        return (f"ctx_edit applied {applied}/{len(ops)} ops:\n" + "\n".join(out))


# -------------------------------------------------------------- agent ----
DEFAULT_SYSTEM_CLM = (
    "You are an autonomous coding agent. Inspect before you write: use "
    "tools to read existing files, then change them. Verify your work by "
    "reading back what you wrote. When a tool call fails, change the "
    "arguments or switch to a different tool instead of repeating it. "
    "Answer in prose when done.\n\n"
    "You own a persistent context file, re-given to you each turn as the "
    "last message ('# CONTEXT FILE ...'). Every sizable tool output is "
    "filed there automatically under ## data (with a stub left in this "
    "conversation) — consult the file instead of re-reading the same file "
    "or re-running the same tool. You maintain the rest of the file "
    "yourself with the ctx_edit tool: keep requirements, conventions and "
    "decisions in ## notes, your working plan in ## plan, and PRUNE "
    "superseded ## data entries and finished scratch at task boundaries — "
    "the file is your only memory between turns, so keep it current and "
    "small. Code you deliver must also be filed into ## code.")


class AgentK9:
    def __init__(self, port: int, system: str, tools: list, mcp,
                 policy: str = "clm", max_turns: int = 12, thinking: bool = False):
        self.port = port
        self.policy = policy
        self.max_turns = max_turns
        self.thinking = thinking
        self.mcp = mcp
        self.tools = list(tools) + ([CTX_EDIT_SPEC] if policy == "clm" else [])
        self.system = system
        self.store = ContextStore({"data": ""}) if policy == "clm" else None
        self.ctx = CtxEditor(self.store) if policy == "clm" else None
        self.msgs: list = []
        self.ledger: list = []
        self.gen_tokens = 0
        self.local_tools = {"ctx_edit"} if policy == "clm" else set()
        self._data_seq = 0
        self.ctx_edits = 0

    # ------------------------------------------------------ file policy --
    def _file_est_tokens(self) -> int:
        return len(self.store.render()) // EST_CHARS_PER_TOKEN

    def _file_render(self, note: str = "") -> str:
        body = self.store.render()
        est = self._file_est_tokens()
        if est > PRESSURE_TOKENS and "prune" not in note:
            note = (note or "") + (f"\n# NOTE FROM MANAGEMENT\nYour context file "
                                   f"is ~{est} tokens — prune superseded ## data "
                                   "entries and finished scratch (use ctx_edit).")
        return FILE_PREAMBLE + body + (note + "\n" if note else "")

    def _file_tool(self, name: str, args_head: str, out: str) -> str | None:
        """File a big tool payload; returns the locator stub for the history."""
        self._data_seq += 1
        eid = f"tool-{self._data_seq}"
        entry = f"### {eid}: {name}({args_head})\n```\n{out}\n```\n"
        prev = self.store.render()
        ok, _err = self.store.apply({"op": "append", "section": "data",
                                     "text": entry}, prev)
        if not ok:                                  # never lose the payload
            return None
        self._enforce_budget()
        return out[:240].rstrip() + STUB_NOTE.format(eid=eid, nb=len(out))

    def _enforce_budget(self) -> int:
        """Tombstone oldest ## data entries while over the hard cap."""
        pruned = 0
        while self._file_est_tokens() > FILE_BUDGET_TOKENS:
            data = self.store.sections.get("data", "")
            i = data.find("### tool-")
            if i < 0:
                break
            j = data.find("### tool-", i + 1)
            end = j if j >= 0 else len(data)
            chunk = data[i:end]
            m = re.match(r"### (tool-\d+): (.{0,80})", chunk)
            body = chunk[((chunk.find("\n```\n") + 4) if "\n```\n" in chunk else 0):]
            body = body[:-5] if body.endswith("```\n") else body
            nb = len(body)
            tomb = (TOMBSTONE.format(nb=nb, eid=m.group(1) if m else "?",
                                     head=(m.group(2) if m else "")) + "\n")
            self.store.sections["data"] = (tomb + data[end:]).strip()
            pruned += 1
        return pruned

    def _refresh_file_message(self, note: str = "") -> None:
        if self.msgs and self.msgs[-1]["role"] == "user" \
                and self.msgs[-1]["content"].startswith(FILE_PREAMBLE):
            self.msgs.pop()
        self.msgs.append({"role": "user", "content": self._file_render(note)})

    def _stub_old_outputs(self) -> None:
        """stub arm: truncate old tool payloads in place (baseline behavior)."""
        n_tool = sum(1 for m in self.msgs if m.get("role") == "tool")
        n_new = 0
        for m in self.msgs:
            if m.get("role") != "tool":
                continue
            n_new += 1
            if n_new <= n_tool - STUB_KEEP_RECENT and \
                    len(m["content"]) > STUB_HEAD:
                m["content"] = (m["content"][:STUB_HEAD]
                                + "\n[elided: full output on disk; re-read if needed]")

    # ---------------------------------------------------------- driven ----
    def run(self, task: str) -> str | None:
        self.msgs = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": task},
        ]
        if self.policy == "clm":
            self._refresh_file_message()
        t0 = time.time()
        for turn in range(self.max_turns):
            r = chat(self.port, self.msgs, self.tools, self.thinking)
            msg = r["choices"][0]["message"]
            usage = r.get("usage", {})
            self.gen_tokens += usage.get("completion_tokens", 0)
            rec = {"turn": turn + 1, "policy": self.policy,
                   "prompt_tokens": usage.get("prompt_tokens", 0),
                   "file_est_tokens": self._file_est_tokens()
                   if self.policy == "clm" else -1,
                   "data_chars": len(self.store.sections.get("data", ""))
                   if self.policy == "clm" else -1}
            self.ledger.append(rec)
            self.msgs.append(msg)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                return (msg.get("content") or "").strip() or None
            for tc in tcs:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": str(tc["function"]["arguments"])[:200]}
                head = json.dumps(args)[:110]
                print(f"[turn {turn + 1}] {fn}({head}{'}' if len(head) <= 110 else ' …'})")
                if fn in self.local_tools:
                    self.ctx_edits += 1
                    out = self.ctx.call(args)
                else:
                    try:
                        out = self.mcp.call(fn, args)
                    except Exception as e:      # noqa: BLE001 — steer on ANY failure
                        out, args = f"error: {e}", {"_note": "switch args/tool"}
                if self.policy == "clm":
                    stub = self._file_tool(fn, head, out) \
                        if len(out) > FILE_PAYLOAD_MIN else None
                    content = stub if stub is not None else out
                else:
                    content = out[:20000]
                self.msgs.append({"role": "tool",
                                  "tool_call_id": tc.get("id") or fn,
                                  "content": content})
            if self.policy == "stub":
                self._stub_old_outputs()
            else:
                self._refresh_file_message()
        return None

    def resident_tool_bytes(self) -> int:
        return sum(len(m.get("content", "")) for m in self.msgs
                   if m.get("role") == "tool")


def run_agent(task: str, port: int = 18445, policy: str = "clm",
              max_turns: int = 12, mcp_spec: str = "ctx=lean-ctx mcp",
              allow: str = "", system: str | None = None,
              thinking: bool = False, mcp=None, verbose: bool = True):
    """Programmatic entry: returns the finished AgentK9 (.answer is None on
    max-turns). `mcp` may inject a pre-connected MCPGroup (tests)."""
    import mcp_client
    if mcp is None:
        servers: dict[str, list[str]] = {}
        for part in mcp_spec.split(","):
            if "=" in part:
                name, cmd = part.split("=", 1)
                servers[name.strip()] = cmd.strip().split()
        mcp = mcp_client.MCPGroup()
        mcp.connect(servers)
        owned = True
    else:
        owned = False
    tools = mcp.tools()
    if allow:
        keep = {x.strip() for x in allow.split(",")}
        tools = [t for t in tools if t["function"]["name"] in keep]
    ag = AgentK9(port, system or (DEFAULT_SYSTEM_CLM if policy == "clm" else
                                  DEFAULT_SYSTEM_BASE),
                 tools, mcp, policy=policy, max_turns=max_turns,
                 thinking=thinking)
    if verbose:
        print(f"tools: {', '.join(t['function']['name'] for t in tools)}")
    t0 = time.time()
    answer = ag.run(task)
    if verbose:
        print()
        if answer:
            print("=== FINAL ===\n" + answer)
        else:
            print(f"[hit --max-turns {max_turns} without a final answer]")
        print(f"[{time.time() - t0:.0f}s wall, {ag.gen_tokens} gen tokens]")
    ag.answer = answer
    if owned:
        mcp.close()
    return ag


DEFAULT_SYSTEM_BASE = (
    "You are an autonomous coding agent. Inspect before you write: use "
    "tools to read existing files, then change them. Verify your work by "
    "reading back what you wrote. When a tool call fails, change the "
    "arguments or switch to a different tool instead of repeating it. "
    "Answer in prose when done.")


# ------------------------------------------------------------ selftest ----
class MockMCP:
    tools_payload = [
        {"type": "function", "function": {"name": "ctx_read", "description": "r",
         "parameters": {"type": "object", "properties": {}}}}]

    def tools(self):
        return self.tools_payload

    def call(self, name, args, timeout=90):
        return (f"### conventions of {args.get('path', '?')}\n" * 30
                + "RULE-needle-at-bottom-value-XYZ")


def selftest() -> int:
    bad = []

    def check(name, cond, detail=""):
        print(f"  [{'ok ' if cond else 'BAD'}] {name} {detail}")
        if not cond:
            bad.append(name)

    ag_holder = {}

    def mock_chat(port, messages, tools, thinking, timeout=900):
        ag_holder["msgs"] = list(messages)
        started = any(m.get("role") in ("assistant", "tool") for m in messages)
        if not started:   # first call: file message present, do a read
            assert messages[-1]["content"].startswith(FILE_PREAMBLE), \
                "file message must be the tail message"
            return {"choices": [{"message": {"role": "assistant", "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "ctx_read",
                    "arguments": json.dumps({"path": "docs/conventions.md"})}}]}}],
                    "usage": {"prompt_tokens": 900 + len(messages) * 10,
                              "completion_tokens": 30}}
        # second call: verify filing, prune check, then answer
        tool_msg = [m for m in messages if m.get("role") == "tool"]
        assert tool_msg and "data entry `tool-1`" in tool_msg[-1]["content"], \
            "big payload must be filed with a locator stub"
        assert "RULE-needle" in messages[-1]["content"], \
            "filed payload must be visible in the tail file"
        return {"choices": [{"message": {"role": "assistant", "content": "done."}}],
                "usage": {"prompt_tokens": 1600, "completion_tokens": 5}}

    old_chat = globals()["chat"]
    globals()["chat"] = mock_chat
    try:
        ag = AgentK9(0, DEFAULT_SYSTEM_CLM, [], MockMCP(), policy="clm")
        ag.run("write a thing")
        check("payload filed to store ## data",
              "RULE-needle" in ag.store.sections["data"])
        check("locator stub in history",
              any("data entry `tool-1`" in m.get("content", "")
                  for m in ag.msgs if m.get("role") == "tool"))
        check("tail file message carries the filing",
              any(m.get("role") == "user"
                  and m["content"].startswith(FILE_PREAMBLE)
                  and "RULE-needle" in m["content"] for m in ag.msgs))
        check("ledger rows + tokens", ag.ledger
              and all("prompt_tokens" in r for r in ag.ledger), str(ag.ledger))
        # ctx editor ops
        ed = CtxEditor(ag.store)
        out = ed.call({"ops": [{"op": "append", "section": "notes",
                                "text": "convention A"}]})
        check("ctx_edit appends", "ok: append(notes)" in out
              and "convention A" in ag.store.render())
        out = ed.call({"ops": [{"op": "append", "section": "notes",
                                "text": "b"}, {"op": "replace",
                                "section": "notes", "find": "absent-X",
                                "text": "y"}]})
        check("ctx_edit surfaces per-op errors", "ERROR on replace" in out)
        # budget: giant entries get tombstoned, oldest first
        ag2 = AgentK9(0, "s", [], MockMCP(), policy="clm")
        for i in range(3):
            ag2._file_tool("ctx_read", f"p{i}", "x" * 70_000)
        check("budget tombstones oldest first",
              ag2._file_est_tokens() <= FILE_BUDGET_TOKENS + 100
              and "[pruned" in ag2.store.sections["data"]
              and "was tool-" in ag2.store.sections["data"]
              and ("x" * 1000) not in ag2.store.sections["data"],
              ag2.store.sections["data"][:140])
        # stub arm keeps old behavior + history truncation
        ag3 = AgentK9(0, DEFAULT_SYSTEM_BASE, [], MockMCP(), policy="stub")

        def mock_chat_stub(port, messages, tools, thinking, timeout=900):
            tools_seen = sum(1 for m in messages if m.get("role") == "tool")
            if tools_seen < 4:
                return {"choices": [{"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"id": f"c{tools_seen}",
                        "function": {"name": "ctx_read",
                        "arguments": json.dumps({"path": f"a{tools_seen}.md"})}}]}}],
                        "usage": {"prompt_tokens": 400, "completion_tokens": 20}}
            return {"choices": [{"message": {"role": "assistant",
                    "content": "ok."}}], "usage": {"prompt_tokens": 500,
                    "completion_tokens": 3}}
        globals()["chat"] = mock_chat_stub
        ag3.run("t")
        tools = [m for m in ag3.msgs if m.get("role") == "tool"]
        elision = "[elided: full output on disk; re-read if needed]"
        check("stub arm truncates old in place, recent intact",
              len(tools) == 4
              and tools[0]["content"].endswith(elision)
              and not tools[1]["content"].endswith(elision)
              and not tools[-1]["content"].endswith(elision))
        check("stub arm has no file message",
              not any(m.get("role") == "user"
                      and m["content"].startswith(FILE_PREAMBLE)
                      for m in ag3.msgs))
    finally:
        globals()["chat"] = old_chat

    print("PASS" if not bad else f"FAIL ({len(bad)}): {bad}")
    return 0 if not bad else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18445)
    ap.add_argument("--task", default=None)
    ap.add_argument("--mcp", default="ctx=lean-ctx mcp",
                    help="name=command[, name2=command2] MCP servers")
    ap.add_argument("--tools", default="",
                    help="comma-separated allow-list (empty = all MCP tools)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--context-policy", choices=["clm", "stub"], default="clm")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.task:
        ap.error("--task is required (or --selftest)")
    run_agent(a.task, port=a.port, policy=a.context_policy,
              max_turns=a.max_turns, mcp_spec=a.mcp, allow=a.tools,
              thinking=a.thinking)


if __name__ == "__main__":
    main()