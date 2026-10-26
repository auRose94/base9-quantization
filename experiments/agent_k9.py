#!/usr/bin/env python3
"""agent_k9 — tool-using agent over the local ninths llama-server.

Speaks OpenAI chat-completions to http://127.0.0.1:<port> (llama-server with
--jinja), executes tool calls through the MCP client (mcp_client.py), and
feeds results back until the model answers in prose or hits --max-turns.
MCP errors are returned to the model as tool output so it can steer — the
session-31 rule: on tool errors change args or switch tool kind.

Usage:
    agent_k9.py --task "..." [--tools ctx_read,ctx_shell,...]
               [--port 18445] [--max-turns 12] [--thinking]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18445)
    ap.add_argument("--task", required=True)
    ap.add_argument("--mcp", default="ctx=lean-ctx mcp",
                    help="name=command[, name2=command2] MCP servers")
    ap.add_argument("--tools", default="", help="comma-separated allow-list")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--system", default=(
        "You are an autonomous coding agent. Inspect before you write: use "
        "tools to read existing files, then change them. Verify your work by "
        "reading back what you wrote. Answer in prose when done."))
    ap.add_argument("--thinking", action="store_true",
                    help="leave the model's thinking channel on")
    a = ap.parse_args()

    import mcp_client
    mcp = mcp_client.MCPGroup()
    servers: dict[str, list[str]] = {}
    for part in a.mcp.split(","):
        if "=" in part:
            name, cmd = part.split("=", 1)
            servers[name.strip()] = cmd.strip().split()
    mcp.connect(servers)
    tools = mcp.tools()
    if a.tools:
        keep = {x.strip() for x in a.tools.split(",")}
        tools = [t for t in tools if t["function"]["name"] in keep]
    print(f"tools: {', '.join(t['function']['name'] for t in tools)}")

    msgs = [{"role": "system", "content": a.system},
            {"role": "user", "content": a.task}]
    t0 = time.time()
    gen_tokens = 0
    for turn in range(a.max_turns):
        # context economy: tool payloads older than the last 2 rounds get
        # stubbed (lean-ctx style eviction — outputs still exist on disk)
        n_tool_rounds = 0
        out_off = 0
        for i, m in enumerate(msgs):
            if m.get("role") == "tool":
                if n_tool_rounds >= 0:
                    pass
            if m.get("role") == "tool":
                out_off += 1
                if out_off <= len(msgs) - 6 and len(m["content"]) > 600:
                    m["content"] = (m["content"][:400]
                                    + "\n[elided: full output on disk; re-read if needed]")
        r = chat(a.port, msgs, tools, a.thinking)
        msg = r["choices"][0]["message"]
        gen_tokens += r["usage"]["completion_tokens"]
        msgs.append(msg)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            print("\n=== FINAL ===")
            print((msg.get("content") or "").strip() or "(empty answer)")
            break
        for tc in tcs:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc["function"]["arguments"][:200]}
            head = json.dumps(args)
            print(f"[turn {turn + 1}] {fn}({head[:110]}{'}' if len(head) <= 110 else ' …'})")
            try:
                out = mcp.call(fn, args)
                status = "ok"
            except Exception as e:  # noqa: BLE001 — steer on ANY failure
                out, status = f"error: {e}", "ERR"
            snippet = out if len(out) <= 200 else out[:120] + f" … (+{len(out) - 120}B)"
            print(f"  -> {status}: {snippet!r}")
            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or fn,
                         "content": out[:20000]})
    else:
        print(f"\n[hit --max-turns {a.max_turns} without a final answer]")
    dt = time.time() - t0
    print(f"[{dt:.0f}s wall, {gen_tokens} gen tokens, "
          f"{gen_tokens / dt:.1f} tok/s incl. tools]")

    mcp.close()


if __name__ == "__main__":
    main()