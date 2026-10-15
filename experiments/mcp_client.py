#!/usr/bin/env python3
"""Minimal MCP (Model Context Protocol) stdio client — stdlib only.

Speaks just enough JSON-RPC 2.0 over a child process's stdio to connect to
any local MCP server (e.g. `lean-ctx mcp`): initialize handshake, tools/list,
tools/call. No MCP SDK dependency (the repo's pip discipline).

The asyncio loop runs in a private daemon thread; the calling thread uses the
blocking wrappers (connect / tools / call) — fits chat_k9's sync Engine.

    mcp = MCPGroup()
    mcp.connect({"ctx": ["lean-ctx", "mcp"]})
    print([t["function"]["name"] for t in mcp.tools()])
    out = mcp.call("ctx_read", {"target_path": "README.md", "mode": "map"})
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    pass


class _Session:
    """One MCP server subprocess + its request bookkeeping."""

    def __init__(self, name: str, command: list[str]):
        self.name = name
        self.command = command
        self.tools: list = []
        self._pending: dict[int, asyncio.Future] = {}
        self._gen = 0

    async def start(self, env: dict | None = None) -> None:
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.command, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, env=env)
        except FileNotFoundError as e:
            raise MCPError(f"server {self.name!r}: {e}") from e
        self._reader_task = asyncio.ensure_future(self._read_loop())

    async def _read_loop(self) -> None:
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue                       # servers may log non-JSON lines
            if isinstance(msg, dict) and msg.get("id") in self._pending:
                fut = self._pending.pop(msg["id"])
                if not fut.done():
                    if "error" in msg:
                        fut.set_exception(MCPError(str(msg["error"])))
                    else:
                        fut.set_result(msg.get("result", {}))
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError(f"server {self.name!r} exited"))

    async def _request(self, method: str, params: dict | None,
                       timeout: float) -> dict:
        self._gen += 1
        rid = self._gen
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        frame = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                            "params": params or {}}) + "\n"
        self.proc.stdin.write(frame.encode())
        await self.proc.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise MCPError(f"timeout on {method} ({self.name})") from None

    async def handshake(self) -> None:
        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "chat_k9", "version": "0.1"}}, timeout=30)
        self.proc.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/'
                              b'initialized"}\n')
        await self.proc.stdin.drain()
        res = await self._request("tools/list", {}, timeout=30)
        self.tools = res.get("tools", [])

    async def call(self, name: str, arguments: dict, timeout: float) -> str:
        res = await self._request("tools/call",
                                  {"name": name, "arguments": arguments},
                                  timeout=timeout)
        if res.get("isError"):
            raise MCPError(_content_text(res) or "tool reported an error")
        return _content_text(res)


def _content_text(result: dict) -> str:
    parts = []
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts).strip()


class MCPGroup:
    """Blocking façade over one or more MCP stdio servers."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        daemon=True)
        self._thread.start()
        self.sessions: dict[str, _Session] = {}
        self.started = False

    def connect(self, servers: dict[str, list[str]], timeout: float = 60.0):
        async def _all():
            for name, cmd in servers.items():
                s = _Session(name, cmd)
                await s.start()
                self.sessions[name] = s
            for name, s in list(self.sessions.items()):
                try:
                    await s.handshake()
                except MCPError as e:
                    await _kill(s)
                    del self.sessions[name]
                    raise MCPError(str(e)) from e
        t0 = time.time()
        fut = asyncio.run_coroutine_threadsafe(_all(), self._loop)
        fut.result(timeout=timeout)
        self.started = True

    def tools(self) -> list:
        """MCP tool descriptors as the OpenAI/Qwen function-tool list."""
        out = []
        for name, s in self.sessions.items():
            for t in s.tools:
                fn = {"name": t["name"],
                      "description": t.get("description", "")}
                params = t.get("inputSchema", {})
                if params.get("type") != "object":
                    params = {"type": "object", "properties": {}}
                req = params.get("required", [])
                # small models need full schemas spelled out
                props = {}
                for prop, spec in params.get("properties", {}).items():
                    spec = dict(spec)
                    if prop in req:
                        spec["required"] = True
                    props[prop] = spec
                fn["parameters"] = {"type": "object", "properties": props}
                out.append({"type": "function", "function": fn})
        return out

    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools()]

    def call(self, name: str, arguments: dict, timeout: float = 90.0) -> str:
        s = self._who(name)
        fut = asyncio.run_coroutine_threadsafe(s.call(name, arguments,
                                                      timeout), self._loop)
        return fut.result(timeout=timeout + 10)

    # NOTE on lean-ctx (observed): an EXACTLY repeated call is answered with a
    # one-line "[cross-agent cache … N tokens avoided]" stub instead of
    # content — persisted state, not the read cache (ctx_cache clear does not
    # lift it). Fresh arguments return full content, so callers should vary
    # calls rather than retry; chat_k9's repeat guard flags this in-band.

    def _who(self, name: str) -> _Session:
        for s in self.sessions.values():
            if any(t["name"] == name for t in s.tools):
                return s
        raise MCPError(f"unknown tool {name!r}")

    def close(self):
        async def _kill_all():
            for s in list(self.sessions.values()):
                await _kill(s)
        if self.started:
            asyncio.run_coroutine_threadsafe(_kill_all(), self._loop) \
                .result(timeout=10)
        self.started = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


async def _kill(s: _Session):
    try:
        s.proc.kill()
    except ProcessLookupError:
        pass


def selftest() -> int:
    """Handshake + call against a tiny stdlib echo server, no GPU/lean-ctx."""
    bad = []
    server = r'''
import json, sys
TOOLS = [{"name": "echo", "description": "echoes its text",
          "inputSchema": {"type": "object",
                          "properties": {"text": {"type": "string", "required": True}},
                          "required": ["text"]}}]
sys.stderr.write("server boot\n")
for line in sys.stdin:
    try:
        m = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "id" not in m:
        continue
    method = m["method"]
    if method == "initialize":
        r = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}
    elif method == "tools/list":
        r = {"tools": TOOLS}
    elif method == "tools/call":
        r = {"content": [{"type": "text", "text": "echo: " + m["params"]["arguments"]["text"]}]}
    else:
        r = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}) + "\n")
    sys.stdout.flush()
'''
    sp = Path("/tmp/mcp_echo_server.py")
    sp.write_text(server)
    mcp = MCPGroup()
    try:
        mcp.connect({"echo": [sys.executable, str(sp)]})
        names = mcp.tool_names()
        out = mcp.call("echo", {"text": "roundtrip"})
        print(f"  [{'ok ' if names == ['echo'] else 'BAD'}] tools/list: {names}")
        print(f"  [{'ok ' if out == 'echo: roundtrip' else 'BAD'}] tools/call: {out}")
        if names != ["echo"]:
            bad.append("list")
        if out != "echo: roundtrip":
            bad.append("call")
    except MCPError as e:
        print(f"  [BAD] MCPError: {e}")
        bad.append("connect")
    finally:
        mcp.close()
    print("PASS" if not bad else f"FAIL ({len(bad)}): {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(selftest())