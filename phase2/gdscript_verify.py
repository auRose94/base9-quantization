#!/usr/bin/env python3
"""Headless GDScript verifier for Phase 2 (data generation + eval).

Two layers, both driven by the installed Godot binary:

  * `check_parse`   — `godot --check-only --script`: the GDScript parser.
                      Catches syntax errors AND undeclared identifiers
                      ("Identifier ... not declared in the current scope"),
                      i.e. a real static check, not just a tokenizer.
  * `check_runtime` — instantiates the script in a throwaway SceneTree and
                      runs a few frames with `--quit-after`: catches null
                      derefs, bad method calls, invalid engine API use.

Exit codes are NOT a reliable signal: Godot returns 0 even when a script
raises `SCRIPT ERROR` at runtime, and returns 1 only for parse failures. So
the verdict comes from parsing stderr, and `Result.status` is derived from
the messages, not `returncode`.

Usage:
    python3 gdscript_verify.py --selftest
    python3 gdscript_verify.py --file snippet.gd [--runtime]
    python3 gdscript_verify.py --jsonl in.jsonl --out out.jsonl [--runtime]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

GODOT = os.environ.get("GODOT_PATH", "/usr/bin/godot-mono")
TARGET = "__target__.gd"

OK = "ok"
PARSE = "parse_error"
RUNTIME = "runtime_error"
TIMEOUT = "timeout"
OTHER = "error"

# "SCRIPT ERROR: Parse Error: <msg>\n          at: GDScript::reload (res://f.gd:4)"
# "SCRIPT ERROR: <msg>\n          at: _ready (res://f.gd:5)"
# The file group must allow ':' — Godot reports "res://name.gd", so a
# `[^:]+` file group silently never matches anything.
_RE_SCRIPT = re.compile(
    r"SCRIPT ERROR: (?:(?P<ptype>Parse Error): )?(?P<msg>[^\n]+)"
    r"\n\s*at: (?P<func>[^(]*?)\((?P<file>[^\n]+?):(?P<line>\d+)\)"
)
_RE_LOADFAIL = re.compile(r'Failed to load script "(?P<file>[^"]+)" with error "(?P<msg>[^"]+)"')
_RE_ENGINE = re.compile(r"^ERROR: (?P<msg>.+)$", re.M)

# Godot's own noise: benign in a headless import/run of a bare project.
_NOISE = (
    "Vulkan", "OpenGL", "display driver", "No loader found for resource",
    "TextServer", "fontconfig", "X11", "WAYLAND_DISPLAY", "ALSA", "PulseAudio",
    "Unable to load", "Mono", "dotnet", "GLES", "shader", "WARNING:",
)


@dataclass
class Result:
    name: str = ""
    status: str = OTHER
    ok: bool = False
    errors: list = field(default_factory=list)
    returncode: int | None = None
    elapsed: float = 0.0
    output: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("output")
        return d


def _first_line(s: str) -> str:
    return " ".join(s.split())


def classify(output: str, returncode: int | None) -> tuple[str, list]:
    """Verdict + structured errors, from stderr/stdout text."""
    errors = []
    for m in _RE_SCRIPT.finditer(output):
        kind = PARSE if m["ptype"] else RUNTIME
        errors.append(dict(kind=kind, message=_first_line(m["msg"]),
                           line=int(m["line"]), func=m["func"].strip(),
                           file=Path(m["file"]).name))
    for m in _RE_LOADFAIL.finditer(output):
        if any(e["file"] == Path(m["file"]).name for e in errors):
            continue
        errors.append(dict(kind=PARSE, message=_first_line(m["msg"]),
                           line=0, func="", file=Path(m["file"]).name))
    if not errors:
        for m in _RE_ENGINE.finditer(output):
            msg = _first_line(m["msg"])
            if any(n.lower() in msg.lower() for n in _NOISE):
                continue
            errors.append(dict(kind=OTHER, message=msg, line=0, file=""))
    if any(e["kind"] == PARSE for e in errors):
        return PARSE, errors
    if errors:
        return (RUNTIME if any(e["kind"] == RUNTIME for e in errors) else OTHER), errors
    return (OK if returncode == 0 else OTHER), errors


_RUNNER = """extends SceneTree

# Instantiates the snippet so its _ready/_process actually execute; a bare
# `--script target.gd` only works for MainLoop/SceneTree subclasses.
func _initialize() -> void:
	var script: GDScript = load("res://__T__")
	if script == null:
		quit(1)
		return
	var inst = script.new()
	if inst is Node:
		root.add_child(inst)
	else:
		print("note: target is not a Node; constructed only: ", inst.get_class())
"""


class GodotVerifier:
    def __init__(self, godot: str = GODOT, workdir: str | os.PathLike | None = None,
                 timeout: float = 25.0, frames: int = 3):
        self.godot = godot
        self.timeout = timeout
        self.frames = frames
        self._own_tmp = workdir is None
        self.dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="gdverify_"))
        self._project_ready = False

    # ---- project scaffolding -------------------------------------------------
    def ensure_project(self) -> None:
        if self._project_ready:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "project.godot").write_text(
            'config_version=5\n\n[application]\n'
            'config/name="gdverify"\n'
            'config/features=PackedStringArray("4.7")\n'
        )
        self._project_ready = True

    def _write(self, text: str, name: str = TARGET) -> None:
        (self.dir / name).write_text(text, encoding="utf-8")

    def _run(self, res: str, extra: list[str]) -> tuple[str, int | None, float]:
        cmd = [self.godot, "--headless", "--path", str(self.dir)] + extra + [res]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            return p.stdout + p.stderr, p.returncode, time.time() - t0
        except subprocess.TimeoutExpired as e:
            got = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")
            if isinstance(got, bytes):
                got = got.decode("utf-8", "replace")
            return got, None, time.time() - t0

    # ---- public checks -------------------------------------------------------
    def check_parse(self, source: str, name: str = "") -> Result:
        self.ensure_project()
        self._write(source)
        out, rc, dt = self._run(f"res://{TARGET}", ["--check-only", "--script"])
        status, errors = classify(out, rc)
        return Result(name=name, status=status, ok=(status == OK), errors=errors,
                      returncode=rc, elapsed=dt, output=out)

    def check_parse_at(self, relpath: str, name: str = "") -> Result:
        """Parse-check a file already present in the project at `relpath`.

        Needed for context-sensitive checks: siblings must stay on disk so
        `extends SiblingClass` and `preload()` can resolve.
        """
        out, rc, dt = self._run(f"res://{relpath}", ["--check-only", "--script"])
        status, errors = classify(out, rc)
        return Result(name=name or relpath, status=status, ok=(status == OK),
                      errors=errors, returncode=rc, elapsed=dt, output=out)

    def import_project(self) -> tuple[str, int | None]:
        """Run the editor's import pass so class_name globals get registered
        (`.godot/global_script_class_cache.cfg`); without it `extends Foo`
        cannot resolve even when Foo.gd is present."""
        cmd = [self.godot, "--headless", "--path", str(self.dir), "--import"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            return (e.stdout or "") if isinstance(e.stdout, str) else "", None

    def check_tree(self, runtime: bool = False) -> list:
        """Parse-check every .gd in the project, in place."""
        return [self.check_parse_at(str(p.relative_to(self.dir)))
                for p in sorted(self.dir.rglob("*.gd"))]

    def check_runtime(self, source: str, name: str = "") -> Result:
        self.ensure_project()
        self._write(source)
        head = source.lstrip()
        if head.startswith("extends SceneTree") or head.startswith("extends MainLoop"):
            out, rc, dt = self._run(
                f"res://{TARGET}", ["--quit-after", str(self.frames), "--script"])
        else:
            self._write(_RUNNER.replace("__T__", TARGET), "__runner__.gd")
            out, rc, dt = self._run(
                "res://__runner__.gd", ["--quit-after", str(self.frames), "--script"])
        status, errors = classify(out, rc)
        if rc is None:
            status, errors = TIMEOUT, [dict(kind=TIMEOUT, message="no quit within budget",
                                            line=0, file="")]
        return Result(name=name, status=status, ok=(status == OK), errors=errors,
                      returncode=rc, elapsed=dt, output=out)

    def check(self, source: str, name: str = "", runtime: bool = False) -> Result:
        """Static check; if it passes and runtime=True, also instantiate + run."""
        r = self.check_parse(source, name)
        if not r.ok or not runtime:
            return r
        rr = self.check_runtime(source, name)
        rr.errors = r.errors + rr.errors
        return rr

    def close(self) -> None:
        if self._own_tmp:
            shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        self.ensure_project()
        return self

    def __exit__(self, *exc):
        self.close()


# ---- self-test ---------------------------------------------------------------
_CASES = [
    ("ok_simple", 'extends Node\n\nfunc _ready() -> void:\n\tprint("hi")\n', OK),
    ("ok_typed", 'extends Node\n\nfunc add(a: int, b: int) -> int:\n\treturn a + b\n', OK),
    ("bad_syntax", 'extends Node\n\nfunc _ready() -> void:\n\tvar x = \n', PARSE),
    ("bad_indent", 'extends Node\n\nfunc _ready() -> void:\nvar x = 1\n', PARSE),
    ("bad_ident", 'extends Node\n\nfunc _ready() -> void:\n\tundefined_thing.call_it()\n', PARSE),
    ("bad_method", 'extends Node\n\nfunc _ready() -> void:\n\tvar n := Node.new()\n\tn.no_such_method()\n', RUNTIME),
    ("null_deref", 'extends Node\n\nfunc _ready() -> void:\n\tvar n = null\n\tn.size()\n', RUNTIME),
]


def selftest() -> int:
    print(f"godot: {GODOT}")
    if not Path(GODOT).exists():
        print(f"FAIL: {GODOT} not found")
        return 2
    bad = 0
    with GodotVerifier() as v:
        print(f"project: {v.dir}")
        for name, src, want in _CASES:
            r = v.check_runtime(src, name)
            mark = "ok " if r.status == want else "BAD"
            if r.status != want:
                bad += 1
            detail = r.errors[0]["message"][:64] if r.errors else ""
            print(f"  [{mark}] {name:12s} want={want:14s} got={r.status:14s}"
                  f" {r.elapsed:.2f}s  {detail}")
    print("PASS" if not bad else f"FAIL ({bad} mismatch)")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--file", help="verify one .gd file")
    ap.add_argument("--jsonl", help="batch: read {id,code} lines")
    ap.add_argument("--out", help="batch output path (default: <jsonl>.results.jsonl)")
    ap.add_argument("--runtime", action="store_true", help="also instantiate + run")
    ap.add_argument("--workdir", help="reuse a project dir (keeps Godot's import cache)")
    ap.add_argument("--timeout", type=float, default=25.0)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.file:
        src = Path(a.file).read_text()
        with GodotVerifier(timeout=a.timeout, workdir=a.workdir) as v:
            r = v.check(src, name=Path(a.file).name, runtime=a.runtime)
        print(json.dumps(r.as_dict(), indent=2))
        return 0 if r.ok else 1

    if a.jsonl:
        src_p, out_p = Path(a.jsonl), Path(a.out or (a.jsonl + ".results.jsonl"))
        n = nok = 0
        t0 = time.time()
        with GodotVerifier(timeout=a.timeout, workdir=a.workdir) as v, \
                open(out_p, "w") as fo:
            for line in src_p.open():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                r = v.check(item.get("code", ""), name=str(item.get("id", n)),
                            runtime=a.runtime)
                n += 1
                nok += r.ok
                fo.write(json.dumps(dict(id=item.get("id"), **r.as_dict())) + "\n")
                fo.flush()
        dt = time.time() - t0
        print(f"{nok}/{n} ok in {dt:.1f}s ({dt/max(n,1):.2f}s/item) → {out_p}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
