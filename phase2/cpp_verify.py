#!/usr/bin/env python3
"""C++ verifier for the phase-2 machinery — the same philosophy as
gdscript_verify.py, judged by the real toolchain instead of Godot.

  * `check_parse`   — `g++ -fsyntax-only -std=c++17` (stdin): full parse +
                      semantics of a self-contained file. 94 ms typical.
                      A missing `#include` of a project header is C++'s
                      "unresolved context": the code may be fine, our
                      reconstruction isn't — verdict `context`, kept out of
                      training but not counted a model failure.
  * `check_runtime` — `g++` (real build) + run with a timeout: verdict on
                      exit status and captured stdout. Catches what
                      -fsyntax-only cannot (link errors, logic, crashes).

g++ return codes ARE trustworthy here (0 = OK), the reverse of Godot's
"0 even on SCRIPT ERROR" — but stderr is still parsed for the errors list
and the context-vs-genuine split.

Usage:
    python3 cpp_verify.py --selftest
    python3 cpp_verify.py --file snippet.cpp [--runtime]
    python3 cpp_verify.py --jsonl in.jsonl --out out.jsonl [--runtime]
    (as a library: CppVerifier().check_parse(code) / .check_runtime(code))
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

GXX = os.environ.get("CXX", "g++")
STD = os.environ.get("CXXSTD", "c++17")

OK = "ok"
PARSE_ERROR = "parse_error"
CONTEXT = "context"          # missing include / unknown external type
RUNTIME_ERROR = "runtime_error"
TIMEOUT = "timeout"
OTHER = "other"

_RE_MISSING = re.compile(r"fatal error:\s*(\S+):\s*No such file or directory")
_RE_ERROR = re.compile(r"^.*?error:\s*(.+)$", re.M)


@dataclass
class Result:
    name: str = ""
    status: str = OTHER
    ok: bool = False
    errors: list = field(default_factory=list)
    stdout: str = None
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return dict(name=self.name, status=self.status, ok=self.ok,
                    errors=self.errors, stdout=self.stdout,
                    seconds=round(self.seconds, 2))


def classify(stderr: str, rc: int | None) -> tuple[str, list]:
    errl = _RE_MISSING.findall(stderr)
    if errl:
        return CONTEXT, [{"kind": CONTEXT, "message": f"missing include {m}",
                          "file": m} for m in errl[:3]]
    errs = [{"kind": PARSE_ERROR, "message": e.strip()[:160]}
            for e in _RE_ERROR.findall(stderr)[:3]]
    if rc == 0 and not errs:
        return OK, []
    return (PARSE_ERROR if rc not in (None, 0) else OTHER), errs


class CppVerifier:
    def __init__(self, workdir: str | os.PathLike | None = None,
                 extra_includes: list[str] = None,
                 runtime_budget: int = 10):
        self.workdir = Path(workdir) if workdir else \
            Path(tempfile.mkdtemp(prefix="cppcheck_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.extra_includes = extra_includes or []
        self.runtime_budget = runtime_budget

    def _includes(self) -> list:
        return [f"-I{d}" for d in self.extra_includes]

    def check_parse(self, source: str, name: str = "") -> Result:
        t0 = time.time()
        try:
            r = subprocess.run([GXX, "-x", "c++", f"-std={STD}", "-fsyntax-only",
                                *self._includes(), "-"],
                               input=source.encode(), capture_output=True,
                               timeout=self.runtime_budget)
            status, errors = classify(r.stderr.decode("utf-8", "replace"), r.returncode)
        except subprocess.TimeoutExpired:
            status, errors = TIMEOUT, [dict(kind=TIMEOUT,
                                            message=f"no parse verdict within "
                                            f"{self.runtime_budget}s")]
        return Result(name=name, status=status, ok=(status == OK),
                      errors=errors, seconds=time.time() - t0)

    def check_parse_at(self, path: str | os.PathLike) -> Result:
        """Parse a file AT its real path — relative quoted includes resolve
        against its own directory, so sibling headers count as context."""
        t0 = time.time()
        try:
            r = subprocess.run([GXX, f"-std={STD}", "-fsyntax-only",
                                *self._includes(), str(path)],
                               capture_output=True, timeout=self.runtime_budget)
            status, errors = classify(r.stderr.decode("utf-8", "replace"), r.returncode)
        except subprocess.TimeoutExpired:
            status, errors = TIMEOUT, [dict(kind=TIMEOUT,
                                            message=f"no parse verdict within "
                                            f"{self.runtime_budget}s")]
        return Result(name=str(path), status=status, ok=(status == OK),
                      errors=errors, seconds=time.time() - t0)

    def check_runtime(self, source: str, name: str = "", stdin: str = "") -> Result:
        t0 = time.time()
        exe = self.workdir / f"r{id(source) % 10**8}.out"
        src = self.workdir / (exe.stem + ".cpp")
        src.write_text(source)
        build = subprocess.run([GXX, f"-std={STD}", *self._includes(),
                                "-o", str(src.with_suffix("")), str(src)],
                               capture_output=True, timeout=self.runtime_budget)
        if build.returncode != 0:
            status, errors = classify(build.stderr.decode("utf-8", "replace"),
                                      build.returncode)
            return Result(name=name, status=status, ok=False, errors=errors,
                          seconds=time.time() - t0)
        try:
            run = subprocess.run([str(src.with_suffix(""))],
                                 input=stdin.encode(), capture_output=True,
                                 timeout=self.runtime_budget)
            out, rc, to = run.stdout.decode("utf-8", "replace"), run.returncode, False
        except subprocess.TimeoutExpired:
            out, rc, to = "", None, True
        status = TIMEOUT if to else (OK if rc == 0 else RUNTIME_ERROR)
        errors = ([dict(kind=TIMEOUT, message=f"no exit within {self.runtime_budget}s")]
                  if to else ([{"kind": RUNTIME_ERROR, "message": f"exit {rc}"}]
                              if rc not in (0, None) else []))
        if src.exists():
            src.unlink()
        if exe.exists():
            exe.unlink()
        return Result(name=name, status=status, ok=(status == OK), errors=errors,
                      stdout=out, seconds=time.time() - t0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


CASES = [
    # (label, code, runtime, verdicts expected)
    ("stdlib parse", "#include <vector>\n#include <algorithm>\n"
             "int main() { std::vector<int> v{3,1,2}; std::sort(v.begin(), v.end());"
             " return v[0]; }", False, OK),
    ("syntax error", "int main() { if ( }", False, PARSE_ERROR),
    ("undeclared", "void f() { return foo(); }", False, PARSE_ERROR),
    ("missing include", '#include "project_local.hpp"\nint main() { return 0; }',
     False, CONTEXT),
    ("python inside", "def f():\n    pass", False, PARSE_ERROR),
    ("runtime ok", "#include <cstdio>\nint main() { printf(\"go\\n\"); return 0; }",
     True, OK),
    ("runtime fails", "int main() { return 2; }", True, RUNTIME_ERROR),
]


def selftest() -> int:
    failed = 0
    with CppVerifier() as v:
        for label, code, runtime, want in CASES:
            r = v.check_runtime(code, name=label) if runtime else v.check_parse(code, name=label)
            got = r.status
            mark = "ok " if got == want else "FAIL"
            failed += got != want
            print(f"  [{mark}] {label:16s} -> {got:14s} (want {want})"
                  f" {r.errors[0]['message'][:60] if r.errors else ''}")
    print("selftest:", "ALL PASS" if not failed else f"{failed} FAILURES")
    return 1 if failed else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--jsonl", help="rows with a `code` field; verdicts out")
    ap.add_argument("--out")
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="jsonl batches via a process pool")
    a = ap.parse_args()
    if a.file:
        code = Path(a.file).read_text()
        with CppVerifier() as v:
            r = v.check_runtime(code) if a.runtime else v.check_parse(code)
        print(f"{r.status}: {r.errors[:2]}")
        return 0 if r.ok else 1
    if a.jsonl:
        from concurrent.futures import ProcessPoolExecutor
        rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
        def one(row):
            with CppVerifier() as v:
                r = v.check_runtime(row["code"]) if a.runtime \
                    else v.check_parse(row["code"])
            return dict(**{k: row[k] for k in row if k != "code"},
                        status=r.status, ok=r.ok,
                        errors=[e["message"] for e in r.errors[:2]])
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            out = list(ex.map(one, rows))
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        print(json.dumps(Counter(r["status"] for r in out), indent=2))
        return 0
    return selftest()


if __name__ == "__main__":
    sys.exit(main())