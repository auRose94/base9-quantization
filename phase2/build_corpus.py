#!/usr/bin/env python3
"""Build a GDScript corpus from the wallstoneai per-project dumps.

`data/gds_repo/files/<project>.txt` is a full dump of one Godot project:

    Godot project
    ### Name: <project>
    ### Godot version: N
    ### Directory structure:
      <tree>
    ### Files:
    File name: <relpath>
    ```<lang>
    <content>
    ```

    File name: <next relpath>
    ...

That `File name:` grouping is what the flat parquet throws away: it lets us
(a) keep only `gdscript` blocks, (b) split by the project's *declared* Godot
version, and (c) write a whole project back to disk so sibling classes and
`preload()` targets resolve — the difference between a 28% and a ~90% parse
rate under Godot 4 (see measure_corpus.py / classify_corpus.py).

Usage:
    python3 build_corpus.py --stats
    python3 build_corpus.py --emit corpus.jsonl [--min-version 4]
    python3 build_corpus.py --reconstruct <project> [--min-block 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "data" / "gds_repo" / "files"
CACHE = HERE / "data" / "corpus.json.gz"

_RE_NAME = re.compile(r"^### Name:\s*(.*)$")
_RE_VER = re.compile(r"^### Godot version:\s*(.*)$")
_RE_FILE = re.compile(r"^File name:\s*(.*)$")
_RE_FENCE = re.compile(r"^```(\w*)\s*$")


@dataclass
class Project:
    name: str
    version: str
    files: dict = field(default_factory=dict)   # relpath -> (lang, content)

    def gd(self) -> dict:
        return {p: c for p, (lang, c) in self.files.items() if lang == "gdscript"}


def parse_dump(text: str) -> Project:
    lines = text.split("\n")
    name = ver = ""
    files: dict = {}
    i = 0
    in_files = False
    while i < len(lines):
        line = lines[i]
        if not in_files:
            m = _RE_NAME.match(line)
            if m:
                name = m.group(1).strip()
            m = _RE_VER.match(line)
            if m:
                ver = m.group(1).strip()
            if line.startswith("### Files:"):
                in_files = True
            i += 1
            continue
        m = _RE_FILE.match(line)
        if m:
            rel = m.group(1).strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                fm = _RE_FENCE.match(lines[j])
                if fm:
                    lang = fm.group(1)
                    k = j + 1
                    while k < len(lines) and not _RE_FENCE.match(lines[k]):
                        k += 1
                    files[rel] = (lang, "\n".join(lines[j + 1:k]))
                    i = k + 1
                    continue
            i = j
            continue
        i += 1
    return Project(name=name or "?", version=ver or "?", files=files)


def load_all(repo: Path = REPO) -> list:
    out = []
    for f in sorted(glob.glob(str(repo / "*.txt"))):
        try:
            out.append(parse_dump(Path(f).read_text(encoding="utf-8", errors="replace")))
        except Exception as e:                                  # noqa: BLE001
            print(f"  ! {Path(f).name}: {e}", file=sys.stderr)
    return out


def major(ver: str) -> str:
    m = re.match(r"(\d+)", ver)
    return m.group(1) if m else "?"


def stats(projects: list) -> None:
    vers: dict = {}
    for p in projects:
        vers.setdefault(major(p.version), []).append(p)
    print(f"projects: {len(projects)}")
    total_gd = total_chars = 0
    for v in sorted(vers, key=lambda x: (x == "?", x)):
        ps = vers[v]
        gd_files = sum(len(p.gd()) for p in ps)
        chars = sum(sum(len(c) for c in p.gd().values()) for p in ps)
        langs: dict = {}
        for p in ps:
            for lang, c in p.files.values():
                langs[lang] = langs.get(lang, 0) + len(c)
        top = ", ".join(f"{k}:{v_:,}" for k, v_ in
                        sorted(langs.items(), key=lambda kv: -kv[1])[:5])
        print(f"  godot {v:>2s}: {len(ps):5d} projects {gd_files:6d} .gd files "
              f"{chars:12,} chars (~{chars/3.5:,.0f} tok)")
        print(f"           langs: {top}")
        if v == "4":
            total_gd, total_chars = gd_files, chars
    v4 = vers.get("4", [])
    print(f"\nGodot-4-only corpus: {len(v4)} projects, {total_gd} .gd files, "
          f"{total_chars:,} chars (~{total_chars/3.5:,.0f} tok)")
    sizes = sorted((sum(len(c) for c in p.gd().values()) for p in v4), reverse=True)
    if sizes:
        print(f"  per-project gd chars: max {sizes[0]:,} "
              f"p50 {sizes[len(sizes)//2]:,} min {sizes[-1]:,}")


def emit(projects: list, out: Path, min_version: int, min_block: int) -> None:
    n = 0
    with open(out, "w") as fo:
        for p in projects:
            if major(p.version) != str(min_version):
                continue
            for rel, code in p.gd().items():
                if len(code) < min_block:
                    continue
                fo.write(json.dumps(dict(id=f"{p.name}::{rel}", project=p.name,
                                         path=rel, version=p.version,
                                         code=code)) + "\n")
                n += 1
    print(f"wrote {n} .gd files → {out}")


def reconstruct(projects: list, name: str, keep_all: bool = False) -> Path:
    """Write one project's files to a Godot project dir and return the path."""
    p = next((x for x in projects if x.name == name), None)
    if p is None:
        cands = [x.name for x in projects if name.lower() in x.name.lower()][:8]
        raise SystemExit(f"no project {name!r}; close matches: {cands}")
    out = HERE / "data" / "recon" / re.sub(r"[^\w.-]", "_", p.name)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "project.godot").write_text(
        'config_version=5\n\n[application]\n'
        f'config/name="{p.name}"\n'
        'config/features=PackedStringArray("4.7")\n')
    wrote = 0
    for rel, (lang, code) in p.files.items():
        if not keep_all and lang not in ("gdscript", "shader", "json", "cfg"):
            continue
        dst = out / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(code)
            wrote += 1
        except OSError:
            pass
    print(f"reconstructed {p.name} (godot {p.version}): {wrote} files → {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--emit", metavar="OUT.jsonl")
    ap.add_argument("--min-version", type=int, default=4)
    ap.add_argument("--min-block", type=int, default=0)
    ap.add_argument("--reconstruct", metavar="PROJECT")
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args()

    projects = load_all(Path(a.repo))
    if a.stats or not (a.emit or a.reconstruct):
        stats(projects)
    if a.emit:
        emit(projects, Path(a.emit), a.min_version, a.min_block)
    if a.reconstruct:
        reconstruct(projects, a.reconstruct, a.keep_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
