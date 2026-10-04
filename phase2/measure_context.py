#!/usr/bin/env python3
"""Does restoring project context lift the Godot-4 parse rate?

The flat parquet gives 28% (measure_corpus.py). classify_corpus.py showed
63.5% of failures are unresolved siblings (`extends Actor`, `preload("res://…")`)
rather than bad code. This reconstructs whole projects from the dumps and
measures the parse rate:

  * as-is (siblings on disk, no import pass)
  * after `godot --import` (registers class_name globals in the cache)

Usage: python3 measure_context.py [N_PROJECTS]
"""
import glob
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_corpus as bc
from gdscript_verify import GodotVerifier, OK, PARSE, RUNTIME

HERE = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

BUILTIN_EXT = re.compile(r"^extends\s+([A-Z]\w*)", re.M)
CLASS_NAME = re.compile(r"^class_name\s+(\w+)", re.M)

DEP_MSG = re.compile(r"Could not find (type|base class)|Identifier .* not declared|"
                     r"Preload file .* does not exist|Cannot open file|Could not resolve")
V3_MSG = re.compile(r"keyword was removed in Godot 4")


def pick(projects):
    """Projects whose class_name globals are actually extended by a sibling —
    i.e. the ones where context restoration can change the verdict."""
    out = []
    for p in projects:
        if bc.major(p.version) != "4":
            continue
        gd = p.gd()
        names = {m.group(1) for c in gd.values() for m in CLASS_NAME.finditer(c)}
        if not names:
            continue
        deps = {m.group(1) for c in gd.values() for m in BUILTIN_EXT.finditer(c)} & names
        if deps:
            out.append(p)
    return out


def bucket(results):
    c = Counter()
    for r in results:
        if r.ok:
            c["ok"] += 1
        elif r.status == PARSE and r.errors and DEP_MSG.search(r.errors[0]["message"]):
            c["missing_context"] += 1
        elif r.status == PARSE and r.errors and V3_MSG.search(r.errors[0]["message"]):
            c["godot3_syntax"] += 1
        else:
            c[r.status] += 1
    return c


def main():
    projects = bc.load_all()
    cands = pick(projects)
    print(f"projects with resolvable class deps: {len(cands)}")
    sample = random.Random(1).sample(cands, min(N, len(cands)))

    tot_before, tot_after = Counter(), Counter()
    per = []
    for p in sample:
        out = bc.reconstruct(projects, p.name)
        files = [str(f.relative_to(out)) for f in sorted(out.rglob("*.gd"))]
        if not files:
            continue
        with GodotVerifier(workdir=str(out)) as v:
            before = [v.check_parse_at(f) for f in files]
            v.import_project()
            after = [v.check_parse_at(f) for f in files]
        b, a = bucket(before), bucket(after)
        tot_before += b
        tot_after += a
        per.append((p.name, len(files), b, a))
        print(f"  {p.name[:34]:34s} {len(files):3d} gd | before ok "
              f"{b['ok']:3d} ctx {b['missing_context']:3d} v3 {b['godot3_syntax']:2d}"
              f" | after ok {a['ok']:3d} ctx {a['missing_context']:3d} v3 {a['godot3_syntax']:2d}")

    def pct(c):
        n = sum(c.values())
        return f"{n} files: ok {100*c['ok']/n:.1f}%  missing_context {100*c['missing_context']/n:.1f}%  godot3 {100*c['godot3_syntax']/n:.1f}%  other {100*(n-c['ok']-c['missing_context']-c['godot3_syntax'])/n:.1f}%"

    print("\nBEFORE import:", pct(tot_before))
    print("AFTER  import:", pct(tot_after))


if __name__ == "__main__":
    main()
