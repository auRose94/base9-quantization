#!/usr/bin/env python3
"""Split the Godot-4 parse failures into genuine version/syntax errors vs
unresolved-context errors (sibling classes, preloaded scenes)."""
import glob, hashlib, random, re, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from gdscript_verify import GodotVerifier, OK

DATA = Path(__file__).parent / "data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rows = []
for f in sorted(glob.glob(str(DATA / "*.parquet"))):
    rows += pq.read_table(f).column("text").to_pylist()
blocks, cur = [], None
for r in rows:
    s = r.strip()
    if s.startswith("```"):
        if cur is None: cur = []
        else: blocks.append("\n".join(cur)); cur = None
        continue
    if cur is not None: cur.append(r)
seen, uniq = set(), []
for b in blocks:
    b = b.strip("\n")
    if len(b.strip()) < 20: continue
    h = hashlib.sha1(b.encode()).hexdigest()
    if h in seen: continue
    seen.add(h); uniq.append(b)
sample = random.Random(0).sample(uniq, min(N, len(uniq)))

DEP = re.compile(r"Could not find (type|base class)|Identifier .* not declared|"
                 r"Preload file .* does not exist|Cannot open file|"
                 r"Could not resolve|Parse Error: Could not find")
V3 = re.compile(r"keyword was removed in Godot 4|Invalid indentation|"
                r"Expected .* after|Unexpected identifier|"
                r"Function .* not found in base|"
                r"Parse Error: Expected|Too many arguments|"
                r"Invalid argument|Cannot use .* as|Cannot assign|"
                r"Unterminated|Invalid escape|Expected end of statement")

buckets = {"ok": [], "dep_context": [], "g3_syntax": [], "other": []}
with GodotVerifier(workdir=str(DATA / ".gdproj")) as v:
    for i, code in enumerate(sample):
        r = v.check_parse(code, name=str(i))
        if r.status == OK:
            buckets["ok"].append(code); continue
        msg = r.errors[0]["message"] if r.errors else "?"
        if DEP.search(msg): buckets["dep_context"].append((msg, code))
        elif V3.search(msg): buckets["g3_syntax"].append((msg, code))
        else: buckets["other"].append((msg, code))

n = len(sample)
for k in ("ok", "dep_context", "g3_syntax", "other"):
    print(f"{k:14s} {len(buckets[k]):4d}  {100*len(buckets[k])/n:5.1f}%")
print("\n-- g3_syntax examples --")
for m, _ in buckets["g3_syntax"][:6]: print("   ", m[:88])
print("\n-- other examples --")
for m, _ in buckets["other"][:8]: print("   ", m[:88])
print("\n-- dep_context examples --")
for m, _ in buckets["dep_context"][:6]: print("   ", m[:88])
