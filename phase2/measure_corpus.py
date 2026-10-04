#!/usr/bin/env python3
"""How much of the wallstoneai GDScript corpus is valid under Godot 4?

The parquet is a line-flattened dump of real Godot projects; the sample
blocks use Godot 3 API (KinematicBody, get_world().direct_space_state).
This measures the Godot-4 parse rate with phase2/gdscript_verify.py.
"""
import glob, hashlib, json, random, re, sys
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

# dedup, drop empties/short
seen, uniq = set(), []
for b in blocks:
    b = b.strip("\n")
    if len(b.strip()) < 20: continue
    h = hashlib.sha1(b.encode()).hexdigest()
    if h in seen: continue
    seen.add(h); uniq.append(b)
print(f"blocks {len(blocks):,} → unique usable {len(uniq):,}")

G3 = re.compile(r"\b(KinematicBody|Spatial|get_world\(\)|OS\.get_ticks|"
                r"PoolStringArray|PoolByteArray|yield\(|\.instance\(\)|"
                r"connect\(\"[a-z_]+\"|export\(|onready var|extends Node2D\b.*)$")
rng = random.Random(0)
sample = rng.sample(uniq, min(N, len(uniq)))

res = {"ok": 0, "parse_error": 0, "runtime_error": 0, "error": 0, "timeout": 0}
fails = []
with GodotVerifier(workdir=str(DATA / ".gdproj")) as v:
    for i, code in enumerate(sample):
        r = v.check_parse(code, name=str(i))
        res[r.status] = res.get(r.status, 0) + 1
        if r.status != OK and len(fails) < 12:
            fails.append((r.errors[0]["message"][:80], code.strip().split("\n")[0][:60]))

ok = res["ok"]; n = len(sample)
print(f"\nGodot 4.7 parse rate: {ok}/{n} = {100*ok/n:.1f}%")
print("  " + json.dumps({k: v for k, v in res.items() if v}))
print("\nfirst failure messages:")
for m, first in fails:
    print(f"  {m}\n      « {first}")
