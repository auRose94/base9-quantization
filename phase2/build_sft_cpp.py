#!/usr/bin/env python3
"""the-stack (v1) C++ → verified SFT corpus — the phase-2 loop in C++.

v2's parquets are a metadata index with NO content column (59.4 M rows of
blob pointers — 9.8 GB learned that the hard way). v1 `bigcode/the-stack`
data/c++/*.parquet embeds the text plus the star-maxed repo licenses and the
classic quality columns. 214 shards of ~57k rows; a spread of every 7th is
streamed (30 shards ≈ 1.7 M rows), then:

  1. row filters: 300..8000 bytes, a permissive license among the repo's
     detected licenses, avg_line <= 200, max_line <= 1000,
     alphanum_fraction in [0.25, 0.95];
  2. deterministic row sampling (sha of hexsha, ~8%) to bound verification;
  3. g++ -fsyntax-only fan-out (threads; subprocess-bound work so threads
     beat the broken forkserver) — flat files, so quoted project includes
     cannot resolve: verdict `context` is honestly dropped, only `ok` trains;
  4. exact-content dedup (fork trees are enormous on github);
  5. SFT format identical to build_sft.py (system + spec + assistant=code),
     spec_cpp deriving types/functions the way the Godot spec derives
     signals — "prompt describes the interface, the file is the target".

    python3 build_sft_cpp.py --shards 30 --sample-rate 0.08 --out out/cpp_sft.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cpp_verify import CppVerifier, OK                     # noqa: E402

os.environ.setdefault("HF_HOME", str(HERE.parent / ".hf-cache"))

LICENSES = ("MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "0BSD",
            "ISC", "BSL-1.0", "Unlicense", "CC0-1.0", "zlib", "MIT-0",
            "BSD-Clause", "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause")

SYSTEM = ("You are an expert C++ developer. You write correct, self-contained "
          "modern C++ (C++17) that compiles cleanly with g++ alone. Reply with "
          "code only.")

_RE_CLASS = re.compile(r"^\s*(?:class|struct)\s+(\w+)", re.M)
_RE_FUNC = re.compile(
    r"^(?!#|\s)[\w:<>&*\s~]*?\b(\w+)\s*\((?:[^()]|\([^()]*\))*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?\{", re.M)


def spec_cpp(code: str, rel: str) -> str:
    lines = [f"Write the complete C++17 source file for `{rel}` — a "
             "self-contained file that compiles with `g++ -std=c++17 "
             "-fsyntax-only` alone."]
    if "int main(" in code:
        lines.append("It is a complete runnable program (it has main()).")
    cls = [c for c in _RE_CLASS.findall(code)][:10]
    if cls:
        lines.append("Types to define: " + ", ".join(cls))
    funcs = []
    for f in _RE_FUNC.findall(code):
        if f not in ("if", "for", "while", "switch", "return", "catch"):
            funcs.append(f)
    funcs = list(dict.fromkeys(funcs))[:16]
    if funcs:
        lines.append("Functions it defines: " + ", ".join(funcs))
    lines.append("Your reply must include the file's #include lines. No TODOs, "
                 "no placeholders, no example usage.")
    return "\n".join(lines)


def verified(record: dict) -> dict | None:
    with CppVerifier() as v:
        r = v.check_parse(record["content"])
    if r.status != OK or not r.ok:
        return None
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=30,
                    help="how many of the 214 C++ shards, spread every ~7th")
    ap.add_argument("--shard-step", type=int, default=7)
    ap.add_argument("--sample-rate", type=float, default=0.08)
    ap.add_argument("--max-candidates", type=int, default=600000)
    ap.add_argument("--max-verified", type=int, default=120000)
    ap.add_argument("--threads", type=int, default=18)
    ap.add_argument("--out", default="out/cpp_sft.jsonl")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    stored = os.path.expanduser("~/.cache/huggingface/token")
    api = HfApi(token=open(stored).read().strip() if os.path.exists(stored) else None)

    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cand, scanned = [], 0
    shard_ids = [i * a.shard_step for i in range(a.shards)]
    for si, shard in enumerate(shard_ids):
        name = f"data/c++/train-{shard:05d}-of-00214.parquet"
        local = api.hf_hub_download("bigcode/the-stack", name, repo_type="dataset")
        pf = pq.ParquetFile(local)
        cols = [c for c in ("content", "hexsha", "size", "avg_line_length",
                            "max_line_length", "alphanum_fraction",
                            "max_stars_repo_licenses", "max_stars_repo_path",
                            "max_stars_repo_name")
                if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=4000, columns=cols):
            scanned += b.num_rows
            d = {c: b.column(c) for c in cols}
            lic_idx = 0 if "max_stars_repo_licenses" in d else None
            for i in range(b.num_rows):
                content = d["content"][i].as_py() if "content" in d else None
                size = d["size"][i].as_py() if "size" in d else None
                if content is None or size is None or not (300 <= size <= 8000):
                    continue
                if lic_idx is not None:
                    lic = d["max_stars_repo_licenses"][i].as_py()
                    lic_s = " ".join(lic) if isinstance(lic, list) else str(lic or "")
                    if not any(x in lic_s for x in LICENSES):
                        continue
                    lic_name = lic_s[:40]
                else:
                    lic_name = "none"
                avg = d["avg_line_length"][i].as_py() if "avg_line_length" in d else 40.0
                mx = d["max_line_length"][i].as_py() if "max_line_length" in d else 80.0
                alpha = d["alphanum_fraction"][i].as_py() if "alphanum_fraction" in d else 0.5
                if avg > 200 or mx > 1000 or not (0.25 <= alpha <= 0.95):
                    continue
                hexsha = d["hexsha"][i].as_py() if "hexsha" in d else content
                if int(hashlib.md5(str(hexsha).encode()).hexdigest()[:8], 16) \
                        / 2**32 >= a.sample_rate:
                    continue
                path = d["max_stars_repo_path"][i].as_py() if "max_stars_repo_path" in d else "src/f.cpp"
                repo = d["max_stars_repo_name"][i].as_py() if "max_stars_repo_name" in d else "github/repo"
                cand.append(dict(repo=str(repo), path=str(path), license=lic_name,
                                 chars=len(content), content=content))
            if len(cand) >= a.max_candidates:
                break
        print(f"shard {shard:3d} ({si+1}/{len(shard_ids)}): scanned {scanned:,} rows, "
              f"candidates {len(cand):,} ({time.time()-t0:.0f}s)", flush=True)
        if len(cand) >= a.max_candidates:
            break

    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        goods = [r for r in ex.map(verified, cand, chunksize=64) if r is not None]
    rate = 100 * len(goods) / max(len(cand), 1)
    print(f"verified {len(goods):,}/{len(cand):,} (clean rate {rate:.1f}%)", flush=True)

    seen, final = set(), []
    for g in goods:
        h = hashlib.md5(g["content"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        final.append(g)
        if len(final) >= a.max_verified:
            break
    print(f"content-unique: {len(final):,} (dedup removed "
          f"{len(goods)-len(final):,})", flush=True)

    with open(out, "w") as f:
        for g in final:
            rel = f"{g['repo']}/{g['path']}"
            row = dict(id=hashlib.md5(rel.encode()).hexdigest()[:12],
                       project=g["repo"], path=g["path"], verdict="clean",
                       license=g["license"], chars=g["chars"],
                       messages=[dict(role="system", content=SYSTEM),
                                 dict(role="user", content=spec_cpp(g["content"], rel)),
                                 dict(role="assistant", content=g["content"])])
            f.write(json.dumps(row) + "\n")
    stats = dict(shard_ids=shard_ids, rows_scanned=scanned, candidates=len(cand),
                 verified=len(goods), clean_rate=round(rate, 1), unique=len(final),
                 written=len(final), minutes=round((time.time() - t0) / 60, 1),
                 sample_rate=a.sample_rate)
    (out.parent / "cpp_sft.stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())