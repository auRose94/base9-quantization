#!/usr/bin/env python3
"""Paired A/B battery for GLM-4.5-Air K9 vs stock q8_0, over HTTP.

Reuses phase2/eval_gdscript.py TASKS verbatim (mechanical graders: required
API names present, forbidden Godot-3 names absent, code present). Greedy,
thinking disabled, token budget equal — the models are the only variable.

Usage: ab_glm45air.py --port 18445 --tag k9v2 [--max-tokens 480]
"""
import argparse, json, re, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2"))
# import only TASKS/extract_code without torch-heavy deps: parse the file
src = (HERE.parent / "phase2" / "eval_gdscript.py").read_text()
ns = {}
body = src[src.index("TASKS = ["):src.index("def extract_code")]
exec(body, {"dict": dict}, ns)
TASKS = ns["TASKS"]

SYSTEM = ("You are an expert Godot 4 game developer. You write correct, "
          "idiomatic GDScript 4 — never Godot 3 APIs. Reply with code only.")

def extract_code(reply):
    m = re.search(r"```(?:gdscript|gd)?\s*\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip()

def chat(port, prompt, max_tokens):
    body = json.dumps({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=480)
    a = ap.parse_args()
    outp = HERE.parent / "results" / f"glm45air_ab_{a.tag}.jsonl"
    rows = []
    t0 = time.time()
    for t in TASKS:
        r = chat(a.port, t["prompt"], a.max_tokens)
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        code = extract_code(content)
        low = code.lower()
        must_ok = all(m in low for m in t["must"])
        bad = [b for b in t["must_not"] if b.lower() in low]
        row = dict(task=t["id"], finish=r["choices"][0]["finish_reason"],
                   tokens=r["usage"]["completion_tokens"],
                   must_ok=must_ok, must_not_hit=bad,
                   has_code=bool(code and "func" in low),
                   correct=bool(must_ok and not bad and code and "func" in low),
                   prompt_tokens=r["usage"]["prompt_tokens"])
        rows.append(row)
        print(f"{t['id']:>15}: {'PASS' if row['correct'] else 'FAIL'} "
              f"({row['tokens']} tok{', g3:' + ','.join(bad) if bad else ''})",
              flush=True)
    ok = sum(r["correct"] for r in rows)
    summary = dict(tag=a.tag, port=a.port, n=len(rows), correct=ok,
                   frac=round(ok / len(rows), 4), wall_s=round(time.time()-t0, 1),
                   max_tokens=a.max_tokens, greedy=True, thinking=False,
                   rows=rows)
    outp.write_text(json.dumps(summary, indent=1))
    print(f"\n{a.tag}: {ok}/{len(rows)} correct ({100*ok/len(rows):.1f}%) "
          f"in {summary['wall_s']}s -> {outp}")

if __name__ == "__main__":
    main()
