#!/usr/bin/env python
"""KV-quant ppl-vs-depth sweep on the fork's llama-perplexity.

Grid = KV configs x context depths. Each cell: one llama-perplexity run on the
GPU with -ctk/-ctv, parse 'Final estimate: PPL =' + the KV buffer lines, append
to CSV (resumable -- done cells are skipped), failures logged not fatal.
"""
import argparse
import csv
import os
import re
import subprocess
import time

P = argparse.ArgumentParser()
P.add_argument("--model", default="/mnt/matrix/Work/models/qwen3.5-9b-q8_0.gguf")
P.add_argument("--corpus", default="/mnt/matrix/Work/base9-quantization/results/wiki2_raw_test.txt")
P.add_argument("--out", default="/mnt/matrix/Work/base9-quantization/results/kv_ppl_9b.csv")
P.add_argument("--bin", default="/mnt/matrix/Work/llama.cpp/build-hip/bin/llama-perplexity")
P.add_argument("--configs", default="f16:f16,q8_0:q8_0,iq4_nl:iq4_nl,iq4_nl:q8_0")
P.add_argument("--depths", default="1024,2048,4096,8192,16384,32768")
P.add_argument("--ngl", type=int, default=1024)
a = P.parse_args()

PPL_RE = re.compile(r"Final estimate:\s*PPL = ([\d.]+) \+/?- ([\d.]+)")
KV_RE = re.compile(r"(?:k|v) size = ([\d.]+) MiB, (\S+)")

configs = [c.split(":") for c in a.configs.split(",")]
depths = [int(d) for d in a.depths.split(",")]

done = set()
if os.path.exists(a.out):
    with open(a.out) as f:
        for row in csv.DictReader(f):
            if float(row.get("ppl", 0) or 0) > 0:
                done.add((row["k_type"], row["v_type"], int(row["ctx"])))

os.makedirs("/tmp/kv_ppl_logs", exist_ok=True)
fields = ["k_type", "v_type", "ctx", "ppl", "ppl_err", "k_mib", "k_ggml_type",
          "v_mib", "v_ggml_type", "exit", "note", "seconds"]
new_file = not os.path.exists(a.out)
csv_f = open(a.out, "a", newline="")
w = csv.DictWriter(csv_f, fieldnames=fields)
if new_file:
    w.writeheader()
    csv_f.flush()

for ck, cv in configs:
    for ctx in depths:
        if (ck, cv, ctx) in done:
            print(f"[sweep] skip done {ck}/{cv}@{ctx}", flush=True)
            continue
        tag = f"{ck}_{cv}_c{ctx}"
        log = f"/tmp/kv_ppl_logs/{tag}.log"
        cmd = [a.bin, "-m", a.model, "-f", a.corpus, "-c", str(ctx),
               "-ctk", ck, "-ctv", cv, "-fa", "on", "-ngl", str(a.ngl)]
        t0 = time.time()
        with open(log, "w") as lf:
            try:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=1800)
            except subprocess.TimeoutExpired:
                r = None
        secs = time.time() - t0
        txt = open(log, errors="replace").read()
        m = PPL_RE.search(txt)
        kv = KV_RE.findall(txt)
        km = next((x for x in kv if x[1].startswith("iq4") or x[1].startswith("q") or x[1].startswith("f")), ("", ""))
        row = {"k_type": ck, "v_type": cv, "ctx": ctx,
               "ppl": m.group(1) if m else "", "ppl_err": m.group(2) if m else "",
               "k_mib": km[0], "k_ggml_type": ck, "v_mib": "", "v_ggml_type": cv,
               "exit": r.returncode if r else 9, "note": "", "seconds": f"{secs:.0f}"}
        if not m:
            tail = txt.strip().splitlines()[-3:] if txt.strip() else ["no output"]
            row["note"] = " | ".join(x.strip()[:120] for x in tail)
        w.writerow(row)
        csv_f.flush()
        print(f"[sweep] {ck}/{cv}@{ctx}: ppl={row['ppl']} exit={r.returncode} "
              f"{row['seconds']}s", flush=True)

csv_f.close()
print(f"[sweep] done -> {a.out}", flush=True)