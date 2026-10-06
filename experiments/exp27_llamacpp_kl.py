#!/usr/bin/env python3
"""exp27 — KL-divergence data for the upstream llama.cpp new-type checklist.

llama.cpp CONTRIBUTING.md's "additional criteria" for a new quantization type
require KL-divergence data vs FP16/BF16 **and** vs types of similar size, on top
of the perplexity table session 24 produced. This script runs llama.cpp's own
two-step protocol with `llama-perplexity`:

  pass 1 (reference):  -m <f16 GGUF> --kl-divergence-base <FILE.kld>          → saves
                       uint16 log-probs (n_vocab + n_chunk header, then one
                       length-nv row per evaluated position: 2nd half of each
                       512-context window)
  passes 2.. (candidate): -m <candidate> --kl-divergence-base <same FILE.kld> --kl-divergence
                       → per-chunk table + "Mean KLD" / PPL(Q)/PPL(base) stats

Models (1.5B trio of the session-24 ppl table, same corpus file):

  | role | file |
  |---|---|
  | reference (F16) | results/ref_qwen_1.5b_f16.gguf |
  | K9 k63 + embed99 | results/qwen_1.5b_k63.gguf |
  | q8_0 (sim-size, generated here) | results/base1p5b_q8_0.gguf |
  | q4_k_m (sim-size, generated here) | results/base1p5b_q4_km.gguf |

The q8_0/q4_k_m GGUFs are produced from the same f16 reference with
`llama-quantize` (session 24 used cached HF quants for these two rows; deriving
them from the identical reference skeleton is reproducible and self-consistent).

Corpus: results/wiki2_raw_test.txt — the exact file the session-24 ppl table ran
on (`-f`). Backend: the fork's CPU build, `-t 8`, everything else default. The
7900 XT is busy with the 15B-mix resume-train and the 5060 Ti stays free for
rose, so CPU is deliberate; the log header says which device the stats come from.

Artifacts (all regenerable):
  results/kld/1p5b_ref_f16.kld        base log-prob file, ~38 GB for 1.5B (152k vocab)
  results/kld/*.pass.log              raw llama-perplexity stdout per pass
  results/kld/quantize_*.log          llama-quantize logs for the two baselines
  results/exp27_llamacpp_kl.csv       parsed statistics, one row per candidate
  results/exp27_llamacpp_kl.md        the same as a table + provenance notes
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
KLD = os.path.join(RESULTS, "kld")
FORK_BIN = "/mnt/matrix/Work/llama.cpp/build-cpu/bin"
CORPUS = os.path.join(RESULTS, "wiki2_raw_test.txt")
THREADS = "8"

REFERENCE = os.path.join(RESULTS, "ref_qwen_1.5b_f16.gguf")
BASE_KLD = os.path.join(KLD, "1p5b_ref_f16.kld")

QUANTIZED = {  # name-suffix -> llama-quantize type, generated from REFERENCE
    "base1p5b_q8_0.gguf": "Q8_0",
    "base1p5b_q4_km.gguf": "Q4_K_M",
}

CANDIDATES = [  # (row label, gguf path)
    ("K9_k63_embed99", os.path.join(RESULTS, "qwen_1.5b_k63.gguf")),
    ("q8_0", os.path.join(RESULTS, "base1p5b_q8_0.gguf")),
    ("q4_k_m", os.path.join(RESULTS, "base1p5b_q4_km.gguf")),
]

NUM = r"[-+0-9.eE]+"

PATTERNS = {
    "kld_mean": rf"Mean\s+KLD:\s*({NUM})\s*±\s*({NUM})",
    "kld_max": rf"Max(?:imum)? KLD:\s*({NUM})",
    "kld_p999": rf"99\.9%\s+KLD:\s*({NUM})",
    "kld_p99": rf"99\.0%\s+KLD:\s*({NUM})",
    "kld_p95": rf"95\.0%\s+KLD:\s*({NUM})",
    "ppl_q": rf"Mean PPL\(Q\)\s*:\s*({NUM})\s*±\s*({NUM})",
    "ppl_base": rf"Mean PPL\(base\)\s*:\s*({NUM})\s*±\s*({NUM})",
    "ppl_ratio": rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*({NUM})\s*±\s*({NUM})",
    "ppl_diff": rf"Mean PPL\(Q\)-PPL\(base\)\s*:\s*({NUM})\s*±\s*({NUM})",
    "ln_ratio": rf"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*({NUM})\s*±\s*({NUM})",
}


def run(cmd: list[str], log_path: str) -> None:
    print(f"[exp27] {' '.join(os.path.basename(_c) for _c in cmd[:-1])} {cmd[-1]} ...", flush=True)
    with open(log_path, "w") as log:
        t0 = time.time()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[exp27]    rc={proc.returncode} ({time.time() - t0:.1f} s) -> {log_path}", flush=True)
    if proc.returncode != 0:
        tail = ""
        try:
            tail = open(log_path, encoding="utf-8", errors="replace").read()[-1500:]
        except OSError:
            pass
        raise SystemExit(f"[exp27] FAILED ({cmd[0]}): see {log_path}\n{tail}")


def parse_pass(log_path: str) -> dict:
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    out: dict = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, txt)
        out[key] = m.groups() if m else None
    m = re.search(rf"Final estimate:\s*PPL == ({NUM}) \+/- ({NUM})", txt)
    out["ppl_estimate"] = m.groups() if m else None
    return out


def main() -> None:
    for path in (os.path.join(FORK_BIN, "llama-perplexity"),
                 os.path.join(FORK_BIN, "llama-quantize"), CORPUS, REFERENCE):
        if not os.path.exists(path):
            raise SystemExit(f"[exp27] missing: {path}")
    os.makedirs(KLD, exist_ok=True)

    # the two similar-size baselines, derived from the reference skeleton
    for fname, qtype in QUANTIZED.items():
        out_path = os.path.join(RESULTS, fname)
        if not os.path.exists(out_path):
            run([os.path.join(FORK_BIN, "llama-quantize"), REFERENCE, out_path, qtype],
                os.path.join(KLD, f"quantize_{qtype}.log"))

    # pass 1 — the reference writes the base log-prob file
    if not os.path.exists(BASE_KLD):
        run([os.path.join(FORK_BIN, "llama-perplexity"), "-m", REFERENCE, "-f", CORPUS,
             "-t", THREADS, "--kl-divergence-base", BASE_KLD],
            os.path.join(KLD, "1p5b_ref_f16.pass.log"))
    ref_ppl = parse_pass(os.path.join(KLD, "1p5b_ref_f16.pass.log"))

    # passes 2.. — candidates against the same base file
    rows = []
    for label, model in CANDIDATES:
        log_path = os.path.join(KLD, f"1p5b_{label}.pass.log")
        run([os.path.join(FORK_BIN, "llama-perplexity"), "-m", model, "-f", CORPUS, "-t",
             THREADS, "--kl-divergence-base", BASE_KLD, "--kl-divergence"], log_path)
        rows.append((label, model, parse_pass(log_path)))

    # ---- emit -------------------------------------------------------------
    fields = ["model", "file_GB", "kld_mean", "kld_unc", "ppl_q", "ppl_q_unc",
              "ppl_base", "ppl_ratio", "ppl_ratio_unc", "ppl_diff", "ln_ratio",
              "kld_p999", "kld_p99", "kld_p95", "kld_max"]
    with open(os.path.join(RESULTS, "exp27_llamacpp_kl.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for label, model, stats in rows:
            kld_mean, kld_unc = stats["kld_mean"] or ("", "")
            ppl_q, ppl_q_unc = stats["ppl_q"] or ("", "")
            ppl_ratio, ppl_ratio_unc = stats["ppl_ratio"] or ("", "")
            ppl_base = (stats["ppl_base"] or ("", ""))[0]
            ppl_diff = (stats["ppl_diff"] or ("", ""))[0]
            ln_ratio = (stats["ln_ratio"] or ("", ""))[0]
            writer.writerow([label, f"{os.path.getsize(model) / 1e9:.3f}",
                             kld_mean, kld_unc, ppl_q, ppl_q_unc, ppl_base,
                             ppl_ratio, ppl_ratio_unc, ppl_diff, ln_ratio,
                             (stats["kld_p999"] or ("",))[0], (stats["kld_p99"] or ("",))[0],
                             (stats["kld_p95"] or ("",))[0], (stats["kld_max"] or ("",))[0]])

    kld_gb = os.path.getsize(BASE_KLD) / 1e9 if os.path.exists(BASE_KLD) else 0.0
    ref_ppl_s = " ± ".join(ref_ppl["ppl_estimate"]) if ref_ppl["ppl_estimate"] else (
        (stats and (stats["ppl_base"] or ("", "")) and (stats["ppl_base"] or ("", ""))[0]) or "n/a")
    md = ["# exp27 — KL divergence vs the f16 reference (upstream checklist item)",
          "",
          "`llama-perplexity` (llama.cpp fork build-cpu), corpus `results/wiki2_raw_test.txt`",
          "(the session-24 ppl corpus), defaults except `-t 8`, CPU backend.",
          f"Reference pass: `{os.path.basename(REFERENCE)}` -> base file "
          f"`{os.path.basename(BASE_KLD)}` ({kld_gb:.1f} GB, regenerable); its final "
          f"estimate: PPL == {ref_ppl_s} (CPU).",
          "",
          "| candidate | file GB | Mean KLD ± | PPL(Q) ± | PPL(Q)/PPL(base) | PPL(Q)−PPL(base) |",
          "|---|---|---|---|---|---|"]
    for label, model, stats in rows:
        kld_mean, kld_unc = stats["kld_mean"] or ("", "")
        ppl_q, ppl_q_unc = stats["ppl_q"] or ("", "")
        ppl_ratio, ppl_ratio_unc = stats["ppl_ratio"] or ("", "")
        ppl_diff = (stats["ppl_diff"] or ("", ""))[0]
        md.append(f"| `{label}` | {os.path.getsize(model) / 1e9:.3f} | "
                  f"{kld_mean} ± {kld_unc} | {ppl_q} ± {ppl_q_unc} | "
                  f"{ppl_ratio} ± {ppl_ratio_unc} | {ppl_diff} |")
    md += ["",
           "Notes:",
           "- F16 reference = f16 GGUF storing Qwen2.5's native bf16 weights exactly",
           "  (bf16's 8-bit mantissa ⊂ f16's 10-bit for in-range values).",
           "- K9_k63_embed99 materializes to Q8_0 on load (exact-Q8_0 theorem); the",
           "  runtime therefore runs the materialized Q8_0/F16 tensors, stock kernels.",
           "- Session-24 ppl table used GPU backends; CPU values shift slightly.",
           "- Per-chunk tables and the full llama-perplexity stdout: results/kld/*.pass.log",
           "- Quantize logs: results/kld/quantize_*.log (baselines = llama-quantize from",
           "  the reference, `Q8_0` / `Q4_K_M`)."]
    with open(os.path.join(RESULTS, "exp27_llamacpp_kl.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print("[exp27] wrote results/exp27_llamacpp_kl.csv + .md", flush=True)
    print("[exp27] NOTE: the ~38 GB base file results/kld/1p5b_ref_f16.kld stays for",
          "possible extra candidate passes; delete when done.", flush=True)


if __name__ == "__main__":
    main()