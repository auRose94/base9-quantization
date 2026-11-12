"""Rung 1: analog write-path survival sweep.

Subjects (all origin = Qwen2.5-Coder-1.5B, ref = ref_qwen_1.5b_f16.gguf):
  f16    — reference weights, no encoding loss (substrate upper bound)
  q8_0   — effective q8_0 weights, RTN re-quantized here on the same tensor
           coverage the k63 container has (LINEAR + embed), fp16 block scales
  q4_km  — effective Q4_K_M weights, dequantized from base1p5b_q4_km.gguf
  k63    — effective k63/e99 container weights, decoded in-process from
           qwen_coder_1.5b_k63_embed99.k9 (k9.py codec)

For each (subject, sigma): apply the analog write-path channel
(analog_channel.py) to every tensor, write an f16-carried GGUF with the
reference's KVs, run the fork's llama-perplexity on the wiki2 corpus, append a
CSV row.  Noise draws are common random numbers per (seed, tensor name), so
survival curves are smooth in sigma.

Run recipe (.venv-glm python; fork gguf-py on PYTHONPATH):
  PYTHONPATH=/home/rose/Work/llama.cpp/gguf-py \
    .venv-glm/bin/python experiments/analog_survival.py [--smoke] [options]

House rules honored: GGML_NO_BACKTRACE=1 on every test binary; results CSV is
append+flush (a killed sweep keeps its rows); heavy runs belong in a capped
unit (systemd-run --user -p MemoryMax=24G).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
# the fork's gguf-py is a superset (K9_4/6/7 types) — always preferred
sys.path.insert(0, "/home/rose/Work/llama.cpp/gguf-py")

import gguf  # noqa: E402

from gguf import GGUFValueType  # noqa: E402
from analog_channel import AnalogParams, analog_channel, tensor_rng  # noqa: E402
from export_k9_gguf import gguf_name  # noqa: E402
import k9  # noqa: E402

REF = REPO / "results/ref_qwen_1.5b_f16.gguf"
K9FILE = REPO / "results/qwen_coder_1.5b_k63_embed99.k9"
Q4KM = REPO / "results/base1p5b_q4_km.gguf"
CORPUS = REPO / "results/wiki2_raw_test.txt"
FORK_BIN = Path("/home/rose/Work/llama.cpp/build-cpu/bin")
OUT = REPO / "results/analog"
LOGS = OUT / "logs"
CSV_PATH = OUT / "survival.csv"

PPL_RE = re.compile(r"Final estimate:\s*PPL =+\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
ALL_SUBJECTS = ("f16", "q8_0", "q4_km", "k63")


# ------------------------------------------------------------------ loading --
def read_reference(path: Path = REF):
    reader = gguf.GGUFReader(str(path))
    arch = str(bytes(reader.get_field("general.architecture").parts[-1]), "utf-8")
    kvs = []
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue  # the writer manages these itself
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        kvs.append((field.name, field.contents(), vtype, sub))
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = np.array(t.data)  # own the bytes (numpy (rows, cols))
    return arch, kvs, tensors


def rtn_q8_0(w: np.ndarray) -> np.ndarray:
    """RTN q8_0 with fp16 block scales (block 32, rows x cols), -> f32."""
    wf = np.ascontiguousarray(w, dtype=np.float32)
    r, c = wf.shape
    blocks = wf.reshape(r, c // 32, 32)
    amax = np.abs(blocks).max(axis=2, keepdims=True)
    d = (amax / np.float32(127.0)).astype(np.float16).astype(np.float32)
    q = np.clip(np.rint(blocks / np.where(d == 0, 1.0, d)), -127, 127)
    return (q * d).reshape(r, c)


def rtn_k9_grid(w: np.ndarray, k: int, group: int = 64,
                dither_rng: np.random.Generator | None = None,
                feedback: float = 0.0) -> np.ndarray:
    """RTN onto the K9 symmetric odd grid: w' = m*(d-H)/H, H=(k-1)//2,
    m = fp16(group amax), d in [0, k).  Always an exact zero level; uniform k
    across tensors (the production containers pin embedding at k99 — these
    uniform-RTN reconstructions exist to measure the k-trend, not to ship).
    dither_rng: stochastic rounding (unbiased snapping).
    feedback: residual error-feedback carry along column blocks (her "snap
    toward the f16 source" idea = sigma-delta / error diffusion: each block's
    snap error is carried into the next block's target, compensating the
    difference so the GROUP's function tracks the f16 source)."""
    H = (k - 1) // 2
    wf = np.ascontiguousarray(w, dtype=np.float32)
    r, c = wf.shape
    nb = c // group
    blocks = wf.reshape(r, nb, group)
    m = np.abs(blocks).max(axis=2, keepdims=True).astype(np.float16).astype(np.float32)
    m = np.where(m == 0.0, np.float32(1.0), m)

    def snap(frac: np.ndarray) -> np.ndarray:
        if dither_rng is None:
            d = np.rint(frac)
        else:
            base_f = np.floor(frac)
            d = base_f + (dither_rng.random(frac.shape, dtype=np.float32)
                          < (frac - base_f)).astype(np.float32)
        return np.clip(d, 0, k - 1)

    if feedback <= 0.0:
        d = snap(blocks / m * np.float32(H) + np.float32(H))
        return (m * (d - H) / np.float32(H)).reshape(r, c)

    out = np.empty_like(blocks)
    carry = np.zeros((r, 1, group), dtype=np.float32)
    for b in range(nb):
        target = (blocks[:, b:b + 1, :] + feedback * carry) \
            / m[:, b:b + 1, :] * np.float32(H) + np.float32(H)
        d = snap(target)
        snapped = m[:, b:b + 1, :] * (d - H) / np.float32(H)
        out[:, b:b + 1, :] = snapped
        carry = (blocks[:, b:b + 1, :] - snapped)  # what f16 wanted, unsnapped
    return out.reshape(r, c)


def build_subject(subject: str, ref_tensors: dict, k9f: k9.K9File | None):
    """Effective weights per subject; ref dtype preserved, overrides on top."""
    weights = dict(ref_tensors)
    covered = set()
    if subject == "f16":
        pass
    elif subject == "q8_0":
        assert k9f is not None
        covered = {gguf_name(m["name"]) for m in k9f.meta}
        for gname in covered:
            if gname not in weights:
                continue
            w = weights[gname].astype(np.float32)
            weights[gname] = rtn_q8_0(w).astype(weights[gname].dtype)
    elif subject == "q4_km":
        r2 = gguf.GGUFReader(str(Q4KM))
        for t in r2.tensors:
            if t.name not in weights:
                continue
            if t.data.dtype in (np.float16, np.float32):
                weights[t.name] = np.array(t.data)
            else:
                flat = gguf.dequantize(np.ascontiguousarray(t.data).reshape(-1),
                                       t.tensor_type)
                # reader t.shape is (ne0, ne1) = (cols, rows) for 2-D tensors;
                # t.data.shape is the packed-BYTE layout, never reshape to it
                rows, cols = int(t.shape[1]), int(t.shape[0])
                weights[t.name] = flat.reshape(rows, cols).astype(
                    weights[t.name].dtype)
        covered = set(weights)  # full coverage; nothing ref-only left
    elif subject == "k63":
        assert k9f is not None
        for m in k9f.meta:
            gname = gguf_name(m["name"])
            if gname not in weights:
                continue  # e.g. tied lm_head absent from the reference
            rec = k9f.record(m["name"])
            r, c = m["shape"]
            w = k9.decode_tensor(rec, m["group"]).numpy().reshape(r, c)
            weights[gname] = w.astype(weights[gname].dtype)
            covered.add(gname)
    elif subject.startswith("k") and subject[1:].rstrip("de").isdigit():
        assert k9f is not None
        dither = subject.endswith("d")
        kv = int(subject[1:].rstrip("de"))
        assert kv >= 3 and kv % 2 == 1, f"k must be odd, got {kv}"
        for m in k9f.meta:
            gname = gguf_name(m["name"])
            if gname not in weights:
                continue
            drng = (tensor_rng(0, f"dither:{subject}:{gname}") if dither else None)
            weights[gname] = rtn_k9_grid(weights[gname], kv, dither_rng=drng,
                                         feedback=(1.0 if subject.endswith("e") else 0.0)
                                         ).astype(weights[gname].dtype)
            covered.add(gname)
    elif subject == "file":
        pass  # tensors already loaded from --ref (e.g. a merged E3 model)
    else:
        raise SystemExit(f"unknown subject {subject}")
    return weights, covered


# ------------------------------------------------------------------ writing --
def write_gguf(path: Path, arch: str, kvs, weights: dict) -> None:
    writer = gguf.GGUFWriter(str(path), arch=arch)
    for name, contents, vtype, sub in kvs:
        writer.add_key_value(name, contents, vtype, sub_type=sub)
    for name, arr in weights.items():
        writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, arr in weights.items():
        writer.write_tensor_data(arr)
    writer.close()


# ------------------------------------------------------------------- ppl ----
def run_ppl(model: Path, threads: int, chunks: int) -> tuple[float, float, float]:
    log = LOGS / f"{model.stem}.log"
    env = {**os.environ, "GGML_NO_BACKTRACE": "1"}
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run(
            [str(FORK_BIN / "llama-perplexity"), "-m", str(model), "-f", str(CORPUS),
             "-t", str(threads), "--chunks", str(chunks)],
            stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=7200)
    secs = time.time() - t0
    if r.returncode != 0:
        sys.stderr.write(open(log).read()[-2000:])
        raise SystemExit(f"llama-perplexity failed for {model.name} ({r.returncode})")
    txt = open(log).read()
    m = PPL_RE.search(txt)
    if not m:
        sys.stderr.write(txt[-2000:])
        raise SystemExit(f"no 'Final estimate: PPL ==' line in {log}")
    return float(m.group(1)), float(m.group(2)), secs


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="f16 only, 3 sigma points, 4 chunks")
    ap.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    ap.add_argument("--sigmas",
                    default="0,0.0025,0.005,0.01,0.02,0.04,0.08",
                    help="sigma_prog sweep, fraction of amax")
    ap.add_argument("--sigma-read", type=float, default=0.0)
    ap.add_argument("--ir-alpha", type=float, default=0.0)
    ap.add_argument("--stuck-p", type=float, default=0.0)
    ap.add_argument("--levels", type=int, default=0)
    ap.add_argument("--scale-ref", choices=("tensor", "group"), default="tensor",
                    help="noise reference: tensor full scale (naive array) or "
                         "per-64-group full scale (gain-calibrated array)")
    ap.add_argument("--boost-factor", type=float, default=1.0,
                    help="amplify every weight by F before storing; rails stay "
                         "at the ORIGINAL tensor full scale (rail_limit) and "
                         "weights are divided by F after readout — the "
                         "amplify-the-weak-signals demo. Forces tensor ref.")
    ap.add_argument("--csv-out", default=str(CSV_PATH))
    ap.add_argument("--dead-zone", type=float, default=0.0,
                    help="exact-zero protection: post-noise cells below "
                         "dead_zone * amax_ref snap to 0")
    ap.add_argument("--only-class", choices=("all", "attn", "ffn", "embed"),
                    default="all",
                    help="apply the channel ONLY to that tensor class "
                         "(others pass through clean) — damage-localization "
                         "ablation")
    ap.add_argument("--rng-key", choices=("subject", "tensor"), default="subject",
                    help="CRN key: 'subject' (legacy; each subject gets its own "
                         "noise realization) or 'tensor' (PAIRED — every subject "
                         "at a seed shares one realization per tensor; required "
                         "for cross-subject comparisons near cliffs)")
    ap.add_argument("--ref", default=str(REF),
                    help="reference GGUF supplying KVs + base tensors; with "
                         "--subjects file, its tensors ARE the subject (any "
                         "trained-model GGUF, e.g. an E3 merge)")
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep every run GGUF")
    args = ap.parse_args()

    subjects = args.subjects.split(",")
    sigmas = [float(s) for s in args.sigmas.split(",")]
    if args.smoke:
        subjects, sigmas, args.chunks = ["f16"], [0.0, 0.01, 0.04], 4

    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    ref_path = Path(args.ref)
    arch, kvs, ref = read_reference(ref_path)
    print(f"ref: {ref_path.name}, {len(ref)} tensors, arch={arch}, "
          f"{sum(v.nbytes for v in ref.values()) / 2**30:.2f} GiB")

    k9f = None
    if any(s in ("q8_0", "k63") or (s.startswith("k") and s[1:].rstrip("de").isdigit())
           for s in subjects):
        k9f = k9.K9File(str(K9FILE))
        print(f"k9: {len(k9f.meta)} records from {K9FILE.name}")

    csv_path = Path(args.csv_out)
    fresh_csv = not csv_path.exists()
    bf = args.boost_factor

    def in_class(name: str) -> bool:
        if args.only_class == "all":
            return True
        if args.only_class == "attn":
            return ".attn_" in name
        if args.only_class == "ffn":
            return ".ffn_" in name
        return name.startswith("token_embd") or name.startswith("output.")
    for subject in subjects:
        weights, _ = build_subject(subject, ref, k9f)
        label = ref_path.stem if subject == "file" else subject
        for sigma in sigmas:
            p = AnalogParams(sigma_prog=sigma, sigma_read=args.sigma_read,
                             ir_alpha=args.ir_alpha, stuck_p=args.stuck_p,
                             levels=args.levels, scale_ref=args.scale_ref,
                             dead_zone=args.dead_zone)
            degraded = {}
            for name, arr in weights.items():
                if not in_class(name):
                    degraded[name] = arr  # ablation: class passes through clean
                    continue
                rng = tensor_rng(args.seed, name if args.rng_key == "tensor"
                                 else f"{label}:{name}")
                if bf != 1.0:
                    rail = float(np.abs(arr.astype(np.float32)).max())
                    p2 = replace(p, rail_limit=rail, scale_ref="tensor")
                    stored = analog_channel(arr.astype(np.float32) * np.float32(bf),
                                            p2, rng) / np.float32(bf)
                    degraded[name] = stored.astype(arr.dtype)
                else:
                    degraded[name] = analog_channel(arr, p, rng)
            sr = {"tensor": "t", "group": "g"}[args.scale_ref]
            boxt = f"_x{bf:g}" if bf != 1.0 else ""
            clst = "" if args.only_class == "all" else f"_{args.only_class}"
            dzst = f"_dz{args.dead_zone:g}" if args.dead_zone else ""
            # chunks+seed in the filename: concurrent units must never share
            # a GGUF path (a cleanup in one unit deletes another's file)
            tag = f"{label}_{sr}_c{args.chunks}_s{args.seed}_sp{sigma:.4f}{boxt}{clst}{dzst}"
            gguf_path = OUT / f"{tag}.gguf"
            t0 = time.time()
            write_gguf(gguf_path, arch, kvs, degraded)
            write_secs = time.time() - t0
            ppl, unc, secs = run_ppl(gguf_path, args.threads, args.chunks)
            row = [label, args.scale_ref, sigma, args.sigma_read, args.ir_alpha,
                   args.stuck_p, args.levels, args.seed, args.chunks, args.threads,
                   f"{ppl:.4f}", f"{unc:.4f}", f"{write_secs:.1f}", f"{secs:.1f}",
                   str(gguf_path)]
            with open(csv_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if fresh_csv:
                    w.writerow(["subject", "scale_ref", "sigma_prog", "sigma_read",
                                "ir_alpha", "stuck_p", "levels", "seed", "chunks",
                                "threads", "ppl", "ppl_unc", "write_s", "ppl_s",
                                "gguf"])
                    fresh_csv = False
                w.writerow(row)
                fh.flush()
            print(f"[{tag}] ppl={ppl:.4f} ±{unc:.4f} "
                  f"(write {write_secs:.0f}s, ppl {secs:.0f}s)", flush=True)
            if not args.keep and sigma != 0.0:
                gguf_path.unlink()
        del weights

    if k9f is not None:
        k9f.close()
    print(f"done -> {csv_path}")


if __name__ == "__main__":
    main()
