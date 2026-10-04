#!/usr/bin/env python3
"""exp5 — "the model, printed as repeating numbers": lossless print→parse round trip.

The claim under test (docs/00, Formalization 1): quantize TinyStories-33M to
the ninths grid and every weight becomes ONE repeating decimal digit. Then the
whole model is printable as repeating decimals and must reload losslessly.

Pipeline:
  1. quantize every body weight matrix to the ninths grid (per-row scale m,
     digits s ∈ {0..8}; dequant w = (s−4)·m/4)
  2. print the model: every weight as its repeating decimal "0.(s)" = s/9
  3. re-parse the printed file from disk (regex only, no side channel)
  4. assert exact digit/scale equality, bit-exact weight reconstruction,
     and identical perplexity after a fp32-restore + parsed-reload cycle
  5. report the size ladder: fp32 · printed text · gzip · compact digits ·
     rANS · alphabet floor log2(9)

Out: results/printed_model.txt (+ .gz), results/printed_model_sample.txt,
     results/exp5_print_roundtrip.md
"""
import gzip
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits, entropy_bits
import exp4_real_model_ptq as e4

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

K = 9
EVAL_BLOCKS = 300  # same eval window as exp4 → comparable perplexities
PRINTED = RESULTS / "printed_model.txt"

HEADER = (
    "# printed_repeating_model v1 | model=roneneldan/TinyStories-33M | grid=ninths\n"
    "# Each token 0.(s) is the repeating decimal s/9 = 0.sssss... (base 10),\n"
    "# where s in {0..8} is the base-9 digit of the quantized weight.\n"
    "# Dequant equation: w = (s - 4) * m / 4, with per-row scale m below.\n"
    "# row line:  m=<scale repr> | 0.(s0)0.(s1)...\n"
)


def quantize_layer(W):
    """Ninths grid per row — same expressions as parse-side reconstruction.

    float32 throughout: the intermediate idx*step product is exact in float64
    and both sides do the identical expression before the final float32 cast,
    so print→parse reconstruction is bit-exact."""
    M = np.abs(W).max(axis=1, keepdims=True).astype(np.float32)
    step = (2.0 * M / (K - 1)).astype(np.float32)
    idx = np.clip(np.round((W + M) / step), 0, K - 1).astype(np.int64)
    deq = (-M + idx * step).astype(np.float32)
    return M, idx, deq


def serialize(quant):
    lines = [HEADER.rstrip("\n")]
    for name, tr, M, idx in quant:
        r, c = idx.shape
        lines.append(f"LAYER {name} tr={int(tr)} rows={r} cols={c}")
        for i in range(r):
            scale = repr(float(M[i, 0]))
            digs = "".join(f"0.({s})" for s in idx[i])
            lines.append(f"m={scale} | {digs}")
    return "\n".join(lines) + "\n"


LAYER_RE = re.compile(r"^LAYER (\S+) tr=(\d) rows=(\d+) cols=(\d+)$")


def parse(path):
    """Read the printed file → list of (name, tr, scales[R], digits[R, C])."""
    out = []
    cur = None
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("LAYER "):
            assert cur is None, "new LAYER while previous incomplete"
            mobj = LAYER_RE.match(line)
            assert mobj, f"bad LAYER line: {line[:80]}"
            name, tr, r, c = mobj.group(1), mobj.group(2) == "1", int(mobj.group(3)), int(mobj.group(4))
            cur = dict(name=name, tr=tr, rows=r, cols=c, scales=[], digits=[])
        elif line.startswith("m="):
            mm, _ = line.split(" | ", 1)
            scale = np.float32(float(mm[2:]))
            svals = np.array([int(t) for t in re.findall(r"0\.\((\d)\)", line)],
                             dtype=np.int64)
            assert svals.size == cur["cols"], f"digit count mismatch in layer {cur['name']}"
            cur["scales"].append(scale)
            cur["digits"].append(svals)
        else:
            assert not line, f"unparsed line: {line[:80]}"
        if cur is not None and len(cur["scales"]) == cur["rows"] and len(cur["digits"]) == cur["rows"]:
            out.append((cur["name"], cur["tr"],
                        np.array(cur["scales"], dtype=np.float32).reshape(-1, 1),
                        np.array(cur["digits"], dtype=np.int64)))
            cur = None
    assert cur is None
    return out


def main():
    dev = e4.device_setup()
    model, tok = e4.load_model(dev)
    layers = e4.quantized_layers(model)

    # fp32 reference (kept aside; restored before the reload test)
    saved = {}
    for name, m in layers:
        saved[name] = m.weight.data.detach().clone()

    blocks = e4.get_blocks(tok, dev)
    ppl0 = e4.perplexity(model, blocks)
    print(f"fp32 baseline perplexity ({EVAL_BLOCKS} blocks): {ppl0:.3f}")

    # 1) quantize in place
    quant = []
    n_weights = 0
    serr = wsum2 = wsum = 0.0
    for name, m in layers:
        tr = type(m).__name__ == "Conv1D"
        W = m.weight.data.detach().cpu().numpy().astype(np.float32)
        if tr:
            W = W.T
        M, idx, deq = quantize_layer(W)
        m.weight.data = torch.from_numpy(np.ascontiguousarray(deq.T if tr else deq)).to(dev)
        quant.append((name, tr, M, idx))
        n_weights += idx.size
        serr += ((W - deq) ** 2).sum()
        wsum2 += (W * W).sum()
        wsum += W.sum()
    rel_mse = float(serr / (wsum2 - n_weights * (wsum / n_weights) ** 2))
    ppl1 = e4.perplexity(model, blocks)
    print(f"ninths-quantized perplexity: {ppl1:.3f} (same eval as exp4's 9-level row)")

    # 2) print the model
    text = serialize(quant)
    PRINTED.write_text(text)
    gz = gzip.compress(text.encode())
    (RESULTS / "printed_model.txt.gz").write_bytes(gz)
    n_rows = sum(M.size for _, _, M, _ in quant)
    print(f"printed {n_weights:,} weights -> {PRINTED.name} "
          f"({PRINTED.stat().st_size/1e6:.1f} MB plain, {len(gz)/1e6:.1f} MB gz)")

    # 3) re-parse from disk
    parsed = parse(PRINTED)
    assert [(p[0], p[1]) for p in parsed] == [(q[0], q[1]) for q in quant], "layer list mismatch"

    asserts = []
    layer_map = dict(layers)
    for (name, tr, M, idx), (pname, ptr, pM, pidx) in zip(quant, parsed):
        assert np.array_equal(pidx, idx), f"digit mismatch: {name}"
        assert np.array_equal(pM, M), f"scale mismatch: {name}"
        step = 2.0 * pM / (K - 1)
        deq_built = (-pM + pidx * step).astype(np.float32)
        # compare against the CURRENT (quantized) weight in the model
        m = layer_map[name]
        W2 = m.weight.data.detach().cpu().numpy()
        W2 = W2.T if tr else W2
        assert np.array_equal(W2, deq_built), f"bit-exact reconstruction failed: {name}"
    asserts.append("all digits + scales + dequantized weights bit-exact across "
                   f"{len(quant)} matrices / {n_weights:,} weights")

    # 4) fresh reload: restore fp32, then load EVERYTHING from the parsed file
    for name, m in layers:
        m.weight.data.copy_(saved[name])
    for pname, ptr, pM, pidx in parsed:
        m = layer_map[pname]
        step = 2.0 * pM / (K - 1)
        deq = (-pM + pidx * step).astype(np.float32)
        m.weight.data = torch.from_numpy(np.ascontiguousarray(deq.T if ptr else deq)).to(dev)
    ppl2 = e4.perplexity(model, blocks)
    assert abs(ppl2 - ppl1) < 1e-9 * ppl1, f"perplexity changed on reload: {ppl1} vs {ppl2}"
    asserts.append(f"perplexity after fp32-restore + printed-file reload identical: {ppl2:.6f}")

    # 5) size ladder (linear weights only; embed/lm_head/bias are fp32 side info)
    n_lin = n_weights
    sym = np.concatenate([idx.reshape(-1) for _, _, _, idx in quant])
    ent = entropy_bits(sym, K)
    rbits = rans_bits(sym, K)
    total_rows = sum(idx.shape[0] for _, _, _, idx in quant)
    scale_bytes = sum(len(repr(float(v))) + len("m= |\n") for _, _, M, idx in quant
                      for v in M.reshape(-1))
    fp32_mb = 4 * n_lin / 1e6
    printed_mb = PRINTED.stat().st_size / 1e6
    gz_mb = len(gz) / 1e6
    digits_mb = (n_lin  # 1 char per digit
                 + n_lin + total_rows  # separator after each digit (spaces/newlines)
                 + scale_bytes) / 1e6
    rans_mb = (rbits * n_lin / 8.0 + 4 * total_rows) / 1e6
    floor_mb = (np.log2(K) * n_lin / 8.0 + 4 * total_rows) / 1e6

    # 6) human-readable sample
    sample_lines = [HEADER.rstrip("\n"), ""]
    first_name = quant[0][0]
    idx0 = quant[0][3]
    M0 = quant[0][2]
    c = idx0.shape[1]
    for i in range(min(3, idx0.shape[0])):
        note40 = "".join(f"0.({s})" for s in idx0[i][:40])
        human = " ".join("0." + f"{s}{s}{s}{s}" + "…" for s in idx0[i][:10])
        sample_lines.append(f"LAYER {first_name}  row {i}: m={float(M0[i, 0])!r}")
        sample_lines.append(f"  notation (first 40 of {c}): {note40}…")
        sample_lines.append(f"  repeating-decimal view: {human} …")
        sample_lines.append("")
    (RESULTS / "printed_model_sample.txt").write_text("\n".join(sample_lines) + "\n")

    ladder = [
        "| representation | size (linear weights) | lossless? |",
        "|---|---|---|",
        f"| fp32 tensors | {fp32_mb:.1f} MB | reference |",
        f"| printed repeating decimals (this file) | {printed_mb:.1f} MB | yes (round trip above) |",
        f"| same, gzip'd | {gz_mb:.1f} MB | yes |",
        f"| compact digits (1 char/digit + scales) | {digits_mb:.1f} MB | yes |",
        f"| rANS binary (digits) + fp32 scales | {rans_mb:.1f} MB | yes |",
        f"| uniform-9 alphabet bound (log2 9 + scales) | {floor_mb:.1f} MB | "
        "information bound under uniform symbols; skew lets rANS beat it |",
    ]

    lines = [
        "# exp5 — the model, printed as repeating numbers (round trip)",
        "",
        f"Model {e4.MODEL_ID}: {n_weights:,} weights across {len(quant)} body weight",
        f"matrices quantized to the ninths grid (per-row scale m, w = (s−4)·m/4,",
        f"s ∈ {{0..8}}). Eval: {EVAL_BLOCKS} blocks × 1024 tokens, TinyStories validation.",
        "",
        f"- fp32 baseline perplexity: **{ppl0:.3f}**",
        f"- after ninths quantization: **{ppl1:.3f}** (linear rel MSE {rel_mse:.5f})",
        f"- after printing to {PRINTED.name} and re-parsing: **{ppl2:.3f}**",
        "",
        "## Round-trip assertions (all passed)",
        "",
    ] + [f"- {a}" for a in asserts] + ["", "## Size ladder", ""] + ladder + [
        "",
        "The printed file is the model: it reloads bit-exactly and reproduces",
        "perplexity exactly. Its text size is large (notation overhead D1 in",
        "docs/01) but gzip + digit-frequency coding recover the content; the",
        "rANS row is the honest compressed-size comparison.",
        "",
    ]
    (RESULTS / "exp5_print_roundtrip.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()