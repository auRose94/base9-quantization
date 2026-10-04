#!/usr/bin/env python3
"""exp12 — RQ4: is there EXPLOITABLE structure in the trained grid-model digit
streams — beyond symbol-frequency coding (which rANS already captures)?

If yes, the "repeating numbers" idea gains a LOSSLESS angle after all: a
context/predictive coder could go below symbol entropy. Inputs are the REAL
trained artifacts: results/qat_digits_{recipe,tern}.npz (exp11's frozen
models; 2-D arrays preserved — rows × cols per weight matrix).

Tests per stream:
  T1  autocorrelation: lag-1 within rows; larger lags (2..64) over the
      flattened stream (row-boundary contamination <1% at these lags)
  T2  order-1 Markov conditional entropy H(X_t | X_{t-1}) within rows, vs
      the marginal entropy H(X) that plain rANS already achieves
  T3  inter-row similarity: aligned match rate of row r vs row r+1 (and the
      within-row lag-1 baseline for comparison)
  T4  zero-structure control: lag-1 stats recomputed on a shuffled stream
Pre-registered: P24 — the within-row order-1 Markov gain over the marginal
H is < 10% (structure exists but is modest; frequency coding captures most).

Also estimated: achievable bytes for the recipe body (k=9 layers) with an
ideal order-1 coder (first symbol of each row at marginal cost, rest at H_cond).

Out: results/exp12_structure_scan.{csv,md}
"""
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
from rans import rans_bits, entropy_bits

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

ARTIFACTS = [("recipe", RESULTS / "qat_digits_recipe.npz"),
             ("tern", RESULTS / "qat_digits_tern.npz")]


def marginal_entropy(X, n_syms):
    return entropy_bits(X, n_syms)


def markov_cond_entropy(Xprev, Xnext, n_syms):
    """H(Xnext | Xprev) from the empirical joint, +1 Laplace smoothing."""
    K = n_syms * n_syms
    joint = np.bincount(Xprev.astype(np.int64) * n_syms + Xnext.astype(np.int64),
                        minlength=K).astype(np.float64) + 1.0
    pxy = joint / joint.sum()
    px = pxy.reshape(n_syms, n_syms).sum(axis=1)
    px = np.repeat(px, n_syms)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(pxy * np.log2(pxy / px))
    return float(np.nansum(h))


def run_stats(mat, rng):
    """mat: 2-D digit array (values 0..n-1). Returns dict of stats."""
    n_syms = int(mat.max()) + 1
    Xprev, Xnext = mat[:, :-1].reshape(-1), mat[:, 1:].reshape(-1)
    flat = mat.reshape(-1)
    p = np.bincount(flat, minlength=n_syms) / flat.size
    shuf = flat[rng.permutation(flat.size)]
    out = {
        "n_syms": n_syms,
        "H_marginal": marginal_entropy(flat, n_syms),
        "H_cond_lag1": markov_cond_entropy(Xprev, Xnext, n_syms),
        "lag1_match": float(np.mean(Xprev == Xnext)),
        "iid_lag1_expected": float(np.sum(p ** 2)),
        "row_align_match": float(np.mean(mat[:-1] == mat[1:])),
    }
    Xs_prev, Xs_next = shuf[:-1], shuf[1:]
    out["H_cond_shuffled"] = markov_cond_entropy(Xs_prev, Xs_next, n_syms)
    out["lag1_match_shuffled"] = float(np.mean(Xs_prev == Xs_next))
    lags = {}
    for lag in (2, 4, 8, 16, 32, 64):
        lags[lag] = float(np.mean(mat[:-lag].reshape(-1) == mat[lag:].reshape(-1))) \
            if lag < mat.shape[1] else 0.0
    out["lags"] = lags
    return out


def shape_for_key(key):
    """exp11's npz artifacts are flattened; restore (rows, cols) from the
    known TinyStories-33M inventory: per layer [q,k,v,out (768,768);
    c_fc (3072,768); c_proj (768,3072)], then wte (50257,768), wpe (2048,768)."""
    n = int(key.split("_")[0])
    k = int(key.split("k")[1])
    if k == 99:
        return (50257, 768) if n == 24 else (2048, 768)
    per = n % 6
    if per < 4:
        return (768, 768)
    return (3072, 768) if per == 4 else (768, 3072)


def main():
    rng = np.random.default_rng(42)
    rows = []
    for arm, path in ARTIFACTS:
        if not path.exists():
            print(f"missing artifact: {path} — skipping {arm}")
            continue
        npz = np.load(path)
        for kk in npz.files:
            flat = npz[kk]
            shape = shape_for_key(kk)
            if int(np.prod(shape)) != flat.size:
                print(f"shape mismatch for {kk}: {flat.size} vs {shape} — skipping")
                continue
            mat = flat.reshape(shape)
            st = run_stats(mat, rng)
            gain = st["H_marginal"] - st["H_cond_lag1"]
            gb = gain / st["H_marginal"] * 100 if st["H_marginal"] > 0 else 0.0
            rows.append(dict(arm=arm, layer=str(kk), n_syms=st["n_syms"],
                             h_marg=st["H_marginal"], h_cond=st["H_cond_lag1"],
                             gain_pct=gb,
                             lag1=st["lag1_match"], lag1_iid=st["iid_lag1_expected"],
                             lag1_shuf=st["lag1_match_shuffled"],
                             row_match=st["row_align_match"],
                             lags=str({l: round(v, 3) for l, v in st["lags"].items()})))
            print(f"{arm:<7} {kk:<10} H {st['H_marginal']:.3f} | H|lag1 {st['H_cond_lag1']:.3f} "
                  f"(−{gb:.1f}%) | lag1 match {st['lag1_match']:.3f} "
                  f"(iid {st['iid_lag1_expected']:.3f}, shuffled {st['lag1_match_shuffled']:.3f}) "
                  f"| row-align {st['row_align_match']:.3f}")

    with open(RESULTS / "exp12_structure_scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "layer", "n_syms", "h_marg",
                                           "h_cond", "gain_pct", "lag1", "lag1_iid",
                                           "lag1_shuf", "row_match", "lags"])
        w.writeheader()
        w.writerows(rows)

    if not rows:
        print("no streams scanned — check that results/qat_digits_*.npz exist")
        return
    gains = [r["gain_pct"] for r in rows]
    max_gain, mean_gain = max(gains), float(np.mean(gains))
    p24 = mean_gain < 10.0
    recipe_npz = np.load(ARTIFACTS[0][1])
    body_rows = [r for r in rows if r["arm"] == "recipe" and r["layer"].endswith("_k9")]
    tot_bits = sum(r["h_marg"] * recipe_npz[r["layer"]].size for r in body_rows) / 8 / 1e6
    tot_cond = sum(r["h_cond"] * recipe_npz[r["layer"]].size for r in body_rows) / 8 / 1e6

    lines = [
        "# exp12 — RQ4: exploitable structure in the trained digit streams?",
        "",
        "Inputs: exp11's frozen trained models (digit artifacts). All match-rates",
        "are within-row; shuffled controls sit at the iid level (sanity).",
        "",
        "| arm | layer | H marginal | H cond (lag1) | Markov gain | lag1 match (iid / shuf) | row-align |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['layer']} | {r['h_marg']:.3f} |"
                     f" {r['h_cond']:.3f} | {r['gain_pct']:.1f}% |"
                     f" {r['lag1']:.3f} ({r['lag1_iid']:.3f} / {r['lag1_shuf']:.3f}) |"
                     f" {r['row_match']:.3f} |")
    # recipe-arm byte estimates (body only: k9 layers)
    lines += [
        "",
        "## Byte estimate — ideal order-1 coder vs plain rANS (recipe body, k=9)",
        "",
        f"- rANS (frequency coding): {tot_bits:.1f} MB",
        f"- ideal order-1 Markov coder: {tot_cond:.1f} MB",
        f"- potential extra saving: {100*(1 - tot_cond/tot_bits):.1f}%",
        "",
        "## Pre-registered verdict",
        "",
        f"- P24 (order-1 Markov gain < 10%): **{'PASS' if p24 else 'FAIL'}** "
        f"(mean {mean_gain:.1f}%, max {max_gain:.1f}%)",
        "",
        "If the shuffled controls did NOT sit at their iid levels, these numbers",
        "are contaminated — check the lag1 columns before quoting.",
    ]
    (RESULTS / "exp12_structure_scan.md").write_text("\n".join(lines))
    print("wrote results/exp12_structure_scan.md")


if __name__ == "__main__":
    main()