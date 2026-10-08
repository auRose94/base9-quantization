#!/usr/bin/env python3
"""eq0 — RQ1a: is there structure for an 'equation' to exploit?

For every 2-D weight tensor of stories260K (and shape-matched iid-Gaussian
controls) measure:
  * DCT-II coefficient-energy concentration: share of total energy held by the
    top f-fraction of |coefficients| (f = 0.10 headline, full curve saved);
  * SVD spectrum: rank holding 99% of squared singular energy (as % of min-dim).

Pre-registered predictions (RESEARCH_LOG session 1):
  P1: >=80% of matrices have top-10%-|DCT| energy share >= 90%, and the mean
      real share is >= 3x the control share.  P2: >=80% of matrices have
      rank@99% energy <= 50% of min-dim (iid controls: < 20% meeting it).
Verdicts: printed PASS/FAIL + results/eq0_verdicts.json.

Outputs: results/eq0_summary.csv, results/eq0_spectra.png.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eq_lib import RESULTS, dct2, load_state_dict  # noqa: E402

FRACTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEED = 0


def energy_curve(X):
    """Share (cumulative) of squared energy held by top-f fraction of |coef|."""
    e = (X.double() ** 2).flatten().numpy()
    e = np.sort(e)[::-1]
    tot = e.sum()
    cum = np.cumsum(e) / tot
    return cum                                    # aligned with FRACTIONS via searchsorted


def share_at(cum, f):
    k = max(1, int(round(f * len(cum))))
    return float(cum[k - 1])


def rank_at(s2, pct=0.99):
    """Smallest rank r with sum of top-r squared singulars >= pct of total."""
    return int(np.searchsorted(np.cumsum(s2), pct * s2.sum()) + 1)


def main():
    torch.manual_seed(SEED)
    sd, ma, ck = load_state_dict()
    rows, curves, svd_curves, svd_ctrl_curves = [], [], {}, {}
    matrices = {k: v for k, v in sd.items() if v.dim() == 2}
    print(f"[eq0] {len(matrices)} 2-D tensors (params total "
          f"{sum(v.numel() for v in sd.values()):,})\n")

    for name, W in matrices.items():
        W = W.double()
        n, m = W.shape
        # --- real ---
        X = dct2(W)
        cum = energy_curve(X)
        top10_real = share_at(cum, 0.10)
        s = torch.linalg.svdvals(W).numpy() ** 2
        s = np.sort(s)[::-1]
        rank99 = rank_at(s) / min(n, m)
        rank99_frac = rank99 / min(n, m)
        cum_e_svd = np.cumsum(s) / s.sum()
        svd_curves[name] = np.interp(FRACTIONS, np.arange(1, len(cum_e_svd) + 1) / len(cum_e_svd), cum_e_svd)
        # --- iid control, same shape + std ---
        C = torch.randn_like(W) * W.std()
        cum_c = energy_curve(dct2(C))
        top10_ctrl = share_at(cum_c, 0.10)
        s_c = torch.linalg.svdvals(C).numpy() ** 2
        s_c = np.sort(s_c)[::-1]
        rank99_c = rank_at(s_c) / min(n, m)
        cum_e_svd_c = np.cumsum(s_c) / s_c.sum()
        svd_ctrl_curves[name] = np.interp(FRACTIONS, np.arange(1, len(cum_e_svd_c) + 1) / len(cum_e_svd_c), cum_e_svd_c)
        curves.append((name, np.interp(FRACTIONS,
                                       np.arange(1, len(cum) + 1) / len(cum), cum),
                       np.interp(FRACTIONS, np.arange(1, len(cum_c) + 1) / len(cum_c), cum_c)))
        rows.append(dict(name=name, shape=f"{n}x{m}", params=n * m,
                         dct_top10_share=round(top10_real, 6),
                         dct_top10_share_ctrl=round(top10_ctrl, 6),
                         rank99_frac=round(rank99_frac, 4),
                         rank99_frac_ctrl=round(rank99_c, 4)))
        print(f"  {name:42s} {n:4d}x{m:<4d} DCT-top10% {top10_real:7.4f} "
              f"(ctrl {top10_ctrl:7.4f})  rank99 {rank99_frac:6.3f} (ctrl {rank99_c:6.3f})")

    # ---------------- verdicts ----------------
    frac_hi = sum(r["dct_top10_share"] >= 0.90 for r in rows) / len(rows)
    r_ctrl = float(np.mean([r["dct_top10_share"] / r["dct_top10_share_ctrl"] for r in rows]))
    r_hi = sum(r["rank99_frac"] <= 0.5 for r in rows) / len(rows)
    c_hi = sum(r["rank99_frac_ctrl"] <= 0.5 for r in rows) / len(rows)
    v1 = dict(prediction="P1", cond1=round(frac_hi, 3), cond2=round(r_ctrl, 3),
              verdict="PASS" if (frac_hi >= 0.8 and r_ctrl >= 3.0) else "FAIL")
    v2 = dict(prediction="P2", real_rank_meets=round(r_hi, 3), ctrl_rank_meets=round(c_hi, 3),
              verdict="PASS" if (r_hi >= 0.8 and c_hi < 0.2) else "FAIL")
    print(f"\n[eq0] P1 ({v1['verdict']}): matrices with DCT-top10% share>=90%: "
          f"{frac_hi:.3f}; mean real/control concentration {r_ctrl:.2f}x")
    print(f"[eq0] P2 ({v2['verdict']}): matrices with rank99<=50% min-dim: real "
          f"{r_hi:.3f} vs control {c_hi:.3f}")

    with open(os.path.join(RESULTS, "eq0_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, "eq0_verdicts.json"), "w") as f:
        json.dump([v1, v2], f, indent=2)

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fr = np.array(FRACTIONS)
    real = np.array([c[1] for c in curves])
    ctrl = np.array([c[2] for c in curves])
    axes[0].loglog(fr, real.mean(0), lw=2.2, label=f"real weights ({len(curves)} matrices)", color="#b45309")
    axes[0].fill_between(fr, real.min(0), real.max(0), color="#b45309", alpha=0.18)
    axes[0].loglog(fr, ctrl.mean(0), lw=2, ls="--", label="iid-Gaussian control", color="#64748b")
    axes[0].set_xlabel("fraction of |DCT coeffs| kept (top order)")
    axes[0].set_ylabel("share of squared energy")
    axes[0].axvline(0.1, color="#f59e0b", lw=1, alpha=0.7)
    axes[0].axhline(0.9, color="#f59e0b", lw=1, alpha=0.7)
    axes[0].set_title(f"DCT-II energy concentration  (P1: {v1['verdict']})")
    axes[0].legend(fontsize=8)

    svd_real = np.array(list(svd_curves.values()))
    svd_ctrl = np.array(list(svd_ctrl_curves.values()))
    # control rank curves averaged per shape class
    axes[1].semilogx(fr, svd_real.mean(0), lw=2.2, color="#b45309", label="real")
    axes[1].fill_between(fr, svd_real.min(0), svd_real.max(0), color="#b45309", alpha=0.18)
    axes[1].semilogx(fr, svd_ctrl.mean(0), lw=2, ls="--", color="#64748b", label="iid control")
    axes[1].axhline(0.99, color="#0f766e", lw=1, ls=":", label="99% energy")
    axes[1].set_xlabel("normalized rank s1..sr (fraction of min-dim)")
    axes[1].set_ylabel("cumulative energy")
    axes[1].set_title(f"SVD spectra (P2: {v2['verdict']})")
    axes[1].legend(fontsize=8)

    names = [r["name"] for r in rows]
    top10 = [r["dct_top10_share"] for r in rows]
    top10c = [r["dct_top10_share_ctrl"] for r in rows]
    yx = np.arange(len(names))
    axes[2].barh(yx, top10, height=0.7, color="#b45309", label="real")
    axes[2].barh(yx, top10c, height=0.28, color="#64748b", label="ctrl")
    axes[2].axvline(0.9, color="#f59e0b", lw=1, alpha=0.7)
    axes[2].set_yticks(yx, [n.replace("layers.", "L").replace(".weight", "").replace("feed_forward", "ffn")
                            for n in names], fontsize=6)
    axes[2].invert_yaxis()
    axes[2].set_title("top-10% |DCT| energy share per matrix")
    axes[2].legend(fontsize=8)
    fig.suptitle("eq0 — weight structure vs iid controls (stories260K)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "eq0_spectra.png"), dpi=130, bbox_inches="tight")
    print("[eq0] wrote results/eq0_summary.csv, eq0_verdicts.json, eq0_spectra.png")


if __name__ == "__main__":
    main()