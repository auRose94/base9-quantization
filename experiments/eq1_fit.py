#!/usr/bin/env python3
"""eq1 — RQ1b: which 'equation' survives quality control?

Reconstruct representations of stories260K and measure paired val loss on the
same 384 x 512-token windows of TinyStories-valid (paired discipline per
base9's eval_harness). Schemes per 2-D tensor (1-D norms carried fp16):

  uniform   W ≈ quant(W, b) symmetric uniform, b bits/param. (no equation, fair baseline)
  svd-r     W ≈ U_r diag(sigma) V_r^T ; sigma/U/V fp16 (learned basis = shipped data)
  dct-rect  W ≈ IDCT( DCT(W) with [0:kx,0:ky] low-pass corner kept, b-bit coefs )  (index-free!)
  dct-topk  W ≈ IDCT( DCT(W) with top-K |coef| kept, quantized b-bit )

Bit accounting is RAW (conservative; no entropy coding credited): kept
coefficient costs b bits (topK adds log2(numel) for its basis index), svd costs
16 bits per (U/V/sigma) entry, plus 32-bit scale/header.

Also: RQ3 probe — are the dominant singular directions shared across layers?
(If u1/v1 cosine ≈ 1 across layers, the equation's basis could be shared.)

Pre-registered P3: exists a point at <= 6 b/p with paired loss <= 1.10x fp32,
and a point at <= 4 b/p with <= 1.30x. Verdict: results/eq1_verdicts.json.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_WINDOWS = 384


# ---------------------------------------------------------------- schemes

def recon_uniform(W, b):
    q, s = L.quantize_sym(W, b)
    return L.dequant(q, s).float(), int(W.numel() * b) + 32


def recon_svd_rank(W, r):
    r = max(1, min(int(r), min(W.shape)))
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    Ur, sr, Vhr = U[:, :r], S[:r], Vh[:r]
    bits = (Ur.numel() + Vhr.numel() + r) * 16 + 32
    Wh = (Ur.half().double() @ torch.diag(sr.half().double())) @ Vhr.half().double()
    return Wh.float(), bits


def recon_dct_rect(W, frac, b):
    n, m = W.shape
    kx = max(1, round(frac * n))
    ky = max(1, round(frac * m))
    X = L.dct2(W.double())
    q, s = L.quantize_sym(X[:kx, :ky], b)
    Xhat = torch.zeros_like(X)
    Xhat[:kx, :ky] = L.dequant(q, s)
    return L.idct2(Xhat).float(), L.tensor_bits_schemeA(kx, ky, b, W.shape)


def recon_dct_topk(W, frac, b):
    X = L.dct2(W.double())
    numel = X.numel()
    k = max(1, round(frac * numel))
    idx = torch.topk(X.abs().flatten(), k).indices
    q, s = L.quantize_sym(X.flatten()[idx], b)
    Xhat = torch.zeros_like(X).flatten()
    Xhat[idx] = L.dequant(q, s)
    return L.idct2(Xhat.reshape(W.shape)).float(), L.tensor_bits_schemeB(*W.shape, k, b)


def fit_all(sd, scheme, frac, b):
    """-> (reconstructed state dict, total bits). Tied tensors fitted once."""
    out, bits = {}, 0
    done = {}
    for k, W in sd.items():
        if id(W) in done:
            out[k] = done[id(W)]
            continue
        if W.dim() != 2:
            Wh = W.half().float()  # norms etc as fp16 variables
            kb = W.numel() * 16
        else:
            if scheme == "uniform":
                Wh, kb = recon_uniform(W.double(), b)
            elif scheme == "svd":
                Wh, kb = recon_svd_rank(W, frac)
            elif scheme == "dct-rect":
                Wh, kb = recon_dct_rect(W, frac, b)
            elif scheme == "dct-topk":
                Wh, kb = recon_dct_topk(W, frac, b)
            else:
                raise ValueError(scheme)
        done[id(W)] = Wh
        out[k] = Wh
        bits += kb
    return out, bits


# ---------------------------------------------------------------- run

def main():
    sd0, ma, ck = L.load_state_dict()
    P = sum(v.numel() for v in L.unique_parameters(sd0).values())
    tokens = L.valid_tokens(N_WINDOWS * ma["max_seq_len"]).astype(np.int64)
    sd0d = L.to_device(sd0, DEV)
    fp32 = L.eval_loss(sd0d, ma, tokens, DEV, N_WINDOWS)
    fp16 = L.eval_loss({k: v.half().float() for k, v in sd0d.items()}, ma, tokens, DEV, N_WINDOWS)
    print(f"[eq1] fp32 anchor loss {fp32[0]:.4f}±{fp32[1]:.4f}  (ckpt best_val_loss "
          f"{float(ck['best_val_loss']):.4f})")
    assert fp32[0] < 2.0, "from-scratch forward does not reproduce model quality — STOP"
    assert fp16[0] < fp32[0] + 0.15, "fp16 anchor unexpectedly bad"

    pts = [dict(scheme="fp32", frac=None, b=32, bits=P * 32, loss=fp32[0], sem=fp32[1]),
           dict(scheme="fp16", frac=None, b=16, bits=P * 16, loss=fp16[0], sem=fp16[1])]
    sweeps = ([("uniform", b, None) for b in (8, 7, 6, 5, 4, 3)]
              + [("svd", 16, r) for r in (1, 2, 4, 8, 16, 32)]
              + [(s, 8, f) for s in ("dct-rect", "dct-topk") for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)]
              + [("dct-topk", 23, 0.05), ("dct-topk", 8, 1.0)])  # precision attribution + machinery control
    for scheme, b, frac in sweeps:
        sdb, bt = fit_all(sd0, scheme, frac, b)
        ev = L.eval_loss(L.to_device(sdb, DEV), ma, tokens, DEV, N_WINDOWS)
        pts.append(dict(scheme=scheme, frac=frac, b=b, bits=bt, loss=ev[0], sem=ev[1]))

    for p in pts:
        p["bpp"] = round(p["bits"] / P, 3)
        p["loss"] = round(p["loss"], 4)
        p["sem"] = round(p["sem"], 4)
    pts_sorted = sorted(pts, key=lambda p: (p["scheme"], p["bpp"]))
    print(f"\n{'scheme':10s} {'frac/r':>7s} {'bpp':>7s} {'loss':>8s} {'sem':>6s} {'x-fp32':>7s}")
    for p in pts_sorted:
        print(f"{p['scheme']:10s} {str(p['frac']):>7s} {p['bpp']:7.2f} {p['loss']:8.4f} "
              f"{p['sem']:6.4f} {p['loss'] / fp32[0]:7.2f}x")

    ok6 = [p for p in pts if p["bpp"] <= 6 and p["loss"] <= 1.10 * fp32[0]]
    ok4 = [p for p in pts if p["bpp"] <= 4 and p["loss"] <= 1.30 * fp32[0]]
    v3 = dict(prediction="P3",
              qualifying_at_6bpp=sorted([(p["scheme"], p["bpp"], p["loss"]) for p in ok6], key=lambda t: t[1]),
              qualifying_at_4bpp=sorted([(p["scheme"], p["bpp"], p["loss"]) for p in ok4], key=lambda t: t[1]),
              verdict="PASS" if ok6 and ok4 else "FAIL")
    print(f"\n[eq1] P3 ({v3['verdict']}): <=6 b/p & +10%: {len(ok6)} point(s); "
          f"<=4 b/p & +30%: {len(ok4)} point(s)")

    # -------- RQ3 probe: dominant singular direction shared across layers? --------
    groups = {}
    for k, W in sd0.items():
        if W.dim() == 2 and k.startswith("layers."):
            name = ".".join(k.split(".")[2:])
            groups.setdefault(name, []).append(k)
    shared = {}
    for grp, keys in sorted(groups.items()):
        if len(keys) < 2:
            continue
        dirs = []
        for k in keys:
            U, S, Vh = torch.linalg.svd(sd0[k].double(), full_matrices=False)
            dirs.append((U[:, 0], Vh[0]))
        uu = [float(abs(dirs[i][0] @ dirs[j][0])) for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
        vv = [float(abs(dirs[i][1] @ dirs[j][1])) for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
        shared[grp] = dict(mean_u1_cos=round(float(np.mean(uu)), 3), mean_v1_cos=round(float(np.mean(vv)), 3))
        print(f"[eq1] shared-basis {grp:24s} mean|cos(u1)| {np.mean(uu):5.3f}  mean|cos(v1)| {np.mean(vv):5.3f}")

    # ---------------- save artifacts ----------------
    with open(os.path.join(L.RESULTS, "eq1_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "frac", "b", "bpp", "loss", "sem"])
        w.writeheader()
        for p in pts_sorted:
            w.writerow({kk: p[kk] for kk in w.fieldnames})
    with open(os.path.join(L.RESULTS, "eq1_verdicts.json"), "w") as f:
        json.dump([v3, dict(prediction="RQ3probe_shared_basis", data=shared, verdict="INFO")], f, indent=2)

    # showcase reconstructions for eq2 (samples) and eq4 (GPU synthesis)
    for tag, (scheme, frac, b) in [("best", ("dct-topk", 0.05, 8)),
                                   ("uniform6", ("uniform", None, 6)),
                                   ("svd8", ("svd", 8, 16))]:
        sdb, _ = fit_all(sd0, scheme, frac, b)
        np.savez_compressed(os.path.join(L.RESULTS, f"eq1_recon_{tag}.npz"),
                            **{f"recon_{k}": W.cpu().numpy() for k, W in sdb.items()})
    plot(pts_sorted, fp32, fp16)
    print("[eq1] wrote results/eq1_results.csv, eq1_verdicts.json, eq1_curve.png, eq1_recon_*.npz")


def plot(pts, fp32, fp16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = {"uniform": ("#64748b", "o--"), "svd": ("#7c3aed", "s-"),
              "dct-rect": ("#0891b2", "^-"), "dct-topk": ("#b45309", "d-")}
    labels = {"svd": "svd-r (learned basis)", "uniform": "uniform quant (no equation)",
              "dct-rect": "DCT low-pass (index-free eq.)", "dct-topk": "DCT top-K (indexed eq.)"}
    for scheme, (c, st) in styles.items():
        pp = sorted([p for p in pts if p["scheme"] == scheme and p["b"] != 23], key=lambda q: q["bpp"])
        if pp:
            ax.errorbar([p["bpp"] for p in pp], [p["loss"] for p in pp],
                        yerr=[max(p["sem"], 1e-3) for p in pp], fmt=st, color=c, capsize=3,
                        label=labels[scheme])
    ax.axvline(16, color="#94a3b8", lw=1, ls=":", label="fp16 anchor (16 b/p)")
    ax.axhline(fp32[0], color="#dc2626", lw=1, ls=":", label=f"fp32 anchor {fp32[0]:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("representation size (bits per parameter, raw accounting)")
    ax.set_ylabel("val loss (384 paired windows)")
    ax.set_title("eq1 — equation-fit vs quality (stories260K)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(L.RESULTS, "eq1_curve.png"), dpi=130)


if __name__ == "__main__":
    main()