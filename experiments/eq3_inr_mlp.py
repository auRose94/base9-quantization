#!/usr/bin/env python3
"""eq3 — RQ3 scoping: a LEARNED 'equation' W ≈ gθ(ξ,η) (INR/NeRN-style field).

Fit per-matrix MLPs with Fourier-featurized grid coordinates on GPU (Adam,
full-batch) and compare against a DCT-topK fit consuming the SAME variable
bytes, per matrix. Answers: do learned fields beat fixed analytic bases at
equal size — before any functional/val-loss fitting (which is next session).
Outputs: results/eq3_table.csv, results/eq3_fieldmap.png, console log.
"""
import csv
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

DEV = "cuda"
TARGETS = ["layers.0.feed_forward.w1.weight", "layers.2.attention.wq.weight",
           "layers.4.feed_forward.w1.weight", "tok_embeddings.weight"]
HIDDENS = (16, 64)


def fourier_feat(xi, eta, Lf=6):
    # xi, eta: (N,) -> (N, 2+2*2L) features = [xi, eta, sin/cos(2^k * pi * x)]
    freqs = (2.0 ** torch.arange(Lf, device=xi.device)) * math.pi
    fx = xi[:, None] * freqs[None, :]
    fy = eta[:, None] * freqs[None, :]
    return torch.cat([xi[:, None], eta[:, None], fx.sin(), fx.cos(), fy.sin(), fy.cos()], 1)


def make_mlp(n_in, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, 1))


@torch.no_grad()
def param_bytes(net):
    return sum(p.numel() * 2 for p in net.parameters())  # fp16 deployment


def fit_mlp(W, hidden, steps=4000, lr=3e-3):
    torch.manual_seed(0)
    n, m = W.shape
    xi = torch.linspace(-1, 1, n, device=DEV)
    eta = torch.linspace(-1, 1, m, device=DEV)
    grid = torch.stack([xi[:, None].expand(n, m).flatten(),
                        eta[None, :].expand(n, m).flatten()], 1)
    feats = fourier_feat(grid[:, 0], grid[:, 1]).double()
    scale = float(W.abs().max())
    Wn = (W.to(DEV) / scale).double().flatten()
    net = make_mlp(feats.shape[1], hidden).to(DEV).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = None
    for t in range(steps + 1):
        pred = net(feats)[:, 0]
        loss = ((pred - Wn) ** 2).mean()
        if t % 500 == 0:
            r = float(((pred - Wn).norm() / Wn.norm()))
            if best is None or r < best[0]:
                best = (r, t)
            if t % 1000 == 0:
                print(f"    step {t:5d} rel-L2 {r:.4f}")
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(feats)[:, 0]
        rel = float(((pred - Wn).norm() / Wn.norm()))
        # deployment bytes: fp16-quantize theta, re-measure
        for p in net.parameters():
            p.copy_(p.half().double())
        pred = net(feats)[:, 0]
        rel_fp16 = float(((pred - Wn).norm() / Wn.norm()))
    Wh = (pred.reshape(n, m) * scale).cpu()
    return dict(rel=rel, rel_fp16=rel_fp16, bytes=param_bytes(net) + 4, Wh=Wh)


def main():
    sd0, ma, _ = L.load_state_dict()
    rows = []
    for name in TARGETS:
        W = sd0[name].double()
        n, m = W.shape
        print(f"[eq3] {name} ({n}x{m}, {n*m:,} params)")
        for H in HIDDENS:
            r = fit_mlp(W, H)
            # DCT-topk fit at the SAME bytes: k coefs * (8b + 4B idx) + scale
            k_bytes = r["bytes"]
            k = max(1, int(k_bytes / (4 + 1)))  # 4B float32 coef + 1B idx (b=8,logceil)
            kd = min(k, n * m)
            X = L.dct2(W)
            idx = torch.topk(X.abs().flatten(), kd).indices
            Xh = torch.zeros_like(X.flatten())
            Xh[idx] = X.flatten()[idx]
            Wh = L.idct2(Xh.reshape(n, m))
            rdct = float(((Wh - W).norm() / W.norm()))
            rows.append(dict(name=name, shape=f"{n}x{m}", hidden=H,
                             mlp_bytes=r["bytes"], mlp_rel=r["rel"], mlp_rel_fp16=r["rel_fp16"],
                             dct_bytes=kd * 5, dct_rel=rdct,
                             winner=("mlp" if r["rel_fp16"] < rdct else "dct")))
            print(f"   H={H:3d}: MLP {r['bytes']:,} B rel-L2 {r['rel']:.4f} "
                  f"(fp16-θ {r['rel_fp16']:.4f})  vs  DCT {kd*5:,} B rel-L2 {rdct:.4f} "
                  f"→ {rows[-1]['winner']} wins")
    with open(os.path.join(L.RESULTS, "eq3_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # field-map figure for the w1 matrix, best MLP
    name = TARGETS[0]
    r = fit_mlp(sd0[name].double(), 64)
    W = sd0[name].double().cpu().numpy(); Wh = r["Wh"].numpy()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    vmax = np.abs(W).max()
    im0 = axs[0].imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axs[1].imshow(Wh, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    im2 = axs[2].imshow(np.abs(W - Wh), aspect="auto", cmap="magma")
    axs[0].set_title(f"{name} (fp32)"); axs[1].set_title(f"MLP field g(ξ,η), rel-L2 {r['rel_fp16']:.4f}")
    axs[2].set_title("|error|")
    fig.colorbar(im0, ax=axs[0], fraction=0.046); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(L.RESULTS, "eq3_fieldmap.png"), dpi=130)
    print("[eq3] wrote results/eq3_table.csv, eq3_fieldmap.png")


if __name__ == "__main__":
    main()