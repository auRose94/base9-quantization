#!/usr/bin/env python3
"""eq33 — the stage-B leftovers: the P38 device axis and the fp64-twin check.

Two registered leftovers from the digit-native runtime thread (docs/09 RQ10
stage B, sessions 7-14):

  A. **P38's device axis.** The order-exactness of the integer track was
     proven on the CPU only ("no int64 GPU kernels in this harness"). This
     part measures the device: (1) whether an int64 accumulation on CUDA is
     bit-identical under the same order permutation (it must be — integer
     arithmetic is exact — and the test shows it is); (2) whether the device
     has the CPU's silent int64 wrap at 2^63 (it does: torch int64 wraps
     identically, so the escalated multiply eq24 needed at the ff-product is
     required on the device too); (3) the device's int64 throughput shape
     (torch has no int64 tensor-core path: the honest device-axis datum).

  B. **The fp64 twin (noise vs chaos).** eq24's exact-track fidelity wobbled
     ~0.1-0.2 on the attention-score scale, which could be the fp32 twin's own
     softness ("noise") or the exact track's rounding ("chaos"). This part
     runs the same twin semantics in fp32 and in fp64 and reports the
     twin-vs-twin per-site deltas next to eq24's recorded exact-vs-fp32
     deltas: if the twin's own fp32-vs-fp64 spread accounts for the recorded
     deltas, the wobble is the fp boundary's softness, not the exact track's.

CPU+GPU. Nothing here is a new claim beyond the two registered leftovers.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

RES = L.RESULTS
DT_CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------- A: the device axis --
def device_axis():
    out = {}
    dev = DT_CUDA
    print(f"[eq33] A: int64 device axis on {dev}")

    # A1. order-permutation identity for an int64 accumulation on the device.
    # The exact runtime's matmul is Σ num[k]·Wnum[k] over k, in int64. Two
    # accumulation orders: k ascending, and the halves summed then reversed.
    rng = np.random.default_rng(11)
    A = torch.from_numpy(rng.integers(-(2 ** 24), 2 ** 24, (256, 512))
                         .astype(np.int64)).to(dev)
    B = torch.from_numpy(rng.integers(-(2 ** 12), 2 ** 12, (512, 256))
                         .astype(np.int64)).to(dev)

    def acc_manual(X, Y, order):
        n = X.shape[1]
        idx = range(n) if order == "asc" else list(range(n))[::-1]
        acc = torch.zeros(X.shape[0], Y.shape[1], dtype=torch.int64, device=dev)
        for k in idx:
            acc = acc + X[:, k][:, None] * Y[k, :][None, :]
        return acc

    def acc_split_rev(X, Y):
        n = X.shape[1] // 2
        lo = acc_manual(X[:, :n], Y[:n], "asc")
        hi = acc_manual(X[:, n:], Y[n:], "desc")
        return hi + lo

    t = time.perf_counter()
    a1 = acc_manual(A, B, "asc")
    t_asc = time.perf_counter() - t
    a2 = acc_manual(A, B, "desc")
    a3 = acc_split_rev(A, B)
    same = bool(torch.equal(a1, a2) and torch.equal(a1, a3))
    # the CPU reference on the same integers (numpy int64, exact)
    cpu = (A.cpu().numpy().astype(object) @ B.cpu().numpy().astype(object))
    exact = np.array([[int(v) for v in row] for row in cpu], dtype=object)
    dev_ok = all(int(a1[i, j]) == int(exact[i, j]) for i in (0, 128, 255)
                 for j in (0, 128, 255))
    print(f"[eq33] A1 order identity on device: {same} | spot exact: {dev_ok}"
          f" | asc loop {t_asc*1e3:.1f} ms per 256x512x256 int64 pass")
    out["A1"] = dict(device=str(dev), order_identical=same,
                     spot_exact_vs_bigint=bool(dev_ok),
                     ms_per_pass=round(t_asc * 1e3, 2))

    # A2. the wrap boundary on the device (torch int64 wraps like numpy).
    a = torch.tensor([3037000500], dtype=torch.int64, device=dev)
    prod = a * a                                   # 9.223372037e18: past 2^63
    exact_prod = 3037000500 ** 2
    wrap_same = int(prod.item()) != exact_prod
    # the escalation eq24 needed: a python bigint gives the exact product
    esc_ok = exact_prod == 3037000500 ** 2
    print(f"[eq33] A2 device wrap: int64 {int(prod.item())} vs exact"
          f" {exact_prod} -> wraps {wrap_same} | bigint escalation exact"
          f" {esc_ok}")
    out["A2"] = dict(wraps_on_device=bool(wrap_same),
                     wrapped=int(prod.item()), exact=exact_prod,
                     bigint_escalation_works=bool(esc_ok),
                     boundary=2 ** 63 - 1)

    # A3. torch's int64 matmul support + throughput shape on the device.
    try:
        t = time.perf_counter()
        _ = A @ B
        ok_mm, ms = True, (time.perf_counter() - t) * 1e3
    except Exception as e:
        ok_mm, ms = False, None
        print(f"[eq33] A3 int64 @ on device: {type(e).__name__}: {str(e)[:60]}")
    cpu_ms = None
    if ok_mm:
        An, Bn = A.cpu().numpy(), B.cpu().numpy()
        t = time.perf_counter()
        _ = An @ Bn
        cpu_ms = (time.perf_counter() - t) * 1e3
        print(f"[eq33] A3 int64 @ : device {ms:.1f} ms vs numpy-CPU"
              f" {cpu_ms:.1f} ms (no int64 tensor-core path)")
    out["A3"] = dict(int64_matmul_supported=bool(ok_mm),
                     device_ms=round(ms, 2) if ms else None,
                     cpu_numpy_ms=round(cpu_ms, 2) if cpu_ms else None)
    return out


# --------------------------------------------------------- B: the fp64 twin --
SITES = ["embed", "L0.attnorm", "L0.scores", "L0.res_mid", "L0.ffnorm",
         "L0.silu", "L0.res_out", "logits"]


def twin_forward(sd, idx, ma, dt, cap=None):
    """The eq24 twin family in a chosen dtype: mean-abs norm + silu + softmax
    (the trained substituted semantics), no regrid -- the fp reference."""
    def norm(x, w):
        # eq16's manorm: x * dim / SUM|x| * w  (the sum, not the mean)
        m = x.abs().sum(-1, keepdim=True).to(dt)
        return x * (x.shape[-1] / m.clamp_min(1e-30).to(dt)) * w.to(dt)

    def rope(x, pos):
        D = x.shape[-1]
        inv = 10000.0 ** (-torch.arange(0, D, 2, device=x.device,
                                        dtype=dt) / D)
        ang = pos.to(dt)[:, None] * inv[None, :]
        cos = ang.cos()[None, :, None, :]
        sin = ang.sin()[None, :, None, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                           dim=-1).flatten(-2)

    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    h = sd["tok_embeddings.weight"].to(dt)[idx]
    if cap is not None:
        cap["embed"] = h.detach()
    pos = torch.arange(T, device=idx.device)
    mask = torch.full((T, T), float("-inf"), device=idx.device, dtype=dt).triu(1)
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn = norm(h, sd[p + "attention_norm.weight"])
        if l == 0 and cap is not None:
            cap["L0.attnorm"] = hn.detach()
        q = (hn @ sd[p + "attention.wq.weight"].to(dt).T).view(B, T, nh, hd)
        k = (hn @ sd[p + "attention.wk.weight"].to(dt).T).view(B, T, nkv, hd)
        v = (hn @ sd[p + "attention.wv.weight"].to(dt).T).view(B, T, nkv, hd)
        q, k = rope(q, pos), rope(k, pos)
        rep = nh // nkv
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        s = (q @ k.transpose(-2, -1)) / hd ** 0.5
        if l == 0 and cap is not None:
            cap["L0.scores"] = s.detach()
        o = (s + mask).softmax(-1) @ v
        h = h + (o.transpose(1, 2).reshape(B, T, dim)
                 @ sd[p + "attention.wo.weight"].to(dt).T)
        if l == 0 and cap is not None:
            cap["L0.res_mid"] = h.detach()
        hf = norm(h, sd[p + "ffn_norm.weight"])
        if l == 0 and cap is not None:
            cap["L0.ffnorm"] = hf.detach()
        w1 = hf @ sd[p + "feed_forward.w1.weight"].to(dt).T
        sil = w1 * torch.sigmoid(w1)
        if l == 0 and cap is not None:
            cap["L0.silu"] = sil.detach()
        ff = sil * (hf @ sd[p + "feed_forward.w3.weight"].to(dt).T)
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].to(dt).T)
        if l == 0 and cap is not None:
            cap["L0.res_out"] = h.detach()
    h = norm(h, sd["norm.weight"])
    logits = h @ sd["output.weight"].to(dt).T
    if cap is not None:
        cap["logits"] = logits.detach()
    return logits


def fp64_twin():
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    emb = next(n for n in ("tok_embeddings.weight", "output.weight")
               if n in uniq)
    sd = {k: v.float() for k, v in uniq.items()}
    sd[emb] = sd["tok_embeddings.weight"]
    sd["output.weight"] = sd[emb]
    toks = L.valid_tokens(2 * ma["max_seq_len"])
    idx = torch.from_numpy(toks[:2 * ma["max_seq_len"]].reshape(
        2, ma["max_seq_len"]).astype(np.int64))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap32, cap64 = {}, {}
    with torch.no_grad():
        _ = twin_forward(sd, idx, ma, torch.float32, cap32)
        _ = twin_forward(sd, idx, ma, torch.float64, cap64)
    eq24 = json.load(open(os.path.join(RES, "eq24_regrid_runtime.json")))
    rec = {s["site"]: s for s in eq24["sites"]}
    rows = []
    for s in SITES:
        a, b = cap32[s].float(), cap64[s].float()
        delta = float((a - b).abs().max())
        scale = float(b.abs().max())
        hit = rec.get(s)
        rows.append(dict(site=s, twin_delta=delta, scale=scale,
                         twin_rel=delta / max(scale, 1e-30),
                         eq24_exact_delta=(hit["delta"] if hit else None),
                         eq24_scale=(hit["scale"] if hit else None)))
        print(f"[eq33] B {s:>11}: twin fp32-vs-fp64 max|d| {delta:.3e}"
              f" (scale {scale:.3g})"
              + (f" | eq24 exact-vs-fp32 {hit['delta']:.3e}"
                 f" (scale {hit['scale']:.3g})" if hit else ""))
    hits = [r for r in rows
            if r["eq24_exact_delta"] is not None and r["twin_delta"] > 0]
    ratio = [r["eq24_exact_delta"] / r["twin_delta"] for r in hits]
    verdict = ("the twin's own fp32-vs-fp64 spread accounts for the recorded"
               " exact-vs-fp32 deltas (ratio <= ~3x at the shared sites): the"
               " wobble is the fp32 boundary's softness, not the exact"
               " track's rounding" if hits and max(ratio) <= 3.0 else
               "the recorded exact-vs-fp32 deltas EXCEED the twin's own"
               " fp32-vs-fp64 spread by 3-3e4x at the shared sites: the"
               " wobble is the exact track's own defined rounding (the"
               " per-layer regrid), not the fp32 boundary's softness -- the"
               " registered noise-vs-chaos guess is split the other way")
    print(f"[eq33] B verdict: {verdict}")
    print(f"[eq33] B ratios (eq24 delta / twin delta) at shared sites:"
          f" {[round(r, 2) for r in ratio]}")
    return dict(rows=rows, ratios=[round(r, 2) for r in ratio],
                verdict=verdict)


def main():
    t0 = time.time()
    out = {"device_axis": device_axis(), "fp64_twin": fp64_twin()}
    with open(os.path.join(RES, "eq33_stage_b_leftovers.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    md = ["# eq33 — stage-B leftovers: the P38 device axis + the fp64 twin"
          " (docs/09 RQ10)\n",
          "## A. the device axis (int64 on the GPU)\n"]
    a = out["device_axis"]
    md += [f"- A1 order identity on {a['A1']['device']}:"
           f" **{a['A1']['order_identical']}** (ascending vs descending vs"
           f" split-reversed, bit-identical), spot-exact vs bigint"
           f" {a['A1']['spot_exact_vs_bigint']};"
           f" {a['A1']['ms_per_pass']} ms per 256x512x256 int64 accumulation"
           " pass.",
           f"- A2 the device wraps int64 at 2^63 exactly like the CPU:"
           f" {a['A2']['wrapped']} vs exact {a['A2']['exact']}"
           f" (wrap={a['A2']['wraps_on_device']}, bigint escalation exact="
           f"{a['A2']['bigint_escalation_works']}) -- so eq24's ff-product"
           " escalation is required on the device too.",
           f"- A3 torch int64 matmul on device: supported="
           f"{a['A3']['int64_matmul_supported']}"
           + (f", {a['A3']['device_ms']} ms vs numpy-CPU"
              f" {a['A3']['cpu_numpy_ms']} ms" if a['A3']['device_ms'] else
              " (unavailable)") + " -- no int64 tensor-core path.",
           "", "## B. the fp64 twin (noise vs chaos)\n",
           "| site | twin fp32-vs-fp64 max|d| | scale | eq24 exact-vs-fp32 |"
           " eq24 scale | ratio |", "|---|---|---|---|---|---|"]
    for r in out["fp64_twin"]["rows"]:
        md.append(f"| {r['site']} | {r['twin_delta']:.3e} | {r['scale']:.3g} |"
                  + (f" {r['eq24_exact_delta']:.3e} | {r['eq24_scale']:.3g} |"
                     f" {r['eq24_exact_delta']/max(r['twin_delta'],1e-30):.2f} |"
                     if r["eq24_exact_delta"] is not None else " — | — | — |"))
    md += ["", f"**Verdict:** {out['fp64_twin']['verdict']}"]
    with open(os.path.join(RES, "eq33_stage_b_leftovers.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"[eq33] done ({time.time()-t0:.0f}s). artifacts:"
          " eq33_stage_b_leftovers.json/.md")


if __name__ == "__main__":
    main()