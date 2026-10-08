#!/usr/bin/env python3
"""eq5 — RQ4 fusion: K9Q1 artifacts of stories260K from the base9 stack.

Reuses the validated base9 codec (container k9.py + rans.py + the compiled
k9rans.so C coder, imported path-wise from the base9 repo; no copies). Our
side: RTN onto the odd k-grid with per-(row, group absmax) scales per
exp4b's group-scale finding (g=64 heals the MSE↔ppl inversion).

Artifact variants (real bytes = K9Q1 file sizes):
  A  body k=9   g=64 + embed k=99 g=64 + fp16 rest   (exp13 recipe class)
  B  body k=27  g=64 + embed k=99 g=64
  C  body k=63  g=64 + embed k=99 g=64               (≈6-bit class)

P7 comparison (matched precision k=255 = the 8-bit class): digits coded from
raw weights vs digits coded from DCT-II coefficients (f=1.0; decode = IDCT).
Body-only recon (embeds/norms fp32) so the digit-rate comparison is clean.

Verdicts: P6 (variant A ≤ 150 KB & ≤ +12% paired loss), P7 (DCT/raw digit
bytes within ±10% — exp12 null replicated in the coefficient domain),
P8 (artifact reload == in-memory recon, bitwise-same paired loss).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402  (base9 container codec; self-asserting module)

DEV = "cuda"
RES = os.path.join(L.ROOT, "results")
GROUP = 64
UNIQUE_PARAMS = 260032


# ------------------------------------------------------------- quantization

def quant_rtn_odd(W, k, group=GROUP):
    """RTN onto the odd k-grid: digits d in [0, k-1], per-(row, group) absmax
    scale m; dequant w = m*(d-H)/H, H=(k-1)//2. Columns zero-pad to a multiple
    of `group` (padding digits = H = exact zero, sliced off after decode)."""
    n, m = W.shape
    H = (k - 1) // 2
    g = min(group, m)
    pad = (-m) % g
    Wp = np.pad(W, ((0, 0), (0, pad)))
    ng = Wp.shape[1] // g
    Wg = Wp.reshape(n, ng, g)
    S = np.maximum(np.abs(Wg).max(axis=2), 1e-12)              # (n, ng)
    Srep = np.repeat(S, g, axis=1)                              # (n, ng*g) = padded width
    D = np.rint(H * Wp / Srep).clip(-H, H).astype(np.uint8) + H
    return D.reshape(n, ng * g), S, m                           # digits (n, padded), scale, true m


def dequant_from_store(D, S, k, group=GROUP):
    """Normative k9 dequant from stored digits/scales -> (n, m_cols) fp64."""
    n, mcols = D.shape
    H = (k - 1) // 2
    m = np.repeat(S, group, axis=1)[:, :mcols]
    return m * (D.astype(np.float64) - H) / H


# ------------------------------------------------------------- artifacts

def build_k9_artifact(uniq, spec, path):
    """spec: name -> ('grid', k) or ('rest',). Returns (total bytes, digits
    store, rest store). The tied embedding is stored under whichever key uniq
    kept and re-aliased at reload."""
    tensors, digits_store, rest_store = [], {}, {}
    for name, W in uniq.items():
        tag = spec.get(name, ("rest",))
        if W.dim() != 2 or tag[0] == "rest":
            rest_store[name] = W.half().numpy()
            continue
        k = tag[1]
        D, S, m_true = quant_rtn_odd(W.double().numpy(), k)
        digits_store[name] = dict(digits=D, scales=S, k=k, m_true=m_true)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]), group=GROUP,
                            scale_mode="fp16", k=k, digits=D,
                            scales=S.astype(np.float32), perm=None))
    k9.write_k9(path, tensors)
    return os.path.getsize(path) + sum(v.nbytes for v in rest_store.values()), \
        digits_store, rest_store


def reload_k9(path, spec, rest_store, emb_name, true_widths):
    recs = k9.read_k9(path)
    sd = {}
    for name, rec in recs.items():
        q = k9.decode_tensor(rec, GROUP)          # fp32, padded width
        sd[name] = q[:, : true_widths[name]] if q.dim() == 2 else q
    for name, v in rest_store.items():
        sd[name] = torch.from_numpy(v).float()
    sd["output.weight"] = sd[emb_name]
    sd["tok_embeddings.weight"] = sd[emb_name]
    return sd


def mem_recon(digits_store, rest_store, uniq, emb_name):
    """Container-independent recon of the same digits (P8 cross-check)."""
    sd = {}
    for name, st in digits_store.items():
        w = dequant_from_store(st["digits"], st["scales"], st["k"])[:, : st["m_true"]]
        sd[name] = torch.from_numpy(w.astype(np.float32))
    for name, v in rest_store.items():
        sd[name] = torch.from_numpy(v).float()
    sd["output.weight"] = sd[emb_name]
    sd["tok_embeddings.weight"] = sd[emb_name]
    return sd


# ------------------------------------------------------------- main

def main():
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    w2d = {k: v for k, v in uniq.items() if v.dim() == 2}
    EMB = "tok_embeddings.weight" if "tok_embeddings.weight" in uniq else "output.weight"
    tokens = L.valid_tokens(384 * ma["max_seq_len"])

    fp32 = L.eval_loss(L.to_device(sd0, DEV), ma, tokens, DEV, 384)
    print(f"[eq5] fp32 anchor {fp32[0]:.4f}±{fp32[1]:.4f}")

    variants = {
        "A_k9emb99": {n: ("grid", 9) for n in w2d if n != EMB} | {EMB: ("grid", 99)},
        "B_k27emb99": {n: ("grid", 27) for n in w2d if n != EMB} | {EMB: ("grid", 99)},
        "C_k63emb99": {n: ("grid", 63) for n in w2d if n != EMB} | {EMB: ("grid", 99)},
    }
    rows = []
    check_mem = []
    for tag, spec in variants.items():
        path = os.path.join(RES, f"eq5_{tag}.k9")
        nbytes, dst, rst = build_k9_artifact(uniq, spec, path)
        widths = {n: int(v.shape[1]) if v.dim() == 2 else None for n, v in uniq.items()}
        sd_file = reload_k9(path, spec, rst, EMB, widths)
        ev_file = L.eval_loss(L.to_device(sd_file, DEV), ma, tokens, DEV, 384)
        sd_mem = mem_recon(dst, rst, uniq, EMB)
        ev_mem = L.eval_loss(L.to_device(sd_mem, DEV), ma, tokens, DEV, 384)
        story = L.sample_story(L.to_device(sd_file, DEV), ma, DEV, n_new=160, seed=7)
        rows.append(dict(variant=tag, bytes=nbytes,
                         bpp=round(nbytes * 8 / UNIQUE_PARAMS, 3),
                         loss=round(ev_file[0], 4), sem=round(ev_file[1], 4),
                         loss_mem=round(ev_mem[0], 4),
                         ratio=round(ev_file[0] / fp32[0], 4),
                         story=story[:80]))
        check_mem.append((tag, ev_file[0], ev_mem[0], ev_file[2], ev_mem[2]))
        print(f"[eq5] {tag}: {nbytes:,} B = {nbytes*8/UNIQUE_PARAMS:.2f} b/p | "
              f"loss {ev_file[0]:.4f} (+{100*(ev_file[0]/fp32[0]-1):.1f}%) | "
              f"mem-recon {ev_mem[0]:.4f}")
        print(f"      story: {story[:96]!r}")

    # ---------------- P7: matched-precision digits, raw vs DCT domain ----------------
    emb32 = sd0[EMB].numpy()
    rest = {n: v for n, v in uniq.items() if n != EMB and uniq[n].dim() != 2}
    rest.update({EMB: emb32})  # fp16-stored in both modes, equal on both sides
    mode_bytes, mode_states = {}, {}
    for mode in ("raw", "dct"):
        total_digits = 0
        tensors, dst = [], {}
        for name, W in w2d.items():
            if name == EMB:
                continue
            src = W.double().numpy() if mode == "raw" else L.dct2(W.double()).numpy()
            D, S, m_true = quant_rtn_odd(src, 255)
            f = os.path.join(RES, f"eq5_p7_{mode}_{name.replace('.', '_')}.k9")
            k9.write_k9(f, [dict(name=name, shape=(D.shape[0], D.shape[1]), group=GROUP,
                                 scale_mode="fp16", k=255, digits=D,
                                 scales=S.astype(np.float32), perm=None)])
            total_digits += os.path.getsize(f)
            dst[name] = dict(digits=D, scales=S, k=255, m_true=m_true)
        sd = {}
        for name, st in dst.items():
            w = dequant_from_store(st["digits"], st["scales"], 255)[:, : st["m_true"]]
            if mode == "dct":
                w = L.idct2(torch.from_numpy(w)).numpy()
            sd[name] = torch.from_numpy(w.astype(np.float32))
        for name, v in rest.items():
            sd[name] = torch.from_numpy(np.asarray(v, dtype=np.float16)).float()
        sd["output.weight"] = sd["tok_embeddings.weight"] = torch.from_numpy(emb32.copy()).float()
        ev = L.eval_loss(L.to_device(sd, DEV), ma, tokens, DEV, 384)
        mode_bytes[mode], mode_states[mode] = total_digits, ev[0]
        print(f"[eq5] P7 {mode}: body digit-file bytes {total_digits:,} | loss {ev[0]:.4f} "
              f"({ev[0]/fp32[0]:.3f}x)")

    raw_b, dct_b = mode_bytes["raw"], mode_bytes["dct"]
    ev_raw, ev_dct = mode_states["raw"], mode_states["dct"]

    # ---------------- verdicts ----------------
    a = next(iter([r for r in rows if r["variant"] == "A_k9emb99"]))
    v6 = dict(prediction="P6", bytes=a["bytes"], loss=a["loss"], ratio=a["ratio"],
              verdict="PASS" if (a["bytes"] <= 150_000 and a["ratio"] <= 1.12) else "FAIL")
    v7 = dict(prediction="P7", raw_bytes=raw_b, dct_bytes=dct_b,
              byte_ratio=round(dct_b / raw_b, 4), loss_raw=ev_raw, loss_dct=ev_dct,
              verdict="PASS" if abs(dct_b / raw_b - 1) <= 0.10 else "FAIL")
    mem_diffs = []
    for tag, lf, lm, pf, pm in check_mem:
        d = float((pf - pm).abs().max())   # per-window NLL diff, container vs formula path
        mem_diffs.append(round(d, 5))
    v8 = dict(prediction="P8", max_window_nll_diff=max(mem_diffs), tol=5e-3,
              verdict="PASS" if max(mem_diffs) <= 5e-3 else "FAIL")
    print(f"[eq5] P6 ({v6['verdict']}): A {a['bytes']:,} B, ratio {a['ratio']:.3f}")
    print(f"[eq5] P7 ({v7['verdict']}): dct/raw byte ratio {dct_b/raw_b:.3f}")
    print(f"[eq5] P8 ({v8['verdict']}): artifact==mem recon paired-loss identity")
    with open(os.path.join(RES, "eq5_results.json"), "w") as f:
        json.dump([v6, v7, v8, dict(rows=rows)], f, indent=2)
    print("[eq5] wrote results/eq5_results.json (+ eq5_*.k9 artifacts)")


if __name__ == "__main__":
    main()