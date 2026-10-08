#!/usr/bin/env python3
"""eq16 — RQ10 stage B, leg 1: the substituted-semantics model (docs/09 P39)
+ the P40 activation-digit iid scan.

The exact-grid runtime must replace the transcendental ops (stage B design,
§0 of docs/09). This leg measures what the SUBSTITUTIONS cost in quality,
in the fast fp32 harness — value semantics only, no precision questions:

  * norm substitution:  mean-abs norm  x · dim/Σ|x| · λ_site  (rational; the
    per-site λ is calibrated on the TRAIN slice — the fit pool, disjoint from
    the 384 eval windows — by matching each site's output rms to rmsnorm's;
    exact zero-vector guard: an all-zero row maps to 0) replacing RMSNorm
    (whose sqrt is irrational in every base);
  * attention substitution:  quadratic rational kernel — weights
    max(0, 1 + s/c)², c = 4.0, row-normalized (implemented as softmax over
    log-weights 2·log1p(relu(s/c)) so the fp path is stable): GROWS on
    positive scores (peaky like exp), exactly 0 below s = −c, and a plain
    per-row rational (numerators over the row sum) for the exact path.
    (Design note: the first kernel tried this session, hinge² 1/(1+re(−s))²,
    saturated at weight 1 for all clearly-matched positions — flat instead of
    peaked — +126% on its own; growing kernels are the registered family.)
  * gate substitution:  silu-r  x·(1 + x/(|x|+1))/2  (odd-symmetric sigmoid
    family; within ~0.04 of silu near 0), replacing silu.

Registered (docs/09 P39, operationalized against eq9's own paired numbers):
  P39 joint: substituted model (artifact weights, all three swaps) within
      +2% (relative loss) of the eq9_QAT_k27emb99 arm (1.3143) on the same
      384x512 windows;
  P39a norm-swap alone <= +1%; P39b attention-swap alone <= +1%;
      (P39c silu-swap, reported but not registered).
P40 (stage B item d): activation digit streams — position-major, the
runtime's own 2-digit base-9 representation with per-position power-of-9
exponent — carry context: order-1 Markov gain >= 5% over static frequency
coding on some layer (contrast: exp12's weight-digit null, 0.0–0.2%).

Anchors (eq9_results.json): CONTROL 1.2962 | QAT-k27 1.3143 | fp32 source
1.9726. CPU-only. NOTE: the attention kernel's score scale c was not pinned
at registration; two values are reported and the formal row uses c = 4.0 —
treat the exact number as indicative, the family result as the claim.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402

RES = L.RESULTS
GROUP = 64
SCORE_C = 4.0


def manorm(x, w, lam=1.0):
    s = x.abs().sum(-1, keepdim=True)
    return torch.where(s > 0, x * (x.shape[-1] / s.clamp_min(1e-30) * lam),
                       torch.zeros_like(x)) * w


def silu_r(x):
    return x * (1.0 + x / (x.abs() + 1.0)) * 0.5


@torch.no_grad()
def forward_sub(sd, idx, ma, use_norm=False, use_att=False, use_silu=False,
                lams=None, c_att=SCORE_C, capture=None):
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh

    def norm(x, w, site):
        if use_norm:
            lam = float(lams[site]) if lams is not None else 1.0
            return manorm(x, w, lam)
        return L.rmsnorm(x, w)

    g = silu_r if use_silu else (lambda x: x * torch.sigmoid(x))
    h = sd["tok_embeddings.weight"][idx]
    pos = torch.arange(T)
    mask = torch.full((T, T), float("-inf")).triu(1)
    keep = torch.ones(T, T).tril().bool()
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn = norm(h, sd[p + "attention_norm.weight"], 2 * l)
        if capture is not None:
            capture.append(hn)
        q = (hn @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k_ = (hn @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v_ = (hn @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = L.rope(q, pos)
        k_ = L.rope(k_, pos)
        rep = nh // nkv
        kk = k_.repeat_interleave(rep, dim=2).transpose(1, 2)
        vv = v_.repeat_interleave(rep, dim=2).transpose(1, 2)
        s_pre = (q.transpose(1, 2) @ kk.transpose(-2, -1)) / hd ** 0.5
        if use_att:
            # registered kernel: w = max(0, 1+s/c)^2 — the clamp is on
            # (1+s/c), not inside log1p: masked/below-floor scores must get
            # weight exactly 0 (log(1+s/c) = -inf), not weight 1.
            s = torch.where(keep, s_pre, torch.full_like(s_pre, -(c_att + 1.0)))
            logw = 2.0 * torch.log(torch.clamp(1.0 + s / c_att, min=1e-20))
            o = torch.softmax(logw, -1) @ vv
        else:
            o = (s_pre + mask).softmax(-1) @ vv
        o = o.transpose(1, 2).reshape(B, T, dim)
        h = h + (o @ sd[p + "attention.wo.weight"].T)
        hn2 = norm(h, sd[p + "ffn_norm.weight"], 2 * l + 1)
        if capture is not None:
            capture.append(hn2)
        w1out = hn2 @ sd[p + "feed_forward.w1.weight"].T
        ff = g(w1out) * (hn2 @ sd[p + "feed_forward.w3.weight"].T)
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].T)
    hnF = norm(h, sd["norm.weight"], 2 * ma["n_layers"])
    if capture is not None:
        capture.append(hnF)
    return hnF @ sd["output.weight"].T


@torch.no_grad()
def calibrate_lams(sd, ma, n_pos=4096):
    """λ per norm site, measured on the TRAIN slice (eq6's fit pool, ⊥ eval):
    λ_site = mean rms of rmsnorm's output / mean rms of manorm's output."""
    from eq6_func_fit import train_tokens  # noqa: E402
    tr = train_tokens(n_pos)
    idx = torch.from_numpy(tr.reshape(-1, ma["max_seq_len"]).astype(np.int64))
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    pos = torch.arange(T)
    mask = torch.full((T, T), float("-inf")).triu(1)
    h = sd["tok_embeddings.weight"][idx]
    lams, site = [], 0
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn_r = L.rmsnorm(h, sd[p + "attention_norm.weight"])
        hn_m = manorm(h, sd[p + "attention_norm.weight"])
        lams.append(float(hn_r.pow(2).mean().sqrt() / hn_m.pow(2).mean().sqrt()))
        q = (hn_r @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k_ = (hn_r @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v_ = (hn_r @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = L.rope(q, pos)
        k_ = L.rope(k_, pos)
        rep = nh // nkv
        kk = k_.repeat_interleave(rep, dim=2).transpose(1, 2)
        vv = v_.repeat_interleave(rep, dim=2).transpose(1, 2)
        s = (q.transpose(1, 2) @ kk.transpose(-2, -1)) / hd ** 0.5 + mask
        o = (s.softmax(-1) @ vv).transpose(1, 2).reshape(B, T, dim)
        h = h + (o @ sd[p + "attention.wo.weight"].T)
        hn_r2 = L.rmsnorm(h, sd[p + "ffn_norm.weight"])
        hn_m2 = manorm(h, sd[p + "ffn_norm.weight"])
        lams.append(float(hn_r2.pow(2).mean().sqrt() / hn_m2.pow(2).mean().sqrt()))
        w1out = hn_r2 @ sd[p + "feed_forward.w1.weight"].T
        ff = (w1out * torch.sigmoid(w1out)) * \
            (hn_r2 @ sd[p + "feed_forward.w3.weight"].T)
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].T)
    hn_rF = L.rmsnorm(h, sd["norm.weight"])
    hn_mF = manorm(h, sd["norm.weight"])
    lams.append(float(hn_rF.pow(2).mean().sqrt() / hn_mF.pow(2).mean().sqrt()))
    return lams


@torch.no_grad()
def eval_sub(sd, ma, tokens, **kw):
    block = ma["max_seq_len"]
    win = torch.from_numpy(tokens[: 384 * block].astype(np.int64)).view(384, block)
    losses = []
    for i in range(0, 384, 16):
        b = win[i:i + 16]
        lg = forward_sub(sd, b, ma, **kw)[:, :-1]
        tg = b[:, 1:]
        nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean(-1).cpu())
    losses = torch.cat(losses)
    return float(losses.mean()), float(losses.std(unbiased=True) / 19)


# ------------------------------------------------------------------ P40

def digitize(t, digits=2):
    """Runtime representation: per position row, exponent e = ceil(log9 max|v|),
    mantissas round(v · 9^(digits−e)). Returns the position-major digit matrix."""
    v = t.reshape(-1, t.shape[-1]).double()
    mx = v.abs().max(dim=1).values
    e = torch.ceil(torch.log(mx.clamp_min(1e-30)) / float(np.log(9.0)))
    scale = torch.pow(9.0, digits - e)[:, None]
    return torch.round(v * scale).long()


def markov_gain(dmat):
    """Order-1 Markov entropy gain (%) over the marginal, along the time axis
    and the channel axis; exp12's estimator shape."""
    x = dmat.numpy()
    gains = {}
    for axis, tag in ((0, "time"), (1, "channel")):
        a = x if axis == 0 else x.T
        s1, s2 = a[:, :-1].reshape(-1), a[:, 1:].reshape(-1)
        u1, i1 = np.unique(s1, return_inverse=True)
        u2, i2 = np.unique(s2, return_inverse=True)
        K = max(len(u1), len(u2))
        joint = np.zeros((K, K))
        np.add.at(joint, (i1, i2), 1)
        p = joint / joint.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            Hjoint = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
            px = p.sum(1)
            py = p.sum(0)
            Hx = float(-(px[px > 0] * np.log2(px[px > 0])).sum())
            Hy = float(-(py[py > 0] * np.log2(py[py > 0])).sum())
        Hcond = Hjoint - Hy
        gains[tag] = round((1.0 - Hcond / max(Hx, 1e-12)) * 100, 2)
    return gains


def main():
    t0 = time.time()
    torch.manual_seed(0)
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    EMB = next(n for n in ("tok_embeddings.weight", "output.weight") if n in uniq)
    widths = {n: int(v.shape[1]) for n, v in uniq.items() if v.dim() == 2}
    tokens = L.valid_tokens(384 * ma["max_seq_len"])

    recs = k9.read_k9(os.path.join(RES, "eq9_QAT_k27emb99.k9"))
    sd = {n: k9.decode_tensor(r, GROUP)[:, : widths[n]].float() for n, r in recs.items()}
    for k2, v in uniq.items():
        if v.dim() == 1:
            sd[k2] = v.half().float()
    sd[EMB] = sd["tok_embeddings.weight"]
    sd["output.weight"] = sd[EMB]
    print(f"[eq16] artifact loaded ({time.time()-t0:.0f}s)")

    arm_k27 = 1.3143247365951538            # eq9's artifact loss (its harness)
    base = eval_sub(sd, ma, tokens)
    print(f"[eq16] artifact re-eval (CPU, this harness): {base[0]:.4f}")

    lams = calibrate_lams(sd, ma)
    print(f"[eq16] calibrated λ (train-slice, 11 sites): "
          f"{[round(x, 4) for x in lams]}")
    lam1 = [1.0] * len(lams)

    variants = [
        ("norm_only", dict(use_norm=True, lams=lams)),
        ("att_only_c4", dict(use_att=True, c_att=4.0)),
        ("att_only_c1", dict(use_att=True, c_att=1.0)),
        ("silu_only", dict(use_silu=True)),
        ("joint_c4", dict(use_norm=True, use_att=True, use_silu=True, lams=lams)),
        ("joint_c1", dict(use_norm=True, use_att=True, use_silu=True, lams=lams,
                          c_att=1.0)),
        ("joint_uncal", dict(use_norm=True, use_att=True, use_silu=True, lams=lam1)),
    ]
    rows = {}
    for tag, kw in variants:
        loss = eval_sub(sd, ma, tokens, **kw)[0]
        rows[tag] = dict(loss=round(loss, 4), vs_arm_pct=round(100 * (loss / arm_k27 - 1), 2))
        print(f"[eq16] {tag:>12}: loss {loss:.4f} ({rows[tag]['vs_arm_pct']:+.2f}% vs k27 arm)")

    p39a = dict(prediction="P39a", **rows["norm_only"], bar=1.0,
                verdict="PASS" if rows["norm_only"]["vs_arm_pct"] <= 1.0 else "FAIL")
    att_rows = (rows["att_only_c4"], rows["att_only_c1"])
    att_best = dict(min(att_rows, key=lambda r: r["vs_arm_pct"]))
    p39b = dict(prediction="P39b", **att_best,
                kernel="max(0,1+s/c)^2, c reported for 4.0 and 1.0", bar=1.0,
                verdict="PASS" if att_best["vs_arm_pct"] <= 1.0 else "FAIL")
    p39c = dict(prediction="P39c", swap="silu-r (reported, not registered)", **rows["silu_only"])
    joint_best = dict(min((rows["joint_c4"], rows["joint_c1"]),
                          key=lambda r: r["vs_arm_pct"]))
    p39 = dict(prediction="P39", **joint_best, bar=2.0,
               verdict="PASS" if joint_best["vs_arm_pct"] <= 2.0 else "FAIL")

    # ---------------- P40: activation digit iid scan ----------------
    print("[eq16] P40 capture pass (fp and substituted-norm streams)...")
    idx = torch.from_numpy(
        tokens[: 8 * ma["max_seq_len"]].reshape(8, ma["max_seq_len"]).astype(np.int64))
    gains = {}
    for tag, use_norm in (("fp", False), ("sub_norm", True)):
        capture = []
        forward_sub(sd, idx, ma, use_norm=use_norm,
                    lams=(lams if use_norm else None), capture=capture)
        per = [markov_gain(digitize(t)) for t in capture]
        gains[tag] = dict(best_time_pct=max(g["time"] for g in per),
                          best_channel_pct=max(g["channel"] for g in per),
                          per_tensor=per)
        print(f"[eq16] P40 {tag}: best time-axis gain {gains[tag]['best_time_pct']:.2f}% "
              f"| channel-axis {gains[tag]['best_channel_pct']:.2f}%")
    best_overall = max(g["best_time_pct"] for g in gains.values())
    p40 = dict(prediction="P40", best_time_gain_pct=best_overall, bar=5.0,
               exp12_null_max_pct=0.2, detail=gains,
               verdict="PASS" if best_overall >= 5.0 else "FAIL")
    print(f"[eq16] P40 ({p40['verdict']}): best order-1 gain {best_overall:.2f}% (bar 5%)")

    out = dict(rows=rows, lams=[round(x, 5) for x in lams],
               P39a=p39a, P39b=p39b, P39c=p39c, P39=p39, P40=p40,
               anchors=dict(arm_k27=arm_k27, cpu_reeval=round(base[0], 5),
                            lambda_note="lams calibrated on train slice (fit pool, "
                                        "disjoint from eval windows)"),
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(RES, "eq16_substituted.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    md = ["# eq16 — RQ10 stage B leg 1: substituted semantics + P40 (docs/09)\n",
          "subject: eq9_QAT_k27emb99.k9 (eq9: ctrl 1.2962 / k27 1.3143 / source 1.9726)\n",
          f"CPU re-eval of the artifact in this harness: {base[0]:.4f}\n",
          f"λ sites (train-slice calibrated): {[round(x, 4) for x in lams]}\n",
          "| variant | loss | vs k27 arm |", "|---|---|---|"]
    for t2, r in rows.items():
        md.append(f"| {t2} | {r['loss']} | {r['vs_arm_pct']:+.2f}% |")
    md += ["",
           f"- P39a ({p39a['verdict']}): mean-abs norm (calibrated) alone "
           f"{rows['norm_only']['loss']} ({rows['norm_only']['vs_arm_pct']:+.2f}%, bar +1%)",
           f"- P39b ({p39b['verdict']}): quadratic rational attention — c=4: "
           f"{rows['att_only_c4']['loss']} ({rows['att_only_c4']['vs_arm_pct']:+.2f}%), "
           f"c=1: {rows['att_only_c1']['loss']} ({rows['att_only_c1']['vs_arm_pct']:+.2f}%), "
           f"bar +1% (formal row = the better c; c was not pinned at registration)",
           f"- P39c (reported): silu-r alone {rows['silu_only']['loss']} "
           f"({rows['silu_only']['vs_arm_pct']:+.2f}%)",
           f"- P39 ({p39['verdict']}): joint, the better c: {joint_best['loss']} "
           f"({joint_best['vs_arm_pct']:+.2f}%, bar +2%)",
           f"- joint without λ calibration: {rows['joint_uncal']['loss']} "
           f"({rows['joint_uncal']['vs_arm_pct']:+.2f}%) — the calibration is doing real work", "",
           f"- P40 ({p40['verdict']}): best order-1 Markov gain on activation digit "
           f"streams: {best_overall:.2f}% (bar 5%; exp12's weight-digit null: 0.0–0.2%)"]
    with open(os.path.join(RES, "eq16_substituted.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[eq16] wrote results/eq16_substituted.{{json,md}} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()