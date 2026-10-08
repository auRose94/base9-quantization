#!/usr/bin/env python3
"""eq14 — RQ10 stage A: the closure census (base9 docs/09-digit-native-coding).

Question: can the forward pass run EXACTLY on the ninths scaling alphabet
(every value a digit-grid rational, no rounding anywhere)? The obstruction is
per-vector relative spread: multiply-accumulate on the grid costs
`spread(x) + row_span(W) + ceil(log9 fan_in)` digits (exactness, not error).
If activation spreads stay bounded, exact-grid runtime fits a machine word;
if spreads compound per layer, exactness means bignum and the design pivots.

Measured here on the stories260K artifact class (eq5 RTN A/C + eq9 QAT arms):
  * per-vector base-9 relative spread (log9 max/min of nonzero |v|, per
    position row) of every activation tensor through the forward pass;
  * per-row weight spreads + cross-group scale spreads from each artifact;
  * the spread map of the non-closure ops (silu ratio effect, attention/softmax
    mass range, rmsnorm eps relative size) — stage B's substitution targets;
  * assembled exact-accumulate digit budgets: regrid-per-layer (n_act sweep)
    vs the measured per-layer spread growth (empirical compounding).

Pre-registered (docs/09): P37-census = worst per-layer exact-precision need
<= 20 base-9 digits at SOME n_act in {1,2,3} (formal P37 waits for stage B's
bit-exact runtime; this measurement decides whether it is possible without
bignum). P38/P39/P40 are stage B. u128 holds 40.4 base-9 digits, u64 20.1.

Data-integrity note: eq9 QAT artifacts persist only the 2-D grid tensors —
the trained 1-D norm weights were not saved, so their census rows use fp16
(source norms) and are flagged norms_reproducible=False; eq5 RTN rows are
exactly reproducible (norms = fp16 of source, as eq5 stored them).

CPU-only (fp64 analysis pass, the repo's analysis convention).
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
import k9  # noqa: E402

RES = L.RESULTS
GROUP = 64
LOG9 = 1.0 / float(np.log(9.0))


def log9(x):
    return np.log(x) * LOG9


def row_spread(t):
    """t: (..., D) tensor -> (per-row base-9 relative spread of nonzero |v|),
    plus the zero fraction. The row is the accumulator unit for exact matmul."""
    a = t.reshape(-1, t.shape[-1]).double().abs().numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rmax = np.where(a > 0, a, -np.inf).max(axis=1)
        rmin = np.where(a > 0, a, np.inf).min(axis=1)
    spread = log9(rmax) - log9(rmin)
    spread[~np.isfinite(spread)] = 0.0
    return spread


def stats(t):
    sp = row_spread(t)
    zf = float((t.abs() == 0).double().mean())
    return dict(spread_p50=round(float(np.percentile(sp, 50)), 2),
                spread_p99=round(float(np.percentile(sp, 99)), 2),
                spread_max=round(float(sp.max()), 2),
                zero_frac=round(zf, 4),
                log9_absmax=round(float(log9(float(t.abs().max()))), 2))


def main():
    torch.manual_seed(0)
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    EMB = next(n for n in ("tok_embeddings.weight", "output.weight") if n in uniq)
    widths = {n: int(v.shape[1]) for n, v in uniq.items() if v.dim() == 2}
    rest = {k: v.half().float().double() for k, v in uniq.items() if v.dim() == 1}
    tokens = L.valid_tokens(16 * ma["max_seq_len"])
    xb = torch.from_numpy(tokens[: 16 * ma["max_seq_len"]]
                          .reshape(16, ma["max_seq_len"]).astype(np.int64))
    B, T = xb.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    pos = torch.arange(T)
    neg_inf_mask = torch.full((T, T), float("-inf")).triu(1)

    subjects = [("eq5_A_k9emb99", "k9 body / k99 embed (RTN)", True),
                ("eq5_C_k63emb99", "k63 body / k99 embed (RTN)", True),
                ("eq9_QAT_k9emb99", "k9 body / k99 embed (QAT-trained digits)", False),
                ("eq9_QAT_k27emb99", "k27 body / k99 embed (QAT-trained digits)", False)]

    report, assemble = {}, {}
    for tag, desc, reps in subjects:
        recs = k9.read_k9(os.path.join(RES, f"{tag}.k9"))
        sd = {n: k9.decode_tensor(r, GROUP)[:, : widths[n]].double()
              for n, r in recs.items()}
        for k, v in rest.items():
            sd[k] = v
        sd[EMB] = sd["tok_embeddings.weight"]
        sd["output.weight"] = sd[EMB]
        print(f"[eq14] {tag}: {desc}, {len(recs)} grid tensors")

        act = {}
        h = sd["tok_embeddings.weight"][xb]
        act["h_in0"] = h.clone()
        for l in range(ma["n_layers"]):
            p = f"layers.{l}."
            hn = L.rmsnorm(h, sd[p + "attention_norm.weight"])
            act[f"hn{l}"] = hn
            q = L.rope((hn @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd), pos)
            k4 = L.rope((hn @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd), pos)
            v4 = (hn @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
            rep_h = nh // nkv
            kk = k4.repeat_interleave(rep_h, dim=2).transpose(1, 2)
            vv = v4.repeat_interleave(rep_h, dim=2).transpose(1, 2)
            att = (q.transpose(1, 2) @ kk.transpose(-2, -1)) / hd ** 0.5
            att = att + neg_inf_mask
            o = (att.softmax(-1) @ vv).transpose(1, 2).reshape(B, T, dim)
            act[f"attnout{l}"] = o
            o_pre = o @ sd[p + "attention.wo.weight"].T
            act[f"o_pre{l}"] = o_pre
            h_mid = h + o_pre
            act[f"mid{l}"] = h_mid
            hn2 = L.rmsnorm(h_mid, sd[p + "ffn_norm.weight"])
            w1out = hn2 @ sd[p + "feed_forward.w1.weight"].T
            gate = hn2 @ sd[p + "feed_forward.w3.weight"].T
            ff = (w1out * torch.sigmoid(w1out)) * gate
            act[f"ff{l}"] = ff
            h = h_mid + (ff @ sd[p + "feed_forward.w2.weight"].T)
            act[f"h_out{l}"] = h
            if l == ma["n_layers"] - 1:
                act["h_final"] = h
                hnF = L.rmsnorm(h, sd["norm.weight"])
                act["hnF"] = hnF
        act_stats = {n: stats(t) for n, t in act.items()}

        wt_stats = {}
        for name, rec in recs.items():
            w = k9.decode_tensor(rec, GROUP).double()[:, : widths[name]]
            rsp = row_spread(w)
            m = np.asarray(k9.decode_scales(rec["scales"], rec["scale_mode"],
                                            rec["shape"][0] * (rec["shape"][1] // GROUP)),
                           dtype=np.float64)
            m = m[m > 0]
            wt_stats[name] = dict(row_spread_p50=round(float(np.percentile(rsp, 50)), 2),
                                  row_spread_p99=round(float(np.percentile(rsp, 99)), 2),
                                  row_spread_max=round(float(rsp.max()), 2),
                                  scale_spread=round(float(log9(m.max() / m.min())), 2),
                                  k=rec["k"])

        # ---- op-substitution instruments ----
        # rmsnorm eps relative size, measured at the norm's INPUT (h_mid; L0 uses
        # the embed output): if eps <= 9^n / 2 relative, the norm stays exact.
        ops = {}
        eps_inputs = [(0, act["h_in0"])]
        eps_inputs += [(l, act[f"mid{l}"]) for l in range(ma["n_layers"])]
        for l, hin in eps_inputs:
            msq = hin.reshape(-1, dim).pow(2).mean(-1)
            ops[f"rmsnorm_eps_rel_L{l}"] = round(float(1e-5 / msq.min()), 6)
        # silu spread map + softmax mass range at the LAST layer (use captured)
        hnL = act[f"hn{ma['n_layers'] - 1}"]
        w1t = hnL @ sd[f"layers.{ma['n_layers'] - 1}.feed_forward.w1.weight"].T
        sil = w1t * torch.sigmoid(w1t)
        s_out, s_in = row_spread(sil).max(), row_spread(w1t).max()
        ops["silu_spread_map_lastL"] = dict(
            w1out_spread_max=round(float(s_in), 2), silu_out_spread_max=round(float(s_out), 2))
        # attention softmax mass range (last layer): per query row max & min nonzero
        p = f"layers.{ma['n_layers'] - 1}."
        hn3 = act[f"hn{ma['n_layers'] - 1}"]
        q4 = L.rope((hn3 @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd), pos)
        k4 = L.rope((hn3 @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd), pos)
        rep_h = nh // nkv
        kk4 = k4.repeat_interleave(rep_h, dim=2).transpose(1, 2)
        sc4 = (q4.transpose(1, 2) @ kk4.transpose(-2, -1)) / hd ** 0.5 \
            + neg_inf_mask
        att4 = sc4.softmax(-1)
        att = att4.reshape(-1, T)
        amax = att.max(-1).values
        # min attended mass per query row, EXCLUDING the causal-mask zeros
        tri = torch.ones(T, T).triu(1).bool()
        attn = torch.where(att4 > 0, att4, torch.tensor(float("inf"))).masked_fill(
            tri.unsqueeze(0).unsqueeze(0), float("inf")).reshape(-1, T).min(-1).values
        amin = attn[attn.isfinite()]
        azero_frac = float((attn == 0).double().mean())
        ops["softmax_mass_lastL"] = dict(
            max_w=round(float(amax.mean()), 3),
            underflow_rows_frac=round(azero_frac, 3),
            min_attended_p50=round(float(amin.median()), 10),
            mass_dynamic_range_log10=round(
                float(np.log10(float(amax.mean()) / amin.median())), 2)
            if amin.numel() and amin.median() > 0 else None)

        # ---- assembled spans ----
        spans = {}
        for n_act in (1, 2, 3):
            worst = 0.0
            for l in range(ma["n_layers"]):
                p = f"layers.{l}."
                s_hn = act_stats[f"hn{l}"]["spread_p99"]
                s_o = act_stats[f"attnout{l}"]["spread_p99"]
                s_ff = act_stats[f"ff{l}"]["spread_p99"] if f"ff{l}" in act_stats else s_hn
                s_hnF = act_stats.get("hnF", s_hn)["spread_p99"] if l == 4 else None
                sites = [("wq", s_hn, 64), ("wk", s_hn, 64), ("wv", s_hn, 64),
                         ("wo", s_o, 64), ("w1", s_hn, 64), ("w3", s_hn, 64),
                         ("w2", s_ff, 172)]
                if l == ma["n_layers"] - 1:
                    sites.append(("output", s_hnF, 64))
                for wn, s_x, din in sites:
                    wname = {"wq": p + "attention.wq.weight", "wk": p + "attention.wk.weight",
                             "wv": p + "attention.wv.weight", "wo": p + "attention.wo.weight",
                             "w1": p + "feed_forward.w1.weight",
                             "w3": p + "feed_forward.w3.weight",
                             "w2": p + "feed_forward.w2.weight",
                             "output": "output.weight"}[wn]
                    r_w = wt_stats[wname]["row_spread_p99"] if wname in wt_stats else 1.7
                    span = s_x + r_w + n_act + np.ceil(log9(din))
                    worst = max(worst, float(span))
            spans[f"n_act={n_act}"] = round(worst, 2)

        report[tag] = dict(desc=desc, norms_reproducible=reps,
                           act=act_stats, weights=wt_stats, ops=ops, spans=spans)
        best = min(spans.values())
        print(f"  spans: {spans} -> {'PASS' if best <= 20 else 'FAIL'} (bar 20 digits)")

    # empirical compounding: residual-stream spread across layers (captured)
    compound = {}
    for tag, rep in report.items():
        a = rep["act"]
        compound[tag] = dict(
            h_in0_p99=a["h_in0"]["spread_p99"],
            h_out_p99=[a[f"h_out{l}"]["spread_p99"] for l in range(ma["n_layers"])],
            o_pre_p99=[a[f"o_pre{l}"]["spread_p99"] for l in range(ma["n_layers"])],
            ff_p99=[a[f"ff{l}"]["spread_p99"] for l in range(ma["n_layers"])])

    out = dict(subjects=report, compounding=compound,
               constants=dict(u128_digits=40.4, u64_digits=20.1))
    with open(os.path.join(RES, "eq14_census.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(RES, "eq14_census.md"), "w") as f:
        f.write("# eq14 — closure census (RQ10 stage A, docs/09)\n\n")
        f.write("Spread = log9(max/min of nonzero |v|) per accumulator row; "
                "budget span = s_x + row_span(W) + n_act + ceil(log9 fan_in).\n\n")
        for tag, rep in report.items():
            f.write(f"## {tag} — {rep['desc']} (norms {'exact' if rep['norms_reproducible'] else 'source-fp16 proxy'})\n\n")
            f.write("| activation | spread p50/p99/max | zeros | log9 absmax |\n|---|---|---|---|\n")
            for n, s in rep["act"].items():
                f.write(f"| {n} | {s['spread_p50']}/{s['spread_p99']}/{s['spread_max']} | "
                        f"{s['zero_frac']} | {s['log9_absmax']} |\n")
            f.write("\n| weight | row spread p50/p99/max | scale spread | k |\n|---|---|---|---|\n")
            for n, s in rep["weights"].items():
                f.write(f"| {n} | {s['row_spread_p50']}/{s['row_spread_p99']}/{s['row_spread_max']} | "
                        f"{s['scale_spread']} | {s['k']} |\n")
            f.write(f"\nops: `{json.dumps(rep['ops'])}`\n\n")
            f.write(f"spans: `{json.dumps(rep['spans'])}`\n\n")
        f.write(f"compounding: `{json.dumps(compound)}`\n")
    print("[eq14] wrote results/eq14_census.json + eq14_census.md")


if __name__ == "__main__":
    main()