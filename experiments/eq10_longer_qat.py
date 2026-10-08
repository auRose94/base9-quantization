#!/usr/bin/env python3
"""eq10 — longer QAT (4000 steps) + multi-seed (exp14 discipline: the matched
CONTROL is THE metric). Runs the eq9 swap-STE machinery with steps/seed args.
Tags encode (budget, seed); artifacts keep the eq9_ prefix (same writer).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402
import eq9_qat as Q  # noqa: E402
from eq6_func_fit import train_tokens  # noqa: E402

DEV = "cuda"
RES = os.path.join(L.ROOT, "results")


def main():
    sd0, ma, _ = L.load_state_dict()
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)
    anchor = L.eval_loss(L.to_device(sd0, DEV), ma, tokens, DEV, 384)
    print(f"[eq10] fp32 source anchor {anchor[0]:.4f}")

    arms = [
        ("q4000s42_CONTROL", None, 42), ("q4000s43_CONTROL", None, 43),
        ("q4000s42_qatk27", 27, 42), ("q4000s43_qatk27", 27, 43),
        ("q4000s42_qatk9", 9, 42),
    ]
    runs = []
    for tag, k, seed in arms:
        print(f"[eq10] === {tag} (k={k}, seed={seed}) ===")
        runs.append(Q.run_arm(tag, k, sd0, ma, fit_tok, tokens, steps=4000, seed=seed))

    ctrl42 = next(r for r in runs if r["arm"] == "q4000s42_CONTROL")["trained_loss"]
    ctrl43 = next(r for r in runs if r["arm"] == "q4000s43_CONTROL")["trained_loss"]
    k27_42 = next(r for r in runs if r["arm"] == "q4000s42_qatk27")["trained_loss"]
    k27_43 = next(r for r in runs if r["arm"] == "q4000s43_qatk27")["trained_loss"]
    k9_42 = next(r for r in runs if r["arm"] == "q4000s42_qatk9")["trained_loss"]
    ctrl_mean = (ctrl42 + ctrl43) / 2
    v20 = dict(prediction="P20", control_mean=round(ctrl_mean, 4),
               verdict="PASS" if ctrl_mean <= 1.20 else "FAIL")
    v21 = dict(prediction="P21", gap_42=round(k27_42 - ctrl42, 4),
               gap_43=round(k27_43 - ctrl43, 4),
               verdict="PASS" if max(k27_42 - ctrl42, k27_43 - ctrl43) <= 0.06 else "FAIL")
    v22 = dict(prediction="P22", gap_k9=round(k9_42 - ctrl42, 4),
               verdict="PASS" if k9_42 - ctrl42 <= 0.20 else "FAIL")
    v23 = dict(prediction="P23", seed_delta=round(abs((k27_42 - ctrl42) - (k27_43 - ctrl43)), 4),
               verdict="PASS" if abs((k27_42 - ctrl42) - (k27_43 - ctrl43)) <= 0.02 else "FAIL")
    for v in (v20, v21, v22, v23):
        print(f"[eq10] {v['prediction']} ({v['verdict']}): " +
              json.dumps({kk: v[kk] for kk in v if kk not in ("prediction", "verdict")}))
    table = [dict(arm=r["arm"], bytes=r["bytes"], loss=round(r["trained_loss"], 4),
                  artifact_loss=round(r["artifact_loss"], 4)) for r in runs]
    with open(os.path.join(RES, "eq10_results.json"), "w") as f:
        json.dump([v20, v21, v22, v23, dict(anchor=anchor[0], runs=table)], f, indent=2)
    print("[eq10] wrote results/eq10_results.json")


if __name__ == "__main__":
    main()