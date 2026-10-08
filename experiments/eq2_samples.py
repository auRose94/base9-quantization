#!/usr/bin/env python3
"""eq2 — story samples: fp32 anchor vs uniform-6 b/p vs the equation point
(dct-topk 0.05, 8-bit). Same seeds/prompts; artifact = results/eq2_samples.md."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_NEW = 320


def sd_from_npz(tag):
    sd0, ma, _ = L.load_state_dict()
    z = np.load(os.path.join(L.RESULTS, f"eq1_recon_{tag}.npz"))
    out = {k[len("recon_"):]: torch.from_numpy(v) for k, v in z.items()
           if k.startswith("recon_")}
    return out, ma


def main():
    sd0, ma, _ = L.load_state_dict()
    variants = [("fp32", sd0)]
    for tag in ("uniform6", "best"):
        sdb, mab = sd_from_npz(tag)
        variants.append((tag, sdb))

    rows = []
    for name, sd in variants:
        sdd = L.to_device({k: v.float() for k, v in sd.items()}, DEV)
        ev = L.eval_loss(sdd, ma, L.valid_tokens(N := 384 * ma["max_seq_len"]), DEV, n_windows=384)
        texts = []
        for seed in (1, 2, 3):
            texts.append(L.sample_story(sdd, ma, DEV, n_new=N_NEW, seed=seed))
        rows.append((name, ev[0], ev[1], texts))
        print(f"[eq2] {name}: loss {ev[0]:.4f}±{ev[1]:.4f}  sample: {texts[0][:70]!r}")

    out = ["# eq2 — stories from the weights-vs-equation models (seeded, temp 0.8, top-k 50)",
           ""
           "prompt: 'Once upon a time,' — 320 new tokens", ""]
    for name, loss, sem, texts in rows:
        out += [f"## {name}  (val loss {loss:.4f} ± {sem:.4f})", ""]
        for si, t in enumerate(texts, 1):
            out += [f"### {name} seed {si}", "", t[:1600], ""]
    with open(os.path.join(L.RESULTS, "eq2_samples.md"), "w") as f:
        f.write("\n".join(out))
    print("[eq2] wrote results/eq2_samples.md")


if __name__ == "__main__":
    main()