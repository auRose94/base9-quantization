#!/usr/bin/env python3
"""Shared eval harness — multi-window perplexity with error bars.

Single-window point estimates were the biggest methodological weakness of
exp17 (24.5k tokens, no CI). This replaces them with:

  * N disjoint windows spread across a corpus (with gaps, so they are not
    adjacent text), each evaluated separately;
  * per-scheme mean +/- standard error over windows;
  * PAIRED comparison on identical windows (`paired_delta`), which cancels the
    window effect — the right way to compare two quantization schemes.

Usage:
    wins = get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                       "test", n_windows=8, toks=1024)
    ppl  = ppl_per_window(model, wins)      # (8,)
    m, se = summarize(ppl)
    dm, dse = paired_delta(ppl_scheme, ppl_fp32)
"""
import math

import numpy as np
import torch

TEXT_FIELDS = ("text", "content", "code", "body")


def get_windows(tok, dev, dataset, config, split, n_windows, toks,
                gap=None, label=""):
    """Stream `dataset`, pack into n_windows disjoint `toks`-token windows."""
    from datasets import load_dataset
    ds = (load_dataset(dataset, config, split=split, streaming=True)
          if config else load_dataset(dataset, split=split, streaming=True))
    feats = list(ds.features)
    field = next((f for f in TEXT_FIELDS if f in feats), None)
    if field is None:
        raise SystemExit(f"no text field in {dataset}: {feats}")
    gap = toks if gap is None else gap
    need = n_windows * (toks + gap)
    ids = []
    for row in ds:
        t = row[field]
        if not t or not t.strip():
            continue
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        if len(ids) >= need:
            break
    arr = np.asarray(ids[:need], dtype=np.int64)
    stride = toks + gap
    wins = np.stack([arr[i * stride: i * stride + toks] for i in range(n_windows)])
    print(f"  {label or dataset}: field '{field}', {n_windows} windows x {toks}"
          f" tok (gap {gap}) = {wins.size:,} tok")
    return torch.from_numpy(wins).to(dev)


@torch.no_grad()
def ppl_per_window(model, wins):
    """Perplexity of each window, one sequence at a time (low memory)."""
    out = np.empty(wins.size(0), dtype=np.float64)
    for i in range(wins.size(0)):
        x = wins[i: i + 1]
        logits = model(x).logits
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1), reduction="sum").item()
        out[i] = math.exp(nll / (x.size(1) - 1))
    return out


def summarize(ppls):
    """Mean and standard error over windows."""
    p = np.asarray(ppls, dtype=np.float64)
    return float(p.mean()), float(p.std(ddof=1) / math.sqrt(p.size))


def paired_delta(a, b):
    """Mean and SE of (a - b) over the same windows (paired)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))
