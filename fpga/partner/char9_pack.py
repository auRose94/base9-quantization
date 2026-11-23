#!/usr/bin/env python3
"""A1 packer: the trained char model -> ninths resident images (K9_7 / K9_4).

Per tensor (row-major (out, in) torch weights):
  per 64-group: m = max|w| (f32); digits = clip(round_half_away(w*H/m)) + H,
  H = (k-1)//2; pack = experiments/ninths_native.pack (the Gate-A packer,
  k9_7 = 8 B fp16 m/H per group + 256-byte signed code stream per superblock).

Outputs in --out (default partner/char9_img):
  nn7.k9img / nn4.k9img  blob concatenation in conf.json's tensor order
  manifest.json          per-tensor {offset, bytes, rows, cols, k} + conf
  quality.md             fp32 vs K9_7 vs K9_4 val losses (same eval split rule)

The board top (A3) reads only the manifest + blob; the eval = the quality
gate that must be green before anything goes to the board.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import ninths_native as native   # noqa: E402  (nn = torch.nn from the model defs)
exec(open(HERE / "char9_train.py").read().split("# ── data")[0])   # the model defs

K_TABLE = {"nn7": 99, "nn4": 15}


def rtn_digits(w: np.ndarray, k: int):
    """RTN to the K9 grid: per-64-group m = amax, half-away rounding."""
    H = (k - 1) // 2
    w = w.reshape(w.shape[0], -1)
    r, c = w.shape
    g = w.reshape(r, c // 64, 64)
    m = np.abs(g).max(axis=2).astype(np.float32)        # (r, groups)
    m16 = np.float16(m / np.float32(H))                  # the DEQUANT scale actually used
    s = m16.view(np.uint16).astype(np.uint32)
    # recompute the scale's f32 value exactly as the dequant will see it:
    s32 = np.frombuffer(np.asarray(m16, dtype="<f2").tobytes(), "<f2").astype(np.float32)
    s32 = s32.reshape(r, c // 64)
    safe = np.where(s32 > 0, s32, 1.0).astype(np.float32)
    q = np.floor(np.abs(g) / safe[..., None] + 0.5) * np.sign(g)   # half away
    digits = np.clip(q, -H, H).astype(np.int16) + H + (g == 0) * 0
    return digits.reshape(r, c).astype(np.uint8), m, s32


def pack_model(state, conf, k):
    names = list(state.keys())
    blobs, entries, off = [], [], 0
    for name in names:
        raw = state[name].numpy().astype(np.float32)
        t = raw.reshape(raw.shape[0], -1) if raw.ndim > 1 else raw.reshape(1, -1)
        r, c = t.shape
        assert c % 256 == 0, (name, c)
        dig, m, s32 = rtn_digits(t, k)
        blob = native.pack(dig, k, m)
        entries.append(dict(name=name, offset=off, bytes=len(blob),
                            rows=r, cols=c, k=k, flat=raw.ndim < 2))
        blobs.append(blob)
        off += len(blob)
    blob = b"".join(blobs)
    # dequant-roundtrip gate
    for e in entries:
        part = blob[e["offset"]:e["offset"] + e["bytes"]]
        w, code, s = native.unpack(np.frombuffer(part, np.uint8),
                                   e['cols'], e['rows'], native.type_for_k(k))
        assert np.isfinite(w).all(), e["name"]
    return blob, entries


def tensor_to_state(entries, blob, conf, dec="k9"):
    """Rebuild a torch state with K9-dequantized fp32 weights (tied embed kept
    explicit; lm_head tied after load)."""
    st = {}
    for e in entries:
        part = blob[e["offset"]:e["offset"] + e["bytes"]]
        w, _, _ = native.unpack(np.frombuffer(part, np.uint8),
                                e['cols'], e['rows'], native.type_for_k(e['k']))
        sh = (e["cols"],) if e.get("flat") else (e["rows"], e["cols"])
        st[e["name"]] = torch.from_numpy(w.reshape(sh).copy())
    return st


def main() -> None:
    ap_out = REPO / "fpga" / "partner" / "char9_img"
    base = HERE / "char9_out3"
    conf = json.loads((base / "conf.json").read_text())
    st0 = load_file(str(base / "state.safetensors"))

    # val split identical to the trainer's val-guard (last 10 %)
    stoi = {c: i for i, c in enumerate(sorted(set(
        (HERE / "data/input.txt").read_text(encoding="utf-8"))))}
    text = (HERE / "data/input.txt").read_text(encoding="utf-8")
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(len(data) * 0.10)
    val = data[len(data) - n_val - conf["seq"]:]

    model = TinyLlama(conf)
    model.tie()

    def val_loss(state):
        cur = {k: v for k, v in state.items()}
        full = {**cur, "lm_head.weight": cur["embed.weight"]}
        model.load_state_dict(full, strict=True)
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(val) - conf["seq"] - 1, conf["seq"] * 8):
                xb = val[i:i + conf["seq"] - 1][None]
                yb = val[i + 1:i + conf["seq"]]
                model._pos = 0
                losses.append(F.cross_entropy(
                    model(xb).reshape(-1, conf["vocab"]), yb).item())
        return float(sum(losses) / len(losses))

    v_fp32 = val_loss(st0)
    print(f"fp32 val loss   = {v_fp32:.5f}")

    ap_out.mkdir(parents=True, exist_ok=True)
    rows = [("fp32 (state.safetensors)", v_fp32, None)]
    for key, k in K_TABLE.items():
        blob, entries = pack_model(st0, conf, k)
        v_q = val_loss(tensor_to_state(entries, blob, conf))
        print(f"{key} (k={k:3d}) val loss = {v_q:.5f}   "
              f"image = {len(blob)/2**20:.2f} MiB")
        rows.append((f"{key} (k={k})", v_q, len(blob)))
        (ap_out / f"{key}.k9img").write_bytes(blob)
        (ap_out / f"{key}.manifest.json").write_text(json.dumps(
            dict(conf=conf, tensors=entries, bytes=len(blob)), indent=2))
    (ap_out / "quality.md").write_text(
        "# char9 resident-image quality (val = last 10% of tinyshakespeare)\n\n"
        "| variant | val loss | image size |\n|---|---:|---:|\n"
        + "\n".join(f"| {n} | {v:.5f} | "
                    f"{(b/2**20 if b else 0):.2f} MiB |" for n, v, b in rows)
        + "\n")
    print(f"images + manifest written to {ap_out}")


if __name__ == "__main__":
    main()