#!/usr/bin/env python3
"""Quantization, visually: classic formats vs ninths (K9) digit containers.

Applies the model-quantization formats to a 2D image (Robonaut 2, NASA PD)
in linear light so the visible differences mirror what each format does to
weight tensors. Classic formats use one per-tensor scale (the naive
baseline); ninths panels use the normative container grid from
docs/12-k9-container-spec.md section 4.1:

    per 64-value group, m = group max-magnitude, H = (k-1)/2,
    W = m * (d - H) / H,  digits d in [0, k-1]

with one addition disclosed in the README: pixels are non-negative while
weights are signed, so each group is mean-centered before gridding.

Outputs (same directory): grid_classic.png, grid_ninths.png, grid_error.png,
source_master.png.
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_CROP = os.path.join(HERE, "source_r2head.png")
MASTER = 768            # quantize at display resolution: 1 shown pixel = 1 quantized value
GROUP = 64              # ninths group width (spec: 64 columns per group)
PANEL = MASTER
FONT_DIR = "/usr/share/fonts/TTF"

# ---------------------------------------------------------------- color space


def srgb_to_lin(v_u8: np.ndarray) -> np.ndarray:
    v = v_u8.astype(np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def lin_to_srgb_u8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def psnr_db(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    mse = np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0**2 / mse)


def digit_entropy(d: np.ndarray, k: int) -> float:
    """Order-0 entropy of the digit stream (one shared freq table per tensor,
    matching the ninths container's rANS table)."""
    h = np.bincount(d.ravel(), minlength=k).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ------------------------------------------------------- classic formats


def _nearest_table(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(table, x)
    lo = np.clip(idx - 1, 0, len(table) - 1)
    hi = np.clip(idx, 0, len(table) - 1)
    pick_hi = np.abs(table[hi] - x) < np.abs(x - table[lo])  # ties -> lower
    return np.where(pick_hi, table[hi], table[lo])


def _e4m3_table() -> np.ndarray:
    """All 126 positive finite E4M3 (fn variant) values."""
    vals = [2.0**-6 * m / 8 for m in range(1, 8)]            # subnormals
    for e in range(-6, 9):                                    # normals
        for m in range(8):
            if e == 8 and m == 7:
                continue                                      # NaN encoding
            vals.append(2.0**e * (1.0 + m / 8.0))
    return np.array(sorted(vals))


E4M3 = _e4m3_table()
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])    # FP4 (E2M1)


def q_fp64(x):  # reference: the tensor itself
    return x.copy()


def q_fp16(x):
    return x.astype(np.float16).astype(np.float64)


def q_fp8(x):  # E4M3, no scale needed: the exponent self-adjusts
    return _nearest_table(x, E4M3)


def q_fp4(x):  # E2M1 with one per-tensor scale, MXFP4-style (block e8m0 scaled up to tensor)
    sigma = x.max() / 6.0
    return _nearest_table(x / sigma, E2M1) * sigma


def q_int8(x):  # one asymmetric [0,255] mapping for the whole image
    s = x.max() / 255.0
    return np.clip(np.rint(x / s), 0, 255) * s


def q_int4(x):  # one asymmetric [0,15] mapping for the whole image
    s = x.max() / 15.0
    return np.clip(np.rint(x / s), 0, 15) * s


def q_ternary(x):  # BitNet b1.58-faithful: absmean scale, digits {-1,0,1}
    gamma = x.mean()
    return np.clip(np.clip(np.rint(x / gamma), -1, 1) * gamma, 0.0, None)


def q_binary(x):  # BNN-faithful: centered sign, absmean-of-deviation scale
    c = x.mean()
    s = np.abs(x - c).mean()
    return np.clip(c + np.sign(x - c) * s, 0.0, None)


# ------------------------------------------------------------ ninths grid


def ninths_digits(x: np.ndarray, k: int, group: int = GROUP) -> np.ndarray:
    """docs/12 section 4.1 grid on mean-centered groups (pixels >= 0).
    Returns digits d in [0, k-1] shaped like the padded (rows, -1, group) view."""
    hh = (k - 1) // 2
    rows, ncol = x.shape[0], int(np.prod(x.shape[1:]))
    m2 = x.reshape(rows, ncol)
    pad = (-ncol) % group
    if pad:
        m2 = np.pad(m2, ((0, 0), (0, pad)), mode="edge")
    rg = m2.reshape(rows, -1, group)
    c = rg.mean(axis=2, keepdims=True)
    dev = rg - c
    gmax = np.maximum(np.abs(dev).max(axis=2, keepdims=True), 1e-12)
    return np.clip(np.rint(dev / gmax * hh), -hh, hh).astype(np.int32) + hh


def ninths_quant(x: np.ndarray, k: int, group: int = GROUP):
    d = ninths_digits(x, k, group)
    hh = (k - 1) // 2
    rows, ncol = x.shape[0], int(np.prod(x.shape[1:]))
    pad = (-ncol) % group
    m2 = x.reshape(rows, ncol)
    if pad:
        m2 = np.pad(m2, ((0, 0), (0, pad)), mode="edge")
    rg = m2.reshape(rows, -1, group)
    c = rg.mean(axis=2, keepdims=True)
    dev = rg - c
    gmax = np.maximum(np.abs(dev).max(axis=2, keepdims=True), 1e-12)
    deq = c + gmax * (d - hh) / hh
    out = deq.reshape(rows, -1)[:, :ncol].reshape(x.shape)
    return out, digit_entropy(d, k)


def q_digits_map(x, k=9):  # digit stream of the luma plane — what rANS codes
    luma = x.mean(axis=2, keepdims=True)
    d = ninths_digits(luma, k)
    g = d.astype(np.float64) / (k - 1)
    return g.reshape(x.shape[0], x.shape[1])


# -------------------------------------------------------------- error map


_MAGMA = [
    (0.00, (0, 0, 4)),
    (0.25, (87, 16, 110)),
    (0.50, (188, 55, 84)),
    (0.75, (249, 142, 9)),
    (1.00, (252, 255, 164)),
]


def magma_lut() -> np.ndarray:
    xs = np.array([a for a, _ in _MAGMA])
    lut = np.zeros((256, 3))
    for c in range(3):
        ys = np.array([rgb[c] for _, rgb in _MAGMA])
        lut[:, c] = np.interp(np.linspace(0, 1, 256), xs, ys)
    return lut


def error_map(deq_lin: np.ndarray, src_lin: np.ndarray, gain: int = 64) -> np.ndarray:
    """Amplified absolute error in LINEAR light — the space quantization happens in."""
    err = np.abs(deq_lin - src_lin).mean(axis=2) * gain
    lut = magma_lut()
    return lut[np.clip(np.rint(err * 255), 0, 255).astype(np.uint8)].astype(np.uint8)


# ----------------------------------------------------------------- layout


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def compose(panels, out_path, title, footer):
    """panels: list of (img_u8 | 'err', heading, statline)"""
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    pad, gut, lab_h, head_h, foot_h = 28, 18, 118, 132, 118
    W = pad * 2 + cols * PANEL + (cols - 1) * gut
    H = head_h + rows * (PANEL + lab_h) + (rows - 1) * gut + foot_h
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)

    f_title = _font("DejaVuSans-Bold.ttf", 58)
    f_head = _font("DejaVuSans-Bold.ttf", 44)
    f_stat = _font("DejaVuSans.ttf", 27)
    f_foot = _font("DejaVuSans.ttf", 24)

    tw = dr.textlength(title, font=f_title)
    dr.text(((W - tw) / 2, 34), title, fill=(17, 17, 17), font=f_title)

    for i, (im, head, stat) in enumerate(panels):
        cx = pad + (i % cols) * (PANEL + gut)
        cy = head_h + (i // cols) * (PANEL + lab_h + gut)
        dr.rectangle((cx - 2, cy - 2, cx + PANEL + 1, cy + PANEL + 1), outline=(200, 200, 200), width=1)
        img.paste(Image.fromarray(im), (cx, cy))
        hw = dr.textlength(head, font=f_head)
        dr.text((cx + (PANEL - hw) / 2, cy + PANEL + 12), head, fill=(17, 17, 17), font=f_head)
        sw = dr.textlength(stat, font=f_stat)
        if sw > PANEL:
            f_stat2 = _font("DejaVuSans.ttf", 23)
            sw = dr.textlength(stat, font=f_stat2)
            dr.text((cx + (PANEL - sw) / 2, cy + PANEL + 68), stat, fill=(90, 90, 90), font=f_stat2)
        else:
            dr.text((cx + (PANEL - sw) / 2, cy + PANEL + 68), stat, fill=(90, 90, 90), font=f_stat)

    y = H - foot_h + 8
    for line in textwrap.wrap(footer, width=150):
        fw = dr.textlength(line, font=f_foot)
        dr.text(((W - fw) / 2, y), line, fill=(70, 70, 70), font=f_foot)
        y += 30
    img.save(out_path)
    print(f"wrote {out_path} ({W}x{H})")


# ------------------------------------------------------------------- main


def main():
    src_img = Image.open(SOURCE_CROP).convert("RGB")
    if src_img.size != (MASTER, MASTER):
        src_img = src_img.resize((MASTER, MASTER), Image.LANCZOS)
    src_img.save(os.path.join(HERE, "source_master.png"))
    src_u8 = np.array(src_img)
    lin = srgb_to_lin(src_u8)

    src_psnr_ref = src_u8
    classic, ninths, errpanels = [], [], []

    def run_classic(tag, fn, head, stat):
        deq = fn(lin)
        u8 = lin_to_srgb_u8(deq)
        p = psnr_db(u8, src_psnr_ref)
        stat = stat.format(psnr=("∞" if p == float("inf") else f"{p:.1f}"))
        classic.append((u8, head, stat))
        errpanels.append((error_map(deq, lin), head, stat))
        print(f"  {tag:<12} PSNR {p:.2f} dB")

    def run_ninths(tag, k, head, stat_extra=""):
        deq, H = ninths_quant(lin, k)
        u8 = lin_to_srgb_u8(deq)
        p = psnr_db(u8, src_psnr_ref)
        raw = np.log2(k)
        stat = f"{raw:.2f} b/v raw · rANS ≈ {H:.2f} b/v · PSNR {p:.1f} dB{stat_extra}"
        ninths.append((u8, head, stat))
        errpanels.append((error_map(deq, lin), head, stat))
        print(f"  {tag:<12} PSNR {p:.2f} dB  H {H:.3f} b/v")

    print("classic formats (per-tensor scale, linear light):")
    run_classic("fp64", q_fp64, "FP64", "64 b/v · PSNR {psnr} dB · reference")
    run_classic("fp16", q_fp16, "FP16", "16 b/v · PSNR {psnr} dB")
    run_classic("fp8", q_fp8, "FP8 (E4M3)", "8 b/v · PSNR {psnr} dB · scale-free")
    run_classic("int8", q_int8, "INT8", "8 b/v · PSNR {psnr} dB · per-tensor scale")
    run_classic("fp4", q_fp4, "FP4 (E2M1)", "4 b/v · PSNR {psnr} dB · per-tensor scale")
    run_classic("int4", q_int4, "INT4", "4 b/v · PSNR {psnr} dB · per-tensor scale")
    run_classic("ternary", q_ternary, "1.58-bit", "1.58 b/v · PSNR {psnr} dB · absmean scale")
    run_classic("binary", q_binary, "1-bit", "1 b/v · PSNR {psnr} dB · absmean scale")

    print("ninths containers (docs/12 §4.1 grid, per-64-group scale):")
    ninths.append((src_u8, "source", "Robonaut 2 · NASA iss026e034306 · public domain"))
    run_ninths("k99", 99, "k99", " · spec range k ≥ 9")
    run_ninths("k63", 63, "k63", " · the workhorse")
    run_ninths("k15", 15, "k15", "")
    run_ninths("k9", 9, "k9", " · spec floor")
    run_ninths("k5", 5, "k5*", " · beyond spec v1.0")
    run_ninths("k3", 3, "k3*", " · beyond spec v1.0")

    dmap = np.clip(q_digits_map(lin, 9), 0, 1)
    du8 = np.clip(np.rint(dmap * 255), 0, 255).astype(np.uint8)
    du8 = np.repeat(du8[..., None], 3, axis=2)
    dl, Hl = ninths_quant(lin.mean(axis=2, keepdims=True), 9)
    ninths.append((du8, "k9 digit map",
                   f"luma digits d ∈ [0, 8] — what rANS codes · {Hl:.2f} b/v after coding"))

    compose(classic, os.path.join(HERE, "grid_classic.png"),
            "Quantization, visually — classic model formats on a 2D image",
            "Quantized in linear light (sRGB decoded), round-to-nearest, no dithering. Classic formats use one scale "
            "for the whole image (per-tensor). b/v = bits per value; PSNR vs the 8-bit source in display space. "
            "FP16 and FP64 are display-identical here (source is an 8-bit photo) — their residual is amplified in the "
            "error sheet. Photo: Robonaut 2 aboard the ISS — NASA, public domain.")
    compose(ninths, os.path.join(HERE, "grid_ninths.png"),
            "Quantization, visually — ninths (K9) digit containers on a 2D image",
            "docs/12 §4.1 grid: per-64-value group max-magnitude scale m, W = m·(d−H)/H, H = (k−1)/2 — plus group "
            "mean-centering (pixels ≥ 0; weights are signed). rANS ≈ measured order-0 digit entropy (one shared freq "
            "table, as in the container). * k5/k3 are extreme rungs outside spec v1.0 (k ≥ 9). Same photo: NASA PD.")
    compose(errpanels, os.path.join(HERE, "grid_error.png"),
            "What quantization throws away — |error| × 64 in linear light",
            "Amplified absolute error of each panel vs the source, in the space quantization happens in (magma "
            "colormap, black = exact). FP64 is exactly zero; FP16's residual is real but sits below 8-bit display "
            "visibility; FP8/INT8 banding and the ternary/binary collapse become obvious once amplified. "
            "Top rows: classic formats · bottom rows: ninths containers. Same photo and pipeline as the other grids.")


if __name__ == "__main__":
    main()
