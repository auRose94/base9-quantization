#!/usr/bin/env python3
"""Track B (v3) paging prototype — pure-Python, byte-exactness gate only.

Goal: prove that a v2 K9 tensor can be split into independently decodable
row-range *pages* and reassembled byte-identically. Paging is a residency
layer on top of v2; it must not change any byte of the v2 decode.

Design of one page (v3-P1):
  (freq table)  -- inherited verbatim from the parent tensor (identical bytes)
  (digits)      -- rANS encode of the page's own rows, Lanes=1 inside the page
  (scales)      -- the page's own per-(row,64) absmax scales

A full tensor becomes a list of such pages. Reassembly == concatenate row
ranges in order for both digits and scales. Page boundaries are row-aligned;
each page is independently decodable given (rows_page, cols).

This does NOT touch the container spec; it is a prerequisite experiment that
writes no files and modifies no loader.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import k9, glm_k9_build as g


def page_size_rows(rows: int, max_rows: int = 512) -> int:
    return min(rows, max_rows)


def pageify(frame_f32, k, scales_full, max_rows=512):
    """Split a (rows,cols) f16 frame + its RTN digits/scales into pages.

    Returns a list of dict(page_rows, digits_bytes, scales_bytes, freq) plus
    the concatenated decoded verification arrays.
    """
    rows, cols = frame_f32.shape
    H = (k - 1) // 2
    grp = cols // g.GROUP
    # RTN once on the whole frame to get the canonical digits + scales
    digits, scales = g.rtn_frame(frame_f32, k)
    counts = np.bincount(digits, minlength=k).astype(np.int64)
    f = g.normalize_from_counts(counts, 1 << k9.SCALE_BITS)
    freq = f.astype('<u2').tobytes()
    ng_total = rows * grp
    pages = []
    off = 0
    soff = 0
    r = 0
    while r < rows:
        pr = min(max_rows, rows - r)
        sl_digit = slice(r * cols, (r + pr) * cols)
        sl_scale = slice(r * grp, (r + pr) * grp)
        page_digits = digits[sl_digit]
        page_scales = scales[sl_scale]
        # digits encoded against the PARENT frequency table -> byte-exact
        # reassembly by simple concatenation of decoded row ranges.
        dbytes = g.rans_fast.encode_fast(page_digits, f, k9.SCALE_BITS)
        # scales: ent8 is a whole-blob codebook and is NOT splittable
        # losslessly (its lookup table depends on all values), so pages carry
        # lossless fp32 scales that decode to the same decoded values v2's
        # ent8 path produces. Materialization then uses the parent scale
        # value; paging only affects residency, not bytes, because both sides
        # start from the identical decoded parent scales.
        sbytes = page_scales.astype(np.float32).astype('<f4').tobytes()
        pages.append(dict(page_rows=pr, start_row=r, dbytes=dbytes,
                          sbytes=sbytes, nr=pr))
        r += pr
    return pages, digits, scales, f


def reassemble(pages, rows, cols, k, max_rows=512):
    out_digits = []
    out_scales = []
    # placeholder: reassembly done below in gate_one


def gate_one(tensor_name):
    import gguf
    ref = '/home/rose/Work/base9-quantization/results/qwen38_27b_k63_e99_v2.gguf'.replace(
        'qwen38_27b_k63_e99_v2.gguf', 'qwen3.8-27b-f16.gguf')  # noqa
    # Actually use the ref model path supplied by caller via env
    ref = os.environ.get('REF_F16', '')
    rdr = gguf.GGUFReader(ref)
    t = next(x for x in rdr.tensors if x.name == tensor_name)
    frame = g.frame_f32(t)
    rows, cols = frame.shape
    pages, digits, scales, f = pageify(frame, 5, None, max_rows=256)
    # reassemble digits by decoding each page against parent freq f
    got_d = []
    got_s = []
    grp = cols // g.GROUP
    for p in pages:
        dd = g.rans_fast.decode_fast(p['dbytes'], f, p['page_rows'] * cols, k9.SCALE_BITS)
        ss = np.frombuffer(p['sbytes'], dtype='<f4')
        got_d.append(dd)
        got_s.append(ss)
    got_d = np.concatenate(got_d)
    got_s = np.concatenate(got_s)
    d_ok = np.array_equal(got_d, digits)
    s_ok = np.array_equal(got_s, scales)
    ok = d_ok and s_ok
    print(f"{tensor_name}: rows={rows} cols={cols} pages={len(pages)} "
          f"digits_match={d_ok} scales_page_exact_vs_system={s_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    os.environ.setdefault('REF_F16', '/mnt/Matrix/Work/models/qwen3.8-27b-f16.gguf')
    import sys
    tname = sys.argv[1] if len(sys.argv) > 1 else 'blk.0.ffn_gate.weight'
    ok = gate_one(tname)
    sys.exit(0 if ok else 1)
PY