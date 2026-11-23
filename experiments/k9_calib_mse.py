#!/usr/bin/env python3
"""Track A experiment 2 prototype: importance-weighted MSE scale search.

For each 64-group, search a set of candidate scales and pick the one that
minimises the importance-weighted reconstruction error. Unlike experiment 1,
importance does NOT multiply the stored scale directly; it weights the error
term in the search, which is the quantity RTN actually wants to minimise.

Run on a single tensor and report weighted MSE gain vs. uncalibrated RTN.
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import glm_k9_build as g

K = 5


def candidates_for_group(w, imp, H, n_cand=16):
    amax = np.maximum(np.abs(w), 1e-30)
    wmax = float(amax.max())
    # log-spaced candidates around the unweighted amax
    base = np.geomspace(wmax * 0.5, wmax * 2.0, n_cand)
    return base


def optimize_tensor(frame, imp_cols, k=K, rows_chunk=64, n_cand=16):
    rows, cols = frame.shape
    ng = cols // g.GROUP
    H = (k - 1) // 2
    imp = imp_cols.astype(np.float32).reshape(1, cols, 1)
    digits = np.empty(rows * cols, dtype=np.uint8)
    scales = np.empty(rows * ng, dtype=np.float32)
    uncal_mse = 0.0
    opt_mse = 0.0
    wimp = imp.reshape(1, ng, g.GROUP)
    for a in range(0, rows, rows_chunk):
        b = min(rows, a + rows_chunk)
        Wc = np.ascontiguousarray(frame[a:b], dtype=np.float32).reshape(b - a, ng, g.GROUP)
        r, GG, G = Wc.shape
        impg = wimp  # (1,ng,G)
        # uncalibrated baseline (plain amax)
        m0 = np.maximum(np.abs(Wc).max(axis=2, keepdims=True), 1e-30)
        idx0 = np.rint((Wc + m0) * (float(H) / m0))
        np.clip(idx0, 0, k - 1, out=idx0)
        dq0 = m0 * (idx0 - H) / H
        uncal_mse += float(np.sum(wimp * (dq0 - Wc) ** 2))
        # candidate scales
        amax_per = np.maximum(np.abs(Wc).max(axis=2), 1e-30)  # (r,GG)
        # shape (cand, r, GG, 1)
        cands = np.geomspace(amax_per * 0.5, amax_per * 2.0, n_cand, axis=0) \
            if False else None
        # simpler: per-group log candidates
        best_err = np.full((r, GG), np.inf)
        best_m = np.zeros((r, GG), dtype=np.float32)
        best_idx = np.zeros((r, GG, G), dtype=np.int64)
        for ci in range(n_cand):
            # candidate scale multiplier factor
            factor = 0.5 + 1.5 * ci / max(n_cand - 1, 1)
            m = amax_per * factor
            m = np.maximum(m, 1e-30)[:, :, None]
            idx = np.rint((Wc + m) * (float(H) / m))
            np.clip(idx, 0, k - 1, out=idx)
            dq = m * (idx - H) / H
            err = np.sum(wimp * (dq - Wc) ** 2, axis=2)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_m = np.where(better, m[:, :, 0], best_m)
            # store idx for rows where better
            # need full idx array update
            # store always last best_idx where better
            best_idx = np.where(better[:, :, None], idx.astype(np.int64), best_idx)
        # write outputs
        digits[a * cols:b * cols] = best_idx.reshape(-1).astype(np.uint8)
        scales[a * ng:b * ng] = best_m.reshape(-1)
        dq_opt = best_m[:, :, None] * (best_idx - H) / H
        opt_mse += float(np.sum(wimp * (dq_opt - Wc) ** 2))
    return digits, scales, uncal_mse, opt_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--tensor', required=True)
    ap.add_argument('--imatrix', required=True)
    a = ap.parse_args()
    import gguf
    rdr = gguf.GGUFReader(a.ref)
    t = next(x for x in rdr.tensors if x.name == a.tensor)
    frame = g.frame_f32(t)
    ne0 = int(t.shape[0])
    assert frame.shape[1] == ne0
    with np.load(a.imatrix) as z:
        imp = np.asarray(z[a.tensor], dtype=np.float32).reshape(-1)
    assert imp.shape[0] == ne0
    digits, scales, umse, omse = optimize_tensor(frame, imp)
    print(f"tensor={a.tensor} frame={frame.shape}")
    print(f"uncalibrated weighted MSE: {umse:.6e}")
    print(f"optimized     weighted MSE: {omse:.6e}  gain {umse/max(omse,1e-30):.3f}x")


if __name__ == '__main__':
    main()
PY