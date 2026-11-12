"""Analog write-path channel — weight-level fidelity (rung 1 of the analog ladder).

Models what a mixed-signal compute-in-memory array does to weights between
"program" and "hold":

  * Noise reference (scale_ref):
      "tensor" — noise is ABSOLUTE, referenced to the tensor's full scale
        (amax = max|w|; the differential-pair rails sit at +-amax).  Models a
        naive array where every cell shares one global gain.  Quantization
        error is level-referenced; analog noise is range-referenced, so small
        weights suffer proportionally more — this is the fundamental
        difference between the analog channel and a k-grid.
      "group"  — noise referenced to per-group (default 64) local full scale:
        a gain-calibrated array (per-tile/per-column reference).  The
        analog-native equivalent of what quantization scales do.
  * Programming / read noise: N(0, sigma * amax_ref), absolute in weight units.
  * Retention drift: multiplicative on each cell (N(1, sigma_drift)).
  * IR drop: systematic gain ramp across columns; the far column loses
    ir_alpha.  Applied to 2-D tensors only.
  * Stuck-at faults: a fraction stuck_p of cells saturate to their rail,
    sign drawn 50/50 (a failed comparator lands on Gmin or Gmax).
  * Optional resolvable levels: snap to a 2*amax_ref/(levels-1) grid — the
    "the array itself is a quantizer" knob.  0 = analog continuum.

Common random numbers: draws are seeded per (master seed, tensor name) and
scale with sigma, so every sweep point sees the same underlying noise
realization scaled by sigma — curves are smooth in sigma by construction.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


def tensor_rng(master: int, name: str) -> np.random.Generator:
    """Deterministic per-tensor generator: same (master, name) -> same draws."""
    return np.random.default_rng([int(master), zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF])


@dataclass
class AnalogParams:
    sigma_prog: float = 0.0   # programming error, fraction of amax_ref
    sigma_read: float = 0.0   # read/thermal noise, fraction of amax_ref
    sigma_drift: float = 0.0  # retention drift, relative to cell value
    ir_alpha: float = 0.0     # fractional gain loss at the far column
    stuck_p: float = 0.0      # fraction of cells stuck at a rail
    levels: int = 0           # resolvable levels across the range (0 = continuum)
    scale_ref: str = "tensor"  # "tensor" | "group"
    group: int = 64           # group size for scale_ref="group"
    rail_limit: float = 0.0   # absolute rail override; also pins the noise
                              # reference (array full scale) — for boost demos
                              # where the input arrives pre-amplified
    dead_zone: float = 0.0    # exact-zero protection: after noise, cells with
                              # |w'| < dead_zone * amax_ref snap to 0 (a
                              # per-group comparator; kills noise-induced
                              # sign flips of tiny weights)

    def noise_active(self) -> bool:
        return (self.sigma_prog > 0.0 or self.sigma_read > 0.0
                or self.sigma_drift > 0.0 or self.ir_alpha > 0.0
                or self.stuck_p > 0.0 or self.levels > 1
                or self.dead_zone > 0.0)


def analog_channel(w: np.ndarray, p: AnalogParams, rng: np.random.Generator) -> np.ndarray:
    """Apply the analog write path to one tensor.  Returns the same shape/dtype."""
    out = w.astype(np.float32, copy=True)
    if out.size == 0:
        return out.astype(w.dtype, copy=False)

    # reference amax: scalar (tensor) or per-group column (nb, 1), broadcastable
    grouped = p.scale_ref == "group" and p.rail_limit <= 0.0 and p.group > 1 \
        and out.ndim >= 1 and out.size % p.group == 0
    shp = out.shape
    if grouped:
        g = p.group
        out = out.reshape(-1, g)
        amax = np.abs(out).max(axis=1, keepdims=True).astype(np.float32)
    elif p.rail_limit > 0.0:
        amax = np.float32(p.rail_limit)  # fixed array full scale (boost demos)
    else:
        amax = np.float32(np.abs(out).max())
        if not p.noise_active():
            return out.astype(w.dtype, copy=False)  # pure identity fast path
    if float(np.max(amax)) == 0.0:
        return out.reshape(shp).astype(w.dtype, copy=False)

    def noise(sigma: float) -> np.ndarray:
        return (np.float32(sigma) * amax) * rng.standard_normal(out.shape, dtype=np.float32)

    if p.sigma_prog > 0.0:
        out += noise(p.sigma_prog)
    if p.sigma_read > 0.0:
        out += noise(p.sigma_read)
    if p.sigma_drift > 0.0:
        out *= np.float32(1.0) + np.float32(p.sigma_drift) * rng.standard_normal(
            out.shape, dtype=np.float32)
    if p.stuck_p > 0.0:
        stuck = rng.random(out.shape, dtype=np.float32) < p.stuck_p
        if stuck.any():
            rail = np.where(rng.random(out.shape, dtype=np.float32) < 0.5,
                            amax, -amax).astype(np.float32)
            out = np.where(stuck, rail, out)
    if p.levels and p.levels > 1:
        step = np.float32(2.0) * amax / np.float32(p.levels - 1)
        out = np.round(out / step) * step
    if p.dead_zone > 0.0:
        out[np.abs(out) < (p.dead_zone * amax)] = np.float32(0.0)
    np.clip(out, -amax, amax, out=out)
    if grouped:
        out = out.reshape(shp)
    # IR drop only scales down, so it cannot push values past the rails
    if p.ir_alpha > 0.0 and w.ndim == 2:
        ramp = np.linspace(np.float32(1.0), np.float32(1.0 - p.ir_alpha),
                           out.shape[1], dtype=np.float32)
        out *= ramp
    return out.astype(w.dtype, copy=False)


def selftest() -> None:
    rng = np.random.default_rng(7)
    w16 = (rng.standard_normal((256, 128)) * 0.02).astype(np.float16)
    w32 = rng.standard_normal((512, 256)).astype(np.float32)
    vec = rng.standard_normal(512).astype(np.float32)

    # 1) zero-noise identity, dtype preserved (f16 and f32, both scale refs)
    for ref in ("tensor", "group"):
        for arr in (w16, w32, vec):
            out = analog_channel(arr, AnalogParams(scale_ref=ref), tensor_rng(0, "t"))
            assert out.dtype == arr.dtype and np.array_equal(out, arr), \
                f"identity failed ({ref}, {arr.dtype})"

    # 2) MSE strictly increases with sigma (fixed CRN seed, both scale refs)
    for ref in ("tensor", "group"):
        mses = []
        for s in (0.01, 0.03, 0.1, 0.3, 1.0):
            out = analog_channel(w32, AnalogParams(sigma_prog=s, scale_ref=ref),
                                 tensor_rng(3, "t"))
            mses.append(float(np.mean((out - w32) ** 2)))
        assert all(a < b for a, b in zip(mses, mses[1:])), f"non-monotone MSE: {ref} {mses}"

    # 3) rails clip: huge noise -> nothing exceeds the local rail and almost
    #    every cell saturates (MSE -> amax^2 + var(w) at tensor reference)
    amax = float(np.abs(w32).max())
    out = analog_channel(w32, AnalogParams(sigma_prog=100.0), tensor_rng(1, "t"))
    assert np.abs(out).max() <= amax + 1e-6
    frac_at_rail = float(np.mean(np.abs(out) >= amax * (1.0 - 1e-6)))
    assert frac_at_rail > 0.95, f"saturation fraction too low: {frac_at_rail}"
    mse = float(np.mean((out - w32) ** 2))
    assert amax * amax * 0.8 < mse < amax * amax * 2.0, f"saturated MSE off: {mse}"

    # 4) finite levels snap onto the grid
    out = analog_channel(w32, AnalogParams(levels=9), tensor_rng(2, "t"))
    step = 2.0 * amax / 8.0
    assert np.allclose(out / step, np.round(out / step), atol=1e-4), "levels not on grid"

    # 5) stuck fraction lands on rails, approximately stuck_p of cells
    out = analog_channel(w32, AnalogParams(stuck_p=0.05), tensor_rng(4, "t"))
    frac = float(np.mean(np.abs(out) >= amax * (1.0 - 1e-6)))
    assert 0.03 < frac < 0.08, f"stuck fraction off: {frac}"

    # 6) IR ramp: columns scale by linspace(1, 1-alpha, cols) exactly
    out = analog_channel(w32, AnalogParams(ir_alpha=0.2), tensor_rng(5, "t"))
    ramp = np.linspace(1.0, 1.0 - 0.2, w32.shape[1], dtype=np.float32)
    assert np.allclose(out, w32 * ramp)

    # 7) common random numbers: same seed -> identical output across calls
    o1 = analog_channel(w32, AnalogParams(sigma_prog=0.1), tensor_rng(9, "same"))
    o2 = analog_channel(w32, AnalogParams(sigma_prog=0.1), tensor_rng(9, "same"))
    assert np.array_equal(o1, o2), "CRN broken"

    # 8) 1-D tensors take noise + drift but no ramp, and survive fine
    out = analog_channel(vec, AnalogParams(sigma_prog=0.02, ir_alpha=0.5), tensor_rng(6, "t"))
    assert out.shape == vec.shape

    # 9b) rail_limit pins rails + noise reference regardless of input scale,
    #     and clips even with zero noise (boost demo correctness)
    big = np.abs(w32).max()
    out = analog_channel(w32 * 10.0,
                         AnalogParams(sigma_prog=0.01, rail_limit=big),
                         tensor_rng(12, "t"))
    assert np.abs(out).max() <= big + 1e-6, "rail_limit not respected"
    assert float(np.mean(np.abs(out) >= big * (1 - 1e-6))) > 0.0, \
        "boosted input never hit the fixed rail"
    out0 = analog_channel(w32 * 10.0, AnalogParams(rail_limit=big),
                          tensor_rng(12, "t"))
    assert np.abs(out0).max() <= big + 1e-6 and not np.array_equal(out0, w32 * 10.0), \
        "zero-noise boost must still clip at the rail"
    err_pin = analog_channel(w32 / 10.0,
                             AnalogParams(sigma_prog=0.01, rail_limit=big),
                             tensor_rng(12, "t")) - w32 / 10.0
    err_def = analog_channel(w32 / 10.0, AnalogParams(sigma_prog=0.01),
                             tensor_rng(12, "t")) - w32 / 10.0
    assert float(np.abs(err_pin).max()) > 5.0 * float(np.abs(err_def).max()), \
        "rail_limit did not pin the noise reference"

    # 9) group reference: local gain calibration protects small-amplitude rows
    #    — build a tensor with two 64-wide groups of very different amax and
    #    check the small group's perturbation is far smaller under "group"
    #    than under "tensor"
    big = rng.standard_normal((1, 64)).astype(np.float32) * 10.0
    small = rng.standard_normal((1, 64)).astype(np.float32) * 0.01
    mix = np.concatenate([big, small], axis=1)
    err_t = analog_channel(mix, AnalogParams(sigma_prog=0.01, scale_ref="tensor"),
                           tensor_rng(11, "t")) - mix
    err_g = analog_channel(mix, AnalogParams(sigma_prog=0.01, scale_ref="group"),
                           tensor_rng(11, "t")) - mix
    small_t = float(np.abs(err_t[0, 64:]).max())
    small_g = float(np.abs(err_g[0, 64:]).max())
    assert small_g < small_t / 100.0, \
        f"group ref did not protect small group: {small_g} vs {small_t}"
    big_t = float(np.abs(err_t[0, :64]).max())
    big_g = float(np.abs(err_g[0, :64]).max())
    assert abs(big_t - big_g) < 0.25 * big_t, \
        f"group ref changed big-group noise scale unexpectedly: {big_g} vs {big_t}"

    # 10) dead zone: tiny cells snap to exact zero, large cells untouched
    #     (tested at sigma=0 for determinism)
    out = analog_channel(w32, AnalogParams(dead_zone=0.02), tensor_rng(13, "t"))
    tiny = np.abs(w32) < 0.02 * np.abs(w32).max()
    assert tiny.any() and np.all(out[tiny] == 0.0), "dead zone missed tiny cells"
    big_cells = np.abs(w32) > 0.06 * np.abs(w32).max()
    assert np.array_equal(out[big_cells], w32[big_cells]), "dead zone ate real weights"

    print("analog_channel selftest: ALL PASS")


if __name__ == "__main__":
    selftest()
