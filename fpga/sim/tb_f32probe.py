"""Fuzz f32_pkg.f32_mul / f32_add against numpy IEEE fp32 (bit-exact)."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from golden import b32f, f32b


def make_clock(dut):
    for kwargs in ({"units": "ns"}, {"unit": "ns"}):
        try:
            return Clock(dut.clk, 2, **kwargs)
        except TypeError:
            continue
    raise TypeError("no Clock signature worked")


def golden_add(a: int, b: int) -> int:
    return f32b(b32f(a) + b32f(b))


def golden_mul(a: int, b: int) -> int:
    return f32b(b32f(a) * b32f(b))


@cocotb.test()
async def fuzz(dut):
    import numpy as np

    cocotb.start_soon(make_clock(dut).start())
    rng = np.random.default_rng(3)
    n_fail = 0
    n_test = 0
    n_skip = 0

    def norm_pair():
        while True:
            ea, eb = rng.integers(64, 255, 2)
            ma, mb = rng.integers(0, 2**23, 2)
            # keep products/sums away from subnormal results ($error scope)
            if ea + eb - 127 < 1:
                continue
            break
        return (ea << 23) | int(ma), (eb << 23) | int(mb)

    cases = []
    for _ in range(2500):
        a, b = norm_pair()
        if rng.random() < 0.5:
            a |= 1 << 31
        if rng.random() < 0.5:
            b |= 1 << 31
        cases.append((a, b, "norm"))
    for _ in range(250):  # adversarial exponent gaps incl. > 47 (sticky path)
        d = int(rng.integers(0, 120))
        while True:
            eb = int(rng.integers(100, 255 - d)) if d > 0 else int(rng.integers(100, 255))
            ea = eb + d
            if ea + eb - 127 < 1:
                continue
            break
        ma, mb = rng.integers(0, 2**23, 2)
        a, b = (ea << 23) | int(mb), (eb << 23) | int(mb)
        if rng.random() < 0.5:
            a |= 1 << 31
        if rng.random() < 0.5:
            b |= 1 << 31
        cases.append((a, b, f"gap{d}"))
    for _ in range(250):  # near-cancellation: equal exponents, close mantissas
        e = int(rng.integers(100, 255))
        ma = int(rng.integers(0, 2**23))
        dlt = int(rng.integers(-4, 5))
        mb = min(max(ma + dlt, 0), 2**23 - 1)
        sgn_a, sgn_b = (int(rng.integers(0, 2)) << 31), (int(rng.integers(0, 2)) << 31)
        cases.append((sgn_a | (e << 23) | ma, sgn_b | (e << 23) | mb, "cancel"))
    specials = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000,
                0xFF800000, 0x4B7FFFFF, 0x00800000, 0x3FC00000, 0x3F800001]
    for a in specials:
        for b in specials:
            cases.append((a, b, "special"))

    for a, b, tag in cases:
        for sel, name in ((0, "mul"), (1, "add")):
            # skip NaN-generating specials (canonical NaN bits differ by platform)
            ae, be = (a >> 23) & 0xFF, (b >> 23) & 0xFF
            am, bm = a & 0x7FFFFF, b & 0x7FFFFF
            if name == "mul" and ((ae == 0xFF and am != 0) or (be == 0xFF and bm != 0)
                                  or (ae == 0xFF and am == 0 and bm == 0 and be == 0xFF)):
                continue
            if name == "add" and ((ae == 0xFF and am != 0) or (be == 0xFF and bm != 0)):
                continue
            if name == "mul" and ((ae == 0xFF and am == 0 and be == 0 and bm == 0)
                                  or (be == 0xFF and bm == 0 and ae == 0 and am == 0)):
                continue
            want = golden_mul(a, b) if sel == 0 else golden_add(a, b)
            # scoped out (unit flushes these with $error): subnormal results
            if ((want >> 23) & 0xFF) == 0 and (want & 0x7FFFFF) != 0:
                n_skip += 1
                continue
            # numpy nan bits are platform-dependent; compare nan-ness instead
            import struct
            def isnan(v):
                return (v & 0x7F800000) == 0x7F800000 and (v & 0x7FFFFF) != 0
            dut.a.value = int(a)
            dut.b.value = int(b)
            dut.sel.value = int(sel)
            await RisingEdge(dut.clk)   # flop captures
            await RisingEdge(dut.clk)   # settle before read
            got = int(dut.y.value)
            n_test += 1
            if got != want and not (isnan(got) and isnan(want)):
                n_fail += 1
                if n_fail <= 10:
                    print(f"  {name} {tag} a={a:#010x} b={b:#010x}: "
                          f"rtl {got:#010x} vs np {want:#010x} FAIL")
    print(f"f32 fuzz: {n_test} cases, {n_fail} mismatches, {n_skip} skipped (subnormal out of scope)")
    assert n_fail == 0