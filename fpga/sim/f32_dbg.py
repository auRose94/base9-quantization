"""Offline replica of f32_pkg.sv (mul/add) to debug vs numpy — same algorithm,
Python integers, so divergences print with internals."""
import numpy as np


def f16_to_f32(h):
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0x1F:
        return (s << 31) | (0xFF << 23) | ((1 << 22) | (m << 12) if m else 0)
    if e == 0:
        if m == 0:
            return s << 31
        p = max(i for i in range(10) if (m >> i) & 1)
        sh = m << (23 - p)
        return (s << 31) | ((103 + p) << 23) | (sh & 0x7FFFFF)
    return (s << 31) | ((e + 112) << 23) | (m << 13)


def int_to_f32(n):
    if n == 0:
        return 0
    mag = -n if n < 0 else n
    p = max(i for i in range(31) if (mag >> i) & 1) if mag < (1 << 31) else 31
    sh = mag << (23 - p)
    return ((1 if n < 0 else 0) << 31) | ((127 + p) << 23) | (sh & 0x7FFFFF)


def f32_mul(xa, xb):
    sa, ea, ma = xa >> 31, (xa >> 23) & 0xFF, xa & 0x7FFFFF
    sb, eb, mb = xb >> 31, (xb >> 23) & 0xFF, xb & 0x7FFFFF
    sgn = sa ^ sb
    if (ea == 0xFF and ma) or (eb == 0xFF and mb):
        return 0x7FC00000
    if ea == 0xFF or eb == 0xFF:
        return (sgn << 31) | (0xFF << 23)
    if (ea == 0 and ma == 0) or (eb == 0 and mb == 0):
        return (sgn << 31)
    fa, fb = (1 << 23) | ma, (1 << 23) | mb
    p = fa * fb
    if (p >> 47) & 1:
        m24, g, sticky, er = p >> 24, (p >> 23) & 1, p & 0x7FFFFF, ea + eb - 126
    else:
        m24, g, sticky, er = (p >> 23) & 0xFFFFFF, (p >> 22) & 1, p & 0x3FFFFF, ea + eb - 127
    if g and (sticky or (m24 & 1)):
        m24 += 1
        if m24 == (1 << 24):
            m24 >>= 1
            er += 1
    if er >= 255:
        return (sgn << 31) | (0xFF << 23)
    assert er > 0, er
    return (sgn << 31) | (er << 23) | (m24 & 0x7FFFFF)


def f32_add(xa, xb):
    sa, ea, ma = xa >> 31, (xa >> 23) & 0xFF, xa & 0x7FFFFF
    sb, eb, mb = xb >> 31, (xb >> 23) & 0xFF, xb & 0x7FFFFF
    if (ea == 0xFF and ma) or (eb == 0xFF and mb):
        return 0x7FC00000
    if ea == 0xFF and eb == 0xFF and sa != sb:
        return 0x7FC00000
    if ea == 0xFF:
        return (sa << 31) | (0xFF << 23)
    if eb == 0xFF:
        return (sb << 31) | (0xFF << 23)
    if (ea == 0 and ma == 0) and (eb == 0 and mb == 0):
        return (1 << 31) if (sa and sb) else 0
    if ea == 0 and ma == 0:
        return xb
    if eb == 0 and mb == 0:
        return xa
    if ea >= eb:
        fbig, fsml, ebig, sbig, ssml = (1 << 23) | ma, (1 << 23) | mb, ea, sa, sb
        d = ea - eb
    else:
        fbig, fsml, ebig, sbig, ssml = (1 << 23) | mb, (1 << 23) | ma, eb, sb, sa
        d = eb - ea
    va = fbig << 24
    if d == 0:
        vb = fsml << 24
    elif d <= 24:
        vb = fsml << (24 - d)
    else:
        sh = d - 24
        vb = fsml >> sh if sh <= 24 else 0
        lost = fsml & ((1 << sh) - 1) if sh < 24 else fsml
        if lost:
            vb |= 1
    if sbig == ssml:
        vsum = va + vb
        if vsum >> 48:
            m24, g, r, stick = (vsum >> 25) & 0xFFFFFF, (vsum >> 24) & 1, (vsum >> 23) & 1, vsum & 0x7FFFFF
            er = ebig + 1
        else:
            m24, g, r, stick = (vsum >> 24) & 0xFFFFFF, (vsum >> 23) & 1, (vsum >> 22) & 1, vsum & 0x3FFFFF
            er = ebig
        sgn = sbig
    else:
        if vb > va:
            mag, sgn = vb - va, ssml
        elif va > vb:
            mag, sgn = va - vb, sbig
        else:
            return 0
        p = mag.bit_length() - 1
        sh = 47 - p
        mag <<= sh
        er = ebig - sh
        m24, g, r, stick = (mag >> 24) & 0xFFFFFF, (mag >> 23) & 1, (mag >> 22) & 1, mag & 0x3FFFFF
    if g and (r or stick or (m24 & 1)):
        m24 += 1
        if m24 == (1 << 24):
            m24 >>= 1
            er += 1
    if er >= 255:
        return (sgn << 31) | (0xFF << 23)
    assert er > 0, er
    return (sgn << 31) | (er << 23) | (m24 & 0x7FFFFF)


NPF = np.frombuffer


def npb(v):
    return int(NPF(struct_pack(v), dtype="<f4")[0].view(np.uint32))


import struct


def struct_pack(v):
    return struct.pack("<I", v & 0xFFFFFFFF)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    fails = {"mul": 0, "add": 0}
    shown = {"mul": 0, "add": 0}
    for it in range(20000):
        ea, eb = rng.integers(64, 255, 2)
        ma, mb = int(rng.integers(0, 2**23)), int(rng.integers(0, 2**23))
        a = (int(ea) << 23) | ma
        b = (int(eb) << 23) | mb
        for name in ("mul", "add"):
            got = (f32_mul if name == "mul" else f32_add)(a, b)
            x = NPF(struct_pack(a), dtype="<f4")[0]
            y = NPF(struct_pack(b), dtype="<f4")[0]
            z = x * y if name == "mul" else x + y
            want = int(z.view(np.uint32))
            if got != want:
                fails[name] += 1
                if shown[name] < 4:
                    shown[name] += 1
                    print(f"{name} a={a:#010x} b={b:#010x}: got {got:#010x} want {want:#010x}")
    print("FAILS:", fails)