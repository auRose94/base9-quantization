#!/usr/bin/env python3
"""rans_fast — ctypes binding for the C rANS coder (rans_fast.c).

Same semantics and byte-identical output to rans.py, ~50-100x faster (the
Python inner loop is the bottleneck, not the algorithm). Builds the shared
object on first import (gcc -O3 -shared -fPIC) and rebuilds when the .c changes;
falls back to the pure-Python coder if no compiler is available.

    from rans_fast import HAVE_FAST, rans_bits_fast, encode_fast, decode_fast
"""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from rans import normalize_freqs, _tables

HERE = Path(__file__).resolve().parent
SRC = HERE / "rans_fast.c"
SO = HERE / "k9rans.so"        # NOT rans_fast.so: that would shadow this module

HAVE_FAST = False
_lib = None


def _build():
    if not SRC.exists():
        return None
    if SO.exists() and SO.stat().st_mtime >= SRC.stat().st_mtime:
        return SO
    cc = os.environ.get("CC", "gcc")
    cmd = [cc, "-O3", "-funroll-loops", "-shared", "-fPIC", str(SRC), "-o", str(SO)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:                                    # no compiler
        print(f"rans_fast: build failed ({e}); using the Python coder",
              file=sys.stderr)
        return None
    return SO


def _load():
    global HAVE_FAST, _lib
    so = _build()
    if so is None:
        return
    try:
        lib = ctypes.CDLL(str(so))
    except Exception as e:
        print(f"rans_fast: load failed ({e}); using the Python coder",
              file=sys.stderr)
        return
    lib.k9_rans_encode.restype = ctypes.c_size_t
    lib.k9_rans_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.k9_rans_decode.restype = None
    lib.k9_rans_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    _lib = lib
    HAVE_FAST = True


_load()


def encode_fast(symbols, f, k):
    """Byte-identical to rans_encode(symbols, f, k) for uint8 alphabets."""
    if not HAVE_FAST:
        from rans import rans_encode
        return rans_encode(symbols, f, k)
    flat = np.ascontiguousarray(np.asarray(symbols, dtype=np.uint8).reshape(-1))
    n = flat.size
    cumul, _ = _tables(f, k)
    freq64 = np.ascontiguousarray(f, dtype=np.uint64)
    cum64 = np.ascontiguousarray(cumul[:-1], dtype=np.uint64)
    out = np.empty(2 * n + 8, dtype=np.uint8)
    m = _lib.k9_rans_encode(flat.ctypes.data, n, freq64.ctypes.data,
                            cum64.ctypes.data, int(k), out.ctypes.data,
                            out.size)
    if m == 0:
        raise RuntimeError("rANS encode buffer too small")
    return out[:m].tobytes()


def decode_fast(stream, f, n, k):
    """Byte-identical to rans_decode(stream, f, n, k)."""
    if not HAVE_FAST:
        from rans import rans_decode
        return rans_decode(stream, f, n, k)
    cumul, inv = _tables(f, k)
    freq64 = np.ascontiguousarray(f, dtype=np.uint64)
    cum64 = np.ascontiguousarray(cumul[:-1], dtype=np.uint64)
    inv32 = np.ascontiguousarray(inv, dtype=np.uint32)
    buf = np.frombuffer(stream, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    _lib.k9_rans_decode(buf.ctypes.data, buf.size, n, freq64.ctypes.data,
                        cum64.ctypes.data, inv32.ctypes.data, int(k),
                        out.ctypes.data)
    return out.astype(np.int64)


def rans_bits_fast(symbols, n_syms, k=12):
    """Compress + roundtrip-assert; returns bits/symbol."""
    f = normalize_freqs(np.asarray(symbols, dtype=np.int64), n_syms, 1 << k)
    enc = encode_fast(symbols, f, k)
    back = decode_fast(enc, f, len(symbols), k)
    assert np.array_equal(back.astype(np.uint8),
                          np.asarray(symbols, dtype=np.uint8)), "roundtrip"
    return 8.0 * (len(enc) + 4) / len(symbols)
