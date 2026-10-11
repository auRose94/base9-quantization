"""Kernel-level real-tensor probe: feed real-tensor npz records straight into
the V1 K9_6 wide-port kernel (the op-for-op C-replicated semantic reference)
and dump per-record partials + chained sumf vs the goldens.

Purpose + history: on the real-model payload the full-top (tb_gate0sd) FAILed
while numpy == verbatim C held; the per-record dump (fpga/analyze_dump.py)
proved every per-superblock partial was bit-exact and the divergence was the
chain-tree protocol, not the hardware. This probe re-checks the per-call
level against any npz (REAL_NPZ env), including the corrected protocol's
npz — cheap isolation run for any future value-dependent divergence.

Run: cd fpga/sim && make SIM=verilator TOP=k96_vec_dot MOD=tb_real96 \
       RTL="../rtl/f32_pkg.sv ../rtl/k96_vec_dot.sv"
Env: REAL_NPZ (default /tmp/gate0_real_vecs.npz)
"""
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import b32f, f32b, k96_partials

REAL_NPZ = os.environ.get("REAL_NPZ", "/tmp/gate0_real_vecs.npz")


def make_clock(dut):
    for kwargs in ({"units": "ns"}, {"unit": "ns"}):
        try:
            return Clock(dut.clk, 2, **kwargs)
        except TypeError:
            continue
    raise TypeError("no Clock signature worked")


async def run_sb(dut, wbytes, acts):
    dut.wblk.value = int.from_bytes(bytes(wbytes), "little")
    dut.act.value = int.from_bytes(b"".join(acts), "little")
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    cyc = 0
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        cyc += 1
    return int(dut.partial_bits.value), cyc


@cocotb.test()
async def probe(dut):
    cocotb.start_soon(make_clock(dut).start())
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    data = np.load(REAL_NPZ)
    import re
    idxs = sorted(int(re.match(r"rec(\d+)_wblk", k).group(1))
                  for k in data.files if re.match(r"rec(\d+)_wblk", k))
    recs = [(data[f"rec{i}_wblk"],
             [bytes(data[f"rec{i}_act"][34 * j: 34 * (j + 1)]) for j in range(8)],
             int(data[f"rec{i}_gold"]), int(data[f"rec{i}_case"])) for i in idxs]

    acc = np.float32(0.0)
    prev_case = None
    n_pdiv = n_cdiv = 0
    cycles = []
    print(f"{'rec':>3} {'cs':>3} {'partial-vs-singleton':>20} {'chained-vs-gold':>16} {'cyc'}")
    for n_rec, (wblk, acts, gold, case) in enumerate(recs):
        got, cyc = await run_sb(dut, wblk, acts)
        cycles.append(cyc)
        sgl, _ = k96_partials([bytes(wblk)], acts)
        acc = np.float32(0.0 if prev_case is None or case != prev_case else acc)
        acc = np.float32(acc + b32f(got))
        p_ok = "OK" if got == int(sgl[0]) else "DIFF"
        c_ok = "OK" if f32b(acc) == gold else "FAIL"
        if p_ok != "OK":
            n_pdiv += 1
        if c_ok != "OK":
            n_cdiv += 1
        print(f"{n_rec:3d} {case:3d} {p_ok:>20} {c_ok:>16} {cyc:3d}")
        prev_case = case
    cy = np.array(cycles)
    print(f"kernel-level probe: partial diffs {n_pdiv}, chain diffs {n_cdiv} "
          f"({'PASS' if n_pdiv == 0 else 'FAIL'})")
    print(f"cycles per superblock: mode {np.bincount(cy).argmax()}, "
          f"min {cy.min()}, max {cy.max()}, mean {cy.mean():.1f} (V1 wide-port shape)")