import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, "/home/rose/Work/base9-quantization/fpga/sim")
sys.path.insert(0, "/home/rose/Work/base9-quantization/experiments")
import ninths_native as nn
from golden import k97_partials, gen_act_blocks

SB_WORDS = 72
ACT_WORDS = 72


def byte_of(word, sel):
    return (word >> (8 * sel)) & 0xFF


@cocotb.test()
async def dbg_ext(dut):
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dalt = np.tile(np.array([0, 98], dtype=np.uint8), 128).reshape(1, 256)
    wb = bytes(nn.pack(dalt, 99, np.full((1, 4), 1.0)))
    acts = [bytes(a) for a in gen_act_blocks(np.random.default_rng(1), 8, codes=-128)]
    w = np.frombuffer(wb, np.uint8)
    d0 = [int.from_bytes(bytes(w[2 * j: 2 * j + 2]), "little") for j in range(4)]
    d1 = [int.from_bytes(bytes(a[:2]), "little") for a in acts]
    ablob = b"".join(acts)
    am = [int.from_bytes((ablob + b"\x00" * ACT_WORDS * 4)[4 * j:4 * j + 4], "little") for j in range(ACT_WORDS)]

    # trace the ACT + CODE ports for the first 24 consume-relevant cycles
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    n = 0
    last = {}
    while int(dut.done.value) == 0 and n < 900:
        await RisingEdge(dut.clk)
        a0 = int(dut.waddr0.value); a1 = int(dut.waddr1.value)
        aa = int(dut.aaddr0.value); sr = int(dut.sreq_addr.value)
        st = int(dut.k_st.value)
        t = int(dut.t.value) if hasattr(dut.t, "value") else -1
        # serve exactly like the tb
        q_w0 = int.from_bytes(wb[4 * ((a0 >> 2) % 66): 4 * ((a0 >> 2) % 66) + 4], "little")
        q_w1 = int.from_bytes(wb[4 * ((a1 >> 2) % 66): 4 * ((a1 >> 2) % 66) + 4], "little")
        q_a = am[(aa >> 2) % ACT_WORDS]
        sa = (aa >> 1) & 1
        w0 = byte_of(q_w0, a0 & 3); w1 = byte_of(q_w1, a1 & 3)
        pair = (((q_a >> (16 * sa + 8)) & 0xFF) << 8) | ((q_a >> (16 * sa)) & 0xFF)
        dut.wdata0.value = w0
        dut.wdata1.value = w1
        dut.adata0.value = pair
        if st == 4 and (t & 31) == 31:
            dut.sdata.value = d0[(t >> 5) & 3]
        else:
            dut.sdata.value = (d0[sr & 3] if (sr >> 3) & 1 else d1[sr & 7])
        cp0 = w0 - 256 if w0 >= 128 else w0
        cp1 = w1 - 256 if w1 >= 128 else w1
        ay0 = pair & 0xFF; ay1 = (pair >> 8) & 0xFF
        ay0 = ay0 - 256 if ay0 >= 128 else ay0
        ay1 = ay1 - 256 if ay1 >= 128 else ay1
        prod = cp0 * ay0 + cp1 * ay1
        last = dict(st=st, t=t, a0=a0, w0=f"{w0:#04x}", w1=f"{w1:#04x}",
                    aa=aa, pair=f"{pair:#06x}", sb=int(dut.sb_cap.value),
                    sblk=int(dut.sumi_block.value), prod=prod)
        if st in (0, 2, 3, 4, 5) or n % 3 == 0:
            print(f"  n={n} st={st} t={t} a0={a0} w=({cp0},{cp1}) act=({ay0},{ay1})"
                  f" prod={prod} sumi_block={last['sblk']} sreq={sr:#04x}")
        n += 1
    got = int(dut.partial_bits.value)
    np_fin = k97_partials([wb], acts)[1]
    print(f"RTL partial {got:#010x} vs numpy {np_fin:#010x}")
    assert got == np_fin, "ext-codes mismatch"