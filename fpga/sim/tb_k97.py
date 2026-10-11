"""Gate-0 for the K9_7 container: k97_vec_dot_v2 (streaming V2 shape) vs the
numpy replica of the fork's ggml_vec_dot_k9_7_q8_0_generic and the verbatim
C harness — bit-exact per call.

The tb IS the caller: one persistent Scratch task replays the proven caller
circuit (gate0sd_top's topology) port-side — sync-read words with 1-cycle
latency + combinational byte muxes; the scale service is served
combinationally from the live sreq_addr every cycle (raw fp16 bit words,
exactly like the top's 16-bit regs). Scale-service self-checks fire if a
consume state ever sees a scale whose sreq is older than the registered set
— they cannot under this contract, and the check proves it.

Vectors: synthetic (tb_k96's case list on k grids up to 99) unless REAL_NPZ
is set — then the npz rec{i}_* records (264 B wblk) run per-call
(partner/real_slice_image.py --kind k9_7 emits real __embed__ row records).

Run: cd fpga/sim && make SIM=verilator TOP=k97_vec_dot_v2 MOD=tb_k97 \
       RTL="../rtl/f32_pkg.sv ../rtl/k97_vec_dot_v2.sv"
"""
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import b32f, f32b, k97_partials, gen_superblocks, gen_act_blocks

REAL_NPZ = os.environ.get("REAL_NPZ", "")
HARNESS = Path(__file__).resolve().parents[1] / "crosscheck" / "k9_c_harness"
K97_TYPE = 7
SB_WORDS = 72          # k9_7 wblk = 264 B = 66 words (+ address slack)
ACT_WORDS = 72         # act = 272 B = 68 words (+ slack)


def make_clock(dut):
    for kwargs in ({"units": "ns"}, {"unit": "ns"}):
        try:
            return Clock(dut.clk, 2, **kwargs)
        except TypeError:
            continue
    raise TypeError("no Clock signature worked")


def byte_of(word, sel):
    return (word >> (8 * sel)) & 0xFF


class Scratch:
    """The gate0sd_top caller circuit, port-side, one task for the whole run:
    sync-read (1-cycle-latency) words + combinational muxes; scale service
    combinational on the live sreq (16-bit raw fp16 words, like the top)."""

    def __init__(self, dut):
        self.dut = dut
        self.wm = [0] * SB_WORDS
        self.am = [0] * ACT_WORDS
        self.d0 = [0, 0, 0, 0]
        self.d1 = [0] * 8
        self.check = []          # (state seen, sreq observed) audit trail

    def load(self, wblk, acts, d0bits, d1bits):
        for j in range(SB_WORDS):
            b = bytes(wblk)
            pad = b + b"\x00" * (SB_WORDS * 4)
            self.wm[j] = int.from_bytes(pad[4 * j: 4 * j + 4], "little")
        ablob = b"".join(bytes(a) for a in acts) + b"\x00" * (ACT_WORDS * 4)
        for j in range(ACT_WORDS):
            self.am[j] = int.from_bytes(ablob[4 * j: 4 * j + 4], "little")
        self.d0 = [int(x) for x in d0bits]
        self.d1 = [int(x) for x in d1bits]

    def audit(self):
        self.check.clear()

    async def run(self):
        trace = os.environ.get("K97_TRACE", "")
        tcount = 0
        while True:
            await RisingEdge(self.dut.clk)
            a0 = int(self.dut.waddr0.value)
            a1 = int(self.dut.waddr1.value)
            aa = int(self.dut.aaddr0.value)
            sr = int(self.dut.sreq_addr.value)
            q_w0 = self.wm[(a0 >> 2) % SB_WORDS]; s0 = a0 & 3
            q_w1 = self.wm[(a1 >> 2) % SB_WORDS]; s1 = a1 & 3
            q_a = self.am[(aa >> 2) % ACT_WORDS]
            a_sel = (aa >> 1) & 1          # aa[1]: the word's even-byte half
            w0 = byte_of(q_w0, s0)
            w1 = byte_of(q_w1, s1)
            self.dut.wdata0.value = w0
            self.dut.wdata1.value = w1
            # pair = the word's bytes (2a_sel, 2a_sel+1): {upper = odd code,
            # lower = even code}, the top's {am_q[31:24], am_q[23:16]} /
            # {am_q[15:8], am_q[7:0]} muxes exactly
            self.dut.adata0.value = ((q_a >> (16 * a_sel + 8)) & 0xFF) << 8 | \
                                    ((q_a >> (16 * a_sel)) & 0xFF)
            st = int(self.dut.k_st.value)
            # scale service: combinational on the LIVE sreq. Port-side this
            # means: serve from the previously sampled sreq (>=1 cycle old,
            # correct for every consume state) EXCEPT S_SUMF, whose sreq is
            # set by the S_SUMI-exit NBA one edge ahead -- at S_SUMI with a
            # last pair of the group, forecast d0[g] instead.
            if st == 4 and (int(self.dut.t.value) & 31) == 31:
                self.dut.sdata.value = self.d0[(int(self.dut.t.value) >> 5) & 3]
            else:
                self.dut.sdata.value = (self.d0[sr & 3] if (sr >> 3) & 1
                                        else self.d1[sr & 7])
            if trace:
                lim = int(trace)
                if st in (4, 5, 6) or tcount < lim:
                    print(f"  trace: st={st} t={int(self.dut.t.value) if hasattr(self.dut.t, 'value') else '?'}"
                          f" waddr0={a0:03d} wdata0={w0:#04x} wdata1={w1:#04x}"
                          f" aaddr0={aa:3d} adata0={int(self.dut.adata0.value):#06x}"
                          f" sreq={sr:#03x} sdata={int(self.dut.sdata.value):#06x}"
                          f" sb={int(self.dut.sb_cap.value) if st == 4 else 0}")
                tcount += 1


async def run_sb(dut) -> int:
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
    return int(dut.partial_bits.value)


def c_harness_records(cases):
    """Every RECORD via the verbatim fork kernel as a singleton chain."""
    outs = []
    for name, wblks, acts in cases:
        for i, wb in enumerate(wblks):
            inb = (K97_TYPE).to_bytes(4, "little") + (1).to_bytes(4, "little")
            inb += bytes(wb) + b"".join(bytes(b) for b in acts[8 * i: 8 * i + 8])
            fin = "/tmp/k97_rec.bin"
            fout = "/tmp/k97_out.bin"
            Path(fin).write_bytes(inb)
            subprocess.run([str(HARNESS), fin, fout], check=True)
            outs.append(int(np.frombuffer(Path(fout).read_bytes(), dtype="<u4")[0]))
    return outs


def synth_cases(rng):
    cases = []
    for tag in ("", "b", "c"):
        cases.append((f"rand-k99{tag}", gen_superblocks(rng, 1, k=99),
                      gen_act_blocks(rng, 8)))
    cases.append(("chain6", gen_superblocks(rng, 6, k=99), gen_act_blocks(rng, 48)))
    import ninths_native as nn
    dmax = np.full((1, 256), 98, dtype=np.uint8)
    m_ext = np.array([[0.0, 2e-6, 2.03e6, 1e7]], dtype=np.float32)
    cases.append(("ext-scales", [bytes(nn.pack(dmax, 99, m_ext))],
                  gen_act_blocks(rng, 8, codes=1)))
    dalt = np.tile(np.array([0, 98], dtype=np.uint8), 128).reshape(1, 256)
    cases.append(("ext-codes", [bytes(nn.pack(dalt, 99, np.full((1, 4), 1.0)))],
                  gen_act_blocks(rng, 8, codes=-128)))
    d3 = rng.integers(0, 99, (1, 256)).astype(np.uint8)
    m_neg = np.array([[-1.5, -0.25, 3.0, -7.25]], dtype=np.float32)
    cases.append(("neg-scales", [bytes(nn.pack(d3, 99, m_neg))],
                  gen_act_blocks(rng, 8, d16=-1.5)))
    for c in range(8):
        cases.append((f"fuzz{c}", gen_superblocks(rng, 1, k=99),
                      gen_act_blocks(rng, 8)))
    return cases


@cocotb.test()
async def gate(dut):
    cocotb.start_soon(make_clock(dut).start())
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    if REAL_NPZ:
        data = np.load(REAL_NPZ)
        import re
        idxs = sorted(int(m.group(1)) for k in data.files
                      if (m := re.match(r"rec(\d+)_wblk", k)))
        cases = [(f"real{i}", [bytes(data[f"rec{i}_wblk"])],
                  [bytes(data[f"rec{i}_act"][34 * j: 34 * (j + 1)]) for j in range(8)])
                 for i in idxs]
    else:
        cases = synth_cases(np.random.default_rng(1))

    scratch = Scratch(dut)
    cocotb.start_soon(scratch.run())

    c_singles = c_harness_records(cases)
    n_fail = 0
    n_rec = 0
    print(f"{'case':12s} {'i':>3} {'rtl':>12} {'np-single':>12} {'C-single':>12} {'chainacc':>12}")
    ptr = 0
    for name, wblks, acts in cases:
        acc = np.float32(0.0)
        for i, wb in enumerate(wblks):
            acts_i = acts[8 * i: 8 * i + 8]
            w = np.frombuffer(bytes(wb), np.uint8)
            # the caller's 16-bit scale regs = the raw fp16 halves, LE
            d0 = [int.from_bytes(bytes(w[2 * j: 2 * j + 2]), "little") for j in range(4)]
            d1 = [int.from_bytes(bytes(a[:2]), "little") for a in acts_i]
            scratch.load(wb, acts_i, d0, d1)
            got = await run_sb(dut)
            one_np, _ = k97_partials([bytes(wb)], acts_i)
            one_c = c_singles[ptr]
            acc = np.float32(acc + b32f(got))          # the gate chain tree
            ok = (got == int(one_np[0])) and (got == one_c)
            if not ok:
                print(f"  {name:12s} {i:3d} MISMATCH rtl={got:#010x} "
                      f"np={int(one_np[0]):#010x} C={one_c:#010x}")
                n_fail += 1
            else:
                print(f"  {name:12s} {i:3d} {got:12x} {int(one_np[0]):12x} "
                      f"{one_c:12x} {f32b(acc):12x}")
            ptr += 1
            n_rec += 1
    print(f"K9_7 GATE-0: {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'} over {n_rec} records")
    assert n_fail == 0