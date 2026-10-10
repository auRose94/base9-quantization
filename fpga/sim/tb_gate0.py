"""Full-top silicon-gate simulation: gate0v2_top (fpga/synth) in Verilator.

Validates the V2 (byte-streamed) on-chip design end to end -- stim.hex load
into scratch, k96_vec_dot_v2's FETCH/SETTLE/MACS consumption, fp32 chaining
with per-case resets, LED pattern and the UART verdict line (with elapsed
cycles as hex). Includes a first-superblock bus trace checked against the C
consumption schedule (kernel states: IDLE=0 FETCH=1 SETTLE=2 MACS=3 SUMI=4
SUMF=5 OUT=6).

Run from fpga/sim:
  make SIM=verilator TOP=gate0v2_top MOD=tb_gate0 \
       RTL="../synth/gate0v2_top.sv ../synth/uart_tx.sv \
            ../rtl/f32_pkg.sv ../rtl/k96_vec_dot_v2.sv"
"""
import os
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

os.chdir(Path(__file__).resolve().parents[1] / "synth")  # $readmemh CWD

BIT_CYCLES = 234  # tx bit period in clock cycles (27 MHz / ~115200 baud)


@cocotb.test()
async def gate_on_chip(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_b.value = 1  # board pull-up keeps it high; drive explicitly in sim

    uarts = bytearray()

    async def uart_rx():
        prev = 1
        while True:
            await RisingEdge(dut.clk)
            cur = int(dut.txd.value) if dut.txd.value.is_resolvable else 1
            if cur == 0 and prev == 1 and len(uarts) < 24:
                for _ in range((3 * BIT_CYCLES) // 2):
                    await RisingEdge(dut.clk)
                b = 0
                for i in range(8):
                    if int(dut.txd.value):
                        b |= 1 << i
                    for _ in range(BIT_CYCLES):
                        await RisingEdge(dut.clk)
                uarts.append(b)
            prev = cur

    recs = []   # (sb, partial, exp, sumf, fail_idx, wm[0:4], am[0:4]) at done

    async def monitor():
        while True:
            await RisingEdge(dut.clk)
            try:
                if int(dut.k_done.value):
                    recs.append((
                        int(dut.sb_idx_r.value),
                        int(dut.k_partial.value),
                        int(dut.exp_r.value),
                        int(dut.sumf.value),
                        int(dut.fail_idx.value),
                        [int(dut.wm[i].value) & 0xFFFFFFFF for i in range(4)],
                        [int(dut.am[i].value) & 0xFFFFFFFF for i in range(4)],
                    ))
            except Exception:
                pass

    trace = []  # (k_st, wdata0, wdata1, adata0, sdata) while rec 0..1 run

    async def tracer():
        while True:
            await RisingEdge(dut.clk)
            try:
                if len(recs) < 2:
                    trace.append((int(dut.k_st.value),
                                  int(dut.wdata0.value), int(dut.wdata1.value),
                                  int(dut.adata0.value), int(dut.sdata.value)))
            except Exception:
                pass

    cocotb.start_soon(uart_rx())
    cocotb.start_soon(monitor())
    cocotb.start_soon(tracer())

    waited = 0
    while len(uarts) < 9 and waited < 600_000:
        await RisingEdge(dut.clk)
        waited += 1
    assert len(uarts) >= 9, f"no verdict line: {bytes(uarts)!r} after {waited} cycles"

    line = bytes(uarts)
    print("UART line:", line.split(b'\r')[0])
    print("LEDs:", f"{int(dut.led.value):06b}")

    npz = np.load("/tmp/gate0_vecs.npz")
    print(f"{'rec':>3} {'sb':>3} {'partial':>10} {'npz_rtl':>10} {'exp_r':>10} {'npz_gold':>10} {'sumf':>10}")
    n_bad = 0
    for rec_no, r in enumerate(recs[:21]):
        sb, part, exp, sumf, fidx = r[0], r[1], r[2], r[3], r[4]
        rtl = int(npz[f"rec{rec_no}_rtl"])
        gold = int(npz[f"rec{rec_no}_gold"])
        pok = (part & 0xFFFFFFFF) == (rtl & 0xFFFFFFFF)
        eok = (exp & 0xFFFFFFFF) == (gold & 0xFFFFFFFF)
        if not (pok and eok):
            n_bad += 1
        print(f"{rec_no:>3} {sb:>3} {part & 0xFFFFFFFF:010x} {rtl & 0xFFFFFFFF:010x} "
              f"{exp & 0xFFFFFFFF:010x} {gold & 0xFFFFFFFF:010x} {sumf & 0xFFFFFFFF:010x} "
              f"{'P-OK' if pok else 'P-DIFF'} {'E-OK' if eok else 'E-DIFF'}")

    # first-superblock consumption trace vs the C schedule
    st_set = sorted({e[0] for e in trace})
    print("trace: len", len(trace), "states seen", st_set)
    macc, sumic, sumfc = [], [], []
    for st_v, wd0, wd1, ad0, sd in trace:
        if st_v == 3:
            macc.append((wd0, wd1, ad0))
        elif st_v == 4:
            sumic.append(sd)
        elif st_v == 5:
            sumfc.append(sd)
    wb = bytes(npz[f"rec0_wblk"])
    ab = bytes(npz[f"rec0_act"])
    exp_mac, exp_sumi, exp_sumf = [], [], []
    for g in range(4):
        for kk in range(2):
            j = 2 * g + kk
            exp_sumi.append(int.from_bytes(ab[34 * j: 34 * j + 2], "little"))
            for bb in range(16):
                tstep = 32 * g + 16 * kk + bb
                c0 = 32 * kk + 2 * bb
                lb = wb[8 + tstep]
                hhb = wb[136 + 16 * g + 8 * kk + (bb >> 1)]
                h0 = (hhb >> (2 * (c0 & 3))) & 3
                h1 = (hhb >> (2 * ((c0 + 1) & 3))) & 3
                ay0 = ab[34 * j + 2 + 2 * bb]
                ay1 = ab[34 * j + 3 + 2 * bb]
                exp_mac.append((lb, hhb, (ay1 << 8) | ay0))
        exp_sumf.append(int.from_bytes(wb[2 * g: 2 * g + 2], "little"))
    shown = 0
    for i, got in enumerate(macc[:len(exp_mac)]):
        if got != exp_mac[i] and shown < 10:
            shown += 1
            print(f"  MAC-DIFF step {i}: got (lb {got[0]:#04x}, hh {got[1]:#04x}, ay {got[2]:#06x}) "
                  f"want (lb {exp_mac[i][0]:#04x}, hh {exp_mac[i][1]:#04x}, ay {exp_mac[i][2]:#06x})")
    for i, got in enumerate(sumic[:len(exp_sumi)]):
        if got != exp_sumi[i] and shown < 14:
            shown += 1
            print(f"  SUMI-DIFF {i}: sdata {got:#06x} want {exp_sumi[i]:#06x}")
    for i, got in enumerate(sumfc[:len(exp_sumf)]):
        if got != exp_sumf[i] and shown < 14:
            shown += 1
            print(f"  SUMF-DIFF {i}: sdata {got:#06x} want {exp_sumf[i]:#06x}")
    print(f"  trace checks: {min(len(macc), len(exp_mac))}/{len(exp_mac)} MACS, "
          f"{len(sumic)}/{len(exp_sumi)} SUMI, {len(sumfc)}/{len(exp_sumf)} SUMF")

    if line.startswith(b"G0 PASS"):
        led_v = int(dut.led.value)
        assert led_v & 0b100000 and not (led_v & 0b10000) and (led_v & 0b1111) == 0b1111, \
            f"PASS but LEDs {led_v:06b}"
    elif line.startswith(b"G0 FAIL"):
        rec = recs[int(dut.fail_idx.value)][0]
        wb2 = bytes(npz[f"rec{rec}_wblk"])
        ab2 = bytes(npz[f"rec{rec}_act"])
        wm_w, am_w = recs[int(dut.fail_idx.value)][5], recs[int(dut.fail_idx.value)][6]
        wbw = [int.from_bytes(wb2[4 * i:4 * i + 4], "little") for i in range(4)]
        abw = [int.from_bytes(ab2[4 * i:4 * i + 4], "little") for i in range(4)]
        print("wm@done got :", " ".join(f"{x:08x}" for x in wm_w))
        print("wm@done want:", " ".join(f"{x:08x}" for x in wbw))
        print("am@done got (w64-67):", " ".join(f"{int(dut.am[i].value)&0xFFFFFFFF:08x}" for i in range(64, 68)))
        print("am want (w64-67):", " ".join(f"{int.from_bytes(ab2[4*i:4*i+4],'little'):08x}" for i in range(64, 68)))
        print("am@done got :", " ".join(f"{x:08x}" for x in am_w))
        print("am@done want:", " ".join(f"{x:08x}" for x in abw))
        raise AssertionError(f"on-chip mismatch at record {rec}; table above")
    else:
        raise AssertionError(f"unexpected line {line!r}")
    assert n_bad == 0, f"monitor found {n_bad} bad records despite PASS line"