"""Full-top silicon-gate simulation: gate0_top (fpga/synth) in Verilator.

Runs the REAL on-chip design -- stim.hex/exp.hex BRAM load, descending
shift-in, k96_vec_dot kernel, fp32 chaining, LED pattern and the UART verdict
line -- to prove the bitstream logic before/regardless of board observation.
UART decode: sample txd at the tx bit rate from start-bit edges.

Run from fpga/sim:
  make SIM=verilator TOP=gate0_top MOD=tb_gate0 \
       RTL="../synth/gate0_top.sv ../synth/uart_tx.sv ../rtl/f32_pkg.sv ../rtl/k96_vec_dot.sv"
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
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())  # counts are all that matter

    uarts = bytearray()

    async def uart_rx():
        prev = 1
        seen_edges = 0
        while True:
            await RisingEdge(dut.clk)
            resolvable = bool(dut.txd.value.is_resolvable)
            cur = int(dut.txd.value) if resolvable else 1
            if cur == 0 and prev == 1:
                seen_edges += 1
                # skip 1.5 bit times, then sample 8 data bits + skip stop
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

    cocotb.start_soon(uart_rx())

    dut.rst_b.value = 1  # board pull-up keeps it high; drive explicitly in sim

    # per-comparison recorder: samples (sb_idx, partial, exp_r, sumf_before)
    recs = []
    async def monitor():
        while True:
            await RisingEdge(dut.clk)
            try:
                if int(dut.k_done.value):
                    recs.append((
                        int(dut.sb_idx.value),
                        int(dut.k_partial.value),
                        int(dut.exp_r.value),
                        int(dut.sumf.value),
                        int(dut.fail_idx.value),
                        int(dut.wblk_sh.value),
                        int(dut.act_sh.value),
                    ))
            except Exception:
                pass

    cocotb.start_soon(monitor())

    # progress probes: S state + LEDs every 20K cycles
    for k in range(20):
        for _ in range(20_000):
            await RisingEdge(dut.clk)
        try:
            st = int(dut.st.value)
        except Exception:
            st = -1
        try:
            led = f"{int(dut.led.value):06b}"
        except Exception:
            led = "?"
        try:
            sb = int(dut.sb_idx.value)
        except Exception:
            sb = -1
        print(f"  t={20_000*(k+1):>6} st={st} sb_idx={sb} led={led} uart={bytes(uarts)!r}")
        if st == 5 or len(uarts) >= 13:
            break

    waited = 0
    while len(uarts) < 13 and waited < 400_000:
        await RisingEdge(dut.clk)
        waited += 1
    assert len(uarts) >= 9, f"no verdict line: {bytes(uarts)!r} after {waited} cycles"

    line = bytes(uarts)
    leds = f"{int(dut.led.value):06b}"
    print("UART line:", line)
    print("LEDs:", leds)

    # cross-check per-record sampling vs the kernel-level npz artifacts
    npz = np.load("/tmp/gate0_vecs.npz")
    print(f"{'rec':>3} {'sb':>3} {'partial':>10} {'npz_rtl':>10} {'exp_r':>10} {'npz_gold':>10} {'sumf':>10}")
    for rec_no, r in enumerate(recs):
        sb, part, exp, sumf, fidx = r[0], r[1], r[2], r[3], r[4]
        wb, ab = r[5], r[6]
        rtl = int(npz[f"rec{rec_no}_rtl"])
        gold = int(npz[f"rec{rec_no}_gold"])
        print(f"{rec_no:>3} {sb:>3} {part & 0xFFFFFFFF:010x} {rtl & 0xFFFFFFFF:010x} "
              f"{exp & 0xFFFFFFFF:010x} {gold & 0xFFFFFFFF:010x} {sumf & 0xFFFFFFFF:010x} "
              f"{'P-OK' if (part & 0xFFFFFFFF) == (rtl & 0xFFFFFFFF) else 'P-DIFF'} "
              f"{'E-OK' if (exp & 0xFFFFFFFF) == (gold & 0xFFFFFFFF) else 'E-DIFF'}")
        # byte-level data diff: which of the 472 stimulus bytes got scrambled?
        want_w = bytes(npz[f"rec{rec_no}_wblk"])
        want_a = bytes(npz[f"rec{rec_no}_act"])
        got_w = (wb & ((1 << 1600) - 1)).to_bytes(200, "little")
        got_a = (ab & ((1 << 2176) - 1)).to_bytes(272, "little")
        dw = [i for i in range(200) if got_w[i] != want_w[i]]
        da = [i for i in range(272) if got_a[i] != want_a[i]]
        if dw or da:
            print(f"    DATA-DIFF rec{rec_no}: wblk bad bytes {dw[:12]}{'...' if len(dw) > 12 else ''}"
                  f" ({len(dw)}/200), act bad {da[:12]}{'...' if len(da) > 12 else ''} ({len(da)}/272)")
        else:
            print(f"    DATA-OK rec{rec_no}: wblk+act bytes identical to npz stimulus")

    if line.startswith(b"G0 PASS"):
        led_v = int(dut.led.value)
        assert led_v & 0b100000 and not (led_v & 0b10000) and (led_v & 0b1111) == 0b1111, \
            f"PASS but LEDs {led_v:06b}"
    elif line.startswith(b"G0 FAIL"):
        assert int(dut.led.value) & 0b10000, "FAIL but led[4] off"
        print(f"FAIL at record {line[9:11].decode()}")
        raise AssertionError("on-chip gate mismatch: compare table above")
    else:
        raise AssertionError(f"unexpected line {line!r}")