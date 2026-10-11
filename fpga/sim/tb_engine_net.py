"""Engine-probe bench: works on BOTH the RTL and the synthesized netlist.
The probe's LED contract: pads ~lit_mask; finished = 0b010000 (bit5 lit +
index dark); otherwise lit_mask[3:0] = the exchange count (0..7) on the
index LEDs. Assert: the count reaches finished.
"""
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

os.chdir(Path(__file__).resolve().parents[1] / "synth")


@cocotb.test()
async def engine_probe(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    hist = []
    for cyc in range(100_000):
        await RisingEdge(dut.clk)
        if cyc % 2000 == 0:
            led = int(dut.led.value) & 0b111111
            cnt = (int(led) ^ 0b111111) & 0b1111
            fin = led == 0b010000
            hist.append((cyc, cnt, fin))
    print("prog:", hist[0], "...", hist[3], hist[-1])
    assert hist[-1][2], f"engine never finished; last {hist[-1]}"
