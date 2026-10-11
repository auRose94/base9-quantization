"""Netlist-mode probe of gate0sd_top (the post-synthesized net).

Discovers the flattened internal net names at runtime, watches the reader/top
FSM states, and EXITS EARLY with a net-name dump the moment the design
freezes (no state change for a long window) or reaches a terminal state.
This is the RTL-vs-netlist divergence detector: the RTL passes the same
scenario, the netlist was observed to never start its card path.

Run from fpga/sim:
  make SIM=verilator TOP=gate0sd_top MOD=tb_net \
       RTL="../sim/gowin_sim_stubs.sv /tmp/gate0sd_net.v \
            /home/rose/Work/tools/oss-cad-suite/share/yosys/gowin/cells_sim.v"
"""
import os
from collections import deque
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

os.chdir(Path(__file__).resolve().parents[1] / "partner")

BIT_CYCLES = 232


@cocotb.test()
async def netlist_probe(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_b.value = 1
    dut.prov_btn.value = 1            # no provisioning: verify+read path only

    # discover flattened names for the interesting internals
    names = [n for n in dir(dut) if not n.startswith("_")]
    print("top-level names:", [n for n in names if any(
        k in n for k in ("st", "sd_", "rd_", "pay", "fail", "card", "kernel"))][:50])

    async def watch(name_prefix):
        hist = deque(maxlen=6000)
        return hist

    # The reader-state net: find objects whose flat name suggests the reader's
    # state register. Verilator flattens as e.g. sd_spi_reader..st_DFFRE_Q*
    cand = [n for n in names if ("st_DFF" in n or n.endswith("st"))]
    print("state-register candidates:", cand[:20])
    for c in cand[:4]:
        try:
            print(f"  {c} = {int(getattr(dut, c).value)}")
        except Exception:
            pass

    # keep only names that actually read
    good = []
    for c in cand[:8]:
        try:
            int(getattr(dut, c).value)
            good.append(c)
        except Exception:
            continue
    print("readable state nets:", good)

    # watch the top's `st` register for a freeze / evolution
    snap = None
    frozen = 0
    for cyc in range(6_000_000):
        await RisingEdge(dut.clk)
        try:
            stv = int(dut.st.value) if hasattr(dut, "st") else int(getattr(dut, "st_DFFRE_Q").value)
        except Exception:
            stv = -1
        if stv == snap:
            frozen += 1
            if frozen >= 1_500_000:
                print(f"FROZEN: st stayed {stv} for {frozen} cycles")
                for c in good:
                    try:
                        print(f"    {c} = {int(getattr(dut, c).value) if hasattr(getattr(dut, c), 'value') else '?'}")
                    except Exception:
                        print(f"    {c} = ??")
                raise AssertionError("netlist frozen; see values")
        else:
            frozen = 0
            snap = stv
        if cyc % 100000 == 0:
            try:
                print(f"  t={cyc} st={int(dut.st.value) if hasattr(dut, 'st') else stv}")
            except Exception:
                pass
    print("netlist survived the window without freezing (or kept changing)")