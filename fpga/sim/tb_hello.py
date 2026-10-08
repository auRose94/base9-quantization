"""Toolchain smoke: cocotb + Verilator flow works end to end."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


def make_clock(dut):
    for kwargs in ({"units": "ns"}, {"unit": "ns"}):
        try:
            return Clock(dut.clk, 2, **kwargs)
        except TypeError:
            continue
    raise TypeError("no Clock signature worked")


@cocotb.test()
async def smoke(dut):
    cocotb.start_soon(make_clock(dut).start())
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    c0 = int(dut.count.value)
    for _ in range(8):
        await RisingEdge(dut.clk)
    c1 = int(dut.count.value)
    # edge sampling vs NBA timing can swallow one increment at the boundary
    assert c1 - c0 >= 6, f"counter advanced {c1 - c0}, expected ~8"
    dut._log.info("hello smoke PASS")