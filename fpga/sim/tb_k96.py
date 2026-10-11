"""Gate-0: K9_6 superblock vec_dot RTL vs the fork's C kernel — bit-exact."""
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import (b32f, f32b, k96_partials, gen_superblocks, gen_act_blocks,
                    act_flat_int)
import ninths_native as nn  # noqa: F401  (pack used below)


def make_clock(dut):
    for kwargs in ({"units": "ns"}, {"unit": "ns"}):
        try:
            return Clock(dut.clk, 2, **kwargs)
        except TypeError:
            continue
    raise TypeError("no Clock signature worked")


async def run_sb(dut, wbytes, acts) -> int:
    dut.wblk.value = int.from_bytes(wbytes, "little")
    dut.act.value = act_flat_int(acts)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
    return int(dut.partial_bits.value)


@cocotb.test()
async def gate(dut):
    cocotb.start_soon(make_clock(dut).start())
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = np.random.default_rng(1)
    cases = []
    for k in (17, 27, 63):
        cases.append((f"rand-normal-k{k}", gen_superblocks(rng, 1, k=k),
                      gen_act_blocks(rng, 8)))
    cases.append(("chain3", gen_superblocks(rng, 3), gen_act_blocks(rng, 24)))

    # scale range incl. fp16 zero / subnormal / max / +inf weight scales
    d62 = np.full((1, 256), 62, dtype=np.uint8)
    m_ext = np.array([[0.0, 2e-6, 2.03e6, 1e7]], dtype=np.float32)
    cases.append(("ext-scales", [bytes(nn.pack(d62, 63, m_ext))],
                  gen_act_blocks(rng, 8, codes=1)))

    # max-magnitude codes, all-negative activation products
    dalt = np.tile(np.array([0, 62], dtype=np.uint8), 128).reshape(1, 256)
    cases.append(("ext-codes", [bytes(nn.pack(dalt, 63, np.full((1, 4), 1.0)))],
                  gen_act_blocks(rng, 8, codes=-128)))

    # negative weight scales (sign paths in fp ops)
    d3 = rng.integers(0, 63, (1, 256)).astype(np.uint8)
    m_neg = np.array([[-1.5, -0.25, 3.0, -7.25]], dtype=np.float32)
    cases.append(("neg-scales", [bytes(nn.pack(d3, 63, m_neg))],
                  gen_act_blocks(rng, 8, d16=-1.5)))

    for c in range(12):
        cases.append((f"fuzz{c}", gen_superblocks(rng, 1), gen_act_blocks(rng, 8)))

    n_fail = 0
    recs = []
    for cidx, (name, wblks, acts) in enumerate(cases):
        row_g, row_final = k96_partials(wblks, acts)   # row-kernel tree (INFO: the
        # C row kernel's own accumulation; crosscheck validates python==C on it)
        acc_rtl = b32f(0)
        acc_np = np.float32(0.0)
        line = []
        for i, wb in enumerate(wblks):
            acts_i = acts[8 * i : 8 * i + 8]
            got = await run_sb(dut, wb, acts_i)
            sgl, _ = k96_partials([wb], acts_i)
            # per-CALL gate: the hardware's single-superblock result vs the
            # numpy replica of the fork's per-call semantics (deterministic --
            # the old chained-vs-row-tree compare held only by luck for
            # multi-superblock chains: 18/21 synth records were singletons).
            sgl_ok = got == sgl[0]
            acc_np = np.float32(acc_np + b32f(sgl[0]))      # gate chain tree
            acc_rtl = np.float32(acc_rtl + b32f(got))
            want = f32b(acc_np)
            recs.append({"case": cidx,
                         "wblk": np.frombuffer(wb, np.uint8),
                         "act": np.frombuffer(b"".join(acts_i), np.uint8),
                         "rtl": got, "gold": want})
            if not sgl_ok or f32b(acc_rtl) != want:
                line.append(f"FAIL({got:#010x}!={sgl[0]:#010x})")
                n_fail += 1
            else:
                line.append("OK")
        print(f"  {name:16s} {' '.join(line)} (row-final {row_final:#010x} "
              f"informational) chain-final={f32b(acc_rtl):#010x}")
    np.savez("/tmp/gate0_vecs.npz",
             **{f"rec{i}_{k}": v for i, r in enumerate(recs) for k, v in r.items()})
    print("GATE-0 K9_6 vec_dot: "
          + ("PASS" if n_fail == 0 else f"FAIL ({n_fail} mismatches)"))
    assert n_fail == 0