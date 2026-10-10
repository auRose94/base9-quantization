"""Cross-check driver: feed the Gate-0 cocotb stimulus to the verbatim C fork
kernel (fpga/crosscheck/k9_c_harness) and require bit-identical cumulative
sumf states vs the golden cums[] the RTL was gated against.

Closes RTL<->C: cocotb already proved RTL == numpy-replica; this proves
numpy-replica == real C (compiled -ffp-contract=off), so RTL == C transitively
over exactly the same stimulus.
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BIN = HERE / "k9_c_harness"
NPZ = Path("/tmp/gate0_vecs.npz")


def f32b(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def main():
    if not BIN.exists():
        raise SystemExit("compile k9_c_harness first (see header comment)")
    data = np.load(NPZ)
    n_recs = 1 + max(int(k.split("_")[0][3:]) for k in data.files if k.startswith("rec"))
    recs = [{f.split("_", 1)[1]: data[f] for f in data.files if f.startswith(f"rec{i}_")}
            for i in range(n_recs)]

    n_fail = 0
    for cidx in sorted({int(r["case"]) for r in recs}):
        chain = [r for r in recs if int(r["case"]) == cidx]
        inb = (6).to_bytes(4, "little") + len(chain).to_bytes(4, "little")
        inb += b"".join(bytes(r["wblk"]) for r in chain)
        inb += b"".join(bytes(r["act"]) for r in chain)
        fin, fout = HERE / f"_in{cidx}.bin", HERE / f"_out{cidx}.bin"
        fin.write_bytes(inb)
        subprocess.run([str(BIN), str(fin), str(fout)], check=True)
        cums_c = np.frombuffer(fout.read_bytes(), dtype="<u4")
        fin.unlink(); fout.unlink()

        got_gold = "".join("OK" if int(x) == int(r["gold"]) else "FAIL" for x, r in zip(cums_c, chain))
        # RTL recs carry per-superblock partials; chain them with fp32 adds like
        # the C kernel (and the cocotb bench) do before comparing to cums[].
        acc = np.float32(0.0)
        got_rtl = []
        for r in chain:
            acc = np.float32(acc + np.frombuffer(struct.pack("<I", int(r["rtl"])), dtype="<f4")[0])
            got_rtl.append("OK" if f32b(acc) == int(r["gold"]) else "FAIL")
        got_rtl = "".join(got_rtl)
        cf = got_gold.count("FAIL"); rf = got_rtl.count("FAIL")
        n_fail += cf + rf
        print(f"  case{cidx} ({len(chain)} sb): C==gold {got_gold or '(empty)'} | "
              f"rtl==gold {got_rtl or '(empty)'}")

    print("CROSSCHECK C-kernel vs numpy-gold vs RTL: "
          + ("PASS (all bits identical)" if n_fail == 0 else f"FAIL ({n_fail} mismatches)"))
    assert n_fail == 0


if __name__ == "__main__":
    main()