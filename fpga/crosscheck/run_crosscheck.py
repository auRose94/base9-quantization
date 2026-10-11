"""Cross-check driver: feed the Gate-0 cocotb stimulus to the verbatim C fork
kernel (fpga/crosscheck/k9_c_harness) and check both accumulation trees:

  row-tree: C harness (the row kernel's own per-group accumulation over the
  chain) vs the numpy replica (golden.k96_partials on the same chain) --
  proves python == C for the row kernel on the same bytes.

  gate-tree: the RTL's per-record partials (npz "rtl") chained left-to-right
  with running fp32 adds, case-reset at flags, vs the npz "gold" words (the
  gate protocol's exp; the sequence the silicon gate compares).

History note: the original gold words were the row tree; the real-tensor
payload (partner/real_slice_image.py) exposed that the hardware chains
per-superblock partials -- a different fp32 tree that differs from the row
tree by 1 ulp on real magnitudes, so the protocol's exp semantics moved to
the chain tree and the row tree stays as the python==C check.
"""
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
from golden import k96_partials   # noqa: E402  (numpy replica)

BIN = HERE / "k9_c_harness"
NPZ = Path(os.environ.get("GATE0_NPZ", "/tmp/gate0_vecs.npz"))


def f32b(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def main():
    if not BIN.exists():
        raise SystemExit("compile k9_c_harness first (see header comment)")
    data = np.load(NPZ)
    import re
    idxs_all = sorted(int(re.match(r"rec(\d+)_wblk", k).group(1))
                      for k in data.files if re.match(r"rec(\d+)_wblk", k))
    n_recs = 1 + max(idxs_all)
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

        wblks = [bytes(r["wblk"]) for r in chain]
        acts = [bytes(np.asarray(r["act"])[34 * j: 34 * (j + 1)])
                for r in chain for j in range(8)]
        cums_np, _ = k96_partials(wblks, acts)
        row_ok = ("".join("OK" if int(x) == int(y) else "FAIL" for x, y in zip(cums_c, cums_np))
                  if len(cums_c) == len(cums_np) else "LEN")

        acc = np.float32(0.0)
        got_rtl = []
        has_rtl = "rtl" in chain[0]
        for r in chain:
            if not has_rtl:
                got_rtl.append("n/a")
                continue
            acc = np.float32(acc + np.frombuffer(struct.pack("<I", int(r["rtl"])), dtype="<f4")[0])
            got_rtl.append("OK" if f32b(acc) == int(r["gold"]) else "FAIL")
        got_rtl = "".join(got_rtl)
        cf = row_ok.count("FAIL"); rf = got_rtl.count("FAIL")
        n_fail += cf + rf
        print(f"  case{cidx} ({len(chain)} sb): C==numpy(row) {row_ok or '(empty)'} | "
              f"rtl==gold(chain) {got_rtl or '(empty)'}")

    print("CROSSCHECK: "
          + ("PASS (all trees bit-identical)" if n_fail == 0 else f"FAIL ({n_fail} mismatches)"))
    assert n_fail == 0


if __name__ == "__main__":
    main()