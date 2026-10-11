#!/usr/bin/env python3
"""Partner-program side of the ninths streamed-model work: builds the microSD
image the FPGA reads and writes it to a real card.

Why a host partner at all: the FPGA datapath consumes bytes in a fixed order
with zero file-system knowledge. Organizing storage, checksums and provenance
is host work; the card then makes the board DEMO STANDALONE (no PC during
inference), while the PC keeps authoring/verifying containers. The same split
scales later: this tool grows into the container-packer for streamed models
(tensor index + payload LBAs), and the board-side reader grows into 4-bit SD
mode when the SPI rate becomes the bottleneck.

Image layout (the FPGA hard-codes it, see gate0sd_top.sv):
  partition 1 starts at LBA 2048 (the FPGA's SD_BASE_LBA anchor);
  relative LBAs 0..19 (20 x 512 B):
    bytes 0..9911   = stim: 2478 x u32 LE (the Gate-0 word stream)
    bytes 9912..9995 = exp: 21 x u32 LE (golden cumulative fp32 bits)
    bytes 9996..10079 = flags: 21 x u32 LE (0/1 case-chain reset)
    rest: 0x00 padding.

Commands:
  mkimage  [npz]  -> ninths_sd.bin (payload, 10 KiB)
  flash-card <dev> -> partition the card (p1: 256 MiB payload, p2: rest FAT32),
                      write the payload, mkfs p2. Requires root + a card
                      reader. SAFETY: refuses anything that looks like it is
                      not a removable/SD media; double-check the device node.
  verify <dev>     -> read back and check the payload bytes.
"""
import struct
import subprocess
import sys
from pathlib import Path

LBA = 512
BASE_LBA = 2048
PART1_SECTORS = 262144  # 256 MiB
NPZ = Path("/tmp/gate0_vecs.npz")
HERE = Path(__file__).resolve().parent


def load_npz_words():
    import numpy as np
    d = np.load(NPZ)
    n = 1 + max(int(k.split("_")[0][3:]) for k in d.files if k.startswith("rec"))
    assert n == 21, n
    stim, exp, flags = [], [], []
    prev_case = -1
    for i in range(n):
        blob = bytes(d[f"rec{i}_wblk"].tobytes()) + bytes(d[f"rec{i}_act"].tobytes())
        assert len(blob) == 472
        stim += [int.from_bytes(blob[4 * j: 4 * j + 4], "little") for j in range(118)]
        exp.append(int(d[f"rec{i}_gold"]))
        cid = int(d[f"rec{i}_case"])
        flags.append(1 if i == 0 or cid != prev_case else 0)   # case-chain start
        prev_case = cid
    return stim, exp, flags


def mkimage(out: Path):
    stim, exp, flags = load_npz_words()
    assert len(stim) == 2478
    buf = bytearray()
    for w in stim:
        buf += struct.pack("<I", w)
    for w in exp:
        buf += struct.pack("<I", w & 0xFFFFFFFF)
    for f in flags:
        buf += struct.pack("<I", int(f))
    assert len(buf) == 10080, len(buf)
    payload = bytes(buf) + bytes(20 * LBA - len(buf))
    out.write_bytes(payload)
    print(f"{out}: {len(payload)} bytes ({len(payload)//LBA} LBAs)")


def flash_card(dev: str, payload: Path):
    if not dev.startswith("/dev/sd") or Path(dev).is_dir():
        raise SystemExit(f"refusing non-sd-like device {dev!r}")
    p = subprocess.run(["lsblk", "-o", "NAME,SIZE,RM,TYPE", "-n", "-b"],
                       capture_output=True, text=True)
    rows = [r.split() for r in p.stdout.strip().splitlines()]
    me = [r for r in rows if r[0] == Path(dev).name]
    if not me or len(me[0]) < 3 or me[0][2] != "1":
        print("device details:", rows)
        raise SystemExit(f"{dev} is not flagged removable (rm=1) -- refusing")
    subprocess.run(["sudo", "bash", "-c", f"""
set -e
sfdisk --yes {dev} <<'TBL'
label: dos
start=2048, size={PART1_SECTORS}, type=81
start=$((2048 + {PART1_SECTORS})), size=0, type=0C
TBL
dd if={payload} of={dev} bs=512 seek={BASE_LBA} oflag=direct conv=fsync status=none
partprobe {dev} || true
partdev={dev}1
mkfs.vfat -F 32 -n NINTHS {partdev}
"""], check=True)
    print("card partitioned; payload written at LBA", BASE_LBA)


def verify(dev: str, payload: Path):
    want = payload.read_bytes()
    got = subprocess.run(["sudo", "dd", f"if={dev}", "bs=512", f"skip={BASE_LBA}",
                          f"count={len(want) // LBA}", "status=none"],
                         capture_output=True, check=True).stdout
    ok = got == want
    print("verify:", "OK" if ok else "MISMATCH")
    return ok


def main():
    argv = sys.argv[1:]
    out = HERE / "ninths_sd.bin"
    if argv and argv[0] == "mkimage":
        return mkimage(out)
    if argv and argv[0] == "flash-card":
        mkimage(out)
        return flash_card(argv[1], out)
    if argv and argv[0] == "verify":
        return verify(argv[1], out)
    print(__doc__)


if __name__ == "__main__":
    main()