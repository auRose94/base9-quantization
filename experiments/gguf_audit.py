#!/usr/bin/env python3
"""Byte-level audit of a GGUF: declared tensor extents vs file reality."""
import struct, os, sys

QT = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
      8: (32, 34), 9: (32, 38), 10: (256, 84), 11: (256, 110), 12: (256, 144),
      13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 164),
      17: (256, 164), 18: (256, 110), 19: (256, 50), 20: (32, 18),
      21: (256, 110), 22: (256, 166), 23: (32, 22), 24: (1, 1), 25: (1, 1),
      26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (1, 2), 30: (1, 4), 43: (1, 1)}


def audit(path):
    f = open(path, "rb")
    def rs():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF"
    ver, ntens, nkv = struct.unpack("<Iqq", f.read(20))
    simple = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
              5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
              12: ("d", 8)}

    def rd_val(ty):
        if ty == 8:
            return rs()
        if ty == 9:
            et, cnt = struct.unpack("<IQ", f.read(12))
            if et in simple:
                f.seek(cnt * simple[et][1], 1)
                return None
            if et == 8:
                for _ in range(cnt):
                    rs()
                return None
            if et == 9:
                et2, cnt2 = struct.unpack("<IQ", f.read(12))
                if et2 in simple:
                    f.seek(cnt2 * simple[et2][1], 1)
                else:
                    for _ in range(cnt2):
                        rs()
                return None
            raise RuntimeError("deep array")
        fm, sz = simple[ty]
        f.read(sz)
        return None

    for _ in range(nkv):
        rs()
        rd_val(struct.unpack("<I", f.read(4))[0])
    hdr_end = f.tell()
    infos = []
    for _ in range(ntens):
        nm = rs()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        ty = struct.unpack("<I", f.read(4))[0]
        offs = struct.unpack("<Q", f.read(8))[0]
        infos.append((nm, dims, ty, offs))
    fsz = os.path.getsize(path)
    pad = (32 - hdr_end % 32) % 32
    data_start = hdr_end + pad
    print(f"{os.path.basename(path)}: v{ver} tensors={ntens} kv={nkv} "
          f"header={hdr_end} data_start={data_start} filesize={fsz}")
    prev_rel = 0
    problems = 0
    for nm, dims, ty, offs in infos:
        bt = QT.get(ty)
        if bt is None:
            print(f"  ?? unhandled type {ty}: {nm}")
            problems += 1
            continue
        bs, tsz = bt
        if dims[0] % bs != 0:
            print(f"  !! {nm}: ne0 {dims[0]} not divisible by block {bs}")
            problems += 1
            continue
        n = (dims[0] // bs) * tsz
        for d in dims[1:]:
            n *= d
        if offs != prev_rel:
            print(f"  !! {nm}: declared offset {offs} != chain {prev_rel} (Δ{offs-prev_rel})")
            problems += 1
        end = offs + n
        if end + (((32 - (end % 32)) % 32)) > fsz - data_start:
            short = end - (fsz - data_start)
            print(f"  !! {nm}: data needs {end} rel, file has {fsz - data_start} "
                  f"(short {max(0, short)})")
            problems += 1
        prev_rel = end + ((32 - (end % 32)) % 32)
    print("audit:", "CLEAN" if problems == 0 else f"{problems} problems")


if __name__ == "__main__":
    audit(sys.argv[1])