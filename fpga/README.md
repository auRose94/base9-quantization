# ninths on FPGA (Tang Nano 20K, GW2AR-18)

The FPGA seam: **a third substrate for the stage-2 resident digit containers**
(docs/08). The device decodes entropy-coded / packed ninths superblocks and
does the MAC — at batch-1 the decode bandwidth *is* the token rate, so a
cheap streaming decode pipeline is a real serving path.

## Layout

- `rtl/f32_pkg.sv` — IEEE-754 binary32 helpers (fp16→fp32 exact incl.
  subnormal renorm, int→f32, mul/add RN-even, separate ops — the contract is
  the C kernel compiled without FMA contraction). Bit-exactness tested by
  6,188-case fuzz in `sim/tb_f32probe.py`.
- `rtl/k96_vec_dot.sv` — V1 correctness-first FSM replicating
  `ggml_vec_dot_k9_6_q8_0_generic` op-for-op (one 200 B superblock + 8 Q8_0
  blocks in, fp32 partial out; caller chains partials in order). V1 uses
  wide parallel ports; see "V2" below.
- `sim/` — cocotb + Verilator benches. `tb_k96.py` is Gate-0 (19 cases,
  bit-exact incl. fp16 inf/zero/subnormal scales, inf products, chained
  superblocks); stimulus saved to `/tmp/gate0_vecs.npz` (deterministic,
  rng seed 1) and re-used by the cross-check and the silicon top. Run:
  `make SIM=verilator TOP=k96_vec_dot MOD=tb_k96 RTL="../rtl/f32_pkg.sv ../rtl/k96_vec_dot.sv"`
- `crosscheck/` — RTL↔C closure. `k9_c_harness.c` carries the fork's
  kernels verbatim (built `-O2 -ffp-contract=off`); `run_crosscheck.py`
  feeds it `gate0_vecs.npz` and requires bit-identical cumulative fp32
  states vs golden. **Status: PASS (all 21 records).**
- `synth/` — silicon gate for the board (see below). `build.sh` = gen memory
  → yosys `synth_gowin` → `nextpnr-himbaechel` → `gowin_pack`.
  Tools: oss-cad-suite (yosys 0.69+/nextpnr 0.11.1/apycula) in
  `~/Work/tools/oss-cad-suite`.

## Status (2026-10-10, session 28)

- [x] Gate-0 cocotb: RTL bit-exact vs numpy replica of the C kernel.
- [x] RTL↔C: verbatim fork kernels bit-exact vs golden on the same stimulus.
- [x] Synth-safe RTL: package calls qualified (`f32_pkg::`), the f32_add
      renormalization while-loop rewritten as a bounded MSB scan (bit-identical,
      gate re-verified).
- [x] `gate0_top.sv` (silicon gate) **passes full simulation** (`sim/tb_gate0.py`):
      "G0 PASS" + LED pattern + UART protocol all verified.
- [x] Board bring-up: JTAG load via Sipeed USB Debugger pod (FT2232) works;
      bits stream to the device (load verified 100%).
- [ ] **PnR**: full gate top fails legal placement at 14914 LUT4-equiv (71%).
      V1's parallel-port byte muxes are the whale. Next: V2 streaming kernel
      (byte-serial wblk/act consumption — all indices are sequential:
      lb = wblk byte 8+t, hh byte per two t, act blocks contiguous), which
      removes both the big muxes and the 1600/2176-bit shift registers.
      k96 datapath alone PnR'd fine (Fmax 30.9 MHz @ 27 MHz target).
- [ ] UART-visible verdict on the board (needs the board's own USB-C plugged
      in for the BL616 ttyACM, or 69/70 wired to the external pod's CH-B).

## Board notes (Tang Nano 20K)

- Pins (per apicula examples CST, authoritative): clk = pin 4 @ 27 MHz,
  LEDs = 15,16,17,18,19,20, BL616 USB-C UART TXD/RXD = 69/70, reset btn = 88.
- The device is `GW2AR-LV18QN88C8/I7` (nextpnr) with `--vopt family=GW2A-18C`;
  `gowin_pack -d GW2A-18C` packs the routed JSON (no `--textcfg` on this
  nextpnr — post-route JSON goes straight into the packer).
- The pod connected here is the Sipeed USB Debugger (FT2232) on the EXPAND
  JTAG pins; the board's BL616 USB-C pod is unplugged (no ttyACM). The
  0489:e10a lsusb entry is the host's Bluetooth module — not the pod.
- LED semantics of `gate0_top`: running = heartbeat walk on 6 LEDs; verdict
  line repeats every ~1 s ("G0 PASS" / "G0 FAIL nn"): PASS = 5 LEDs on
  (led[4] stays off), FAIL = led[4] on + led[3:0] = first mismatching record.
  `beacon_top.sv` (walking LED + repeated "HELLO n=NNNN") is the wiring
  probe; it is what's currently loaded.
- Host-side UART without ttyUSB group perms: pyusb + FTDI CH-B (baud divisor
  25 → 3 MHz/26 = 115384 baud, identical to the FPGA's 27e6/234).

## Gotchas ledger (add on every new catch)

- Package imports: yosys accepts `import f32_pkg::*` at file scope but NOT in
  the module header or body; qualified `f32_pkg::fn()` works everywhere and
  is the portable form.
- Data-dependent `while` loops are non-synthesizable in yosys (Verilator is
  fine); express priority scans as bounded for-loops.
- $readmemh into BRAM-inferred memories: yosys + Verilator both honor it;
  Verilator resolves paths relative to the *runtime* CWD (tb chdirs).
- cocotb access to internal signals on Verilator: works via flat names
  (`dut.k_done`, `dut.wblk_sh`), incl. submodule ports flattened.
- Verilator lint is errors-by-default in the cocotb flow: size-cast
  (`4'(fi / 4'd10)`) and explicit widths (`16'(DIV) - 16'd1`) keep it quiet.
- Shift-in direction: descending word reads pair with LEFT-shift-in
  (`{reg[w-33:0], word}`) to get byte 0 at the LSB. (Caught by tb_gate0's
  byte-diff instrumentation — that harness is why silico-validation is
  mandatory before every bitstream.)
- Golden per-case chaining: golden `cums[]` reset at case boundaries; any
  streaming comparator needs a case-start flag alongside the expected value.