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
- [x] `gate0_top.sv` (V1 silicon gate) passes full simulation, incl. the
      three pre-silicon catches (shift direction, per-case chain reset,
      done_flag) — kept as the sim-shape reference for tb_k96.
- [x] **V2 streaming kernel `rtl/k96_vec_dot_v2.sv`** (session 28 cont.):
      byte-streamed consumption from scratch memories (FETCH/SETTLE/MACS
      3-cycle byte steps; scale service port with pre-registered d0/d1; same
      fp ops/op order as V1). Validated by `sim/tb_gate0.py` end to end:
      **G0 PASS, 21/21 records, plus a full 128-step bus trace byte-identical
      to the C consumption schedule** (`wm`/`am` scratch + per-cycle wdata/
      adata/sdata). `gate0v2_top.sv` = BRAM load -> scratch -> kernel ->
      chain/compare -> LEDs/UART (verdict repeats every ~5.5 ms with elapsed
      cycles in hex).
- [x] **PnR FITS**: V2 = 6618 LUT4-eq (31% vs V1's failing 14914/71%), 11
      BSRAM, post-route Fmax 28.33 MHz at the 27 MHz target. Packed and
      **flashed to the board** via the Sipeed FT2232 pod — the gate design
      runs on silicon; LEDs carry the verdict.
- [x] Board bring-up: JTAG load via Sipeed USB Debugger pod (FT2232) works;
      bits stream to the device (load verified 100%).
- [ ] UART-visible verdict on the board (needs the board's own USB-C plugged
      in for the BL616 ttyACM, or 69/70 wired to the external pod's CH-B;
      an EXPAND "tx fan" shotgun was tried and abandoned — several header
      pins are vendor-reserved, and the wiring position is unknown anyway).
- [x] **microSD streaming backend (session 29)**: the Tang's slot is wired
      to the FPGA (SD_CS=81, CMD=82, CLK=83, DAT0=84 — LiteX sipeed map).
      `rtl/spi_byte_master.sv` + `rtl/sd_spi_reader.sv` implement SDHC/SDXC
      SPI-mode init (375 kHz init window → 6.75 MHz SCK data, CMD0/8/41-hcs/58,
      CMD17 blocks-as-words) and `synth/gate0sd_top.sv` boots the card and
      fills stim/exp/flags from it — **validated end-to-end in cocotb with a
      modeled card (G0 PASS, fill byte-exact)**. PnR: 11441 LUT4 (55%), 32
      BSRAM + 512 RAM16, closes 23.1 MHz on a 13.5 MHz fabric clock (27 MHz
      post-route was 17.4 MHz → divided clock through BUFG). **Flashed.**
      Card absent = walking blink + "G0 SERR"; card with payload = verdict.
- [x] **Nonvolatile config + LED polarity (session 29 cont.)**: SRAM JTAG
      loads are volatile — a power-cycle wiped the design (the "dark board").
      `openFPGALoader --board tangnano20k -f gate0sd.fs` writes the bitstream
      into the board's SPI config flash, so every power-up boots it. Board
      LEDs are ACTIVE-LOW (LiteX names them led_n): pads inverted; visuals —
      card-error = a walking light plus the four index LEDs showing a binary
      error code (1 = card init failed, 2 = a write failed, 3 = the write/
      readback roundtrip mismatched, 4 = watchdog); boot/provision/fill =
      led[5]+led[4] lit and the index LEDs show WHERE the card path is
      (reader FSM state nibble during init, block index during provision/
      fill — a frozen boot reads off exactly which state stuck); **G0 PASS =
      led[5] lit + all four index LEDs lit (five lit, led[4] dark); G0 FAIL
      = led[5] blinking (fail_idx+2) with the index LEDs holding the failing
      record** — countable with no tools. SDXC
      write-busy budget widened to ~1 s/phase (WR_BUDGET) — a real card can
      sit busy far past a 333 ms NAC budget. Also fix: `openFPGALoader`
      writes flash via `-f`, there is no `--flash` flag.
- [x] **Host partner program** (`partner/ninths_sd_image.py`): mkimage builds
      the SD payload (stim + exp + case-start flags at LBA 2048, 20 blocks);
      flash-card/verify need a USB reader + root (none attached at run time —
      no card was formatted yet; tool ready).
- [ ] Next rungs: K9_7/K9_4 V2 kernels; format the 256 GB card via the
      partner tool when a reader is attached; 4-bit SD mode (DAT1/2 = 85/80)
      for ~5× SPI throughput; fetch/compute overlap + skid buffer; SDRAM
      streaming stage (the streamed-model ladder's first rung).

## Board notes (Tang Nano 20K)

- Pins (apicula CST + official datasheet v1.3, cross-checked with LiteX's
  sipeed_tang_nano_20k): clk = pin 4 @ 27 MHz (on-board OSC), LEDs =
  15..20 = LED0..LED5 (6 discrete, next to button S1), BL616 USB-C UART
  TXD/RXD = 69/70, user buttons S1/S2 = pins 87/88 (both plain GPIOs; the
  small button near the flash is UPDATE = hardware reconfig, unused here).
  Reset polarity: we use pin 88 as active-low reset with pull-up.
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