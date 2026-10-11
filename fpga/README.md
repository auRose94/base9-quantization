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
- [x] **REAL TENSORS through the whole gate (session 30)**: the card payload
      is no longer rng-seeded — `partner/real_slice_image.py` slices REAL
      K9_6 superblocks out of the SERVED 1.5B coder container
      (`results/qwen_coder_1.5b_k63_embed99.k9`, k9-decoder digits packed by
      the Gate-A packer) and crosses them with a real activation h — default
      = the runtime-chain dequant of the `__embed__` record's row
      (`--act embd:2048`); `--act file:/tmp/ninths_h.npy` = the fork's
      `llama-embedding` last-token state of a real prompt, captured through
      the real loader (`--pooling last`) — a genuine forward-pass hidden
      state. Plan: 4 chains = self_attn.o_proj rows of layers 0/1/2 (6 sb
      each) + a 3-sb partial = 21 records. Three-level bit-exact closure
      before anything is written: (1) per-CALL numpy == verbatim fork C
      harness on every singleton; (2) row-tree numpy == C harness on every
      chain; (3) the image's exp = the gate's own chain tree. Full-top sims:
      synthetic G0 PASS under the corrected exps; real-tensor PASS on BOTH
      `gate0sd_top` (NIN_SD_IMG/NIN_SD_NPZ/NIN_SD_PAY_POKE envs) and the new
      `synth/gate0sd_real_top.sv` (embedded `pay_real.hex`, full telegraph,
      all three silicon fixes, no domain probes). `sim/tb_real96.py` = fast
      kernel-level probe over any npz (V1 shape) with a cycle meter.
      **PnR'd both**: `gate0sd_top.fs` rebuilt under the corrected exps (the
      formal-retest bitstream) and `gate0sd_real_top.fs` = flash-ready;
      the real-top netlist closes 13.14 MHz at the 7 MHz target (1.9×
      margin, critical path = the sumf f32 chain — expected).
- [!] **PROTOCOL FIX found by real data**: the original gate compared the
      hardware's per-superblock chained sumf against the C ROW-kernel tree
      (the golden's per-group accumulation interleaved along the chain).
      Those are DIFFERENT fp32 add trees — they matched on the synthetic
      vectors only because 18/21 records were singleton chains; real
      magnitudes diverge by 1-ulp-class. NOT a hardware bug: every per-
      superblock partial matched the numpy singleton exactly (per-record
      dump). Protocol now: exp = the chain tree (deterministic IEEE of the
      exact adds the FPGA runs); the row tree remains the python==C check.
      Consequence for the formal retest: the payload's exp+flags words
      changed (42 of 2520), so the post-S1-repair reflash needs one
      S2-provision boot on the new image (already the reflash plan).
- [x] **K9_7 V2 streaming kernel (session 30 cont 1)** —
      `rtl/k97_vec_dot_v2.sv`: 264-byte container, one contiguous 256-code
      signed-int8 stream (d − H, H = 49), 9-bit code addresses, same FSM +
      fp ops + scale-service shape as the k9_6 V2. Bit-exact vs
      `golden.k97_partials` and the verbatim fork kernel: synthetic 20/20
      (fp16-inf-scale, exact-zero and ±49-code extremes included) and REAL
      12/12 — the 1.5B is tied-embed so `__embed__` (k99) rows ARE the
      unembed; the real vectors = embed rows 0/1 × the llama-embedding h
      (`real_slice_image.py --kind k9_7` → `/tmp/gate0_k97_real_vecs.npz`).
      tb = port-side caller bench (tb_k97.py: the top's sync-read + mux
      circuit in one Scratch task, S_SUMF one-cycle scale forecast,
      K97_TRACE per-cycle debug; tb_k97dbg = the ext-codes zero-invariant
      single-record case). This is the rung the streamed-unembed logits
      demo needs (no k9_7 card top yet).
- [ ] Next rungs: the k9_7 card top + streamed-unembed loop; K9_4 V2
      kernel; 4-bit SD mode (DAT1/2 = 85/80) for 4× SPI throughput;
      fetch/compute overlap + skid buffer; SDRAM streaming stage.

## Honest token economics of this device class (2026-10-11, session 30)

Batch-1 decode of entropy-coded weights = every weight streamed once per
token, so the card lane and the fabric rate cap the token rate together.
The V2 kernel's schedule costs ~400 cycles per 200-byte superblock plus a
non-overlapped 118-cycle record load (the skid/overlap rung removes the
load bubble). The card lane is the real limiter: `spi_byte_master` spends
~33 clk13 cycles per byte at HALF_FAST=1, so a 1-bit lane ≈ 420 kB/s at the
measured ~13.4 MHz fabric clock (≈ 210 kB/s at the intended 6.75 MHz) —
the fabric itself feeds ~3.4–6.3 MB/s depending on the same question, so
today the card, not the MAC, is the bottleneck by an order of magnitude.

| pass per token (streaming, K9_6) | bytes | 1-bit today (~420 kB/s) | 4-bit, 1.7 MB/s |
|---|---:|---:|---:|
| unembed/logits pass only, 1.5B | 182 MB | 7.2 min | 107 s |
| full model, 1.5B | 1.2 GB | 47 min | 11.8 min |
| full model, 7B | 5.5 GB | 3.6 h | 54 min |
| full model, 14B | 11.2 GB | 7.4 h | 1.8 h |

With the planned rungs (4-bit + SD high-speed mode to a ~25 MHz SCK = ~12.5
MB/s) the fabric becomes the cap: ~3–6 min per token on 1.5B (~0.2 t/s) at
this chip class, ~29 min on 14B. The 5060 Ti serves the same tuned
containers at 109 t/s. **So stated plainly: this $15 board (≈0.4 W) is the
bit-exact validator, protocol substrate and novelty demo for ninths
containers — not an LLM serving engine; even DDR3-class FPGAs stay
sub-1 t/s at 7B via streaming. The useful rungs here are exactly what
shipped: a silicon proof on real container bytes, a real-model logits demo,
then SDRAM as a working-set cache; the serving-engine tier would be a
different board class entirely.**

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

**Silicon-vs-sim divergences (the debugging that matters, session 29):**
1. **The fp32 datapath breaks if synthesized to DSP** — yosys's gowin flow
   maps `*` onto the Gowin DSP macro with the pipeline/enable/operand-select
   ports tied to whatever net was free; the real DSP's operand-flapping
   diverges from any RTL sim (the gate failed its FIRST record on silicon
   while every RTL sim passed). **Build with `synth_gowin -nodsp`; const
   multiplies were rewritten as shifts.** Verifiable by netlist simulation
   (yosys `write_verilog` + verilator over `share/yosys/gowin/cells_sim.v`
   + `fpga/sim/gowin_sim_stubs.sv`).
2. **Runtime-written memories need an explicit zero `$readmemh` init** or
   synth_gowin's memory opt keeps/loses them differently than the RTL.
3. Self-gated loop conditions (`if (mag[i] && p == 0)`) — write loops as
   unconditional ascending scans instead.

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