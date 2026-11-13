# analog_emu — rung 2¾: RTL mixed-signal emulator

A synthesizable SystemVerilog emulation of one analog synapse tile — the
rung between the statistical channel simulator (`experiments/analog_channel.py`,
Phase A/B) and physical hardware (rung 2½ breadboard, later the appliance
PCB).  No noisy hardware required: the "analog" semantics are exact integer
arithmetic that both python and the FPGA reproduce bit-for-bit.

## The key architectural decision

**Degradation lives in the array image, not in the datapath.**  Programming
noise, stuck cells, column-gain mismatch — python bakes all of it into the
weight/gain memory image (any distribution it likes: the channel model's
Gaussians today, *measured histograms from rung 2½'s parts* tomorrow).  The
RTL then just does what a real array does: compute with whatever conductances
were programmed.  Consequences:

- the gate is bit-exact with no cross-language RNG (the classic trap),
- the same image replays identically forever (deterministic = gateable),
- degradation models evolve in python without touching RTL.

## The arithmetic contract (v1)

Everything is signed fixed-point, floor semantics (`>>>` in SV, `>>` in
python — both floor; **no division anywhere**, rails are powers of two):

| stage | definition |
|---|---|
| weight image | `W[r][c]` signed 16-bit Q4.12, k9-grid digits `w = m·(d−H)/H` quantized, then degradation baked (uniform ±σ LSBs, stuck rails) |
| column gain | `G[c]` unsigned 8-bit Q0.7, 256 = ×1.0 (the per-column calibration the write-path findings demand) |
| DAC in | `xq = clip(x >>> (WW−DACB), ±(2^(DACB−1)−1))` — symmetric-odd grid (always an exact zero level, per exp22) on fixed rails ±8.0, then `xq_d = xq <<< (WW−DACB)` |
| column apply | `xg = (xq_d · G[c]) >>> 7` |
| MAC | `acc[r] += W[r][c] · xg` into signed 40-bit |
| ADC out | `yq = clip(acc >>> OUT_SHIFT, ±(2^(ADCB−1)−1)); y = yq <<< OUT_SHIFT` — fixed rails (Phase B used per-token auto-range; the tile models the real fixed-rail converter; delta noted) |

Delta vs Phase B, stated honestly: the numpy channel referenced noise to
per-token amax (an idealized auto-ranging converter); the tile uses fixed
rails — more physical, simpler, and division-free.  Fixed rails are the
price ladder the appliance will actually pay.

## Files

    rtl/synapse_tile.sv   the tile (parameters: ROWS COLS WW DACB ADCB OUT_SHIFT)
    rtl/tb_tile.sv        testbench: streams tokens, compares to golden
    golden.py             builds the image (k9 grid + baked degradation),
                          generates tokens + expected outputs, runs the
                          reference datapath in pure python ints
    Makefile              `make gate` (16×16) / `make gate64` (64×64)

## Gate

    make gate      # verilator --binary (pacman -S verilator); iverilog v13
                   # dropped -G, so the house sim is verilator, like fpga/sim

PASS = every output word of every token matches the python reference
bit-for-bit.  This is the char9 discipline one abstraction level down:
golden vectors, diff, PASS/FAIL.

## Status

- **v1 DONE** — fixed-point MVM tile + DAC/ADC, gate bit-exact in sim.
- **v2 DONE** — on-tile LFSR16 read-noise (xorshift 7,9,8, symmetric-odd
  ±2^(k−1)−1 ADC-LSBs, drawn fresh per row per token — the deterministic
  RTL twin of the Phase-B read path), drift step (exponential conductance
  relaxation `W -= W >> s` swept in-place between token batches), and a
  per-row LIF core (leak → integrate → threshold → reset-to-zero spike,
  BrainScaleS-style).  Gate bit-exact at 16×16 and 64×64 with the spike
  path exercised (519 spikes verified word-for-word).  A `--sweep` mode
  prints the tile-level survival curve (RMS vs read-noise power — doubles
  per noise-power step, as the physics demands).
- **Fixed-rail finding** (v2, physical): rails sized to the max accumulator
  quantize typical outputs to zero — the heavy tail forces either
  auto-ranging (Phase-B idealization) or accepting outlier clipping
  (`--out-shift` override).  Same trade a real ADC board makes.
- **v3 next**: Tang Nano 20K bring-up — one 64×64 tile on silicon
  (UART-programmed images), then the `.k9` connection: the weight image
  format IS the container — digits plus group scales; `experiments/k9.py`
  records feed it directly.  The container research becomes the array's
  programming format.
