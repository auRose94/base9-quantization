#!/bin/bash
# analog_emu silicon build — Tang Nano 20K, oss-cad-suite flow (G0 pattern).
# NOTE: no -nodsp here — the tile's 16x18 INTEGER multiplier maps to one DSP
# macro and is exact (the G0 -nodsp guard was about fp32 mantissa ties).
set -e
cd "$(dirname "$0")"
source ~/Work/tools/oss-cad-suite/environment
yosys -ql gate_synth.log -p "read_verilog -sv ../rtl/synapse_tile_ser.sv ../rtl/uart_rx.sv ../rtl/uart_tx.sv ../rtl/emu_top.sv; synth_gowin -top emu_top -json gate.json -family gw2a"
nextpnr-himbaechel --json gate.json --device GW2AR-LV18QN88C8/I7 \
  --vopt family=GW2A-18C --vopt cst=analog.cst --freq 27 --write gate_pnr.json
gowin_pack -c -d GW2A-18C -o emu_top.fs gate_pnr.json
echo "emu_top.fs written; flash: openFPGALoader --board tangnano20k emu_top.fs"
