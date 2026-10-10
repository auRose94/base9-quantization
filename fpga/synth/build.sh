#!/bin/bash
# Gate-0 silicon gate for Tang Nano 20K (GW2AR-LV18QN88C8/I7).
# gen memory -> yosys -> nextpnr-himbaechel -> apicula gowin_pack.
set -e
cd "$(dirname "$0")"
source ~/Work/tools/oss-cad-suite/environment

[ -f stim.hex ] || python3 gen_mem.py

yosys -ql gate0_synth.log -p \
  "read_verilog -sv ../rtl/f32_pkg.sv ../rtl/k96_vec_dot.sv uart_tx.sv gate0_top.sv; \
   synth_gowin -top gate0_top -json gate0.json -family gw2a"

nextpnr-himbaechel --json gate0.json \
  --device GW2AR-LV18QN88C8/I7 \
  --vopt family=GW2A-18C --vopt cst=board.cst \
  --freq 27 --write gate0_pnr.json 2>&1 | tail -30

gowin_pack -c -d GW2A-18C -o gate0.fs gate0_pnr.json
echo "gate0.fs written"