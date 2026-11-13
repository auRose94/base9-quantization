#!/bin/bash
# analog_emu SD-gate build — no -nodsp (the tile's integer multiply is exact)
set -e
cd "$(dirname "$0")"
source ~/Work/tools/oss-cad-suite/environment
yosys -ql sd_synth.log -p "read_verilog -sv ../rtl/synapse_tile_ser.sv ../../rtl/sd_spi_reader.sv ../../rtl/spi_byte_master.sv ../rtl/emu_sd_top.sv; synth_gowin -top emu_sd_top -json sd.json -family gw2a"
nextpnr-himbaechel --json sd.json --device GW2AR-LV18QN88C8/I7 \
  --vopt family=GW2A-18C --vopt cst=analog-sd.cst --freq 27 --write sd_pnr.json
gowin_pack -c -d GW2A-18C -o emu_sd_top.fs sd_pnr.json
echo "emu_sd_top.fs written"
