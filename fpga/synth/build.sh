#!/bin/bash
# Gate-0 silicon variants. Default: the SD-card-fed gate (gate0sd_top) —
# boots the microSD, verifies the payload roundtrip, runs the gate, reports
# on LEDs + UART. Hold the user key (pin 87) during boot to RE-PROVISION the
# card (writes 20 LBAs from the bitstream's embedded payload at LBA 2048).
# Targets: gate0sd_top | gate0v2_top (BRAM-only gate) | beacon_top.
TARGET=${1:-gate0sd_top}
set -e
cd "$(dirname "$0")"
source ~/Work/tools/oss-cad-suite/environment

[ -f stim.hex ] || echo "warning: stim.hex missing (needed for $TARGET)"

if [ "$TARGET" = "gate0sd_top" ]; then
  RTL="../rtl/f32_pkg.sv ../rtl/k96_vec_dot_v2.sv ../rtl/spi_byte_master.sv ../rtl/sd_spi_reader.sv uart_tx.sv gate0sd_top.sv"
elif [ "$TARGET" = "gate0v2_top" ]; then
  RTL="../rtl/f32_pkg.sv ../rtl/k96_vec_dot_v2.sv uart_tx.sv gate0v2_top.sv"
else
  RTL="uart_tx.sv"
fi

# -nodsp: yosys maps the fp32 datapath's mantissa products onto the Gowin DSP
# macros with operand/enable ties that do not behave like the RTL on silicon
# (the S2-hold gate-FAIL trace); -nobram/-nolutram are NOT set (BRAM is fine).
yosys -ql gate_synth.log -p "read_verilog -sv $RTL; synth_gowin -top $TARGET -json gate.json -family gw2a -nodsp"
nextpnr-himbaechel --json gate.json --device GW2AR-LV18QN88C8/I7 \
  --vopt family=GW2A-18C --vopt cst=board.cst --freq 13 --write gate_pnr.json
gowin_pack -c -d GW2A-18C -o $TARGET.fs gate_pnr.json
echo "$TARGET.fs written; flash: openFPGALoader --board tangnano20k $TARGET.fs"
