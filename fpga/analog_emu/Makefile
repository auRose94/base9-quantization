# analog_emu gate — golden.py -> simulator -> bit-exact compare.
# `make gate` auto-picks iverilog, else verilator --binary.
PY      ?= python3
ROWS    ?= 16
COLS    ?= 16
NTOK    ?= 8
SEED    ?= 7
BUILD   ?= sim_build

.PHONY: gate gate64 golden clean

golden:
	$(PY) golden.py --rows $(ROWS) --cols $(COLS) --tokens $(NTOK) --seed $(SEED)

gate: golden
	$(MAKE) gate-verilator

# (iverilog v13 dropped -G parameter overrides — verilator is the house sim,
# same convention as fpga/sim/Makefile.)

gate-verilator:
	mkdir -p $(BUILD)
	verilator --binary --timing -Wno-fatal -DSIM_LOAD --top-module tb_tile \
	    $$(cat sim_flags.txt) rtl/tb_tile.sv rtl/synapse_tile.sv \
	    --Mdir $(BUILD) -o Vtb_tile
	$(BUILD)/Vtb_tile | tee $(BUILD)/gate.log
	@grep -q "GATE PASS" $(BUILD)/gate.log && echo "== gate: PASS ==" \
	    || { echo "== gate: FAIL =="; exit 1; }

gate64:
	$(MAKE) gate ROWS=64 COLS=64 NTOK=4

clean:
	rm -rf $(BUILD) golden_w.memh golden_g.memh golden_tokens.hex \
	    golden_expected.hex sim_flags.txt

# serial-tile gate (silicon bring-up variant): goldens with the SILICON
# constants (out-shift 14, thresh 96 — matches emu_sd_top), not calibrated
gate-ser:
	$(PY) golden.py --rows $(ROWS) --cols $(COLS) --tokens $(NTOK) --seed $(SEED) --out-shift 14 --thresh 96
	$(PY) flat16.py 14 $(ROWS)
	verilator --binary --timing -Wno-fatal -DSIM_LOAD --top-module tb_tile_ser \
	    $$(cat sim_flags.txt) -GNTOK=$$(grep -o 'NTOK=[0-9]*' sim_flags.txt | cut -d= -f2) \
	    rtl/tb_tile_ser.sv rtl/synapse_tile_ser.sv --Mdir $(BUILD) -o Vtb_ser
	$(BUILD)/Vtb_ser | tee $(BUILD)/gate_ser.log
	@grep -q "GATE PASS" $(BUILD)/gate_ser.log && echo "== gate-ser: PASS ==" \
	    || { echo "== gate-ser: FAIL =="; exit 1; }
