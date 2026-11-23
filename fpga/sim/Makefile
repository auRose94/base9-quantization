# Gate-0 sims. Usage:  make SIM=verilator TOP=hello MOD=tb_hello RTL=hello.sv
SIM                 ?= verilator
TOPLEVEL_LANG       ?= verilog
TOP                 ?= hello
MOD                 ?= tb_hello
RTL                 ?= hello.sv

TOPLEVEL            = $(TOP)
COCOTB_TOPLEVEL     = $(TOP)
MODULE              = $(MOD)
COCOTB_TEST_MODULES = $(MOD)
VERILOG_SOURCES     = $(RTL)
VERILATOR_FLAGS     += --timing
SIM_BUILD           = sim_build/$(TOP)
# sdram_top: sim-shim define (DQ injection port, Verilator can't do the
# real top-level tristate) + the shortened hold phase. Kept here (not the
# CLI) so the makefile's own COMPILE_ARGS += survive.
ifeq ($(TOP),sdram_top)
COMPILE_ARGS        += -DSIMULATION -GHOLD_CYC=3000
endif
ifeq ($(TOP),sdram_net_top)
COMPILE_ARGS        += -Wno-SPECIFYIGN -Wno-UNOPTFLAT
endif

include $(shell cocotb-config --makefiles)/Makefile.sim