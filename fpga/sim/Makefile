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

include $(shell cocotb-config --makefiles)/Makefile.sim