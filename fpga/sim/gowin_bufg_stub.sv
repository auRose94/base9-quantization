// BUFG pass-through for simulation (yosys supplies the real Gowin cell)
`ifdef SYNTHESIS
`else
module BUFG (input wire I, output wire O);
  assign O = I;
endmodule
`endif
