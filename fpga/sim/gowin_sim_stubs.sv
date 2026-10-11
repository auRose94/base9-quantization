`ifdef SYNTHESIS
`else
// pass-through for the fabric-toggled clock (apicula routes it globally)
module BUFG (input wire I, output wire O);
  assign O = I;
endmodule

// behavioral models of the Gowin DSP cells, with the full port list yosys
// emits in netlists; the unused pipeline/enable inputs are tied off by the
// flow, so the model is just the exact integer product.
module MULT9X9 (output reg [17:0] DOUT, input [8:0] A, input [8:0] B,
                input ASIGN, input BSIGN, input ASEL, input BSEL,
                input [8:0] SIA, input [8:0] SIB,
                input CE, input CLK, input RESET);
  wire [17:0] p = (ASIGN ? $signed(A) * $signed(B) : A * B);
  always @* DOUT = p;
endmodule

module MULT18X18 (output reg [35:0] DOUT, input [17:0] A, input [17:0] B,
                  input ASIGN, input BSIGN, input ASEL, input BSEL,
                  input [17:0] SIA, input [17:0] SIB,
                  input CE, input CLK, input RESET);
  wire [35:0] p = (ASIGN ? $signed(A) * $signed(B) : A * B);
  always @* DOUT = p;
endmodule

module MULT36X36 (output reg [71:0] DOUT, input [35:0] A, input [35:0] B,
                  input ASIGN, input BSIGN,
                  input [35:0] SIA, input [35:0] SIB,
                  input CE, input CLK, input RESET);
  wire [71:0] p = (ASIGN ? $signed(A) * $signed(B) : A * B);
  always @* DOUT = p;
endmodule
`endif
