// Pad-binding probe for the GW2AR in-package (SIP) SDRAM. Carries the exact
// port names the apicula chipdb lists (IO_sdram_dq/O_sdram_*) with constant
// idle drives: all commands high (deasserted), CKE low (die power-down until
// a later design runs init), DQ tri-stated. Purpose = answer whether
// nextpnr-himbaechel binds these pads by port NAME (no IO_LOC lines).
module sdram_probe_top (
  input  wire clk,          // 27 MHz
  output wire [5:0] led,
  output wire        O_sdram_clk,
  output wire        O_sdram_cke,
  output wire        O_sdram_cs_n,
  output wire        O_sdram_ras_n,
  output wire        O_sdram_cas_n,
  output wire        O_sdram_wen_n,
  output wire [10:0] O_sdram_addr,
  output wire [1:0]  O_sdram_ba,
  output wire [3:0]  O_sdram_dqm,
  inout  wire [31:0] IO_sdram_dq
);
  reg [1:0] divc = 2'd0;
  always @(posedge clk) divc <= divc + 2'd1;
  wire clk13_r = divc[1];

  // 100 ms walking light + a fixed heartbeat bit, just a visible identity
  reg [5:0] blink = 6'b000001;
  reg [23:0] bcnt = 24'd0;
  always @(posedge clk13_r) begin
    bcnt <= bcnt + 24'd1;
    if (bcnt == 24'd1350000) begin bcnt <= 24'd0; blink <= {blink[4:0], blink[5]}; end
  end
  assign led = ~blink;

  // die held in power-down; DQ tri-stated; commands deasserted
  assign O_sdram_clk  = clk13_r;
  assign O_sdram_cke  = 1'b0;
  assign O_sdram_cs_n  = 1'b1;
  assign O_sdram_ras_n = 1'b1;
  assign O_sdram_cas_n = 1'b1;
  assign O_sdram_wen_n = 1'b1;
  assign O_sdram_addr  = 11'd0;
  assign O_sdram_ba    = 2'd0;
  assign O_sdram_dqm   = 4'hF;
  assign IO_sdram_dq   = 32'bz;
endmodule