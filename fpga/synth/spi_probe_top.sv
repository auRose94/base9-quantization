// SPI-engine subsystem test: the sd reader's byte master runs through the
// reader's C80 phase (10 exchanges, CS high, slow regime) while the LEDs
// show both the exchange count (upper 4) and the engine FSM (lower 2).
// A live design: the count climbs 0..10 and the reader completes into its
// next states; a frozen count = a silicon-specific engine stall.
module spi_probe_top (
  input  wire        clk,        // 27 MHz
  output wire [5:0]  led,
  output wire        sd_cs_n,
  output wire        sd_sck,
  output wire        sd_mosi,
  input  wire        sd_miso
);
  reg [7:0] pwr = 8'd0;
  wire rst_n = pwr[7];
  always @(posedge clk) if (!pwr[7]) pwr <= pwr + 8'd1;

  reg        m_go = 1'b0, m_slow = 1'b1;
  reg [7:0]  m_tx = 8'hFF;
  reg        m_busy, m_done;
  reg [7:0]  m_rx;

  spi_byte_master engine (
    .clk(clk), .rst_n(rst_n),
    .byte_go(m_go), .tx(m_tx), .slow(m_slow),
    .busy(m_busy), .done(m_done), .rx(m_rx),
    .sck(sd_sck), .mosi(sd_mosi), .miso(sd_miso)
  );

  assign sd_cs_n = 1'b1;   // C80 phase: CS high

  reg [2:0] cnt = 3'd0;
  always @(posedge clk) begin
    m_go <= 1'b0;
    if (m_done && cnt != 3'd7) begin
      cnt <= cnt + 3'd1;
      m_go <= 1'b1;
    end
  end

  wire [5:0] lit_mask = (m_done ? 1'b1 : 1'b0) | {1'b0, cnt} ;
  assign led = ~lit_mask;
endmodule
