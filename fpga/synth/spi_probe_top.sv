// SPI byte-engine subsystem probe: the LEDs show the exchange count directly.
// Bit5 lit + all four index LEDs dark = all 8 exchanges completed and the
// engine retired (engine works on silicon); a frozen lower count = the
// netlist/silicon-reproducible engine stall. SCK toggles on its card pin
// through the same slow-regime waveform the gate's init phase uses.
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
  reg        m_busy, m_done;    // m_done is a 1-cycle pulse from the engine
  reg [7:0]  m_rx;

  spi_byte_master engine (
    .clk(clk), .rst_n(rst_n),
    .byte_go(m_go), .tx(m_tx), .slow(m_slow),
    .busy(m_busy), .done(m_done), .rx(m_rx),
    .sck(sd_sck), .mosi(sd_mosi), .miso(sd_miso)
  );

  assign sd_cs_n = 1'b1;   // probe: CS high (the C80 phase's condition)

  reg [3:0] cnt = 4'd0;      // exchanges completed, 0..8
  reg       started = 1'b0;
  reg [6:0] warm = 7'd0;
  always @(posedge clk) begin
    m_go <= 1'b0;
    if (!rst_n) begin
      cnt <= 4'd0; started <= 1'b0; warm <= 7'd0;
    end else if (!warm[6]) begin
      warm <= warm + 7'd1;              // settle; device regs come up zeroed
    end else if (!started) begin
      m_go <= 1'b1;                     // first exchange, unconditionally
      started <= 1'b1;
    end else if (m_done && cnt != 4'd7) begin
      cnt <= cnt + 4'd1;                // 8 total exchanges: 7 follow-ups
      m_go <= 1'b1;
    end else if (m_done) begin
      cnt <= 4'd8;                      // all done: engine proven live
    end
  end

  // LED contract (active-low pads): bit5 lit + index dark = finished;
  // otherwise the count shows on the index LEDs, bit5 dark.
  wire [5:0] lit_mask = (cnt == 4'd8) ? 6'b101111 : {1'b0, 1'b0, cnt[3:0]};
  assign led = ~lit_mask;
endmodule