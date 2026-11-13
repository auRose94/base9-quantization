// Minimal UART RX, 8N1, 16x-ish oversample via bit counter.  DIV = clk/baud
// (234 @ 27 MHz -> 115384 baud, matches uart_tx).  valid is a 1-clk pulse.
module uart_rx #(parameter integer DIV = 234) (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       rx,
  output reg        valid,
  output reg  [7:0] data
);
  reg [2:0]  sync = 3'b111;      // 2FF sync + history for edge detect
  reg        active = 1'b0;
  reg [15:0] cnt;
  reg [2:0]  bits;
  reg [7:0]  sh;

  wire rx_s  = sync[1];
  wire fall  = sync[2] & ~sync[1];

  always @(posedge clk) begin
    if (!rst_n) begin
      sync <= 3'b111; active <= 1'b0; valid <= 1'b0;
    end else begin
      sync <= {sync[1:0], rx};
      valid <= 1'b0;
      if (!active) begin
        if (fall) begin
          active <= 1'b1;
          cnt <= 16'd0; bits <= 3'd0;
        end
      end else begin
        cnt <= cnt + 1;
        if (bits == 0 && cnt == (DIV >> 1)) begin
          if (!rx_s) begin
            cnt <= 16'd0;      // confirmed start; sample at DIV boundaries
            bits <= 3'd1;
          end else begin
            active <= 1'b0;    // glitch
          end
        end else if (bits != 0 && cnt == DIV - 1) begin
          cnt <= 16'd0;
          if (bits <= 8) begin
            sh  <= {rx_s, sh[7:1]};
            bits <= bits + 1;
          end else begin
            active <= 1'b0;
            data  <= sh;       // stop-bit position; not checked (host is
            valid <= 1'b1;     // deterministic) — ledger note
          end
        end
      end
    end
  end
endmodule
