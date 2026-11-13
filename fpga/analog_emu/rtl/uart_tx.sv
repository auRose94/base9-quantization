// Minimal UART TX, 8N1, latch-on-idle. DIV = clk / baud (e.g. 234 @ 27 MHz
// -> 115384 baud, 0.16% off 115200 -- within tolerance).
module uart_tx #(parameter integer DIV = 234) (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       start,
  input  wire [7:0] data,
  output reg        tx = 1'b1,
  output reg        busy
);
  reg [15:0] div_cnt = 16'd0;
  reg [3:0]  bit_idx = 4'd0;
  reg [9:0]  sh;

  always @(posedge clk) begin
    if (!rst_n) begin
      tx <= 1'b1; busy <= 1'b0; div_cnt <= 16'd0; bit_idx <= 4'd0;
    end else if (!busy) begin
      tx <= 1'b1;
      if (start) begin
        sh <= {1'b1, data, 1'b0};   // {stop, 8 data, start}; start = sh[0]
        bit_idx <= 4'd0; div_cnt <= 16'd0;
        tx <= 1'b0;                 // start bit
        busy <= 1'b1;
      end
    end else if (div_cnt == 16'(DIV) - 16'd1) begin
      div_cnt <= 16'd0;
      if (bit_idx == 4'd9) begin
        busy <= 1'b0;
        tx <= 1'b1;
      end else begin
        bit_idx <= bit_idx + 4'd1;
        tx <= sh[bit_idx + 4'd1];
      end
    end else div_cnt <= div_cnt + 16'd1;
  end
endmodule