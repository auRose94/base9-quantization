// Wiring probe: raw UART beacon, no gate logic. Repeats "HELLO n=NNNN\r\n"
// forever (n = 16-bit hex counter) from the instant the bitstream loads, with
// a walking-LED heartbeat. If the pod CH B UART is wired to pins 69/70, the
// host captures lines continuously.
module beacon_top (
  input  wire        clk,        // 27 MHz
  output wire [5:0]  led,
  output wire        txd
);
  reg [7:0] pwr = 8'd0;
  wire rst_n = pwr[7];
  always @(posedge clk) if (!pwr[7]) pwr <= pwr + 8'd1;

  logic        u_start = 1'b0, u_busy;
  logic [7:0]  u_data;
  uart_tx #(.DIV(234)) utx (.clk(clk), .rst_n(rst_n), .start(u_start),
                            .data(u_data), .tx(txd), .busy(u_busy));

  reg [15:0] n_cnt = 16'd0;

  function [7:0] ch_at(input [3:0] i);
    case (i)
      4'd0:  ch_at = "H";
      4'd1:  ch_at = "E";
      4'd2:  ch_at = "L";
      4'd3:  ch_at = "L";
      4'd4:  ch_at = "O";
      4'd5:  ch_at = " ";
      4'd6:  ch_at = "n";
      4'd7:  ch_at = "=";
      4'd8:  ch_at = {4'h3, n_cnt[15:12]};
      4'd9:  ch_at = {4'h3, n_cnt[11:8]};
      4'd10: ch_at = {4'h3, n_cnt[7:4]};
      4'd11: ch_at = {4'h3, n_cnt[3:0]};
      4'd12: ch_at = "\r";
      default: ch_at = "\n";
    endcase
  endfunction

  localparam [3:0] LINE_LEN = 4'd13;

  logic [3:0] tx_ptr = 4'd0;
  logic [1:0] tx_st = 2'd0;   // 0=issue, 1=wait ack, 2=wait done
  always @(posedge clk) begin
    if (!rst_n) begin
      tx_ptr <= 4'd0; tx_st <= 2'd0; u_start <= 1'b0;
    end else case (tx_st)
      2'd0: begin
        if (tx_ptr == LINE_LEN) begin
          tx_ptr <= 4'd0;
          n_cnt  <= n_cnt + 16'd1;
          tx_st  <= 2'd0;
        end else begin
          u_start <= 1'b1;
          u_data  <= ch_at(tx_ptr);
          tx_ptr  <= tx_ptr + 4'd1;
          tx_st   <= 2'd1;
        end
      end
      2'd1: begin
        u_start <= 1'b0;
        if (u_busy) tx_st <= 2'd2;
      end
      default: if (!u_busy) tx_st <= 2'd0;
    endcase
  end

  // walking-LED heartbeat at ~10 Hz
  reg [22:0] hbt = 23'd0;
  always @(posedge clk) begin
    if (!rst_n) hbt <= 23'd0;
    else hbt <= hbt + 23'd1;
  end
  reg [2:0] led_pos = 3'd0;
  reg [22:0] hbt_q = 23'd0;
  always @(posedge clk) begin
    hbt_q <= hbt;
    if (hbt[22] && !hbt_q[22]) led_pos <= led_pos + 3'd1;
  end
  assign led = 6'b1 << led_pos;
endmodule