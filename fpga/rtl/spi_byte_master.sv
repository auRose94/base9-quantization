// Single-byte SPI exchange engine for the SD reader (and any 3-wire master).
//
// Per bit: drive MOSI while SCK stays low (`half` cycles), raise SCK for
// `half` cycles (the card latches MOSI at the rise), drop SCK and keep it low
// `half` cycles, then latch MISO (a full half-cycle after the fall -- the SD
// card changes MISO exactly on the fall, so sampling at the fall would race).
// After the eighth bit one extra low half-period elapses and `done` pulses.
// SCK duty is 3:1 low:high within a bit; the SD spec bounds only frequency.
// Frequencies: half=1 -> 9 MHz, half=36 -> 250 kHz (init window, < 400 kHz).
module spi_byte_master (
  input  wire        clk, rst_n,
  input  wire        byte_go,     // 1-cycle pulse, only accepted while idle
  input  wire [7:0]  tx,
  input  wire        slow,        // 1 = long half-period (init clock)
  output reg         busy,
  output reg         done,        // 1-cycle pulse; rx valid the same cycle
  output reg  [7:0]  rx,
  output reg         sck,
  output reg         mosi,
  input  wire        miso
);
  localparam integer HALF_SLOW = 36;
  localparam integer HALF_FAST = 1;

  wire [15:0] half = slow ? HALF_SLOW[15:0] : HALF_FAST[15:0];

  localparam S_IDLE = 3'd0, S_DRIVE = 3'd1, S_HIGH = 3'd2,
             S_SAMPLE = 3'd3, S_FIN = 3'd4;
  reg [2:0]  st;
  reg [2:0]  bit_i;
  reg [15:0] hcnt;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; busy <= 1'b0; done <= 1'b0; bit_i <= 3'd0;
      hcnt <= 16'd0; rx <= 8'hFF; sck <= 1'b0; mosi <= 1'b1;
    end else begin
      done <= 1'b0;
      case (st)
        S_IDLE: begin
          sck <= 1'b0;
          if (byte_go) begin
            busy <= 1'b1; bit_i <= 3'd0; hcnt <= 16'd0;
            mosi <= tx[7];
            st <= S_DRIVE;
          end
        end
        S_DRIVE: begin              // MOSI set, SCK low
          hcnt <= hcnt + 16'd1;
          if (hcnt >= half - 16'd1) begin
            hcnt <= 16'd0; sck <= 1'b1; st <= S_HIGH;
          end
        end
        S_HIGH: begin               // card latches MOSI at the rise
          hcnt <= hcnt + 16'd1;
          if (hcnt >= half - 16'd1) begin
            hcnt <= 16'd0; sck <= 1'b0; st <= S_SAMPLE;
          end
        end
        S_SAMPLE: begin             // a half-cycle after the fall: safe latch
          hcnt <= hcnt + 16'd1;
          if (hcnt >= half - 16'd1) begin
            hcnt <= 16'd0;
            rx <= {rx[6:0], miso};
            if (bit_i == 3'd7) st <= S_FIN;
            else begin
              bit_i <= bit_i + 3'd1;
              mosi  <= tx[7 - (bit_i + 3'd1)];
              st <= S_DRIVE;
            end
          end
        end
        default: begin              // S_FIN: one extra low half, then done
          hcnt <= hcnt + 16'd1;
          if (hcnt >= half - 16'd1) begin
            hcnt <= 16'd0; busy <= 1'b0; done <= 1'b1; st <= S_IDLE;
          end
        end
      endcase
    end
  end
endmodule