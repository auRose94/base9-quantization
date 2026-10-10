// On-silicon Gate-0: K9_6 superblock vec_dot bit-exactness wall test.
//
// Stimulus (the exact cocotb Gate-0 records, /tmp/gate0_vecs.npz via gen_mem.py)
// lives in BRAM as 21 x 118 words; the top streams each record into the
// k96_vec_dot kernel, chains its fp32 partial into the running sumf with the
// same sequential f32_add the C kernel does, and compares against the
// precomputed golden cumulative state per record. Verdict on LEDs + UART.
//
// LEDs: while running led[5]=1, led[3:0] = record index; on PASS all six on;
// on FAIL led[4]=1, led[3:0] = first mismatching record index (polarity of
// pins 15/16/17 -- RGB on some boards -- may be inverted; UART is the
// authoritative channel).
module gate0_top (
  input  wire        clk,        // 27 MHz
  input  wire        rst_b,      // active-low board reset (pull-up)
  output wire [5:0]  led,
  output wire        txd
);
  localparam integer N_RECS    = 21;
  localparam integer W_PER_REC = 118;   // 50 wblk + 68 act words

  // power-up settle + manual reset; device regs come up zeroed
  reg [7:0] pwr = 8'd0;
  wire rst_n = pwr[7] & rst_b;
  always @(posedge clk) begin
    if (!rst_b) pwr <= 8'd0;
    else if (!pwr[7]) pwr <= pwr + 8'd1;
  end

  // ------------------------------------------------------------- BRAM -----
  reg [31:0] stim [0:4095];
  reg [31:0] exp_mem [0:N_RECS-1];
  reg        flags_mem [0:N_RECS-1];   // 1 = record starts a new case chain
  reg [31:0] stim_q;
  reg [31:0] exp_q;
  reg        flags_q;
  initial begin
    $readmemh("stim.hex", stim);
    $readmemh("exp.hex", exp_mem);
    $readmemh("flags.hex", flags_mem);
  end
  always @(posedge clk) stim_q  <= stim[rd_addr];
  always @(posedge clk) exp_q   <= exp_mem[exp_addr];
  always @(posedge clk) flags_q <= flags_mem[exp_addr];

  reg [11:0] rd_addr  = 12'd0;
  reg [4:0]  exp_addr = 5'd0;

  // ------------------------------------------------------------- kernel ---
  logic        k_busy, k_done;
  logic [31:0] k_partial;
  logic        k_start;

  k96_vec_dot kernel (
    .clk(clk), .rst_n(rst_n),
    .start(k_start), .wblk(wblk_sh), .act(act_sh),
    .busy(k_busy), .done(k_done), .partial_bits(k_partial)
  );

  // ------------------------------------------------------------- chain ----
  logic [31:0]  sumf = 32'd0;
  logic [4:0]   sb_idx = 5'd0;
  logic         fail = 1'b0;
  logic [4:0]   fail_idx = 5'd0;
  logic         done_flag = 1'b0;

  logic [7:0]   ld_cnt = 8'd0;      // captures completed in current record
  logic [1599:0] wblk_sh;
  logic [2175:0] act_sh;
  logic [31:0]  exp_r;
  logic         cs_r;               // case-start flag captured with exp_r

  localparam S_BOOT=3'd0, S_LOAD=3'd1, S_RUNA=3'd2, S_RUNW=3'd3,
             S_WAIT=3'd4, S_DONE=3'd5;
  reg [2:0] st = S_BOOT;

  // fp32 chaining add: same sequential add op as the C kernel's sumf updates
  logic [31:0] sumf_next;
  assign sumf_next = f32_pkg::f32_add(sumf, k_partial);

  always @(posedge clk) begin
    if (!rst_n) begin
      st <= S_BOOT; sb_idx <= 5'd0; sumf <= 32'd0; fail <= 1'b0;
      ld_cnt <= 8'd0; fail_idx <= 5'd0; done_flag <= 1'b0; k_start <= 1'b0;
    end else begin
      k_start <= 1'b0;
      case (st)
        S_BOOT: begin
          ld_cnt <= 8'd0; sb_idx <= 5'd0;
          rd_addr  <= 12'd117;          // top word of record 0
          exp_addr <= 5'd0;
          st <= S_LOAD;
        end
        // Descending stream with LEFT-shift-in: the newest word (lowest index)
        // lands at the LSB end each capture, so after all 118 captures the
        // register holds wblk/act with byte 0 at bit 0 -- exactly the port
        // layout of k96_vec_dot. Words 50..117 belong to act, 49..0 to wblk.
        S_LOAD: begin
          if (ld_cnt != 8'd0) begin
            if (ld_cnt <= 8'd68) act_sh  <= {act_sh[2143:0],  stim_q};
            else                 wblk_sh <= {wblk_sh[1567:0], stim_q};
          end
          if (ld_cnt == 8'd118) begin
            exp_r <= exp_q;
            cs_r  <= flags_q;
            st <= S_RUNA;
          end else begin
            ld_cnt  <= ld_cnt + 8'd1;
            rd_addr <= rd_addr - 12'd1;
          end
        end
        S_RUNA: st <= S_RUNW;    // settle cycle: wblk_sh/act_sh/exr_r stable
        S_RUNW: begin
          k_start <= 1'b1;
          st <= S_WAIT;
        end
        S_WAIT: begin
          // exp_r/flags_r still hold this record's expected value here; the
          // exp_addr/flags_addr update below lands one cycle later.
          if (k_done && !k_start) begin
            // golden cums[] reset per case; at a case boundary the chain is
            // 0.0 + partial == partial (bit-exact in IEEE), so compare the
            // partial directly and restart the chain from it.
            if (cs_r) begin
              sumf <= k_partial;
              if (k_partial != exp_r) begin
                if (!fail) fail_idx <= sb_idx;
                fail <= 1'b1;
              end
            end else begin
              sumf <= sumf_next;
              if (sumf_next != exp_r) begin
                if (!fail) fail_idx <= sb_idx;
                fail <= 1'b1;
              end
            end
            if (sb_idx == 5'd20) begin
              done_flag <= 1'b1;
              st <= S_DONE;
            end else begin
              sb_idx   <= sb_idx + 5'd1;
              exp_addr <= sb_idx + 5'd1;
              rd_addr  <= 12'(235 + sb_idx * 12'd118);   // next record's top word
              ld_cnt   <= 8'd0;
              st <= S_LOAD;
            end
          end
        end
        default: st <= S_DONE;
      endcase
    end
  end

  // wait-state guard: k_done pulses one cycle after the kernel's final add;
  // k_start must be low (it always is by S_WAIT) -- condition kept explicit.

  // ------------------------------------------------------------- LEDs -----
  wire run_leds  = ~done_flag;
  assign led[5] = done_flag ? (fail ? 1'b0 : 1'b1) : 1'b1;
  assign led[4] = done_flag ? fail : 1'b0;
  assign led[3:0] = done_flag ? (fail ? fail_idx[3:0] : 4'b1111) : sb_idx[3:0];
  // (run_leds kept for clarity only)

  // ------------------------------------------------------------- UART -----
  localparam integer UART_DIV = 234;    // 27 MHz / 234 ~= 115200 8N1

  logic        u_start, u_busy;
  logic [7:0]  u_data;
  uart_tx #(.DIV(UART_DIV)) utx (
    .clk(clk), .rst_n(rst_n),
    .start(u_start), .data(u_data), .tx(txd), .busy(u_busy)
  );

  // verdict line: pass  "G0 PASS\r\n"        (9 bytes)
  //               fail  "G0 FAIL nn\r\n"     (13 bytes, nn = fail_idx dec)
  function [7:0] hexnyb(input [3:0] n);
    hexnyb = (n < 10) ? (8'h30 + {4'b0, n}) : (8'h37 + {4'b0, n});
  endfunction
  function [7:0] tx_char(input fail_v, input [3:0] dig_hi, input [3:0] dig_lo,
                         input [3:0] p);
    case (p)
      4'd0: tx_char = "G";
      4'd1: tx_char = "0";
      4'd2: tx_char = " ";
      4'd3: tx_char = fail_v ? "F" : "P";
      4'd4: tx_char = "A";
      4'd5: tx_char = fail_v ? "I" : "S";
      4'd6: tx_char = fail_v ? "L" : "S";
      4'd7: tx_char = fail_v ? " " : "\r";
      4'd8: tx_char = fail_v ? hexnyb(dig_hi) : "\n";
      4'd9: tx_char = fail_v ? hexnyb(dig_lo) : 8'h00;
      4'd10: tx_char = "\r";
      4'd11: tx_char = "\n";
      default: tx_char = 8'h00;
    endcase
  endfunction

  logic [3:0] tx_ptr = 4'd0;
  logic       tx_run = 1'b0;
  logic [1:0] tx_st = 2'd0;   // 0=issue, 1=wait ack (busy), 2=wait done
  reg  [21:0] gap = 22'd0;    // ~1 s idle between repeated verdict lines
  wire [4:0]  fi = fail_idx;
  always @(posedge clk) begin
    if (!rst_n) begin
      tx_ptr <= 4'd0; tx_run <= 1'b0; tx_st <= 2'd0; gap <= 22'd0;
    end else if (st == S_DONE && !tx_run) begin
      if (gap != 22'd0) gap <= gap - 22'd1;
      else begin
        tx_ptr <= 4'd0; tx_run <= 1'b1; tx_st <= 2'd0;
      end
    end else if (tx_run) begin
      case (tx_st)
        2'd0: begin
          if (tx_ptr == (fail ? 4'd13 : 4'd9)) begin
            tx_run <= 1'b0;
            gap <= 22'h1FFFE;
          end else begin
            u_start <= 1'b1;
            u_data  <= tx_char(fail, 4'(fi / 4'd10), 4'(fi % 4'd10), tx_ptr);
            tx_ptr  <= tx_ptr + 4'd1;
            tx_st   <= 2'd1;
          end
        end
        2'd1: begin
          u_start <= 1'b0;
          if (u_busy) tx_st <= 2'd2;
        end
        default: begin
          if (!u_busy) tx_st <= 2'd0;
        end
      endcase
    end else u_start <= 1'b0;
  end

endmodule