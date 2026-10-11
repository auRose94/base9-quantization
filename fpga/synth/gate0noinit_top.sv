// On-silicon Gate-0, V2 shape: byte-streamed kernel against scratch memories.
//
// Per record: copy the record's 200 B wblk + 272 B act from the stimulus BRAM
// into small scratch memories (wm/am, word k = bytes 4k..4k+3), pre-register
// the 4 group + 8 block fp16 scales, then run k96_vec_dot_v2 (V2 consumes
// bytes in exact C-consumption order). Chain the fp32 partial into the
// running sumf with the same sequential f32_add the C kernel does, compare
// against the golden cumulative state per record (case-start resets from
// flags.hex), verdict on LEDs + UART with elapsed cycles in hex.
//
// Load loop timing: rd_addr leads the copy by one registered-read cycle, so
// each copy cycle (cp_cnt = c >= 1) writes stim[base + c - 1] into slot c-1;
// cp_cnt == 0 is the read warm-up. exp_r/flags are captured at load end from
// exp/flags memories already addressed for this record.
import f32_pkg::*;
module gate0noinit_top (
  input  wire        clk,        // 27 MHz
  input  wire        rst_b,      // active-low board reset (pull-up)
  output wire [5:0]  led,
  output wire        txd
);
  localparam integer N_RECS    = 21;
  localparam integer W_PER_REC = 118;   // 50 wblk + 68 act words

  reg [7:0] pwr = 8'd0;
  wire rst_n = pwr[7] & rst_b;
  always @(posedge clk) begin
    if (!rst_b) pwr <= 8'd0;
    else if (!pwr[7]) pwr <= pwr + 8'd1;
  end

  // ------------------------------------------------------- main stim ------
  reg [31:0] stim   [0:4095];
  reg [31:0] exp_mem [0:N_RECS-1];
  reg        flags_mem [0:N_RECS-1];    // 1 = record starts a new case chain
  reg [31:0] stim_q;
  reg [31:0] exp_q;
  reg        flags_q;
  always @(posedge clk) stim_q  <= stim[rd_addr];
  always @(posedge clk) exp_q   <= exp_mem[exp_addr];
  always @(posedge clk) flags_q <= flags_mem[exp_addr];

  reg [11:0] rd_addr  = 12'd0;
  reg [4:0]  exp_addr = 5'd0;

  // ------------------------------------------------------- scratch --------
  reg [31:0] wm [0:63];    // wblk bytes, word k = bytes 4k..4k+3
  reg [31:0] am [0:127];   // act bytes  (words past 67 = fetch slack)

  // pre-registered scales (LE fp16 bit patterns)
  reg [15:0] d0regs [0:3];
  reg [15:0] d1regs [0:7];

  reg [7:0]  cp_cnt = 8'd0;    // copy counter; slot < c-1 >, c == 0 warm-up
  reg [4:0]  sb_idx_r = 5'd0;
  reg [31:0] wm_q0, wm_q1, am_q0;
  reg [1:0]  wm_sel0, wm_sel1;     // sel captured together with the word
  reg        a_sel0;               // (address already advances next cycle)

  always @(posedge clk) begin
    wm_q0 <= wm[waddr0[7:2]]; wm_sel0 <= waddr0[1:0];
    wm_q1 <= wm[waddr1[7:2]]; wm_sel1 <= waddr1[1:0];
    am_q0 <= am[aaddr0[8:2]]; a_sel0 <= aaddr0[1];
    stim_q <= stim[rd_addr];
  end

  always @(posedge clk) begin
    if (!rst_n) begin
      d0regs[0] <= 16'd0; d0regs[1] <= 16'd0; d0regs[2] <= 16'd0; d0regs[3] <= 16'd0;
      d1regs[0] <= 16'd0; d1regs[1] <= 16'd0; d1regs[2] <= 16'd0; d1regs[3] <= 16'd0;
      d1regs[4] <= 16'd0; d1regs[5] <= 16'd0; d1regs[6] <= 16'd0; d1regs[7] <= 16'd0;
    end else if (st == S_LOAD && cp_cnt != 8'd0) begin
      case (cp_cnt - 8'd1)   // stim word index being written this cycle
        8'd0: begin d0regs[0] <= stim_q[15:0]; d0regs[1] <= stim_q[31:16]; end
        8'd1: begin d0regs[2] <= stim_q[15:0]; d0regs[3] <= stim_q[31:16]; end
        8'd50:  d1regs[0] <= stim_q[15:0];
        8'd58:  d1regs[1] <= stim_q[31:16];
        8'd67:  d1regs[2] <= stim_q[15:0];
        8'd75:  d1regs[3] <= stim_q[31:16];
        8'd84:  d1regs[4] <= stim_q[15:0];
        8'd92:  d1regs[5] <= stim_q[31:16];
        8'd101: d1regs[6] <= stim_q[15:0];
        8'd109: d1regs[7] <= stim_q[31:16];
        default: ;
      endcase
    end
  end

  // ------------------------------------------------------- kernel ---------
  logic [7:0]  waddr0, waddr1;
  logic [8:0]  aaddr0;
  logic [3:0]  sreq_addr;
  logic [7:0]  wdata0, wdata1;
  logic [15:0] adata0, sdata;
  logic        k_busy, k_done, k_start;
  logic [2:0]  k_st;
  logic [31:0] k_partial;

  always_comb begin
    case (wm_sel0)
      2'd0: wdata0 = wm_q0[7:0];
      2'd1: wdata0 = wm_q0[15:8];
      2'd2: wdata0 = wm_q0[23:16];
      2'd3: wdata0 = wm_q0[31:24];
    endcase
    case (wm_sel1)
      2'd0: wdata1 = wm_q1[7:0];
      2'd1: wdata1 = wm_q1[15:8];
      2'd2: wdata1 = wm_q1[23:16];
      2'd3: wdata1 = wm_q1[31:24];
    endcase
    adata0 = a_sel0 ? {am_q0[31:24], am_q0[23:16]}
                    : {am_q0[15:8],  am_q0[7:0]};
    sdata  = sreq_addr[3] ? d0regs[sreq_addr[1:0]] : d1regs[sreq_addr[2:0]];
  end

  k96_vec_dot_v2 kernel (
    .clk(clk), .rst_n(rst_n), .start(k_start),
    .waddr0(waddr0), .waddr1(waddr1), .aaddr0(aaddr0), .sreq_addr(sreq_addr),
    .wdata0(wdata0), .wdata1(wdata1), .adata0(adata0), .sdata(sdata),
    .busy(k_busy), .done(k_done), .k_st(k_st), .partial_bits(k_partial)
  );

  // ------------------------------------------------------- chain ----------
  logic [31:0] sumf = 32'd0;
  logic        fail = 1'b0;
  logic [4:0]  fail_idx = 5'd0;
  logic        done_flag = 1'b0;
  logic [31:0] exp_r;
  logic        cs_r;
  logic [31:0] cyc = 32'd0;

  localparam S_BOOT=3'd0, S_LOAD=3'd1, S_RUNA=3'd2, S_RUNW=3'd3,
             S_WAIT=3'd4, S_DONE=3'd5;
  reg [2:0] st = S_BOOT;

  logic [31:0] sumf_next;
  assign sumf_next = f32_pkg::f32_add(sumf, k_partial);

  always @(posedge clk) begin
    if (!rst_n) begin
      st <= S_BOOT; sb_idx_r <= 5'd0; sumf <= 32'd0; fail <= 1'b0;
      cp_cnt <= 8'd0; fail_idx <= 5'd0; done_flag <= 1'b0; k_start <= 1'b0;
      cyc <= 32'd0; rd_addr <= 12'd0; exp_addr <= 5'd0;
    end else begin
      if (st != S_BOOT && !done_flag) cyc <= cyc + 32'd1;
      k_start <= 1'b0;
      case (st)
        S_BOOT: begin
          sb_idx_r <= 5'd0;
          rd_addr  <= 12'd0;              // stim[0] read for the first copy
          exp_addr <= 5'd0;
          cp_cnt   <= 8'd0;
          st <= S_LOAD;
        end
        S_LOAD: begin
          // copy cycle cp_cnt = c: stim_q carries stim[base + c - 1] -> slot c-1
          if (cp_cnt != 8'd0) begin
            if (cp_cnt <= 8'd50)      wm[6'(cp_cnt - 8'd1)] <= stim_q;
            else if (cp_cnt <= 8'd118) am[7'(cp_cnt - 8'd51)] <= stim_q;
          end
          if (cp_cnt == 8'd118) begin
            exp_r <= exp_q;
            cs_r  <= flags_q;
            st <= S_RUNA;
          end else begin
            rd_addr  <= 12'(32'(sb_idx_r) * W_PER_REC + 32'(cp_cnt) + 32'd1);
            cp_cnt <= cp_cnt + 8'd1;
          end
        end
        S_RUNA: begin k_start <= 1'b1; st <= S_RUNW; end
        S_RUNW: st <= S_WAIT;
        S_WAIT: begin
          if (k_done && !k_start) begin
            if (cs_r) begin
              sumf <= k_partial;
              if (k_partial != exp_r) begin
                if (!fail) fail_idx <= sb_idx_r;
                fail <= 1'b1;
              end
            end else begin
              sumf <= sumf_next;
              if (sumf_next != exp_r) begin
                if (!fail) fail_idx <= sb_idx_r;
                fail <= 1'b1;
              end
            end
            if (sb_idx_r == 5'd20) begin
              done_flag <= 1'b1;
              st <= S_DONE;
            end else begin
              sb_idx_r <= sb_idx_r + 5'd1;
              exp_addr <= 5'(sb_idx_r) + 5'd1;
              rd_addr  <= 12'(32'(sb_idx_r + 5'd1) * W_PER_REC);
              cp_cnt   <= 8'd0;
              st <= S_LOAD;
            end
          end
        end
        default: st <= S_DONE;
      endcase
    end
  end

  // ------------------------------------------------------- LEDs -----------
  assign led[5] = done_flag ? (fail ? 1'b0 : 1'b1) : 1'b1;
  assign led[4] = done_flag ? fail : 1'b0;
  assign led[3:0] = done_flag ? (fail ? fail_idx[3:0] : 4'b1111) : sb_idx_r[3:0];

  // ------------------------------------------------------- UART -----------
  logic        u_start, u_busy;
  logic [7:0]  u_data;
  uart_tx #(.DIV(234)) utx (
    .clk(clk), .rst_n(rst_n),
    .start(u_start), .data(u_data), .tx(txd), .busy(u_busy)
  );

  // verdict line (17 chars): "G0 PASS" / "G0 FAIL" + 8 hex of cyc + CRLF
  function [7:0] hexnyb(input [3:0] n);
    hexnyb = (n < 10) ? (8'h30 + {4'b0, n}) : (8'h37 + {4'b0, n});
  endfunction
  function [7:0] tx_char(input [4:0] p);
    case (p)
      5'd0: tx_char = "G";
      5'd1: tx_char = "0";
      5'd2: tx_char = " ";
      5'd3: tx_char = fail ? "F" : "P";
      5'd4: tx_char = "A";
      5'd5: tx_char = fail ? "I" : "S";
      5'd6: tx_char = fail ? "L" : "S";
      5'd15: tx_char = "\r";
      5'd16: tx_char = "\n";
      default: tx_char = hexnyb(cyc[31 - 4 * (p - 5'd7) -: 4]);
    endcase
  endfunction

  logic [4:0] tx_ptr = 5'd0;
  logic       tx_run = 1'b0;
  logic [1:0] tx_st = 2'd0;
  reg  [21:0] gap = 22'd0;
  always @(posedge clk) begin
    if (!rst_n) begin
      tx_ptr <= 5'd0; tx_run <= 1'b0; tx_st <= 2'd0; gap <= 22'd0;
    end else if (st == S_DONE && !tx_run) begin
      if (gap != 22'd0) gap <= gap - 22'd1;
      else begin
        tx_ptr <= 5'd0; tx_run <= 1'b1; tx_st <= 2'd0;
      end
    end else if (tx_run) begin
      case (tx_st)
        2'd0: begin
          if (tx_ptr == 5'd17) begin
            tx_run <= 1'b0;
            gap    <= 22'h1FFFE;
          end else begin
            u_start <= 1'b1;
            u_data  <= tx_char(tx_ptr);
            tx_ptr  <= tx_ptr + 5'd1;
            tx_st   <= 2'd1;
          end
        end
        2'd1: begin
          u_start <= 1'b0;
          if (u_busy) tx_st <= 2'd2;
        end
        default: if (!u_busy) tx_st <= 2'd0;
      endcase
    end else u_start <= 1'b0;
  end
endmodule