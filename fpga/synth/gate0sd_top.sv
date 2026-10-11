// On-silicon Gate-0 with the stimulus resident on the microSD card (SPI mode).
//
// Clean-rewrite variant of gate0v2_top (the shape that synthesizes fully):
// boot = sd_spi_reader brings the card up (SDHC/SDXC SPI, CCS confirmed) and
// streams 20 blocks of the ninths image into the stimulus BRAM: words 0..2477
// = stim; inside LBA 19, words 46..66 = exp (21 words) and 67..87 = flags
// (21 x u32, bit0 used). Image layout: partner/ninths_sd_image.py. After boot
// the machinery equals gate0v2_top: per-record scratch copy, k96_vec_dot_v2
// in exact C-consumption order, fp32 chaining with per-case resets, compare
// vs the card-borne golden cums, LEDs + UART verdict. Card errors = walking
// blink on the LEDs and "G0 SERR" on the UART, no gate run.
//
// Memories carry a zero $readmemh init: the X-init variant prunes to a stub
// under synth_gowin (writes exist but the flow folds a no-init memory), while
// the boot overwrites every word the gate reads.
import f32_pkg::*;
module gate0sd_top (
  input  wire        clk,        // 27 MHz
  input  wire        rst_b,      // active-low board reset (pull-up)
  output wire [5:0]  led,
  output wire        txd,
  output wire        sd_cs_n,    // microSD slot: CS 81, CMD 82, CLK 83, DAT0 84
  output wire        sd_sck,
  output wire        sd_mosi,
  input  wire        sd_miso,
  input  wire        prov_btn     // user key (pin 87, pull-up, low = pressed)
);
  localparam integer N_RECS      = 21;
  localparam integer W_PER_REC   = 118;  // 50 wblk + 68 act words
  localparam integer SD_BASE_LBA = 2048; // payload partition 1 start

  // post-route Fmax of the SD-fed design is ~17 MHz; run the fabric at half
  // the 27 MHz oscillator through a BUFG (26% margin at 13.5 MHz).
  reg clkdiv = 1'b0;
  always @(posedge clk) clkdiv <= ~clkdiv;
  wire clk13;
  BUFG u_clkbuf (.I(clkdiv), .O(clk13));

  // sim shim: yosys/apicula map this cell to the fabric clock buffer
  `ifdef SIMULATION
   `endif

  reg [7:0] pwr = 8'd0;
  wire rst_n = pwr[7] & rst_b;
  always @(posedge clk) begin
    if (!rst_b) pwr <= 8'd0;
    else if (!pwr[7]) pwr <= pwr + 8'd1;
  end

  // ------------------------------------------------------- main stim ------
  reg [31:0] stim   [0:4095];
  reg [31:0] exp_mem [0:N_RECS-1];
  reg        flags_mem [0:N_RECS-1];   // 1 = record starts a new case chain
  reg [31:0] stim_q;
  reg [31:0] exp_q;
  reg        flags_q;
  always @(posedge clk13) stim_q  <= stim[rd_addr];
  always @(posedge clk13) exp_q   <= exp_mem[exp_addr];
  always @(posedge clk13) flags_q <= flags_mem[exp_addr];

  initial begin                // X-init pruning workaround: boot overwrites
    $readmemh("stim_zero.hex", stim);
    $readmemh("exp_zero.hex", exp_mem);
    $readmemh("flags_zero.hex", flags_mem);
  end
  // embedded ninths payload: 2560 words = 20 LBAs (the SD image verbatim:
  // stim words 0..2477, exp words 2478..2498, case-start flags 2499..2519,
  // zero padding to the block end). Source for provisioning AND the
  // byte-exact readback reference.
  reg [31:0] pay [0:2559];
  initial $readmemh("pay.hex", pay);

  reg [11:0] rd_addr  = 12'd0;
  reg [4:0]  exp_addr = 5'd0;

  // ------------------------------------------------------- SD reader ------
  reg        sd_start;
  reg [31:0] sd_lba;
  reg        sd_busy, sd_done, sd_error, sd_word_v, sd_rd_done;
  reg [6:0]  sd_blkw;                      // word index within block
  reg [5:0]  sd_blk;
  reg [31:0] sd_word;
  reg [23:0] boot_to;
  reg        card_err, sd_pend;
  reg [3:0]  err_code;       // 0 ok; 1 init; 2 write; 3 verify; 4 watchdog
  reg        sd_done_seen;   // sticky: never miss the reader's init-done pulse       // 0 ok; 1 init; 2 write; 3 verify; 4 watchdog

  wire [31:0] sd_word_r, sd_wdata;
  wire        sd_word_v_r, sd_rd_done_r, sd_done_r, sd_error_r, sd_busy_r;
  wire [5:0]  rd_st;
  wire        sd_wdata_req;
  reg        sd_wpend;                   // write request pending (level)
  reg [11:0] pay_a;                      // payload read address (shared)
  reg [31:0] pay_q;
  reg        prov;                       // provisioning requested (button)
  reg [5:0]  prov_blk;                   // blocks written this run
  reg        provv_err;

  sd_spi_reader sdr (
    .clk(clk13), .rst_n(rst_n),
    .spi_cs_n(sd_cs_n), .miso(sd_miso), .sck(sd_sck), .mosi(sd_mosi),
    .start(sd_start), .read_req(sd_pend), .write_req(sd_wpend), .lba(sd_lba),
    .wdata_req(sd_wdata_req), .wdata(sd_wdata),
    .busy(sd_busy_r), .done(sd_done_r), .error(sd_error_r),
    .word_valid(sd_word_v_r), .word_out(sd_word_r), .rd_done(sd_rd_done_r),
    .dbg_st(rd_st)
  );

  assign sd_wdata = pay_q;   // written-word stream (stable 4 exchanges)

  // payload read port: during provisioning the write stream needs the next
  // word when wdata_req arrives; during fill/verify the address tracks the
  // arriving card words
  always @(posedge clk13) begin
    if (st == S_PROV) begin
      if (sd_wdata_req) pay_q <= pay[wpay];
    end else if (st == S_SDFILL && sd_word_v_r) begin
      pay_q <= pay[pay_a];
    end
  end

  // reader outputs are registered once here (the fill logic then sees them
  // one cycle late -- deliberately simple, the fill is not speed-critical)
  always @(posedge clk13) begin
    if (!rst_n) begin
      sd_word_v <= 1'b0; sd_rd_done <= 1'b0; sd_done <= 1'b0; sd_error <= 1'b0;
      sd_word <= 32'd0; sd_busy <= 1'b0;
    end else begin
      sd_word_v  <= sd_word_v_r;
      sd_rd_done <= sd_rd_done_r;
      sd_done    <= sd_done_r;
      sd_error   <= sd_error_r;
      sd_busy    <= sd_busy_r;
      sd_word    <= sd_word_r;
      if (sd_done_r) sd_done_seen <= 1'b1;
    end
  end

  // ------------------------------------------------------- scratch --------
  reg [31:0] wm [0:63];    // wblk bytes, word k = bytes 4k..4k+3
  reg [31:0] am [0:127];   // act bytes  (words past 67 = fetch slack)
  reg [15:0] d0regs [0:3];
  reg [15:0] d1regs [0:7];
  reg [31:0] wm_q0, wm_q1, am_q0;
  reg [1:0]  wm_sel0, wm_sel1;
  reg        a_sel0;
  reg [7:0]  cp_cnt = 8'd0;    // copy counter; slot c-1, c == 0 warm-up
  reg [4:0]  sb_idx_r = 5'd0;

  always @(posedge clk13) begin
    wm_q0 <= wm[waddr0[7:2]]; wm_sel0 <= waddr0[1:0];
    wm_q1 <= wm[waddr1[7:2]]; wm_sel1 <= waddr1[1:0];
    am_q0 <= am[aaddr0[8:2]]; a_sel0 <= aaddr0[1];
    stim_q <= stim[rd_addr];
  end

  always @(posedge clk13) begin
    if (!rst_n) begin
      d0regs[0] <= 16'd0; d0regs[1] <= 16'd0; d0regs[2] <= 16'd0; d0regs[3] <= 16'd0;
      d1regs[0] <= 16'd0; d1regs[1] <= 16'd0; d1regs[2] <= 16'd0; d1regs[3] <= 16'd0;
      d1regs[4] <= 16'd0; d1regs[5] <= 16'd0; d1regs[6] <= 16'd0; d1regs[7] <= 16'd0;
    end else if (st == S_LOAD && cp_cnt != 8'd0) begin
      case (cp_cnt - 8'd1)
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
    .clk(clk13), .rst_n(rst_n), .start(k_start),
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

  localparam S_BOOT=4'd0, S_SDBOOT=4'd1, S_SDFILL=4'd2, S_LOAD=4'd3,
             S_RUNA=4'd4, S_RUNW=4'd5, S_WAIT=4'd6, S_DONE=4'd7,
             S_PROV=4'd8;
  reg [3:0] st = S_BOOT;
  reg [3:0] prov_ph = S_PROV;

  logic [31:0] sumf_next;
  assign sumf_next = f32_pkg::f32_add(sumf, k_partial);

  // SD fill bookkeeping (the reader's outputs are 1 cycle delayed here)
  reg [31:0] sd_wr;
  reg [11:0] wpay;                     // payload word index while writing
  always @(posedge clk13) begin
    if (!rst_n) begin
      sd_start  <= 1'b0;
      sd_done_seen <= 1'b0;
      sd_wr     <= 32'd0;
      sd_blkw   <= 7'd0;
      sd_blk    <= 6'd0;
      boot_to   <= 24'd0;
      card_err  <= 1'b0;
      err_code  <= 4'd0;
      sd_pend   <= 1'b0;
      sd_lba    <= 32'(SD_BASE_LBA);
      prov      <= 1'b0;
      prov_blk  <= 6'd0;
      provv_err <= 1'b0;
      sd_wpend  <= 1'b0;
      wpay      <= 12'd0;
      pay_a     <= 12'd0;
      pay_q     <= 32'd0;
    end else begin
      if (sd_busy_r && sd_start)              // reader captured the start
        sd_start <= 1'b0;
      case (st)
        S_SDBOOT: boot_to <= boot_to + 24'd1;
        S_SDFILL: begin
          if (sd_busy && sd_pend) begin    // reader captured the read request
            sd_pend <= 1'b0;
            sd_lba  <= sd_lba + 32'd1;
          end
        end
        S_PROV: begin
          if (sd_busy && sd_wpend) begin   // reader captured the write request
            sd_wpend <= 1'b0;
            sd_lba   <= sd_lba + 32'd1;
            wpay     <= 12'(32'(prov_blk) * 32'd128);   // block k starts at k*128
          end
          if (sd_wdata_req) begin          // caller side of the write stream
            pay_q <= pay[wpay];
            wpay <= wpay + 12'd1;
          end
        end
        default: ;
      endcase
    end
  end

  always @(posedge clk13) begin
    if (!rst_n) begin
      st <= S_BOOT; sb_idx_r <= 5'd0; sumf <= 32'd0; fail <= 1'b0;
      cp_cnt <= 8'd0; fail_idx <= 5'd0; done_flag <= 1'b0; k_start <= 1'b0;
      cyc <= 32'd0; rd_addr <= 12'd0; exp_addr <= 5'd0;
    end else begin
      if (st != S_BOOT && st != S_SDBOOT && st != S_SDFILL && !done_flag)
        cyc <= cyc + 32'd1;
      k_start <= 1'b0;
      case (st)
        S_BOOT: begin                     // bring the card up
          sd_start <= 1'b1;
          rd_addr  <= 12'd0;
          exp_addr <= 5'd0;
          cp_cnt   <= 8'd0;
          st <= S_SDBOOT;
        end
        S_SDBOOT: begin
          boot_to <= boot_to + 24'd1;
          if (boot_to >= 24'd810000000) begin  // 30 s watchdog (stuck card)
            card_err <= 1'b1;
            err_code <= 4'd4;
            st <= S_DONE;
          end else if (sd_done_seen) begin     // sticky init-done
            sd_done_seen <= 1'b0;
            if (sd_error) begin
              card_err <= 1'b1;
              err_code <= 4'd1;
              st <= S_DONE;
            end else begin
              prov <= !prov_btn;              // key held = provision this boot
              prov_blk <= 6'd0;
              provv_err <= 1'b0;
              sd_blk  <= 6'd0;
              sd_blkw <= 7'd0;
              sd_wr   <= 32'd0;
              sd_lba  <= 32'(SD_BASE_LBA);
              wpay    <= 12'd0;
              pay_a   <= 12'd0;
              if (!prov_btn) begin
                sd_wpend <= 1'b1;
                st <= S_PROV;
              end else begin
                sd_pend <= 1'b1;
                st <= S_SDFILL;
              end
            end
          end
        end
        S_PROV: begin
          if (sd_error) begin                 // write failed (reader fatal)
            card_err <= 1'b1;
            err_code <= 4'd2;
            st <= S_DONE;
          end else if (sd_rd_done) begin      // one block written
            prov_blk <= prov_blk + 6'd1;
            if (prov_blk == 6'd19) begin
              sd_blk  <= 6'd0;
              sd_blkw <= 7'd0;
              sd_wr   <= 32'd0;
              sd_lba  <= 32'(SD_BASE_LBA);
              pay_a   <= 12'd0;
              sd_pend <= 1'b1;              // now read back + verify
              st <= S_SDFILL;
            end else begin
              sd_wpend <= 1'b1;
            end
          end
        end
        S_SDFILL: begin
          // payload read address tracks the arriving stream (registered once)
          if (sd_word_v_r) pay_a <= pay_a + 12'd1;
          if (sd_word_v && pay_q != sd_word)
            provv_err <= 1'b1;              // card roundtrip is not byte-exact
          if (sd_word_v) begin              // reader pulses one word/cycle
            case (sd_blk)
              6'd19: begin                  // stim tail + exp + flags
                if (sd_blkw < 7'd46)
                  stim[12'(32'd2432 + 32'(sd_blkw))] <= sd_word;
                else if (sd_blkw < 7'd67)
                  exp_mem[5'(sd_blkw - 7'd46)] <= sd_word;
                else if (sd_blkw < 7'd88)
                  flags_mem[5'(sd_blkw - 7'd67)] <= sd_word[0];
              end
              default:
                stim[12'(32'(sd_blk) * 32'd128 + 32'(sd_blkw))] <= sd_word;
            endcase
            sd_wr   <= sd_wr + 32'd1;
            sd_blkw <= (sd_blkw == 7'd127) ? 7'd0 : (sd_blkw + 7'd1);
            if (sd_blkw == 7'd127) sd_blk <= sd_blk + 6'd1;
          end
          if (sd_rd_done) begin
            if (sd_blk >= 6'd20) begin     // 20 blocks resident
              if (provv_err) begin
                card_err <= 1'b1;          // card != embedded payload
                err_code <= 4'd3;
                st <= S_DONE;
              end else st <= S_LOAD;
            end else begin
              sd_pend <= 1'b1;             // level request for the next block
            end
          end
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
        S_DONE: ;
        default: st <= S_DONE;
      endcase
    end
  end

  // ------------------------------------------------------- LEDs -----------
  reg [5:0] blink = 6'b000001;
  reg [23:0] bcnt = 24'd0;
  always @(posedge clk13) begin
    if (!rst_n) begin
      blink <= 6'b000001; bcnt <= 24'd0;
    end else begin
      bcnt <= bcnt + 24'd1;
      if (bcnt == 24'd6000000) begin       // ~222 ms
        bcnt <= 24'd0;
        blink <= {blink[4:0], blink[5]};
      end
    end
  end
  // LED telegraph (pads ACTIVE-LOW: 1 in lit_mask = LED lit):
  //   boot / provision / fill -> led[5], led[4] lit + the four index LEDs
  //                              show WHERE the card path is: reader FSM
  //                              state nibble (init), provisioned-block or
  //                              streamed-block index
  //   gate run                -> led[5] lit, led[4] dark, index = record
  //   G0 PASS                 -> led[5] + four index LEDs lit (5 lit)
  //   G0 FAIL                 -> led[5] blinks (fail_idx+2) slowly, the four
  //                              index LEDs hold the failing record
  //   card errors             -> walking light + binary error code on the
  //                              index LEDs (1 init, 2 write, 3 roundtrip, 4 watchdog)
  reg [23:0] bt = 24'd0;
  always @(posedge clk13) bt <= (bt == 24'd405000) ? 24'd0 : bt + 24'd1;
  wire beat_on = (bt < 24'd135000);            // 10 ms lit per 30 ms beat
  reg [3:0]  bcn_i = 4'd0;
  wire [3:0] blinks_tot = 4'(fail_idx + 5'd2);  // every fail blinks >= 2
  always @(posedge clk13) begin
    if (st != S_DONE || !fail) bcn_i <= 4'd0;
    else if (bt == 24'd0) bcn_i <= bcn_i + 4'd1;
  end
  wire beacon_on = done_flag && fail && (bcn_i < blinks_tot) && beat_on;

  wire [5:0] lit_mask =
    card_err ?      (blink | {2'b00, err_code[3:0]}) :
    st == S_PROV   ? {2'b11, rd_st[3:0]} :   // stuck-write state readable
    st == S_SDFILL ? {2'b11, rd_st[3:0]} :
    st == S_SDBOOT ? {2'b11, rd_st[3:0]} :   // init telegraph: reader FSM
    st == S_BOOT   ? {2'b11, 4'd0} :
    done_flag      ? (fail ? {1'b0, beacon_on, fail_idx[3:0]}
                           : {1'b1, 1'b0, 4'b1111})
                   : {1'b1, 1'b0, sb_idx_r[3:0]};
  assign led = ~lit_mask;

  // ------------------------------------------------------- UART -----------
  logic        u_start, u_busy;
  logic [7:0]  u_data;
  uart_tx #(.DIV(117)) utx (   // 13.5 MHz / 117 = 115384 baud
    .clk(clk13), .rst_n(rst_n),
    .start(u_start), .data(u_data), .tx(txd), .busy(u_busy)
  );

  // verdict: "G0 PASS"/"G0 FAIL"/"G0 SERR" + 8 hex of cyc + CRLF (17 chars)
  function [7:0] hexnyb(input [3:0] n);
    hexnyb = (n < 10) ? (8'h30 + {4'b0, n}) : (8'h37 + {4'b0, n});
  endfunction
  function [7:0] tx_char(input [4:0] p);
    case (p)
      5'd0: tx_char = "G";
      5'd1: tx_char = "0";
      5'd2: tx_char = " ";
      5'd3: tx_char = card_err ? "S" : fail ? "F" : "P";
      5'd4: tx_char = card_err ? "E" : "A";
      5'd5: tx_char = card_err ? "R" : fail ? "I" : "S";
      5'd6: tx_char = card_err ? "R" : fail ? "L" : "S";
      5'd15: tx_char = "\r";
      5'd16: tx_char = "\n";
      default: tx_char = hexnyb(cyc[31 - 4 * (p - 5'd7) -: 4]);
    endcase
  endfunction

  logic [4:0] tx_ptr = 5'd0;
  logic       tx_run = 1'b0;
  logic [1:0] tx_st = 2'd0;
  reg  [21:0] gap = 22'd0;
  always @(posedge clk13) begin
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
