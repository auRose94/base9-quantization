// SDRAM bring-up gate (A2): die init, geometry probe, pattern walk, hold
// test, verdict over the LED telegraph + UART — the session-29 SD bring-up
// protocol applied to the in-package die. Board-SD-independent: canaries are
// constants and the walk pattern is a keyed per-address xorshift generated
// in fabric, so this runs regardless of the microSD's provision state.
//
// Phases (index LEDs show the phase while running):
//   1 PROB  10 canary word addresses (distinguishing corners of geometry
//           hypothesis H1 = 2 banks x 2048 rows x 512 cols) written, then
//           read back + compared against constants.
//   2 DUMP  the ten canary READBACKS dump on the UART once ("k0=0x????????")
//           — offline these decode the die's real geometry.
//   3 WRIT  32K words written as two sweeps (bank0 rows + bank1 rows under
//           H1), pattern = per-address xorshift, mirrored EXACTLY in
//           sim/tb_sdram.py.
//   4 READ  the sweeps read back, expected recomputed per address, first
//           mismatch recorded.
//   5 HOLD  an idle span (2 s silicon / 3000 cycles sim), banks closed —
//           refresh carries the content; then re-read + compare.
// Telegraph (pads ACTIVE-LOW; "1 in lit_mask = lit"):
//   run: led[5],led[4] lit, index LEDs = phase 1..6; PASS: led[5] + index
//   LEDs lit, led[4] dark (5 lit) and led[0] blinking = the 27 MHz-domain
//   probe (board-alive signature); FAIL: led[5] blinks (code+2) per beat,
//   index LEDs hold the failing address's low nibble. Codes: 1 probe,
//   2 walk, 3 hold, 4 watchdog (30 s). led[0] is fed from the 27 MHz pad
//   clock alone (blip) — no led pin crosses domains.
// UART: probe dump pass first, a live 4-byte trace frame (0x55, phase,
//   controller state, word low bits) every ~20 ms while running, then a
//   repeating 27-char verdict line:
//     "SD PASS a=0 1234ABCD" / "SD FAIL c=n a=????????" + 8 hex cyc + CRLF
//   (verdict layout: "SD PASS " or "SD FAIL ", "c=<digit> a=<8 hex> <8 hex
//   cyc>" where the first hex = fail_addr, second = cycles).
// Reset pattern = dbg3-v2: pwr is a pure power-on free-runner; rst_b is
// consumed NOWHERE (S1 jam-safe; the UPDATE button = effective reset).
module sdram_top #(
  parameter integer HOLD_CYC = 27000000   // 2.0 s @ 13.5 MHz; tb passes -G
) (
  input  wire        clk,        // 27 MHz
  input  wire        rst_b,      // pin 88 — deliberately UNUSED below
  output wire [5:0]  led,
  output wire        txd,
  output wire        O_sdram_clk,
  output wire        O_sdram_cke,
  output wire        O_sdram_cs_n,
  output wire        O_sdram_ras_n,
  output wire        O_sdram_cas_n,
  output wire        O_sdram_wen_n,
  output wire [10:0] O_sdram_addr,
  output wire [1:0]  O_sdram_ba,
  output wire [3:0]  O_sdram_dqm,
  `ifdef SIMULATION
  input  wire [31:0] SIM_dq_in,  // tb injects the die's DQ here (Verilator
                                 // cannot model the top-level tristate + two
                                 // drivers faithfully; synthesis keeps the
                                 // real IO_sdram_dq inout below, unconnected)
  `endif
  inout  wire [31:0] IO_sdram_dq
);

  // ── clocks ────────────────────────────────────────────────────────────────
  // divc[0] = 13.5 MHz = SDRAM/UART domain (same plain-counter MSB -> BUFG
  // shape that fixed the cont-8 dead-clock silicon bug; no CE gates).
  reg [1:0] divc = 2'd0;
  always @(posedge clk) divc <= divc + 2'd1;
  wire clk_sd;
  BUFG u_bufg_sd (.I(divc[0]), .O(clk_sd));

  reg [7:0] pwr = 8'd0;        // power-on free-runner (never reset-gated)
  always @(posedge clk) if (!pwr[7]) pwr <= pwr + 8'd1;

  // 27 MHz-domain alive probe: ~0.5 Hz blip on led[0]
  reg p27 = 1'b0;
  reg [26:0] c27 = 27'd0;
  always @(posedge clk) begin
    c27 <= (c27 == 27'd13499999) ? 27'd0 : c27 + 27'd1;
    if (c27 == 27'd13499999) p27 <= ~p27;
  end

  // ── DQ bus + controller ───────────────────────────────────────────────────
  `ifdef SIMULATION
  wire [31:0] dq_in = SIM_dq_in;
  `else
  wire [31:0] dq_in = IO_sdram_dq;
  assign IO_sdram_dq = dq_oe ? dq_out : 32'bz;
  `endif
  wire [31:0] dq_out;
  wire        dq_oe;
  assign O_sdram_clk = clk_sd;   // die clock forwarded (v1: BUFG net direct)

  logic        req, wr_n, busy, ready, rdata_v;
  logic [20:0] addr;
  logic [31:0] wdata, rdata;
  logic [7:0]  c_st;

  sdram_ctrl #(.BANK_LOG2(1), .ROW_LOG2(11), .COL_LOG2(9)) u_sdram (
    .clk(clk_sd), .rst_n(pwr[7]),
    .sd_cke(O_sdram_cke), .sd_cs_n(O_sdram_cs_n), .sd_ras_n(O_sdram_ras_n),
    .sd_cas_n(O_sdram_cas_n), .sd_we_n(O_sdram_wen_n), .sd_addr(O_sdram_addr),
    .sd_ba(O_sdram_ba), .sd_dqm(O_sdram_dqm),
    .dq_out(dq_out), .dq_oe(dq_oe), .dq_in(dq_in),
    .req(req), .wr_n(wr_n), .addr(addr), .wdata(wdata),
    .rdata(rdata), .rdata_v(rdata_v), .busy(busy), .ready(ready),
    .st(c_st)
  );

  // ── expected pattern — MIRRORED EXACTLY in sim/tb_sdram.py ────────────────
  // x = (w ^ 0x5A5A5A5A); x ^= x<<13; x ^= x>>17; x ^= x<<5; pat = x ^ 0x1234ABCD
  function [31:0] exp_pat(input [20:0] w);
    logic [31:0] x;
    begin
      x = 32'(w) ^ 32'h5A5A5A5A;
      x = x ^ {x[18:0], 13'b0};    // x ^= x << 13
      x = x ^ {17'b0, x[31:17]};   // x ^= x >> 17
      x = x ^ {x[26:0], 5'b0};     // x ^= x << 5
      exp_pat = x ^ 32'h1234ABCD;
    end
  endfunction

  // ── probes: constants mirrored in the tb ─────────────────────────────────
  localparam integer N_CANARY = 10;
  function [20:0] canary_addr(input [3:0] i);
    case (i)
      4'd0: canary_addr = 21'd0;        4'd1: canary_addr = 21'd1;
      4'd2: canary_addr = 21'd2;        4'd3: canary_addr = 21'd3;
      4'd4: canary_addr = 21'd512;      4'd5: canary_addr = 21'd1024;
      4'd6: canary_addr = 21'd65536;    // H1: row 128, col 0
      4'd7: canary_addr = 21'd1048576;  // H1: bank 1 start (w >> 20)
      4'd8: canary_addr = 21'd1048577;  // H1: bank 1, col 1
      4'd9: canary_addr = 21'd2097151;  // H1: last word
      default: canary_addr = 21'd0;
    endcase
  endfunction
  function [31:0] canary_val(input [3:0] i);
    canary_val = 32'hCAFE0000 + {28'd0, i};
  endfunction
  localparam integer WALK_WORDS = 16384;
  function [20:0] walk_addr(input half, input [13:0] ii);
    walk_addr = half ? (21'h100000 + 21'(32'(ii))) : 21'(32'(ii));
  endfunction

  // ── test FSM ─────────────────────────────────────────────────────────────
  localparam P_BOOT=4'd0, P_PROB_W=4'd1, P_PROB_R=4'd2, P_DUMP=4'd3,
             P_WALK_W=4'd4, P_WALK_R=4'd5, P_HOLD=4'd6, P_HOLD_R=4'd7,
             P_DONE=4'd8;
  reg [3:0]  ph = P_BOOT;
  reg [3:0]  ci;                  // canary index
  reg [13:0] wi;                  // word index within both sweeps
  reg        half;                // 0 = bank0 sweep, 1 = bank1 sweep
  reg        req_r = 1'b0;
  reg        wr_req = 1'b0;
  reg [20:0] req_a = 21'd0;
  reg [31:0] req_d = 32'd0;       // driven wdata (write phases recompute)
  reg [15:0] mism = 16'd0;
  reg [20:0] fail_addr = 21'd0;
  reg [3:0]  err_code = 4'd0;     // 1 probe, 2 walk, 3 hold, 4 watchdog
  reg        err_seen = 1'b0;
  reg [25:0] hold_cnt = 26'd0;
  reg [31:0] wd_cnt = 32'd0;
  reg [31:0] cyc = 32'd0;
  wire       done_flag = (ph == P_DONE);
  wire       fail = err_seen;

  reg [31:0] canary_q0,  canary_q1,  canary_q2,  canary_q3,  canary_q4;
  reg [31:0] canary_q5,  canary_q6,  canary_q7,  canary_q8,  canary_q9;
  // (ten plain regs, not an array: no BRAM inference, no runtime-write gotcha)

  // HOLD_CYC = module parameter (default 2.0 s @ 13.5 MHz; the tb passes
  // -GHOLD_CYC=3000; keep it under the real-time 64 ms refresh boundary per phase).

  assign req   = req_r;
  assign wr_n  = wr_req;
  assign addr  = req_a;
  assign wdata = req_d;

  // advance-then-fire: the request-driver must never share an edge with the
  // index-advance (both read pre-edge state -> stale indices). Scheme: the
  // advance runs on the ready pulse's first cycle (done_p) or on a phase-
  // path entry, and SETS adv_mark; the next request fires ONE CYCLE LATER
  // (reading only committed state) and clears the mark.
  reg done_p = 1'b0;
  reg adv_mark = 1'b0;

  always @(posedge clk_sd) begin
    if (!pwr[7]) begin
      req_r <= 1'b0; done_p <= 1'b0; adv_mark <= 1'b0;
    end else begin
      done_p  <= ready;

      // (1) advance on the completion pulse (pre-edge state = pre-txn state)
      if (done_p) begin
        adv_mark <= 1'b1;        // arm the next request (fires one cycle later)
        case (ph)
          P_BOOT: begin                  // init handshake done: start probing
            ci <= 4'd0; wi <= 14'd0; half <= 1'b0;
            mism <= 16'd0; err_seen <= 1'b0; err_code <= 4'd0;
            adv_mark <= 1'b1;
            ph <= P_PROB_W;
          end
          P_PROB_W: begin
            ci <= ci + 4'd1;
            if (ci == 4'(N_CANARY) - 4'd1) begin ci <= 4'd0; ph <= P_PROB_R; end
          end
          P_PROB_R: begin
            ci <= ci + 4'd1;
            if (ci == 4'(N_CANARY) - 4'd1) ph <= P_DUMP;
          end
          P_WALK_W: begin
            wi <= wi + 14'd1;
            if (wi == 14'(WALK_WORDS) - 14'd1) begin
              wi <= 14'd0;
              if (half) begin ph <= P_WALK_R; half <= 1'b0; end
              else half <= 1'b1;
            end
          end
          P_WALK_R: begin
            wi <= wi + 14'd1;
            if (wi == 14'(WALK_WORDS) - 14'd1) begin
              wi <= 14'd0;
              if (half) ph <= P_HOLD; else half <= 1'b1;
            end
          end
          P_HOLD_R: begin
            wi <= wi + 14'd1;
            if (wi == 14'(WALK_WORDS) - 14'd1) begin
              wi <= 14'd0;
              if (half) ph <= P_DONE;       // verdict reflects err_seen/mism
              else half <= 1'b1;
            end
          end
          default: ;
        endcase
      end

      // (2) fire the next request one cycle after an advance/entry commit
      if (adv_mark && !done_p && req_r == 1'b0 && busy == 1'b0) begin
        adv_mark <= 1'b0;      // cleared with the fire (args = committed state)
        case (ph)
          P_PROB_W, P_PROB_R: begin
            req_r <= 1'b1;
            wr_req <= !(ph == P_PROB_W);       // wr_n: 0 = write
            req_a <= canary_addr(ci);
            req_d <= canary_val(ci);
          end
          P_WALK_W: begin
            req_r <= 1'b1; wr_req <= 1'b0;
            req_a <= walk_addr(half, wi);
            req_d <= exp_pat(walk_addr(half, wi));
          end
          P_WALK_R, P_HOLD_R: begin
            req_r <= 1'b1; wr_req <= 1'b1;     // wr_n = 1 = read
            req_a <= walk_addr(half, wi);
          end
          default: ;
        endcase
      end
      if (req_r && busy) req_r <= 1'b0;   // controller latched: release level

      // transaction service
      if (rdata_v) begin
        if (ph == P_PROB_R) begin
          case (ci)
            4'd0: canary_q0 <= rdata; 4'd1: canary_q1 <= rdata;
            4'd2: canary_q2 <= rdata; 4'd3: canary_q3 <= rdata;
            4'd4: canary_q4 <= rdata; 4'd5: canary_q5 <= rdata;
            4'd6: canary_q6 <= rdata; 4'd7: canary_q7 <= rdata;
            4'd8: canary_q8 <= rdata; default: canary_q9 <= rdata;
          endcase
          if (rdata != canary_val(ci)) begin
            if (!err_seen) begin
              err_seen <= 1'b1; err_code <= 4'd1;
              fail_addr <= 21'(32'(ci));
            end
          end
        end
        if ((ph == P_WALK_R || ph == P_HOLD_R) && rdata != exp_pat(req_a)) begin
          if (!err_seen) begin
            err_seen <= 1'b1; err_code <= (ph == P_WALK_R) ? 4'd2 : 4'd3;
            fail_addr <= req_a;
          end
          mism <= mism + 16'd1;
        end
      end

      // non-transaction transitions
      case (ph)
        P_DUMP: if (dump_done) begin
          ph <= P_WALK_W; wi <= 14'd0; half <= 1'b0; adv_mark <= 1'b1;
        end
        P_HOLD: begin
          hold_cnt <= hold_cnt + 26'd1;
          if (hold_cnt == 26'(HOLD_CYC)) begin
            hold_cnt <= 26'd0;
            wi <= 14'd0; half <= 1'b0;
            adv_mark <= 1'b1;
            ph <= P_HOLD_R;
          end
        end
        default: ;
      endcase

      // watchdog: 30 s total; any stall = code 4
      wd_cnt <= wd_cnt + 32'd1;
      if (wd_cnt == 32'd405000000) begin
        if (!err_seen) begin
          err_seen <= 1'b1; err_code <= 4'd4; fail_addr <= 21'd0;
        end
        ph <= P_DONE;
      end
      if (ph != P_BOOT) cyc <= cyc + 32'd1;
    end
  end

  // (dump_done is a reg declared with the UART block below)

  // ── LEDs ─────────────────────────────────────────────────────────────────
  reg [5:0] blink_sd = 6'b000001;
  reg [23:0] bcnt = 24'd0;
  always @(posedge clk_sd) begin
    bcnt <= (bcnt == 24'd3374999) ? 24'd0 : bcnt + 24'd1;   // ~250 ms walk
    if (bcnt == 24'd3374999) blink_sd <= {blink_sd[4:0], blink_sd[5]};
  end
  reg [23:0] bt = 24'd0;    // 30 ms beats (SD-top convention)
  always @(posedge clk_sd) bt <= (bt >= 24'd404999) ? 24'd0 : bt + 24'd1;
  wire beat_on = (bt < 24'd135000);      // 10 ms lit per 30 ms beat
  reg [3:0] bcn_i = 4'd0;
  wire [3:0] blinks_tot = 4'(err_code + 4'd2);
  always @(posedge clk_sd) begin
    if (!done_flag || !fail) bcn_i <= 4'd0;
    else if (bt == 24'd0) bcn_i <= bcn_i + 4'd1;
  end
  wire beacon_on = done_flag && fail && (bcn_i < blinks_tot) && beat_on;

  wire [3:0] run_idx =
    (ph == P_BOOT)                      ? 4'd0 :
    (ph == P_PROB_W || ph == P_PROB_R)  ? 4'd1 :
    (ph == P_DUMP)                      ? 4'd2 :
    (ph == P_WALK_W || ph == P_WALK_R)  ? 4'd3 :
    (ph == P_HOLD)                      ? 4'd5 :
    (ph == P_HOLD_R)                    ? 4'd6 : 4'd7;
  wire [5:0] lit_mask =
    done_flag ? (fail ? {1'b0, beacon_on, fail_addr[3:0]}
                      : {1'b1, 1'b0, 4'b1111})
              : {1'b1, 1'b1, run_idx};
  // led[0] = 27 MHz probe; led[5:1] = telegraph (ACTIVE-LOW pads: invert)
  assign led = {~lit_mask[5], ~lit_mask[4], ~lit_mask[3],
                ~lit_mask[2], ~lit_mask[1], p27};

  // ── UART: dump pass, live trace, verdict loop ────────────────────────────
  uart_tx #(.DIV(117)) utx (    // 13.5 MHz / 117 = 115384 baud
    .clk(clk_sd), .rst_n(pwr[7]),
    .start(u_start), .data(u_data), .tx(txd), .busy(u_busy)
  );
  logic       u_start, u_busy;
  logic [7:0] u_data;

  reg  [3:0] dump_i = 4'd0;
  reg        dump_done = 1'b0;    // tx FSM writes; test FSM reads (same domain)
  function [31:0] canary_word(input [3:0] i);
    case (i)
      4'd0: canary_word = canary_q0;  4'd1: canary_word = canary_q1;
      4'd2: canary_word = canary_q2;  4'd3: canary_word = canary_q3;
      4'd4: canary_word = canary_q4;  4'd5: canary_word = canary_q5;
      4'd6: canary_word = canary_q6;  4'd7: canary_word = canary_q7;
      4'd8: canary_word = canary_q8;  default: canary_word = canary_q9;
    endcase
  endfunction
  function [7:0] hexnyb(input [3:0] n);
    hexnyb = (n < 10) ? (8'h30 + {4'b0, n}) : (8'h37 + {4'b0, n});
  endfunction

  // dump line (15 ch): "kX=0x????????\r\n"
  function [7:0] dump_char(input [5:0] p);
    case (p)
      6'd0: dump_char = "k";
      6'd1: dump_char = hexnyb(dump_i[3:0]);
      6'd2: dump_char = "=";
      6'd3: dump_char = "0";
      6'd4: dump_char = "x";
      6'd5, 6'd6, 6'd7, 6'd8, 6'd9, 6'd10, 6'd11, 6'd12: begin
        logic [31:0] cw;                        // yosys: no part-select on a
        cw = canary_word(dump_i[3:0]);          // call result — round-trip a local
        dump_char = hexnyb(cw[31 - 4*(p - 6'd5) -: 4]);
      end
      6'd13: dump_char = "\r";
      default: dump_char = "\n";
    endcase
  endfunction
  // verdict (29 ch, DONE): "SD PASS a=00000000 0123ABCD\r\n"
  // ("SD FAIL a=???????? ????cyc"); a = fail_addr zero-extended to 8 hex;
  // the err digit lives in the LED blink count and the trace frames.
  function [7:0] done_char(input [5:0] p);
    logic [31:0] fa;
    fa = {11'd0, fail_addr};
    case (p)
      6'd0: done_char = "S";
      6'd1: done_char = "D";
      6'd2: done_char = " ";
      6'd3: done_char = fail ? "F" : "P";
      6'd4: done_char = "A";
      6'd5: done_char = fail ? "I" : "S";
      6'd6: done_char = fail ? "L" : "S";
      6'd7: done_char = " ";
      6'd8: done_char = "a";
      6'd9: done_char = "=";
      6'd10, 6'd11, 6'd12, 6'd13, 6'd14, 6'd15, 6'd16, 6'd17:
            done_char = hexnyb(fa[31 - 4*(p - 6'd10) -: 4]);
      6'd18: done_char = " ";
      6'd19, 6'd20, 6'd21, 6'd22, 6'd23, 6'd24, 6'd25, 6'd26:
            done_char = hexnyb(cyc[31 - 4*(p - 6'd19) -: 4]);
      6'd27: done_char = "\r";
      default: done_char = "\n";
    endcase
  endfunction
  // live trace (4 ch): 0x55, phase, ctrl state, word index low
  function [7:0] trace_char(input [5:0] p);
    case (p)
      6'd0: trace_char = 8'h55;
      6'd1: trace_char = {4'd0, ph};
      6'd2: trace_char = c_st;
      default: trace_char = {4'd0, wi[3:0]};
    endcase
  endfunction

  wire [5:0] tx_len = (ph == P_DUMP) ? 6'd15 :
                      (ph == P_DONE) ? 6'd29 : 6'd4;
  function [7:0] tx_byte(input [5:0] p);
    tx_byte = (ph == P_DUMP) ? dump_char(p) :
              (ph == P_DONE) ? done_char(p) : trace_char(p);
  endfunction

  logic [5:0] tx_ptr = 6'd0;
  logic       tx_run = 1'b0;
  logic [1:0] tx_st = 2'd0;
  reg  [23:0] gap = 24'd0;
  always @(posedge clk_sd) begin
    if (!pwr[7]) begin
      tx_ptr <= 6'd0; tx_run <= 1'b0; tx_st <= 2'd0; gap <= 24'd0;
      dump_done <= 1'b0;
    end else if (!tx_run) begin
      if (gap != 24'd0) gap <= gap - 24'd1;
      else begin tx_ptr <= 6'd0; tx_run <= 1'b1; tx_st <= 2'd0; end
    end else begin
      case (tx_st)
        2'd0: begin
          if (tx_ptr == tx_len) begin
            tx_run <= 1'b0;
            if (ph == P_DUMP) begin
              if (dump_i == 4'(N_CANARY) - 4'd1) dump_done <= 1'b1;
              else dump_i <= dump_i + 4'd1;
              gap <= 24'd270000;                                // 20 ms/line
            end else gap <= (ph == P_DONE) ? 24'd6750000        // 0.5 s
                                           : 24'd270000;        // 20 ms trace
          end else begin
            u_start <= 1'b1;
            u_data  <= tx_byte(tx_ptr);
            tx_ptr  <= tx_ptr + 6'd1;
            tx_st   <= 2'd1;
          end
        end
        2'd1: begin u_start <= 1'b0; if (u_busy) tx_st <= 2'd2; end
        default: if (!u_busy) tx_st <= 2'd0;
      endcase
    end
  end
endmodule