// analog_emu — Path C: SD-boot silicon gate for the Tang Nano 20K.
//
// Boots the payload from the microSD (21 LBAs at LBA 2048, written by the
// host via dd — see make_sd_payload.py), fills the tile's image + token +
// expected memories, runs ALL tokens with the golden drift schedule
// internally, compares every yq word / spike word / membrane against the
// card-borne expectations, and reports the verdict on the LEDs:
//
//   running   led[5] heartbeat blink
//   PASS      led[0..4] on, led[5] slow blink
//   FAIL      led[4] on + led[3:0] = err code
//             (1 card-init, 2 card-read, 3 yq, 4 spike, 5 vmem)
//
// Skeleton + clock topology follow gate0sd_top.sv (G0-proven): fabric clock
// straight off the divider MSB into BUFG (the clock-enable variant was dead
// on silicon), POR counter, zero-initialized memories (synth_gowin prunes
// written-only memories otherwise).  Payload words arrive BIG-ENDIAN from
// the sector bytes (first byte on the wire = word MSB).
module emu_sd_top (
  input  wire        clk,        // 27 MHz
  output wire [5:0]  led,
  output wire        sd_cs_n,
  output wire        sd_sck,
  output wire        sd_mosi,
  input  wire        sd_miso
);
  localparam int ROWS = 64, COLS = 64, NT = 8;
  localparam int DRIFT_EVERY = 2;
  localparam integer SD_BASE_LBA = 2048;
  localparam integer PAY_WORDS = 2688;   // 21 LBAs

  // fabric clock = 27 MHz / 4 = 6.75 MHz (G0-proven: MSB into BUFG, no CE)
  reg [1:0] divc = 2'd0;
  always @(posedge clk) divc <= divc + 2'd1;
  wire clk13;
  BUFG u_clkbuf (.I(divc[1]), .O(clk13));

  reg [7:0] pwr = 8'd0;
  wire rst_n = pwr[7];
  always @(posedge clk) if (!pwr[7]) pwr <= pwr + 8'd1;

  // ------------------------------------------------- tile memories -------
  reg signed [15:0] wmem    [0:4095];
  reg [7:0]         gmem    [0:63];
  reg signed [15:0] tokmem  [0:NT*64-1];
  reg signed [15:0] expyq   [0:NT*64-1];
  reg [63:0]        expsp   [0:NT-1];
  reg signed [31:0] expvmem [0:63];

  integer zi;
  initial begin
    for (zi = 0; zi < 4096; zi = zi + 1) wmem[zi] = 16'd0;
    for (zi = 0; zi < 64;   zi = zi + 1) gmem[zi] = 8'd0;
    for (zi = 0; zi < NT*64; zi = zi + 1) begin
      tokmem[zi] = 16'd0; expyq[zi] = 16'd0;
    end
    for (zi = 0; zi < NT; zi = zi + 1) expsp[zi] = 64'd0;
    for (zi = 0; zi < 64;  zi = zi + 1) expvmem[zi] = 32'd0;
  end

  // ------------------------------------------------- the tile ------------
  reg         t_start, t_drift, t_tok_we;
  reg  [5:0]  t_tok_addr, t_yq_addr, t_vaddr;
  reg  [15:0] t_tok_data;
  wire        t_done;
  wire [ROWS-1:0] t_spikes;
  wire signed [15:0] t_yq;
  wire signed [31:0] t_vmem;

  synapse_tile_ser #(
    .ROWS(ROWS), .COLS(COLS), .OUT_SHIFT(14), .THRESH(96)
  ) u_tile (
    .clk(clk13), .rst_n(rst_n),
    .we(1'b0), .waddr(13'd0), .wdata(16'd0),
    .tok_we(t_tok_we), .tok_addr(t_tok_addr), .tok_data(t_tok_data),
    .start(t_start), .drift_step(t_drift),
    .done(t_done), .spikes(t_spikes),
    .yq_addr(t_yq_addr), .yq(t_yq),
    .vaddr(t_vaddr), .vmem_out(t_vmem));

  // ------------------------------------------------- SD reader -----------
  reg        sd_start, sd_pend;
  reg [31:0] sd_lba;
  wire       sd_busy_r, sd_done_r, sd_error_r, sd_word_v_r, sd_rd_done_r;
  wire [31:0] sd_word_r;
  wire [5:0]  dbg_st;
  reg        sd_word_v, sd_rd_done, sd_done, sd_error, sd_busy;
  reg        sd_done_seen;

  sd_spi_reader sdr (
    .clk(clk13), .rst_n(rst_n),
    .spi_cs_n(sd_cs_n), .miso(sd_miso), .sck(sd_sck), .mosi(sd_mosi),
    .start(sd_start), .read_req(sd_pend), .write_req(sd_wpend), .lba(sd_lba),
    .wdata_req(sd_wdata_req), .wdata(sd_wdata),
    .dbg_st(dbg_st),
    .busy(sd_busy_r), .done(sd_done_r), .error(sd_error_r),
    .word_valid(sd_word_v_r), .word_out(sd_word_r), .rd_done(sd_rd_done_r),
    .dbg_st());

  always @(posedge clk13) begin
    sd_word_v  <= sd_word_v_r;
    sd_rd_done <= sd_rd_done_r;
    sd_done    <= sd_done_r;
    sd_error   <= sd_error_r;
    sd_busy    <= sd_busy_r;
    if (sd_done_r) sd_done_seen <= 1'b1;
  end

  reg [31:0] sd_word;   // registered reader word (1 cycle late)
  always @(posedge clk13) sd_word <= sd_word_r;

  // verdict-report write stream (1 LBA at LBA 4600, after the gate)
  reg        sd_wpend;
  wire       sd_wdata_req;
  reg  [31:0] sd_wdata;
  reg  [6:0] rcnt;        // report word 0..127
  reg        report_done;
  reg  [3:0] rep_task;    // 1 fill, 2+tok per-token, 3 final
  reg  [3:0] rep_ret;     // state to resume after the write
  reg [15:0] rep_watch;   // write watchdog

  // ------------------------------------------------- fill dispatch -------
  // payload word wcnt (0..2687) — dispatch by segment:
  //   0..2047 W pairs | 2048..2063 gains x4 | 2064..2319 tokens x2
  //   2320..2575 exp yq x2 | 2576..2591 exp spikes | 2592..2655 exp vmem
  reg [12:0] wcnt;
  reg [5:0]  fill_blk;
  reg [31:0] boot_to;

  always @(posedge clk13) begin
    if (!rst_n) begin
      wcnt <= 13'd0;
    end else if (sd_word_v && state == S_FILL) begin
      wcnt <= wcnt + 13'd1;
      if (wcnt < 2048) begin
        wmem[{wcnt[11:1], 1'b0}] <= sd_word[15:0];
        wmem[{wcnt[11:1], 1'b1}] <= sd_word[31:16];
      end else if (wcnt < 2064) begin
        gmem[{wcnt[3:0], 2'b00}] <= sd_word[7:0];
        gmem[{wcnt[3:0], 2'b01}] <= sd_word[15:8];
        gmem[{wcnt[3:0], 2'b10}] <= sd_word[23:16];
        gmem[{wcnt[3:0], 2'b11}] <= sd_word[31:24];
      end else if (wcnt < 2320) begin
        tokmem[{wcnt[8:1], 1'b0}]  <= sd_word[15:0];
        tokmem[{wcnt[8:1], 1'b1}]  <= sd_word[31:16];
      end else if (wcnt < 2576) begin
        expyq[{wcnt[8:1], 1'b0}]   <= sd_word[15:0];
        expyq[{wcnt[8:1], 1'b1}]   <= sd_word[31:16];
      end else if (wcnt < 2592) begin
        expsp[wcnt[4:1]][wcnt[0]*32 +: 32] <= sd_word;
      end else if (wcnt < 2656) begin
        expvmem[wcnt[5:0]] <= sd_word;
      end
    end
  end

  // ------------------------------------------------- run FSM -------------
  localparam [3:0] S_BOOT=0, S_FILL=1, S_TLOADA=2, S_TLOADB=3, S_TWAIT=4,
                   S_YQ=5, S_YQC=6, S_SPK=7, S_DRIFT=8, S_VMEM=9, S_VMQ=10,
                   S_DONE=11;
  reg [3:0]  state = S_BOOT;
  reg [3:0]  err_code = 4'd0;
  reg [5:0]  fail_idx = 6'd0;
  reg [5:0]  tok;
  reg [7:0]  qcnt;
  reg [15:0] drift_wait;
  reg [31:0] boot_to;

  reg signed [15:0] tok_q;
  reg [5:0]  fill_blk;

  always @(posedge clk13) begin
    if (sd_wdata_req && state == S_REPORT) begin
      case (rcnt)
        7'd0: sd_wdata <= {8'hA9, 8'hA9, rep_task, err_code};  // header
        7'd1: sd_wdata <= {3'b000, wcnt[12:0]};                // fill progress
        7'd2: sd_wdata <= {tok[5:0], qcnt[5:0], 20'd0};        // run position
        7'd3: sd_wdata <= t_spikes[31:0];
        7'd4: sd_wdata <= t_spikes[63:32];
        default: sd_wdata <= 32'd0;                            // pad
      endcase
      rcnt <= rcnt + 1;
    end
  end

  always @(posedge clk13) begin
    if (!rst_n) begin
      state <= S_BOOT; sd_start <= 1'b0; sd_pend <= 1'b0;
      sd_lba <= 32'(SD_BASE_LBA); boot_to <= 0; sd_done_seen <= 1'b0;
      err_code <= 0; fail_idx <= 0; tok <= 0; qcnt <= 0; drift_wait <= 0;
      fill_blk <= 0; sd_wpend <= 1'b0; rcnt <= 0; report_done <= 1'b0;
      t_start <= 1'b0; t_drift <= 1'b0; t_tok_we <= 1'b0; t_tok_addr <= 0;
      t_tok_data <= 0; t_yq_addr <= 0; t_vaddr <= 0;
    end else begin
      t_start <= 1'b0; t_drift <= 1'b0; t_tok_we <= 1'b0;
      if (sd_busy_r && sd_start) sd_start <= 1'b0;
      if (sd_busy_r && sd_pend)  sd_pend  <= 1'b0;
      case (state)
        S_BOOT: begin
          boot_to <= boot_to + 1;
          sd_start <= 1'b1;
          if (boot_to >= 32'd202500000) begin      // 30 s watchdog
            err_code <= 4'd1; state <= S_DONE;
          end else if (sd_done_seen) begin
            sd_done_seen <= 1'b0;
            if (sd_error) begin err_code <= 4'd1; state <= S_DONE; end
            else begin
              sd_lba <= 32'(SD_BASE_LBA);
              sd_pend <= 1'b1;
              wcnt <= 0;
              state <= S_FILL;
            end
          end
        end
        S_FILL: begin
          boot_to <= boot_to + 1;
          if (boot_to >= 32'd405000000) begin      // 60 s read watchdog
            err_code <= 4'd2; state <= S_DONE;
          end else if (sd_rd_done) begin           // one LBA streamed
            boot_to <= 0;
            fill_blk <= fill_blk + 6'd1;
            if (fill_blk == 6'd20) begin           // 21 LBAs done
              rep_task <= 4'd1; rep_ret <= S_TLOADA;
              rep_watch <= 0; rcnt <= 0;
              sd_lba <= 32'd4600; sd_wpend <= 1'b1;
              state <= S_REPORT;
            end else begin
              sd_pend <= 1'b1;
            end
          end
        end
        // token load: 2 cycles per word (addr+capture, then drive)
        S_TLOADA: begin
          t_tok_addr <= qcnt[5:0];
          tok_q      <= tokmem[qcnt[5:0]];   // capture THIS slot's value
          state <= S_TLOADB;
        end
        S_TLOADB: begin
          t_tok_data <= tok_q;                     // tokmem[qcnt] captured
          t_tok_we   <= 1'b1;
          qcnt <= qcnt + 1;
          if (qcnt[5:0] == 6'd63) begin
            t_tok_we <= 1'b0;
            qcnt <= 0;
            t_start <= 1'b1;
            state <= S_TWAIT;
          end else begin
            state <= S_TLOADA;
          end
        end
        S_TWAIT: if (t_done) begin
          qcnt <= 0; t_yq_addr <= 0;
          state <= S_YQ;
        end
        // yq compare: 64 rows, 2-phase (set addr / compare readback)
        S_YQ: begin
          t_yq_addr <= qcnt[5:0];
          state <= S_YQC;
        end
        S_YQC: begin
          if (t_yq !== $signed(expyq[{tok[3:0], qcnt[5:0]}])) begin
            if (err_code == 0) begin
              err_code <= 4'd3;
              fail_idx <= {tok[3:0], qcnt[3:0]};
            end
          end
          qcnt <= qcnt + 1;
          if (qcnt[5:0] == 6'd63) begin
            qcnt <= 0;
            state <= S_SPK;
          end else begin
            state <= S_YQ;
          end
        end
        S_SPK: begin
          if (t_spikes !== expsp[tok[3:0]]) begin
            if (err_code == 0) begin
              err_code <= 4'd4;
              fail_idx <= {2'b00, tok[3:0]};
            end
          end
          if ((tok + 1) % DRIFT_EVERY == 0 && (tok + 1) < NT) begin
            t_drift <= 1'b1;
            drift_wait <= 16'd0;
            state <= S_DRIFT;
          end else if (tok == NT-1) begin
            qcnt <= 0; t_vaddr <= 0;
            state <= S_VMEM;
          end else begin
            rep_task <= 4'd2 + tok[3:0]; rep_ret <= S_TLOADA;
            rep_watch <= 0; rcnt <= 0;
            sd_lba <= 32'd4600; sd_wpend <= 1'b1;
            state <= S_REPORT;                     // flight recorder
            tok <= tok + 1;
          end
        end
        // drift sweep wait: 3 cycles/cell x 4096 + margin
        S_DRIFT: begin
          drift_wait <= drift_wait + 1;
          if (drift_wait >= 16'd14000) begin
            tok <= tok + 1;
            qcnt <= 0;
            state <= S_TLOADA;
          end
        end
        // membrane compare: 64 rows, 2-phase
        S_VMEM: begin
          t_vaddr <= qcnt[5:0];
          state <= S_VMQ;
        end
        S_VMQ: begin
          if (t_vmem !== $signed(expvmem[qcnt[5:0]])) begin
            if (err_code == 0) begin
              err_code <= 4'd5;
              fail_idx <= qcnt[5:0];
            end
          end
          qcnt <= qcnt + 1;
          if (qcnt[5:0] == 6'd63) begin
            rep_task <= 4'd3; rep_ret <= S_DONE;
            rep_watch <= 0; rcnt <= 0;
            sd_lba <= 32'd4600; sd_wpend <= 1'b1;
            state <= S_REPORT;                     // final report
          end
          else state <= S_VMEM;
        end
        // FLIGHT RECORDER: 1 LBA report per checkpoint; the last write
        // wins, so the card always holds the furthest completed step
        S_REPORT: begin
          rep_watch <= rep_watch + 1;
          if (sd_error || rep_watch >= 16'd81000) begin
            // write failed or stalled — mark and continue the run anyway
            if (err_code == 0 && sd_error) err_code <= 4'd2;
            state <= rep_ret;
          end else if (sd_rd_done) begin
            state <= rep_ret;
          end
        end
        S_DONE: ;   // verdict latched
      endcase
    end
  end

  // ------------------------------------------------- LEDs ----------------
  // polarity/order-proof verdict protocol:
  //   running          led[5] heartbeat, others dark
  //   PASS (err 0)     ALL six LEDs pulse together once per second
  //   FAIL (err 1..5)  ALL six LEDs blink out the code: N quick pulses,
  //                    1.5 s dark, repeat — count the pulses per burst
  reg [25:0] hb = 0;
  always @(posedge clk13) hb <= hb + 1;

  wire running = (state != S_DONE);

  // blink-out sequencer: 6.75 MHz -> 6750 cycles = 1 ms
  reg [12:0] ms_cyc   = 0;     // cycles within this millisecond (0..6749)
  reg [9:0]  ms_cnt   = 0;     // milliseconds within the current phase
  reg [3:0]  pulses_left = 0;  // pulses remaining in this burst
  reg        burst_on = 0;    // in the pulse burst (vs the 2 s gap)
  reg        led_blink = 0;

  wire ms_tick = (ms_cyc == 13'd6749);
  always @(posedge clk13) begin
    if (!running) begin
      if (ms_tick) begin
        ms_cyc <= 0;
        if (err_code == 0) begin
          // PASS: 500 ms lit / 500 ms dark — phase-split by ms_cnt, VISIBLE
          ms_cnt <= (ms_cnt >= 10'd999) ? 10'd0 : ms_cnt + 1;
          led_blink <= (ms_cnt < 10'd500);
        end else if (!burst_on) begin
          led_blink <= 0;                       // 2 s gap, then arm the burst
          if (ms_cnt >= 10'd2000) begin
            burst_on <= 1; pulses_left <= err_code; ms_cnt <= 0;
          end else begin
            ms_cnt <= ms_cnt + 1;
          end
        end else begin
          // FAIL burst: err_code pulses of 500 ms lit / 500 ms dark
          led_blink <= (ms_cnt < 10'd500);
          if (ms_cnt >= 10'd999) begin
            ms_cnt <= 0;
            pulses_left <= pulses_left - 1;
            if (pulses_left == 1) begin
              burst_on <= 0; ms_cnt <= 0;       // burst done -> gap
            end
          end else begin
            ms_cnt <= ms_cnt + 1;
          end
        end
      end
    end else begin
      ms_cyc <= 0; ms_cnt <= 0; burst_on <= 0; pulses_left <= 0; led_blink <= 0;
    end
  end

  // RUNNING: the LEDs carry the SD engine's live state (6-bit dbg_st —
  // a frozen pattern after a stall = the wedge address, readable both
  // polarities); DONE: the verdict blink-out takes over
  // clock discriminator: led[5] blinks from the 27 MHz domain, led[4] from
  // the 6.75 MHz fabric (clk13) — if 5 blinks and 4 doesn't, clk13 is dead
  reg [25:0] hb27 = 0;
  always @(posedge clk) hb27 <= hb27 + 1;
  reg [25:0] hb13 = 0;
  always @(posedge clk13) hb13 <= hb13 + 1;

  assign led[0] = running ? dbg_st[0] : led_blink;
  assign led[1] = running ? dbg_st[1] : led_blink;
  assign led[2] = running ? dbg_st[2] : led_blink;
  assign led[3] = running ? dbg_st[3] : led_blink;
  assign led[4] = running ? hb13[23] : led_blink;   // clk13 alive indicator
  assign led[5] = running ? hb27[24] : led_blink;   // 27 MHz alive indicator

endmodule
