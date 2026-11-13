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
  reg        sd_word_v, sd_rd_done, sd_done, sd_error, sd_busy;
  reg        sd_done_seen;

  sd_spi_reader sdr (
    .clk(clk13), .rst_n(rst_n),
    .spi_cs_n(sd_cs_n), .miso(sd_miso), .sck(sd_sck), .mosi(sd_mosi),
    .start(sd_start), .read_req(sd_pend), .write_req(1'b0), .lba(sd_lba),
    .wdata_req(), .wdata(32'd0),
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
    if (!rst_n) begin
      state <= S_BOOT; sd_start <= 1'b0; sd_pend <= 1'b0;
      sd_lba <= 32'(SD_BASE_LBA); boot_to <= 0; sd_done_seen <= 1'b0;
      err_code <= 0; fail_idx <= 0; tok <= 0; qcnt <= 0; drift_wait <= 0;
      fill_blk <= 0;
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
          if (sd_rd_done) begin                    // one LBA streamed
            fill_blk <= fill_blk + 6'd1;
            if (fill_blk == 6'd20) begin           // 21 LBAs done
              tok <= 0; qcnt <= 0;
              state <= S_TLOADA;
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
            tok <= tok + 1;
            qcnt <= 0;
            state <= S_TLOADA;
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
          if (qcnt[5:0] == 6'd63) state <= S_DONE;
          else state <= S_VMEM;
        end
        S_DONE: ;   // verdict latched
      endcase
    end
  end

  // ------------------------------------------------- LEDs ----------------
  reg [25:0] hb = 0;
  always @(posedge clk13) hb <= hb + 1;

  wire running = (state != S_DONE);

  assign led[0] = !running ? ((err_code == 0) ? 1'b1 : err_code[0]) : 1'b0;
  assign led[1] = !running ? ((err_code == 0) ? 1'b1 : err_code[1]) : 1'b0;
  assign led[2] = !running ? ((err_code == 0) ? 1'b1 : err_code[2]) : 1'b0;
  assign led[3] = !running ? ((err_code == 0) ? 1'b1 : err_code[3]) : 1'b0;
  assign led[4] = !running ? 1'b1 : 1'b0;
  assign led[5] = running  ? hb[21]
                : (err_code == 0) ? hb[22] : hb[16];

endmodule
