// Minimal correctness-first SDR SDRAM controller for the GW2AR in-package
// (SIP) die — Tang Nano 20K's 64 Mbit embedded SDRAM, exposed to the fabric
// as the named pads O_sdram_* / IO_sdram_dq (apicula binds them by port name;
// verified: strict, conflict-checked placements in the dedicated pad clusters).
//
// COMMAND-VISIBILITY RULE: every command asserted inside a state's body is
// visible on the die pins during the NEXT cycle (registered outputs) and is
// sampled by the die at that cycle's terminating posedge. State bodies run
// during their own state's cycle. Command field order in cmd_drive = the bit
// order of CMD_* below.
//
// At the 13.5 MHz domain one cycle = 74 ns, so tRCD/tRP/tWR/tRAS(min)/tRC are
// all satisfied inside single cycles; every command still gets >= 2 gaps.
// CAS = 2, burst length = 1, sequential. No auto-precharge: every access
// PREcharges explicitly, so banks are always closed in S_IDLE and refresh
// needs no bank bookkeeping.
//
// App contract (level handshake, same 1-cycle skew idiom as the SD reader):
//   hold req=1 (+ {wr_n, addr, wdata}) until busy=1 (latch seen); ready
//   pulses when done; holding req across ready runs words back-to-back.
//   Reads pulse rdata_v once with rdata.
module sdram_ctrl #(
  parameter integer BANK_LOG2 = 1,   // banks = 2
  parameter integer ROW_LOG2  = 11,  // rows  = 2048
  parameter integer COL_LOG2  = 9    // cols  = 512
) (
  input  wire clk,          // SDRAM-domain clock
  input  wire rst_n,
  // die pins (named-pad convention)
  output reg         sd_cke,
  output reg         sd_cs_n,
  output reg         sd_ras_n,
  output reg         sd_cas_n,
  output reg         sd_we_n,
  output reg  [ROW_LOG2-1:0] sd_addr,
  output reg  [1:0]  sd_ba,
  output reg  [3:0]  sd_dqm,
  output reg  [31:0] dq_out,
  output reg         dq_oe,     // 1 = fabric drives DQ this cycle
  input  wire [31:0] dq_in,     // DQ as sampled by the top (combinational pad)
  // app port
  input  wire        req,
  input  wire        wr_n,    // 0 = write, 1 = read
  input  wire [20:0] addr,    // linear word address (geometry decode below)
  input  wire [31:0] wdata,
  output reg  [31:0] rdata,
  output reg         rdata_v,
  output reg         busy,
  output reg         ready,
  // debug
  output reg  [7:0]  st       // registered copy of the FSM state (1 behind)
);
  // ── geometry decode (H1 default; the board's probe tests alternatives) ──
  wire [BANK_LOG2-1:0] bank_of = addr[20 : ROW_LOG2+COL_LOG2];
  wire [ROW_LOG2-1:0]  row_of  = addr[ROW_LOG2+COL_LOG2-1 : COL_LOG2];
  wire [COL_LOG2-1:0]  col_of  = addr[COL_LOG2-1 : 0];

  // ── refresh scheduler ───────────────────────────────────────────────────
  // 64 ms / 4096 rows = one AUTO REFRESH per 15.625 us (211 cycles @13.5 MHz;
  // span slightly eager). Counts idle cycles only — the bring-up's phases
  // individually stay far under 64 ms, and the hold test validates coverage.
  localparam integer REF_SPAN = 208;
  reg [15:0] ref_cnt = 16'd0;
  wire       ref_due = (ref_cnt >= 16'(REF_SPAN) - 16'd1);

  // ── init: 100 us CKE-off, settle, PRE-ALL, 4x AREF burst, LMR ────────────
  // Mode register: A[2:0]=000 (BL1 seq), A[6:4]=010 (CAS2), rest 0.
  localparam integer PWR_CYC   = 1600;   // > 100 us @ 13.5 MHz
  localparam integer INIT_REFS = 4;
  localparam [10:0]  MR_CAS2_BL1 = 11'h020;

  // commands on {cs_n, ras_n, cas_n, we_n} (all active low)
  localparam [3:0] CMD_NOP = 4'b1111;
  localparam [3:0] CMD_ACT = 4'b0011;
  localparam [3:0] CMD_RD  = 4'b0101;
  localparam [3:0] CMD_WR  = 4'b0100;
  localparam [3:0] CMD_PRE = 4'b0010;
  localparam [3:0] CMD_REF = 4'b0000;
  localparam [3:0] CMD_LMR = 4'b0001;

  localparam [7:0] S_INIT_PWR=8'd0, S_INIT_W1=8'd1,  S_INIT_PRE=8'd2,
                   S_INIT_PRW=8'd3, S_INIT_REF=8'd4, S_INIT_RFW=8'd5,
                   S_INIT_LMR=8'd6, S_INIT_LMW=8'd7, S_IDLE=8'd8,
                   S_ACT=8'd9,      S_ACTW=8'd10,     S_CMD=8'd11,
                   S_CAS1=8'd12,    S_CAS2=8'd13,     S_CAS3=8'd14,
                   S_WRW=8'd15,     S_PRE=8'd16,      S_PREW=8'd17,
                   S_REF=8'd18,     S_REFW=8'd19;
  reg [7:0] cs = S_INIT_PWR;

  logic [11:0] pwr_cnt;
  logic [2:0]  sub;
  logic [2:0]  refs_left;
  logic [20:0] addr_r;
  logic [31:0] wdata_r;
  logic        wr_n_r;

  task cmd_drive(input [3:0] c, input [1:0] ba0, input [10:0] a);
    sd_cs_n  <= c[3];  sd_ras_n <= c[2];
    sd_cas_n <= c[1];  sd_we_n  <= c[0];
    sd_ba    <= ba0;
    sd_addr  <= ROW_LOG2'(a);
  endtask

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cs <= S_INIT_PWR; sd_cke <= 1'b0; sd_cs_n <= 1'b1; sd_ras_n <= 1'b1;
      sd_cas_n <= 1'b1; sd_we_n <= 1'b1; sd_addr <= 11'd0; sd_ba <= 2'd0;
      sd_dqm <= 4'd0; dq_out <= 32'd0; dq_oe <= 1'b0;
      rdata <= 32'd0; rdata_v <= 1'b0; busy <= 1'b0; ready <= 1'b0;
      st <= 8'd0; pwr_cnt <= 12'd0; sub <= 3'd0; refs_left <= 3'd0;
      addr_r <= 21'd0; wdata_r <= 32'd0; wr_n_r <= 1'b1;
      ref_cnt <= 16'd0;
    end else begin
      rdata_v <= 1'b0;
      ready   <= 1'b0;
      if (cs == S_IDLE) ref_cnt <= ref_cnt + 16'd1;
      case (cs)
        // ── init ──────────────────────────────────────────────────────────
        S_INIT_PWR: begin
          if (pwr_cnt == 12'(PWR_CYC)) begin
            pwr_cnt <= 12'd0;
            sd_cke  <= 1'b1;
            sub     <= 3'd0;
            cs <= S_INIT_W1;
          end else pwr_cnt <= pwr_cnt + 12'd1;
        end
        S_INIT_W1: begin      // settle after CKE raise
          sub <= sub + 3'd1;
          if (sub == 3'd7) begin sub <= 3'd0; cs <= S_INIT_PRE; end
        end
        S_INIT_PRE: begin
          cmd_drive(CMD_PRE, 2'd0, {11{1'b1}});   // A10 = 1: all banks
          refs_left <= 3'(INIT_REFS);
          cs <= S_INIT_PRW;
        end
        S_INIT_PRW: begin
          cmd_drive(CMD_NOP, 2'd0, 11'd0);
          sub <= sub + 3'd1;
          if (sub == 3'd3) begin sub <= 3'd0; cs <= S_INIT_REF; end
        end
        S_INIT_REF: begin
          if (refs_left != 3'd0) begin
            refs_left <= refs_left - 3'd1;
            cmd_drive(CMD_REF, 2'd0, 11'd0);
            sub <= 3'd0;
            cs <= S_INIT_RFW;
          end else cs <= S_INIT_LMR;
        end
        S_INIT_RFW: begin     // tRC between init refreshes
          cmd_drive(CMD_NOP, 2'd0, 11'd0);
          sub <= sub + 3'd1;
          if (sub == 3'd4) begin sub <= 3'd0; cs <= S_INIT_REF; end
        end
        S_INIT_LMR: begin
          cmd_drive(CMD_LMR, 2'd0, 11'h020);   // CAS2, BL1, sequential
          ref_cnt <= 16'd0;
          cs <= S_INIT_LMW;
        end
        S_INIT_LMW: begin     // tMRD
          cmd_drive(CMD_NOP, 2'd0, 11'd0);
          sub <= sub + 3'd1;
          if (sub == 3'd3) begin
            ready <= 1'b1; busy <= 1'b0;
            sub <= 3'd0;
            cs <= S_IDLE;
          end
        end
        // ── app transactions ──────────────────────────────────────────────
        S_IDLE: begin
          if (ref_due && !req) begin
            ref_cnt <= 16'd0;
            sub <= 3'd0;
            cs <= S_REF;
          end else if (req) begin
            busy <= 1'b1;
            addr_r  <= addr;
            wdata_r <= wdata;
            wr_n_r  <= wr_n;
            sub <= 3'd0;
            cs <= S_ACT;
          end
        end
        S_ACT: begin
          cmd_drive(CMD_ACT, 2'(bank_of), 11'(row_of));
          cs <= S_ACTW;
        end
        S_ACTW: begin        // tRCD: ACT visible 2 cycles ago; 1 more gap
          cmd_drive(CMD_NOP, 2'(bank_of), 11'(row_of));
          sub <= sub + 3'd1;
          if (sub == 3'd1) begin sub <= 3'd0; cs <= S_CMD; end
        end
        S_CMD: begin
          if (wr_n_r) begin
            cmd_drive(CMD_RD, 2'(bank_of), 11'(col_of));
            cs <= S_CAS1;
          end else begin
            cmd_drive(CMD_WR, 2'(bank_of), 11'(col_of));
            dq_oe   <= 1'b1;
            dq_out  <= wdata_r;
            cs <= S_WRW;
          end
        end
        // CAS = 2: the RD command is visible during S_CAS1's cycle; the die
        // samples it at that cycle's end (edge T). Data valid T+2, i.e. the
        // cycle ending at edge T+2 = S_CAS3's cycle: sample there, PRE next.
        S_CAS1: begin
          cmd_drive(CMD_NOP, 2'(bank_of), 11'(col_of));
          cs <= S_CAS2;
        end
        S_CAS2: begin
          cmd_drive(CMD_NOP, 2'(bank_of), 11'(col_of));
          cs <= S_CAS3;
        end
        S_CAS3: begin        // last data-valid cycle: sample + precharge
          cmd_drive(CMD_PRE, 2'(bank_of), {11{1'b0}});
          rdata  <= dq_in;
          rdata_v <= 1'b1;
          cs <= S_PREW;
        end
        S_WRW: begin         // tWR: 2 waits, then PRE
          cmd_drive(CMD_NOP, 2'(bank_of), 11'(col_of));
          dq_oe <= 1'b0;
          sub <= sub + 3'd1;
          if (sub == 3'd1) begin sub <= 3'd0; cs <= S_PRE; end
        end
        S_PRE: begin          // tWR satisfied (2 waits in S_WRW); PRE now
          cmd_drive(CMD_PRE, 2'(bank_of), {11{1'b0}});
          cs <= S_PREW;
        end
        S_PREW: begin        // tRP: 3 waits, then ready
          cmd_drive(CMD_NOP, 2'd0, 11'd0);
          sub <= sub + 3'd1;
          if (sub == 3'd2) begin
            sub <= 3'd0;
            ready <= 1'b1; busy <= 1'b0;
            cs <= S_IDLE;
          end
        end
        // ── refresh ───────────────────────────────────────────────────────
        S_REF: begin
          cmd_drive(CMD_REF, 2'd0, 11'd0);
          cs <= S_REFW;
        end
        S_REFW: begin        // tRC
          cmd_drive(CMD_NOP, 2'd0, 11'd0);
          sub <= sub + 3'd1;
          if (sub == 3'd4) begin sub <= 3'd0; cs <= S_IDLE; end
        end
        default: cs <= S_IDLE;
      endcase
      st <= cs;  // telegraph copy: st shows the state DURING this cycle
    end
  end
endmodule