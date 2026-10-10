// V2 streaming kernel: ggml_vec_dot_k9_6_q8_0 against byte-served scratch
// memories instead of V1's wide parallel ports (the V1 shape synthesized to
// ~15k LUT4 of byte muxes and failed GW2A-18 placement).
//
// The FSM and every fp op are UNCHANGED from V1 (the bit-exactness contract:
// op-for-op replication of the C kernel, no FMA contraction; V1/tb_k96 is
// the semantic reference). What changed is byte routing:
//
//   waddr0 (wm)  lb code byte      = wblk byte 8+t             every MAC step
//   waddr1 (wm)  hh pair byte      = wblk byte 136+16g+8kk+(bb>>1) every step
//   aaddr0 (am)  act code pair     = bytes (34j+2+2bb, +1), j=2g+kk every step
//   sreq_addr    scale service     = d1 block scale (j) at S_SUMI, d0 group
//                scale (g) at S_SUMF — the caller pre-registers all 4+8
//                scales at scratch-load time, so this port never touches
//                the streaming pipelines (no flush, no recovery bubble)
//
// Timing contract with the caller (sync-read scratch, 1-cycle latency):
// streaming addresses are REGISTERED two steps ahead of consumption (each
// S_MAC sets the addr for t+2) so data is valid exactly on the consume cycle
// and stays valid while the address holds across fp-boundary cycles. The
// scale port is combinational (registered values + mux), consumed in-cycle.
import f32_pkg::*;
module k96_vec_dot_v2 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic [7:0]  waddr0,      // lb stream
  output logic [7:0]  waddr1,      // hh stream
  output logic [8:0]  aaddr0,      // act code-pair stream (low byte both in word)
  output logic [3:0]  sreq_addr,   // {d0_mode, sel}: d1 j=2g+kk (sel 0..7) | d0 g (sel 0..3)
  input  logic [7:0]  wdata0,
  input  logic [7:0]  wdata1,
  input  logic [15:0] adata0,      // {code 2bb+1, code 2bb} LE
  input  logic [15:0] sdata,       // selected fp16 scale, LE
  output logic        busy,
  output logic        done,
  output logic [2:0]  k_st,       // debug: FSM state
  output logic [31:0] partial_bits
);
  typedef enum logic [2:0] {S_IDLE, S_FETCH, S_SETTLE, S_MACS, S_SUMI,
                            S_SUMF, S_OUT} st_t;   // [2:0] holds 0..6, fine
  st_t st;

  int t;        // 0..127: 4 groups x 2 activation blocks x 16 byte-pairs
  int g, rr, kk, bb, c0;

  logic signed [7:0]  cp0_w, cp1_w;    // weight codes (-32..31)
  logic signed [7:0]  ay0_w, ay1_w;    // activation int8
  logic signed [15:0] prod_w;
  logic signed [31:0] sb_next_w;

  logic signed [31:0] sumi_block;
  logic signed [31:0] sb_cap;
  logic [1:0]  g_cap;
  logic        kk_cap;
  logic [31:0] sumi_r, sumf_r, out_r;  // fp32 bit patterns

  always_comb begin
    g  = t / 32;
    rr = t % 32;
    kk = rr / 16;
    bb = rr % 16;
    c0 = 32 * kk + 2 * bb;               // first code index of the pair in block j
    ay0_w = $signed(adata0[7:0]);
    ay1_w = $signed(adata0[15:8]);
    cp0_w = $signed({2'b00, wdata1[((c0 & 3) * 2) +: 2], wdata0[3:0]}) - 32;
    cp1_w = $signed({2'b00, wdata1[(((c0 + 1) & 3) * 2) +: 2], wdata0[7:4]}) - 32;
    prod_w = cp0_w * ay0_w + cp1_w * ay1_w;
    sb_next_w = sumi_block + {{16{prod_w[15]}}, prod_w};
  end

  // address generators over the flat t counter (values past the schedule are
  // harmless: the caller's scratch has slack and nothing consumes them)
  function automatic logic [7:0] lb_of(input int tt);
    lb_of = 8'(8 + tt);
  endfunction
  function automatic logic [7:0] hh_of(input int tt);   // yosys: no decl initializers
    int gg, rrr, kkk, bbb;
    begin
      gg = tt / 32; rrr = tt % 32; kkk = rrr / 16; bbb = rrr % 16;
      hh_of = 8'(136 + 16*gg + 8*kkk + (bbb >> 1));
    end
  endfunction
  function automatic logic [8:0] ay_of(input int tt);
    int gg, rrr, kkk, bbb;
    begin
      gg = tt / 32; rrr = tt % 32; kkk = rrr / 16; bbb = rrr % 16;
      ay_of = 9'(68*gg + 34*kkk + 2 + 2*bbb);
    end
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; t <= 0; busy <= 1'b0; done <= 1'b0;
      sumi_block <= 32'd0; sb_cap <= 32'd0;
      sumi_r <= 32'b0; sumf_r <= 32'b0; out_r <= 32'b0;
      g_cap <= 2'd0; kk_cap <= 1'b0;
      waddr0 <= 8'd0; waddr1 <= 8'd0; aaddr0 <= 9'd0; sreq_addr <= 4'd0;
    end else begin
      done <= 1'b0;
      case (st)
        S_IDLE: if (start) begin
          t <= 0; sumi_block <= 32'd0;
          sumi_r <= 32'b0; sumf_r <= 32'b0;
          sreq_addr <= 4'b0_000;        // d1 of block 0 (assigned fresh at FETCH)
          busy <= 1'b1; st <= S_FETCH;
        end
        // 3-cycle byte step: FETCH issues addresses, SETTLE lets the sync
        // read sample them, MACS consumes (the C kernel's one byte iteration)
        S_FETCH: begin
          waddr0 <= lb_of(t);
          waddr1 <= hh_of(t);
          aaddr0 <= ay_of(t);
          sreq_addr <= {1'b0, g[1:0], kk[0]};   // d1 consumed at next S_SUMI
          st <= S_SETTLE;
        end
        S_SETTLE: st <= S_MACS;
        S_MACS: begin
          sumi_block <= sb_next_w;
          if (bb == 15) begin
            sb_cap <= sb_next_w;
            g_cap <= g[1:0]; kk_cap <= kk[0];
            st <= S_SUMI;
          end else begin
            t <= t + 1;
            st <= S_FETCH;
          end
        end
        S_SUMI: begin
          sumi_r <= f32_pkg::f32_add(sumi_r, f32_pkg::f32_mul(
                    f32_pkg::f16_to_f32(sdata), f32_pkg::int_to_f32(sb_cap)));
          sumi_block <= 32'd0;
          if (kk_cap) begin
            sreq_addr <= {1'b1, 1'b0, g_cap};   // d0 consumed at S_SUMF (sel = g)
            st <= S_SUMF;
          end else begin
            t <= t + 1; st <= S_FETCH;
          end
        end
        S_SUMF: begin
          sumf_r <= f32_pkg::f32_add(sumf_r, f32_pkg::f32_mul(
                    f32_pkg::f16_to_f32(sdata), sumi_r));
          sumi_r <= 32'b0;                  // C: float sumi is per-group
          if (g_cap == 2'd3) st <= S_OUT;
          else begin
            sreq_addr <= {1'b0, g_cap + 2'd1, 1'b0};  // d1 of next group's first block
            t <= t + 1; st <= S_FETCH;
          end
        end
        S_OUT: begin
          out_r <= sumf_r;
          busy <= 1'b0; done <= 1'b1;
          st <= S_IDLE;
        end
        default: st <= S_IDLE;
      endcase
    end
  end

  assign partial_bits = out_r;
  assign k_st = st;
endmodule