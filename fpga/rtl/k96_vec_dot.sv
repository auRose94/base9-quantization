// Gate-0 datapath: ggml_vec_dot_k9_6_q8_0 in RTL, bit-exact against the fork
// (docs/08-ninths-stage2-spec.md, ggml/src/ggml-cpu/quants.c). One invocation
// consumes one 256-code superblock paired with 8 Q8_0 activation blocks and
// produces that superblock's fp32 partial sum; the caller chains partials
// across superblocks in order (same sequential adds as the C kernel).
//
// V1 is a correctness-first FSM: one byte-iteration per clock plus register
// stages for the fp adds at (g,k) boundaries, op order identical to the C.
// Integer MACs are exact, so V2 vectorization (lanes over b) cannot change
// any bits; only the fp32 add order is load-bearing.
//
// Port layout (little-endian bit order):
//   wblk[8*b +: 8]        = wblk byte b  (sc[8] ql[128] qh[64], 200 B)
//   act[272*j +: 16]      = fp16 scale of Q8_0 block j (bytes 2j, 2j+1 LE)
//   act[272*j+16+8*m +: 8]= int8 code m of Q8_0 block j (two's complement)
module k96_vec_dot
  import f32_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [1599:0] wblk,
  input  logic [2175:0] act,
  output logic        busy,
  output logic        done,
  output logic [31:0] partial_bits
);
  typedef enum logic [2:0] {S_IDLE, S_MAC, S_SUMI, S_SUMF, S_OUT} st_t;
  st_t st;

  int t;        // 0..127: 4 groups x 2 activation blocks x 16 bytes
  int g, rr, kk, bb, j, c0;

  logic [7:0] lb_w, hh0_w, hh1_w;
  logic [1:0] h0_w, h1_w;
  logic [5:0] u60_w, u61_w;
  logic signed [7:0]  cp0_w, cp1_w;    // weight codes (-32..31, -31..31 real)
  logic signed [7:0]  ay0_w, ay1_w;    // activation int8
  logic signed [15:0] prod_w;
  logic signed [31:0] sb_next_w;
  logic [15:0] d0_w, d1_w;

  logic signed [31:0] sumi_block;      // int accumulator of the current (g,k) pair
  logic signed [31:0] sb_cap;
  logic [15:0] d0_cap, d1_cap;
  logic [1:0]  g_cap;
  logic        kk_cap;
  logic [31:0] sumi_r, sumf_r, out_r;  // fp32 bit patterns

  always_comb begin
    g  = t / 32;
    rr = t % 32;
    kk = rr / 16;
    bb = rr % 16;
    j  = 2 * g + kk;                   // activation block index (C: g*2 + k)
    c0 = 32 * kk + 2 * bb;
    lb_w  = wblk[8 * (8 + 32*g + 16*kk + bb) +: 8];
    hh0_w = wblk[8 * (136 + 16*g + (c0 >> 2)) +: 8];
    hh1_w = wblk[8 * (136 + 16*g + ((c0 + 1) >> 2)) +: 8];
    h0_w  = hh0_w[((c0 & 3) * 2) +: 2];
    h1_w  = hh1_w[(((c0 + 1) & 3) * 2) +: 2];
    u60_w = {h0_w, lb_w[3:0]};         // u6 = code + 32
    u61_w = {h1_w, lb_w[7:4]};
    cp0_w = $signed({2'b00, u60_w}) - 32;
    cp1_w = $signed({2'b00, u61_w}) - 32;
    ay0_w = $signed(act[272*j + 16 + 8*(2*bb) +: 8]);
    ay1_w = $signed(act[272*j + 16 + 8*(2*bb + 1) +: 8]);
    prod_w = cp0_w * ay0_w + cp1_w * ay1_w;
    sb_next_w = sumi_block + {{16{prod_w[15]}}, prod_w};
    d0_w = wblk[16*g +: 16];           // sc[g], LE fp16
    d1_w = act[272*j +: 16];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; t <= 0; busy <= 1'b0; done <= 1'b0;
      sumi_block <= 32'd0; sb_cap <= 32'd0;
      sumi_r <= 32'b0; sumf_r <= 32'b0; out_r <= 32'b0;
      g_cap <= 2'd0; kk_cap <= 1'b0;
      d0_cap <= 16'b0; d1_cap <= 16'b0;
    end else begin
      done <= 1'b0;
      case (st)
        S_IDLE: if (start) begin
          t <= 0; sumi_block <= 32'd0;
          sumi_r <= 32'b0; sumf_r <= 32'b0;
          busy <= 1'b1; st <= S_MAC;
        end
        S_MAC: begin
          sumi_block <= sb_next_w;
          if (bb == 15) begin
            sb_cap <= sb_next_w;
            g_cap <= g[1:0]; kk_cap <= kk[0];
            d0_cap <= d0_w; d1_cap <= d1_w;
            st <= S_SUMI;
          end else t <= t + 1;
        end
        S_SUMI: begin
          sumi_r <= f32_add(sumi_r, f32_mul(f16_to_f32(d1_cap),
                                            int_to_f32(sb_cap)));
`ifdef GATE0_DEBUG
          $display("[k96] g=%0d k=%0d sb=%0d d1=%h sumi_before=%h sumi_after=%h",
                   g_cap, kk_cap, sb_cap, d1_cap, sumi_r, 
                   f32_add(sumi_r, f32_mul(f16_to_f32(d1_cap), int_to_f32(sb_cap))));
`endif
          sumi_block <= 32'd0;
          if (kk_cap) st <= S_SUMF;
          else begin t <= t + 1; st <= S_MAC; end
        end
        S_SUMF: begin
          sumf_r <= f32_add(sumf_r, f32_mul(f16_to_f32(d0_cap), sumi_r));
          sumi_r <= 32'b0;             // C: float sumi is per-group (declared in g-loop)
          if (g_cap == 2'd3) st <= S_OUT;
          else begin t <= t + 1; st <= S_MAC; end
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
endmodule