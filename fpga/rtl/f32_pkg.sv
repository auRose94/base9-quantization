// IEEE-754 binary32 helpers for the ninths Gate-0/1 datapath.
//
// Scope (by construction of the data): operands are fp16-derived scales and
// exact-integer floats (< 2**24), so fp32-subnormal inputs never occur;
// subnormal results are out of scope and guarded with $error. add/mul are
// round-to-nearest-even, computed as separate mul and add ops — the contract
// is ggml_vec_dot_k9_6_q8_0_generic compiled without FMA contraction.
package f32_pkg;

  // exact: fp16 -> fp32 (subnormals renormalized; inf preserved; nan quiet)
  function automatic logic [31:0] f16_to_f32(input logic [15:0] h);
    logic        s;
    logic [4:0]  e;
    logic [9:0]  m;
    logic [31:0] sh;
    int          p;
    begin
      s = h[15]; e = h[14:10]; m = h[9:0];
      if (e == 5'h1F) begin
        f16_to_f32 = m != '0 ? {s, 8'hFF, 1'b1, m, 12'b0} : {s, 8'hFF, 23'b0};
      end else if (e == 5'h0) begin
        if (m == '0) f16_to_f32 = {s, 31'b0};
        else begin
          // value = m * 2**-24; msb of m at bit p -> biased exponent 103 + p
          p = 0;
          for (int i = 0; i < 10; i++) if (m[i]) p = i;
          sh = 32'(m) << (23 - p);         // leading bit lands on bit 23
          f16_to_f32 = {s, 8'(103 + p), sh[22:0]};
        end
      end else begin
        f16_to_f32 = {s, 8'(e) + 8'd112, m, 13'b0};
      end
    end
  endfunction

  // exact for |n| < 2**24 (integer partial sums qualify by wide margin)
  function automatic logic [31:0] int_to_f32(input int n);
    logic [31:0] mag;
    logic [31:0] sh;
    int          p;
    begin
      if (n == 0) int_to_f32 = 32'b0;
      else begin
        mag = n < 0 ? 32'(-n) : 32'(n);
        p = 0;
        for (int i = 0; i < 31; i++) if (mag[i]) p = i;
        sh = mag << (23 - p);
        int_to_f32 = {n[31], 8'(127 + p), sh[22:0]};
      end
    end
  endfunction

  function automatic logic [31:0] f32_mul(input logic [31:0] xa,
                                          input logic [31:0] xb);
    logic        sa, sb, sgn, az, bz, ai, bi, an, bn, g, sticky, inc;
    logic [7:0]  ea, eb;
    logic [22:0] ma, mb;
    logic [23:0] fa, fb;
    logic [24:0] m24;
    logic [47:0] p;
    int          er;
    begin
      sa = xa[31]; ea = xa[30:23]; ma = xa[22:0];
      sb = xb[31]; eb = xb[30:23]; mb = xb[22:0];
      sgn = sa ^ sb;
      az = (ea == 8'd0) && (ma == 0);
      bz = (eb == 8'd0) && (mb == 0);
      an = (ea == 8'hFF) && (ma != 0);
      bn = (eb == 8'hFF) && (mb != 0);
      ai = (ea == 8'hFF) && (ma == 0);
      bi = (eb == 8'hFF) && (mb == 0);
      if (an || bn || (ai && bz) || (bi && az)) f32_mul = 32'h7FC00000;
      else if (ai || bi) f32_mul = {sgn, 8'hFF, 23'b0};
      else if (az || bz) f32_mul = {sgn, 31'b0};
      else begin
        fa = {1'b1, ma}; fb = {1'b1, mb};
        p = fa * fb;                       // in [2^46, 2^48)
        if (p[47]) begin
          m24 = 25'(p[47:24]); g = p[23]; sticky = |p[22:0]; er = int'(ea) + int'(eb) - 126;
        end else begin
          m24 = 25'(p[46:23]); g = p[22]; sticky = |p[21:0]; er = int'(ea) + int'(eb) - 127;
        end
        inc = g && (sticky || m24[0]);     // round-half-even
        if (inc) begin
          m24 = m24 + 24'd1;
          if (m24[24]) begin m24 = 25'h800000; er = er + 1; end
        end
        if (er >= 255) f32_mul = {sgn, 8'hFF, 23'b0};
        else if (er <= 0) f32_mul = {sgn, 31'b0};   // FTZ (out of scope, guarded by fuzz)
        else f32_mul = {sgn, er[7:0], m24[22:0]};
      end
    end
  endfunction

  // round-to-nearest-even fp32 add. Alignment uses a 48-bit exact zone
  // (24-bit mantissas << 24): shifts of d <= 24 lose nothing; d > 24 leaves
  // only a sticky flag, which the standard theorem says cannot demand a
  // renormalization shift beyond 1 ulp, so sticky marking stays correct.
  function automatic logic [31:0] f32_add(input logic [31:0] xa,
                                          input logic [31:0] xb);
    logic        sa, sb, az, bz, ai, bi, an, bn, canceled;
    logic        sgn, s_big, s_sml, g, r, stick, inc;
    logic [7:0]  ea, eb, e_big;
    logic [22:0] ma, mb;
    logic [23:0] fbig, fsml;
    logic [24:0] m24;
    logic [47:0] va, vb, mag;
    logic [48:0] vsum;
    int          d, sh, p, lost, er;
    begin
      sa = xa[31]; ea = xa[30:23]; ma = xa[22:0];
      sb = xb[31]; eb = xb[30:23]; mb = xb[22:0];
      az = (ea == 8'd0) && (ma == 0);
      bz = (eb == 8'd0) && (mb == 0);
      ai = (ea == 8'hFF) && (ma == 0);
      bi = (eb == 8'hFF) && (mb == 0);
      an = (ea == 8'hFF) && (ma != 0);
      bn = (eb == 8'hFF) && (mb != 0);
      if (an || bn)                       f32_add = 32'h7FC00000;
      else if (ai && bi && (sa != sb))    f32_add = 32'h7FC00000;
      else if (ai || bi) f32_add = ai ? {sa, 8'hFF, 23'b0} : {sb, 8'hFF, 23'b0};
      else if (az && bz) f32_add = (sa && sb) ? {1'b1, 31'b0} : 32'b0;
      else if (az) f32_add = xb;
      else if (bz) f32_add = xa;
      else begin
        if (ea >= eb) begin
          fbig = {1'b1, ma}; fsml = {1'b1, mb};
          e_big = ea; s_big = sa; s_sml = sb;
          d = int'(ea) - int'(eb);
        end else begin
          fbig = {1'b1, mb}; fsml = {1'b1, ma};
          e_big = eb; s_big = sb; s_sml = sa;
          d = int'(eb) - int'(ea);
        end
        va = fbig << 24;
        if (d == 0)       vb = fsml << 24;
        else if (d <= 24) vb = 48'(fsml) << (24 - d);
        else begin
          sh = d - 24;
          if (sh <= 24) begin
            vb = 48'(fsml) >> sh;
            lost = int'(fsml) & ((1 << sh) - 1);
          end else begin
            vb = '0; lost = int'(fsml);
          end
          if (lost != 0) vb[0] = 1'b1;   // sticky
        end
        if (s_big == s_sml) begin
          canceled = 1'b0;
          sgn = s_big;
          vsum = {1'b0, va} + {1'b0, vb};
          if (vsum[48]) begin
            m24 = 25'(vsum[48:25]); g = vsum[24]; r = vsum[23]; stick = |vsum[22:0];
            er = int'(e_big) + 1;
          end else begin
            m24 = 25'(vsum[47:24]); g = vsum[23]; r = vsum[22]; stick = |vsum[21:0];
            er = int'(e_big);
          end
        end else begin
          if (vb > va)      begin mag = vb - va; sgn = s_sml; end
          else if (va > vb) begin mag = va - vb; sgn = s_big; end
          else              mag = '0;
          canceled = (mag == '0);
          if (!canceled) begin
            p = 47; while (p > 0 && !mag[p]) p--;
            sh = 47 - p;
            mag = mag << sh;               // vacated bits are exact zeros
            er = int'(e_big) - sh;
            m24 = 25'(mag[47:24]); g = mag[23]; r = mag[22]; stick = |mag[21:0];
          end
        end
        if (canceled) f32_add = 32'b0;     // exact cancellation -> +0 (RN)
        else begin
          inc = g && (r || stick || m24[0]);
          if (inc) begin
            m24 = m24 + 24'd1;
            if (m24[24]) begin m24 = 25'h800000; er = er + 1; end
          end
          if (er >= 255) f32_add = {sgn, 8'hFF, 23'b0};
          else if (er <= 0) f32_add = {sgn, 31'b0};  // FTZ (out of scope, guarded by fuzz)
          else f32_add = {sgn, er[7:0], m24[22:0]};
        end
      end
    end
  endfunction

endpackage