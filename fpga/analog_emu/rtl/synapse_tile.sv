// analog_emu — synapse tile v2.  Contract: fpga/analog_emu/README.md
// v1: fixed-point MVM, DAC/ADC symmetric-odd quantizers, per-column gain,
//     degradation baked into the image by golden.py.
// v2 adds: on-tile LFSR16 READ noise (deterministic, drawn fresh per row per
//     token — the Phase-B read-path model, symmetric-odd like the
//     converters), a drift step (exponential conductance relaxation swept
//     in-place), and a per-row LIF neuron core (spike trigger integrating
//     tile outputs over time, BrainScaleS-style).
// All floor semantics, no division.  Every op has a python twin in golden.py.
module synapse_tile #(
    parameter int ROWS      = 16,
    parameter int COLS      = 16,
    parameter int WW        = 16,  // weight/token word bits (Q4.12)
    parameter int DACB      = 8,   // input DAC: symmetric-odd levels on +-8.0
    parameter int ADCB      = 8,   // output ADC: symmetric-odd levels
    parameter int OUT_SHIFT = 11,  // ADC rails = +-(2^(ADCB-1)-1) << OUT_SHIFT
    parameter int RN_POW    = 3,   // read noise k: +-2^(k-1)-1 ADC-LSBs
                                   // (k=1 -> +-0 = off; k>=2 for actual noise)
    parameter int DRIFT_SHIFT = 6, // drift decay = 1/2^DRIFT_SHIFT per pass
    parameter int LIF_EN    = 1,   // per-row LIF core on/off
    parameter int LEAK_SHIFT = 3,  // LIF leak = 1/2^LEAK_SHIFT per token
    parameter int THRESH    = 384, // LIF spike threshold (in yq units)
    parameter int LFSR_SEED = 16'hACE1
)(
    input  wire                      clk,
    input  wire                      rstn,
    input  wire                      start,
    input  wire                      xv,     // token column valid
    input  wire signed [WW-1:0]      x,      // token column value (Q4.12)
    input  wire [7:0]                g,      // column gain Q0.7 (256 = x1.0)
    input  wire                      drift_step,  // pulse: one decay sweep
    output wire                      yv,     // output row valid
    output wire [$clog2(ROWS)-1:0]   yrow,
    output wire signed [39:0]        y,      // ADC-dequantized accumulator
    output wire [ROWS-1:0]           spikes, // LIF spikes for the current token
    output wire                      done,
    input  wire [$clog2(ROWS)-1:0]   vaddr,  // membrane scanback
    output wire signed [31:0]        vmem_out
);
    localparam int SHR_IN = WW - DACB;              // DAC step = 2^SHR_IN
    localparam int L_IN   = (1 << (DACB-1)) - 1;
    localparam int L_ADC  = (1 << (ADCB-1)) - 1;
    localparam int RN_MSK = (1 << RN_POW) - 1;      // symmetric-odd read noise
    localparam int RN_OFF = (1 << (RN_POW-1)) - 1;

    // weight image (degradation baked in by golden.py); loaded by the
    // testbench context under SIM_LOAD — synthesis replaces this with a
    // real programming port later.
    reg signed [WW-1:0] wmem [0:ROWS*COLS-1];
`ifdef SIM_LOAD
    initial $readmemh("golden_w.memh", wmem);
`endif

    reg [7:0]  c;                  // column counter
    reg [$clog2(ROWS)-1:0] r;      // row counter (emit + drift sweep)
    reg [2:0]  state;              // 0 idle, 1 mac, 2 emit, 3 drift
    reg signed [39:0] acc [0:ROWS-1];
    reg signed [31:0] V    [0:ROWS-1];   // LIF membranes (persist across tokens)
    reg [ROWS-1:0]    spike_vec;

    // --- LFSR16 (xorshift 7,9,8) — the tile's deterministic noise source ---
    reg [15:0] lfsr;
    function automatic [15:0] lfsr_next(input [15:0] s);
        reg [15:0] d1, d2;
        d1 = s ^ {s[8:0], 7'b0};
        d2 = d1 ^ (d1 >> 9);
        lfsr_next = d2 ^ {d2[7:0], 8'b0};
    endfunction

    // DAC: floor-quantize to the symmetric-odd grid, then dequantize
    function automatic signed [WW-1:0] dac_q(input signed [WW-1:0] v);
        reg signed [WW-1:0] q;
        q = v >>> SHR_IN;
        if (q >  L_IN) q =  L_IN;
        if (q < -L_IN) q = -L_IN;
        dac_q = q <<< SHR_IN;
    endfunction

    // ADC: floor-quantize the accumulator to the symmetric-odd grid
    // (returns yq units — caller dequantizes with <<< OUT_SHIFT)
    function automatic signed [39:0] adc_q(input signed [39:0] v);
        reg signed [39:0] q;
        q = v >>> OUT_SHIFT;
        if (q >  L_ADC) q =  L_ADC;
        if (q < -L_ADC) q = -L_ADC;
        adc_q = q;
    endfunction

    // read-noise draw for the CURRENT emit row, in ADC-LSB units:
    // n = (lfsr & RN_MSK) - RN_OFF  in [-(2^(k-1)-1), +(2^(k-1)-1)]
    wire [15:0]       n_raw = lfsr & RN_MSK[15:0];
    wire signed [39:0] n_adc = (RN_POW < 2) ? 40'sd0
        : ($signed({24'd0, n_raw}) - $signed({24'd0, RN_OFF[15:0]}));

    genvar gi;
    generate
        for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_row
            wire signed [WW-1:0]   wr  = wmem[gi*COLS + c];
            wire signed [WW-1:0]   xq  = dac_q(x);
            wire signed [24:0]     xg_f = xq * $signed({1'b0, g});  // 16b x 9b
            wire signed [17:0]     xg  = $signed(xg_f[24:7]);       // >>>7 exact
            wire signed [33:0]     prod = wr * xg;                  // 16b x 18b

            // noisy accumulator for this row (noise lands pre-quantization,
            // in ADC-LSB units — exactly n << OUT_SHIFT raw LSBs)
            wire signed [39:0] accn = acc[gi] + (n_adc <<< OUT_SHIFT);
            wire signed [39:0] yq   = adc_q(accn);

            // LIF: candidate = leak(V) + yq; spike if >= THRESH; reset to 0
            wire signed [31:0] cand = (V[gi] >>> LEAK_SHIFT)
                                    + $signed(yq[31:0]);
            wire spk = LIF_EN && (cand >= THRESH);

            always @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    acc[gi] <= 40'sd0;
                    V[gi]   <= 32'sd0;
                end else if (state == 3'd1 && xv) begin
                    acc[gi] <= acc[gi] + prod;
                end else if (state == 3'd0 && start) begin
                    acc[gi] <= 40'sd0;    // V deliberately NOT cleared:
                end else if (state == 3'd2 && r == gi) begin
                    // emit cycle of this row: LIF update with its own yq
                    V[gi] <= spk ? 32'sd0 : cand;
                end else if (state == 3'd3 && r == gi) begin
                    // drift sweep of this row: exponential relaxation to 0
                    for (integer c2 = 0; c2 < COLS; c2 = c2 + 1)
                        wmem[gi*COLS + c2] <= wmem[gi*COLS + c2]
                            - (wmem[gi*COLS + c2] >>> DRIFT_SHIFT);
                end
            end

            // spike vector: each row owns its bit (clear at emit start)
            always @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    spike_vec[gi] <= 1'b0;
                end else if (state == 3'd2) begin
                    if (r == 0)    spike_vec[gi] <= 1'b0;
                    if (r == gi && spk) spike_vec[gi] <= 1'b1;
                end
            end
        end
    endgenerate

    // FSM
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state <= 3'd0; c <= 8'd0; r <= 0;
            lfsr  <= LFSR_SEED[15:0];
        end else begin
            case (state)
                3'd0: if (start) begin
                    state <= 3'd1; c <= 8'd0;
                end else if (drift_step) begin
                    state <= 3'd3; r <= 0;
                end
                3'd1: if (xv) begin
                    c <= c + 8'd1;
                    if (c == COLS-1) begin
                        state <= 3'd2; r <= 0;
                    end
                end
                3'd2: begin
                    if (RN_POW >= 2) lfsr <= lfsr_next(lfsr);
                    if (r == ROWS-1) state <= 3'd0;
                    r <= r + 1'b1;
                end
                3'd3: begin
                    if (r == ROWS-1) state <= 3'd0;
                    r <= r + 1'b1;
                end
            endcase
        end
    end

    // current row's output (EMIT)
    wire signed [39:0] accn_cur = acc[r] + (n_adc <<< OUT_SHIFT);
    wire signed [39:0] yq_cur   = adc_q(accn_cur);

    assign yv        = (state == 3'd2);
    assign yrow      = r;
    assign y         = yq_cur <<< OUT_SHIFT;
    assign spikes    = spike_vec;
    assign done      = (state == 3'd2) && (r == ROWS-1);
    assign vmem_out  = V[vaddr];

endmodule
