// analog_emu — synapse tile v2 SERIAL variant (silicon bring-up for the
// Tang Nano 20K: one shared multiplier, row-major sweep — identical
// arithmetic order per row as the parallel v2 tile, so the python goldens
// are valid unchanged).  Contract: fpga/analog_emu/README.md.
//
// Silicon-lesson ledger baked into this file:
//  * products need FULL-WIDTH intermediates (assigning a*b straight to a
//    narrower LHS wraps the whole expression at the LHS width)
//  * sync-read BRAM = 2-cycle read latency: issue/capture/accumulate are
//    three frozen-pointer phases, and row boundaries must RE-ISSUE
//  * one state per role — a shared capture state between the token and
//    drift paths silently kills one of them (duplicate case labels are
//    legal and take the first match)
//  * $signed on every branch of a ternary, or one unsigned branch makes
//    the shifts logical and negative weights decay wrong
//  * accumulators need an explicit clear per token
module synapse_tile_ser #(
    parameter int ROWS      = 64,
    parameter int COLS      = 64,
    parameter int WW        = 16,  // weight/token word bits (Q4.12)
    parameter int DACB      = 8,   // input DAC: symmetric-odd levels on +-8.0
    parameter int ADCB      = 8,   // output ADC: symmetric-odd levels
    parameter int OUT_SHIFT = 14,  // ADC rails = +-(2^(ADCB-1)-1) << OUT_SHIFT
    parameter int RN_POW    = 3,   // read noise k: +-2^(k-1)-1 ADC-LSBs
    parameter int DRIFT_SHIFT = 6, // drift decay = 1/2^DRIFT_SHIFT per pass
    parameter int LIF_EN    = 1,
    parameter int LEAK_SHIFT = 3,
    parameter int THRESH    = 96,
    parameter int LFSR_SEED = 16'hACE1
)(
    input  wire clk, rst_n,
    input  wire        we,           // image programming (0..4095 W, 4096+ G)
    input  wire [12:0] waddr,
    input  wire [15:0] wdata,
    input  wire        tok_we,
    input  wire [5:0]  tok_addr,
    input  wire [15:0] tok_data,
    input  wire        start,        // run one token (idle only)
    input  wire        drift_step,   // one decay sweep (idle only)
    output wire        done,
    output wire [ROWS-1:0] spikes,
    input  wire [5:0]  yq_addr,
    output wire signed [15:0] yq,
    input  wire [5:0]  vaddr,
    output wire signed [31:0] vmem_out
);
    localparam int SHR_IN = WW - DACB;
    localparam int L_IN   = (1 << (DACB-1)) - 1;
    localparam int L_ADC  = (1 << (ADCB-1)) - 1;
    localparam int RN_MSK = (1 << RN_POW) - 1;
    localparam int RN_OFF = (1 << (RN_POW-1)) - 1;

    reg signed [WW-1:0] wmem [0:ROWS*COLS-1];
`ifdef SIM_LOAD
    initial $readmemh("golden_w.memh", wmem);
`endif
    reg [7:0]           gmem [0:COLS-1];
    reg signed [WW-1:0] xmem [0:COLS-1];
    reg signed [17:0]   xgb  [0:COLS-1];
    reg signed [15:0]   yq_mem [0:ROWS-1];

    reg [3:0]  state;   // 0 idle, 1 precompute, 2 issue, 5 capture,
                        // 6 accumulate, 3 emit, 4/8/9 drift
    reg [8:0]  c;
    reg [$clog2(ROWS)-1:0] r;
    reg [12:0] d_cnt;
    reg [12:0] d_cnt_r;
    reg signed [39:0] acc [0:ROWS-1];
    reg signed [31:0] V [0:ROWS-1];
    reg [ROWS-1:0] spike_vec;
    reg [15:0] lfsr;
    reg        done_r;

    function automatic [15:0] lfsr_next(input [15:0] s);
        reg [15:0] d1, d2;
        d1 = s ^ {s[8:0], 7'b0};
        d2 = d1 ^ (d1 >> 9);
        lfsr_next = d2 ^ {d2[7:0], 8'b0};
    endfunction

    function automatic signed [WW-1:0] dac_q(input signed [WW-1:0] v);
        reg signed [WW-1:0] q;
        q = v >>> SHR_IN;
        if (q >  L_IN) q =  L_IN;
        if (q < -L_IN) q = -L_IN;
        dac_q = q <<< SHR_IN;
    endfunction

    function automatic signed [39:0] adc_q(input signed [39:0] v);
        reg signed [39:0] q;
        q = v >>> OUT_SHIFT;
        if (q >  L_ADC) q =  L_ADC;
        if (q < -L_ADC) q = -L_ADC;
        adc_q = q;
    endfunction

    wire [15:0]        n_raw = lfsr & RN_MSK[15:0];
    wire signed [39:0] n_adc = (RN_POW < 2) ? 40'sd0
        : ($signed({24'd0, n_raw}) - $signed({24'd0, RN_OFF[15:0]}));

    // sync-read BRAM: address mux issues in states 2/4, holds elsewhere
    reg [12:0]  rd_addr = 0;
    reg signed [WW-1:0] rd_data;
    wire [12:0] rd_addr_next = (state == 4'd2) ? (r*COLS + c[5:0])
                             : (state == 4'd4) ? d_cnt
                             : rd_addr;
    always @(posedge clk) begin
        rd_data <= wmem[rd_addr];
        if (!we) rd_addr <= rd_addr_next;
    end

    wire signed [17:0] xg_d = xgb[c[5:0]];
    wire signed [33:0] prod = rd_data * xg_d;

    wire signed [39:0] yq_i = adc_q(acc[r] + (n_adc <<< OUT_SHIFT));
    wire signed [31:0] cand = (V[r] >>> LEAK_SHIFT) + $signed(yq_i[31:0]);
    wire spk = LIF_EN && (cand >= THRESH);

    // unified write port: loader (we) has priority over the drift sweeper
    wire        drift_we = (state == 4'd9) && !we;
    wire [12:0] wr_addr  = we ? waddr : d_cnt_r;
    // $signed on every ternary branch: one unsigned branch makes the
    // shifts logical and negative weights decay wrong
    wire [15:0] wr_data  = we ? $signed(wdata)
                            : ($signed(rd_data)
                               - ($signed(rd_data) >>> DRIFT_SHIFT));

    always @(posedge clk) begin
        if (we || drift_we) begin
            if (we && waddr[12]) gmem[waddr[5:0]] <= wdata[7:0];
            else                 wmem[wr_addr]    <= wr_data;
        end
        if (tok_we) xmem[tok_addr] <= tok_data;
    end

    reg signed [24:0] prod25;   // full-width product scratch (see ledger)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= 0; c <= 0; r <= 0; d_cnt <= 0; d_cnt_r <= 0;
            lfsr <= LFSR_SEED[15:0];
            spike_vec <= {ROWS{1'b0}}; done_r <= 1'b0;
        end else begin
            done_r <= 1'b0;
            case (state)
                0: if (start) begin
                    state <= 1; c <= 0;
                    for (integer g2 = 0; g2 < ROWS; g2 = g2 + 1)
                        acc[g2] <= 40'sd0;
                end else if (drift_step) begin
                    state <= 4; d_cnt <= 0;
                end
                // xg precompute: DAC + column gain, full-width product
                1: begin
                    prod25 = dac_q(xmem[c[5:0]])
                             * $signed({1'b0, gmem[c[5:0]]});
                    xgb[c[5:0]] <= prod25 >>> 7;
                    c <= c + 1;
                    if (c == COLS-1) begin
                        state <= 2; c <= 0; r <= 0;
                    end
                end
                // sweep: 3 frozen-pointer phases per cell; row boundaries
                // RE-ISSUE (state 2) or the row's first product mispairs
                2: state <= 5;
                5: state <= 6;
                6: begin
`ifdef DBGMAC
                    if (r == 9 && c < 4)
                        $display("M9 c=%0d rd_addr=%0d rd_data=%h xg=%h accpre=%h",
                                 c, rd_addr, rd_data, xg_d, acc[9]);
`endif
                    acc[r] <= acc[r] + prod;
                    if (c == COLS-1) begin
                        c <= 0;
                        if (r == ROWS-1) begin
                            state <= 3; r <= 0; spike_vec <= {ROWS{1'b0}};
                        end else begin
                            r <= r + 1'b1;
                            state <= 2;
                        end
                    end else begin
                        c <= c + 1'b1;
                        state <= 2;
                    end
                end
                // emit: noise draw + ADC + LIF per row (r order = golden)
                3: begin
                    yq_mem[r] <= yq_i[15:0];
`ifdef DBGMAC
                    if (r < 2)
                        $display("V tok_r=%0d Vpre=%h yq=%0d cand=%0d spk=%b",
                                 r, V[r], yq_i >>> OUT_SHIFT, cand, spk);
`endif
                    V[r] <= spk ? 32'sd0 : cand;
                    if (r == 0) spike_vec <= {ROWS{1'b0}};
                    if (spk) spike_vec[r] <= 1'b1;
                    if (RN_POW >= 2) lfsr <= lfsr_next(lfsr);
                    if (r == ROWS-1) begin
                        state <= 0;
                        done_r <= 1'b1;
                    end
                    r <= r + 1'b1;
                end
                // drift: issue/capture/write per cell, row-major
                4: begin
                    rd_addr <= d_cnt;
                    d_cnt_r <= d_cnt;
                    state <= 8;
                end
                8: state <= 9;   // rd_data captures wmem[d_cnt] here
                9: begin
                    d_cnt <= d_cnt + 1;
                    if (d_cnt == ROWS*COLS-1) state <= 0;
                    else state <= 4;
                end
            endcase
        end
    end

    assign done     = done_r;
    assign spikes   = spike_vec;
    assign yq       = yq_mem[yq_addr];
    assign vmem_out = V[vaddr];

endmodule
