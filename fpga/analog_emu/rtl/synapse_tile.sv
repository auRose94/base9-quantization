// analog_emu — synapse tile v1.  Arithmetic contract: fpga/analog_emu/README.md
// All fixed-point, floor semantics, no division (power-of-two rails).
// Degradation is baked into the loaded image (see golden.py), not computed here.
module synapse_tile #(
    parameter int ROWS      = 16,
    parameter int COLS      = 16,
    parameter int WW        = 16,  // weight/token word bits (Q4.12)
    parameter int DACB      = 8,   // input DAC: symmetric-odd levels on +-8.0
    parameter int ADCB      = 8,   // output ADC: symmetric-odd levels
    parameter int OUT_SHIFT = 11   // ADC rails = +-(2^(ADCB-1)-1) << OUT_SHIFT
)(
    input  wire                      clk,
    input  wire                      rstn,
    input  wire                      start,
    input  wire                      xv,     // token column valid
    input  wire signed [WW-1:0]      x,      // token column value (Q4.12)
    input  wire [7:0]                g,      // column gain Q0.7 (256 = x1.0)
    output wire                      yv,     // output row valid
    output wire [$clog2(ROWS)-1:0]   yrow,
    output wire signed [39:0]        y,      // ADC-dequantized accumulator
    output wire                      done
);
    localparam int SHR_IN = WW - DACB;              // DAC step = 2^SHR_IN
    localparam int L_IN   = (1 << (DACB-1)) - 1;    // symmetric-odd DAC limit
    localparam int L_ADC  = (1 << (ADCB-1)) - 1;

    // weight image (degradation baked in by golden.py); loaded by the
    // testbench context under SIM_LOAD — synthesis replaces this with a
    // real programming port later.
    reg signed [WW-1:0] wmem [0:ROWS*COLS-1];
`ifdef SIM_LOAD
    initial $readmemh("golden_w.memh", wmem);
`endif

    reg [7:0] c;                  // column counter
    reg [$clog2(ROWS)-1:0] r;     // row counter (emit)
    reg [1:0]  state;             // 0 idle, 1 mac, 2 emit
    reg signed [39:0] acc [0:ROWS-1];

    // DAC: floor-quantize to the symmetric-odd grid, then dequantize
    function automatic signed [WW-1:0] dac_q(input signed [WW-1:0] v);
        reg signed [WW-1:0] q;
        q = v >>> SHR_IN;
        if (q >  L_IN) q =  L_IN;
        if (q < -L_IN) q = -L_IN;
        dac_q = q <<< SHR_IN;
    endfunction

    // ADC on the accumulator
    function automatic signed [39:0] adc_q(input signed [39:0] v);
        reg signed [39:0] q;
        q = v >>> OUT_SHIFT;
        if (q >  L_ADC) q =  L_ADC;
        if (q < -L_ADC) q = -L_ADC;
        adc_q = q <<< OUT_SHIFT;
    endfunction

    genvar gi;
    generate
        for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_row
            wire signed [WW-1:0]   wr  = wmem[gi*COLS + c];
            wire signed [WW-1:0]   xq  = dac_q(x);
            wire signed [24:0]     xg_f = xq * $signed({1'b0, g});  // 16b x 9b
            wire signed [17:0]     xg  = $signed(xg_f[24:7]);       // >>>7, exact
            wire signed [33:0]     prod = wr * xg;                  // 16b x 18b
            always @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    acc[gi] <= 40'sd0;
                end else if (state == 2'd1 && xv) begin
                    acc[gi] <= acc[gi] + prod;
                end else if (state == 2'd0 && start) begin
                    acc[gi] <= 40'sd0;
                end
            end
        end
    endgenerate

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state <= 2'd0; c <= 8'd0; r <= 0;
        end else begin
            case (state)
                2'd0: if (start) begin
                    state <= 2'd1; c <= 8'd0;
                end
                2'd1: if (xv) begin
                    c <= c + 8'd1;
                    if (c == COLS-1) begin
                        state <= 2'd2; r <= 0;
                    end
                end
                2'd2: begin
                    if (r == ROWS-1) state <= 2'd0;
                    r <= r + 1'b1;
                end
            endcase
        end
    end

    assign yv   = (state == 2'd2);
    assign yrow = r;
    assign y    = adc_q(acc[r]);
    assign done = (state == 2'd2) && (r == ROWS-1);

endmodule
