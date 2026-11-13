// analog_emu — tile testbench (port-level only, runs under both simulators).
// Streams tokens from golden_tokens.hex (via local readmemh), compares every
// emitted row against golden_expected.hex.  Prints "GATE PASS" / "GATE FAIL"
// — the Makefile greps for it.
`timescale 1ns/1ps
module tb_tile #(
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int WW = 16,
    parameter int DACB = 8,
    parameter int ADCB = 8,
    parameter int OUT_SHIFT = 11,
    parameter int NTOK = 8
);
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;

    reg        start = 0, xv = 0;
    reg [7:0]  g = 8'd0;  // driven per-column by feed_token
    reg signed [WW-1:0] x = 0;
    wire       yv, done;
    wire [$clog2(ROWS)-1:0] yrow;
    wire signed [39:0]      y;

    synapse_tile #(
        .ROWS(ROWS), .COLS(COLS), .WW(WW),
        .DACB(DACB), .ADCB(ADCB), .OUT_SHIFT(OUT_SHIFT)
    ) dut (
        .clk(clk), .rstn(rstn), .start(start), .xv(xv), .x(x), .g(g),
        .yv(yv), .yrow(yrow), .y(y), .done(done));

    reg [15:0] tokmem [0:NTOK*COLS-1];
    reg [39:0] expmem [0:NTOK*ROWS-1];
    reg [7:0]  gmem  [0:COLS-1];

    integer n_mis = 0, t, tcol, trow;

    task feed_token(input integer ti);
        integer c0;
        begin
            @(negedge clk); start <= 1'b1;
            @(negedge clk); start <= 1'b0;
            for (c0 = 0; c0 < COLS; c0 = c0 + 1) begin
                g  <= gmem[c0];
                x  <= $signed(tokmem[ti*COLS + c0]);
                xv <= 1'b1;
                @(negedge clk);
            end
            xv <= 1'b0;
        end
    endtask

    task collect(input integer ti);
        integer r0;
        begin
            @(posedge yv);
            for (r0 = 0; r0 < ROWS; r0 = r0 + 1) begin
                @(negedge clk);
                if (y !== $signed(expmem[ti*ROWS + r0])) begin
                    n_mis = n_mis + 1;
                    if (n_mis <= 8)
                        $display("MISMATCH tok %0d row %0d exp=%h got=%h",
                                 ti, r0, expmem[ti*ROWS + r0], y);
                end
            end
        end
    endtask

    initial begin
        $readmemh("golden_g.memh", gmem);
        $readmemh("golden_tokens.hex", tokmem);
        $readmemh("golden_expected.hex", expmem);
`ifdef TRACE
        $dumpfile("sim_build/tile.vcd");
        $dumpvars(0, tb_tile);
`endif
        repeat (4) @(negedge clk);
        rstn = 1;
        repeat (4) @(negedge clk);

        for (t = 0; t < NTOK; t = t + 1) begin
            fork
                feed_token(t);
                collect(t);
            join
            @(negedge clk);
        end

        if (n_mis == 0) $display("GATE PASS (%0d tokens x %0d rows, bit-exact)",
                                 NTOK, ROWS);
        else            $display("GATE FAIL %0d", n_mis);
        $finish;
    end
endmodule
