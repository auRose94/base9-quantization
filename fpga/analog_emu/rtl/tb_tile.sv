// analog_emu — tile testbench v2 (port-level only, runs under verilator
// --binary).  Streams tokens, compares outputs + LIF spikes per token and
// membranes at the end, applies drift sweeps between token batches on the
// golden schedule.  Prints "GATE PASS" / "GATE FAIL <n>".
`timescale 1ns/1ps
module tb_tile #(
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int WW = 16,
    parameter int DACB = 8,
    parameter int ADCB = 8,
    parameter int OUT_SHIFT = 11,
    parameter int RN_POW = 3,
    parameter int DRIFT_SHIFT = 6,
    parameter int LIF_EN = 1,
    parameter int LEAK_SHIFT = 3,
    parameter int THRESH = 384,
    parameter int LFSR_SEED = 16'hACE1,
    parameter int NTOK = 8,
    parameter int DRIFT_EVERY = 2   // drift sweep after every k-th token
);
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;

    reg        start = 0, xv = 0, drift_step = 0;
    reg [7:0]  g = 8'd0;  // driven per-column by feed_token
    reg signed [WW-1:0] x = 0;
    reg [$clog2(ROWS)-1:0] vaddr = 0;
    wire       yv, done;
    wire [$clog2(ROWS)-1:0] yrow;
    wire signed [39:0]      y;
    wire [ROWS-1:0]         spikes;
    wire signed [31:0]      vmem_out;

    synapse_tile #(
        .ROWS(ROWS), .COLS(COLS), .WW(WW),
        .DACB(DACB), .ADCB(ADCB), .OUT_SHIFT(OUT_SHIFT),
        .RN_POW(RN_POW), .DRIFT_SHIFT(DRIFT_SHIFT),
        .LIF_EN(LIF_EN), .LEAK_SHIFT(LEAK_SHIFT), .THRESH(THRESH),
        .LFSR_SEED(LFSR_SEED)
    ) dut (
        .clk(clk), .rstn(rstn), .start(start), .xv(xv), .x(x), .g(g),
        .drift_step(drift_step),
        .yv(yv), .yrow(yrow), .y(y), .spikes(spikes), .done(done),
        .vaddr(vaddr), .vmem_out(vmem_out));

    reg [15:0] tokmem [0:NTOK*COLS-1];
    reg [39:0] expmem  [0:NTOK*ROWS-1];
    reg [7:0]  gmem    [0:COLS-1];
    reg [(ROWS+3)/4*4-1:0] spkmem [0:NTOK-1];   // spike words
    reg [31:0] vmemexp [0:ROWS-1];              // final membranes

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
            wait (dut.state == 3'd2);   // v1-proven: level wait, no edge race
            for (r0 = 0; r0 < ROWS; r0 = r0 + 1) begin
                @(negedge clk);
                if (y !== $signed(expmem[ti*ROWS + r0])) begin
                    n_mis = n_mis + 1;
                    if (n_mis <= 8)
                        $display("MISMATCH tok %0d row %0d exp=%h got=%h",
                                 ti, r0, expmem[ti*ROWS + r0], y);
                end
            end
            // spike_vec is stable here (last row's emit posedge passed)
            if (spikes !== spkmem[ti]) begin
                n_mis = n_mis + 1;
                $display("SPIKE MISMATCH tok %0d exp=%h got=%h",
                         ti, spkmem[ti], spikes);
            end
        end
    endtask

    task drift_sweep;
        integer d0;
        begin
            @(negedge clk); drift_step <= 1'b1;
            for (d0 = 0; d0 < ROWS; d0 = d0 + 1) @(negedge clk);
            drift_step <= 1'b0;
            @(negedge clk);
        end
    endtask

    initial begin
        $readmemh("golden_g.memh", gmem);
        $readmemh("golden_tokens.hex", tokmem);
        $readmemh("golden_expected.hex", expmem);
        $readmemh("golden_spikes.hex", spkmem);
        $readmemh("golden_vmem.hex", vmemexp);
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
            if ((t + 1) % DRIFT_EVERY == 0 && t + 1 < NTOK)
                drift_sweep;
        end

        // final membrane scanback
        for (trow = 0; trow < ROWS; trow = trow + 1) begin
            vaddr <= trow[$clog2(ROWS)-1:0];
            @(negedge clk);
            if (vmem_out !== $signed(vmemexp[trow])) begin
                n_mis = n_mis + 1;
                if (n_mis <= 8)
                    $display("VMEM MISMATCH row %0d exp=%h got=%h",
                             trow, vmemexp[trow], vmem_out);
            end
        end

        if (n_mis == 0) $display("GATE PASS (%0d tokens x %0d rows, bit-exact)",
                                 NTOK, ROWS);
        else            $display("GATE FAIL %0d", n_mis);
        $finish;
    end
endmodule
