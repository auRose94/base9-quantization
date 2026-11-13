// analog_emu — SERIAL tile testbench: same goldens as the parallel gate,
// adapted to the tok_we/load + readback interface.  Proves the silicon
// bring-up tile (one multiplier, different schedule) is bit-exact with the
// python reference before it ever sees the SD flow.
`timescale 1ns/1ps
module tb_tile_ser #(
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
    parameter int THRESH = 96,
    parameter int LFSR_SEED = 16'hACE1,
    parameter int NTOK = 8,
    parameter int DRIFT_EVERY = 2
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg               we = 0, tok_we = 0, start = 0, drift_step = 0;
    reg  [12:0]       waddr = 0;
    reg  [15:0]       wdata = 0;
    reg  [5:0]        tok_addr = 0;
    reg  [15:0]       tok_data = 0;
    reg  [5:0]        yq_addr = 0, vaddr = 0;
    wire              done;
    wire [ROWS-1:0]   spikes;
    wire signed [15:0] yq;
    wire signed [31:0] vmem_out;

    synapse_tile_ser #(
        .ROWS(ROWS), .COLS(COLS), .WW(WW), .DACB(DACB), .ADCB(ADCB),
        .OUT_SHIFT(OUT_SHIFT), .RN_POW(RN_POW), .DRIFT_SHIFT(DRIFT_SHIFT),
        .LIF_EN(LIF_EN), .LEAK_SHIFT(LEAK_SHIFT), .THRESH(THRESH),
        .LFSR_SEED(LFSR_SEED)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .we(we), .waddr(waddr), .wdata(wdata),
        .tok_we(tok_we), .tok_addr(tok_addr), .tok_data(tok_data),
        .start(start), .drift_step(drift_step),
        .done(done), .spikes(spikes),
        .yq_addr(yq_addr), .yq(yq),
        .vaddr(vaddr), .vmem_out(vmem_out));

    // golden images: 16-bit words (weights/tokens/yq), spikes as words
    reg [15:0] g_w [0:ROWS*COLS-1];
    reg [7:0]  g_g [0:COLS-1];
    reg [15:0] g_tok [0:NTOK*COLS-1];
    reg [15:0] g_yq [0:NTOK*ROWS-1];
    reg [(ROWS+3)/4*4-1:0] g_spk [0:NTOK-1];
    reg [31:0] g_vmem [0:ROWS-1];

    integer n_mis = 0, t, i, r0;

    // load goldens into local arrays (from the flat hex files)
    reg [31:0] word;
    integer wi;
    task load_pair(input [8*20-1:0] fname_w, input [8*20-1:0] fname_unused);
        begin end
    endtask

    initial begin
        #5000000;
        $display("HB t=%0t st=%0d c=%0d r=%0d done=%b", $time,
                 dut.state, dut.c, dut.r, done);
        #5000000;
        $display("HB t=%0t st=%0d c=%0d r=%0d done=%b", $time,
                 dut.state, dut.c, dut.r, done);
        $finish;
    end

    initial begin
        repeat (4) @(negedge clk);
        rst_n = 1;                 // (was missing: tile sat in reset forever)
        repeat (4) @(negedge clk);
        // weights: golden_w.memh is ROWS lines of ROWS 4-hex words
        $readmemh("golden_w16.memh", g_w);
        $readmemh("golden_g.memh", g_g);
        $readmemh("golden_tokens16.memh", g_tok);
        $readmemh("golden_yq16.memh", g_yq);
        $readmemh("golden_spikes16.memh", g_spk);
        $readmemh("golden_vmem.hex", g_vmem);

        // program the tile image
        @(negedge clk);
        for (i = 0; i < ROWS*COLS; i = i + 1) begin
            we <= 1; waddr <= i[12:0]; wdata <= g_w[i];
            @(negedge clk);
        end
        for (i = 0; i < COLS; i = i + 1) begin
            we <= 1; waddr <= 13'h1000 + i[12:0]; wdata <= {8'd0, g_g[i]};
            @(negedge clk);
        end
        we <= 0;

        for (t = 0; t < NTOK; t = t + 1) begin
            // load token
            for (i = 0; i < COLS; i = i + 1) begin
                tok_we   <= 1;
                tok_addr <= i[5:0];
                tok_data <= g_tok[t*COLS + i];
                @(negedge clk);
            end
            tok_we <= 0;
            // run
            start <= 1; @(negedge clk); start <= 0;
            wait (done);
            @(negedge clk);
            if (t == 0) begin
                $display("ACC9 %h ACC15 %h", dut.acc[9], dut.acc[15]);
                for (i = 144; i < 160; i = i + 1)
                    $display("WM9 %0d %h", i, dut.wmem[i]);
                $display("WM %h %h %h | XGB %h %h | X %h | G %h",
                         dut.wmem[0], dut.wmem[1], dut.wmem[16],
                         dut.xgb[0], dut.xgb[1], dut.xmem[0], dut.gmem[0]);
            end
            // compare yq
            for (r0 = 0; r0 < ROWS; r0 = r0 + 1) begin
                yq_addr <= r0[5:0];
                @(negedge clk);
                if (yq !== $signed(g_yq[t*ROWS + r0])) begin
                    n_mis = n_mis + 1;
                    if (n_mis <= 8)
                        $display("YQ MISMATCH tok %0d row %0d exp=%h got=%h",
                                 t, r0, g_yq[t*ROWS + r0], yq);
                end
            end
            if (spikes !== g_spk[t]) begin
                n_mis = n_mis + 1;
                $display("SPIKE MISMATCH tok %0d exp=%h got=%h",
                         t, g_spk[t], spikes);
            end
            if ((t + 1) % DRIFT_EVERY == 0 && t + 1 < NTOK) begin
                drift_step <= 1; @(negedge clk); drift_step <= 0;
                wait (state_idle);
                @(negedge clk);
                if (t == 1)
                    for (i = 112; i < 128; i = i + 1)
                        $display("DWM %0d %h", i, dut.wmem[i]);
            end
        end

        // membrane scanback
        for (r0 = 0; r0 < ROWS; r0 = r0 + 1) begin
            vaddr <= r0[5:0];
            @(negedge clk);
            if (vmem_out !== $signed(g_vmem[r0])) begin
                n_mis = n_mis + 1;
                if (n_mis <= 8)
                    $display("VMEM MISMATCH row %0d exp=%h got=%h",
                             r0, g_vmem[r0], vmem_out);
            end
        end

        if (n_mis == 0) $display("GATE PASS (serial, %0d tokens x %0d rows)",
                                 NTOK, ROWS);
        else            $display("GATE FAIL %0d", n_mis);
        $finish;
    end

    wire state_idle = (dut.state == 3'd0);
endmodule
