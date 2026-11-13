// analog_emu — Tang Nano 20K top: one 64x64 serial synapse tile behind a
// UART command interpreter.  Protocol (115200 8N1, LE multi-byte words):
//   'I'            -> id string "analog-emu-v3\n"
//   'P' + 8256 B   -> program image: 4096 x u16 LE weights, then 64 x u8
//                     gains -> 'K'
//   'X' + 128 B    -> token: 64 x i16 LE (Q4.12) -> run -> 64 x i16 LE yq,
//                     8 B spike word (LE) -> 'K'
//   'D'            -> one drift sweep (~4096 cycles) -> 'K'
//   'V'            -> 64 x i32 LE membranes -> 'K'
// The python host (silicon_gate.py) is the checker: golden files from
// golden.py, diffed word-for-word — the char9 gate, over a wire.
module emu_top #(
    parameter integer UART_DIV = 234      // 27 MHz / 115200
) (
    input  wire       clk,
    input  wire       uart_rxd,
    input  wire       uart_rxd2,
    output wire       uart_txd,
    output wire [5:0] led
);
    localparam int ROWS = 64, COLS = 64;

    // power-on reset: regs without bitstream init (the tile LFSR seed!)
    // get a real reset pulse
    reg [7:0] por_cnt = 8'd0;
    always @(posedge clk) if (!por_done) por_cnt <= por_cnt + 1;
    wire por_done = &por_cnt;
    wire rst_n = por_done;

    // UART — two RX instances: pod-RX pin is a bring-up unknown, so listen
    // on BOTH candidates (70 and 68) and let the protocol filter garbage
    reg  rxv1, rxv2;
    reg  [7:0] rxd1, rxd2;
    uart_rx #(.DIV(UART_DIV)) u_rx1 (
        .clk(clk), .rst_n(rst_n), .rx(uart_rxd),
        .valid(rxv1), .data(rxd1));
    uart_rx #(.DIV(UART_DIV)) u_rx2 (
        .clk(clk), .rst_n(rst_n), .rx(uart_rxd2),
        .valid(rxv2), .data(rxd2));
    wire rx_valid = rxv1 | rxv2;
    wire [7:0] rx_data = rxv1 ? rxd1 : rxd2;
    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy;
    uart_tx #(.DIV(UART_DIV)) u_tx (
        .clk(clk), .rst_n(rst_n), .start(tx_start), .data(tx_data),
        .tx(uart_txd), .busy(tx_busy));

    // tile
    reg         t_we, t_tok_we, t_start, t_drift;
    reg  [12:0] t_waddr;
    reg  [15:0] t_wdata;
    reg  [5:0]  t_tok_addr, t_yq_addr, t_vaddr;
    reg  [15:0] t_tok_data;
    wire        t_done;
    wire [ROWS-1:0] t_spikes;
    wire signed [15:0] t_yq;
    wire signed [31:0] t_vmem;

    synapse_tile_ser #(
        .ROWS(ROWS), .COLS(COLS), .OUT_SHIFT(14), .THRESH(96)
    ) u_tile (
        .clk(clk), .rst_n(rst_n),
        .we(t_we), .waddr(t_waddr), .wdata(t_wdata),
        .tok_we(t_tok_we), .tok_addr(t_tok_addr), .tok_data(t_tok_data),
        .start(t_start), .drift_step(t_drift),
        .done(t_done), .spikes(t_spikes),
        .yq_addr(t_yq_addr), .yq(t_yq),
        .vaddr(t_vaddr), .vmem_out(t_vmem));

    // dispatcher
    localparam [3:0] S_CMD=0, S_ID=1, S_PROG=2, S_TOK=3, S_RUN=4,
                     S_YQ=5, S_YQ2=9, S_VMEM=6, S_DRIFT=7, S_WAITK=8;
    reg [3:0]  state;
    reg [15:0] bcnt;        // byte counter within a payload
    reg [15:0] btotal;
    reg [7:0]  b0;          // low byte of the pair being assembled
    reg        have_b0;
    reg        yq_phase;
    reg [8:0]  ms_cnt = 9'd256;  // first broadcast ~256 ms after boot
    reg [2:0]  id_idx;
    reg [7:0]  out_byte;
    reg        out_req;
    reg [15:0] wait_cnt;

    wire [63:0] spikes_w = t_spikes;

    task tx_byte(input [7:0] b);
        begin
            out_byte <= b;
            out_req  <= 1'b1;
        end
    endtask

    always @(posedge clk) begin
        tx_start <= 1'b0;
        t_we <= 1'b0; t_tok_we <= 1'b0; t_start <= 1'b0; t_drift <= 1'b0;
        if (out_req && !tx_busy) begin
            tx_data <= out_byte; tx_start <= 1'b1; out_req <= 1'b0;
        end
        case (state)
            S_CMD: if (bcnt == 16'd27000) begin
                // 1 ms tick; broadcast the id every 512 ms via ms counter
                bcnt <= 16'd0;
                ms_cnt <= ms_cnt + 1;
                if (&ms_cnt) begin
                    state <= S_ID; id_idx <= 0;
                end
            end else begin
                bcnt <= bcnt + 1;
                if (rx_valid) begin
                case (rx_data)
                        "I": begin state <= S_ID; id_idx <= 0;
                                  bcnt <= 0; ms_cnt <= 0; end
                        "P": begin state <= S_PROG; bcnt <= 0; have_b0 <= 0; end
                        "X": begin state <= S_TOK; bcnt <= 0; have_b0 <= 0; end
                        "D": begin state <= S_DRIFT; t_drift <= 1'b1;
                                   wait_cnt <= 0; end
                        "V": begin state <= S_VMEM; bcnt <= 0; t_vaddr <= 0; end
                        default: ;   // ignore
                    endcase
                end
            end
            S_ID: if (!out_req && !tx_busy) begin
                if (id_idx == 14) begin
                    state <= S_CMD;
                end else begin
                    tx_byte(id_str[111 - id_idx*8 +: 8]);
                    id_idx <= id_idx + 1;
                end
            end
            // program: 8192 W bytes + 64 G bytes
            S_PROG: if (rx_valid) begin
                if (bcnt < 8192) begin
                    if (!have_b0) begin
                        b0 <= rx_data; have_b0 <= 1;
                    end else begin
                        t_we <= 1'b1;
                        t_waddr <= bcnt[15:1];
                        t_wdata <= {rx_data, b0};
                        have_b0 <= 0;
                    end
                end else begin
                    t_we    <= 1'b1;
                    t_waddr <= 13'd4096 + (bcnt - 8192);
                    t_wdata <= {8'd0, rx_data};
                end
                bcnt <= bcnt + 1;
                if (bcnt == 8255) begin
                    state <= S_WAITK;
                    tx_byte("K");
                end
            end
            // token: 128 bytes -> xmem, then run
            S_TOK: if (rx_valid) begin
                if (!have_b0) begin
                    b0 <= rx_data; have_b0 <= 1;
                end else begin
                    t_tok_we   <= 1'b1;
                    t_tok_addr <= bcnt[5:1];
                    t_tok_data <= {rx_data, b0};
                    have_b0    <= 0;
                end
                bcnt <= bcnt + 1;
                if (bcnt == 127) begin
                    state <= S_RUN;
                    t_start <= 1'b1;
                end
            end
            S_RUN: if (t_done) begin
                state <= S_YQ; bcnt <= 0; t_yq_addr <= 0;
            end
            // results: 128 yq bytes + 8 spike bytes + 'K'
            // two-phase per word: set yq_addr, read the registered yq next posedge
            S_YQ: if (!out_req && !tx_busy) begin
                if (bcnt < 128) begin
                    if (yq_phase) begin
                        state <= S_YQ2;   // address settled: emit the word
                    end else if (bcnt[0] == 0) begin
                        t_yq_addr <= bcnt[7:1];
                        yq_phase  <= 1'b1;
                    end else begin
                        out_byte  <= b0;
                        out_req   <= 1'b1;
                    end
                end else if (bcnt < 136) begin
                    out_byte <= spikes_w[bcnt[5:0]*8 +: 8];
                    out_req  <= 1'b1;
                end else begin
                    state <= S_CMD;
                    tx_byte("K");
                end
                bcnt <= bcnt + 1;
            end
            S_DRIFT: begin
                wait_cnt <= wait_cnt + 1;
                if (wait_cnt == 16'd12000) begin   // > 4096 sweep cycles
                    state <= S_WAITK;
                    tx_byte("K");
                end
            end
            S_VMEM: if (!out_req && !tx_busy) begin
                case (bcnt[1:0])
                    2'd0: out_byte <= t_vmem[7:0];
                    2'd1: out_byte <= t_vmem[15:8];
                    2'd2: out_byte <= t_vmem[23:16];
                    2'd3: out_byte <= t_vmem[31:24];
                endcase
                out_req <= 1'b1;
                if (bcnt[1:0] == 2'd3 && bcnt != 255)
                    t_vaddr <= t_vaddr + 1'b1;
                bcnt <= bcnt + 1;
                if (bcnt == 255) begin
                    state <= S_WAITK;
                    tx_byte("K");
                end
            end
            // yq phase 2: address has settled — emit low byte, stash high
            S_YQ2: if (!out_req && !tx_busy) begin
                out_byte  <= t_yq[7:0];
                b0        <= t_yq[15:8];
                out_req   <= 1'b1;
                yq_phase  <= 1'b0;
                state     <= S_YQ;
            end
            S_WAITK: if (!out_req && !tx_busy) state <= S_CMD;
        endcase
    end

    // id string
    wire [111:0] id_str = "analog-emu-v3\n";  // 14 chars, MSB-first

    // LEDs: heartbeat, rx activity, tile busy
    reg [24:0] hb;
    always @(posedge clk) hb <= hb + 1;
    assign led[0] = hb[22];
    assign led[1] = rx_valid;
    assign led[2] = (state != S_CMD) || t_done;
    assign led[3] = uart_rxd;    // candidate pin 70 raw level
    assign led[4] = uart_rxd2;   // candidate pin 68 raw level
    assign led[5] = rxv1 | rxv2; // a validated byte arrived (either pin)
endmodule
