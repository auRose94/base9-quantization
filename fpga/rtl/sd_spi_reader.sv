// SPI-mode SDHC/SDXC card reader over spi_byte_master.
//
// Init handshake (slow clock): ~1 ms CS-high settle, 80 clocks of 0xFF with
// CS high, CMD0 (R1=0x01, budgeted), CMD8 arg 0x1AA (R7 echo check), a
// CMD55+ACMD41 (HCS=1) loop until R1 clears (< 1 s budget -- a global cycle
// counter runs while busy), then CMD58 capturing OCR bit30 = CCS (block
// addressing; a 256 GB SDXC card sets it; SDSC is refused). SPI-mode CRC
// checking is off by default, so frame CRCs are the convention constants
// (0x95 for CMD0, 0x87 for CMD8, 0x01 elsewhere).
//
// Reads (fast clock): caller pulses read_req with `lba`; the reader sends
// CMD17, requires R1 == 0x00, waits for the 0xFE data token (NAC budget
// ~333 ms), streams the 512 bytes as little-endian words (128 word_valid
// pulses per block), eats the 2 CRC bytes, raises CS with one idle byte and
// pulses rd_done. CS stays low from the CMD17 frame through the data phase.
//
// error latches on any fault; done pulses once when init ends (success or
// fatal) and again when the reader fully exits; rd_done pulses per block and
// the caller checks error.
module sd_spi_reader (
  input  wire        clk, rst_n,
  // card wires
  output reg         spi_cs_n,
  input  wire        miso,
  output wire        sck,
  output reg         mosi,
  // control
  input  wire        start,
  input  wire        read_req,
  input  wire        write_req,    // CMD24: caller holds lba + supplies wdata
  input  wire [31:0] lba,
  output reg         wdata_req,    // 1-cycle pulse per 512-B word request
  input  wire [31:0] wdata,        // caller returns the word after the pulse
  output reg         busy,
  output reg         done,
  output reg         error,
  // block stream
  output reg         word_valid,
  output reg [31:0]  word_out,
  output reg         rd_done,
  output wire [5:0]  dbg_st       // debug telegraph: FSM state
);
  // ---------------- byte master ----------------
  reg        m_go, m_slow;
  reg [7:0]  m_tx;
  reg        m_done;
  reg [7:0]  m_rx;
  reg        m_busy;

  spi_byte_master engine (
    .clk(clk), .rst_n(rst_n),
    .byte_go(m_go), .tx(m_tx), .slow(m_slow),
    .busy(m_busy), .done(m_done), .rx(m_rx),
    .sck(sck), .mosi(mosi), .miso(miso)
  );

  localparam S_IDLE=0,  S_PSET=1,  S_C80=2,   S_C80W=3,
             S_F=4,     S_FW=5,    S_P=6,    S_PW=7,
             S_RESP=8,  S_RESPW=9, S_AFTER=10, S_CSW=11, S_CSWW=12,
             S_RRDY=13, S_TOK=14,  S_TOKW=15, S_DATA=16, S_DATAW=17,
             S_CRC=18,  S_CRCW=19, S_RDEND=20, S_RDENDW=21, S_EXIT=22,
             S_WIDL=23, S_WTOK=24, S_WDAT=25, S_WDAT0=26, S_WDAT1=27,
             S_WDAT2=28, S_WDAT3=29, S_WCRC1=30, S_WCRC2=31, S_WRESP=32,
             S_WRESPW=33, S_WBUSY=34;
  reg [5:0] st;
  reg [5:0] post;        // state to jump to after the CS resync byte

  localparam PL_CMD0=0, PL_CMD8=1, PL_CMD55=2, PL_CMD41=3,
             PL_CMD58=4, PL_CMD17=5, PL_CMD24=6;
  reg [2:0] plan;

  reg [5:0]  cmdsel;
  reg [31:0] arg;
  reg [7:0]  crc8;
  reg [47:0] frame;
  reg [3:0]  txi;
  reg [5:0]  poll_i;
  reg [1:0]  resp_need;  // R7/R3 = 4 arg bytes pending (nonzero flag)
  reg [1:0]  resp_cnt;   // byte position 0..3 being captured
  reg [7:0]  r1q;
  reg [7:0]  b1, b2, b3, b4;   // response arg big-endian bytes
  reg        ccs, arg_echo_ok;
  reg        fast;
  reg [31:0] to;         // global cycles while busy
  reg [31:0] toc;        // token-wait deadline
  reg [31:0] bcyc;       // byte counter within the data phase
  reg [31:0] wsh;
  reg [31:0] lba_q;

  always @* frame = {2'b01, cmdsel, arg, crc8};

  localparam integer PSET_CYC    = 27000;    // ~1 ms
  localparam integer ACMD_BUDGET = 27000000; // ~1 s of slow-clock init
  localparam integer TOK_BUDGET  = 9000000;  // ~333 ms NAC (reads)
  localparam integer WR_BUDGET   = 27000000; // ~1 s per phase: SDXC writes can
                                             // sit busy for hundreds of ms

  wire bccs = b1[6];       // OCR[31:24] bit 30 after the 4 bytes arrive

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; busy <= 1'b0; done <= 1'b0; error <= 1'b0;
      spi_cs_n <= 1'b1; word_valid <= 1'b0; rd_done <= 1'b0; word_out <= 32'd0;
      m_go <= 1'b0; m_slow <= 1'b1; m_tx <= 8'hFF;
      plan <= PL_CMD0; cmdsel <= 6'd0; arg <= 32'd0; crc8 <= 8'h01;
      txi <= 4'd0; poll_i <= 6'd0; resp_need <= 2'd0; resp_cnt <= 2'd0;
      r1q <= 8'hFF; b1 <= 0; b2 <= 0; b3 <= 0; b4 <= 0;
      ccs <= 1'b0; arg_echo_ok <= 1'b0; fast <= 1'b0; post <= S_F;
      to <= 32'd0; toc <= 32'd0; bcyc <= 32'd0; wsh <= 32'd0; lba_q <= 32'd0;
    end else begin
      m_go <= 1'b0;
      done <= 1'b0;
      word_valid <= 1'b0;
      rd_done <= 1'b0;
      wdata_req <= 1'b0;
      if (busy) to <= to + 32'd1;
      case (st)
        S_IDLE: if (start) begin
          busy <= 1'b1; to <= 32'd0; st <= S_PSET;
        end
        S_PSET: if (to >= PSET_CYC) begin
          txi <= 4'd0; st <= S_C80;
        end
        S_C80: begin                      // 80 clocks, CS high, slow regime
          m_tx <= 8'hFF; m_slow <= 1'b1; m_go <= 1'b1; st <= S_C80W;
        end
        S_C80W: if (m_done) begin
          if (txi == 4'd9) begin
            txi <= 4'd0;
            plan <= PL_CMD0; cmdsel <= 6'd0; arg <= 32'd0; crc8 <= 8'h95;
            resp_need <= 2'd0;
            st <= S_F;
          end else begin
            txi <= txi + 4'd1; st <= S_C80;
          end
        end
        S_F: begin                        // frame byte txi, CS low
          spi_cs_n <= 1'b0;
          m_slow <= ~fast;   // fast flag -> NOT-slow selector (init = slow)
          m_tx <= frame[47 - 8 * txi -: 8];
          m_go <= 1'b1;
          st <= S_FW;
        end
        S_FW: if (m_done) begin
          if (txi == 4'd5) begin
            poll_i <= 6'd0; st <= S_P;
          end else begin
            txi <= txi + 4'd1; st <= S_F;
          end
        end
        S_P: begin                        // poll for a non-0xFF response byte
          m_tx <= 8'hFF; m_go <= 1'b1; st <= S_PW;
        end
        S_PW: if (m_done) begin
          poll_i <= poll_i + 6'd1;
          if (m_rx != 8'hFF) begin
            r1q <= m_rx;
            if (resp_need != 2'd0) begin
              resp_need <= 2'd0;
              resp_cnt <= 2'd0;
              st <= S_RESP;
            end else st <= S_AFTER;
          end else if (poll_i == 6'h3F) begin
            error <= 1'b1; post <= S_EXIT; spi_cs_n <= 1'b1;
            m_tx <= 8'hFF; m_go <= 1'b1;
            st <= S_CSW;
          end else st <= S_P;
        end
        S_RESP: begin
          m_tx <= 8'hFF; m_go <= 1'b1; st <= S_RESPW;
        end
        S_RESPW: if (m_done) begin        // capture arg bytes 1..4 big-endian
          case (resp_cnt)
            2'd0: b1 <= m_rx;
            2'd1: b2 <= m_rx;
            2'd2: b3 <= m_rx;
            default: b4 <= m_rx;
          endcase
          if (resp_cnt == 2'd3) st <= S_AFTER;
          else begin
            resp_cnt <= resp_cnt + 2'd1;
            st <= S_RESP;
          end
        end
        S_AFTER: begin                    // R1 captured: dispatch `plan`
          if (plan == PL_CMD24) begin
            if (r1q == 8'h00) begin
              toc <= to + WR_BUDGET;      // SDXC write phases may sit long
              st <= S_WIDL;               // CS stays LOW through the write
            end else begin
              error <= 1'b1; post <= S_RDEND;
              spi_cs_n <= 1'b1;
              m_tx <= 8'hFF; m_go <= 1'b1;
              st <= S_CSW;
            end
          end else if (plan == PL_CMD17) begin
            if (r1q == 8'h00) begin
              toc <= to + TOK_BUDGET;
              st <= S_TOK;                // CS stays LOW through the block
            end else begin
              error <= 1'b1; post <= S_RDEND;
              spi_cs_n <= 1'b1;
              m_tx <= 8'hFF; m_go <= 1'b1;
              st <= S_CSW;
            end
          end else begin
            spi_cs_n <= 1'b1;             // resync CS between commands
            m_tx <= 8'hFF; m_go <= 1'b1;
            case (plan)
              PL_CMD0: begin
                if (r1q == 8'h01) begin
                  plan <= PL_CMD8; cmdsel <= 6'd8; arg <= 32'h1AA; crc8 <= 8'h87;
                  resp_need <= 2'd1;
                  txi <= 4'd0;
                  post <= S_F;
                end else if (to < ACMD_BUDGET) begin
                  plan <= PL_CMD0; cmdsel <= 6'd0; arg <= 32'd0; crc8 <= 8'h95;
                  resp_need <= 2'd0;
                  txi <= 4'd0;
                  post <= S_F;
                end else begin
                  error <= 1'b1; post <= S_EXIT;
                end
              end
              PL_CMD8: begin
                arg_echo_ok <= (r1q == 8'h01) && (b4 == 8'hAA);
                plan <= PL_CMD55; cmdsel <= 6'd55; arg <= 32'd0; crc8 <= 8'h01;
                resp_need <= 2'd0;
                txi <= 4'd0;
                post <= S_F;
              end
              PL_CMD55: begin
                plan <= PL_CMD41; cmdsel <= 6'd41;
                arg <= arg_echo_ok ? 32'h40000000 : 32'h00000000;
                crc8 <= 8'h01; resp_need <= 2'd0;
                txi <= 4'd0;
                post <= S_F;
              end
              PL_CMD41: begin
                if (r1q == 8'h00) begin         // card is now addressable
                  fast <= 1'b1;
                  plan <= PL_CMD58; cmdsel <= 6'd58; arg <= 32'd0; crc8 <= 8'h01;
                  resp_need <= 2'd1;
                  txi <= 4'd0;
                  post <= S_F;
                end else if (to < ACMD_BUDGET) begin
                  plan <= PL_CMD55; cmdsel <= 6'd55; arg <= 32'd0; crc8 <= 8'h01;
                  resp_need <= 2'd0;
                  txi <= 4'd0;
                  post <= S_F;
                end else begin
                  error <= 1'b1; post <= S_EXIT;
                end
              end
              PL_CMD58: begin
                if (r1q == 8'h00 && bccs) begin   // SDHC/SDXC confirmed
                  ccs <= 1'b1;
                  busy <= 1'b0; done <= 1'b1;
                  post <= S_RRDY;
                end else begin
                  error <= 1'b1; post <= S_EXIT;
                end
              end
              default: begin
                error <= 1'b1; post <= S_EXIT;
              end
            endcase
            st <= S_CSW;
          end
        end
        S_CSW: st <= S_CSWW;              // idle byte with CS high, in flight
        S_CSWW: if (m_done) begin
          if (post == S_RRDY) st <= S_RRDY;
          else st <= post;
        end
        S_RRDY: if (!busy && read_req) begin
          lba_q <= lba;
          plan <= PL_CMD17; cmdsel <= 6'd17; arg <= lba; crc8 <= 8'h01;
          resp_need <= 2'd0;
          txi <= 4'd0;
          busy <= 1'b1;
          st <= S_F;
        end else if (!busy && write_req) begin
          lba_q <= lba;
          plan <= PL_CMD24; cmdsel <= 6'd24; arg <= lba; crc8 <= 8'h01;
          resp_need <= 2'd0;
          txi <= 4'd0;
          bcyc <= 32'd0;              // reused: word index in the block
          busy <= 1'b1;
          st <= S_F;
        end
        S_TOK: if (to >= toc) begin
          error <= 1'b1; post <= S_RDEND;
          spi_cs_n <= 1'b1;
          m_tx <= 8'hFF; m_go <= 1'b1;
          st <= S_CSW;
        end else begin
          m_tx <= 8'hFF; m_go <= 1'b1; st <= S_TOKW;
        end
        S_TOKW: if (m_done) begin
          if (m_rx == 8'hFE) begin
            bcyc <= 32'd0; wsh <= 32'd0;
            st <= S_DATA;
          end else if (m_rx != 8'hFF) begin
            error <= 1'b1; post <= S_RDEND;
            spi_cs_n <= 1'b1;
            m_tx <= 8'hFF; m_go <= 1'b1;
            st <= S_CSW;
          end else st <= S_TOK;
        end
        S_DATA: begin
          m_tx <= 8'hFF; m_go <= 1'b1; st <= S_DATAW;
        end
        S_DATAW: if (m_done) begin
          wsh <= {m_rx, wsh[31:8]};          // first byte lands in the LSB (LE)
          bcyc <= bcyc + 32'd1;
          if (bcyc[1:0] == 2'd3) begin
            word_out <= {m_rx, wsh[31:8]};
            word_valid <= 1'b1;
          end
          if (bcyc == 32'd511) st <= S_CRC;
          else st <= S_DATA;
        end
        S_CRC: begin
          m_tx <= 8'hFF; m_go <= 1'b1; st <= S_CRCW;
        end
        S_CRCW: if (m_done) begin
          if (bcyc == 32'd513) begin
            spi_cs_n <= 1'b1;
            m_tx <= 8'hFF;
            m_go <= 1'b1;
            st <= S_RDEND;
          end else begin
            bcyc <= bcyc + 32'd1;
            st <= S_CRC;
          end
        end
        // ------------------------- CMD24 write block --------------------
        // After CMD24's R1: one 0xFF idle byte (NCR), the 0xFE start token,
        // 512 data bytes (little-endian words fetched through wdata_req ->
        // wdata; the caller holds the word stable across its four exchanges),
        // a dummy CRC16 (cards default CRC-off), then poll the data-response
        // token ((rx & 0x1F) == 5 accepted) and the busy state (rx == 0x00).
        S_WIDL: begin
          m_tx <= 8'hFF; m_go <= 1'b1; bcyc <= 32'd0; st <= S_WTOK;
        end
        S_WTOK: begin
          m_tx <= 8'hFE; m_go <= 1'b1; st <= S_WDAT;
        end
        S_WDAT: begin
          wdata_req <= 1'b1;
          st <= S_WDAT0;
        end
        S_WDAT0: if (m_done) begin        // FE byte completed here
          m_tx <= wdata[7:0];
          m_go <= 1'b1;
          st <= S_WDAT1;
        end
        S_WDAT1: if (m_done) begin
          m_tx <= wdata[15:8];
          m_go <= 1'b1;
          st <= S_WDAT2;
        end
        S_WDAT2: if (m_done) begin
          m_tx <= wdata[23:16];
          m_go <= 1'b1;
          st <= S_WDAT3;
        end
        S_WDAT3: if (m_done) begin
          m_tx <= wdata[31:24];
          m_go <= 1'b1;
          bcyc <= bcyc + 32'd1;
          if (bcyc == 32'd127) st <= S_WCRC1;
          else st <= S_WDAT;
        end
        S_WCRC1: if (m_done) begin
          m_tx <= 8'hAA;                  // dummy CRC16, part 1
          m_go <= 1'b1;
          st <= S_WCRC2;
        end
        S_WCRC2: if (m_done) begin
          m_tx <= 8'h6D;                  // dummy CRC16, part 2
          m_go <= 1'b1;
          st <= S_WRESP;
        end
        S_WRESP: if (m_done) begin        // poll the data-response token --
          // EVERY poll exchange's completion is tested here: an accepted
          // card token (0bxxx05) can land on the very first poll.
          m_tx <= 8'hFF;
          m_go <= 1'b1;
          if (m_rx != 8'hFF && (m_rx & 8'h1F) == 8'h05) begin
            toc <= to + TOK_BUDGET;     // budget now covers the busy wait
            st <= S_WBUSY;
          end else if (m_rx != 8'hFF) begin
            error <= 1'b1; post <= S_RDEND;
            spi_cs_n <= 1'b1;
            m_tx <= 8'hFF; m_go <= 1'b1;
            st <= S_CSW;
          end else if (to >= toc) begin
            error <= 1'b1; post <= S_RDEND;
            spi_cs_n <= 1'b1;
            m_tx <= 8'hFF; m_go <= 1'b1;
            st <= S_CSW;
          end else st <= S_WRESP;
        end
        S_WBUSY: if (m_done) begin        // card holds MISO low while flashing
          m_tx <= 8'hFF;
          m_go <= 1'b1;
          if (to >= toc) begin
            error <= 1'b1; post <= S_RDEND;
            spi_cs_n <= 1'b1;
            m_tx <= 8'hFF; m_go <= 1'b1;
            st <= S_CSW;
          end else if (m_rx != 8'h00) begin
            spi_cs_n <= 1'b1;
            m_tx <= 8'hFF; m_go <= 1'b1;
            st <= S_RDEND;
          end else st <= S_WBUSY;
        end
        S_RDEND: begin
          m_tx <= 8'hFF; m_go <= 1'b1;
          st <= S_RDENDW;
        end
        S_RDENDW: if (m_done) begin
          rd_done <= 1'b1;
          busy <= 1'b0;
          spi_cs_n <= 1'b1;
          st <= S_RRDY;
        end
        S_EXIT: begin
          busy <= 1'b0; done <= 1'b1;
          st <= S_IDLE;
        end
        default: st <= S_IDLE;
      endcase
    end
  end
  assign dbg_st = st;
endmodule