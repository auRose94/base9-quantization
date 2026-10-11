"""Full-top simulation of gate0sd_top: the Gate-0 stimulus streams from a
modeled SPI-mode SD card.

The model implements the card side of the reader: latch MOSI on SCK rises,
drive MISO ahead of the reader's safe sample point, answer the SDHC/SDXC init
handshake, and for CMD17 serve each complete block (stuffing + 0xFE token +
512 payload bytes + CRC) as a pre-queued bit stream. Payload =
partner/ninths_sd.bin (partner/ninths_sd_image.py mkimage).

Verification: reader state history sane, card->BRAM fill byte-identical vs
the npz source artifacts, and the gate verdict = G0 PASS (fill check skipped
with an explicit error line if the card path reports G0 SERR).

Run from fpga/sim:
  make SIM=verilator TOP=gate0sd_top MOD=tb_gate0sd \
       RTL="../synth/gate0sd_top.sv ../synth/uart_tx.sv \
            ../rtl/f32_pkg.sv ../rtl/k96_vec_dot_v2.sv \
            ../rtl/spi_byte_master.sv ../rtl/sd_spi_reader.sv"
"""
import os
from collections import deque
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

os.chdir(Path(__file__).resolve().parents[1] / "partner")  # card image CWD

BIT_CYCLES = 234  # 117 clk13 cycles = 234 clk pin cycles per tx bit
BASE_LBA = 2048
N_BLOCKS = 20
LBA = 512


class SdCardModel:
    "SPI-mode SDHC/SDXC responder over the card pads."

    def __init__(self, dut, payload):
        assert len(payload) == N_BLOCKS * LBA
        self.dut = dut
        self.payload = payload
        self.bits = []
        self.resp = deque()      # MISO bit queue
        self.frame = []
        self.fhist = []
        self.acmd41_seen = 0
        self.wr_active = None    # LBA being written (SPI write capture)
        self.wr_i = -1
        self.wr_bytes = bytearray()
        self.wr_done_logs = []

    def queue_byte(self, v):
        for k in range(7, -1, -1):
            self.resp.append((v >> k) & 1)

    def queue_r1(self, r1, arg=None):
        self.queue_byte(0xFF)    # NCR stuffing
        self.queue_byte(r1)
        if arg is not None:
            for b in arg:
                self.queue_byte(b)

    def on_frame(self, f):
        assert f[0] >> 6 == 1, f
        cmd = f[0] & 0x3F
        arg = (f[1] << 24) | (f[2] << 16) | (f[3] << 8) | f[4]
        if cmd == 0:
            self.queue_r1(0x01)
        elif cmd == 8:
            assert arg == 0x1AA
            self.queue_r1(0x01, arg=(0x00, 0x00, 0x01, 0xAA))
        elif cmd == 55:
            self.queue_r1(0x01)
        elif cmd == 41:
            self.acmd41_seen += 1
            self.queue_r1(0x01 if self.acmd41_seen < 4 else 0x00)
        elif cmd == 58:
            self.queue_r1(0x00, arg=(0xC0, 0xFF, 0x80, 0x00))  # CCS=1
        elif cmd == 24:
            rel = arg - BASE_LBA
            assert 0 <= rel < N_BLOCKS, (arg, rel)
            self.wr_active = rel                 # model captures card writes
            self.wr_i = -1
            self.wr_bytes = bytearray()
            self.queue_r1(0x00)
        elif cmd == 17:
            self.dut._log.info("CMD17 lba=%d", arg)
            rel = arg - BASE_LBA
            assert 0 <= rel < N_BLOCKS, (arg, self.dut.sd_blk.value, self.dut.sd_lba.value)
            self.queue_r1(0x00)
            for _ in range(4):
                self.queue_byte(0xFF)              # read-latency stuffing
            self.queue_byte(0xFE)                  # data token
            for b in self.payload[rel * LBA: (rel + 1) * LBA]:
                self.queue_byte(b)
            self.queue_byte(0x9A)                  # crc16 (unchecked)
            self.queue_byte(0x6D)
        else:
            raise AssertionError(f"unexpected cmd {cmd} arg {arg:#x}")

    async def run(self):
        prev_sck = 0
        while True:
            await RisingEdge(self.dut.clk)
            try:
                cs = int(self.dut.sd_cs_n.value)
                sck = int(self.dut.sd_sck.value)
                mosi = int(self.dut.sd_mosi.value)
            except Exception:
                continue
            if cs == 1:
                self.bits = []                     # frames never span CS high
                self.dut.sd_miso.value = 1
                prev_sck = 0
                continue
            if prev_sck == 0 and sck == 1:
                if self.resp:
                    self.dut.sd_miso.value = self.resp.popleft()
                else:
                    self.dut.sd_miso.value = 1
                self.bits.append(mosi)
                if len(self.bits) == 8:
                    b = 0
                    for i, bit in enumerate(self.bits):
                        b |= bit << (7 - i)
                    self.bits = []
                    if self.wr_active is not None:
                        if self.wr_i == -1:
                            # pre-token: polls and NCR stuffing, all 0xFF;
                            # the first 0xFE after CMD24's R1 is the token
                            if b == 0xFE:
                                self.wr_i = 0
                            else:
                                assert b == 0xFF, f"expected FF pre-token, got {b:#x}"
                        elif self.wr_i < 512:
                            self.wr_bytes.append(b)
                            self.wr_i += 1
                        elif self.wr_i < 514:
                            self.wr_i += 1    # dummy CRC16 bytes
                        else:
                            # write done: payload + CRC consumed
                            print(f"  MODEL-RESPOND: LBA {self.wr_active} "
                                  f"bytes {len(self.wr_bytes)} queued 05+busy")
                            base = (self.wr_active) * 512
                            got = bytes(self.wr_bytes)
                            want = self.payload[base:base+512]
                            assert got == want, (
                                f"write mismatch LBA {self.wr_active}: "
                                f"len {len(got)} vs {len(want)}; first diff at "
                                f"{next((i for i in range(512) if got[i] != want[i]), None)}, "
                                f"got {got[:12].hex()} want {want[:12].hex()}, "
                                f"tail {got[-8:].hex()} vs {want[-8:].hex()}")
                            self.wr_done_logs.append(self.wr_active)
                            self.queue_byte(0x05)        # data accepted
                            for _ in range(3):
                                self.queue_byte(0xFF)
                            self.wr_active = None
                            self.wr_i = -1
                    # a command frame STARTS with a byte whose top two bits
                    # are 01; other transactions (R1 polls, data fills, write
                    # payloads) are invisible to the frame assembly -- exactly
                    # how a real card finds its frames. Writes lock the
                    # assembly out entirely (data bytes can look like frames).
                    if self.wr_active is None:
                        if len(self.frame) == 0:
                            if (b >> 6) == 0b01:
                                self.frame.append(b)
                        else:
                            self.frame.append(b)
                            if len(self.frame) == 6:
                                f = self.frame
                                self.frame = []
                                self.fhist.append(f[0] & 0x3F)
                                if len(self.fhist) > 40000:
                                    self.fhist.pop(0)
                                self.on_frame(f)
            prev_sck = sck


@cocotb.test()
async def gate_from_card(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_b.value = 1
    dut.prov_btn.value = 0            # provisioning key held during boot

    payload = Path("ninths_sd.bin").read_bytes()
    model = SdCardModel(dut, payload)
    cocotb.start_soon(model.run())

    rlog = []

    async def rreqlog():
        last = None
        while True:
            await RisingEdge(dut.clk)
            try:
                v = (int(dut.sdr.st.value), int(dut.sd_read_req.value), int(dut.sd_lba.value),
                     int(dut.sdr.busy.value), int(dut.sd_blk.value))
            except Exception:
                continue
            if v != last:
                rlog.append((len(rlog), v))
                if len(rlog) > 40:
                    rlog.pop(0)
                last = v

    cocotb.start_soon(rreqlog())
    sthist = []

    async def strec():
        last = -1
        while True:
            await RisingEdge(dut.clk)
            try:
                s = int(dut.sdr.st.value)
            except Exception:
                s = -2
            if s != last:
                sthist.append(s)
                if len(sthist) > 20000:
                    sthist.pop(0)
                last = s

    cocotb.start_soon(strec())

    uarts = bytearray()

    async def uart_rx():
        prev = 1
        while True:
            await RisingEdge(dut.clk)
            cur = int(dut.txd.value) if dut.txd.value.is_resolvable else 1
            if cur == 0 and prev == 1 and len(uarts) < 40:
                for _ in range((3 * BIT_CYCLES) // 2):
                    await RisingEdge(dut.clk)
                b = 0
                for i in range(8):
                    if int(dut.txd.value):
                        b |= 1 << i
                    for _ in range(BIT_CYCLES):
                        await RisingEdge(dut.clk)
                uarts.append(b)
            prev = cur

    wire_cap = []
    async def wcapture():
        armed = False
        got = 0
        prev = None
        while True:
            await RisingEdge(dut.clk)
            try:
                stv = int(dut.sdr.st.value)
                v = (int(dut.sd_cs_n.value), int(dut.sd_sck.value),
                     int(dut.sd_mosi.value), int(dut.sd_miso.value),
                     int(dut.sdr.m_done.value))
            except Exception:
                continue
            if stv == 32 or armed:
                armed = True
                if v != prev:
                    wire_cap.append(v)
                    prev = v
                    got += 1
                if got > 150:
                    for w in wire_cap[:90]:
                        print(f"  W({w[0]}{w[1]}{w[2]}{w[3]} md{w[4]})")
                    return

    cocotb.start_soon(wcapture())
    async def wpaywatch():
        n = 0
        while True:
            await RisingEdge(dut.clk)
            try:
                if int(dut.sd_wdata_req.value) and n < 5:
                    n += 1
                    print(f"  wreq#{n}: wpay={int(dut.wpay.value)} pay_q="
                          f"{int(dut.pay_q.value) & 0xFFFFFFFF:08x} sd_wdata="
                          f"{int(dut.sd_wdata.value) & 0xFFFFFFFF:08x} "
                          f"pay[0]={int(dut.pay[0].value) & 0xFFFFFFFF:08x}")
            except Exception:
                pass

    async def wrespwatch():
        seen = 0
        while True:
            await RisingEdge(dut.clk)
            try:
                if int(dut.sdr.st.value) == 33 and seen < 8:
                    seen += 1
                    q = model.resp
                    md = int(dut.sdr.m_done.value)
                    print(f"  WRESPW#{seen}: md={md} m_rx={int(dut.sdr.m_rx.value):#04x} "
                          f"miso={int(dut.sd_miso.value)} qbits={len(q)} "
                          f"wr_i={model.wr_i} wact={model.wr_active} "
                          f"sck={int(dut.sd_sck.value)} mosi={int(dut.sd_mosi.value)}")
            except Exception:
                pass

    cocotb.start_soon(wrespwatch())
    cocotb.start_soon(wpaywatch())
    cocotb.start_soon(uart_rx())

    waited = 0
    while len(uarts) < 9 and waited < 4_000_000:
        await RisingEdge(dut.clk)
        waited += 1
        if waited in (500_000, 1_000_000, 2_000_000, 3_000_000, 3_600_000, 3_700_000):
            try:
                print(f"  probe@{waited}: top.st={int(dut.st.value)} rd.st={int(dut.sdr.st.value)} "
                      f"blk={int(dut.sdr.st.value) and int(dut.sd_blk.value)} lba={int(dut.sd_lba.value)} "
                      f"blkw={int(dut.sd_blkw.value)} wr={int(dut.sd_wr.value)} rreq={int(dut.sd_read_req.value)} "
                      f"wv={int(dut.sd_word_v.value)} rdd={int(dut.sd_rd_done.value)} "
                      f"fast={int(dut.sdr.fast.value)} mslow={int(dut.sdr.m_slow.value)} "
                      f"bcyc={int(dut.sdr.bcyc.value)} "
                      f"rdst={int(dut.sdr.st.value)} rderr={int(dut.sdr.error.value)}")
            except Exception as e:
                print("probe fail:", e)
        if waited == 500_000:
            print("  probe@500k: reader.st:", sthist[-14:])
    assert len(uarts) >= 9, f"no verdict line after {waited} cycles; reader states so far: {sthist[-20:]}"

    line = bytes(uarts)
    vline = line.split(b'\r')[0]
    print("UART line:", vline)
    print("LEDs:", f"{int(dut.led.value):06b}")

    print("req/rdy log:", [(i, f"st{a} rq{b} lba{c} bs{d} blk{e}") for i, (a, b, c, d, e) in rlog])
    if vline.startswith(b"G0 SERR"):
        from collections import Counter
        print("reader state histogram:", dict(Counter(sthist)))
        try:
            print("reader regs: plan", int(dut.sdr.plan.value), "r1q", hex(int(dut.sdr.r1q.value)),
                  "args:", [hex(int(dut.sdr.b[i].value)) for i in (1, 2, 3, 4)],
                  "arg_echo_ok", int(dut.sdr.arg_echo_ok.value),
                  "resp_need", int(dut.sdr.resp_need.value),
                  "poll_i", int(dut.sdr.poll_i.value), "to", int(dut.sdr.to.value))
        except Exception as e:
            print("regs probe failed:", e)
        print("model frames seen:", len(model.fhist), "last cmd ids:", model.fhist[-8:])
        raise AssertionError("card init failed (G0 SERR); see diagnostics above")

    npz = np.load("/tmp/gate0_vecs.npz")

    # byte-exactness of the card -> BRAM fill
    stim_w = []
    exp_w = []
    flg = []
    for i in range(21):
        blob = bytes(npz[f"rec{i}_wblk"].tobytes()) + bytes(npz[f"rec{i}_act"].tobytes())
        stim_w += [int.from_bytes(blob[4 * j: 4 * j + 4], "little") for j in range(118)]
        exp_w.append(int(npz[f"rec{i}_gold"]))
        cid = int(npz[f"rec{i}_case"])
        flg.append(1) if (i == 0 or cid != int(npz[f"rec{i - 1}_case"])) else flg.append(0)
    mism = 0
    for i in range(2478):
        got = int(dut.stim[i].value) & 0xFFFFFFFF
        if got != stim_w[i]:
            mism += 1
            if mism <= 6:
                print(f"  stim[{i}] {got:08x} != {stim_w[i]:08x}")
    fbad = 0
    for i in range(21):
        got = int(dut.exp_mem[i].value) & 0xFFFFFFFF
        if got != (exp_w[i] & 0xFFFFFFFF):
            fbad += 1
            print(f"  exp[{i}] {got:08x} != {exp_w[i] & 0xFFFFFFFF:08x}")
        fg = int(dut.flags_mem[i].value) != 0
        if fg != (flg[i] != 0):
            fbad += 1
            print(f"  flags[{i}] {fg} != {flg[i] != 0}")
    assert mism == 0 and fbad == 0, f"card fill not byte-exact ({mism} stim, {fbad} exp/flags)"
    print("card fill: byte-exact (2478 stim words, 21 exp, 21 flags)")

    if vline.startswith(b"G0 PASS"):
        led_v = int(dut.led.value)
        # pads are active-low: PASS = exactly one pad low (led[5]),
        # led[4] high (dark), the four index pads low on
        assert led_v == 0b010000, f"PASS but LED pads {led_v:06b}"
    elif vline.startswith(b"G0 FAIL"):
        npz_all = bytearray()
        for i in range(21):
            npz_all += bytes(npz[f"rec{i}_wblk"].tobytes()) + bytes(npz[f"rec{i}_act"].tobytes())
        npz_all = bytes(npz_all)
        rec0 = bytes(npz[f"rec0_wblk"]) + bytes(npz[f"rec0_act"])
        print("wm[0..2]:", [f"{int(dut.wm[i].value)&0xFFFFFFFF:08x}" for i in range(3)],
              "want:", [f"{int.from_bytes(rec0[4*i:4*i+4],'little'):08x}" for i in range(3)])
        print("am[0..2]:", [f"{int(dut.am[i].value)&0xFFFFFFFF:08x}" for i in range(3)],
              "want:", [f"{int.from_bytes(rec0[200+4*i:200+4*i+4],'little'):08x}" for i in range(3)])
        print("stim[0..2]:", [f"{int(dut.stim[i].value)&0xFFFFFFFF:08x}" for i in range(3)])
        print("stim[128..130]:", [f"{int(dut.stim[i].value)&0xFFFFFFFF:08x}" for i in range(128, 131)],
              "want:", [f"{int.from_bytes(npz_all[4*i:4*i+4],'little'):08x}" for i in range(128, 131)])
        print("exp[0..2]:", [f"{int(dut.exp_mem[i].value)&0xFFFFFFFF:08x}" for i in range(3)],
              "want:", [f"{int(npz[f'rec{i}_gold'] & 0xFFFFFFFF):08x}" for i in range(3)])
        print("wm[0..2] got/want above; flags[0..1]:",
              [int(dut.flags_mem[i].value) for i in range(2)])
        raise AssertionError(f"on-chip mismatch at record {vline[9:10].decode()}; dumps above")
    else:
        raise AssertionError(f"unexpected line {vline!r}")