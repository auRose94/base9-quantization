"""Full-top simulation of gate0sd_top: provisioning + verify + gate with a
modeled SPI-mode SD card.

Model card side: latches MOSI on SCK rises, drives MISO ahead of the reader's
safe sample point, answers the SDHC/SDXC init handshake, serves CMD17 blocks
from partner/ninths_sd.bin, and CAPTURES CMD24 writes byte-for-byte (asserting
each written block against the image). The design streams 4-byte UART trace
frames (0x55, phase, reader-state, block) while the card path is running and
the verdict line at the end; the tb strips the frames and requires the
verdict.

Run from fpga/sim:
  make SIM=verilator TOP=gate0sd_top MOD=tb_gate0sd \
       RTL="../sim/gowin_bufg_stub.sv ../synth/gate0sd_top.sv ../synth/uart_tx.sv \
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

BIT_CYCLES = 234          # tx bit period in clk-pin cycles (DIV 117 @ clk13)
BASE_LBA = 2048
N_BLOCKS = 20
LBA = 512


class SdCardModel:
    "SPI-mode SDHC/SDXC responder over the card pads."

    def __init__(self, dut, payload):
        self.dut = dut
        self.payload = payload
        self.bits = []
        self.resp = deque()
        self.frame = []
        self.fhist = []
        self.acmd41_seen = 0
        self.wr_active = None    # LBA of an in-progress CMD24 (None otherwise)
        self.wr_i = -1           # -1 = pre-token scan; else data offset
        self.wr_bytes = bytearray()
        self.writes = {}         # lba -> 512 written bytes

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
        elif cmd == 17:
            rel = arg - BASE_LBA
            assert 0 <= rel < N_BLOCKS, arg
            self.queue_r1(0x00)
            for _ in range(4):
                self.queue_byte(0xFF)              # read-latency stuffing
            self.queue_byte(0xFE)                  # data token
            for b in self.payload[rel * LBA:(rel + 1) * LBA]:
                self.queue_byte(b)
            self.queue_byte(0x9A)                  # crc16 (unchecked)
            self.queue_byte(0x6D)
        elif cmd == 24:
            rel = arg - BASE_LBA
            assert 0 <= rel < N_BLOCKS, arg
            self.wr_active = rel
            self.wr_i = -1
            self.wr_bytes = bytearray()
            self.queue_r1(0x00)
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
                        # inside a write: R1 polls (FF...) precede the 0xFE
                        # token; then 512 data bytes and a dummy CRC16
                        if self.wr_i == -1:
                            if b == 0xFE:
                                self.wr_i = 0
                            else:
                                assert b == 0xFF, f"pre-token byte {b:#x}"
                        elif self.wr_i <= 1024 and len(self.wr_bytes) < 512:
                            self.wr_bytes.append(b)
                            self.wr_i += 1
                        elif len(self.wr_bytes) == 512 and self.wr_i < 514:
                            self.wr_i += 1         # dummy CRC16 (2 bytes)
                        else:
                            base = self.wr_active * LBA
                            got, want = bytes(self.wr_bytes), self.payload[base:base + LBA]
                            assert got == want, (
                                f"write mismatch LBA {self.wr_active}: "
                                f"first diff {next((i for i in range(LBA) if got[i] != want[i]), None)}")
                            self.writes[self.wr_active] = got
                            self.wr_active = None
                            self.wr_i = -1
                            self.queue_byte(0x05)  # data accepted
                            for _ in range(3):
                                self.queue_byte(0xFF)
                    else:
                        # a command frame starts 01xxxxxx; other bytes ignored
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

    uarts = bytearray()

    async def uart_rx():
        prev = 1
        while True:
            await RisingEdge(dut.clk)
            cur = int(dut.txd.value) if dut.txd.value.is_resolvable else 1
            if cur == 0 and prev == 1 and len(uarts) < 200:
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

    cocotb.start_soon(uart_rx())

    def strip_traces(b):
        out = bytearray()
        i = 0
        while i < len(b):
            if b[i] == 0x55 and i + 4 <= len(b) and i < 4_000_000:
                # trace frame: 0x55, phase, reader-state, block
                out_note = None
                i += 4
                continue
            out.append(b[i])
            i += 1
        return bytes(out)

    def findings(b):
        return strip_traces(b).find(b"G0 ")

    waited = 0
    while (strip := strip_traces(bytes(uarts))).find(b"G0 \r") < 0 \
            and waited < 16_000_000:
        await RisingEdge(dut.clk)
        waited += 1
        if waited == 750_000:
            try:
                print(f"  probe: top.st={int(dut.st.value)} reader.st="
                      f"{int(dut.sdr.st.value)} prov_blk={int(dut.prov_blk.value)} "
                      f"sd_blk={int(dut.sd_blk.value)} rd_st={int(dut.rd_st.value)} "
                      f"writes={len(model.writes)} err={int(dut.sdr.error.value)}")
            except Exception:
                pass

    line = bytes(uarts)
    print("writes captured by model:", sorted(model.writes))
    clean = strip_traces(line)
    vline = clean[:clean.find(b"\r")] if b"\r" in clean else clean[:17]
    print("UART verdict:", vline)

    # byte-exact roundtrip checks
    assert len(model.writes) == N_BLOCKS, f"only {len(model.writes)} blocks written"
    assert model.writes == {i: payload[i * LBA:(i + 1) * LBA] for i in range(N_BLOCKS)}, \
        "model-captured writes deviate from the image"
    npz = np.load("/tmp/gate0_vecs.npz")
    stim_w = []
    for i in range(21):
        blob = bytes(npz[f"rec{i}_wblk"].tobytes()) + bytes(npz[f"rec{i}_act"].tobytes())
        stim_w += [int.from_bytes(blob[4 * j:4 * j + 4], "little") for j in range(118)]
    mism = 0
    for i in range(2478):
        got = int(dut.stim[i].value) & 0xFFFFFFFF
        if got != stim_w[i]:
            mism += 1
            if mism <= 6:
                print(f"  stim[{i}] {got:08x} != {stim_w[i]:08x}")
    assert mism == 0, f"card fill not byte-exact ({mism} stim words)"
    print("card fill: byte-exact (2478 stim words)")

    if vline.startswith(b"G0 PASS"):
        led_v = int(dut.led.value)
        assert led_v == 0b010000, f"PASS but LED pads {led_v:06b}"
    elif vline.startswith(b"G0 FAIL"):
        raise AssertionError(f"on-chip mismatch at record {vline[9:11].decode()}")
    else:
        raise AssertionError(f"unexpected line {vline!r}")