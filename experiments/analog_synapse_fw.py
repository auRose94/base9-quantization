"""Rung 2-half firmware — synapse write-path probe (MicroPython).

Target: any MicroPython board (RP2040 Pi Pico reference pins below; port =
change the Pin numbers).  Copy to the board as main.py.

Parts (against the cart):
  MCP41010  SPI digital pot, 10k, 256 taps — the MCU-writable weight
  ADS1115   I2C 16-bit ADC, 4ch + PGA — the readback
  (later)   LM13700N OTA synapse + TL074 stage extend this protocol; stage A
            (digipot channel characterization) needs no OTA — first-night
            experiment runs with just digipot + ADC + MCU.

Wiring (Pi Pico reference):
  MCP41010  VDD->3V3  CS->GP17  SCK->GP18  SDI->GP19  SHDN->VDD  VSS->GND
            PA0->3V3  PB0->GND  PW0->ADS1115 A0   (voltage-divider mode:
            wiper = rail * tap/255, source impedance <= 2.5k, ADC-safe)
  ADS1115   VDD->3V3  GND->GND  SCL->GP21  SDA->GP20  ADDR->GND (0x48)
            A1->3V3 rail sense (ratios out supply drift in analysis)
  Both grounds common with the Pico.  Rail decoupling 100n at each VDD.

Serial protocol (115200, line-based, CSV replies):
  ID            -> id,synapse-probe,1
  TAP <n>       -> ok               (program digipot tap 0..255)
  READ <n>      -> n x v,<a0_volts>,<a1_volts>
  PROG <tap> <n> -> n x p,<a0_volts>,<a1_volts>   (re-program before each read:
                    programming-noise distribution; READ = read-noise only)
"""
from machine import Pin, SPI, I2C
import time

CS = Pin(17, Pin.OUT, value=1)
spi = SPI(0, baudrate=1_000_000, sck=Pin(18), mosi=Pin(19))
i2c = I2C(0, sda=Pin(20), scl=Pin(21), freq=400_000)
ADS = 0x48
PGA = {6.144: 0b000, 4.096: 0b001, 2.048: 0b010, 1.024: 0b011}


def set_tap(t: int) -> None:
    """Write pot-0 tap (command 0x11 = write data to pot 0)."""
    CS.low()
    spi.write(bytes([0x11, t & 0xFF]))
    CS.high()


def read_ads(ch: int, gain: float = 4.096) -> float:
    """One single-shot conversion, channel ch vs GND, returns volts."""
    cfg = (0x8000 | ((0b1000 | (ch & 7)) << 12) | (PGA[gain] << 9)
           | (1 << 8) | (0b111 << 5) | 0b00011)  # single-shot, 860 SPS
    i2c.writeto_mem(ADS, 1, cfg.to_bytes(2, "big"))
    time.sleep_ms(2)               # ~1.2 ms conversion at 860 SPS
    while True:
        if i2c.readfrom_mem(ADS, 1, 2)[0] & 0x80:
            break
    raw = int.from_bytes(i2c.readfrom_mem(ADS, 0, 2), "big", True)  # signed
    return raw * gain / 32768.0


def emit(prefix: str, n: int) -> None:
    for _ in range(n):
        a0 = read_ads(0)
        a1 = read_ads(1)
        print(f"{prefix},{a0:.6f},{a1:.6f}")


def main() -> None:
    print("id,synapse-probe,1")
    while True:
        line = input().strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].upper()
        if cmd == "ID":
            print("id,synapse-probe,1")
        elif cmd == "TAP":
            set_tap(int(parts[1]))
            print("ok")
        elif cmd == "READ":
            emit("v", int(parts[1]))
        elif cmd == "PROG":
            tap, n = int(parts[1]), int(parts[2])
            for _ in range(n):
                set_tap(tap)
                print(f"p,{read_ads(0):.6f},{read_ads(1):.6f}")
        else:
            print(f"err,unknown:{line}")


main()
