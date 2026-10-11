## 2026-10-10, session 29 (cont. 2) — why her power-up was dark, and the fixes

Rose power-cycled holding the user key and saw nothing: **the SRAM-loaded
design is volatile** — every power-up boots a blank chip unless the bitstream
sits in the config flash. Two fixes shipped: `openFPGALoader --board
tangnano20k -f gate0sd.fs` writes the design into the board's SPI flash
(the flag is `-f/--write-flash`; there is no `--flash`), now done — every
power-up boots the SD-fed gate; and the LED pads are ACTIVE-LOW (the LiteX
board map names them led_n), so the pad logic is inverted with an explicit
lit-mask — sim re-verified (G0 PASS) and re-flashed to the flash. She can now
provision with the user key held at power-up and see the verdict LEDs directly;
UART line on pin 69 whenever the USB-C or the pod's CH-B pair is wired.
