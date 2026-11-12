"""Rung 2-half host harness — characterize the analog write path on real parts.

Stages (against `analog_synapse_fw.py` firmware, wired per its header):
  sweep  256 taps x N reads  -> per-tap mean (grid transfer curve), read
         noise per tap, integral nonlinearity, stuck-tap census
  prog   subset of taps re-programmed before each read -> programming noise
         via variance subtraction: sigma_prog = sqrt(var_obs - var_read)
  ratio  A1 (rail) divides out supply drift
  export synapse_channel.json — measured per-tap means + noise distributions,
         the future `measured` input for analog_channel.AnalogParams

Run against hardware:
  .venv-glm/bin/python experiments/synapse_char.py --port /dev/ttyACM0 --unit U1
Run the no-hardware gate (mock firmware with known ground truth):
  .venv-glm/bin/python experiments/synapse_char.py --mock --gate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/synapse_char"
TAPS = 256


# ----------------------------------------------------------------- probes --
class MockProbe:
    """In-process firmware simulator with known ground truth (gate mode)."""

    GROUND_TRUTH = dict(
        sigma_read_volts=300e-6,      # ADS1115-ish read noise
        sigma_prog_taps=0.030,        # programming jitter, in taps
        rail=3.3000, rail_drift=20e-6,
        stuck=(77, 200),
        endl_lsbs=0.4,                # track nonlinearity amplitude
    )

    def __init__(self):
        rng = np.random.default_rng(42)
        self.rng = rng
        t = np.arange(TAPS)
        self.grid_err = (self.GROUND_TRUTH["endl_lsbs"]
                         * np.sin(t * 0.37 + 1.1))          # deterministic NL
        self.eff_tap = 0.0
        self._stuck_voltage = {}
        for s in self.GROUND_TRUTH["stuck"]:
            self._stuck_voltage[s] = None                   # learned on 1st write

    def _volts(self, tap_float: float) -> float:
        return self.GROUND_TRUTH["rail"] * tap_float / (TAPS - 1)

    def cmd(self, line: str) -> list[str]:
        parts = line.split()
        c = parts[0].upper()
        g = self.GROUND_TRUTH
        rail = lambda: g["rail"] + self.rng.normal(0, g["rail_drift"])
        if c == "ID":
            return ["id,synapse-probe,1"]
        if c == "TAP":
            t = int(parts[1])
            self._apply(t)
            return ["ok"]
        if c == "READ":
            return [f"v,{self._sample():.6f},{rail():.6f}"
                    for _ in range(int(parts[1]))]
        if c == "PROG":
            t, n = int(parts[1]), int(parts[2])
            out = []
            for _ in range(n):
                self._apply(t)
                out.append(f"p,{self._sample():.6f},{rail():.6f}")
            return out
        return [f"err,unknown:{line}"]

    def _apply(self, t: int) -> None:
        g = self.GROUND_TRUTH
        self.eff_tap = t + self.grid_err[t] + self.rng.normal(0, g["sigma_prog_taps"])
        if t in self._stuck_voltage:
            if self._stuck_voltage[t] is None:
                # a stuck wiper freezes ~25 taps away from where it should be
                self._stuck_voltage[t] = self.eff_tap + 25.0
            self.eff_tap = self._stuck_voltage[t]

    def _sample(self) -> float:
        return self._volts(self.eff_tap) + self.rng.normal(
            0, self.GROUND_TRUTH["sigma_read_volts"])


class SerialProbe:
    """pyserial transport with the same cmd() interface."""

    def __init__(self, port: str):
        import serial  # pyserial; install into .venv-glm when hardware lands
        self.ser = serial.Serial(port, 115200, timeout=5)

    def cmd(self, line: str) -> list[str]:
        self.ser.reset_input_buffer()
        self.ser.write((line + "\n").encode())
        n = {"READ": int(line.split()[1]) if line.startswith("READ") else 0,
             "PROG": int(line.split()[2]) if line.startswith("PROG") else 0}
        want = {"ID": 1, "TAP": 1, "READ": n["READ"], "PROG": n["PROG"]}
        lines = []
        for _ in range(want.get(line.split()[0].upper(), 0)):
            s = self.ser.readline().decode().strip()
            if s.startswith(("id,", "ok", "v,", "p,")):
                lines.append(s)
        return lines


# -------------------------------------------------------------- acquisition --
def run_sweep(probe, n_reads: int):
    rows = []
    for tap in range(TAPS):
        probe.cmd(f"TAP {tap}")
        samples = [(float(l.split(",")[1]), float(l.split(",")[2]))
                   for l in probe.cmd(f"READ {n_reads}")]
        a0 = np.array([s[0] for s in samples])
        a1 = np.array([s[1] for s in samples])
        ratio = a0 / a1 * 3.3          # rail-ratio corrected volts
        rows.append(dict(tap=tap, mean=float(ratio.mean()),
                         std=float(ratio.std()), n=len(ratio)))
    return rows


def run_prog(probe, taps, n_cycles: int):
    rows = []
    for tap in taps:
        samples = [(float(l.split(",")[1]), float(l.split(",")[2]))
                   for l in probe.cmd(f"PROG {tap} {n_cycles}")]
        ratio = np.array([a / b * 3.3 for a, b in samples])
        rows.append(dict(tap=tap, mean=float(ratio.mean()),
                         std=float(ratio.std()), n=len(ratio)))
    return rows


# ----------------------------------------------------------------- analysis --
def analyze(sweep, prog, unit: str):
    means = np.array([r["mean"] for r in sweep])
    read_std = np.array([r["std"] for r in sweep])
    sigma_read = float(read_std.mean())

    # robust line fit (OLS, then one outlier-rejection pass)
    taps = np.arange(TAPS)
    keep = np.ones(TAPS, bool)
    for _ in range(2):
        p = np.polyfit(taps[keep], means[keep], 1)
        resid = means - np.polyval(p, taps)
        s = float(np.std(resid[keep]))
        keep_new = np.abs(resid) < 5 * max(s, sigma_read)
        if (keep_new == keep).all():
            break
        keep = keep_new
    stuck = [int(t) for t in taps[~keep]]
    gain_v_per_tap = float(p[0])
    lsb_volts = abs(gain_v_per_tap)          # 1 tap ~ 1 LSB of the 256-grid

    # programming noise by variance subtraction
    prog_std = {}
    for r in prog:
        t = r["tap"]
        var_read = float(read_std[t]) ** 2
        v = max(r["std"] ** 2 - var_read, 0.0)
        prog_std[t] = math.sqrt(v)
    sigma_prog_volts = float(np.mean(list(prog_std.values())))
    sigma_prog_taps = sigma_prog_volts / lsb_volts

    endl = float(np.max(np.abs(resid[keep]))) / lsb_volts
    report = dict(
        unit=unit, sigma_read_volts=sigma_read, sigma_prog_volts=sigma_prog_volts,
        sigma_prog_taps=sigma_prog_taps, v_per_tap=gain_v_per_tap,
        rail_volts=3.3, endl_lsbs=endl, stuck_taps=stuck,
        monotonic=bool(np.all(np.diff(means[keep]) > 0)),
    )
    return report, resid


def export(unit: str, sweep, prog, report) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{unit}_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tap", "mean", "std", "n"])
        w.writeheader()
        w.writerows(sweep)
    with open(OUT / f"{unit}_prog.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tap", "mean", "std", "n"])
        w.writeheader()
        w.writerows(prog)
    path = OUT / f"{unit}_channel.json"
    prog_by_tap = {p["tap"]: p for p in prog}
    sigma_prog_per_tap = {}
    for r in sweep:
        t = r["tap"]
        if t in prog_by_tap:
            v = max(prog_by_tap[t]["std"] ** 2 - r["std"] ** 2, 0.0)
            sigma_prog_per_tap[str(t)] = math.sqrt(v) / abs(report["v_per_tap"])
    path.write_text(json.dumps(
        dict(report=report,
             v_mean=[r["mean"] for r in sweep],
             sigma_read=[r["std"] for r in sweep],
             sigma_prog_taps_per_tap=sigma_prog_per_tap),
        indent=1))
    return path


# --------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--unit", default="U1", help="board label (multi-unit runs)")
    ap.add_argument("--reads", type=int, default=200)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="with --mock: assert recovery of ground truth")
    args = ap.parse_args()

    probe = MockProbe() if args.mock else SerialProbe(args.port)
    print("probe:", probe.cmd("ID")[0])

    sweep = run_sweep(probe, args.reads)
    prog_taps = list(range(0, TAPS, 8))
    prog = run_prog(probe, prog_taps, args.cycles)

    report, _ = analyze(sweep, prog, args.unit)
    for k, v in report.items():
        print(f"  {k}: {v}")
    path = export(args.unit, sweep, prog, report)
    print(f"exported -> {path}")

    if args.gate:
        gt = MockProbe.GROUND_TRUTH
        assert abs(report["sigma_read_volts"] - gt["sigma_read_volts"]) \
            < 0.10 * gt["sigma_read_volts"], "sigma_read recovery failed"
        assert abs(report["sigma_prog_volts"] - gt["sigma_prog_taps"]
                   * abs(report["v_per_tap"])) \
            < 0.30 * gt["sigma_prog_taps"] * abs(report["v_per_tap"]), \
            "sigma_prog recovery failed"
        assert sorted(report["stuck_taps"]) == list(gt["stuck"]), "stuck census failed"
        assert 0 < report["endl_lsbs"] < 1.5, "ENDL estimate failed"
        print(f"GATE PASS (mock): sigma_read ±10%, sigma_prog ±30%, "
              f"stuck census exact, ENDL {report['endl_lsbs']:.2f} LSB")


if __name__ == "__main__":
    main()
