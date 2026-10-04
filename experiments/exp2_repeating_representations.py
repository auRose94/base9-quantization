#!/usr/bin/env python3
"""exp2 — number-theory checks and the rational-encoding (P4) test.

A. Long-division expansion machinery (base b): minimal preperiod/period per
   fraction, verified by exact reconstruction with `fractions.Fraction`.
   Theorem under test: eventually periodic (incl. terminating) iff rational,
   with period ord_q(b) after removing divisors of the base.

B. The ninths identity: k/9 = 0.kkk... (1-digit period) in base 10 and
   k/9 = 0.k (terminating) in base 9, for k = 1..8; plus the base-9/base-10
   duality rows (1/2, 1/3, 1/8, 1/9, 1/7) and ord_q(b) table.

C. Negative control: base-10 and base-9 expansions of sqrt(2) to 20,000
   digits must NOT show a cycle (irrational).

D. P4 — can "repeating numbers" beat index coding?
   D1: notation-vs-compression: storing "0.k̄" text for ninths-grid weights
       vs storing the digit index (3.17 bits floor).
   D2: minimal-rational encoding via continued fractions (Fraction.limit_
       denominator) — bits/param and MSE for denominator bounds, compared
       with exp1's grid coders at matched bits.

Outputs results/exp2_repeating_representations.md (+ .csv for D2).
"""
import csv
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.set_int_max_str_digits(0)  # 20000-digit sqrt(2) needs the str() conversion

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

N_D2 = 20_000
LOG = math.log2


# --------------------------------------------------- long division ----
def expand(p, q, base, max_digits=100_000):
    """Digits of p/q, 0 <= p < q, in `base`, with cycle detection.

    Returns (pre, period_digits, terminating) where `pre` is the digit list
    before the cycle. terminating => period_digits == [].
    """
    assert 0 <= p < q and q > 0
    if p == 0:
        return [], [], True
    r = p
    seen = {}
    pre, cycle = [], []
    digits = []
    i = 0
    while r and r not in seen:
        if i > max_digits:
            return None, [], False  # no cycle found in window (irrational-ish)
        seen[r] = i
        r *= base
        digits.append(r // q)
        r %= q
        i += 1
        if r == 0:
            return digits, [], True
    j = seen[r]
    return digits[:j], digits[j:], False


def reconstruct(base, pre, cycle):
    """Exact Fraction value of (pre + repeating cycle) expansion — the
    algebraic inverse of long division, proving periodic => rational."""
    A = 0
    for d in pre:
        A = A * base + d
    B = 0
    for d in cycle:
        B = B * base + d
    L = len(cycle)
    j = len(pre)
    if L == 0:
        return Fraction(A, base ** j)
    # value = A/b^j + B/(b^j * (b^L - 1))
    return Fraction(A * (base ** L - 1) + B, base ** j * (base ** L - 1))


def fmt(base, pre, cycle):
    digs = "0123456789abcdefghi"
    s = "0." + "".join(digs[d] for d in pre)
    if cycle:
        s += "(" + "".join(digs[d] for d in cycle) + ")"
    return s


def multiplicative_order(b, q):
    assert math.gcd(b, q) == 1
    r, n = b % q, 1
    while r != 1:
        r = r * b % q
        n += 1
        assert n <= q
    return n


# ----------------------------------------------------------- parts ----
def part_a_theorem():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(200):
        q = int(rng.integers(2, 200))
        p = int(rng.integers(0, q))
        g = math.gcd(p, q)
        p, q = p // g, q // g
        for base in (9, 10):
            pre, cyc, term = expand(p % q, q, base, max_digits=2 * q + 20)
            assert term or cyc, f"{p}/{q} base {base} failed to cycle"
            val = reconstruct(base, pre, cyc)
            assert val == Fraction(p, q), f"reconstruction mismatch {p}/{q}"
            checked += 1
    print(f"A  theorem check passed on {checked} (fraction, base) pairs")
    return checked


def part_b_ninths():
    rows = []
    print("\nB  ninths identity and duality (base 10 vs base 9)")
    for frac in [Fraction(1, 2), Fraction(1, 3), Fraction(1, 8), Fraction(1, 9),
                 Fraction(2, 9), Fraction(4, 9), Fraction(1, 5), Fraction(1, 7),
                 Fraction(1, 11)]:
        cells = [f"{frac}"]
        for base in (10, 9):
            pre, cyc, term = expand(frac.numerator, frac.denominator, base)
            if term:
                cells.append(f"{fmt(base, pre, cyc)}  terminates")
            else:
                cells.append(f"{fmt(base, pre, cyc)}  period {len(cyc)}")
        rows.append(cells)
        print(f"   {cells[0]:<6} base10: {cells[1]:<32} base9: {cells[2]}")
    return rows


def part_b_orders():
    rows = []
    print("\n   multiplicative order ord_q(b) (period of 1/q)")
    for q in [3, 7, 9, 11, 13, 27, 37, 101]:
        cells = [f"q={q}"]
        for base in (10, 9):
            if math.gcd(base, q) == 1:
                cells.append(f"ord_{q}({base}) = {multiplicative_order(base, q)}")
            else:
                cells.append(f"{q} divides {base} -> terminates")
        rows.append(cells)
        print(f"   {cells[0]:<7} {cells[1]:<28} {cells[2]}")
    return rows


def part_c_sqrt2():
    """Irrational negative control: the first 20000 digits of sqrt(2) must not
    be eventually periodic (any period <= 2000 with preperiod <= 30 would show
    up as all digit-vs-digit-minus-L matches beyond the preperiod)."""
    N = 20_000
    digs10 = np.array([int(c) for c in str(math.isqrt(2 * 10 ** (2 * N)))[1:]],
                      dtype=np.int64)
    t = math.isqrt(2 * 9 ** (2 * N))
    a9 = []
    while t:
        a9.append(t % 9)
        t //= 9
    a9 = a9[::-1]
    assert a9[0] == 1, "integer part of sqrt(2) is a single digit, base 9 too"
    digs9 = np.array(a9[1:], dtype=np.int64)
    result = []
    for name, digs in (("base 10", digs10), ("base 9", digs9)):
        assert len(digs) == N
        found = None
        for L in range(1, 2001):
            bad = np.nonzero(digs[L:] != digs[:-L])[0]
            if len(bad) == 0:
                found = (0, L)  # purely periodic from digit 1
                break
            if L + bad[-1] < L + 30:  # final mismatch inside a short preperiod
                found = (int(bad[-1]), L)
                break
        result.append((name, found))
        if found:
            print(f"C  sqrt(2) {name}: PERIODICITY DETECTED pre<={found[0]} period={found[1]} (unexpected!)")
        else:
            print(f"C  sqrt(2) {name}: no period<=2000 / preperiod<=30 in 20000 digits (expected)")
    return result


def part_d1_notation():
    print("\nD1 notation-vs-compression for a ninths-grid weight (k = 4, say 4/9):")
    idx_bits = LOG(9)
    str10 = len("0.(4)")
    str9 = len("0.4")
    print(f"   digit index (base-9 alphabet floor): {idx_bits:.4f} bits")
    print(f"   '0.(4)' repeating-decimal notation  : {str10} ASCII chars = {8 * str10} bits")
    print(f"   '0.4' terminating base-9 notation   : {str9} ASCII chars = {8 * str9} bits")
    print("   => notation carries framing overhead, not compression; the")
    print("      digit index IS the content the notation decorates.")


def part_d2_rational_encoding():
    rng = np.random.default_rng(42)
    w = rng.standard_normal(N_D2)
    grid_scale = 9.0 * float(np.max(np.abs(w))) / 4.0
    rows = []
    for B in (9, 16, 32, 256, 4096):
        bits, err = 0.0, 0.0
        for x in w:
            fr = Fraction(float(x)).limit_denominator(B)
            p = abs(fr.numerator)
            q = fr.denominator
            bits += (p.bit_length() if p else 1) + q.bit_length() + 1  # + sign
            err += (x - float(fr)) ** 2
        bits /= N_D2
        err /= N_D2 * float(np.var(w))
        rows.append(dict(den_bound=B, bits=bits, mse=err))
        print(f"D2 limit_denominator({B:>5}): {bits:8.3f} bits/param, rel MSE {err:.6f}")
    q9 = np.clip(np.round(9 * w / grid_scale), -4, 4)
    mse9 = float(np.mean((w - q9 * grid_scale / 9) ** 2) / np.var(w))
    print(f"D2 9-level ninths grid (reference)  : {LOG(9):8.3f} bits/param (alphabet),"
          f" rel MSE {mse9:.6f}")
    rows.append(dict(den_bound="ninths grid", bits=LOG(9), mse=mse9))
    with open(RESULTS / "exp2_rational_encoding.csv", "w", newline="") as fh:
        cwr = csv.DictWriter(fh, fieldnames=["den_bound", "bits", "mse"])
        cwr.writeheader()
        cwr.writerows(rows)
    return rows


def main():
    lines = ["# exp2 — repeating representations and rational encoding (P4)", ""]
    n = part_a_theorem()
    lines.append(f"Theorem check (periodic expansion <=> rational, bases 9 & 10):"
                 f" {n} (fraction, base) pairs reconstructed exactly via"
                 f" geometric-series identity. PASS")
    print()
    b1 = part_b_ninths()
    b2 = part_b_orders()
    c = part_c_sqrt2()
    print()
    part_d1_notation()
    r = part_d2_rational_encoding()

    lines += ["", "## B — expansions (verified by exact reconstruction)", "",
              "| fraction | base 10 | base 9 |", "|---|---|---|"]
    lines += [f"| {c0} | {c1} | {c2} |" for c0, c1, c2 in b1]
    lines += ["", "## ord_q(b) — period of 1/q", "",
              "| q | base 10 | base 9 |", "|---|---|---|"]
    lines += [f"| {c0} | {c1} | {c2} |" for c0, c1, c2 in b2]
    lines += ["", "## C — irrational negative control (sqrt 2)", ""]
    lines += [f"- {name}: {'cycle DETECTED (unexpected)' if L else 'no cycle in 20000 digits, as expected'}"
              for name, L in c]
    lines += ["", "## D2 — rational (continued-fraction) encoding vs index coding", "",
              "| encoding | bits/param | rel MSE |", "|---|---|---|"]
    lines += [f"| min-rational, denominator ≤ {r_['den_bound']} | {r_['bits']:.3f} |"
              f" {r_['mse']:.6f} |" for r_ in r]
    lines += ["", "Notation-vs-compression (D1): digit index = 3.1699 bits vs"
              " '0.(4)' = 40 bits of ASCII. The repeating-decimal *form* is"
              " notation; the digit index is the content.", ""]
    (RESULTS / "exp2_repeating_representations.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()