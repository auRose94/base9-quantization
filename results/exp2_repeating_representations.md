# exp2 — repeating representations and rational encoding (P4)

Theorem check (periodic expansion <=> rational, bases 9 & 10): 400 (fraction, base) pairs reconstructed exactly via geometric-series identity. PASS

## B — expansions (verified by exact reconstruction)

| fraction | base 10 | base 9 |
|---|---|---|
| 1/2 | 0.5  terminates | 0.(4)  period 1 |
| 1/3 | 0.(3)  period 1 | 0.3  terminates |
| 1/8 | 0.125  terminates | 0.(1)  period 1 |
| 1/9 | 0.(1)  period 1 | 0.1  terminates |
| 2/9 | 0.(2)  period 1 | 0.2  terminates |
| 4/9 | 0.(4)  period 1 | 0.4  terminates |
| 1/5 | 0.2  terminates | 0.(17)  period 2 |
| 1/7 | 0.(142857)  period 6 | 0.(125)  period 3 |
| 1/11 | 0.(09)  period 2 | 0.(07324)  period 5 |

## ord_q(b) — period of 1/q

| q | base 10 | base 9 |
|---|---|---|
| q=3 | ord_3(10) = 1 | 3 divides 9 -> terminates |
| q=7 | ord_7(10) = 6 | ord_7(9) = 3 |
| q=9 | ord_9(10) = 1 | 9 divides 9 -> terminates |
| q=11 | ord_11(10) = 2 | ord_11(9) = 5 |
| q=13 | ord_13(10) = 6 | ord_13(9) = 3 |
| q=27 | ord_27(10) = 3 | 27 divides 9 -> terminates |
| q=37 | ord_37(10) = 3 | ord_37(9) = 9 |
| q=101 | ord_101(10) = 4 | ord_101(9) = 50 |

## C — irrational negative control (sqrt 2)

- base 10: no cycle in 20000 digits, as expected
- base 9: no cycle in 20000 digits, as expected

## D2 — rational (continued-fraction) encoding vs index coding

| encoding | bits/param | rel MSE |
|---|---|---|
| min-rational, denominator ≤ 9 | 6.416 | 0.000308 |
| min-rational, denominator ≤ 16 | 7.561 | 0.000059 |
| min-rational, denominator ≤ 32 | 9.292 | 0.000007 |
| min-rational, denominator ≤ 256 | 14.917 | 0.000000 |
| min-rational, denominator ≤ 4096 | 22.839 | 0.000000 |
| min-rational, denominator ≤ ninths grid | 3.170 | 0.099640 |

Notation-vs-compression (D1): digit index = 3.1699 bits vs '0.(4)' = 40 bits of ASCII. The repeating-decimal *form* is notation; the digit index is the content.
