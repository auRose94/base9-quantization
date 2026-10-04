# 00 — Where the idea came from

**Date:** 2026-10-03 (session start)
**Author:** rose
**License intent:** MIT — this whole folder is meant to be publishable as-is.

## Original idea, verbatim

> I want you to make a folder to research the idea of using a quantized model
> that fits in the repeating numbers of the result of a math equition. I was
> thinking a base 9 number system 0-9 fitting within the numbers might prove
> useful for model compression. And I wanted help to explore that as
> legitimate research I'm willing to share under MIT license.

Provenance note for future readers: this file preserves the idea as first
stated, before any formalization, so later refinement can be checked against
the origin.

## Terminology note

"Base 9 number system 0-9": digits 0-9 are ten symbols, which is base *10*.
Base 9 uses digits **0-8**. This project interprets the idea generously and
works with the base-9 alphabet **0-8** (one decimal digit's worth of symbols),
and separately considers the reading where decimal digits 0-9 act as
*carriers* for base-9 payload digits (see `03-open-questions.md`, RQ6).

## How the idea was unfolded

The verbatim idea names two mechanisms that feel independent at first but
turn out to be two views of one quantization grid:

1. **Quantized model fits in the repeating numbers** — infinitely repeating
   decimal expansions are exactly the rational numbers; a grid of multiples
   of 1/9 makes every weight a *single-digit* repeating decimal in base 10
   (k/9 = 0.kkkk...), because 10 ≡ 1 (mod 9).
2. **Base-9 digit system** — a base-9 digit is exactly a *pair of ternary
   (3-level) weights*: two trits = 9 states. Modern ternary LLM quantization
   (BitNet b1.58, "1.58 bits") has log2(3) ≈ 1.585 bits per weight, so one
   base-9 digit (log2 9 = 3.170 bits) carries exactly two of them.

Formalization, math, and falsifiable predictions live in
`01-hypothesis.md`. The literature context lives in `02-related-work.md`.
Open questions and planned experiments live in `03-open-questions.md`.