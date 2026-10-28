//! ninths-codec — Rust port of the ninths (k9) container codec.
//!
//! Normative references: `experiments/k9.py` and `experiments/rans.py` in the
//! base9-quantization repo. This crate is byte-exact parity with those: every
//! stream the Python reference encodes, this crate decodes identically, and
//! vice versa (enforced by `tests/parity.rs` against golden fixtures).
//!
//! Semantics that MUST NOT drift (see rans.py):
//! - rANS, byte renorm, state domain x ∈ [2^24, 2^32).
//! - renorm-before-write pushes the LOW byte (`out.push(x as u8); x >>= 8`)
//!   while `x >= (fs << 32) / M` (integer division).
//! - encode step: `x = ((x / fs) << scale_bits) + (x % fs) + cumul[s]`,
//!   symbols processed in reverse.
//! - final state appended as 4 bytes, big-endian; decoder walks `pos` downward.
//! - `normalize_freqs` uses banker's rounding, tie-break by descending count
//!   then ascending symbol index.

pub mod rans;
pub mod scales;

pub use rans::{rans_decode, rans_encode, normalize_freqs};
pub use scales::{encode_scales_ent8, decode_scales_ent8};

pub const SCALE_BITS: u32 = 12;
pub const SCALE_M: usize = 1 << SCALE_BITS;
pub const SCALE_MODES: &[(u8, &str)] = &[(0, "fp32"), (1, "fp16"), (4, "ent8")];

pub const DEQUANT_GROUP: usize = 64;

#[inline]
pub fn dequant_half(k: usize) -> usize {
    (k - 1) / 2
}

/// Normative dequantization: `w = m * (d - H) / H`, H = (k-1)/2.
#[inline]
pub fn dequant_level(m: f32, d: u8, k: usize) -> f32 {
    let h = dequant_half(k) as f32;
    m * (i32::from(d) - h as i32) as f32 / h
}

/// docs/11 rule: L = max(1, min(want, bytes >> 21, rows)).
#[inline]
pub fn lane_plan(nbytes: usize, rows: usize, want_lanes: usize) -> usize {
    want_lanes
        .min((nbytes + (1 << 21) - 1) >> 21)
        .min(rows.max(1))
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_plan_matches_doc() {
        // python: L = max(1, min(want, (bytes + 2^21 - 1) >> 21, rows))
        assert_eq!(lane_plan(0, 0, 16), 1);
        assert_eq!(lane_plan(1 << 21, 4, 16), 1);
        assert_eq!(lane_plan(3 << 21, 4, 16), 3);
        assert_eq!(lane_plan(1 << 22, 2, 16), 2);
        assert_eq!(lane_plan(3 << 20, 8, 16), 2);
        assert_eq!(lane_plan(5 << 21, 1000, 16), 5);
        assert_eq!(lane_plan(1 << 21, 0, 16), 1);
    }

    #[test]
    fn dequant_half_values() {
        assert_eq!(dequant_half(3), 1);
        assert_eq!(dequant_half(5), 2);
        assert_eq!(dequant_half(63), 31);
        assert_eq!(dequant_half(99), 49);
    }
}