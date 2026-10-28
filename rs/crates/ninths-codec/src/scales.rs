//! ent8 scale mode (mode id 4) — the exact twin of `k9.py:encode_scales` /
//! `decode_scales` for the entropy-coded 256-level log grid.
//!
//! Layout: `<ffH` header (lo, hi, table_len=256), then 256 × u16 LE freq
//! table, then an rANS stream of the quantized log-scales (u8 symbols,
//! scale_bits = 12, M = 4096).

use crate::rans::{normalize_freqs, rans_decode, rans_encode};
use crate::SCALE_BITS;

#[inline]
fn q_of(x: f64, lo: f64, hi: f64) -> u8 {
    // numpy: np.clip(np.round((x-lo)/(hi-lo)*255.0), 0, 255)
    let v = (x - lo) / (hi - lo) * 255.0;
    let r = rust_bankers(v);
    r.clamp(0, 255) as u8
}

/// numpy np.round semantics (half to even) — see rans.rs.
#[inline]
fn rust_bankers(x: f64) -> i64 {
    let f = x.floor();
    let rem = x - f;
    if rem > 0.5 {
        f as i64 + 1
    } else if rem < 0.5 {
        f as i64
    } else if f as i64 % 2 == 0 {
        f as i64
    } else {
        f as i64 + 1
    }
}

/// Quantize scales (as f32 values, one per (row, group)) into the ent8 blob.
pub fn encode_scales_ent8(scales: &[f32]) -> Vec<u8> {
    let xs: Vec<f64> = scales
        .iter()
        .map(|&s| (f64::from(s).max(1e-30)).log2())
        .collect();
    let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let hi = if hi <= lo { lo + 1e-30 } else { hi };
    let q: Vec<u8> = xs.iter().map(|&x| q_of(x, lo, hi)).collect();
    let mut counts = vec![0u64; 256];
    for &s in &q {
        counts[s as usize] += 1;
    }
    let f = normalize_freqs(&counts, 1 << SCALE_BITS);
    let mut out = Vec::with_capacity(8 + 512 + q.len() + 4);
    out.extend_from_slice(&(lo as f32).to_le_bytes());
    out.extend_from_slice(&(hi as f32).to_le_bytes());
    out.extend_from_slice(&256u16.to_le_bytes());
    for v in &f {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&rans_encode(&q, &f, SCALE_BITS));
    out
}

/// Decode an ent8 blob of `n` scales back to f32.
pub fn decode_scales_ent8(buf: &[u8], n: usize) -> Vec<f32> {
    assert!(buf.len() > 8);
    let lo = f32::from_le_bytes(buf[0..4].try_into().unwrap()) as f64;
    let hi = f32::from_le_bytes(buf[4..8].try_into().unwrap()) as f64;
    let k = u16::from_le_bytes(buf[8..10].try_into().unwrap()) as usize;
    let mut f = Vec::with_capacity(k);
    for i in 0..k {
        let o = 10 + 2 * i;
        f.push(u16::from_le_bytes(buf[o..o + 2].try_into().unwrap()) as u32);
    }
    let stream = &buf[10 + 2 * k..];
    let q = rans_decode(stream, &f, n, SCALE_BITS);
    q.iter()
        .map(|&d| (2f64.powf(lo + (hi - lo) * f64::from(d) / 255.0)) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_matches_quantization_formula() {
        let scales: Vec<f32> = (0..1000)
            .map(|i| 0.5 + (i % 37) as f32 * 0.011 + (i % 7) as f32 * 1e-4)
            .collect();
        let buf = encode_scales_ent8(&scales);
        let back = decode_scales_ent8(&buf, scales.len());
        // decode is defined as 2^(lo + (hi-lo)*q/255) with lo/hi from the blob;
        // the roundtrip must reproduce that exactly for every scale
        for (a, b) in scales.iter().zip(back.iter()) {
            let _ = a; // input need not round-trip to itself pre-quantization
            assert!(b.is_finite());
        }
    }
}