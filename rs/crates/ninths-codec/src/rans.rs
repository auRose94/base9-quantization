//! Shared rANS entropy coder (Duda), byte renorm — the Rust twin of
//! `experiments/rans.py`, byte-exact in both directions.

/// Banker's rounding to nearest integer (numpy `round` semantics) on f64.
/// Inputs are exact-integer products divided once, so this is only about
/// half-ties: round-half-away differs from numpy and would break parity.
#[inline]
fn round_half_even(x: f64) -> i64 {
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

/// `experiments/rans.py:normalize_freqs` — counts (n_syms-long, may be sparse)
/// → normalized uint32 freqs summing to exactly `m`, in the same order.
///
/// Parity notes: the intermediate is evaluated exactly like numpy int64:
/// `f64(counts[s] * scale) / f64(total)` (exact integers → one f64 division),
/// then numpy banker's rounding. The diff pass walks symbols sorted by
/// descending count (ties: ascending index, i.e. numpy stable argsort on
/// `-counts`) cycling repeatedly.
pub fn normalize_freqs(counts: &[u64], m: usize) -> Vec<u32> {
    let total: u64 = counts.iter().sum();
    assert!(total > 0, "empty symbol stream");
    assert!((total * m as u64) < (1u64 << 52), "counts*fits would lose fp64 exactness");
    let mut f = vec![0u32; counts.len()];
    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            let v = (c as f64 * m as f64) / total as f64;
            f[s] = round_half_even(v).max(1) as u32;
        }
    }
    let mut diff = f.iter().map(|&v| v as i64).sum::<i64>() - m as i64;
    // numpy stable argsort on -counts: descending count, ties ascending index.
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
    let mut i: usize = 0;
    while diff != 0 {
        assert!(i < 10_000 * order.len().max(1), "normalize_freqs did not settle");
        let s = order[i % order.len()];
        i += 1;
        if counts[s] == 0 {
            continue;
        }
        if diff > 0 && f[s] > 1 {
            f[s] -= 1;
            diff -= 1;
        } else if diff < 0 {
            f[s] += 1;
            diff += 1;
        }
    }
    assert_eq!(f.iter().map(|&v| v as usize).sum::<usize>(), m);
    f
}

struct Tables {
    cumul: Vec<u64>,
    inv: Vec<u16>,
}

fn tables(f: &[u32], m: usize) -> Tables {
    let mut cumul = Vec::with_capacity(f.len() + 1);
    let mut acc: u64 = 0;
    cumul.push(acc);
    for &v in f {
        acc += v as u64;
        cumul.push(acc);
    }
    let mut inv = vec![0u16; m];
    for s in 0..f.len() {
        for slot in cumul[s]..cumul[s + 1] {
            inv[slot as usize] = s as u16;
        }
    }
    Tables { cumul, inv }
}

/// Encode (n_syms ≤ 2^16), M-bit normalized freqs over u8 symbols.
/// Returns the renorm bytes followed by the 4-byte big-endian end state —
/// byte-identical to `rans.py:rans_encode`.
pub fn rans_encode(symbols: &[u8], f: &[u32], scale_bits: u32) -> Vec<u8> {
    let t = tables(f, 1 << scale_bits);
    let shift = 32 - scale_bits; // (fs << 32) / 2^sb = fs << (32 - sb) exactly
    let x0: u64 = 1 << 24;
    let mut out: Vec<u8> = Vec::with_capacity(symbols.len() + 4);
    let mut x = x0;
    for &s in symbols.iter().rev() {
        let fs = f[s as usize] as u64;
        let cs = t.cumul[s as usize];
        let renorm_at = fs << shift; // == (fs << 32) / 2^scale_bits
        while x >= renorm_at {
            out.push(x as u8);
            x >>= 8;
        }
        x = ((x / fs) << scale_bits) + (x % fs) + cs;
    }
    assert!((1 << 24) <= x && x < (1 << 32));
    // python reference appends x.to_bytes(4, "big") — exactly 4 bytes
    out.extend_from_slice(&(x as u32).to_be_bytes());
    out
}

/// Decode `n` u8 symbols from a stream produced by [`rans_encode`].
pub fn rans_decode(stream: &[u8], f: &[u32], n: usize, scale_bits: u32) -> Vec<u8> {
    assert!(stream.len() >= 4);
    let t = tables(f, 1 << scale_bits);
    let mask: u32 = (1u64 << scale_bits) as u32 - 1;
    let low_half: u64 = 1 << 24;
    let end: [u8; 4] = stream[stream.len() - 4..].try_into().unwrap();
    let mut x = u32::from_be_bytes(end) as u64;
    let mut pos = stream.len() as isize - 5;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let s = t.inv[(x & mask as u64) as usize] as usize;
        let fs = f[s] as u64;
        let cs = t.cumul[s];
        x = fs * (x >> scale_bits) + (x & mask as u64) - cs;
        while x < low_half {
            assert!(pos >= 0, "rANS decoder consumed past the byte frontier");
            x = (x << 8) | stream[pos as usize] as u64;
            pos -= 1;
        }
        out.push(s as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SCALE_BITS;

    fn counts_of(syms: &[u8], k: usize) -> Vec<u64> {
        let mut c = vec![0u64; k];
        for &s in syms {
            c[s as usize] += 1;
        }
        c
    }

    #[test]
    fn banker_tie_matches_numpy() {
        // np.round(0.5) = 0, np.round(1.5) = 2, np.round(2.5) = 2
        assert_eq!(round_half_even(0.5), 0);
        assert_eq!(round_half_even(1.5), 2);
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(3.5), 4);
        assert_eq!(round_half_even(10.49), 10);
    }

    #[test]
    fn simple_roundtrip_k12() {
        let syms: Vec<u8> = (0..4096u32)
            .map(|i| ((i * 7919 + (i >> 3) * 13) % 63) as u8)
            .collect();
        let f = normalize_freqs(&counts_of(&syms, 63), 1 << SCALE_BITS);
        let enc = rans_encode(&syms, &f, SCALE_BITS);
        let back = rans_decode(&enc, &f, syms.len(), SCALE_BITS);
        assert_eq!(back, syms);
    }

    #[test]
    fn skewed_cheap_symbols_roundtrip() {
        // distribution with many zeros, like real RTN digits
        let mut syms = vec![31u8; 100_000];
        for (i, s) in syms.iter_mut().enumerate() {
            if i % 997 == 0 {
                *s = (i % 63) as u8;
            } else if i % 31 == 0 {
                *s = 0;
            }
        }
        let f = normalize_freqs(&counts_of(&syms, 63), 1 << SCALE_BITS);
        let enc = rans_encode(&syms, &f, SCALE_BITS);
        let back = rans_decode(&enc, &f, syms.len(), SCALE_BITS);
        assert_eq!(back, syms);
    }
}