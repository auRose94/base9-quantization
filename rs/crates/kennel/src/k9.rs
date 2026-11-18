//! ninths/K9 container directory + decode — kennel's view of docs/12.
//!
//! Directory KV: `ninths.directory2` (v2, version-prefixed) or
//! `ninths.directory` / `k9.directory` (v1, legacy alias). Records point at
//! the GGUF K9 blobs by name; sections in-blob are
//! `[freq | digits | scales | perm]` with byte lengths tbl/dig/sca/plen.
//! Digit streams are lane-interleaved rANS (docs/11): u32 lane_off[L+1]
//! then L independent v1 sub-streams, lane i owning rows
//! [i*ceil(r/L), (i+1)*ceil(r/L)) — decoded here with one thread per lane.

use anyhow::{bail, Context, Result};

use crate::gguf::{Gguf, Value};

#[derive(Debug, Clone)]
pub struct Record {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub k: usize,
    pub group: u8,
    pub scale_mode: u8,
    pub perm_flag: u8,
    pub lanes: usize,
    pub tbl: usize,
    pub dig: usize,
    pub sca: usize,
    pub plen: usize,
}

impl Record {
    /// blob byte ranges: (freq, digits, scales, perm). The GGUF tensor
    /// slice may carry alignment padding after the blob — the record's
    /// section lengths are authoritative.
    pub fn sections<'b>(&self, blob: &'b [u8]) -> Result<(&'b [u8], &'b [u8], &'b [u8], &'b [u8])> {
        let sum = self.tbl + self.dig + self.sca + self.plen;
        if sum > blob.len() {
            bail!(
                "record {}: section lengths {sum} > blob len {}",
                self.name,
                blob.len()
            );
        }
        let blob = &blob[..sum];
        let a = blob.split_at(self.tbl);
        let b = a.1.split_at(self.dig);
        let c = b.1.split_at(self.sca);
        Ok((a.0, b.0, c.0, c.1))
    }
}

struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Cur<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.pos + n > self.b.len() {
            bail!("directory truncated at {} (+{n})", self.pos);
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

/// Parse the directory bytes. `v2` = the KV carried a version prefix
/// (directory2); v1 directories start directly at n_records.
pub fn parse_directory(bytes: &[u8], v2: bool) -> Result<Vec<Record>> {
    let mut c = Cur { b: bytes, pos: 0 };
    if v2 {
        let ver = c.u16()?;
        if ver != 2 {
            bail!("directory2 version {ver}, kennel reads 2");
        }
    }
    let n_records = c.u16()? as usize;
    let mut out = Vec::with_capacity(n_records);
    for _ in 0..n_records {
        let name = c.str().context("record name")?;
        let rank = c.u8()?;
        if rank != 2 {
            bail!("record {name}: rank {rank}, spec says 2");
        }
        let rows = c.u32()? as usize;
        let cols = c.u32()? as usize;
        let k = c.u8()? as usize;
        let group = c.u8()?;
        let scale_mode = c.u8()?;
        let perm_flag = c.u8()?;
        let lanes = if v2 { c.u8()? as usize } else { 1 };
        let tbl = c.u32()? as usize;
        let dig = c.u32()? as usize;
        let sca = c.u32()? as usize;
        let plen = c.u32()? as usize;
        if group != 64 {
            bail!("record {name}: group {group}, normative is 64");
        }
        if cols % 64 != 0 {
            bail!("record {name}: cols {cols} not 64-aligned");
        }
        // docs/12 says k >= 9, but the published extreme rungs (flint k5/k3)
        // postdate that line — accept the full odd range the math supports.
        if k % 2 == 0 || !(3..=255).contains(&k) {
            bail!("record {name}: k {k} outside odd 3..=255");
        }
        if ![0, 1, 4].contains(&scale_mode) {
            bail!("record {name}: scale_mode {scale_mode} not permitted");
        }
        if !(1..=64).contains(&lanes) {
            bail!("record {name}: lanes {lanes} outside 1..=64");
        }
        out.push(Record { name, rows, cols, k, group, scale_mode, perm_flag, lanes, tbl, dig, sca, plen });
    }
    Ok(out)
}

impl Gguf {
    /// (directory bytes, is_v2) — directory2 first, then the legacy spellings.
    pub fn k9_directory(&self) -> Result<Option<(Vec<u8>, bool)>> {
        for (key, v2) in [
            ("ninths.directory2", true),
            ("k9.directory2", true),
            ("ninths.directory", false),
            ("k9.directory", false),
        ] {
            if let Some(v) = self.kv.get(key) {
                let Value::Array(items) = v else {
                    bail!("{key} is not an array");
                };
                let mut bytes = Vec::with_capacity(items.len());
                for it in items {
                    let Value::U8(b) = it else {
                        bail!("{key} holds a non-u8 element");
                    };
                    bytes.push(*b);
                }
                return Ok(Some((bytes, v2)));
            }
        }
        Ok(None)
    }
}

/// Decode the record's digit stream (u8 codes, stored row-major order).
/// Lane-parallel for v2 (docs/11: lanes are independent rANS states over
/// one shared frequency table — thread-per-lane needs no synchronization).
pub fn decode_digits(rec: &Record, blob: &[u8]) -> Result<Vec<u8>> {
    let (freq, dig, _, _) = rec.sections(blob)?;
    if freq.len() != rec.k * 2 {
        bail!("record {}: freq table {} B, expected {}", rec.name, freq.len(), rec.k * 2);
    }
    let f: Vec<u32> = freq
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as u32)
        .collect();
    let sum: u32 = f.iter().sum();
    if sum != ninths_codec::SCALE_M as u32 {
        bail!("record {}: freq sum {sum} != M", rec.name);
    }
    let n = rec.rows * rec.cols;

    if rec.lanes <= 1 {
        return Ok(ninths_codec::rans_decode(dig, &f, n, ninths_codec::SCALE_BITS));
    }

    let l = rec.lanes;
    if dig.len() < 4 * (l + 1) {
        bail!("record {}: digit area {} too small for {l} lanes", rec.name, dig.len());
    }
    let otab = 4 * (l + 1);
    let offs: Vec<usize> = dig[..otab]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    if offs[0] != 0 || offs[l] != dig.len() - otab {
        bail!("record {}: v2 lane offset table broken", rec.name);
    }
    let rows_per = rec.rows.div_ceil(l);
    let base = otab;

    std::thread::scope(|s| -> Result<Vec<Vec<u8>>> {
        let mut handles = Vec::new();
        for li in 0..l {
            let a = (li * rows_per).min(rec.rows);
            let b = ((li + 1) * rows_per).min(rec.rows);
            if a >= b {
                continue;
            }
            let seg = &dig[base + offs[li]..base + offs[li + 1]];
            let f_ref = &f;
            handles.push(s.spawn(move || {
                ninths_codec::rans_decode(seg, f_ref, (b - a) * rec.cols, ninths_codec::SCALE_BITS)
            }));
        }
        let mut chunks = Vec::with_capacity(handles.len());
        for h in handles {
            chunks.push(h.join().expect("lane decode thread panicked"));
        }
        Ok(chunks)
    })
    .map(|chunks| chunks.concat())
    .map(|mut v| {
        v.truncate(n);
        v
    })
}

/// Decode the record's scales to f32 (modes 0/1/4; ent8 via ninths-codec).
pub fn decode_scales(rec: &Record, blob: &[u8]) -> Result<Vec<f32>> {
    let (_, _, sca, _) = rec.sections(blob)?;
    let n_scales = rec.rows * (rec.cols / rec.group as usize);
    match rec.scale_mode {
        0 => {
            if sca.len() < n_scales * 4 {
                bail!("record {}: scales blob short", rec.name);
            }
            Ok(sca
                .chunks_exact(4)
                .take(n_scales)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        1 => {
            if sca.len() < n_scales * 2 {
                bail!("record {}: scales blob short", rec.name);
            }
            Ok(sca
                .chunks_exact(2)
                .take(n_scales)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect())
        }
        4 => Ok(ninths_codec::decode_scales_ent8(sca, n_scales)),
        other => bail!("record {}: scale_mode {other}", rec.name),
    }
}

/// half-precision -> f32 bit-exact expansion (no external crates).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal: value = frac * 2^-24; renormalize into f32
            let hbit = 31 - frac.leading_zeros(); // highest set bit, 0..=9
            sign << 31 | (103 + hbit) << 23 | (frac << (23 - hbit)) & 0x7f_ffff
        }
    } else if exp == 31 {
        sign << 31 | 0x7f80_0000 | frac << 13 // inf / nan
    } else {
        sign << 31 | (exp + 112) << 23 | frac << 13
    };
    f32::from_bits(bits)
}

/// Normative f32 dequant over the whole stored frame, row-parallel.
/// Uses ninths_codec::dequant_level so the CPU side IS the shared norm.
pub fn dequant_f32(rec: &Record, digits: &[u8], scales: &[f32]) -> Result<Vec<f32>> {
    let n = rec.rows * rec.cols;
    if digits.len() != n {
        bail!("record {}: {} digits, expected {n}", rec.name, digits.len());
    }
    let n_groups = rec.cols / rec.group as usize;
    if scales.len() != rec.rows * n_groups {
        bail!("record {}: {} scales, expected {}", rec.name, scales.len(), rec.rows * n_groups);
    }
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let chunk_rows = rec.rows.div_ceil(threads).max(1);
    let mut out = vec![0f32; n];
    std::thread::scope(|s| {
        for (t, out_chunk) in out.chunks_mut(chunk_rows * rec.cols).enumerate() {
            let row0 = t * chunk_rows;
            let digits = &digits[row0 * rec.cols..row0 * rec.cols + out_chunk.len()];
            let scales = &scales;
            let k = rec.k;
            let cols = rec.cols;
            s.spawn(move || {
                for (i, w) in out_chunk.iter_mut().enumerate() {
                    // r must be the GLOBAL row — chunk-local indexing was the
                    // M1 gate's first catch (every chunk re-read row-0 scales)
                    let r = row0 + i / cols;
                    let c = i % cols;
                    *w = ninths_codec::dequant_level(scales[r * n_groups + c / 64], digits[i], k);
                }
            });
        }
    });
    Ok(out)
}
