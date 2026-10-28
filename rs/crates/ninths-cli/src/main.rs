//! ninths-cli — command-line front for the ninths container tooling.
//! Stage 1: codec parity driver (encode/decode roundtrips on files).
//! Later stages: gguf pass1/pass2, gates, materialize.


#[derive(clap::Parser, Debug)]
#[command(name = "ninths", version, about = "ninths container tooling")]
enum Cli {
    /// rANS roundtrip check over a raw u8 digit file (self-test).
    RansCheck {
        /// optional raw symbols file; omitted = deterministic synthetic
        #[arg(short, long)]
        digits: Option<String>,
        /// symbol count (alphabet size), default 63
        #[arg(short = 'k', long, default_value_t = 63)]
        k: usize,
    },
    /// ent8 scale blob roundtrip check over a raw f32 little-endian file
    Ent8Check {
        scales: String,
    },
}

fn main() -> anyhow::Result<()> {
    use clap::Parser as _;
    let cli = Cli::parse();
    match cli {
        Cli::RansCheck { digits, k } => {
            let syms: Vec<u8> = match digits {
                Some(p) => std::fs::read(p)?,
                None => (0..500_000u32)
                    .map(|i| ((i * 7919u32 + (i >> 3) * 13) % k as u32) as u8)
                    .collect(),
            };
            let mut counts = vec![0u64; k];
            for &s in &syms {
                counts[s as usize] += 1;
            }
            let f = ninths_codec::normalize_freqs(&counts, ninths_codec::SCALE_M);
            let t0 = std::time::Instant::now();
            let enc = ninths_codec::rans_encode(&syms, &f, ninths_codec::SCALE_BITS);
            let dec = ninths_codec::rans_decode(&enc, &f, syms.len(), ninths_codec::SCALE_BITS);
            assert_eq!(dec, syms, "roundtrip mismatch");
            println!(
                "{} symbols, {} bytes ({} b/sym), {} ms — roundtrip OK",
                syms.len(),
                enc.len(),
                8.0 * enc.len() as f64 / syms.len() as f64,
                t0.elapsed().as_millis(),
            );
        }
        Cli::Ent8Check { scales } => {
            let raw = std::fs::read(scales)?;
            let n = raw.len() / 4;
            let vals: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let t0 = std::time::Instant::now();
            let blob = ninths_codec::encode_scales_ent8(&vals);
            let back = ninths_codec::decode_scales_ent8(&blob, vals.len());
            let drift: f64 = vals
                .iter()
                .zip(back.iter())
                .map(|(a, b)| {
                    let a = f64::from(*a);
                    let b = f64::from(*b);
                    (a - b).abs() / b.max(1e-30)
                })
                .fold(0.0, f64::max);
            println!(
                "{} scales → {} bytes blob ({} B/scale), max rel drift {:.4}%, {} ms — OK",
                n,
                blob.len(),
                blob.len() as f64 / n as f64,
                drift * 100.0,
                t0.elapsed().as_millis(),
            );
        }
    }
    Ok(())
}