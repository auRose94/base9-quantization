//! kennel — experimental LLM engine for the ninths/k9 family.
//!
//! Thesis: co-design the engine with the container format. K9/ninths can
//! MATERIALIZE weights in any per-tensor layout, so kennel is free to
//! demand the layout its kernels want, per model, instead of inheriting
//! one-size-fits-all GGUF shapes.
//!
//! Hardware target: exactly one machine — the A9 Mega (Ryzen AI Max+ 395,
//! Radeon 8060S / gfx1151, 128 GB unified LPDDR5X, ROCm 10.1). Lessons
//! from the k9 fork's M3-HIP work (anonymous memory invisible to gfx1151
//! kernels; hipHostRegister regions are the coherent path) are load-bearing
//! here from day one.
//!
//! Stage 1 (llmhamster-limited CLI): device probe → GGUF/container read →
//! GPU dequant, byte-exact-gated against ninths-codec. Later: kernels,
//! forward pass, serve.

mod gguf;
mod hip;
mod k9;

use clap::Parser as _;
use std::path::PathBuf;

const KERNEL_CO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kennel_kernels.co"));

#[derive(clap::Parser, Debug)]
#[command(name = "kennel", version, about = "experimental ninths engine for Strix Halo (gfx1151)")]
enum Cli {
    /// List HIP devices visible to the runtime (expect gfx1151)
    Devices,
    /// Dump a GGUF container: metadata KV + tensor census
    Info {
        /// path to a .gguf container (any: stock, K9-typed, .k9s)
        file: PathBuf,
        /// show this many tensor rows (default 12)
        #[arg(short = 'n', long, default_value_t = 12)]
        tensors: usize,
    },
    /// Decode + dequant one K9 tensor, GPU vs CPU, byte-exact gate
    Dequant {
        /// path to a K9/ninths container (.gguf with a k9 directory)
        file: PathBuf,
        /// substring of the directory record name (must match exactly one)
        tensor: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli {
        Cli::Devices => {
            let rt = hip::runtime_version()?;
            println!("HIP runtime version: {rt}");
            for d in hip::devices()? {
                println!();
                println!("[{}] {} ({})", d.ordinal, d.name, d.gcn_arch);
                println!(
                    "  compute capability {}.{}  |  {} CUs  |  warp {}  |  {} threads/CU",
                    d.cc.0, d.cc.1, d.cu_count, d.warp_size, d.max_threads_per_cu
                );
                println!(
                    "  clocks: gpu {:.0} MHz, mem {:.0} MHz  |  bus {} bit  |  L2 {} KiB",
                    d.clock_mhz,
                    d.mem_clock_mhz,
                    d.mem_bus_bits,
                    d.l2_bytes / 1024
                );
                println!(
                    "  memory: {:.1} GiB unified (integrated: {})",
                    d.total_mem as f64 / (1 << 30) as f64,
                    d.integrated
                );
                println!(
                    "  LDS opt-in: {} KiB/block",
                    d.shared_per_block_optin / 1024
                );
                // the two flags the M3-HIP rung lived and died by:
                println!(
                    "  pageableMemoryAccess: {}  |  hostRegisterSupported: {}",
                    d.pageable_memory_access, d.host_register_supported
                );
            }
            Ok(())
        }
        Cli::Info { file, tensors } => {
            let g = gguf::Gguf::open(&file)?;
            println!(
                "{} — gguf v{}, {} kv, {} tensors, align {}, data @{:#x}",
                file.display(),
                g.version,
                g.kv.len(),
                g.tensors.len(),
                g.alignment,
                g.data_offset
            );
            println!("--- container metadata ---");
            for (k, v) in &g.kv {
                let interesting = k.starts_with("general.")
                    || k.starts_with("ninths.")
                    || k.starts_with("k9.")
                    || k.ends_with(".embedding_length")
                    || k.ends_with(".block_count")
                    || k.ends_with(".expert_count")
                    || k.ends_with(".expert_used_count")
                    || k.ends_with(".attention.head_count");
                if interesting {
                    println!("  {k} = {}", v.brief());
                }
            }
            // census by type
            let mut by_type: std::collections::BTreeMap<String, usize> = Default::default();
            for t in &g.tensors {
                *by_type.entry(gguf::type_name(t.ty)).or_default() += 1;
            }
            println!("--- tensor census ---");
            let mut rows: Vec<(&String, &usize)> = by_type.iter().collect();
            rows.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
            for (ty, n) in rows {
                println!("  {n:4}  {ty}");
            }
            println!("--- first tensors ---");
            for t in g.tensors.iter().take(tensors) {
                let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
                println!("  {:52} {:>10}  [{}]", t.name, gguf::type_name(t.ty), dims.join("×"));
            }
            Ok(())
        }
        Cli::Dequant { file, tensor } => {
            let t0 = std::time::Instant::now();
            // Scope the container: its multi-GiB mmap must be DEAD before
            // any GPU call — ROCm 10.1/gfx1151 segfaults inside pageable
            // HtoD when a giant file VMA is alive (hip::tests::va_d2_*),
            // even with largest-first device allocations. Engine-level
            // fix later: pread or pinned staging (ggml's approach).
            let (dir, rec, blob) = {
                let g = gguf::Gguf::open(&file)?;
                let (dir_bytes, v2) = g
                    .k9_directory()?
                    .ok_or_else(|| anyhow::anyhow!("no ninths/k9 directory in {}", file.display()))?;
                let dir = k9::parse_directory(&dir_bytes, v2)?;
                let matches: Vec<&k9::Record> =
                    dir.iter().filter(|r| r.name.contains(&tensor)).collect();
                let rec = match matches.as_slice() {
                    [one] => (*one).clone(),
                    [] => anyhow::bail!(
                        "no record matches {:?}; e.g. {}",
                        tensor,
                        dir[0].name
                    ),
                    many => anyhow::bail!(
                        "{} records match {:?}: {}",
                        many.len(),
                        tensor,
                        many.iter().map(|r| r.name.as_str()).take(4).collect::<Vec<_>>().join(", ")
                    ),
                };
                if rec.perm_flag != 0 {
                    anyhow::bail!("record {} is permuted — F16 materialization path is a later rung", rec.name);
                }
                let tinfo = g
                    .tensors
                    .iter()
                    .find(|t| t.name == rec.name)
                    .ok_or_else(|| anyhow::anyhow!("record {} has no GGUF tensor entry", rec.name))?;
                let blob = g.tensor_bytes(tinfo)?.to_vec();
                (dir, rec, blob)
            }; // <- Gguf (and its mmap) drops here
            if std::env::var("KENNEL_DEBUG_MAPS").is_ok() {
                let maps = std::fs::read_to_string("/proc/self/maps")?;
                let big: Vec<&str> = maps
                    .lines()
                    .filter(|l| {
                        let Some((range, _)) = l.split_once(' ') else { return false };
                        let (a, b) = range.split_once('-').unwrap();
                        let lo = u64::from_str_radix(a, 16).unwrap();
                        let hi = u64::from_str_radix(b, 16).unwrap();
                        hi - lo > (1 << 30)
                    })
                    .collect();
                eprintln!("VMAs > 1 GiB at GPU phase: {}", big.len());
                for l in big.iter().take(4) {
                    eprintln!("  {l}");
                }
            }
            println!(
                "container: {} — k9 directory ({} records), mmap released [{} ms]",
                file.display(),
                dir.len(),
                t0.elapsed().as_millis()
            );
            println!(
                "record: {} — {}x{} k={} lanes={} scale_mode={} blob {} B",
                rec.name, rec.rows, rec.cols, rec.k, rec.lanes, rec.scale_mode, blob.len()
            );

            // ---- CPU: digits, scales, normative dequant ----
            let t = std::time::Instant::now();
            let digits = k9::decode_digits(&rec, &blob)?;
            let n = digits.len();
            println!(
                "  digits decode: {n} in {} ms ({:.0} MB/s)",
                t.elapsed().as_millis(),
                n as f64 / 1e6 / t.elapsed().as_secs_f64().max(1e-9)
            );

            let t = std::time::Instant::now();
            let scales = k9::decode_scales(&rec, &blob)?;
            println!(
                "  scales decode: {} f32 in {} ms",
                scales.len(),
                t.elapsed().as_millis()
            );

            let t = std::time::Instant::now();
            let cpu = k9::dequant_f32(&rec, &digits, &scales)?;
            println!(
                "  cpu dequant:   {n} w in {} ms ({:.0} MW/s)",
                t.elapsed().as_millis(),
                n as f64 / 1e6 / t.elapsed().as_secs_f64().max(1e-9)
            );

            // ---- GPU: upload, kernel, download ----
            // Two ROCm 10.1/gfx1151 rules live here (hip::tests::* document
            // both): allocate device buffers LARGEST-FIRST, and never copy
            // pageable memory directly — everything goes through pinned
            // staging (the ggml pattern).
            hip::set_device(0)?;
            let d_out = hip::DevBuf::alloc(n * 4)?;
            let d_digits = hip::DevBuf::alloc(n)?;
            let d_scales = hip::DevBuf::alloc(scales.len() * 4)?;
            let staging = hip::Staging::new(16 << 20)?;

            let t = std::time::Instant::now();
            staging.upload(&digits, &d_digits)?;
            let scale_bytes =
                unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4) };
            staging.upload(scale_bytes, &d_scales)?;
            println!(
                "  upload: {} + {} B in {} ms (pinned staged)",
                n,
                scale_bytes.len(),
                t.elapsed().as_millis()
            );

            let module = hip::Module::load(KERNEL_CO)?;
            let func = module.function("k9_dequant_f32")?;
            let mut p_digits = d_digits.as_ptr();
            let mut p_scales = d_scales.as_ptr();
            let mut p_out = d_out.as_ptr();
            let mut n64 = n as u64;
            let mut cols = rec.cols as u32;
            let mut n_groups = (rec.cols / rec.group as usize) as u32;
            let mut h_int = ((rec.k - 1) / 2) as i32;
            let mut h_flt = h_int as f32;
            let mut args: [*mut std::ffi::c_void; 8] = [
                (&mut p_digits as *mut *mut std::ffi::c_void).cast(),
                (&mut p_scales as *mut *mut std::ffi::c_void).cast(),
                (&mut p_out as *mut *mut std::ffi::c_void).cast(),
                (&mut n64 as *mut u64).cast(),
                (&mut cols as *mut u32).cast(),
                (&mut n_groups as *mut u32).cast(),
                (&mut h_int as *mut i32).cast(),
                (&mut h_flt as *mut f32).cast(),
            ];

            let block: u32 = 256;
            let grid_x = (n64 + block as u64 - 1) as u32 / block;
            let t = std::time::Instant::now();
            hip::launch(func, (grid_x, 1, 1), (block, 1, 1), 0, &mut args)?;
            hip::sync()?;
            let kernel_ms = t.elapsed().as_secs_f64() * 1e3;
            let bytes_touched = n + scale_bytes.len() + n * 4;
            println!(
                "  kernel: {n} w in {kernel_ms:.2} ms ({:.0} GB/s effective of ~200)",
                bytes_touched as f64 / 1e9 / t.elapsed().as_secs_f64().max(1e-9)
            );

            let mut gpu_bytes = vec![0u8; n * 4];
            let t = std::time::Instant::now();
            staging.download(&d_out, &mut gpu_bytes)?;
            println!("  download: {} B in {} ms (pinned staged)", gpu_bytes.len(), t.elapsed().as_millis());

            // ---- gate: bit-exact vs CPU ----
            let gpu: &[f32] = unsafe {
                std::slice::from_raw_parts(gpu_bytes.as_ptr() as *const f32, n)
            };
            let mut mismatches = 0usize;
            let mut first: Option<(usize, f32, f32)> = None;
            for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
                if a.to_bits() != b.to_bits() {
                    mismatches += 1;
                    if first.is_none() {
                        first = Some((i, *a, *b));
                    }
                }
            }
            if mismatches == 0 {
                println!("GATE PASS: {n} weights bit-exact (0 ulp)");
            } else {
                if let Some((i, a, b)) = first {
                    println!("  first mismatch @ {i}: cpu {a:e} vs gpu {b:e}");
                }
                println!("GATE FAIL: {mismatches} / {n} mismatched");
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
