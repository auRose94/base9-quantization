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

use clap::Parser as _;
use std::path::PathBuf;

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
    }
}
