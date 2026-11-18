//! Link against ROCm's HIP runtime and compile the gate kernels for the
//! ONE target kennel exists for: gfx1151 (A9 Mega / 8060S).
//! ROCM_PATH overrides the default /opt/rocm prefix.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // rpath so the binary runs without ldconfig knowing about /opt/rocm/lib
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernel_src = manifest.join("kernels/dequant.hip");
    let co = out_dir.join("kennel_kernels.co");

    // hipcc needs a sanitized env on this box (gcc cc1 fails under the
    // login PATH; verified recipe: clean env, PATH=/opt/rocm/bin:/usr/bin:/bin)
    let status = Command::new(format!("{rocm}/bin/hipcc"))
        .env_clear()
        .env("PATH", format!("{rocm}/bin:/usr/bin:/bin"))
        .env("HOME", "/tmp")
        .args(["--genco", "--offload-arch=gfx1151", "-o"])
        .arg(&co)
        .arg(&kernel_src)
        .status()
        .expect("failed to spawn hipcc — is ROCm installed?");

    if !status.success() {
        panic!("hipcc --genco failed for kernels/dequant.hip ({status})");
    }

    println!("cargo:rustc-env=KENNEL_KERNEL_CO={}", co.display());
    println!("cargo:rerun-if-changed={}", kernel_src.display());
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=build.rs");
}
