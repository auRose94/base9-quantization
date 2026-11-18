//! Link against ROCm's HIP runtime (libamdhip64).
//! ROCM_PATH overrides the default /opt/rocm prefix.

fn main() {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // rpath so the binary runs without ldconfig knowing about /opt/rocm/lib
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=build.rs");
}
