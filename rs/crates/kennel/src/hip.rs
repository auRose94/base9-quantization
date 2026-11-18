//! Minimal hand-rolled HIP FFI — grown as kennel needs it.
//!
//! Ground truth: /opt/rocm/include/hip/hip_runtime_api.h on the A9 Mega
//! (ROCm 10.1.0). `hipGetDeviceProperties` is a macro alias for
//! `hipGetDevicePropertiesR0600`; we bind the versioned symbol directly.
//! The props struct is replicated field-for-field down to `gcnArchName` /
//! `clockInstructionRate`; everything after (`hipDeviceArch_t arch`, HDP
//! pointers, `isLargeBar`, `asicRevision`) is opaque tail padding we never
//! read. If a future ROCm moves to R0700, linkage fails loudly here.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_void};

pub type hipError_t = c_int;
pub const HIP_SUCCESS: hipError_t = 0;

// hipMemcpyKind (driver_types.h)
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// hipDeviceProp_tR0600 (partial — see module docs). repr(C), natural
/// alignment, matching the C declaration order exactly.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct hipDeviceProp_tR0600 {
    pub name: [u8; 256],
    pub uuid: [u8; 16],
    pub luid: [u8; 8],
    pub luidDeviceNodeMask: u32,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: i32,
    pub warpSize: i32,
    pub memPitch: usize,
    pub maxThreadsPerBlock: i32,
    pub maxThreadsDim: [i32; 3],
    pub maxGridSize: [i32; 3],
    pub clockRate: i32, // kHz
    pub totalConstMem: usize,
    pub major: i32,
    pub minor: i32,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: i32,
    pub multiProcessorCount: i32, // CUs (or WGPs — see header note)
    pub kernelExecTimeoutEnabled: i32,
    pub integrated: i32,
    pub canMapHostMemory: i32,
    pub computeMode: i32,
    pub maxTexture1D: i32,
    pub maxTexture1DMipmap: i32,
    pub maxTexture1DLinear: i32,
    pub maxTexture2D: [i32; 2],
    pub maxTexture2DMipmap: [i32; 2],
    pub maxTexture2DLinear: [i32; 3],
    pub maxTexture2DGather: [i32; 2],
    pub maxTexture3D: [i32; 3],
    pub maxTexture3DAlt: [i32; 3],
    pub maxTextureCubemap: i32,
    pub maxTexture1DLayered: [i32; 2],
    pub maxTexture2DLayered: [i32; 3],
    pub maxTextureCubemapLayered: [i32; 2],
    pub maxSurface1D: i32,
    pub maxSurface2D: [i32; 2],
    pub maxSurface3D: [i32; 3],
    pub maxSurface1DLayered: [i32; 2],
    pub maxSurface2DLayered: [i32; 3],
    pub maxSurfaceCubemap: i32,
    pub maxSurfaceCubemapLayered: [i32; 2],
    pub surfaceAlignment: usize,
    pub concurrentKernels: i32,
    pub ECCEnabled: i32,
    pub pciBusID: i32,
    pub pciDeviceID: i32,
    pub pciDomainID: i32,
    pub tccDriver: i32,
    pub asyncEngineCount: i32,
    pub unifiedAddressing: i32,
    pub memoryClockRate: i32, // kHz
    pub memoryBusWidth: i32,  // bits
    pub l2CacheSize: i32,     // bytes
    pub persistingL2CacheMaxSize: i32,
    pub maxThreadsPerMultiProcessor: i32,
    pub streamPrioritiesSupported: i32,
    pub globalL1CacheSupported: i32,
    pub localL1CacheSupported: i32,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: i32,
    pub managedMemory: i32,
    pub isMultiGpuBoard: i32,
    pub multiGpuBoardGroupID: i32,
    pub hostNativeAtomicSupported: i32,
    pub singleToDoublePrecisionPerfRatio: i32,
    pub pageableMemoryAccess: i32,
    pub concurrentManagedAccess: i32,
    pub computePreemptionSupported: i32,
    pub canUseHostPointerForRegisteredMem: i32,
    pub cooperativeLaunch: i32,
    pub cooperativeMultiDeviceLaunch: i32,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: i32,
    pub directManagedMemAccessFromHost: i32,
    pub maxBlocksPerMultiProcessor: i32,
    pub accessPolicyMaxWindowSize: i32,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: i32,
    pub sparseHipArraySupported: i32,
    pub hostRegisterReadOnlySupported: i32,
    pub timelineSemaphoreInteropSupported: i32,
    pub memoryPoolsSupported: i32,
    pub gpuDirectRDMASupported: i32,
    pub gpuDirectRDMAFlushWritesOptions: u32,
    pub gpuDirectRDMAWritesOrdering: i32,
    pub memoryPoolSupportedHandleTypes: u32,
    pub deferredMappingHipArraySupported: i32,
    pub ipcEventSupported: i32,
    pub clusterLaunch: i32,
    pub unifiedFunctionPointers: i32,
    pub reserved: [i32; 63],
    pub hipReserved: [i32; 32],
    pub gcnArchName: [u8; 256],
    pub maxSharedMemoryPerMultiProcessor: usize,
    pub clockInstructionRate: i32, // kHz
    /// opaque tail: hipDeviceArch_t arch, HDP flush pointers, cooperative
    /// flags, isLargeBar, asicRevision — plus headroom. Never read.
    tail: [u8; 512],
}

const _: () = assert!(std::mem::size_of::<hipDeviceProp_tR0600>() >= 1024);

extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipGetDevicePropertiesR0600(
        prop: *mut hipDeviceProp_tR0600,
        deviceId: c_int,
    ) -> hipError_t;
    pub fn hipRuntimeGetVersion(version: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(deviceId: c_int) -> hipError_t;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        sizeBytes: usize,
        kind: c_int,
    ) -> hipError_t;
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> hipError_t;
    pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: *mut c_void) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut *mut c_void,
        module: *mut c_void,
        kname: *const c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: *mut c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        stream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipGetErrorString(hipError: hipError_t) -> *const c_char;
}

/// The fields kennel actually consumes, post-FFI.
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub gcn_arch: String,
    pub cc: (i32, i32),
    pub cu_count: i32,
    pub warp_size: i32,
    pub clock_mhz: f64,
    pub mem_clock_mhz: f64,
    pub mem_bus_bits: i32,
    pub total_mem: usize,
    pub l2_bytes: i32,
    pub shared_per_block_optin: usize,
    pub max_threads_per_cu: i32,
    pub integrated: bool,
    pub pageable_memory_access: bool,
    pub host_register_supported: bool,
}

fn cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

pub fn check(err: hipError_t, what: &str) -> anyhow::Result<()> {
    if err == HIP_SUCCESS {
        Ok(())
    } else {
        Err(anyhow::anyhow!("HIP error {err} in {what}"))
    }
}

pub fn runtime_version() -> anyhow::Result<i32> {
    let mut v: c_int = 0;
    unsafe { check(hipRuntimeGetVersion(&mut v), "hipRuntimeGetVersion")? };
    Ok(v)
}

pub fn devices() -> anyhow::Result<Vec<DeviceInfo>> {
    let mut n: c_int = 0;
    unsafe { check(hipGetDeviceCount(&mut n), "hipGetDeviceCount")? };
    let mut out = Vec::new();
    for i in 0..n {
        let mut p = hipDeviceProp_tR0600 {
            name: [0; 256],
            uuid: [0; 16],
            luid: [0; 8],
            luidDeviceNodeMask: 0,
            totalGlobalMem: 0,
            sharedMemPerBlock: 0,
            regsPerBlock: 0,
            warpSize: 0,
            memPitch: 0,
            maxThreadsPerBlock: 0,
            maxThreadsDim: [0; 3],
            maxGridSize: [0; 3],
            clockRate: 0,
            totalConstMem: 0,
            major: 0,
            minor: 0,
            textureAlignment: 0,
            texturePitchAlignment: 0,
            deviceOverlap: 0,
            multiProcessorCount: 0,
            kernelExecTimeoutEnabled: 0,
            integrated: 0,
            canMapHostMemory: 0,
            computeMode: 0,
            maxTexture1D: 0,
            maxTexture1DMipmap: 0,
            maxTexture1DLinear: 0,
            maxTexture2D: [0; 2],
            maxTexture2DMipmap: [0; 2],
            maxTexture2DLinear: [0; 3],
            maxTexture2DGather: [0; 2],
            maxTexture3D: [0; 3],
            maxTexture3DAlt: [0; 3],
            maxTextureCubemap: 0,
            maxTexture1DLayered: [0; 2],
            maxTexture2DLayered: [0; 3],
            maxTextureCubemapLayered: [0; 2],
            maxSurface1D: 0,
            maxSurface2D: [0; 2],
            maxSurface3D: [0; 3],
            maxSurface1DLayered: [0; 2],
            maxSurface2DLayered: [0; 3],
            maxSurfaceCubemap: 0,
            maxSurfaceCubemapLayered: [0; 2],
            surfaceAlignment: 0,
            concurrentKernels: 0,
            ECCEnabled: 0,
            pciBusID: 0,
            pciDeviceID: 0,
            pciDomainID: 0,
            tccDriver: 0,
            asyncEngineCount: 0,
            unifiedAddressing: 0,
            memoryClockRate: 0,
            memoryBusWidth: 0,
            l2CacheSize: 0,
            persistingL2CacheMaxSize: 0,
            maxThreadsPerMultiProcessor: 0,
            streamPrioritiesSupported: 0,
            globalL1CacheSupported: 0,
            localL1CacheSupported: 0,
            sharedMemPerMultiprocessor: 0,
            regsPerMultiprocessor: 0,
            managedMemory: 0,
            isMultiGpuBoard: 0,
            multiGpuBoardGroupID: 0,
            hostNativeAtomicSupported: 0,
            singleToDoublePrecisionPerfRatio: 0,
            pageableMemoryAccess: 0,
            concurrentManagedAccess: 0,
            computePreemptionSupported: 0,
            canUseHostPointerForRegisteredMem: 0,
            cooperativeLaunch: 0,
            cooperativeMultiDeviceLaunch: 0,
            sharedMemPerBlockOptin: 0,
            pageableMemoryAccessUsesHostPageTables: 0,
            directManagedMemAccessFromHost: 0,
            maxBlocksPerMultiProcessor: 0,
            accessPolicyMaxWindowSize: 0,
            reservedSharedMemPerBlock: 0,
            hostRegisterSupported: 0,
            sparseHipArraySupported: 0,
            hostRegisterReadOnlySupported: 0,
            timelineSemaphoreInteropSupported: 0,
            memoryPoolsSupported: 0,
            gpuDirectRDMASupported: 0,
            gpuDirectRDMAFlushWritesOptions: 0,
            gpuDirectRDMAWritesOrdering: 0,
            memoryPoolSupportedHandleTypes: 0,
            deferredMappingHipArraySupported: 0,
            ipcEventSupported: 0,
            clusterLaunch: 0,
            unifiedFunctionPointers: 0,
            reserved: [0; 63],
            hipReserved: [0; 32],
            gcnArchName: [0; 256],
            maxSharedMemoryPerMultiProcessor: 0,
            clockInstructionRate: 0,
            tail: [0; 512],
        };
        unsafe { check(hipGetDevicePropertiesR0600(&mut p, i), "hipGetDeviceProperties")? };
        out.push(DeviceInfo {
            ordinal: i as usize,
            name: cstr(&p.name),
            gcn_arch: cstr(&p.gcnArchName),
            cc: (p.major, p.minor),
            cu_count: p.multiProcessorCount,
            warp_size: p.warpSize,
            clock_mhz: p.clockRate as f64 / 1000.0,
            mem_clock_mhz: p.memoryClockRate as f64 / 1000.0,
            mem_bus_bits: p.memoryBusWidth,
            total_mem: p.totalGlobalMem,
            l2_bytes: p.l2CacheSize,
            shared_per_block_optin: p.sharedMemPerBlockOptin,
            max_threads_per_cu: p.maxThreadsPerMultiProcessor,
            integrated: p.integrated != 0,
            pageable_memory_access: p.pageableMemoryAccess != 0,
            host_register_supported: p.hostRegisterSupported != 0,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------- runtime helpers

pub fn error_string(err: hipError_t) -> String {
    unsafe {
        let p = hipGetErrorString(err);
        if p.is_null() {
            format!("hip error {err}")
        } else {
            let mut len = 0;
            while *p.add(len) != 0 {
                len += 1;
            }
            String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len)).into_owned()
        }
    }
}

pub fn check2(err: hipError_t, what: &str) -> anyhow::Result<()> {
    if err == HIP_SUCCESS {
        Ok(())
    } else {
        Err(anyhow::anyhow!("HIP error {} ({}) in {what}", error_string(err), err))
    }
}

/// Owned device allocation.
pub struct DevBuf {
    ptr: *mut c_void,
    pub len: usize,
}

impl DevBuf {
    pub fn alloc(len: usize) -> anyhow::Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe { check2(hipMalloc(&mut ptr, len), "hipMalloc")? };
        Ok(DevBuf { ptr, len })
    }

    /// Chunked HtoD: ROCm 10.1 on gfx1151 segfaults inside libamdhip64 on
    /// large single-pageable hipMemcpy calls (~268 MB+ reproducibly);
    /// 8 MiB pieces dodge the broken path at negligible cost.
    const COPY_CHUNK: usize = 8 << 20;

    pub fn upload(&self, host: &[u8]) -> anyhow::Result<()> {
        if host.len() > self.len {
            anyhow::bail!("upload: {} bytes into {} byte buffer", host.len(), self.len);
        }
        for (off, piece) in host.chunks(Self::COPY_CHUNK).enumerate() {
            unsafe {
                check2(
                    hipMemcpy(
                        (self.ptr as *mut u8).add(off * Self::COPY_CHUNK).cast(),
                        piece.as_ptr() as *const c_void,
                        piece.len(),
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "hipMemcpy HtoD",
                )?;
            }
        }
        Ok(())
    }

    pub fn download(&self, host: &mut [u8]) -> anyhow::Result<()> {
        if host.len() > self.len {
            anyhow::bail!("download: {} bytes into {} byte buffer", host.len(), self.len);
        }
        for (off, piece) in host.chunks_mut(Self::COPY_CHUNK).enumerate() {
            unsafe {
                check2(
                    hipMemcpy(
                        piece.as_mut_ptr() as *mut c_void,
                        (self.ptr as *const u8).add(off * Self::COPY_CHUNK).cast(),
                        piece.len(),
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "hipMemcpy DtoH",
                )?;
            }
        }
        Ok(())
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hipFree(self.ptr) };
        }
    }
}

/// Pinned staging buffer (hipHostMalloc) — the ggml-style safe path for
/// pageable HtoD/DtoH on this box: raw pageable hipMemcpy SIGSEGVs inside
/// libamdhip64 depending on process VA layout (see tests below), while
/// copies sourced from pinned memory take the direct runtime path.
pub struct Staging {
    ptr: *mut c_void,
    pub len: usize,
}

impl Staging {
    pub fn new(len: usize) -> anyhow::Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe { check2(hipHostMalloc(&mut ptr, len, 0), "hipHostMalloc")? };
        Ok(Staging { ptr, len })
    }

    /// host -> staged -> device, one staging-sized piece at a time.
    pub fn upload(&self, host: &[u8], dev: &DevBuf) -> anyhow::Result<()> {
        for (off, piece) in host.chunks(self.len).enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(piece.as_ptr(), self.ptr as *mut u8, piece.len());
                check2(
                    hipMemcpy(
                        (dev.as_ptr() as *mut u8).add(off * self.len).cast(),
                        self.ptr,
                        piece.len(),
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "hipMemcpy staged HtoD",
                )?;
            }
        }
        Ok(())
    }

    /// device -> staged -> host.
    pub fn download(&self, dev: &DevBuf, host: &mut [u8]) -> anyhow::Result<()> {
        for (off, piece) in host.chunks_mut(self.len).enumerate() {
            unsafe {
                check2(
                    hipMemcpy(
                        self.ptr,
                        (dev.as_ptr() as *const u8).add(off * self.len).cast(),
                        piece.len(),
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "hipMemcpy staged DtoH",
                )?;
                std::ptr::copy_nonoverlapping(self.ptr as *const u8, piece.as_mut_ptr(), piece.len());
            }
        }
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hipHostFree(self.ptr) };
        }
    }
}

/// A loaded code object (hipModuleLoadData over embedded .co bytes).
pub struct Module {
    handle: *mut c_void,
}

impl Module {
    pub fn load(image: &[u8]) -> anyhow::Result<Self> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        unsafe { check2(hipModuleLoadData(&mut handle, image.as_ptr() as *const c_void), "hipModuleLoadData")? };
        Ok(Module { handle })
    }

    pub fn function(&self, name: &str) -> anyhow::Result<*mut c_void> {
        let cname = std::ffi::CString::new(name)?;
        let mut f: *mut c_void = std::ptr::null_mut();
        unsafe { check2(hipModuleGetFunction(&mut f, self.handle, cname.as_ptr()), "hipModuleGetFunction")? };
        if f.is_null() {
            anyhow::bail!("kernel {name} not found in module");
        }
        Ok(f)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { hipModuleUnload(self.handle) };
        }
    }
}

/// Launch `f` on the null stream. `args` = one pointer-to-argument per
/// kernel parameter (the classic CUDA kernelParams contract).
pub fn launch(
    f: *mut c_void,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    args: &mut [*mut c_void],
) -> anyhow::Result<()> {
    unsafe {
        check2(
            hipModuleLaunchKernel(
                f, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared,
                std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut(),
            ),
            "hipModuleLaunchKernel",
        )
    }
}

pub fn sync() -> anyhow::Result<()> {
    unsafe { check2(hipDeviceSynchronize(), "hipDeviceSynchronize") }
}

pub fn set_device(i: c_int) -> anyhow::Result<()> {
    unsafe { check2(hipSetDevice(i), "hipSetDevice") }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal repro for the flaky SIGSEGV seen in big pageable HtoD:
    /// no container, no threads, no mmap — just alloc + chunked upload +
    /// download at the sizes the gate uses.
    #[test]
    fn big_upload_roundtrip() {
        set_device(0).unwrap();
        let n = 268_435_456;
        let host = vec![0x5au8; n];
        let mut back = vec![0u8; n];
        for round in 0..3 {
            let d = DevBuf::alloc(n).unwrap();
            d.upload(&host).unwrap();
            d.download(&mut back).unwrap();
            assert_eq!(back[..1024], host[..1024]);
            drop(d);
            eprintln!("round {round} ok");
        }
    }

    /// Gate-shaped: three live DevBufs (digits, scales, out) with the big
    /// out buffer allocated LAST, then chunked upload into the FIRST.
    /// REPRODUCES the ROCm 10.1 / gfx1151 allocation-order bug (SIGSEGV in
    /// hipMemcpy's CPU-side GTT alias). Kept #[ignore] as living
    /// documentation + regression probe for future ROCm updates: un-ignore
    /// and re-run when the driver changes. Workaround: allocate big-first
    /// (see `va_big_first`).
    #[test]
    #[ignore = "reproduces ROCm 10.1 gfx1151 alloc-order bug — see docs/16"]
    fn gate_shaped_three_live_allocs() {
        set_device(0).unwrap();
        let n = 268_435_456;
        let digits = vec![0x5au8; n];
        let scales = vec![1.0f32; 4_194_304];
        let d_digits = DevBuf::alloc(n).unwrap();
        let d_scales = DevBuf::alloc(scales.len() * 4).unwrap();
        let d_out = DevBuf::alloc(n * 4).unwrap();
        d_digits.upload(&digits).unwrap();
        let sb = unsafe {
            std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
        };
        d_scales.upload(sb).unwrap();
        let mut back = vec![0u8; n * 4];
        d_out.download(&mut back).unwrap();
        drop((d_digits, d_scales, d_out));
        eprintln!("gate-shaped ok");
    }

    /// Variant A: same three sizes, but BIG allocated FIRST.
    #[test]
    fn va_big_first() {
        set_device(0).unwrap();
        let host = vec![0x5au8; 268_435_456];
        let d_out = DevBuf::alloc(268_435_456 * 4).unwrap();
        let d_digits = DevBuf::alloc(268_435_456).unwrap();
        let d_scales = DevBuf::alloc(16 << 20).unwrap();
        d_digits.upload(&host).unwrap();
        d_scales.upload(&host[..16 << 20]).unwrap();
        drop((d_out, d_digits, d_scales));
        eprintln!("big-first ok");
    }

    /// Variant B: two allocs only (268 MB + 1 GiB), upload the small one.
    #[test]
    fn va_two_allocs() {
        set_device(0).unwrap();
        let host = vec![0x5au8; 268_435_456];
        let d_small = DevBuf::alloc(268_435_456).unwrap();
        let d_big = DevBuf::alloc(268_435_456 * 4).unwrap();
        d_small.upload(&host).unwrap();
        drop((d_small, d_big));
        eprintln!("two-allocs ok");
    }

    /// Size-threshold probe: uploads at 1/8/32/64/128/256 MiB — after
    /// driver-state degradation only the big ones should die.
    #[test]
    fn va_size_ladder() {
        set_device(0).unwrap();
        for log2 in [20, 23, 25, 26, 27, 28] {
            let n = 1usize << log2;
            let host = vec![0x5au8; n];
            let d = DevBuf::alloc(n).unwrap();
            d.upload(&host).unwrap();
            drop(d);
            drop(host);
            eprintln!("{} MiB ok", n >> 20);
        }
    }

    /// Variant C: three allocs, all under 512 MiB.
    #[test]
    fn va_all_small() {
        set_device(0).unwrap();
        let host = vec![0x5au8; 1 << 28];
        let a = DevBuf::alloc(1 << 28).unwrap();
        let b = DevBuf::alloc(1 << 24).unwrap();
        let c = DevBuf::alloc(1 << 29).unwrap();
        a.upload(&host).unwrap();
        drop((a, b, c));
        eprintln!("all-small ok");
    }

    /// Variant D: full-run context — big file mmap + 32-thread phase +
    /// ~1.3 GiB live heap, then BIG-FIRST allocs + upload. Bisect from
    /// here if it crashes.
    #[test]
    fn va_full_context() {
        set_device(0).unwrap();
        let n = 268_435_456usize;
        // mmap the local flint k5 container (5.3 GiB VA)
        let f = std::fs::File::open("/home/rose/Work/models/flint_k5_k99_v2.gguf").unwrap();
        let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
        let _ = map[..4096].to_vec(); // touch it
        // threaded phase over a 1 GiB vec, like dequant_f32
        let mut cpu = vec![0f32; n];
        std::thread::scope(|s| {
            for chunk in cpu.chunks_mut(n / 32) {
                s.spawn(move || {
                    for (i, w) in chunk.iter_mut().enumerate() {
                        *w = i as f32 * 1e-9;
                    }
                });
            }
        });
        let host = vec![0x5au8; n];
        let d_out = DevBuf::alloc(n * 4).unwrap();
        let d_digits = DevBuf::alloc(n).unwrap();
        let d_scales = DevBuf::alloc(16 << 20).unwrap();
        d_digits.upload(&host).unwrap();
        drop((d_out, d_digits, d_scales));
        drop((cpu, host, map));
        eprintln!("full-context ok");
    }

    /// Bisect helpers for va_full_context: each drops ONE context piece.
    #[test]
    fn va_d1_no_mmap() {
        set_device(0).unwrap();
        let n = 268_435_456usize;
        let mut cpu = vec![0f32; n];
        std::thread::scope(|s| {
            for chunk in cpu.chunks_mut(n / 32) {
                s.spawn(move || {
                    for (i, w) in chunk.iter_mut().enumerate() {
                        *w = i as f32 * 1e-9;
                    }
                });
            }
        });
        let host = vec![0x5au8; n];
        let d_out = DevBuf::alloc(n * 4).unwrap();
        let d_digits = DevBuf::alloc(n).unwrap();
        let d_scales = DevBuf::alloc(16 << 20).unwrap();
        d_digits.upload(&host).unwrap();
        drop((d_out, d_digits, d_scales));
        eprintln!("d1 (no mmap) ok");
    }

    #[test]
    fn va_d2_no_threads() {
        set_device(0).unwrap();
        let n = 268_435_456usize;
        let f = std::fs::File::open("/home/rose/Work/models/flint_k5_k99_v2.gguf").unwrap();
        let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
        let _ = map[..4096].to_vec();
        let cpu = vec![0f32; n]; // single-threaded, never written by threads
        let host = vec![0x5au8; n];
        let d_out = DevBuf::alloc(n * 4).unwrap();
        let d_digits = DevBuf::alloc(n).unwrap();
        let d_scales = DevBuf::alloc(16 << 20).unwrap();
        d_digits.upload(&host).unwrap();
        drop((d_out, d_digits, d_scales));
        drop((cpu, host, map));
        eprintln!("d2 (no threads) ok");
    }

    #[test]
    fn va_d3_no_heap() {
        set_device(0).unwrap();
        let n = 268_435_456usize;
        let f = std::fs::File::open("/home/rose/Work/models/flint_k5_k99_v2.gguf").unwrap();
        let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
        let _ = map[..4096].to_vec();
        let mut cpu = vec![0f32; 1 << 20]; // small heap instead of 1 GiB
        std::thread::scope(|s| {
            for chunk in cpu.chunks_mut(1 << 14) {
                s.spawn(move || {
                    for (i, w) in chunk.iter_mut().enumerate() {
                        *w = i as f32 * 1e-9;
                    }
                });
            }
        });
        let host = vec![0x5au8; n];
        let d_out = DevBuf::alloc(n * 4).unwrap();
        let d_digits = DevBuf::alloc(n).unwrap();
        let d_scales = DevBuf::alloc(16 << 20).unwrap();
        d_digits.upload(&host).unwrap();
        drop((d_out, d_digits, d_scales));
        eprintln!("d3 (no big heap) ok");
    }
}
