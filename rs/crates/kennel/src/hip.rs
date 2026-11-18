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

use std::ffi::c_int;

pub type hipError_t = c_int;
pub const HIP_SUCCESS: hipError_t = 0;

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
