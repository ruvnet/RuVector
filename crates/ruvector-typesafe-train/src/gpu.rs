//! CUDA memory-pool hygiene.
//!
//! cudarc 0.19 allocates with `cuMemAllocAsync` from the device's default
//! memory pool and frees back into it. With per-batch shapes varying (dynamic
//! padding) plus periodic full-validation passes, the pool's reserved size
//! ratchets up through fragmentation until a 16 GB card OOMs (observed after
//! ~1500 v0 steps on the RTX 5080). Trimming the pool to 0 at quiet points
//! returns the unused reserve to the driver. No-ops on CPU / non-CUDA builds.

use anyhow::Result;
use candle_core::Device;

/// Synchronize the stream and release all unused pool memory.
#[cfg(feature = "cuda")]
pub fn trim(device: &Device) -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    let Device::Cuda(d) = device else {
        return Ok(());
    };
    let stream = d.cuda_stream();
    stream.synchronize()?;
    let dev = stream.context().cu_device();
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: plain driver-API calls on the live context's device with a valid
    // out-pointer; the pool handle is owned by the driver and not retained.
    unsafe {
        let r = sys::cuDeviceGetDefaultMemPool(&mut pool, dev);
        anyhow::ensure!(
            r == sys::CUresult::CUDA_SUCCESS,
            "cuDeviceGetDefaultMemPool: {r:?}"
        );
        let r = sys::cuMemPoolTrimTo(pool, 0);
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuMemPoolTrimTo: {r:?}");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn trim(_device: &Device) -> Result<()> {
    Ok(())
}

/// Device memory in use (MiB), for the curves; None on CPU.
#[cfg(feature = "cuda")]
pub fn used_mib(device: &Device) -> Option<u64> {
    use candle_core::cuda::cudarc::driver::sys;
    let Device::Cuda(d) = device else {
        return None;
    };
    d.cuda_stream().context().bind_to_thread().ok()?;
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: out-pointers to locals; the context is bound to this thread above.
    let r = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) };
    (r == sys::CUresult::CUDA_SUCCESS).then(|| ((total - free) / (1 << 20)) as u64)
}

#[cfg(not(feature = "cuda"))]
pub fn used_mib(_device: &Device) -> Option<u64> {
    None
}
