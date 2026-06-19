//! Fused CUDA kernel for one Adaptive Computation Time iteration.
//!
//! # What is fused here
//!
//! One iteration of the ACT halting loop (in [`super::recurrent`]) computes:
//!
//! ```text
//! p_eff     = p  * not_halted
//! new_cum   = cum + p_eff
//! will_halt = (new_cum >= threshold) * not_halted   → {0,1}
//! still_run = not_halted - will_halt
//! w         = will_halt*(1-cum) + still_run*p_eff   ← weight for h accumulation
//! cum       += still_run * p_eff
//! not_halted = still_run
//! depth     += will_halt * ((t+1) - depth)
//! ```
//!
//! With Candle tensor ops this is 8–10 separate kernel dispatches per iteration.
//! This module collapses the 8 element-wise steps into **one CUDA kernel** that
//! operates on `n = b * seq` elements, fitting comfortably in L1 cache for the
//! common case (seq ≤ 512).
//!
//! # Staging-buffer integration (ADR-258 option 3)
//!
//! `ruvllm` reaches the GPU through Candle's abstraction.  Candle does not
//! expose raw device pointers, so this module creates its own
//! `cudarc::driver::CudaDevice::new(0)` alongside Candle's device.  Per
//! iteration:
//!
//! 1. `p` (halt probability, F32 or BF16) is pulled from the Candle tensor via
//!    `to_vec1()` and pushed to a cudarc staging buffer (`htod_sync_copy`).
//! 2. The fused kernel launches (all 8 ACT ops in one pass).
//! 3. `w_out` is copied back (`dtoh_sync_copy`) and wrapped in a new Candle
//!    tensor for the `h * w` accumulation.
//!
//! Two blocking H2D + D2H transfers per iteration are the staging overhead.
//! They are small (~0.5–1 µs for n ≤ 512) but they prevent true zero-copy.
//! The "upstream `Tensor::cuda_device_ptr()`" path eliminates them; see
//! ADR-258 near-term plan.
//!
//! # Safety
//!
//! Creating a second `CudaDevice` context for device 0 is safe in a
//! single-threaded benchmark or model-forward pass.  In a concurrent setting
//! the caller must ensure candle and this module do not issue operations
//! concurrently to the same device without explicit synchronisation.

#![cfg(all(feature = "candle", feature = "cuda", feature = "fused-act"))]

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use once_cell::sync::OnceCell;

use crate::error::{Result, RuvLLMError};

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled at runtime via nvrtc)
// ---------------------------------------------------------------------------
//
// BF16 conversion is handled by a bit-cast helper so no cuda_bf16.h include
// is required — nvrtc finds headers from the CUDA toolkit install detected by
// cudarc's build.rs, which may not be available in all environments.

const ACT_KERNEL_SRC: &str = r#"
// act_fused.cu — fused ACT step for OpenMythos / RDT recurrent loops.
//
// Each CUDA thread handles one token position.  The state arrays live entirely
// in registers or L1 cache for the common case (n <= 512), so global-memory
// traffic is reduced to one read and one write per state element per iteration.

// BF16 → F32 without cuda_bf16.h: BF16 occupies the upper 16 bits of F32.
__device__ __forceinline__ float bf16_to_f32(unsigned short x) {
    return __int_as_float((unsigned int)x << 16);
}

// F32 variant: p tensor already in F32.
extern "C" __global__ void act_fused_step_f32(
    const float*         p,            // [n] halt prob (F32, read-only)
    float* __restrict__  cum,          // [n] cumulative prob       (in-out)
    float* __restrict__  not_halted,   // [n] 1.0=running, 0.0=done (in-out)
    float* __restrict__  depth,        // [n] halt iteration index  (in-out)
    float* __restrict__  w_out,        // [n] weight for h_out accum (out)
    int   n,
    float threshold,
    float step_plus_one
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float pi = p[i];
    float ci = cum[i];
    float ni = not_halted[i];
    float di = depth[i];

    float p_eff   = pi * ni;
    float new_cum = ci + p_eff;
    float wh      = (new_cum >= threshold && ni > 0.5f) ? 1.0f : 0.0f;
    float still   = ni - wh;

    w_out[i]      = wh * (1.0f - ci) + still * p_eff;
    cum[i]        = ci + still * p_eff;
    not_halted[i] = still;
    depth[i]      = di + wh * (step_plus_one - di);
}

// BF16 variant: p tensor in BF16, passed as raw u16 bits.
extern "C" __global__ void act_fused_step_bf16(
    const unsigned short* p,            // [n] halt prob (BF16 as u16, read-only)
    float* __restrict__   cum,
    float* __restrict__   not_halted,
    float* __restrict__   depth,
    float* __restrict__   w_out,
    int   n,
    float threshold,
    float step_plus_one
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float pi = bf16_to_f32(p[i]);
    float ci = cum[i];
    float ni = not_halted[i];
    float di = depth[i];

    float p_eff   = pi * ni;
    float new_cum = ci + p_eff;
    float wh      = (new_cum >= threshold && ni > 0.5f) ? 1.0f : 0.0f;
    float still   = ni - wh;

    w_out[i]      = wh * (1.0f - ci) + still * p_eff;
    cum[i]        = ci + still * p_eff;
    not_halted[i] = still;
    depth[i]      = di + wh * (step_plus_one - di);
}
"#;

// ---------------------------------------------------------------------------
// cudarc 0.19 API:
//   CudaContext::new(ordinal)   → Arc<CudaContext>  (was CudaDevice)
//   ctx.default_stream()        → Arc<CudaStream>
//   ctx.load_module(ptx)        → Arc<CudaModule>   (was dev.load_ptx)
//   module.load_function(name)  → CudaFunction      (was dev.get_func)
//   stream.clone_htod(&slice)   → CudaSlice<T>      (was dev.htod_sync_copy)
//   stream.clone_dtoh(&dev)     → Vec<T>            (was dev.dtoh_sync_copy)
//   stream.launch_builder(&f).arg(&x).launch(cfg)   (was f.launch(cfg, tuple))
//
// PTX is compiled once (OnceCell<Ptx>) but loaded per CudaContext instance
// (each has its own module table).
// ---------------------------------------------------------------------------

use cudarc::nvrtc::Ptx;

static COMPILED_PTX: OnceCell<Ptx> = OnceCell::new();

fn get_or_compile_ptx() -> Result<Ptx> {
    COMPILED_PTX
        .get_or_try_init(|| {
            compile_ptx(ACT_KERNEL_SRC)
                .map_err(|e| RuvLLMError::Model(format!("nvrtc compile act_fused: {e}")))
        })
        .cloned()
}

fn load_module(ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>> {
    let ptx = get_or_compile_ptx()?;
    ctx.load_module(ptx)
        .map_err(|e| RuvLLMError::Model(format!("cudarc load_module: {e}")))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Persistent ACT state on device 0 across loop iterations.
///
/// Create one per forward pass (or cache it in the model struct for repeated
/// inference at the same batch×sequence shape).
pub struct FusedActKernel {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    cum: CudaSlice<f32>,
    not_halted: CudaSlice<f32>,
    depth: CudaSlice<f32>,
    w_out: CudaSlice<f32>,
    /// `b * seq` — total token positions managed by this kernel context.
    pub n: usize,
}

impl FusedActKernel {
    /// Allocate device state for `n = b * seq` token positions.
    ///
    /// Compiles and loads the PTX on the first call (cached for subsequent
    /// calls in the same process).
    pub fn new(n: usize) -> Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| RuvLLMError::Model(format!("cudarc CudaContext::new: {e}")))?;
        let stream = ctx.default_stream();
        let module = load_module(&ctx)?;

        // Initialise state: cum=0, not_halted=1, depth=0, w_out=0.
        let cum = stream
            .clone_htod(vec![0.0f32; n].as_slice())
            .map_err(|e| RuvLLMError::Model(format!("htod cum: {e}")))?;
        let not_halted = stream
            .clone_htod(vec![1.0f32; n].as_slice())
            .map_err(|e| RuvLLMError::Model(format!("htod not_halted: {e}")))?;
        let depth = stream
            .clone_htod(vec![0.0f32; n].as_slice())
            .map_err(|e| RuvLLMError::Model(format!("htod depth: {e}")))?;
        let w_out = stream
            .clone_htod(vec![0.0f32; n].as_slice())
            .map_err(|e| RuvLLMError::Model(format!("htod w_out: {e}")))?;

        Ok(Self {
            stream,
            module,
            cum,
            not_halted,
            depth,
            w_out,
            n,
        })
    }

    /// Run one ACT iteration.
    ///
    /// - `p_tensor`: Candle tensor shaped `[b, seq, 1]`, dtype F32 or BF16.
    /// - Returns `w`: Candle F32 tensor `[b, seq, 1]` on the **same device** as
    ///   `p_tensor`, ready for `h.broadcast_mul(&w)`.
    pub fn step(
        &mut self,
        p_tensor: &Tensor,
        b: usize,
        seq: usize,
        threshold: f32,
        t: usize,
    ) -> Result<Tensor> {
        let p_flat = p_tensor
            .flatten_all()
            .map_err(|e| RuvLLMError::Model(format!("flatten p: {e}")))?;

        // Scalar kernel args must be stack-local so their addresses are valid
        // through the launch_builder().launch() call.
        let n_i32 = self.n as i32;
        let step_f32 = (t + 1) as f32;
        let cfg = LaunchConfig::for_num_elems(self.n as u32);

        match p_flat.dtype() {
            DType::F32 => {
                let p_host: Vec<f32> = p_flat
                    .to_vec1()
                    .map_err(|e| RuvLLMError::Model(format!("to_vec1 f32: {e}")))?;
                let p_dev = self
                    .stream
                    .clone_htod(p_host.as_slice())
                    .map_err(|e| RuvLLMError::Model(format!("htod p_f32: {e}")))?;
                let f = self
                    .module
                    .load_function("act_fused_step_f32")
                    .map_err(|e| RuvLLMError::Model(format!("load_function f32: {e}")))?;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(&p_dev)
                        .arg(&mut self.cum)
                        .arg(&mut self.not_halted)
                        .arg(&mut self.depth)
                        .arg(&mut self.w_out)
                        .arg(&n_i32)
                        .arg(&threshold)
                        .arg(&step_f32)
                        .launch(cfg)
                        .map_err(|e| RuvLLMError::Model(format!("launch f32: {e}")))?;
                }
            }

            DType::BF16 => {
                let p_host: Vec<half::bf16> = p_flat
                    .to_vec1()
                    .map_err(|e| RuvLLMError::Model(format!("to_vec1 bf16: {e}")))?;
                let p_u16: Vec<u16> = p_host.iter().map(|x| x.to_bits()).collect();
                let p_dev = self
                    .stream
                    .clone_htod(p_u16.as_slice())
                    .map_err(|e| RuvLLMError::Model(format!("htod p_bf16: {e}")))?;
                let f = self
                    .module
                    .load_function("act_fused_step_bf16")
                    .map_err(|e| RuvLLMError::Model(format!("load_function bf16: {e}")))?;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(&p_dev)
                        .arg(&mut self.cum)
                        .arg(&mut self.not_halted)
                        .arg(&mut self.depth)
                        .arg(&mut self.w_out)
                        .arg(&n_i32)
                        .arg(&threshold)
                        .arg(&step_f32)
                        .launch(cfg)
                        .map_err(|e| RuvLLMError::Model(format!("launch bf16: {e}")))?;
                }
            }

            other => {
                return Err(RuvLLMError::Model(format!(
                    "fused-act: p dtype must be F32 or BF16, got {other:?}"
                )));
            }
        }

        // D2H staging copy — w_out → Candle F32 tensor.
        let w_host = self
            .stream
            .clone_dtoh(&self.w_out)
            .map_err(|e| RuvLLMError::Model(format!("dtoh w_out: {e}")))?;
        let w_cpu = Tensor::from_slice(&w_host, (b, seq, 1), &Device::Cpu)
            .map_err(|e| RuvLLMError::Model(format!("from_slice w: {e}")))?;
        w_cpu
            .to_device(p_tensor.device())
            .map_err(|e| RuvLLMError::Model(format!("to_device w: {e}")))
    }

    /// True if all tokens have halted (`sum(not_halted) < 0.5`).
    pub fn all_halted(&self) -> Result<bool> {
        let v = self
            .stream
            .clone_dtoh(&self.not_halted)
            .map_err(|e| RuvLLMError::Model(format!("dtoh not_halted: {e}")))?;
        Ok(v.iter().sum::<f32>() < 0.5)
    }

    /// Per-token halt iteration (for depth telemetry). One D2H copy.
    pub fn depths(&self) -> Result<Vec<usize>> {
        let v = self
            .stream
            .clone_dtoh(&self.depth)
            .map_err(|e| RuvLLMError::Model(format!("dtoh depth: {e}")))?;
        Ok(v.iter().map(|&d| d as usize).collect())
    }
}

// ---------------------------------------------------------------------------
// Zero-copy path: use candle's public `Tensor::storage_and_layout()` +
// `CudaDevice::cuda_stream()` to extract raw device pointers without H2D/D2H
// staging transfers.
//
// candle 0.9 public surface used:
//   Tensor::as_cuda_device() → &CudaDevice           (device.rs:238)
//   CudaDevice::cuda_stream() → Arc<CudaStream>       (device.rs)
//   Tensor::storage_and_layout() → (Guard<Storage>, &Layout)
//   CudaStorage::as_cuda_slice::<T>() → &CudaSlice<T>
//   DevicePtr::device_ptr(&stream) → (CUdeviceptr, SyncOnDrop)
//
// No workspace patch to candle is required.  The SyncOnDrop guard MUST be
// kept alive through the kernel launch (it syncs the stream on drop, which
// ensures the kernel sees the pointer).
// ---------------------------------------------------------------------------

/// Call `f(raw_ptr_u64)` with the raw CUDA device pointer for a contiguous
/// F32 tensor, holding all lifetime guards alive for the duration of the call.
///
/// The `SyncOnDrop` guard returned by `device_ptr()` is dropped AFTER `f`
/// returns — it triggers a stream sync that serializes any downstream reads.
///
/// # Safety
/// Caller must ensure the tensor is contiguous, on CUDA, and dtype F32.
pub unsafe fn with_tensor_f32_ptr<R, F: FnOnce(u64) -> R>(
    tensor: &Tensor,
    f: F,
) -> Result<R> {
    use candle_core::Storage;
    use cudarc::driver::DevicePtr;

    let cuda_dev = tensor
        .device()
        .as_cuda_device()
        .map_err(|e| RuvLLMError::Model(format!("not CUDA: {e}")))?;
    let stream = cuda_dev.cuda_stream();

    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(ref cs) = *storage else {
        return Err(RuvLLMError::Model("tensor not on CUDA device".into()));
    };
    let slice = cs
        .as_cuda_slice::<f32>()
        .map_err(|e| RuvLLMError::Model(format!("dtype: {e}")))?;

    let offset_bytes = (layout.start_offset() * 4) as u64;
    let (base_ptr, _guard) = slice.device_ptr(&stream);
    // f is called before _guard (and storage, stream) are dropped — pointer valid.
    let result = f(base_ptr + offset_bytes);
    // _guard dropped here: syncs stream so downstream operations see the data.
    Ok(result)
}

/// Same callback pattern for BF16 tensors (pointer is `*const u16` equivalent).
pub unsafe fn with_tensor_bf16_ptr<R, F: FnOnce(u64) -> R>(
    tensor: &Tensor,
    f: F,
) -> Result<R> {
    use candle_core::Storage;
    use cudarc::driver::DevicePtr;
    use half::bf16;

    let cuda_dev = tensor
        .device()
        .as_cuda_device()
        .map_err(|e| RuvLLMError::Model(format!("not CUDA: {e}")))?;
    let stream = cuda_dev.cuda_stream();

    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(ref cs) = *storage else {
        return Err(RuvLLMError::Model("tensor not on CUDA device".into()));
    };
    let slice = cs
        .as_cuda_slice::<bf16>()
        .map_err(|e| RuvLLMError::Model(format!("dtype: {e}")))?;

    let offset_bytes = (layout.start_offset() * 2) as u64;
    let (base_ptr, _guard) = slice.device_ptr(&stream);
    let result = f(base_ptr + offset_bytes);
    Ok(result)
}

/// Zero-copy ACT kernel: `p` and `w_out` are Candle tensors on the same CUDA
/// device — no H2D/D2H staging copies. State (`cum`, `not_halted`, `depth`)
/// lives in cudarc `CudaSlice<f32>` buffers on a separate context but the
/// same physical GPU (device 0).
///
/// Returns the weight tensor `w_out` as a pre-allocated Candle tensor that the
/// caller passes directly to `h.broadcast_mul(&w)`.
///
/// # Remaining limitation
/// The ACT state buffers (`cum`, `not_halted`, `depth`) are still on a separate
/// cudarc context from candle's tensors.  A full zero-copy solution requires
/// allocating state via candle or unifying contexts — tracked in ADR-258.
pub struct FusedActZeroCopy {
    kernel: FusedActKernel,
    /// Pre-allocated Candle tensor for `w_out` on the model's CUDA device.
    w_candle: Tensor,
}

impl FusedActZeroCopy {
    /// Allocate zero-copy ACT state for `n = b * seq` positions.
    /// `device` must be a CUDA device.
    pub fn new(n: usize, device: &candle_core::Device) -> Result<Self> {
        let kernel = FusedActKernel::new(n)?;
        let w_candle = candle_core::Tensor::zeros((n,), DType::F32, device)
            .map_err(|e| RuvLLMError::Model(format!("w_candle alloc: {e}")))?;
        Ok(Self { kernel, w_candle })
    }

    /// Run one ACT step with zero-copy tensor access for `p`.
    ///
    /// `p_tensor`: `[b, seq, 1]` F32 or BF16, must be contiguous, on CUDA.
    /// Returns `w`: the pre-allocated `[n]` F32 Candle tensor (re-used each call).
    pub fn step_zero_copy(
        &mut self,
        p_tensor: &Tensor,
        threshold: f32,
        t: usize,
    ) -> Result<&Tensor> {
        let n_i32 = self.kernel.n as i32;
        let step_f32 = (t + 1) as f32;
        let cfg = LaunchConfig::for_num_elems(self.kernel.n as u32);

        // Zero-copy: get raw device pointers directly, no H2D/D2H staging.
        // The callback guards sync their respective streams on return.
        match p_tensor.dtype() {
            DType::F32 => {
                let f = self
                    .kernel
                    .module
                    .load_function("act_fused_step_f32")
                    .map_err(|e| RuvLLMError::Model(format!("load_function: {e}")))?;
                let kernel = &mut self.kernel;
                let w_candle = &self.w_candle;
                unsafe {
                    with_tensor_f32_ptr(p_tensor, |p_ptr| {
                        with_tensor_f32_ptr(w_candle, |w_ptr| {
                            kernel
                                .stream
                                .launch_builder(&f)
                                .arg(&p_ptr)
                                .arg(&mut kernel.cum)
                                .arg(&mut kernel.not_halted)
                                .arg(&mut kernel.depth)
                                .arg(&w_ptr)
                                .arg(&n_i32)
                                .arg(&threshold)
                                .arg(&step_f32)
                                .launch(cfg)
                        })
                    })
                }
                .map_err(|e| RuvLLMError::Model(format!("zero-copy launch: {e}")))?
                .map_err(|e| RuvLLMError::Model(format!("inner launch: {e}")))?
                .map_err(|e| RuvLLMError::Model(format!("kernel: {e}")))?;
            }
            other => {
                return Err(RuvLLMError::Model(format!(
                    "zero-copy ACT: dtype {other:?} not yet supported (add BF16 with_tensor_bf16_ptr)"
                )));
            }
        }
        Ok(&self.w_candle)
    }

    pub fn all_halted(&self) -> Result<bool> {
        self.kernel.all_halted()
    }
    pub fn depths(&self) -> Result<Vec<usize>> {
        self.kernel.depths()
    }
}
