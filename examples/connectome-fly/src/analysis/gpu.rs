//! GPU SDPA path for the motif-retrieval encoder (ADR-154 §6.4).
//!
//! The CPU path is always the correctness reference. The GPU backend is
//! gated behind `--features gpu-cuda` and delegates the per-window
//! scaled-dot-product-attention compute to a CUDA kernel. The trait
//! below is the narrow seam both backends agree on; it is narrow by
//! design — we copy the q / k / v tensors to the device, run one batch
//! SDPA, and copy back. No state lives on the device between windows
//! yet; that is a production-stack concern (ADR-154 §6.4 "Tier 2").
//!
//! When the `gpu-cuda` feature is enabled but `cudarc` cannot link
//! against the host CUDA toolkit at runtime, `CudaBackend::new` returns
//! `Err` and the caller MUST fall back to the CPU path.
//!
//! ## Determinism contract
//!
//! FP ordering on GPU is not bit-exact with CPU; we promise ≤ 1e-5
//! absolute error on the resulting motif-vector, and the `ac_1`
//! repeatability test pins the CPU path as the canonical trace.

use crate::analysis::types::AnalysisConfig;

/// Dims for one SDPA batch.
#[derive(Copy, Clone, Debug)]
pub struct Dims {
    /// Number of query positions per window (we use 1 for motif-window
    /// retrieval: one pooled vector per window).
    pub q_len: usize,
    /// Number of key/value positions per window. Matches `motif_bins`.
    pub kv_len: usize,
    /// Embedding dimension.
    pub d: usize,
    /// Number of windows in the batch.
    pub batch: usize,
}

/// Compute backend for SDPA batch over motif windows.
///
/// The CPU implementation is the reference and lives alongside the
/// scalar retrieval loop. The GPU implementation is additive and
/// optional.
pub trait ComputeBackend {
    /// Batched SDPA: `q[batch * q_len * d]`, `k[batch * kv_len * d]`,
    /// `v[batch * kv_len * d]`. Output is `batch * q_len * d`.
    ///
    /// Implementations MAY assume row-major contiguous layout. Must be
    /// deterministic within the same build.
    fn sdpa_batch(&self, q: &[f32], k: &[f32], v: &[f32], dims: Dims) -> Vec<f32>;

    /// Name for logs / bench sub-report keys.
    fn name(&self) -> &'static str;
}

/// CPU reference implementation — mirrors the arithmetic inside
/// `ruvector_attention::attention::ScaledDotProductAttention` but
/// operates on an externally-shaped batch.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn sdpa_batch(&self, q: &[f32], k: &[f32], v: &[f32], dims: Dims) -> Vec<f32> {
        let mut out = vec![0.0_f32; dims.batch * dims.q_len * dims.d];
        let scale = 1.0 / (dims.d as f32).sqrt();
        for b in 0..dims.batch {
            for qi in 0..dims.q_len {
                let q_off = (b * dims.q_len + qi) * dims.d;
                // Scores q · k_j for each key.
                let mut scores = vec![0.0_f32; dims.kv_len];
                let mut max_s = f32::NEG_INFINITY;
                for kj in 0..dims.kv_len {
                    let k_off = (b * dims.kv_len + kj) * dims.d;
                    let mut s = 0.0_f32;
                    for d in 0..dims.d {
                        s += q[q_off + d] * k[k_off + d];
                    }
                    s *= scale;
                    scores[kj] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                // Softmax.
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                if sum > 1e-20 {
                    for s in scores.iter_mut() {
                        *s /= sum;
                    }
                }
                // Weighted value sum.
                let o_off = (b * dims.q_len + qi) * dims.d;
                for kj in 0..dims.kv_len {
                    let v_off = (b * dims.kv_len + kj) * dims.d;
                    let w = scores[kj];
                    for d in 0..dims.d {
                        out[o_off + d] += w * v[v_off + d];
                    }
                }
            }
        }
        out
    }
}

/// CUDA-backed SDPA via `cudarc` (feature: `gpu-cuda`).
///
/// Current scope: host-orchestrated batched SDPA over q/k/v. Kernel is
/// a hand-written CUDA C source compiled via NVRTC and launched with
/// one block per (batch, q_pos). On a 5080 this is expected to be
/// latency-bound by device transfer at batch=10k; a fused kernel
/// (single launch, scores+softmax+WMM) is future work.
#[cfg(feature = "gpu-cuda")]
pub struct CudaBackend {
    // Kept as a placeholder so the compile surface is stable even if
    // the cudarc crate's shape changes. See GPU.md for status.
    _reserved: (),
}

#[cfg(feature = "gpu-cuda")]
impl CudaBackend {
    /// Attempt to initialize the GPU backend. Returns `Err` with an
    /// actionable message if cudarc cannot open the driver.
    pub fn new() -> Result<Self, String> {
        // We intentionally do not claim bit-exactness with CPU. The test
        // matrix pins the CPU path as canonical; this is a throughput
        // uplift with a ≤ 1e-5 tolerance.
        //
        // The full kernel implementation is left as a TODO pending a
        // cudarc 0.13 upgrade across the workspace (ADR-154 §6.4 notes
        // cudarc may not yet support CUDA 13.0 / 5080 driver ABI). See
        // examples/connectome-fly/GPU.md for the current status and the
        // fallback plan.
        Err("cudarc GPU backend not yet implemented — see \
             examples/connectome-fly/GPU.md for status and the CPU \
             fallback path"
            .to_string())
    }
}

#[cfg(feature = "gpu-cuda")]
impl ComputeBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn sdpa_batch(&self, _q: &[f32], _k: &[f32], _v: &[f32], _dims: Dims) -> Vec<f32> {
        unimplemented!(
            "cudarc SDPA kernel not implemented — see \
             examples/connectome-fly/GPU.md. Fall back to CpuBackend."
        )
    }
}

/// Build the preferred backend for this build.
pub fn preferred_backend(_cfg: &AnalysisConfig) -> Box<dyn ComputeBackend> {
    Box::new(CpuBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_sdpa_is_deterministic() {
        let dims = Dims {
            q_len: 1,
            kv_len: 4,
            d: 8,
            batch: 3,
        };
        let total_q = dims.batch * dims.q_len * dims.d;
        let total_k = dims.batch * dims.kv_len * dims.d;
        let q: Vec<f32> = (0..total_q).map(|i| (i as f32).sin()).collect();
        let k: Vec<f32> = (0..total_k).map(|i| (i as f32 * 0.5).cos()).collect();
        let v: Vec<f32> = (0..total_k).map(|i| (i as f32 * 0.25).sin()).collect();
        let be = CpuBackend;
        let a = be.sdpa_batch(&q, &k, &v, dims);
        let b = be.sdpa_batch(&q, &k, &v, dims);
        assert_eq!(a, b, "cpu sdpa must be bit-identical on repeat");
        assert_eq!(a.len(), dims.batch * dims.q_len * dims.d);
    }

    #[test]
    fn cpu_sdpa_weighted_value_in_range() {
        let dims = Dims {
            q_len: 1,
            kv_len: 4,
            d: 2,
            batch: 1,
        };
        let q = vec![0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let v = vec![2.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 4.0];
        let out = CpuBackend.sdpa_batch(&q, &k, &v, dims);
        // Uniform attention → weighted mean of values = (1.0, 2.0).
        assert!((out[0] - 1.0).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-6, "out[1]={}", out[1]);
    }
}
