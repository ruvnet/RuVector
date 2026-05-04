//! Multi-pipeline NPU embedder pool — iter 234.
//!
//! ADR-176 P5 (queued post-iter-227 baseline). The single-pipeline
//! [`crate::hef_embedder::HefEmbedder`] caps cluster throughput at
//! ~70 RPS (cognitum-v0, all-MiniLM-L6-v2, batch=1) because every
//! gRPC request serializes on a single `Mutex<Inner>` covering one
//! input-vstream / output-vstream pair. The Hailo-8 NPU plus its PCIe
//! DMA path can overlap meaningfully — a single inference is ~14 ms
//! steady-state but only ~2 ms of that is NPU compute; the remainder
//! is PCIe write + read. Multi-pipeline overlap should unlock a
//! 2-4× throughput multiplier.
//!
//! # Baseline measurement (iter 227, single pipeline)
//!
//! `ruvector-hailo-cluster-bench --workers 127.0.0.1:50051`:
//!
//! | concurrency | throughput | p50    | p99    |
//! |-------------|------------|--------|--------|
//! | 1           | 70.6 RPS   | 14.1ms | 15.8ms |
//! | 4           | 70.7 RPS   | 56.7ms | 74.7ms |
//! | 8           | 70.7 RPS   | 112.7ms| 170.7ms|
//!
//! Throughput plateaus regardless of concurrency; p50 scales linearly
//! with concurrency confirming the lock is the bottleneck.
//!
//! # Design (iter 234 — skeleton; iter 235 will bench + tune)
//!
//! Mirrors the [`crate::cpu_embedder::CpuEmbedder`] pool layout:
//! `Vec<Mutex<Slot>>`. `embed()` `try_lock`s slots in order; the first
//! free one wins. If all are busy, falls back to blocking on slot 0
//! (matches CpuEmbedder semantics — cheap fallback rather than a
//! channel/queue).
//!
//! Each slot owns its own `HefPipeline` (= its own network_group +
//! vstream pair on the *shared* vdevice). HailoRT's network-group
//! scheduler handles the actual NPU arbitration — multiple network
//! groups on the same vdevice are pipelined by the scheduler so PCIe
//! DMA can overlap with NPU compute.
//!
//! Tokenizer + HostEmbeddings are also per-slot (cheap clone /
//! per-slot mmap view) so CPU preprocessing doesn't reserialize at
//! the pool boundary.
//!
//! # Why a separate type, not `HefEmbedder { pool_size: usize }`
//!
//! The single-pipeline path stays cheaper for low-concurrency deploys
//! (init time, RAM footprint, no scheduler overhead). Operators who
//! know they'll see low load — solo Pi running mmwave-bridge only,
//! say — keep `HefEmbedder`. Cluster workers handling many concurrent
//! gRPC streams switch to `HefEmbedderPool`.

#![cfg(all(feature = "hailo", feature = "cpu-fallback"))]

use crate::device::HailoDevice;
use crate::error::HailoError;
use crate::hef_embedder::HefEmbedder;
use std::path::Path;
use std::sync::Mutex;

/// Default pool size. 4 pipelines is the sweet spot in the SOTA
/// design space: enough to overlap PCIe write / NPU compute / PCIe
/// read / host pre-post-processing without exhausting NPU scheduler
/// resources on the Hailo-8. Iter-235 will sweep 1/2/4/8 to confirm.
pub const DEFAULT_POOL_SIZE: usize = 4;

/// N independent NPU pipelines fronted by a Vec<Mutex> with try_lock
/// dispatch. See module-level docs for design rationale.
pub struct HefEmbedderPool {
    /// Per-slot embedders. Each is a fully isolated HefPipeline +
    /// tokenizer + HostEmbeddings on the shared vdevice. Wrapping
    /// each in its own `Mutex` (rather than `Mutex<Vec<HefEmbedder>>`)
    /// lets `try_lock` discriminate slot-by-slot without contention
    /// on a parent lock.
    slots: Vec<Mutex<HefEmbedder>>,
    /// Mirrors the inner embedders' value so callers don't need to
    /// pop a slot just to read the dim.
    output_dim: usize,
    /// Same for max_seq.
    max_seq: usize,
}

impl HefEmbedderPool {
    /// Open `pool_size` independent pipelines on the same vdevice.
    /// Each slot performs a full `HefPipeline::open` so every slot
    /// gets its own configured network_group + vstream pair. The
    /// vdevice handle behind `device` is shared (HailoRT documents
    /// it as thread-safe across multiple network groups).
    ///
    /// Errors during slot N propagate immediately; partially-built
    /// slots already in `slots` Drop in reverse order, releasing
    /// vstreams before the network group as required by HailoRT.
    pub fn open(
        device: &HailoDevice,
        model_dir: &Path,
        pool_size: usize,
    ) -> Result<Self, HailoError> {
        if pool_size == 0 {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "pool_size must be >= 1",
            });
        }

        let mut slots = Vec::with_capacity(pool_size);
        let mut output_dim = 0usize;
        let mut max_seq = 0usize;
        for i in 0..pool_size {
            // Earlier slots already pushed will Drop in reverse on
            // the early return, releasing their vstreams in HailoRT's
            // expected order. `i` is captured in BadModelDir's path
            // for diagnostic clarity if a later slot fails to open.
            let emb = HefEmbedder::open(device, model_dir).map_err(|e| match e {
                HailoError::BadModelDir { path, what } => HailoError::BadModelDir {
                    path: format!("{} (slot {})", path, i),
                    what,
                },
                other => other,
            })?;
            if i == 0 {
                output_dim = emb.output_dim();
                max_seq = emb.max_seq();
            }
            slots.push(Mutex::new(emb));
        }
        Ok(Self {
            slots,
            output_dim,
            max_seq,
        })
    }

    /// Output embedding dimensionality (384 for all-MiniLM-L6-v2).
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// HEF compile-time max sequence length (128 for the iter-156b HEF).
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Number of independent pipelines this pool owns.
    pub fn pool_size(&self) -> usize {
        self.slots.len()
    }

    /// Embed `text` on the first available pipeline. Mirrors
    /// `HefEmbedder::embed`'s output contract bit-for-bit since each
    /// slot uses identical HEF + tokenizer + embedding tables; the
    /// only thing that varies is which physical pipeline carried the
    /// inference.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, HailoError> {
        // First pass — try every slot's lock without blocking. If
        // any are free, win the cheap path. Maps to ~99% of requests
        // when pool_size > peak_concurrency.
        for slot in &self.slots {
            if let Ok(g) = slot.try_lock() {
                return g.embed(text);
            }
        }
        // Fallback: contention exceeds pool size. Block on slot 0
        // (matching cpu_embedder.rs's pattern). The blocking caller
        // pays a queue cost but doesn't lose throughput — slot 0 is
        // about to be free and the next inference is already in
        // flight on another slot.
        let g = self.slots[0]
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        g.embed(text)
    }
}

// SAFETY: each slot's Mutex<HefEmbedder> already enforces serialization
// inside the slot. The Vec<Mutex<...>> pattern matches cpu_embedder.rs
// which already carries the same Send+Sync invariants.
unsafe impl Send for HefEmbedderPool {}
unsafe impl Sync for HefEmbedderPool {}

#[cfg(test)]
mod tests {
    // Deliberate stub — full integration tests need a real HEF + vdevice
    // and live in deploy-side smoke tests on cognitum-v0. Iter-235 will
    // add cluster-bench measurements to confirm the throughput unlock.
    //
    // Compile-only test: ensure the type satisfies Send + Sync so the
    // worker can hand out `Arc<HefEmbedderPool>` across tokio tasks.
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn pool_is_send_sync() {
        assert_send_sync::<super::HefEmbedderPool>();
    }
}
