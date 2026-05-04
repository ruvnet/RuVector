//! Pure-Rust CPU fallback for sentence-transformers/all-MiniLM-L6-v2
//! (iter 133, ADR-167 path C).
//!
//! Runs real BERT-6 inference on the host CPU (Cortex-A76 NEON on the
//! Pi 5, AVX2 on x86 dev hosts) via candle-transformers. The Hailo NPU
//! stays idle — this is a fallback, not the primary path. Use when
//! the operator has the model weights but not (yet) a compiled HEF.
//!
//! # Artifacts expected in `model_dir`
//!
//! ```text
//!   model_dir/
//!     model.safetensors       # ~90 MB BERT-6 weights from HF
//!     tokenizer.json          # HF tokenizers JSON (not the WordPiece text vocab)
//!     config.json             # BERT config — hidden_size, layers, heads, etc.
//! ```
//!
//! These are the standard HuggingFace artifacts for
//! `sentence-transformers/all-MiniLM-L6-v2`. No HEF / Hailo Dataflow
//! Compiler dependency.
//!
//! # Realistic latency (measured iter 149 on real hardware)
//!
//! AMD Ryzen 9 9950X (AVX2/AVX-512 x86_64), 128-token sequence:
//!   * cold first embed:  ~45 ms (model warm-up + JIT)
//!   * warm steady-state: ~38-40 ms
//!   * sustained throughput, pool=1: 25.7 embeds/sec
//!   * sustained throughput, pool=4: **45.0 embeds/sec** (1.75×)
//!
//! Pi 5 (Cortex-A76 @ 2.4 GHz, 4 cores), measured against deployed
//! aarch64 release build:
//!   * cold first embed:  ~510 ms (model load + JIT)
//!   * warm steady-state: ~570 ms p50
//!   * sustained throughput, pool=4: **7.0 embeds/sec** at 4 concurrent clients
//!     (latency p50=572ms, p99=813ms — A76 cores split between 4 parallel
//!     forwards; memory bandwidth limits keep us at ~70% of theoretical)
//!
//! Slow vs Hailo's 1-3 ms NPU target, but real semantic vectors today.
//! Scale horizontally with workers — a 4-worker Pi cluster reaches
//! **~28 embeds/sec aggregate** (each contributes 7/sec via pool=4),
//! which covers most ingest workloads.

#![cfg(feature = "cpu-fallback")]

use crate::error::HailoError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// CPU-side BERT-6 embedder. Held in `HailoEmbedder` as a fallback
/// when no HEF is loaded.
///
/// **Iter 147 — parallel inference pool.** Holds N independent
/// `Inner` instances behind separate `Mutex` locks. The `embed()` call
/// round-robins through them with `try_lock`, falling back to the
/// designated next slot if all are busy. This unblocks N concurrent
/// embed() callers from queueing on a single mutex (which capped
/// throughput at ~25 embeds/sec on x86 release, regardless of how
/// many cluster threads were dispatching).
///
/// Memory cost: N safetensors mmaps that the OS dedupes when they
/// open the same file, so the 90 MB weight blob is shared. Each
/// `Inner` only adds the candle `BertModel` graph structure (~few
/// hundred KB) so 4 instances ≈ 100 MB resident vs 90 MB for 1.
///
/// Pool size is set via `RUVECTOR_CPU_FALLBACK_POOL_SIZE` (default 1
/// for backward compat; set to 4 on a Pi 5 for ~4× throughput).
pub struct CpuEmbedder {
    /// One Mutex per pool slot. `Vec<Mutex<Inner>>` (not
    /// `Mutex<Vec<Inner>>`) so try_lock on individual slots doesn't
    /// serialize through a single outer lock.
    pool: Vec<Mutex<Inner>>,
    /// Round-robin index for fair dispatch + bounded blocking when
    /// all try_locks fail. Incremented on every embed() call.
    next_slot: AtomicUsize,
    output_dim: usize,
    /// 128-token sequence cap matches all-MiniLM-L6-v2's training-time
    /// max. Raising this breaks RoPE/positional baked into the weights.
    max_seq: usize,
}

struct Inner {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CpuEmbedder {
    /// Load the model with the default pool size (1, or whatever
    /// `RUVECTOR_CPU_FALLBACK_POOL_SIZE` is set to).
    pub fn open(model_dir: &Path) -> Result<Self, HailoError> {
        let n = std::env::var("RUVECTOR_CPU_FALLBACK_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1);
        Self::open_with_pool(model_dir, n)
    }

    /// Load the model from `model_dir` with `pool_size` parallel
    /// inference slots. Errors if the three required files
    /// (model.safetensors, tokenizer.json, config.json) aren't all
    /// present + parseable, or if `pool_size == 0`.
    pub fn open_with_pool(model_dir: &Path, pool_size: usize) -> Result<Self, HailoError> {
        if pool_size == 0 {
            return Err(HailoError::Tokenizer("pool_size must be >= 1".to_string()));
        }

        let weights_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");

        if !weights_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "model.safetensors",
            });
        }
        if !tokenizer_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "tokenizer.json",
            });
        }
        if !config_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "config.json",
            });
        }

        // CPU device — NPU is dormant in this fallback. Future iter
        // could pick up GPU via candle's CUDA/Metal backends, but the
        // Pi 5 + AI HAT+ deploy doesn't have one.
        let device = Device::Cpu;

        // BERT config drives the model topology. all-MiniLM-L6-v2
        // ships hidden_size=384, num_hidden_layers=6, num_heads=12.
        //
        // Iter 213 — cap at 64 KB (same as host_embeddings; legit BERT
        // config is <1 KB). Operator-controlled path; misconfig
        // protection against an accidental large-file pointer.
        const CONFIG_CAP: u64 = 64 * 1024;
        if let Ok(meta) = std::fs::metadata(&config_path) {
            if meta.len() > CONFIG_CAP {
                return Err(HailoError::Tokenizer(format!(
                    "config.json at {} is {} bytes, exceeds {} byte cap \
                     (iter 213 — likely a misconfig pointing at the wrong file)",
                    config_path.display(),
                    meta.len(),
                    CONFIG_CAP
                )));
            }
        }
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| HailoError::Tokenizer(format!("read config.json: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| HailoError::Tokenizer(format!("parse config.json: {}", e)))?;
        let output_dim = config.hidden_size;

        // Load N independent BertModel instances. Each calls
        // `from_mmaped_safetensors` against the same file; OS-level
        // mmap dedupes the 90 MB weight blob into shared physical
        // pages, so the per-slot memory cost is just the candle
        // BertModel graph structure (~few hundred KB) — see iter-147
        // commit message for the empirical breakdown.
        let mut pool: Vec<Mutex<Inner>> = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                    .map_err(|e| HailoError::Tokenizer(format!("load safetensors: {}", e)))?
            };
            let model = BertModel::load(vb, &config)
                .map_err(|e| HailoError::Tokenizer(format!("BertModel::load: {}", e)))?;
            // Tokenizer per slot — they're cheap (~few MB) and using
            // one per slot avoids any internal mutability gotchas.
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| HailoError::Tokenizer(format!("Tokenizer::from_file: {}", e)))?;
            pool.push(Mutex::new(Inner {
                model,
                tokenizer,
                device: device.clone(),
            }));
        }
        // Suppress unused-import warning when PathBuf isn't used in this build
        let _ = std::mem::size_of::<PathBuf>();

        Ok(Self {
            pool,
            next_slot: AtomicUsize::new(0),
            output_dim,
            max_seq: 128,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Pool size — number of parallel inference slots. Iter 147.
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Acquire an inference slot. Iter 147 dispatch:
    ///   1. round-robin pick a starting slot via the AtomicUsize
    ///   2. try_lock that slot first (best case: parallel work)
    ///   3. fall through with try_lock on every other slot
    ///   4. if all are busy, block on the originally-picked slot
    ///      (bounded wait, fair-ish under load)
    fn acquire_slot(&self) -> std::sync::MutexGuard<'_, Inner> {
        let n = self.pool.len();
        let start = self.next_slot.fetch_add(1, Ordering::Relaxed) % n;
        for i in 0..n {
            let idx = (start + i) % n;
            if let Ok(g) = self.pool[idx].try_lock() {
                return g;
            }
        }
        self.pool[start].lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Embed `text` into a unit-norm `output_dim`-length f32 vector.
    /// Mean-pools the BERT-6 output across the masked sequence and
    /// L2-normalises — matches what `sentence-transformers/all-MiniLM-L6-v2`
    /// produces from its native Python pipeline.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, HailoError> {
        let mut g = self.acquire_slot();
        let inner = &mut *g;

        let mut encoding = inner
            .tokenizer
            .encode(text, true)
            .map_err(|e| HailoError::Tokenizer(format!("encode: {}", e)))?;
        encoding.truncate(self.max_seq, 1, tokenizers::TruncationDirection::Right);

        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let token_t = Tensor::new(token_ids.as_slice(), &inner.device)
            .map_err(|e| HailoError::Tokenizer(format!("token tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| HailoError::Tokenizer(format!("token unsqueeze: {}", e)))?;
        let token_type = Tensor::zeros((1, token_ids.len()), DType::I64, &inner.device)
            .map_err(|e| HailoError::Tokenizer(format!("token type: {}", e)))?;
        let attention_t = Tensor::new(attention.as_slice(), &inner.device)
            .map_err(|e| HailoError::Tokenizer(format!("attention tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| HailoError::Tokenizer(format!("attention unsqueeze: {}", e)))?;

        // Forward pass — returns (1, seq_len, hidden_size).
        let output = inner
            .model
            .forward(&token_t, &token_type, Some(&attention_t))
            .map_err(|e| HailoError::Tokenizer(format!("BertModel::forward: {}", e)))?;

        // Mean-pool over the sequence dim, weighted by the attention
        // mask. Standard sentence-transformers operation.
        let attention_f = attention_t
            .to_dtype(DType::F32)
            .map_err(|e| HailoError::Tokenizer(format!("mask f32: {}", e)))?;
        let mask = attention_f
            .unsqueeze(2)
            .map_err(|e| HailoError::Tokenizer(format!("mask unsqueeze: {}", e)))?
            .broadcast_as(output.shape())
            .map_err(|e| HailoError::Tokenizer(format!("mask broadcast: {}", e)))?;
        let masked = output
            .broadcast_mul(&mask)
            .map_err(|e| HailoError::Tokenizer(format!("masked mul: {}", e)))?;
        let summed = masked
            .sum(1)
            .map_err(|e| HailoError::Tokenizer(format!("sum: {}", e)))?;
        let denom = mask
            .sum(1)
            .map_err(|e| HailoError::Tokenizer(format!("denom sum: {}", e)))?;
        let pooled = summed
            .broadcast_div(&denom)
            .map_err(|e| HailoError::Tokenizer(format!("div: {}", e)))?;

        // Squeeze batch + read out as Vec<f32>.
        let v: Vec<f32> = pooled
            .squeeze(0)
            .map_err(|e| HailoError::Tokenizer(format!("squeeze: {}", e)))?
            .to_vec1()
            .map_err(|e| HailoError::Tokenizer(format!("to_vec1: {}", e)))?;

        // L2-normalise — matches sentence-transformers convention.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v = if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        };
        Ok(v)
    }
}
