//! FlashAttention-3 IO-aware tiled attention.
//!
//! Implements the FlashAttention algorithm which reduces HBM (High Bandwidth Memory)
//! reads from O(N^2 d) to O(N^2 d^2 / M) where M is SRAM size, by tiling Q, K, V
//! into blocks and fusing the softmax rescaling with the matmul accumulation.
//!
//! The key insight is that standard attention materializes the full N x N attention
//! matrix in HBM, causing O(N^2) memory. FlashAttention never materializes this
//! matrix, instead computing attention in tiles using an online softmax algorithm
//! that maintains running statistics (row-max and log-sum-exp) to avoid the
//! two-pass softmax.
//!
//! This module provides:
//! - [`FlashConfig`]: Configuration for block sizes, causal masking, and dropout
//! - [`FlashAttention3`]: IO-aware tiled forward pass returning output + LSE
//! - [`IOStats`]: Tracking of FLOPs and memory transfer for IO analysis
//! - [`RingAttention`]: Simplified ring-based distributed attention across devices

use crate::error::{AttentionError, AttentionResult};

/// Configuration for FlashAttention tiled computation.
#[derive(Clone, Debug)]
pub struct FlashConfig {
    /// Block size along the query dimension (Br).
    pub block_size_q: usize,
    /// Block size along the key/value dimension (Bc).
    pub block_size_kv: usize,
    /// Whether to apply causal masking (upper-triangular mask).
    pub causal: bool,
    /// Dropout probability (0.0 = no dropout). Applied conceptually but not
    /// stochastically in this CPU implementation.
    pub dropout_p: f32,
}

impl Default for FlashConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            causal: false,
            dropout_p: 0.0,
        }
    }
}

impl FlashConfig {
    /// Creates a config with custom block sizes.
    pub fn new(block_size_q: usize, block_size_kv: usize) -> AttentionResult<Self> {
        if block_size_q == 0 || block_size_kv == 0 {
            return Err(AttentionError::InvalidConfig(
                "Block sizes must be > 0".into(),
            ));
        }
        Ok(Self {
            block_size_q,
            block_size_kv,
            ..Default::default()
        })
    }

    /// Returns a causal variant of this config.
    pub fn with_causal(mut self) -> Self {
        self.causal = true;
        self
    }

    /// Sets the dropout probability.
    pub fn with_dropout(mut self, p: f32) -> AttentionResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(AttentionError::InvalidConfig(
                "Dropout must be in [0, 1]".into(),
            ));
        }
        self.dropout_p = p;
        Ok(self)
    }
}

/// IO statistics for comparing tiled vs naive attention.
#[derive(Clone, Debug, Default)]
pub struct IOStats {
    /// Total floating-point operations performed.
    pub total_flops: u64,
    /// Total elements read from main memory.
    pub memory_reads: u64,
    /// Total elements written to main memory.
    pub memory_writes: u64,
    /// Sequence length used for the computation.
    seq_len: usize,
    /// Head dimension used for the computation.
    head_dim: usize,
    /// Block size Q used.
    #[allow(dead_code)]
    block_size_q: usize,
    /// Block size KV used.
    #[allow(dead_code)]
    block_size_kv: usize,
}

impl IOStats {
    /// Returns the ratio of naive FLOPs to tiled FLOPs (should be ~1.0 since
    /// FLOPs are the same; the advantage is in memory IO).
    pub fn flop_ratio(&self) -> f32 {
        if self.total_flops == 0 {
            return 1.0;
        }
        // Naive attention has same FLOPs but materializes N^2 attention matrix.
        // The IO ratio compares memory transfers: naive reads/writes O(N^2 + Nd),
        // tiled reads/writes O(N^2 d / M) where M ~ block_size.
        let n = self.seq_len as f64;
        let d = self.head_dim as f64;
        let naive_io = n * n + n * d; // attention matrix + QKV
        let tiled_io = self.memory_reads as f64 + self.memory_writes as f64;
        if tiled_io < 1.0 {
            return 1.0;
        }
        (naive_io / tiled_io) as f32
    }

    /// Returns the memory complexity class as a string.
    /// Tiled: O(N) working memory. Naive: O(N^2).
    pub fn memory_complexity(&self) -> &'static str {
        "O(N)"
    }

    /// Returns the naive attention memory complexity for comparison.
    pub fn naive_memory_complexity(&self) -> &'static str {
        "O(N^2)"
    }
}

/// FlashAttention-3: IO-aware tiled attention.
///
/// Processes Q in blocks of Br rows and K/V in blocks of Bc rows, never
/// materializing the full N x N attention matrix. Uses online softmax with
/// running max and log-sum-exp to maintain numerical stability.
pub struct FlashAttention3;

/// Output of a flash attention forward pass.
#[derive(Clone, Debug)]
pub struct FlashOutput {
    /// The attention output matrix, shape [num_queries, dim].
    pub output: Vec<Vec<f32>>,
    /// Log-sum-exp per query row (m_i + ln(l_i)), used for backward pass.
    pub lse: Vec<f32>,
    /// IO statistics for this computation.
    pub stats: IOStats,
}

impl FlashAttention3 {
    /// Computes IO-aware tiled attention.
    ///
    /// # Algorithm
    ///
    /// 1. Split Q into Tr blocks of Br rows, K/V into Tc blocks of Bc rows.
    /// 2. For each Q block i, iterate over all K/V blocks j:
    ///    - Compute S_ij = Q_i @ K_j^T / sqrt(d)
    ///    - Apply causal mask if configured
    ///    - Update running max, sum-exp, and output using online softmax
    /// 3. Return output and log-sum-exp for backward pass.
    ///
    /// # Arguments
    ///
    /// * `q` - Query matrix, shape [n_q, d]
    /// * `k` - Key matrix, shape [n_kv, d]
    /// * `v` - Value matrix, shape [n_kv, d]
    /// * `config` - Flash attention configuration
    pub fn forward(
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        config: &FlashConfig,
    ) -> AttentionResult<FlashOutput> {
        if q.is_empty() {
            return Err(AttentionError::EmptyInput("queries".into()));
        }
        if k.is_empty() || v.is_empty() {
            return Err(AttentionError::EmptyInput("keys or values".into()));
        }
        if k.len() != v.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: k.len(),
                actual: v.len(),
            });
        }
        let d = q[0].len();
        if d == 0 {
            return Err(AttentionError::InvalidConfig("Dimension must be > 0".into()));
        }
        let scale = 1.0 / (d as f32).sqrt();
        let n_q = q.len();
        let n_kv = k.len();
        let br = config.block_size_q;
        let bc = config.block_size_kv;

        let mut output = vec![vec![0.0f32; d]; n_q];
        let mut lse = vec![f32::NEG_INFINITY; n_q];
        let mut row_max = vec![f32::NEG_INFINITY; n_q];
        let mut row_sum = vec![0.0f32; n_q];

        let mut stats = IOStats {
            seq_len: n_q.max(n_kv),
            head_dim: d,
            block_size_q: br,
            block_size_kv: bc,
            ..Default::default()
        };

        // Outer loop: iterate over Q blocks
        for qi_start in (0..n_q).step_by(br) {
            let qi_end = (qi_start + br).min(n_q);

            // Inner loop: iterate over K/V blocks
            for kj_start in (0..n_kv).step_by(bc) {
                let kj_end = (kj_start + bc).min(n_kv);

                // Track memory reads: Q block + K block + V block
                stats.memory_reads += ((qi_end - qi_start) * d
                    + (kj_end - kj_start) * d * 2) as u64;

                // For each query row in this Q block
                for qi in qi_start..qi_end {
                    // Compute S_ij = Q_i @ K_j^T / sqrt(d) for each key in block
                    let mut block_scores = Vec::with_capacity(kj_end - kj_start);
                    for kj in kj_start..kj_end {
                        let mut dot = 0.0f32;
                        for dd in 0..d {
                            dot += q[qi][dd] * k[kj][dd];
                        }
                        let mut score = dot * scale;

                        // Apply causal mask: mask out positions where kj > qi
                        if config.causal && kj > qi {
                            score = f32::NEG_INFINITY;
                        }
                        block_scores.push(score);
                        stats.total_flops += (2 * d) as u64; // dot product
                    }

                    // Block row-max
                    let m_ij = block_scores
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);

                    if !m_ij.is_finite() {
                        continue; // Fully masked block
                    }

                    // Exponentiate and sum
                    let exp_scores: Vec<f32> =
                        block_scores.iter().map(|&s| (s - m_ij).exp()).collect();
                    let l_ij: f32 = exp_scores
                        .iter()
                        .filter(|x| x.is_finite())
                        .sum();

                    // Online softmax rescaling
                    let m_old = row_max[qi];
                    let m_new = m_old.max(m_ij);

                    let exp_old = if m_old.is_finite() {
                        (m_old - m_new).exp()
                    } else {
                        0.0
                    };
                    let exp_new = (m_ij - m_new).exp();

                    let l_new = exp_old * row_sum[qi] + exp_new * l_ij;

                    // Rescale existing output and add new contribution
                    // O_i = (exp(m_old - m_new) * l_old * O_i
                    //      + exp(m_ij - m_new) * P_ij @ V_j) / l_new
                    if l_new > 0.0 {
                        let inv_l_new = 1.0 / l_new;
                        let scale_old = exp_old * row_sum[qi] * inv_l_new;
                        let scale_new = exp_new * inv_l_new;

                        for dd in 0..d {
                            let mut pv = 0.0f32;
                            for (local_j, kj) in (kj_start..kj_end).enumerate() {
                                if exp_scores[local_j].is_finite() {
                                    pv += exp_scores[local_j] * v[kj][dd];
                                }
                            }
                            output[qi][dd] =
                                scale_old * output[qi][dd] + scale_new * pv;
                            stats.total_flops += (2 * (kj_end - kj_start)) as u64;
                        }
                    }

                    row_max[qi] = m_new;
                    row_sum[qi] = l_new;
                }
            }

            // Track memory writes: output block
            stats.memory_writes += ((qi_end - qi_start) * d) as u64;
        }

        // Compute LSE = m + ln(l) for backward pass
        for i in 0..n_q {
            if row_sum[i] > 0.0 && row_max[i].is_finite() {
                lse[i] = row_max[i] + row_sum[i].ln();
            }
        }

        Ok(FlashOutput {
            output,
            lse,
            stats,
        })
    }
}

/// Generates a causal mask for block (qi_start..qi_end) x (kj_start..kj_end)
/// without materializing a full N x N mask.
///
/// Returns `true` for positions that should be attended to (kj <= qi).
pub fn causal_block_mask(
    qi_start: usize,
    qi_end: usize,
    kj_start: usize,
    kj_end: usize,
) -> Vec<Vec<bool>> {
    let mut mask = Vec::with_capacity(qi_end - qi_start);
    for qi in qi_start..qi_end {
        let mut row = Vec::with_capacity(kj_end - kj_start);
        for kj in kj_start..kj_end {
            row.push(kj <= qi);
        }
        mask.push(row);
    }
    mask
}

/// Simplified ring attention for distributed sequence parallelism.
///
/// In ring attention, the sequence is sharded across devices. Each device holds
/// a local Q shard and rotates K/V shards around a ring, accumulating partial
/// attention using the same online softmax as FlashAttention.
pub struct RingAttention;

/// Result from a single device in ring attention.
#[derive(Clone, Debug)]
pub struct RingDeviceOutput {
    /// Output for this device's Q shard.
    pub output: Vec<Vec<f32>>,
    /// LSE for this device's Q shard.
    pub lse: Vec<f32>,
    /// Number of simulated ring transfers.
    pub transfers: usize,
}

impl RingAttention {
    /// Runs ring attention across simulated devices.
    ///
    /// Each device holds a Q shard and processes all K/V shards by rotating
    /// them around the ring. This simulates the communication pattern of
    /// distributed ring attention.
    ///
    /// # Arguments
    ///
    /// * `q_shards` - Q shards, one per device
    /// * `k_shards` - K shards, one per device
    /// * `v_shards` - V shards, one per device
    pub fn ring_forward(
        q_shards: &[Vec<Vec<f32>>],
        k_shards: &[Vec<Vec<f32>>],
        v_shards: &[Vec<Vec<f32>>],
    ) -> AttentionResult<Vec<RingDeviceOutput>> {
        let num_devices = q_shards.len();
        if num_devices == 0 {
            return Err(AttentionError::EmptyInput("shards".into()));
        }
        if k_shards.len() != num_devices || v_shards.len() != num_devices {
            return Err(AttentionError::DimensionMismatch {
                expected: num_devices,
                actual: k_shards.len().min(v_shards.len()),
            });
        }

        let config = FlashConfig {
            block_size_q: 32,
            block_size_kv: 32,
            causal: false,
            dropout_p: 0.0,
        };

        let mut results = Vec::with_capacity(num_devices);

        // Each device processes its local Q against all K/V shards
        for device_id in 0..num_devices {
            let local_q = &q_shards[device_id];
            if local_q.is_empty() {
                return Err(AttentionError::EmptyInput(
                    format!("Q shard on device {device_id}"),
                ));
            }
            let d = local_q[0].len();
            let n_q = local_q.len();

            let mut output = vec![vec![0.0f32; d]; n_q];
            let mut row_max = vec![f32::NEG_INFINITY; n_q];
            let mut row_sum = vec![0.0f32; n_q];
            let mut lse = vec![f32::NEG_INFINITY; n_q];
            let mut transfers = 0usize;

            // Rotate through all K/V shards (ring communication)
            for step in 0..num_devices {
                let kv_idx = (device_id + step) % num_devices;
                if step > 0 {
                    transfers += 1; // Simulated device-to-device transfer
                }

                let partial = FlashAttention3::forward(
                    local_q,
                    &k_shards[kv_idx],
                    &v_shards[kv_idx],
                    &config,
                )?;

                // Merge partial results using online softmax
                for qi in 0..n_q {
                    let m_partial = if partial.lse[qi].is_finite() {
                        // Recover max from lse: we stored lse = m + ln(l),
                        // but for merging we use the partial output directly.
                        partial.lse[qi]
                    } else {
                        continue;
                    };

                    let m_old = row_max[qi];
                    let m_new = m_old.max(m_partial);

                    let exp_old = if m_old.is_finite() {
                        (m_old - m_new).exp()
                    } else {
                        0.0
                    };
                    let exp_partial = (m_partial - m_new).exp();

                    // partial.output is already normalized, so we need to
                    // un-normalize: partial_unnorm = partial.output * exp(partial.lse)
                    // For simplicity, use the sum approach:
                    let l_partial = if partial.lse[qi].is_finite() {
                        partial.lse[qi].exp()
                    } else {
                        0.0
                    };
                    let l_old = row_sum[qi];

                    let l_new = exp_old * l_old + exp_partial * l_partial;

                    if l_new > 0.0 {
                        let inv_l = 1.0 / l_new;
                        for dd in 0..d {
                            output[qi][dd] = (exp_old * l_old * output[qi][dd]
                                + exp_partial * l_partial * partial.output[qi][dd])
                                * inv_l;
                        }
                    }

                    row_max[qi] = m_new;
                    row_sum[qi] = l_new;
                }
            }

            // Final LSE
            for qi in 0..n_q {
                if row_sum[qi] > 0.0 && row_max[qi].is_finite() {
                    lse[qi] = row_max[qi] + row_sum[qi].ln();
                }
            }

            results.push(RingDeviceOutput {
                output,
                lse,
                transfers,
            });
        }

        Ok(results)
    }
}

/// Computes naive (standard) attention for correctness comparison.
/// Returns (output, attention_weights) where output is [n_q, d].
fn naive_attention(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    causal: bool,
) -> Vec<Vec<f32>> {
    let n_q = q.len();
    let n_kv = k.len();
    let d = q[0].len();
    let scale = 1.0 / (d as f32).sqrt();

    let mut output = vec![vec![0.0f32; d]; n_q];

    for qi in 0..n_q {
        // Compute scores
        let mut scores = Vec::with_capacity(n_kv);
        for kj in 0..n_kv {
            let mut dot = 0.0f32;
            for dd in 0..d {
                dot += q[qi][dd] * k[kj][dd];
            }
            let mut s = dot * scale;
            if causal && kj > qi {
                s = f32::NEG_INFINITY;
            }
            scores.push(s);
        }

        // Softmax
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_s: f32 = exp_s.iter().sum();

        // Weighted sum
        for dd in 0..d {
            let mut val = 0.0f32;
            for kj in 0..n_kv {
                val += (exp_s[kj] / sum_s) * v[kj][dd];
            }
            output[qi][dd] = val;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq(n: usize, d: usize, seed: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i as f32 + 1.0) * (j as f32 + 1.0) * seed).sin() * 0.5)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_forward_matches_naive() {
        let d = 16;
        let n = 12;
        let q = make_seq(n, d, 0.1);
        let k = make_seq(n, d, 0.2);
        let v = make_seq(n, d, 0.3);

        let config = FlashConfig::new(4, 4).unwrap();
        let flash = FlashAttention3::forward(&q, &k, &v, &config).unwrap();
        let naive = naive_attention(&q, &k, &v, false);

        for qi in 0..n {
            for dd in 0..d {
                let diff = (flash.output[qi][dd] - naive[qi][dd]).abs();
                assert!(diff < 1e-4, "row={qi} col={dd} flash={} naive={} diff={diff}",
                    flash.output[qi][dd], naive[qi][dd]);
            }
        }
    }

    #[test]
    fn test_causal_masking() {
        let d = 8;
        let n = 6;
        let q = make_seq(n, d, 0.4);
        let k = make_seq(n, d, 0.5);
        let v = make_seq(n, d, 0.6);

        let config = FlashConfig::new(2, 2).unwrap().with_causal();
        let flash = FlashAttention3::forward(&q, &k, &v, &config).unwrap();
        let naive = naive_attention(&q, &k, &v, true);

        for qi in 0..n {
            for dd in 0..d {
                let diff = (flash.output[qi][dd] - naive[qi][dd]).abs();
                assert!(diff < 1e-4, "causal row={qi} col={dd} diff={diff}");
            }
        }
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let d = 8;
        let n = 4;
        // Use large values that could cause overflow without stable softmax
        let q: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![100.0 * (i as f32 + 1.0); d])
            .collect();
        let k = q.clone();
        let v: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32; d]).collect();

        let config = FlashConfig::new(2, 2).unwrap();
        let result = FlashAttention3::forward(&q, &k, &v, &config).unwrap();

        // Output should contain finite values (no NaN/Inf)
        for row in &result.output {
            for &val in row {
                assert!(val.is_finite(), "Non-finite output: {val}");
            }
        }
        for &l in &result.lse {
            assert!(l.is_finite(), "Non-finite LSE: {l}");
        }
    }

    #[test]
    fn test_block_size_variations() {
        let d = 8;
        let n = 10;
        let q = make_seq(n, d, 0.7);
        let k = make_seq(n, d, 0.8);
        let v = make_seq(n, d, 0.9);

        let block_sizes = [(2, 2), (3, 5), (1, 1), (10, 10), (7, 3)];
        let naive = naive_attention(&q, &k, &v, false);

        for (bq, bk) in block_sizes {
            let config = FlashConfig::new(bq, bk).unwrap();
            let flash = FlashAttention3::forward(&q, &k, &v, &config).unwrap();

            for qi in 0..n {
                for dd in 0..d {
                    let diff = (flash.output[qi][dd] - naive[qi][dd]).abs();
                    assert!(
                        diff < 1e-4,
                        "blocks=({bq},{bk}) row={qi} col={dd} diff={diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_io_stats_tracking() {
        let d = 8;
        let n = 16;
        let q = make_seq(n, d, 1.0);
        let k = make_seq(n, d, 1.1);
        let v = make_seq(n, d, 1.2);

        let config = FlashConfig::new(4, 4).unwrap();
        let result = FlashAttention3::forward(&q, &k, &v, &config).unwrap();

        assert!(result.stats.total_flops > 0, "FLOPs should be tracked");
        assert!(result.stats.memory_reads > 0, "Reads should be tracked");
        assert!(result.stats.memory_writes > 0, "Writes should be tracked");
        assert_eq!(result.stats.memory_complexity(), "O(N)");
        assert_eq!(result.stats.naive_memory_complexity(), "O(N^2)");

        let ratio = result.stats.flop_ratio();
        assert!(ratio > 0.0, "IO ratio should be positive");
    }

    #[test]
    fn test_ring_attention() {
        let d = 8;
        let shard_size = 4;
        let num_devices = 3;

        let q_shards: Vec<Vec<Vec<f32>>> = (0..num_devices)
            .map(|dev| make_seq(shard_size, d, 0.1 * (dev as f32 + 1.0)))
            .collect();
        let k_shards: Vec<Vec<Vec<f32>>> = (0..num_devices)
            .map(|dev| make_seq(shard_size, d, 0.2 * (dev as f32 + 1.0)))
            .collect();
        let v_shards: Vec<Vec<Vec<f32>>> = (0..num_devices)
            .map(|dev| make_seq(shard_size, d, 0.3 * (dev as f32 + 1.0)))
            .collect();

        let results =
            RingAttention::ring_forward(&q_shards, &k_shards, &v_shards).unwrap();

        assert_eq!(results.len(), num_devices);
        for (dev_id, res) in results.iter().enumerate() {
            assert_eq!(res.output.len(), shard_size);
            assert_eq!(res.output[0].len(), d);
            // Each device except first does (num_devices - 1) transfers
            assert_eq!(res.transfers, num_devices - 1,
                "Device {dev_id} should have {} transfers", num_devices - 1);
            for row in &res.output {
                for &val in row {
                    assert!(val.is_finite(), "Device {dev_id} has non-finite output");
                }
            }
        }
    }

    #[test]
    fn test_single_block() {
        // When block size >= sequence length, should behave identically to naive
        let d = 4;
        let n = 3;
        let q = make_seq(n, d, 1.5);
        let k = make_seq(n, d, 1.6);
        let v = make_seq(n, d, 1.7);

        let config = FlashConfig::new(n, n).unwrap();
        let flash = FlashAttention3::forward(&q, &k, &v, &config).unwrap();
        let naive = naive_attention(&q, &k, &v, false);

        for qi in 0..n {
            for dd in 0..d {
                let diff = (flash.output[qi][dd] - naive[qi][dd]).abs();
                assert!(diff < 1e-5, "single block row={qi} col={dd} diff={diff}");
            }
        }
    }

    #[test]
    fn test_large_sequence() {
        let d = 16;
        let n = 128;
        let q = make_seq(n, d, 2.0);
        let k = make_seq(n, d, 2.1);
        let v = make_seq(n, d, 2.2);

        let config = FlashConfig::new(16, 16).unwrap();
        let flash = FlashAttention3::forward(&q, &k, &v, &config).unwrap();
        let naive = naive_attention(&q, &k, &v, false);

        let mut max_diff = 0.0f32;
        for qi in 0..n {
            for dd in 0..d {
                max_diff = max_diff.max((flash.output[qi][dd] - naive[qi][dd]).abs());
            }
        }
        assert!(max_diff < 1e-3, "Large seq max diff: {max_diff}");
    }

    #[test]
    fn test_lse_correctness() {
        let d = 8;
        let n = 6;
        let q = make_seq(n, d, 3.0);
        let k = make_seq(n, d, 3.1);
        let v = make_seq(n, d, 3.2);
        let scale = 1.0 / (d as f32).sqrt();

        let config = FlashConfig::new(2, 3).unwrap();
        let result = FlashAttention3::forward(&q, &k, &v, &config).unwrap();

        // Verify LSE: for each query, compute log(sum(exp(scores))) manually
        for qi in 0..n {
            let mut scores = Vec::with_capacity(n);
            for kj in 0..n {
                let dot: f32 = (0..d).map(|dd| q[qi][dd] * k[kj][dd]).sum();
                scores.push(dot * scale);
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
            let expected_lse = max_s + sum_exp.ln();

            let diff = (result.lse[qi] - expected_lse).abs();
            assert!(diff < 1e-3, "LSE row={qi} flash={} expected={expected_lse} diff={diff}",
                result.lse[qi]);
        }
    }

    #[test]
    fn test_causal_block_mask_utility() {
        let mask = causal_block_mask(2, 5, 0, 4);
        // qi=2: kj 0,1,2 allowed, 3 not
        assert_eq!(mask[0], vec![true, true, true, false]);
        // qi=3: kj 0,1,2,3 allowed
        assert_eq!(mask[1], vec![true, true, true, true]);
        // qi=4: all allowed
        assert_eq!(mask[2], vec![true, true, true, true]);
    }

    #[test]
    fn test_empty_input_errors() {
        let config = FlashConfig::default();
        let empty: Vec<Vec<f32>> = vec![];
        let q = vec![vec![1.0; 4]];

        assert!(FlashAttention3::forward(&empty, &q, &q, &config).is_err());
        assert!(FlashAttention3::forward(&q, &empty, &q, &config).is_err());
        assert!(FlashAttention3::forward(&q, &q, &empty, &config).is_err());
    }

    #[test]
    fn test_config_validation() {
        assert!(FlashConfig::new(0, 4).is_err());
        assert!(FlashConfig::new(4, 0).is_err());
        assert!(FlashConfig::new(4, 4).is_ok());

        assert!(FlashConfig::default().with_dropout(1.5).is_err());
        assert!(FlashConfig::default().with_dropout(-0.1).is_err());
        assert!(FlashConfig::default().with_dropout(0.5).is_ok());
    }
}
