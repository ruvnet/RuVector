# ADR-003: Flash Attention Implementation

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-12 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-002 (RuvLLM Integration), ADR-015 (Coherence-Gated Transformer) |

## 1. Context

### 1.1 The Attention Memory Problem

Standard attention computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

For sequence length N and head dimension d:

| Operation | Memory | Compute |
|-----------|--------|---------|
| QK^T | O(N^2) | O(N^2 * d) |
| softmax | O(N^2) | O(N^2) |
| Softmax * V | O(N * d) | O(N^2 * d) |
| **Total** | **O(N^2)** | **O(N^2 * d)** |

For a 128K context (common in modern LLMs):
- Attention matrix: 128K x 128K x 4 bytes = **65 GB per head per layer**
- A 32-head, 80-layer model would need **166 TB** for attention alone

### 1.2 Memory Hierarchy Reality

| Memory Type | Capacity | Bandwidth | Latency |
|-------------|----------|-----------|---------|
| Registers | ~1 KB | infinite | 1 cycle |
| L1 Cache | 64-128 KB | ~2 TB/s | ~4 cycles |
| L2 Cache | 256 KB - 1 MB | ~1 TB/s | ~12 cycles |
| L3/LLC | 8-64 MB | ~500 GB/s | ~40 cycles |
| HBM/RAM | 16-192 GB | ~2 TB/s (HBM3) | ~100+ cycles |

The quadratic attention matrix forces constant HBM traffic, wasting 90%+ of compute waiting for memory.

### 1.3 Flash Attention Insight

Flash Attention (Dao et al., 2022) observes that:
1. Softmax can be computed **incrementally** via the log-sum-exp trick
2. We never need to materialize the full N x N attention matrix
3. By tiling Q, K, V into blocks that fit in SRAM, we eliminate HBM round-trips

## 2. Decision

### 2.1 Implement Tiled Flash Attention

We implement Flash Attention using a **block-sparse tiled algorithm** that:

1. Divides Q, K, V into blocks of size B_r x B_c
2. Computes attention within SRAM (no HBM writes for intermediate matrices)
3. Uses online softmax to accumulate results incrementally

### 2.2 Algorithm Overview

```
Algorithm: FlashAttention (Forward Pass)

Input: Q, K, V in HBM (N x d), block sizes B_r, B_c
Output: O in HBM (N x d)

1. Initialize O = 0, l = 0, m = -inf in HBM (running softmax state)

2. For each block K_j, V_j (j = 1 to ceil(N/B_c)):
   Load K_j, V_j from HBM to SRAM

   For each block Q_i (i = 1 to ceil(N/B_r)):
       Load Q_i, O_i, l_i, m_i from HBM to SRAM

       # Compute block attention
       S_ij = Q_i * K_j^T / sqrt(d)     # B_r x B_c in SRAM
       m_ij = rowmax(S_ij)               # B_r in SRAM
       P_ij = exp(S_ij - m_ij)           # B_r x B_c in SRAM
       l_ij = rowsum(P_ij)               # B_r in SRAM

       # Update running softmax
       m_new = max(m_i, m_ij)
       l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij

       # Update output (online softmax rescaling)
       O_i = (l_i * exp(m_i - m_new) * O_i + exp(m_ij - m_new) * P_ij * V_j) / l_new

       Write O_i, l_new, m_new back to HBM

3. Return O
```

### 2.3 Memory Analysis

| Standard Attention | Flash Attention |
|-------------------|-----------------|
| O(N^2) for attention matrix | O(N) for output only |
| O(N^2) HBM reads/writes | O(N^2 / M) HBM accesses (M = SRAM size) |

For 128K sequence with 1 MB SRAM:
- Standard: 65 GB memory, ~65 TB HBM I/O
- Flash: ~2 GB memory (Q+K+V+O), ~200 GB HBM I/O

**Result: 300x reduction in memory I/O**

### 2.4 Block Size Selection

Optimal block sizes depend on hardware SRAM:

| Hardware | SRAM | B_r | B_c | Tiles/SM |
|----------|------|-----|-----|----------|
| A100 | 192 KB | 128 | 64 | 4 |
| H100 | 256 KB | 128 | 128 | 8 |
| Apple M4 (GPU) | 96 KB | 64 | 64 | 2 |
| CPU (L2) | 256 KB | 256 | 128 | - |

Block size formula:
```
B_r * B_c * sizeof(f32) + B_r * d * sizeof(f32) + B_c * d * sizeof(f32) <= SRAM
```

## 3. Implementation

### 3.1 Rust Implementation (CPU)

```rust
/// Flash Attention forward pass (CPU implementation)
pub fn flash_attention(
    q: &Tensor,  // [batch, heads, seq_q, dim]
    k: &Tensor,  // [batch, heads, seq_k, dim]
    v: &Tensor,  // [batch, heads, seq_k, dim]
    scale: f32,
    block_size: usize,
) -> Tensor {
    let (batch, heads, seq_q, dim) = q.shape();
    let seq_k = k.shape().2;

    let mut output = Tensor::zeros(&[batch, heads, seq_q, dim]);
    let mut lse = Tensor::full(&[batch, heads, seq_q], f32::NEG_INFINITY);

    let num_blocks_k = (seq_k + block_size - 1) / block_size;
    let num_blocks_q = (seq_q + block_size - 1) / block_size;

    for b in 0..batch {
        for h in 0..heads {
            for j in 0..num_blocks_k {
                let k_start = j * block_size;
                let k_end = (k_start + block_size).min(seq_k);

                // Load K_j, V_j blocks
                let k_block = k.slice(b, h, k_start..k_end);
                let v_block = v.slice(b, h, k_start..k_end);

                for i in 0..num_blocks_q {
                    let q_start = i * block_size;
                    let q_end = (q_start + block_size).min(seq_q);

                    // Load Q_i block and current output state
                    let q_block = q.slice(b, h, q_start..q_end);
                    let o_block = output.slice_mut(b, h, q_start..q_end);
                    let lse_block = lse.slice_mut(b, h, q_start..q_end);

                    // Compute S = Q_i @ K_j^T * scale
                    let s = matmul(&q_block, &k_block.transpose()) * scale;

                    // Online softmax update
                    let m_new = s.max(axis: -1);  // [block_size]
                    let p = (s - m_new.unsqueeze(-1)).exp();
                    let l_new = p.sum(axis: -1);

                    // Rescale existing output
                    let m_old = lse_block.clone();
                    let m_max = m_old.maximum(&m_new);
                    let scale_old = (m_old - &m_max).exp();
                    let scale_new = (m_new - &m_max).exp();

                    let l_old = lse_block.exp();
                    let l_combined = l_old * &scale_old + l_new * &scale_new;

                    // Update output: O = (O * scale_old + P @ V * scale_new) / l_combined
                    let pv = matmul(&p, &v_block);
                    *o_block = (o_block.clone() * scale_old.unsqueeze(-1)
                              + pv * scale_new.unsqueeze(-1))
                              / l_combined.unsqueeze(-1);

                    // Update log-sum-exp
                    *lse_block = m_max + l_combined.ln();
                }
            }
        }
    }

    output
}
```

### 3.2 SIMD-Optimized Block Operations

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn block_attention_neon(
    q: &[f32],      // [B_r, d]
    k: &[f32],      // [B_c, d]
    v: &[f32],      // [B_c, d]
    output: &mut [f32],  // [B_r, d]
    lse: &mut [f32],     // [B_r]
    br: usize,
    bc: usize,
    d: usize,
    scale: f32,
) {
    // Compute S = Q @ K^T using NEON GEMM
    let mut s = vec![0.0f32; br * bc];
    neon_gemm(q, k, &mut s, br, bc, d, scale);

    // Row-wise online softmax
    for i in 0..br {
        let row_start = i * bc;
        let row = &s[row_start..row_start + bc];

        // Find max (for numerical stability)
        let m_new = neon_max(row);

        // Compute exp(s - m) and sum
        let mut p = vec![0.0f32; bc];
        let l_new = neon_exp_sum(row, m_new, &mut p);

        // Online softmax update
        let m_old = lse[i];
        let m_max = m_old.max(m_new);
        let scale_old = (m_old - m_max).exp();
        let scale_new = (m_new - m_max).exp();

        let l_old_scaled = lse[i].exp() * scale_old;
        let l_new_scaled = l_new * scale_new;
        let l_combined = l_old_scaled + l_new_scaled;

        // Update output row: O[i] = (O[i] * scale_old + P[i,:] @ V) / l_combined
        let o_row = &mut output[i * d..(i + 1) * d];
        neon_scale(o_row, scale_old / l_combined);

        // Accumulate P @ V contribution
        for j in 0..bc {
            let v_row = &v[j * d..(j + 1) * d];
            let weight = p[j] * scale_new / l_combined;
            neon_axpy(o_row, v_row, weight);  // O += weight * V[j]
        }

        // Update LSE
        lse[i] = m_max + l_combined.ln();
    }
}
```

### 3.3 Metal GPU Implementation (Apple Silicon)

```metal
// flash_attention.metal
kernel void flash_attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant FlashAttentionParams& params [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
    const uint batch = gid.z;
    const uint head = gid.y;
    const uint block_i = gid.x;

    threadgroup float s_tile[BLOCK_SIZE][BLOCK_SIZE];
    threadgroup float k_tile[BLOCK_SIZE][HEAD_DIM];
    threadgroup float v_tile[BLOCK_SIZE][HEAD_DIM];

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[HEAD_DIM] = {0.0f};

    // Load Q row for this thread
    float q_row[HEAD_DIM];
    const uint q_idx = batch * params.heads * params.seq_len * params.head_dim
                     + head * params.seq_len * params.head_dim
                     + (block_i * BLOCK_SIZE + tid.x) * params.head_dim;
    for (uint d = 0; d < HEAD_DIM; d++) {
        q_row[d] = Q[q_idx + d];
    }

    // Iterate over K, V blocks
    for (uint block_j = 0; block_j < params.num_blocks_k; block_j++) {
        // Cooperative load of K, V tiles
        load_kv_tile(K, V, k_tile, v_tile, batch, head, block_j, tid, params);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S_ij = Q_i @ K_j^T
        for (uint j = 0; j < BLOCK_SIZE; j++) {
            float dot = 0.0f;
            for (uint d = 0; d < HEAD_DIM; d++) {
                dot += q_row[d] * k_tile[j][d];
            }
            s_tile[tid.x][j] = dot * params.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        float m_ij = -INFINITY;
        for (uint j = 0; j < BLOCK_SIZE; j++) {
            m_ij = max(m_ij, s_tile[tid.x][j]);
        }

        float l_ij = 0.0f;
        for (uint j = 0; j < BLOCK_SIZE; j++) {
            s_tile[tid.x][j] = exp(s_tile[tid.x][j] - m_ij);
            l_ij += s_tile[tid.x][j];
        }

        // Update running softmax
        float m_new = max(m_i, m_ij);
        float scale_old = exp(m_i - m_new);
        float scale_new = exp(m_ij - m_new);
        float l_new = l_i * scale_old + l_ij * scale_new;

        // Update output
        for (uint d = 0; d < HEAD_DIM; d++) {
            o_i[d] = o_i[d] * scale_old / l_new;
            for (uint j = 0; j < BLOCK_SIZE; j++) {
                o_i[d] += s_tile[tid.x][j] * v_tile[j][d] * scale_new / l_new;
            }
        }

        m_i = m_new;
        l_i = l_new;
    }

    // Write output
    for (uint d = 0; d < HEAD_DIM; d++) {
        O[q_idx + d] = o_i[d];
    }
}
```

## 4. Block-Sparse Attention Patterns

### 4.1 Supported Patterns

For efficiency with structured sparsity, we support:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| Dense | Full attention | Short sequences |
| Sliding Window | Attend to local context | Long sequences |
| Global + Local | Global tokens + sliding window | Document understanding |
| Strided | Every k-th token | Periodic patterns |
| Block Diagonal | Independent blocks | Batched independent sequences |

### 4.2 Sparse Mask API

```rust
pub enum AttentionMask {
    Dense,
    SlidingWindow { window_size: usize },
    GlobalLocal { global_tokens: Vec<usize>, window_size: usize },
    Strided { stride: usize, window: usize },
    BlockDiagonal { block_sizes: Vec<usize> },
    Custom { blocks: Vec<(usize, usize, usize, usize)> },  // (q_start, q_end, k_start, k_end)
}

impl AttentionMask {
    /// Returns which (i, j) block pairs should be computed
    pub fn block_pairs(&self, seq_q: usize, seq_k: usize, block_size: usize)
        -> Vec<(usize, usize)>;
}
```

## 5. Consequences

### 5.1 Benefits

1. **Memory Efficiency**: O(N) vs O(N^2) memory usage
2. **Speed**: 2.5x-7.5x faster than standard attention (memory-bound -> compute-bound)
3. **Long Context**: Enables 128K+ context without OOM
4. **Exact Output**: Mathematically identical to standard attention (no approximation)

### 5.2 Costs

1. **Implementation Complexity**: Tiled algorithm harder to implement/debug
2. **Block Size Tuning**: Optimal B_r, B_c depend on hardware
3. **Causal Masking**: Requires additional logic for triangular masks
4. **Gradient Computation**: Backward pass requires recomputation or checkpointing

### 5.3 Performance Targets

| Sequence Length | Standard Attention | Flash Attention | Speedup |
|-----------------|-------------------|-----------------|---------|
| 1K | 2.1 ms | 1.8 ms | 1.2x |
| 4K | 34 ms | 12 ms | 2.8x |
| 16K | 540 ms | 95 ms | 5.7x |
| 64K | OOM | 380 ms | - |
| 128K | OOM | 1.5 s | - |

## 6. Related Decisions

- **ADR-001-simd-first**: SIMD optimization for block operations
- **ADR-015-coherence-gated-transformer**: Flash attention in coherence transformer
- **ADR-002-ruvllm-integration**: LLM inference pipeline

## 7. References

1. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.
2. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."
3. Implementation: `/crates/ruvllm/src/attention/flash.rs`
4. Metal shaders: `/crates/ruvllm/src/metal/shaders/attention.metal`

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-12 | Architecture Team | Initial decision record |
