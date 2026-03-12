# QuIP: 2-Bit LLM Quantization via Incoherence Processing

## Abstract

QuIP (Quantization with Incoherence Processing), developed by Cornell University's
RelaxML group, is the first framework to make 2-bit weight quantization viable for
large language models. It combines two key innovations: incoherence processing
(decorrelating weights and Hessian matrices) and adaptive rounding using
proxy-based optimization. This document analyzes QuIP's approach, its evolution
to QuIP#, and implications for ruvLLM integration.

## 1. The Incoherence Principle

### 1.1 Why Naive 2-Bit Fails

Standard round-to-nearest (RTN) quantization at 2 bits produces catastrophic
error because LLM weight matrices have highly non-uniform structure:

- **Outlier channels**: Some channels have 10-100x larger magnitude
- **Correlated columns**: Adjacent weight columns are often highly correlated
- **Structured sparsity**: Weight matrices have low-rank + sparse structure

When you round a structured matrix to 4 values, the structure is destroyed.

### 1.2 Incoherence Processing

QuIP's key insight: if weight matrices were "incoherent" (uniform, decorrelated),
2-bit quantization would work much better. So we transform them:

```
Original weights:     W  (structured, correlated, non-uniform)
                       |
Incoherence transform: W' = U @ W @ V^T
                       |    (random orthogonal rotations)
                       |
Quantize:             W'_q = Quantize_2bit(W')
                       |    (now works well because W' is incoherent)
                       |
Inference:            y = (U^T @ W'_q @ V) @ x
                       |    (apply inverse rotations during inference)
```

Where U and V are random orthogonal matrices (from the Haar measure).

### 1.3 Mathematical Foundation

**Definition (Incoherence)**: A matrix W in R^{m x n} is mu-incoherent if:

```
max_ij |W_ij| <= mu * ||W||_F / sqrt(m * n)
```

After random orthogonal rotation, with high probability:

```
mu(U @ W @ V^T) <= O(sqrt(log(m + n)))
```

This means the maximum entry of the rotated matrix is only sqrt(log)-factor
larger than the average -- eliminating outliers without information loss.

### 1.4 Hessian Incoherence

QuIP also decorrelates the Hessian matrix H used for rounding decisions:

```
Standard GPTQ:  Minimize (W - W_q)^T @ H @ (W - W_q)
                where H = X^T @ X (input covariance)

QuIP:           H' = V @ H @ V^T  (decorrelate Hessian too)
                W' = U @ W @ V^T   (decorrelate weights)
                Minimize (W' - W'_q)^T @ H' @ (W' - W'_q)
```

With both weight and Hessian incoherent, the rounding problem becomes
nearly element-wise independent, making greedy rounding near-optimal.

## 2. Adaptive Rounding (LDPC Rounding)

### 2.1 Beyond Round-to-Nearest

Even with incoherence, round-to-nearest (RTN) is suboptimal. QuIP uses
adaptive rounding inspired by LDPC codes:

```
Standard RTN:    q_i = argmin_{c in C} |w_i - c|   (independent per weight)

Adaptive:        q = argmin_{q in C^n} ||W' - Q||_H'
                 subject to LDPC parity constraints
```

The LDPC constraints create dependencies between rounding decisions,
allowing error cancellation across weights.

### 2.2 Proxy Rounding Algorithm

```
Input:  Incoherent weights W', Hessian H', codebook C
Output: Quantized weights Q

1. Initialize Q = RTN(W')  // start with naive rounding
2. For t = 1 to T (iterations):
   a. Select weight index i (highest remaining error)
   b. Compute error: e_i = W'_i - Q_i
   c. For each alternative value c in C:
      - Compute delta_loss if Q_i were changed to c
      - Include LDPC constraint satisfaction
   d. If any change reduces total loss, apply it
3. Return Q
```

This iterative refinement typically converges in 3-5 passes.

## 3. QuIP# -- Lattice Codebooks

### 3.1 E8 Lattice

QuIP# replaces the uniform codebook with the E8 lattice:

```
Uniform 2-bit codebook:  {0, 1, 2, 3} (4 values, mapped to weight range)

E8 lattice codebook:     Points from the E8 lattice in R^8
                          Quantize 8 weights at a time as a vector
                          Find nearest E8 lattice point
```

The E8 lattice is the densest known packing in 8 dimensions -- it provides
optimal quantization error for the same number of bits.

### 3.2 Vector Quantization Benefits

Instead of quantizing each weight independently:

```
Scalar 2-bit:   4 values per weight    = 4^1 = 4 options per weight
Vector 2-bit:   2^16 lattice points for 8 weights = 2^2 effective bits/weight
```

Vector quantization captures correlations between groups of 8 weights,
even after incoherence processing removes most correlation.

### 3.3 QuIP# Results

```
Model: LLaMA-2 7B

Method          Bits   WikiText-2 PPL   C4 PPL
-----------------------------------------------
FP16            16     5.47             6.97
GPTQ            4      5.68             7.21
GPTQ            3      6.29             7.89
GPTQ            2      459.2            412.7   (catastrophic)
QuIP            2      8.33             9.85    (viable!)
QuIP#           2      6.85             8.41    (much better)
AQLM            2      7.11             8.63
```

QuIP# at 2-bit approaches GPTQ at 3-bit quality while using 33% less memory.

## 4. Implementation Analysis

### 4.1 Computational Cost

**Incoherence Transform:**
- Generate random orthogonal matrices U (m x m), V (n x n): O(m^2 + n^2)
- Apply rotation: O(m * n * max(m, n)) per layer
- One-time cost, amortized across all inference

**For LLaMA-7B (4096 x 11008 FFN layer):**
```
U generation:   4096^2 * 8 bytes = 128 MB, ~50ms
V generation:   11008^2 * 8 bytes = 968 MB, ~200ms
W' = U @ W @ V: ~150ms (BLAS GEMM)
Total per layer: ~400ms
Total model (32 layers): ~13 seconds one-time
```

### 4.2 Inference Overhead

The rotation matrices must be applied during inference:

```
Standard inference:   y = W_q @ x
QuIP inference:       y = U^T @ (W'_q @ (V @ x))
```

This adds two matrix-vector multiplications per layer:
- V @ x: (n x n) @ (n x 1) -- but x is already in memory
- U^T @ y: (m x m) @ (m x 1)

**Mitigation strategies:**

1. **Fuse rotations**: Pre-compute U^T and V into adjacent layers
2. **Kronecker decomposition**: Factor U, V into smaller rotations
3. **Hadamard rotations**: Replace random orthogonal with structured Hadamard
   (O(n log n) instead of O(n^2))

### 4.3 Hadamard Rotation Optimization

QuIP# uses the fast Walsh-Hadamard transform instead of full orthogonal matrices:

```
Cost reduction:
  Full orthogonal: O(d^2) per layer forward
  Hadamard:        O(d * log(d)) per layer forward

For d=4096:
  Full:     16.7M operations
  Hadamard: 49K operations (340x speedup)
```

The Hadamard transform preserves most incoherence benefits while being
efficient enough for real-time inference.

## 5. Relevance to ruvLLM

### 5.1 Current ruvLLM Quantization Stack

ruvLLM currently supports:

```
Method              Bits     Implementation
-------------------------------------------
K-quants (Q2_K)     2.56     quantize/ruvltra_quant.rs
K-quants (Q3_K)     3.44     quantize/ruvltra_quant.rs
K-quants (Q4_K_M)   4.5      quantize/ruvltra_quant.rs (primary)
GGUF IQ2_XXS        2.06     gguf/quantization.rs (parser only)
GGUF IQ1_S           1.56     gguf/quantization.rs (parser only)
BitNet b1.58        ~2.06    bitnet/ (full pipeline)
```

### 5.2 What QuIP Would Add

1. **Incoherence processing** as a preprocessing step before any quantization
2. **Hadamard-accelerated rotations** for inference-time compensation
3. **E8 lattice codebooks** for vector quantization of 8-weight groups
4. **Better Q2_K quality** -- applying incoherence before K-quant quantization

### 5.3 Integration Strategy

```rust
// Proposed new module: ruvllm/src/quantize/incoherence.rs

/// Hadamard rotation matrix for incoherence processing
pub struct HadamardRotation {
    /// Log2 of dimension (must be power of 2)
    log_dim: u32,
    /// Random sign flips for randomized Hadamard
    signs: Vec<i8>,  // +1 or -1
}

impl HadamardRotation {
    /// Apply fast Walsh-Hadamard transform with random signs
    /// O(d * log(d)) complexity
    pub fn transform(&self, input: &mut [f32]) {
        // Apply random sign flips
        for (x, s) in input.iter_mut().zip(self.signs.iter()) {
            *x *= *s as f32;
        }
        // Fast Walsh-Hadamard in-place
        let n = 1 << self.log_dim;
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    let x = input[j];
                    let y = input[j + h];
                    input[j] = x + y;
                    input[j + h] = x - y;
                }
            }
            h *= 2;
        }
        // Normalize
        let norm = (n as f32).sqrt();
        for x in input.iter_mut() {
            *x /= norm;
        }
    }
}

/// Apply incoherence processing to a weight matrix before quantization
pub fn make_incoherent(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, HadamardRotation, HadamardRotation) {
    let row_rotation = HadamardRotation::new(rows);
    let col_rotation = HadamardRotation::new(cols);

    // W' = H_row @ W @ H_col^T
    let mut w_prime = weights.to_vec();
    // Apply row rotation (left multiply)
    for r in 0..rows {
        let row = &mut w_prime[r * cols..(r + 1) * cols];
        row_rotation.transform(row);
    }
    // Apply column rotation (right multiply by transpose)
    // ... transpose, transform columns, transpose back

    (w_prime, row_rotation, col_rotation)
}
```

### 5.4 SIMD Optimization for Hadamard

The Walsh-Hadamard butterfly can be vectorized with NEON/AVX2:

```rust
#[cfg(target_arch = "aarch64")]
pub fn hadamard_butterfly_neon(a: &mut [f32], b: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        for i in (0..a.len()).step_by(4) {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let sum = vaddq_f32(va, vb);
            let diff = vsubq_f32(va, vb);
            vst1q_f32(a.as_mut_ptr().add(i), sum);
            vst1q_f32(b.as_mut_ptr().add(i), diff);
        }
    }
}
```

## 6. Benchmarks and Projections

### 6.1 Expected Performance with ruvLLM + QuIP

```
Model: RuvLTRA-Small (0.5B parameters)

Config                  Memory    Prefill tok/s   Decode tok/s
--------------------------------------------------------------
FP16 (baseline)         1.0 GB    3500            120
Q4_K_M (current)        300 MB    4200            150
Q2_K (current)          160 MB    3800            135
QuIP 2-bit (projected)  130 MB    3600            140
BitNet 1.58 (current)   130 MB    4500            170
```

### 6.2 Quality Projections

```
Config                  WikiText-2 PPL   Tool Use Accuracy
----------------------------------------------------------
FP16                    12.3             89%
Q4_K_M                  12.8             87%
Q2_K (no incoherence)   18.5             71%
QuIP 2-bit (projected)  14.1             82%
QAT 2-bit (projected)   13.2             85%
QuIP + QAT (projected)  12.9             86%
```

## 7. Open Questions for ruvLLM Integration

1. **Dimension padding**: Hadamard requires power-of-2 dimensions. LLaMA's
   4096 is fine, but intermediate sizes (e.g., 11008) need padding strategy.

2. **Rotation storage**: Where to store Hadamard signs? In GGUF metadata
   or as separate files?

3. **Fusion with BitNet**: Can incoherence processing improve BitNet b1.58
   ternary quality? The ternary quantization might benefit from decorrelation.

4. **Per-layer or global rotations**: Should each layer have its own rotation
   matrices, or can a single global rotation suffice?

5. **ANE compatibility**: Do Hadamard transforms map efficiently to Apple
   Neural Engine operations?
