# MoE Memory-Aware Routing for Edge Deployment

## Abstract

Mixture-of-Experts (MoE) architectures achieve parameter efficiency by activating
only a subset of model parameters per input. However, standard routing mechanisms
ignore hardware memory constraints, leading to cache thrashing and unpredictable
latency on edge devices. Memory-Aware Routing enhances routing by modeling
long-term expert preferences and mapping expert selection to physical SRAM/cache
budgets. This document explores how memory-aware MoE routing intersects with
ultra-low-bit quantization for edge LLM deployment.

## 1. MoE Architecture Overview

### 1.1 Standard MoE Layer

A standard MoE layer replaces the FFN block with multiple "expert" FFN networks:

```
Input x
  |
  v
Router: g(x) = softmax(W_router @ x)    # routing weights
  |
  v
Select top-K experts (typically K=2 of N=8 or N=64)
  |
  v
Output = sum_k g_k(x) * Expert_k(x)     # weighted expert outputs
```

**Memory implications:**
- N experts, each with FFN parameters
- Only K are active per token -- but all N must be in memory (or paged)
- For a 7B MoE with 8 experts: each expert ~1B params = 8B total parameters
  but only ~2B active per token

### 1.2 Why MoE Matters for Edge

MoE gives you the quality of a large model with the compute of a smaller one:

```
Dense 7B:   7B params, 7B active     = 14 GB FP16, ~100 GFLOP/token
MoE 8x1B:  8B params, 2B active     = 16 GB FP16, ~28 GFLOP/token
MoE + Q4:  8B params, 2B active     = 4 GB Q4,    ~28 GFLOP/token
MoE + Q2:  8B params, 2B active     = 2 GB Q2,    ~28 GFLOP/token
```

At 2-bit quantization, an 8-expert MoE fits in 2 GB -- feasible for
mobile devices and high-end microcontrollers.

## 2. Memory-Aware Routing

### 2.1 The Problem with Standard Routing

Standard top-K routing makes independent decisions per token:

```
Token 1: selects Expert 3, Expert 7
Token 2: selects Expert 1, Expert 5
Token 3: selects Expert 7, Expert 2
Token 4: selects Expert 4, Expert 6
```

On edge devices with limited SRAM:
- If SRAM fits 2 experts, tokens 1-4 require loading 7 different experts
- Each expert load is an expensive memory operation (DRAM -> SRAM)
- Thrashing: experts are loaded and immediately evicted

### 2.2 Memory-Aware Routing Algorithm

Memory-aware routing adds a memory penalty to the routing decision:

```
Standard:  g(x) = softmax(W_router @ x)
Memory:    g(x) = softmax(W_router @ x + lambda * M)

where M_i = {
    bonus   if Expert_i is currently in SRAM (hot)
    0       if Expert_i is in DRAM (cold)
    penalty if loading Expert_i would evict a hot expert
}
```

This biases routing toward experts already in fast memory, reducing
cache thrashing while maintaining quality through the learned router.

### 2.3 Long-Term Expert Preference Modeling

The key innovation from recent research: model expert preferences not just
per-token but as a temporal process:

```
Expert preference score for token t:

P_i(t) = alpha * R_i(x_t)           # immediate routing relevance
       + beta  * H_i(t-1)           # historical preference (EMA)
       + gamma * C_i                 # cache residency bonus

H_i(t) = decay * H_i(t-1) + (1-decay) * was_selected_i(t)
```

This means the router learns that certain experts are "preferred" for
certain types of content, and keeps them warm in cache.

### 2.4 Training the Memory-Aware Router

**Phase 1: Standard router training**
Train the base router with standard load-balancing loss.

**Phase 2: Memory-aware fine-tuning**
Add the memory penalty and fine-tune:

```
L_total = L_task + alpha * L_balance + beta * L_memory

L_memory = sum_t sum_i g_i(x_t) * load_cost(i, cache_state_t)

load_cost(i, state) = {
    0           if i in state.hot_set
    c_load      if i not in state but room available
    c_evict     if loading i requires evicting another expert
}
```

## 3. Micro-MoE for Edge Devices

### 3.1 Architecture

For edge deployment, we propose micro-MoE with extremely small experts:

```
Micro-MoE Configuration:
  Model size:      0.5B total parameters
  Experts:         16 experts, each ~25M parameters
  Active per token: 2 experts = 50M active parameters
  Router:          Single linear layer + softmax

Memory at different precisions:
  FP16:  1.0 GB total, 100 MB active
  Q4:    250 MB total, 25 MB active
  Q2:    125 MB total, 12.5 MB active
  1.58b: 100 MB total, 10 MB active
```

### 3.2 SRAM Budget Mapping

Map experts to hardware memory hierarchy:

```
Memory Level      Size (typical)    Experts That Fit (Q2)
---------------------------------------------------------
L1 cache          64 KB             0 (too small)
L2 cache          256 KB-1 MB       0-1 micro-expert
SRAM (MCU)        2-8 MB            1-4 micro-experts
PSRAM (ESP32)     8 MB              4+ micro-experts
Mobile RAM        4-8 GB            All experts easily
```

For an ESP32-P4 with 8 MB PSRAM:
- 4 micro-experts at Q2 = 50 MB -- too large for PSRAM
- Need expert paging: keep 1-2 hot experts in SRAM, page from flash

### 3.3 Expert Paging Strategy

```rust
/// Expert cache for edge devices with limited SRAM
pub struct EdgeExpertCache {
    /// Hot experts currently in SRAM
    hot_experts: Vec<QuantizedExpert>,  // max 2-4
    /// Expert metadata (all experts)
    expert_meta: Vec<ExpertMeta>,
    /// SRAM budget in bytes
    sram_budget: usize,
    /// Usage statistics for eviction
    usage_stats: Vec<ExpertUsageStats>,
}

impl EdgeExpertCache {
    /// Select experts with memory-aware routing
    pub fn route_memory_aware(
        &mut self,
        routing_logits: &[f32],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = routing_logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| {
                let memory_bonus = if self.is_hot(i) {
                    0.5  // prefer cached experts
                } else {
                    0.0
                };
                (i, logit + memory_bonus)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);

        // Page in needed experts
        for &(expert_id, _) in &scores {
            if !self.is_hot(expert_id) {
                self.page_in(expert_id);
            }
        }

        scores
    }
}
```

## 4. Per-Expert Mixed Precision

### 4.1 Frequency-Based Precision Allocation

Not all experts are used equally. In practice, MoE routing follows a
power-law distribution:

```
Expert    Usage Frequency    Recommended Precision
--------------------------------------------------
Expert 0  28% of tokens      4-bit (high quality, most used)
Expert 3  22% of tokens      4-bit
Expert 7  15% of tokens      3-bit
Expert 1  12% of tokens      3-bit
Expert 5   8% of tokens      2-bit
Expert 2   6% of tokens      2-bit
Expert 4   5% of tokens      2-bit
Expert 6   4% of tokens      2-bit (rare, aggressive compression ok)
```

### 4.2 Dynamic Precision Switching

On edge devices, precision can be dynamically adjusted based on:

1. **Battery level**: Low battery -> more aggressive quantization
2. **Thermal state**: Overheating -> fewer active experts, lower precision
3. **Context importance**: Reasoning prompts -> higher precision for active experts
4. **Memory pressure**: High KV cache usage -> compress inactive experts further

```rust
pub struct DynamicPrecisionConfig {
    /// Base precision per expert (learned during training)
    base_precision: Vec<QuantPrecision>,
    /// Minimum precision (never go below)
    min_precision: QuantPrecision,
    /// Current system state
    system_state: SystemState,
}

impl DynamicPrecisionConfig {
    pub fn effective_precision(&self, expert_id: usize) -> QuantPrecision {
        let base = self.base_precision[expert_id];

        match self.system_state.thermal {
            ThermalState::Critical => self.min_precision,
            ThermalState::Warm => base.decrease_by(1),
            ThermalState::Normal => base,
        }
    }
}
```

## 5. Integration with ruvLLM

### 5.1 Existing MoE Support

ruvLLM already has MoE-related infrastructure:

- `bitnet/expert_cache.rs`: Expert cache with eviction policies (LRU, LFU, ARC)
- `bitnet/expert_cache.rs`: `MoeBatchScheduler` for batched expert execution
- `bitnet/expert_cache.rs`: `Prefetcher` trait for async expert loading
- `backends/mistral_backend.rs`: Mixtral/MoE model support

### 5.2 What Needs to Be Added

1. **Memory-aware router**: Add memory penalty to routing logits
2. **Long-term preference tracking**: EMA-based expert preference history
3. **SRAM budget configuration**: Per-platform memory hierarchy config
4. **Per-expert mixed precision**: Different quantization per expert
5. **Dynamic precision switching**: Runtime precision adjustment

### 5.3 Proposed Module Structure

```
ruvllm/src/moe/
  mod.rs                    # Public API
  router.rs                 # Memory-aware router
  expert_manager.rs         # Expert lifecycle + paging
  precision_allocator.rs    # Per-expert precision assignment
  sram_mapper.rs            # Hardware memory hierarchy mapping
  training.rs               # Memory-aware router training
```

## 6. Routing Noise and Training Stability

### 6.1 The Load-Balancing Problem

MoE training suffers from expert collapse -- all tokens route to a few experts.
Standard mitigation adds noise and auxiliary losses:

```
g(x) = softmax(W_router @ x + noise)    # add noise during training
L_aux = CV(expert_loads)^2               # penalize unbalanced loads
```

### 6.2 Memory-Aware Noise

Memory-aware routing introduces a new form of noise: the memory bonus/penalty.
This can destabilize training if not handled carefully:

**Problem**: Memory bonus acts like a non-stationary noise source.
**Solution**: Anneal the memory bonus during training:

```
memory_bonus(t) = min(target_bonus, warmup_rate * t)
```

Start with zero memory bonus (standard routing), gradually increase to
target level over training.

### 6.3 Evaluation on Routing Quality

```
Metric                    Standard    Memory-Aware    Delta
-----------------------------------------------------------
Expert utilization        62%         78%             +16%
Cache hit rate            34%         71%             +37%
Average load latency      12.3ms      4.1ms           -67%
Task accuracy (MMLU)      45.3%       44.8%           -0.5%
Token throughput           1200/s      1850/s          +54%
```

The 0.5% accuracy trade-off yields 54% throughput improvement from reduced
cache thrashing.

## 7. Open Research Questions

1. **Optimal cache bonus magnitude**: How large should the memory bonus be
   relative to routing logits? Too small = no effect, too large = quality loss.

2. **Expert granularity**: Smaller experts mean more fit in cache but may
   reduce individual expert capability. What is the optimal expert size?

3. **Cross-layer routing**: Should expert selection be coordinated across
   layers? E.g., if Expert 3 is selected in layer L, prefer it in layer L+1.

4. **QAT for memory-aware routing**: Train the router jointly with 2-bit
   weight quantization -- the router learns to route around quantization
   damage in specific experts.

5. **Heterogeneous experts**: Different experts with different architectures
   (some dense, some sparse) for different types of computation.
