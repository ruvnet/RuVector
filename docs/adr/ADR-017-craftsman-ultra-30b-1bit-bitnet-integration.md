# ADR-017: Craftsman Ultra 30b 1bit — BitNet Integration with RuvLLM

**Status:** Proposed
**Date:** 2026-02-03
**Decision Makers:** Ruvector Architecture Team
**Technical Area:** 1-Bit LLM Inference / MoE Architecture / CPU-Native Serving

---

## Context and Problem Statement

Large language models require substantial GPU resources for inference, limiting deployment to cloud environments and specialized hardware. Recent advances in 1-bit quantization — specifically Microsoft Research's BitNet b1.58 — demonstrate that ternary-weight models ({-1, 0, +1}) can match full-precision performance at 3B+ parameters while enabling CPU-only inference at human-readable speeds.

Concurrently, Zhipu AI's GLM-4.7-Flash introduces a 30B-A3B Mixture-of-Experts architecture that activates only ~3B parameters per token while storing 30B total knowledge, achieving strong coding and agentic benchmarks (SWE-bench Verified: 59.2%, LiveCodeBench v6: 64.0%) with 200K context.

**Craftsman Ultra 30b 1bit** is a proposed model that combines these two paradigms: a 30B-A3B MoE architecture with native BitNet b1.58 ternary quantization, purpose-built for CPU inference within the RuvLLM serving runtime. This ADR evaluates the integration path, architectural decisions, and trade-offs.

### Strategic Goal

Deliver a 30B-class coding/agentic model that runs entirely on consumer CPUs (no GPU required) at 5-15 tokens/second decode, with memory footprint under 8GB, integrated into the RuvLLM + Ruvector ecosystem with SONA self-learning capabilities.

---

## Decision Drivers

### Performance Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| Decode throughput (CPU) | 5-15 tok/s | Human-readable speed per BitNet 100B benchmarks |
| Prefill latency (1K tokens) | <2s | Interactive coding assistant responsiveness |
| Memory footprint (model) | <8 GB | Fits in 16GB system RAM with OS + KV cache |
| Memory footprint (KV cache, 4K ctx) | <2 GB | Q8 KV cache for 4096-token context |
| Active parameter GEMM | Addition-only | BitNet eliminates multiplication in W×A |
| Energy per inference | <0.05J | BitNet CPU efficiency benchmarks |

### Architecture Requirements

- **MoE routing must remain full-precision**: Expert selection requires accurate gating scores
- **Expert weights are ternary**: Each expert's linear layers use BitLinear (W1.58A8)
- **Activations quantized to INT8**: Per-token absmax scaling
- **Shared layers (embeddings, LM head) remain FP16**: Critical for quality preservation
- **GGUF-compatible**: Must serialize to/load from GGUF v3 format with custom metadata

### Ecosystem Requirements

- Integrate with RuvLLM's existing backend abstraction (`backends/mod.rs`)
- Leverage existing GGUF parser (`gguf/parser.rs`, `gguf/quantization.rs`)
- Support SONA learning loops for per-session adaptation
- Compatible with Claude Flow agent routing for task delegation
- NAPI bindings for Node.js consumption via `npm/packages/ruvllm`

---

## Research Summary

### BitNet b1.58 Architecture

**Source**: Microsoft Research, "The Era of 1-bit LLMs" (Feb 2024), bitnet.cpp (Oct 2024)

BitNet b1.58 replaces standard `nn.Linear` with `BitLinear` layers:

```
Forward Pass:
  1. W_ternary = RoundClip(W / (gamma + epsilon), -1, 1)
     where gamma = mean(|W|) (absmean quantization)
  2. X_int8 = Quant(X, absmax)  (per-token 8-bit activation)
  3. Y = W_ternary @ X_int8      (integer addition only, no multiplication)
  4. Y_float = Dequant(Y)         (rescale to float)
```

**Key properties:**
- Weights: ternary {-1, 0, +1} → 1.58 bits per parameter
- Activations: INT8 per-token (absmax scaling)
- Matrix multiply becomes **addition and subtraction only** (no FP multiply)
- Zero weights enable **feature filtering** (sparse activation within dense layers)
- Must be **trained from scratch** — post-training quantization to 1-bit destroys quality

**Inference kernels (bitnet.cpp):**

| Kernel | Method | Compression | Best For |
|--------|--------|-------------|----------|
| I2_S | 2-bit pack, unpack-and-multiply | 2 bits/weight | Bandwidth-limited |
| TL1 | 2-weight → 4-bit LUT index | 2 bits/weight | Balanced CPU |
| TL2 | 3-weight → 5-bit LUT index | 1.67 bits/weight | Memory-limited |

**CPU performance (bitnet.cpp benchmarks):**

| Platform | Speedup vs FP16 | Energy Reduction |
|----------|-----------------|-----------------|
| ARM (NEON) | 1.37x – 5.07x | 55-70% |
| x86 (AVX2) | 2.37x – 6.17x | 72-82% |
| x86 (AVX512) | ~6x+ | ~85% |

### GLM-4.7-Flash Architecture

**Source**: Zhipu AI / Z.AI (Jan 2026)

| Property | Value |
|----------|-------|
| Total parameters | ~30B (31B reported) |
| Active parameters | ~3B (A3B) |
| Architecture | Mixture of Experts (MoE) |
| Shared layers | ~2B parameters |
| Expert layers | ~28B (distributed across experts) |
| Context window | 200K tokens (MLA-based) |
| Training data | 15T general + 7T reasoning/code tokens |
| Attention | Multi-head Latent Attention (MLA) with QK-Norm |
| Activation | SwiGLU |
| Position encoding | RoPE |
| Speculative decoding | Multi-Token Prediction (MTP) layer |
| Reasoning | Interleaved + Retention-Based + Round-Level |

**Benchmark performance:**

| Benchmark | Score |
|-----------|-------|
| AIME 25 | 91.6% |
| GPQA | 75.2% |
| SWE-bench Verified | 59.2% |
| LiveCodeBench v6 | 64.0% |
| HLE | 14.4% |
| tau2-Bench | 79.5% |

### RuvLLM Current Capabilities (Relevant)

- **GGUF v3 parser**: Full format support including IQ1_S (1.56 bits/weight, type 19)
- **Quantization pipeline**: Q4_K_M, Q5_K_M, Q8_0, F16 (no native ternary training)
- **Backends**: Candle (Metal/CUDA), mistral-rs (PagedAttention), CoreML (ANE)
- **No CPU-optimized ternary kernel**: Current backends target GPU acceleration
- **SIMD kernels**: Existing NEON/SSE4.1/AVX2 infrastructure in `crates/ruvllm/src/kernels/`
- **MicroLoRA**: Rank 1-2 adapters with <1ms adaptation (compatible with BitNet)
- **SONA**: Three-tier learning (instant/background/deep) — can drive ternary adapter training

### RuvLLM RLM Training Stack (Reusable for Distillation)

RuvLLM contains a mature reinforcement-learning-from-model-feedback (RLM) training stack that directly accelerates Craftsman Ultra distillation. These components are production-tested and reduce net-new code by ~70%.

**GRPO — Group Relative Policy Optimization** (`training/grpo.rs`, 897 lines)
- Critic-free RL: computes relative advantages within sample groups
- Adaptive KL divergence penalty (`kl_target`, `clip_range`) controls teacher-student divergence
- PPO-style clipping prevents catastrophic updates
- Preset configs: `GrpoConfig::stable()` (safe distillation), `GrpoConfig::for_tool_use()` (expert routing)
- Thread-safe batch processing via `RwLock<VecDeque<SampleGroup>>`

**RealContrastiveTrainer** (`training/real_trainer.rs`, 1000 lines)
- Candle-based training loop with GGUF model loading and GGUF weight export
- Combined loss: Triplet (margin) + InfoNCE (contrastive) + GRPO reward scaling
- AdamW optimizer with gradient clipping, LR warmup, checkpointing
- `GrpoEvaluator` computes per-prediction rewards (1.0 correct, -0.5 wrong)
- Metal/CUDA acceleration via Candle device dispatch

**MicroLoRA + EWC++ Training Pipeline** (`lora/training.rs`, 798 lines)
- Single-example gradient computation (batch_size=1 for real-time)
- EWC++ regularizer: `λ/2 * Σ F_i * (w_i - w*_i)²` prevents catastrophic forgetting
- Fisher diagonal tracking with exponential decay (`fisher_decay: 0.999`)
- 7 learning rate schedules (Cosine, OneCycle, Step, etc.)
- Async adaptation with buffered gradient accumulation

**Memory Distillation** (`reasoning_bank/distillation.rs`, 856 lines)
- Compresses trajectories to `KeyLesson` objects with semantic embeddings
- Smart extraction: explicit lessons, implicit patterns, error patterns, recovery patterns
- Semantic deduplication (Jaccard + cosine similarity, threshold 0.85)
- Quality-gated: only trajectories above `min_quality_threshold` are preserved

**Policy Store** (`policy_store.rs`, 474 lines)
- Ruvector-backed semantic policy persistence with HNSW indexing
- Policy types: `Quantization`, `Router`, `Ewc`, `Pattern`
- Per-layer `QuantizationPolicy` with precision, activation thresholds, quality-latency tradeoff
- Policy source tracking: `InstantLoop`, `BackgroundLoop`, `DeepLoop`, `Federated`

**Contrastive Training** (`training/contrastive.rs`, 634 lines)
- Two-stage: Triplet Loss (margin=0.5) + InfoNCE (temperature=0.07)
- 13 agent types with 1,078 training triplets (578 base + 500 hard negatives)
- Hard negative mining at 48.4% ratio (Claude-generated confusing pairs)
- Proven 100% routing accuracy with hybrid keyword-first + embedding fallback

---

## Considered Options

### Option A: Post-Training Quantization of GLM-4.7-Flash (PTQ Tiers)

Take the existing BF16 GLM-4.7-Flash weights and quantize to low-bit formats without full distillation training.

**Critical distinction — IQ1_S ≠ BitNet b1.58:**

| Property | GGUF IQ1_S | BitNet b1.58 |
|----------|-----------|--------------|
| Encoding | Codebook-based importance quantization | Ternary {-1, 0, +1} via absmean |
| Bits/weight | 1.56 bpw | 1.58 bpw |
| Inference | **Dequantize → FP multiply** | **Integer addition only (no multiply)** |
| Speed benefit | Memory bandwidth only | Bandwidth + compute (multiplication-free) |
| How obtained | Post-training quantization | Trained from scratch or distilled |
| Quality at 7B | Near-random / broken outputs | Matches FP16 |

**Existing GLM-4.7-Flash GGUF quantizations available** (community-published):

| Repository | Lowest Quant | Size | Notes |
|-----------|-------------|------|-------|
| [bartowski/zai-org_GLM-4.7-Flash-GGUF](https://huggingface.co/bartowski/zai-org_GLM-4.7-Flash-GGUF) | IQ2_XXS (2.06 bpw) | 7.62 GB | No IQ1_S published |
| [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) | UD-Q2_K_XL (2.7 bpw dynamic) | ~11 GB | Dynamic quant, recommended |
| [ngxson/GLM-4.7-Flash-GGUF](https://huggingface.co/ngxson/GLM-4.7-Flash-GGUF) | Q4_K_M (4.5 bpw) | 18.1 GB | 55 variants available |

**No IQ1_S quantization** has been published for GLM-4.7-Flash by any community quantizer — this itself is a signal (too aggressive for practical use).

**Sub-options ranked by increasing effort:**

**Sub-option 0A: Download existing IQ2_XXS GGUF**
- Download bartowski's IQ2_XXS at 7.62 GB
- Cost: $0, time: 5 minutes (just download)
- Quality: ~75-80% of FP16 (2.06 bpw is usable per community reports)
- NOT 1-bit, NOT BitNet — just aggressive 2-bit compression
- RuvLLM gap: IQ2_XXS dequantization not implemented (falls to error catch-all in `quantization.rs:358`)
- RuvLLM Q2_K dequantization IS implemented and works

**Sub-option 0B: Quantize to IQ1_S via llama.cpp**
- Run `llama-quantize GLM-4.7-Flash-F16.gguf IQ1_S` with importance matrix
- Cost: $0, time: ~30 minutes on CPU
- Quality: **SEVERE degradation** — blind testing shows IQ1_S is "broken rather than just bad" on 7B; outputs contain garbled text despite acceptable perplexity scores. 30B MoE may survive better due to parameter redundancy, but expert routing is highly sensitive to weight perturbation
- RuvLLM gap: IQ1_S dequantization not implemented (`quantization.rs:358` catch-all)
- Does NOT achieve BitNet multiplication-free inference

**Sub-option 0C: PT-BitNet ternary PTQ** (per [PT-BitNet paper](https://www.sciencedirect.com/science/article/abs/pii/S089360802500735X))
- Apply absmean ternary quantization (BitNet's native method) to pre-trained weights with calibration data
- Cost: **$0** (runs locally on Mac Studio via mmap + Metal; 1-4 hours wall time)
- Alternative: ~$50-200 on cloud GPU if no local Apple Silicon hardware
- Quality: ~55-65% downstream accuracy (PT-BitNet reports 61% on 70B; GLM-4.7-Flash's 30B-A3B may differ)
- THIS IS proper BitNet ternary format → **enables multiplication-free inference with AD-4 kernels**
- Requires implementing absmean ternary quantizer (~200-300 lines of new code)
- Requires calibration dataset (WikiText-2 or similar, ~1M tokens)
- Mac Studio M4 Max 64GB+ or M3 Ultra 96GB+ recommended (see AD-18)

**Sub-option 0D: BitDistill Lite (10B tokens)** (per [BitDistill paper](https://arxiv.org/html/2510.13998v1))
- 3-stage: SubLN insertion → 10B-token continued pre-training → KL + attention distillation
- Cost: ~$200-500 (8× GPU hours on Mi300X/A100 class)
- Quality: **~90-95% of FP16** (BitDistill reports 88.17% vs 88.01% FP16 on MNLI at 0.6B)
- Near-full quality recovery with only 10B tokens (vs 200B+ for Phase 1 full distillation)
- Requires SubLN module insertion + distillation fine-tuning loop
- Bridges gap between pure PTQ and full expert distillation (Phase 1)

**Summary comparison:**

| Sub-option | Cost | Time | Quality (est.) | BitNet Speedup | RuvLLM Ready |
|-----------|------|------|---------------|----------------|-------------|
| 0A: IQ2_XXS download | $0 | 5 min | ~75-80% | No | No (missing dequant) |
| 0B: IQ1_S quantize | $0 | 30 min | ~40-50% | No | No (missing dequant) |
| 0C: PT-BitNet PTQ | **$0 (Mac Studio)** | 1-4 hrs | ~55-65% | **Yes** | Needs quantizer impl |
| 0D: BitDistill Lite | $0 local / ~$300 cloud | 2-4 wks / 1-2 days | ~90-95% | **Yes** | Needs SubLN + KD loop |

**Pros (of PTQ approach generally):**
- Immediate or near-immediate results ($0-$300, minutes to days)
- No large-scale training infrastructure
- Validates inference pipeline and kernels before investing in full distillation
- Sub-option 0C produces genuine BitNet ternary format for kernel development

**Cons:**
- Sub-options 0A/0B: Quality too degraded for production coding tasks
- Sub-options 0A/0B: No BitNet multiplication-free inference (still dequant-then-multiply)
- Sub-option 0C: Significant quality loss (~35-45%) vs teacher — adequate for kernel validation, not production
- Sub-option 0D: Requires non-trivial training code (SubLN, KD loss) but much less than full Phase 1
- IQ1_S blind test results: statistically indistinguishable from random on smaller models

**Verdict: Recommended as Phase 0 rapid prototype** — Sub-option 0C (PT-BitNet PTQ) is the optimal entry point: $100, 2-4 hours, produces genuine BitNet ternary format for kernel development and inference validation. Sub-option 0D (BitDistill Lite) bridges to Phase 1 if higher quality is needed before committing to full expert distillation. Sub-options 0A/0B are useful only as baselines for comparison.

### Option B: Native BitNet Training of GLM-4.7-Flash Architecture (Full)

Train Craftsman Ultra 30b 1bit from scratch using BitNet b1.58 methodology on the GLM-4.7-Flash MoE architecture.

**Approach:**
1. Implement BitLinear layers for all expert MLPs and attention projections
2. Keep MoE router, embeddings, and LM head in FP16
3. Train on 4T+ tokens with ternary weight updates via straight-through estimator
4. Export to custom GGUF with ternary tensor metadata

**Pros:**
- Maximum quality — matches FP16 at 3B+ active parameter scale
- True multiplication-free inference for expert forward passes
- Full TL1/TL2 kernel optimization possible
- Scientifically validated approach (BitNet b1.58 2B4T results)

**Cons:**
- Massive training compute: estimated 4,000-8,000 A100-hours for 4T tokens
- Requires custom training framework (BitNet + MoE + MLA integration)
- 6-12 month timeline for training pipeline + training run
- No pre-existing GLM-4.7-class BitNet training recipe

**Verdict: Recommended long-term** — Highest quality but requires significant investment.

### Option C: Hybrid Approach — BitNet Distillation from GLM-4.7-Flash (RLM-Accelerated)

Use knowledge distillation to transfer GLM-4.7-Flash capabilities into a BitNet architecture, reducing training cost by 5-10x. **Leverages the existing RLM training stack** to eliminate ~70% of net-new training code.

**Approach:**
1. Initialize Craftsman Ultra with GLM-4.7-Flash architecture (30B-A3B MoE)
2. Replace all expert linear layers with BitLinear (ternary {-1, 0, +1})
3. Keep router, embeddings, LM head in FP16
4. **Extend `RealContrastiveTrainer`** with KD loss (KL div + hard-label CE) replacing triplet+InfoNCE
5. **Use `GrpoOptimizer`** for per-expert quality rewards during distillation — each `SampleGroup` maps to one expert's teacher vs student outputs
6. **Apply `EwcRegularizer`** across distillation phases to prevent early-trained experts from being overwritten
7. **Log distillation trajectories** to `MemoryDistiller` for quality tracking and `KeyLesson` extraction
8. **Persist per-layer ternary policies** via `PolicyStore` (quantization thresholds, scale distributions)
9. Export to GGUF with ternary tensor metadata and TL1/TL2 kernel hints via existing `GgufExportResult`

**RLM Component Reuse:**

| Existing Component | Reuse | Adaptation Needed |
|-------------------|-------|-------------------|
| `RealContrastiveTrainer` | Training loop, GGUF export, checkpointing | Replace triplet+InfoNCE with KD loss |
| `GrpoOptimizer` | Reward scaling, adaptive KL, PPO clipping | Map `SampleGroup` to per-expert outputs |
| `EwcRegularizer` | Fisher diagonal, forgetting prevention | Apply across expert distillation phases |
| `MemoryDistiller` | Trajectory compression, lesson extraction | Map `Verdict` to teacher-student quality delta |
| `PolicyStore` | Semantic policy persistence | Add `PolicyType::TernaryScale` for per-block absmean tracking |
| `ContrastiveTrainer` | Hard negative mining framework | Reuse for expert-routing contrastive pre-training |

**Pros:**
- 5-10x less compute than training from scratch (~800-1,600 A100-hours)
- **~70% existing code reuse** — only BitLinear forward/backward and MoE data loading are net-new
- Leverages GLM-4.7-Flash's proven architecture and routing
- GRPO's adaptive KL prevents ternary student from diverging too far from teacher
- EWC++ ensures sequential expert distillation doesn't corrupt earlier experts
- Teacher model provides strong supervision signal for ternary convergence
- Can incrementally improve with more distillation tokens
- `PolicyStore` enables learned per-layer quantization decisions
- Distillation quality tracked end-to-end via `MemoryDistiller` trajectory logging

**Cons:**
- Slight quality gap vs native training (estimated 2-5% on benchmarks)
- `RealContrastiveTrainer` embedding_dim (896) must scale to GLM-4.7-Flash hidden_size
- Teacher inference cost during distillation
- Distillation may not perfectly transfer MoE routing behavior

**Verdict: Recommended near-term** — Best balance of quality, cost, and timeline. RLM reuse eliminates the "custom framework" risk.

### Option D: BitNet Expert Replacement (Incremental, RLM-Accelerated)

Keep GLM-4.7-Flash structure but replace only the expert MLP layers with BitLinear, leaving attention in FP16. **Reuses existing RLM stack for the entire distillation loop.**

**Approach:**
1. Load GLM-4.7-Flash architecture
2. Replace expert FFN layers (gate_proj, up_proj, down_proj) with BitLinear
3. Keep attention (Q/K/V/O projections) in FP16
4. **Use `RealContrastiveTrainer` + `GrpoOptimizer`** for expert-only distillation (~200B tokens)
5. **Apply `EwcRegularizer`** to prevent expert N+1 distillation from corrupting expert N
6. Attention weights loaded directly from GLM-4.7-Flash (no distillation needed)
7. **Use contrastive pre-training** to validate MoE routing still selects correct experts after ternary conversion

**Pros:**
- Fastest path to working model
- Attention quality preserved exactly
- Expert FFN is 60-70% of active parameters — gets most BitNet benefits
- Simpler distillation (only FFN layers)
- Lower memory: ~5.5 GB for ternary experts + FP16 attention
- **Minimal net-new code**: BitLinear layer + GGUF ternary type only; training loop is 100% reused

**Cons:**
- Attention layers still require FP multiply (not fully multiplication-free)
- Mixed-precision inference path complexity
- ~40% of compute still in FP16 attention

**Verdict: Recommended as Phase 1** — Enables rapid prototyping and validation. RLM reuse makes this achievable with only ~30% new code.

---

## Decision

**Phased approach: A(0C) → D → C → B**

### Phase 0: PTQ Rapid Prototype (Option A, Sub-option 0C)
- **Timeline**: 1-2 weeks
- **Cost**: **$0** (runs entirely on Mac Studio locally)
- **Platform**: Mac Studio (M4 Max 64GB+ or M3 Ultra 96GB+)
- **Goal**: Produce a genuine BitNet ternary GGUF of GLM-4.7-Flash for kernel development, inference pipeline validation, and baseline quality measurement
- **Deliverables**:
  - PT-BitNet ternary quantized GLM-4.7-Flash GGUF file (~6-7 GB)
  - Absmean ternary quantizer implementation (~200-300 lines)
  - IQ1_S / BITNET_T158 dequantization kernel in RuvLLM
  - Baseline quality benchmarks (HumanEval, MMLU) to compare against Phase 1+
  - Functional TL1 kernel validated against ternary model
- **Expected quality**: ~55-65% of GLM-4.7-Flash (adequate for kernel validation, not production)
- **Key value**: De-risks Phase 1 by validating the entire inference pipeline (GGUF loading → ternary dequant → TL1 kernel → MoE routing → token generation) at zero cost before committing to $1,300+ distillation training
- **Why Mac Studio works**: Phase 0 is PTQ (no training loop) — just load FP16 weights via mmap, compute absmean per block, round to ternary, export. The absmean computation is trivial math; the bottleneck is memory bandwidth, not compute. Calibration forward pass uses Metal GPU acceleration via existing Candle integration.
- **Optional upgrade (0D)**: If 0C quality is too low for meaningful testing, apply BitDistill Lite (10B tokens, ~$300 cloud or ~$0 on Mac Studio over several weeks) to reach ~90-95% quality

### Phase 1: BitNet Expert Replacement (Option D)
- **Timeline**: 3-4 months
- **Cost**: ~$1,300-$2,000 (4× A100 spot, ~46 days)
- **Goal**: Full-quality ternary experts via distillation, validated against Phase 0 baseline
- **Deliverables**: Working Craftsman Ultra 30b 1bit (mixed: ternary experts, FP16 attention)
- **Expected quality**: ~90-95% of GLM-4.7-Flash on coding benchmarks
- **Prerequisites**: Phase 0 validates inference pipeline works end-to-end

### Phase 2: Full BitNet Distillation (Option C)
- **Timeline**: 4-6 months after Phase 1
- **Cost**: ~$2,500-$5,000 (4× H100, 16-32 days)
- **Goal**: Full ternary model with complete BitNet inference optimization
- **Deliverables**: Craftsman Ultra 30b 1bit v2 (full ternary except router/embed/head)
- **Expected quality**: ~95-98% of GLM-4.7-Flash

### Phase 3: Native BitNet Training (Option B)
- **Timeline**: 6-12 months after Phase 2, contingent on funding/compute
- **Cost**: ~$15,000-$30,000 (8× H100 cluster, 90-180 days)
- **Goal**: Surpass GLM-4.7-Flash quality with native ternary training
- **Deliverables**: Craftsman Ultra 30b 1bit v3 (trained from scratch)
- **Expected quality**: 100%+ of GLM-4.7-Flash (BitNet at scale exceeds FP16)

---

## Architectural Decisions

### AD-1: Ternary Weight Representation

**Decision**: Use BitNet b1.58 absmean quantization for weight ternary encoding.

```
W_ternary = RoundClip(W / (mean(|W|) + epsilon), -1, 1)
```

Each weight is one of {-1, 0, +1}, stored as 2-bit packed integers (I2_S format) in GGUF tensors. Per-block scale factor stored as FP16.

**Storage format per block (256 elements):**
- 64 bytes for ternary weights (2 bits × 256)
- 2 bytes for absmean scale (FP16)
- Total: 66 bytes / 256 weights = **2.06 bits/weight**

### AD-2: MoE Router Precision

**Decision**: MoE gating/routing network remains in FP16.

**Rationale**: Expert selection requires high-precision softmax scores to maintain routing quality. Quantizing the router to ternary would collapse expert selection, effectively turning a 30B model into a random-expert 3B model. The router is <0.1% of total parameters.

**Components kept in FP16:**
- Expert gating weights (router)
- Token embedding table
- LM head (output projection)
- RoPE frequency table
- LayerNorm/RMSNorm parameters

### AD-3: Activation Quantization

**Decision**: INT8 per-token absmax quantization for activations flowing through BitLinear layers.

```
X_int8 = clamp(round(X * 127 / max(|X|)), -128, 127)
```

**Rationale**: Consistent with BitNet b1.58 specification. INT8 activations enable integer-only GEMM in expert forward passes. Attention activations remain in FP16/BF16 for KV cache compatibility.

### AD-4: CPU Inference Kernel Strategy

**Decision**: Implement all three bitnet.cpp kernel types, with runtime selection based on hardware detection.

| Kernel | Target Hardware | Selection Criteria |
|--------|----------------|-------------------|
| **I2_S** | x86 AVX512, ARM SVE | Systems with wide SIMD and high bandwidth |
| **TL1** | x86 AVX2, ARM NEON | General-purpose, balanced performance |
| **TL2** | Memory-constrained | Systems with <16GB RAM or high cache pressure |

**Implementation path**: Adapt bitnet.cpp's kernel generation scripts (Python codegen) to produce Rust SIMD intrinsics compatible with RuvLLM's existing `kernels/` module structure.

**Key kernel operations:**
1. Pack ternary weights into 2-bit (I2_S) or LUT index (TL1: 4-bit, TL2: 5-bit)
2. Generate lookup tables for activation sums at model load time
3. Execute GEMM via table lookup + integer addition (no floating-point multiply)
4. Accumulate in INT16 with pack-and-unpack technique (lossless, no quantization of partials)
5. Dequantize output with per-block FP16 scale

### AD-5: GGUF Tensor Format Extension

**Decision**: Extend RuvLLM's GGUF format with BitNet-specific metadata and a new `BITNET_TERNARY` quantization type.

**New GGUF metadata keys:**
```
craftsman.bitnet.version = 1
craftsman.bitnet.weight_encoding = "absmean_ternary"
craftsman.bitnet.activation_bits = 8
craftsman.bitnet.router_precision = "f16"
craftsman.bitnet.kernel_hint = "tl1"  // preferred kernel
craftsman.moe.total_params = 30000000000
craftsman.moe.active_params = 3000000000
craftsman.moe.num_experts = <N>
craftsman.moe.active_experts = <K>
```

**Tensor storage**: Map to existing `IQ1_S` (type 19) for ternary expert weights, with additional metadata distinguishing post-training IQ1_S from native BitNet ternary. Alternatively, register a new type `BITNET_T158 = 29` if the existing IQ1_S block format is incompatible with absmean-scale-per-block layout.

### AD-6: RuvLLM Backend Integration

**Decision**: Create a new `BitNetBackend` alongside existing Candle and mistral-rs backends.

```
backends/
├── mod.rs                 // Backend trait + dispatch
├── candle_backend.rs      // GPU (Metal/CUDA)
├── mistral_backend.rs     // PagedAttention + ISQ
├── coreml_backend.rs      // Apple Neural Engine
└── bitnet_backend.rs      // NEW: CPU ternary inference
```

**BitNetBackend responsibilities:**
1. Load GGUF with ternary tensor detection
2. Initialize TL1/TL2/I2_S lookup tables per layer
3. Execute MoE routing in FP16 → select active experts
4. Run selected expert forward passes using ternary GEMM kernels
5. Attention in FP16 (Phase 1) or ternary (Phase 2+)
6. KV cache management (Q8 two-tier, existing infrastructure)

**Backend trait compliance:**
```rust
impl InferenceBackend for BitNetBackend {
    fn load_model(&mut self, path: &Path, config: ModelConfig) -> Result<()>;
    fn generate(&self, prompt: &str, params: GenerateParams) -> Result<Response>;
    fn get_embeddings(&self, text: &str) -> Result<Vec<f32>>;
    fn supports_architecture(&self, arch: &str) -> bool;
}
```

### AD-7: MoE Forward Pass Pipeline

**Decision**: Split MoE forward pass into FP16 routing + ternary expert execution.

```
Input Token Embedding (FP16)
  │
  ▼
┌─────────────────────────────────────────┐
│ For each transformer layer:             │
│                                         │
│  1. RMSNorm (FP16)                      │
│  2. Self-Attention                      │
│     ├─ Q/K/V projection (Phase 1: FP16, │
│     │                    Phase 2: Ternary)│
│     ├─ RoPE (FP16)                      │
│     ├─ Scaled dot-product attention      │
│     └─ Output projection                │
│  3. RMSNorm (FP16)                      │
│  4. MoE Block:                          │
│     ├─ Router (FP16 gating network)     │
│     │   → Select top-K experts          │
│     ├─ Expert FFN (TERNARY BitLinear)   │
│     │   ├─ gate_proj: W_ternary @ X_int8│
│     │   ├─ up_proj:   W_ternary @ X_int8│
│     │   ├─ SwiGLU activation            │
│     │   └─ down_proj: W_ternary @ X_int8│
│     └─ Weighted sum of expert outputs   │
│  5. Residual connection                 │
└─────────────────────────────────────────┘
  │
  ▼
LM Head (FP16) → Logits → Token
```

### AD-8: SONA Integration for Ternary Adaptation

**Decision**: MicroLoRA adapters applied as FP16 deltas on top of ternary base weights.

**Rationale**: Ternary weights cannot be directly fine-tuned at inference time (gradient updates don't map to {-1, 0, +1}). Instead, SONA's MicroLoRA applies rank-1 FP16 adapters whose output is added to the ternary forward pass output:

```
Y = BitLinear(X) + LoRA_B @ LoRA_A @ X
```

Where `BitLinear(X)` uses ternary GEMM and `LoRA_B @ LoRA_A @ X` is a small FP16 correction. This preserves BitNet's efficiency for 99%+ of computation while enabling per-session adaptation.

### AD-9: Memory Budget Analysis

**Decision**: Target <8GB model + 2GB KV cache = 10GB total for 4K context.

| Component | Precision | Size | Notes |
|-----------|-----------|------|-------|
| Expert weights (28B params) | 1.58-bit | ~5.5 GB | 28B × 2.06 bits = ~7.2 GB raw, but only routing metadata for inactive experts |
| Shared layers (2B params) | FP16 | ~4 GB | Embeddings, LM head, router, norms |
| Expert routing tables | FP16 | ~50 MB | Gating network weights |
| TL1/TL2 lookup tables | INT16 | ~200 MB | Pre-computed at load time |
| KV cache (4K context) | Q8 | ~1.5 GB | Two-tier cache (hot FP16 + warm Q8) |
| MicroLoRA adapters | FP16 | ~10 MB | Rank-1, <1MB per target module |
| **Total** | — | **~7.8 GB** | Fits in 16GB system with headroom |

**Note**: Full 30B ternary weights on disk are ~7.2 GB. At runtime, only active expert weights (~3B active) are in hot memory for any given token, with inactive expert pages memory-mapped and demand-loaded.

### AD-10: Platform-Specific Kernel Dispatch

**Decision**: Runtime hardware detection drives kernel selection.

```rust
pub fn select_kernel(caps: &HardwareCaps) -> BitNetKernel {
    if caps.has_avx512() {
        BitNetKernel::I2S_AVX512
    } else if caps.has_avx2() {
        BitNetKernel::TL1_AVX2
    } else if caps.has_neon() {
        if caps.cache_size_l2 >= 2 * 1024 * 1024 {
            BitNetKernel::TL1_NEON
        } else {
            BitNetKernel::TL2_NEON  // memory-constrained
        }
    } else if caps.has_sse41() {
        BitNetKernel::TL1_SSE41
    } else {
        BitNetKernel::I2S_Scalar  // fallback
    }
}
```

**Integration**: Leverages RuvLLM's existing `autodetect.rs` hardware capability detection module.

### AD-11: GRPO-Guided Distillation Loss

**Decision**: Use `GrpoOptimizer` to compute per-expert reward scaling during knowledge distillation, replacing a traditional fixed-weight KD loss.

**Rationale**: Standard KD uses a static `alpha` to blend KL divergence and hard-label cross-entropy. GRPO adds a dynamic reward signal that upweights expert-student pairs where ternary output closely matches the teacher, and downweights divergent pairs. This is achieved by mapping each expert's teacher-vs-student output comparison to a `SampleGroup`:

```
Combined Loss = KD_base + GRPO_scale
Where:
  KD_base  = α * KL(teacher_logits/T, student_logits/T)
           + (1-α) * CE(labels, student_logits)
  GRPO_scale = (1 + reward * 0.1)

  reward = GrpoEvaluator.evaluate(student_expert_output, teacher_expert_output)
         → 1.0 when cosine_sim > 0.95
         → -0.5 when cosine_sim < 0.7
```

**Key configuration** (extending `GrpoConfig::stable()`):
```rust
GrpoConfig {
    group_size: num_experts,        // One group per MoE layer
    learning_rate: 1e-6,            // Conservative for distillation
    kl_coefficient: 0.1,            // Tight teacher adherence
    kl_target: 0.02,                // Low divergence target
    clip_range: 0.1,                // Narrow clipping for stability
    normalize_advantages: true,     // Normalize across experts in group
    adaptive_kl: true,              // Auto-adjust KL penalty
    ..GrpoConfig::stable()
}
```

**Reused**: `GrpoOptimizer`, `GrpoConfig`, `SampleGroup`, `GrpoEvaluator` from `training/grpo.rs`.
**New**: `BitNetGrpoAdapter` that maps expert forward pass outputs to `GrpoSample` structs.

### AD-12: Contrastive Pre-Training for Expert Routing Validation

**Decision**: After ternary conversion of expert weights, use the existing `ContrastiveTrainer` to verify that MoE routing still selects the correct experts.

**Rationale**: Replacing expert FFN weights with ternary approximations changes the output distribution of each expert. If expert N's ternary output becomes more similar to expert M's output, the router may misroute tokens. Contrastive pre-training on expert embeddings detects and corrects this.

**Approach**:
1. For each token in a calibration set, record which expert the teacher model's router selects
2. Generate `TrainingTriplet`s: anchor = hidden state, positive = correct expert output, negative = wrong expert output
3. Use existing hard negative mining to find expert pairs that become confusable after ternary conversion
4. Fine-tune the FP16 router gating weights using contrastive loss to restore correct expert selection

**Reused**: `ContrastiveTrainer`, `ContrastiveConfig`, `TrainingTriplet` from `training/contrastive.rs`.
**New**: `ExpertTripletGenerator` that produces triplets from MoE routing decisions.

### AD-13: EWC++ Cross-Expert Stability During Sequential Distillation

**Decision**: Apply `EwcRegularizer` from `lora/training.rs` during sequential expert distillation to prevent catastrophic forgetting across experts.

**Rationale**: Distilling 30B MoE experts sequentially (expert 0, then 1, ..., then N) risks overwriting shared representations. EWC++ computes Fisher information diagonals for each expert's contribution to the shared attention layers, then regularizes subsequent expert distillation to not deviate from previously-learned important weights.

**Configuration**:
```rust
TrainingConfig {
    ewc_lambda: 5000.0,           // Higher than default (2000) for cross-expert stability
    fisher_decay: 0.995,           // Slower decay to preserve Fisher across expert phases
    quality_threshold: 0.5,        // Only learn from high-quality distillation samples
    lr_schedule: LearningRateSchedule::Cosine,
    warmup_steps: 500,             // Longer warmup for 30B scale
    ..Default::default()
}
```

**Concrete protection**:
- After distilling expert 0: compute Fisher diagonal `F_0` over validation set
- When distilling expert 1: add penalty `ewc_lambda/2 * Σ F_0_i * (w_i - w*_0_i)²`
- Accumulate: `F_cumulative = fisher_decay * F_prev + (1-fisher_decay) * F_new`

**Reused**: `EwcRegularizer`, `TrainingPipeline`, `TrainingConfig`, `FisherDiagonal` from `lora/training.rs`.
**New**: `SequentialExpertDistiller` that wraps `EwcRegularizer` across expert phases.

### AD-14: Policy Store for Per-Layer Ternary Scale Tracking

**Decision**: Extend `PolicyStore` with a new `PolicyType::TernaryScale` to persist per-block absmean scale distributions and learned quantization decisions.

**Rationale**: Not all layers quantize equally well to ternary. Attention layers may need different scale clipping than FFN layers. The policy store enables the distillation pipeline to learn and persist per-layer quantization strategies that can be retrieved and applied in future distillation runs or model updates.

**New policy type**:
```rust
pub enum PolicyType {
    Quantization,
    Router,
    Ewc,
    Pattern,
    TernaryScale,      // NEW: Per-layer ternary quantization metadata
}

pub struct TernaryScalePolicy {
    pub layer_idx: usize,
    pub module: String,              // "gate_proj", "up_proj", "down_proj", "q_proj", etc.
    pub mean_absmean: f32,           // Average scale factor across blocks
    pub std_absmean: f32,            // Variance in scale factors
    pub sparsity: f32,               // Fraction of zero weights
    pub quality_vs_teacher: f32,     // Cosine similarity to teacher output
    pub distillation_loss: f32,      // Final loss for this layer
    pub recommended_block_size: usize, // 256 default, may vary
}
```

**Reused**: `PolicyStore`, `PolicyEntry`, `PolicySource` from `policy_store.rs`.
**New**: `TernaryScalePolicy` struct and `PolicyType::TernaryScale` variant.

### AD-15: Memory Distillation for Training Quality Tracking

**Decision**: Log all distillation teacher-student comparisons as `Trajectory` objects in the `ReasoningBank`, enabling `MemoryDistiller` to extract `KeyLesson`s about which layers, experts, and configurations produce the best ternary quality.

**Rationale**: Distillation is iterative — understanding which experts converge quickly, which resist ternary conversion, and what scale distributions correlate with quality enables intelligent scheduling of future distillation runs.

**Mapping**:

| ReasoningBank Concept | Distillation Mapping |
|----------------------|---------------------|
| `Trajectory` | One expert's distillation run (N steps) |
| `Verdict` | `Success` if cosine_sim > 0.9, `Failure` if < 0.7 |
| `PatternCategory` | Expert index + layer type (e.g., "expert_3_gate_proj") |
| `KeyLesson` | "Expert 7 gate_proj converges fastest with lr=2e-6 and block_size=128" |
| `CompressedTrajectory` | Summary of entire expert distillation phase |

**Reused**: `MemoryDistiller`, `DistillationConfig`, `CompressedTrajectory`, `KeyLesson` from `reasoning_bank/distillation.rs`.
**New**: `DistillationTrajectoryRecorder` that adapts expert training steps to `Trajectory` format.

### AD-16: Distillation Pipeline Composition

**Decision**: Compose the full Craftsman Ultra distillation pipeline from existing RLM components wired through a new `CraftsmanDistiller` orchestrator.

**Pipeline architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│                  CraftsmanDistiller (NEW orchestrator)           │
│                                                                 │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│  │ TeacherModel  │───▶│BitLinearTrainer   │───▶│ GGUFExporter │ │
│  │(GLM-4.7-Flash)│    │(NEW: STE+shadow)  │    │(REUSED)      │ │
│  └───────┬───────┘    └────────┬─────────┘    └──────────────┘ │
│          │                     │                                │
│          │   ┌─────────────────┼─────────────────┐              │
│          │   │                 │                  │              │
│          ▼   ▼                 ▼                  ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │GrpoOptimizer │   │EwcRegularizer│   │ContrastiveTrainer│    │
│  │(REUSED)      │   │(REUSED)      │   │(REUSED)          │    │
│  │Per-expert    │   │Cross-expert  │   │Router validation │    │
│  │reward scaling│   │stability     │   │post-ternary      │    │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘    │
│         │                  │                     │              │
│         ▼                  ▼                     ▼              │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Quality Feedback Loop                    │      │
│  │                                                       │      │
│  │  MemoryDistiller ──▶ KeyLesson extraction             │      │
│  │  PolicyStore    ──▶ TernaryScale persistence          │      │
│  │  (BOTH REUSED)                                        │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘

Net-new code:  BitLinearTrainer (STE + shadow weights), CraftsmanDistiller (orchestrator)
Reused code:   GrpoOptimizer, EwcRegularizer, ContrastiveTrainer, MemoryDistiller,
               PolicyStore, GGUFExporter, TrainingConfig, LR schedules
Reuse ratio:   ~70% existing / ~30% new
```

**Optimization: Expert-Parallel Distillation**

Experts are independent during forward pass. Distill multiple experts concurrently across CPU cores:

```rust
// Distill experts in parallel (independent FFN weights)
let expert_results: Vec<DistillResult> = experts
    .par_iter()                          // rayon parallel iterator
    .enumerate()
    .map(|(idx, expert)| {
        let mut trainer = BitLinearTrainer::new(expert, &teacher_expert[idx]);
        let mut ewc = EwcRegularizer::new_with_fisher(cumulative_fisher[idx]);
        let mut grpo = GrpoOptimizer::new(GrpoConfig::stable());

        for batch in dataset.batches() {
            let student_out = trainer.forward_ternary(batch);
            let teacher_out = teacher.forward_expert(idx, batch);

            let reward = grpo.evaluate(&student_out, &teacher_out);
            let kd_loss = kd_loss_fn(&student_out, &teacher_out, alpha, temperature);
            let ewc_penalty = ewc.penalty(&trainer.shadow_weights());
            let total_loss = kd_loss * reward.scale() + ewc_penalty;

            trainer.backward_ste(total_loss);
        }

        ewc.update_fisher(&trainer);      // Update Fisher for next expert
        DistillResult { idx, weights: trainer.export_ternary(), fisher: ewc.fisher() }
    })
    .collect();
```

### AD-17: Training Infrastructure — Cloud GPU over Local SIMD

**Decision**: Use Google Cloud A100/H100 GPU instances for distillation training. Reserve local CPU/SIMD for inference validation, MicroLoRA adaptation, and GGUF export only.

**Rationale**: Local CPU/SIMD training is mathematically infeasible at the 200B+ token scale required for expert distillation. The existing RuvLLM SIMD kernels (`kernels/`) are inference-only — no backpropagation or gradient computation. The training code (`real_trainer.rs:178-184`) supports Metal (macOS) or CPU but not CUDA, and CPU throughput at ~50-100 tok/s training would require ~65 years for 200B tokens.

**Memory analysis (per-expert distillation):**

| Component | Size | Notes |
|-----------|------|-------|
| Single expert FFN shadow weights (FP16) | ~2 GB | ~1B params per expert (28B ÷ N experts) |
| Gradients (FP32) | ~4 GB | Full precision for STE backprop |
| AdamW optimizer state (2× FP32) | ~8 GB | First + second moment |
| Teacher activations cache | ~1 GB | Per-batch FP16 |
| EWC++ Fisher diagonal | ~0.5 GB | Per-expert accumulated |
| **Per-expert total** | **~15.5 GB** | Fits in A100 40GB with headroom |

**Full model simultaneous (Phase 2+):**

| Component | Size | Notes |
|-----------|------|-------|
| 30B shadow weights (FP16) | ~60 GB | Requires A100 80GB or H100 |
| Gradients + optimizer | ~360 GB | Requires multi-GPU parallelism |
| **Total** | **~430 GB** | 4× A100 80GB or 4× H100 80GB |

**Throughput and cost comparison:**

| Platform | Training tok/s | Time (200B tok, Phase 1) | Cost | Phase 0 PTQ? |
|----------|---------------|--------------------------|------|-------------|
| **Mac Studio M4 Max (Metal)** | ~500-1000 | ~6.5 years | N/A | **Yes — 1-4 hrs, $0** |
| **Mac Studio M3 Ultra (Metal)** | ~800-1500 | ~4.2 years | N/A | **Yes — 1-1.5 hrs, $0** |
| CPU AVX2 (Ryzen 9) | ~50-100 | ~65 years | N/A | Yes — 2-6 hrs, $0 |
| 1× A100 80GB (GCP on-demand) | ~15,000 | ~155 days | ~$3,700 | Yes — 30 min, ~$5 |
| 4× A100 80GB (GCP on-demand) | ~50,000 | ~46 days | ~$4,400 | Overkill for PTQ |
| 4× A100 80GB (GCP spot) | ~50,000 | ~46 days | **~$1,300** | Overkill for PTQ |
| 1× H100 (DataCrunch) | ~40,000 | ~58 days | ~$2,900 | Overkill for PTQ |
| 4× H100 (DataCrunch) | ~140,000 | ~16 days | **~$3,200** | Overkill for PTQ |

**Key insight**: Mac Studio is infeasible for Phase 1+ training (years of wall time) but **ideal for Phase 0 PTQ** (hours, $0). This separation justifies the phased approach.

**Recommended infrastructure per phase:**

| Phase | Instance | Duration | Estimated Cost | Strategy |
|-------|----------|----------|----------------|----------|
| **Phase 0 (PTQ)** | **Mac Studio (M4 Max/M3 Ultra)** | **1-4 hours** | **$0** | **Mmap FP16 weights → absmean quantize → export GGUF; Metal GPU for calibration pass** |
| Phase 0D (BitDistill Lite, 10B tok) | Mac Studio Metal or 1× A100 spot | 2-4 weeks (local) / 1-2 days (cloud) | $0 (local) / ~$300 (cloud) | Optional quality upgrade if Phase 0C too degraded |
| Phase 1 (expert FFN, 200B tok) | 4× A100 80GB spot (GCP) | ~46 days | $1,300-$2,000 | Per-expert sequential with EWC++; each expert fits 1 GPU |
| Phase 1 (router validation) | Mac Studio Metal or 1× A100 | ~2-4 hours | $0 (local) / <$10 (cloud) | Contrastive training on router only (~2B params) |
| Phase 2 (full ternary, 500B tok) | 4× H100 (DataCrunch) | ~16-32 days | $2,500-$5,000 | All layers; model-parallel across GPUs |
| Phase 3 (native training, 4T tok) | 8× H100 cluster | ~90-180 days | $15,000-$30,000 | Full from-scratch; depends on funding |
| Inference validation | Mac Studio (NEON) | Continuous | $0 | TL1/TL2 kernel testing on ARM NEON |
| MicroLoRA adaptation | Mac Studio | <1ms/update | $0 | Existing ndarray-based EWC++ pipeline |

**Required code change**: Add CUDA device dispatch to `RealContrastiveTrainer`:
```rust
// Current (real_trainer.rs:178-184):
let device = if config.use_metal {
    Device::new_metal(0).unwrap_or(Device::Cpu)
} else {
    Device::Cpu
};

// Required for cloud GPU training:
let device = if config.use_cuda {
    Device::new_cuda(config.cuda_device_id).unwrap_or(Device::Cpu)
} else if config.use_metal {
    Device::new_metal(0).unwrap_or(Device::Cpu)
} else {
    Device::Cpu
};
```

This is a single-line addition to `RealTrainingConfig` (`use_cuda: bool`, `cuda_device_id: usize`) and a 3-line change to device selection. The rest of the Candle training pipeline (tensors, optimizer, loss computation) works identically across CPU/Metal/CUDA.

**Cost optimization strategies:**
1. **Spot instances**: GCP A100 spot at ~$1/GPU-hr (70% off on-demand) — requires checkpointing every 30 min
2. **DataCrunch / Lambda Labs**: H100 at $1.99-$2.10/hr (40-50% below GCP on-demand)
3. **Expert-sequential on fewer GPUs**: Distill 1 expert at a time on 1× A100 80GB (~$1.50/hr), increasing wall time but reducing per-hour cost
4. **Mixed precision training**: FP16 shadow weights + BF16 activations reduces memory, enabling smaller instances
5. **Gradient checkpointing**: Trade compute for memory to fit on fewer GPUs

### AD-18: Phase 0 — PT-BitNet Post-Training Quantization on Mac Studio

**Decision**: Implement a PT-BitNet ternary post-training quantizer as Phase 0, running entirely on a local Mac Studio, producing a rapid prototype GGUF for inference pipeline validation before investing in full distillation.

**Rationale**: The original Option A ("Rejected") assumed only generic IQ1_S quantization, which produces garbled outputs at 1.56 bpw. However, PT-BitNet (2025) demonstrates that applying BitNet's native absmean ternary quantization to pre-trained weights with calibration data achieves significantly better results (61% downstream at 70B) than generic codebook PTQ. This produces genuine BitNet ternary format that enables multiplication-free inference with TL1/TL2 kernels — unlike IQ1_S which still requires dequant-then-multiply.

**Target platform: Mac Studio (Apple Silicon)**

Phase 0 is pure quantization (no training loop), making it ideal for local execution on Mac Studio:

| Config | Unified RAM | FP16 Load | PTQ? | Calibration? | Notes |
|--------|------------|-----------|------|-------------|-------|
| M4 Max 36GB | 36 GB | mmap (demand-paged) | **Yes** | Slow (paging) | Minimum viable; mmap means only active tensor pages in RAM |
| M4 Max 64GB | 64 GB | Fits with mmap assist | **Yes** | **Yes** | Comfortable for PTQ; calibration may page |
| M4 Max 128GB | 128 GB | Fits entirely | **Yes** | **Yes** | Ideal — FP16 model (60GB) + ternary output (7GB) + calibration buffers all in RAM |
| M3 Ultra 96GB | 96 GB | Fits entirely | **Yes** | **Yes** | Good headroom |
| M3 Ultra 192GB+ | 192+ GB | Fits entirely | **Yes** | **Yes** | Ample room for full model + calibration + inference validation |

**Why Mac Studio works for Phase 0 (but not Phase 1+):**
- **PTQ is not training**: No gradient computation, no optimizer state, no backpropagation — just load → quantize → export
- **Memory-mapped I/O**: FP16 weights can be mmap'd from disk; only the current tensor's pages need to be in RAM
- **Per-tensor processing**: Quantize one tensor at a time (read FP16 block → compute absmean → round to ternary → write output) — working memory is ~2-4 MB per tensor regardless of total model size
- **Metal GPU for calibration**: RuvLLM's existing `RealContrastiveTrainer` and `kernels/matmul.rs` support Metal via Candle (`use_metal: true` default, 3x speedup on M4 Pro GEMV)
- **ARM NEON for TL1 kernels**: Mac Studio's Apple Silicon has NEON SIMD — the same target ISA as the TL1 kernel for ternary inference validation
- **Phase 1 still needs cloud GPU**: 200B token distillation at ~500-1000 tok/s (Metal) = ~6.5 years locally vs ~46 days on 4× A100

**Estimated Phase 0 wall time on Mac Studio:**

| Step | M4 Max 128GB | M4 Max 64GB | M3 Ultra 192GB |
|------|-------------|-------------|----------------|
| Download GLM-4.7-Flash FP16 (~60GB) | ~30 min (1Gbps) | ~30 min | ~30 min |
| Absmean ternary quantization | ~5-15 min | ~10-30 min (paging) | ~5-10 min |
| Calibration pass (1000 samples, Metal) | ~30-60 min | ~60-120 min | ~20-40 min |
| GGUF export | ~2-5 min | ~2-5 min | ~2-5 min |
| TL1 kernel validation inference | ~10-20 min | ~10-20 min | ~10-20 min |
| **Total** | **~1-2 hours** | **~2-4 hours** | **~1-1.5 hours** |

**Implementation approach**:

```
Phase 0 Pipeline (runs on Mac Studio):
  1. Load GLM-4.7-Flash FP16/BF16 weights via mmap
  2. For each linear layer in expert FFNs:
     a. Compute gamma = mean(|W|)  (absmean scale)
     b. W_ternary = RoundClip(W / (gamma + epsilon), -1, 1)
     c. Store: 2-bit packed ternary weights + FP16 scale per block
  3. Calibration pass (optional, improves quality, uses Metal GPU):
     a. Run ~1000 calibration samples through teacher model
     b. Record activation statistics per layer
     c. Optimize scale factors to minimize MSE between teacher and ternary outputs
  4. Export to GGUF with BITNET_T158 tensor type + metadata
  5. Validate: load in BitNetBackend → TL1 NEON kernel → generate tokens
```

**Absmean ternary quantizer (core algorithm)**:
```
Input:  W ∈ R^{m×n} (FP16 weight matrix)
Output: W_t ∈ {-1,0,+1}^{m×n}, scale ∈ R (per-block FP16)

For each block of 256 elements:
  1. gamma = mean(|block|) + 1e-8
  2. normalized = block / gamma
  3. ternary = round(clamp(normalized, -1, 1))  → {-1, 0, +1}
  4. Pack: 2 bits per weight (00=-1, 01=0, 10=+1)
  5. Store scale = gamma as FP16
```

**What stays FP16** (same as AD-2):
- MoE router gating weights
- Token embeddings + LM head
- RoPE frequencies
- LayerNorm/RMSNorm parameters

**RuvLLM implementation gaps to fill**:

| Gap | Effort | Details |
|-----|--------|---------|
| Absmean ternary quantizer | ~200-300 lines | New function in `gguf/quantization.rs` or new module |
| IQ1_S / BITNET_T158 dequantization | ~80-120 lines | Add to `dequantize_tensor` match arm (currently falls to error at line 358) |
| GGUF export with ternary metadata | ~100-150 lines | Extend `GgufExportResult` with BitNet metadata keys from AD-5 |
| TL1 kernel smoke test | ~200 lines | Validate ternary GEMM produces correct output on PTQ model |

**Total new code**: ~600-800 lines (vs ~15,000+ for Phase 1 full distillation pipeline)

**Quality expectations (conservative estimates for GLM-4.7-Flash 30B-A3B)**:

| Benchmark | FP16 Baseline | Phase 0 PTQ (est.) | Phase 1 Distill (est.) |
|-----------|--------------|-------------------|----------------------|
| HumanEval pass@1 | ~65% | ~35-45% | ~55-60% |
| MMLU | ~75% | ~45-55% | ~65-70% |
| SWE-bench Verified | 59.2% | ~25-35% | ~50-55% |
| LiveCodeBench v6 | 64.0% | ~30-40% | ~55-60% |

**Why Phase 0 quality is still useful**:
1. **Kernel validation**: Ternary GEMM correctness doesn't depend on model quality
2. **Memory profiling**: Real-world memory usage measurement with actual MoE activation patterns
3. **Throughput benchmarking**: Measure real tok/s with TL1/TL2/I2_S kernels on target hardware
4. **Pipeline testing**: End-to-end GGUF load → inference → token output
5. **Baseline measurement**: Quantitative quality floor establishes improvement target for Phase 1
6. **Cost**: $0 on Mac Studio vs ~$1,300 for Phase 1 — validates infrastructure at zero cost before committing to cloud GPU

**Key configuration**:
```rust
pub struct PtBitnetConfig {
    pub calibration_samples: usize,     // 1000 default (WikiText-2 or code corpus)
    pub block_size: usize,              // 256 (matches AD-1)
    pub optimize_scales: bool,          // true: MSE-optimized scales; false: raw absmean
    pub layers_to_quantize: LayerMask,  // ExpertsOnly (Phase 0) or All (future)
    pub export_format: TernaryFormat,   // BitnetT158 (native) or IQ1S (llama.cpp compat)
    pub router_precision: Precision,    // FP16 (always, per AD-2)
    pub use_mmap: bool,                 // true: memory-map FP16 weights (required for <128GB systems)
    pub use_metal_calibration: bool,    // true: Metal GPU for calibration pass (Mac Studio)
    pub max_memory_gb: Option<f32>,     // Cap memory usage; enables streaming quantization
}
```

**Reused**: GGUF parser, tensor metadata, `GgufQuantType` enum, export pipeline.
**New**: `PtBitnetQuantizer`, `absmean_ternary()`, `BITNET_T158` dequantization kernel.

---

## Consequences

### Positive

1. **CPU-only deployment**: 30B-class model running on commodity hardware without GPU
2. **Energy efficiency**: 55-82% reduction in inference energy vs FP16
3. **Memory efficiency**: ~8GB vs ~60GB for FP16 30B model (7.5x reduction)
4. **Multiplication-free expert GEMM**: Integer addition only in expert forward passes
5. **SONA compatibility**: MicroLoRA adaptation preserves per-session learning
6. **GGUF ecosystem**: Compatible with existing model distribution infrastructure
7. **Incremental path**: Phase 0 validates at ~$100; Phase 1 delivers quality; Phases 2-3 optimize
8. **~70% RLM code reuse**: GRPO, EWC++, ContrastiveTrainer, MemoryDistiller, PolicyStore are production-tested — only BitLinear layer and orchestrator are net-new
9. **Adaptive distillation**: GRPO reward scaling dynamically focuses compute on hard-to-distill experts
10. **Cross-expert stability**: EWC++ Fisher diagonal prevents catastrophic forgetting during sequential expert distillation
11. **Learned quantization policies**: PolicyStore persists per-layer ternary scale distributions for reproducible future distillation runs
12. **Expert-parallel distillation**: Independent expert FFNs enable rayon-parallel distillation across CPU cores
13. **Phase 0 de-risks Phase 1 at zero cost**: Mac Studio PTQ prototype validates entire inference pipeline (GGUF → dequant → kernel → MoE → generation) for $0 before committing $1,300+ to cloud GPU distillation
14. **Existing GGUF ecosystem**: Community-published GLM-4.7-Flash GGUFs (bartowski, unsloth) available as comparison baselines

### Negative

1. **Training cost**: Even distillation requires 800-1,600 A100-hours (~$2K-$5K cloud cost)
2. **Custom kernels**: Must implement and maintain platform-specific SIMD kernels in Rust
3. **Quality gap**: Phase 1 may be 5-10% below GLM-4.7-Flash on some benchmarks
4. **No GPU acceleration**: BitNet kernels are CPU-specific; GPU path requires separate optimization
5. **Mixed-precision complexity**: Router (FP16) + experts (ternary) + attention (FP16/ternary) adds dispatch complexity
6. **WASM limitation**: Ternary lookup table kernels may not translate efficiently to WASM SIMD
7. **RLM scale gap**: Existing `RealContrastiveTrainer` targets 0.5B models (embedding_dim=896); scaling to 30B requires distributed data loading and increased batch sizes

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase 0 PTQ quality too low for meaningful testing | Medium | Low | Phase 0 is for kernel/pipeline validation, not quality; upgrade to 0D (BitDistill Lite) if needed |
| MoE routing degrades with ternary experts | Medium | High | Phase 0 detects routing issues early; Phase 1 validates routing; router stays FP16; AD-12 contrastive validation |
| bitnet.cpp kernel translation to Rust introduces bugs | Medium | Medium | Phase 0 PTQ model provides cheap test fixture; extensive kernel unit tests; validate against reference impl |
| Distillation fails to converge for MoE | Low | High | GRPO reward scaling + per-expert distillation fallback; EWC++ stability (AD-13) |
| GLM-4.7-Flash architecture changes break compatibility | Low | Medium | Pin to specific HF revision; architecture abstraction layer |
| IQ1_S GGUF format insufficient for absmean metadata | Medium | Low | Register custom GGUF type (BITNET_T158); backward-compatible extension |
| EWC++ Fisher accumulation OOM at 30B scale | Medium | Medium | Sparse Fisher (top-k diagonal entries); per-expert rather than global Fisher |
| GRPO reward signal too noisy for distillation | Low | Low | Fall back to static KD loss; GRPO reward as optional multiplier |
| `RealContrastiveTrainer` doesn't scale to 30B | Medium | Medium | Extract training loop; replace Candle Linear with BitLinear; keep optimizer/scheduler |
| Calibration data bias in Phase 0 PTQ | Low | Low | Use diverse calibration corpus (WikiText + code); measure variance across calibration sets |

---

## Validation Criteria

### Phase 0 Exit Criteria
- [ ] Absmean ternary quantizer produces valid {-1, 0, +1} weights from GLM-4.7-Flash FP16
- [ ] Quantization runs successfully on Mac Studio via mmap (no cloud GPU required)
- [ ] GGUF export with BITNET_T158 tensor type loads without error in BitNetBackend
- [ ] TL1 NEON kernel produces non-zero, bounded output on PTQ ternary weights
- [ ] MoE routing selects experts (not all-zero or all-same-expert degenerate routing)
- [ ] End-to-end token generation produces coherent (if degraded) text
- [ ] Memory usage measured and documented for real MoE activation patterns
- [ ] Throughput measured: tok/s on Mac Studio (ARM NEON) and optionally x86 AVX2
- [ ] Baseline quality benchmarks recorded (HumanEval, MMLU) as Phase 1 improvement target
- [ ] Total Phase 0 cost = $0 (local Mac Studio execution)

### Phase 1 Exit Criteria
- [ ] BitNet backend loads GGUF with ternary expert weights
- [ ] TL1 kernel produces bit-exact output vs reference float implementation
- [ ] Decode speed >= 5 tok/s on x86_64 AVX2 (AMD Ryzen 7 / Intel i7 class)
- [ ] HumanEval pass@1 >= 50% (GLM-4.7-Flash baseline: ~65%)
- [ ] Memory usage < 10GB for 4K context inference
- [ ] GRPO-guided expert distillation converges (loss < 0.5 for all experts)
- [ ] EWC++ prevents cross-expert interference (Fisher-regularized loss delta < 5%)
- [ ] Contrastive router validation: >= 95% expert routing accuracy vs teacher
- [ ] PolicyStore contains TernaryScale entries for all distilled expert layers

### Phase 2 Exit Criteria
- [ ] Full ternary model (attention + experts) running on CPU
- [ ] Decode speed >= 8 tok/s on x86_64 AVX2
- [ ] SWE-bench Verified >= 52% (90%+ of GLM-4.7-Flash's 59.2%)
- [ ] SONA MicroLoRA adaptation functional on ternary base
- [ ] MemoryDistiller has extracted >= 50 KeyLessons from distillation trajectories
- [ ] GRPO adaptive KL stabilizes below kl_target (0.02) for all experts

### Phase 3 Exit Criteria
- [ ] Native-trained model matches or exceeds GLM-4.7-Flash benchmarks
- [ ] Published on HuggingFace (ruv/craftsman-ultra-30b-1bit)
- [ ] GGUF + bitnet kernel distributed via npm/packages/ruvllm
- [ ] Full distillation pipeline reproducible from PolicyStore policies (no manual tuning)

---

## References

1. Ma, S. et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (arXiv:2402.17764, Feb 2024)
2. Ma, S. et al., "BitNet b1.58 2B4T Technical Report" (arXiv:2504.12285, Apr 2025)
3. Microsoft Research, "bitnet.cpp: Efficient Edge Inference for Ternary LLMs" (arXiv:2502.11880, Feb 2025)
4. Microsoft, bitnet.cpp — https://github.com/microsoft/BitNet
5. Zhipu AI, GLM-4.7-Flash — https://huggingface.co/zai-org/GLM-4.7-Flash
6. Zhipu AI, "GLM-4.7: Advancing the Coding Capability" — https://z.ai/blog/glm-4.7
7. RuvLLM ADR-002: RuvLLM Integration with Ruvector
8. RuvLLM GGUF Quantization Module: `crates/ruvllm/src/gguf/quantization.rs`
9. Microsoft, bitnet-b1.58-2B-4T-gguf — https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
10. RuvLLM GRPO Implementation: `crates/ruvllm/src/training/grpo.rs`
11. RuvLLM RealContrastiveTrainer: `crates/ruvllm/src/training/real_trainer.rs`
12. RuvLLM EWC++ Training Pipeline: `crates/ruvllm/src/lora/training.rs`
13. RuvLLM Memory Distillation: `crates/ruvllm/src/reasoning_bank/distillation.rs`
14. RuvLLM Policy Store: `crates/ruvllm/src/policy_store.rs`
15. RuvLLM Contrastive Training: `crates/ruvllm/src/training/contrastive.rs`
16. PT-BitNet: "Scaling up the 1-Bit large language model with post-training quantization" (2025) — https://www.sciencedirect.com/science/article/abs/pii/S089360802500735X
17. BitDistill: "BitNet Distillation" (arXiv:2510.13998, Oct 2025) — https://arxiv.org/html/2510.13998v1
18. bartowski, GLM-4.7-Flash-GGUF quantizations — https://huggingface.co/bartowski/zai-org_GLM-4.7-Flash-GGUF
19. unsloth, GLM-4.7-Flash-GGUF dynamic quantizations — https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF
20. llama.cpp IQ1_S blind testing (Discussion #5962) — https://github.com/ggml-org/llama.cpp/discussions/5962
21. STBLLM: "Breaking the 1-bit Barrier" (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf
22. Apple Mac Studio Technical Specifications (2025) — https://www.apple.com/mac-studio/specs/
23. RuvLLM Metal GEMV integration: `crates/ruvllm/src/kernels/matmul.rs:1444-1582`
