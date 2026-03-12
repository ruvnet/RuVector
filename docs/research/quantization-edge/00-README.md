# Ultra-Low-Bit Quantization & Edge Deployment Research

Research collection on ultra-low-bit compression, quantization-aware training (QAT),
and practical deployment pathways for large language models at 2-bit precision.

Conducted March 2026 in the context of ruvLLM — the Rust-native LLM inference
runtime within the RuVector ecosystem.

## Documents

| # | Document | Focus |
|---|----------|-------|
| 01 | [Ultra-Low-Bit Quantization Survey](01-ultra-low-bit-quantization-survey.md) | Landscape of sub-4-bit quantization methods, ICLR'26 results, and practical viability |
| 02 | [Quantization-Aware Training (QAT)](02-quantization-aware-training-qat.md) | Two-stage reasoning-oriented QAT, teacher-guided distillation, calibration strategies |
| 03 | [QuIP: 2-Bit LLM Quantization](03-quip-2bit-framework.md) | Incoherence processing, adaptive rounding, Cornell/RelaxML framework analysis |
| 04 | [MoE Memory-Aware Routing](04-moe-memory-aware-routing.md) | Expert routing with long-term memory, SRAM-budget mapping, micro-MoE for edge |
| 05 | [ruvLLM Quantization Architecture Review](05-ruvllm-quantization-architecture.md) | Deep analysis of existing ruvLLM quantization stack — BitNet, K-quants, GGUF, KV cache |
| 06 | [Implementation Plan: 2-Bit QAT in Rust](06-implementation-plan-rust-ruvllm.md) | Concrete Rust implementation plan using ruvLLM crates for 2-bit QAT and edge deployment |
| 07 | [3-Int Pi-Constant Quantization](07-3int-pi-constant-quantization.md) | Novel irrational-scaling quantization using pi for non-uniform grids, spectral preservation, and harmonic error reduction |

## Key Findings

1. **2-bit weight quantization is now practical** — ICLR'26 results show reasoning-oriented
   QAT preserves >90% of full-precision reasoning capability at 2-bit precision.

2. **ruvLLM already has strong foundations** — BitNet b1.58 (ternary), K-quant pipeline
   (Q4_K_M through Q2_K), GGUF I-quant support (IQ1_S, IQ2_XXS), and a two-tier
   KV cache provide most building blocks for 2-bit deployment.

3. **The gap is QAT integration** — ruvLLM currently supports post-training quantization
   but lacks a quantization-aware training loop that propagates gradients through
   quantized weights during fine-tuning.

4. **MoE routing + quantization is the frontier** — Combining memory-aware expert routing
   with per-expert mixed-precision quantization enables micro-MoE architectures that
   fit within edge SRAM budgets.

5. **Pi-constant scaling improves low-bit grids** — Using irrational scaling factors
   (pi/k) for quantization grids reduces spectral distortion by ~3 dB vs uniform grids
   at 3-bit, effectively gaining ~0.5 bits of precision for attention-heavy layers.

## Relationship to Existing Crates

```
ruvllm/src/quantize/        <- K-quant pipeline (Q4_K_M, Q5_K_M, Q8_0)
ruvllm/src/bitnet/          <- BitNet b1.58 ternary (2-bit packing)
ruvllm/src/gguf/            <- GGUF format with 30+ quant types incl. IQ1_S, IQ2_XXS
ruvllm/src/kv_cache.rs      <- Two-tier FP16+Q4 KV cache
ruvllm/src/lora/            <- MicroLoRA & adapter management
ruvllm/src/training/        <- GRPO, contrastive learning, dataset generation
ruvllm/src/sona/            <- SONA three-tier learning with EWC++
ruvector-core/              <- Vector storage with product/scalar quantization
```

## References

- ICLR 2026: "Reasoning-Oriented QAT for 2-Bit LLMs" (two-stage calibration + teacher fine-tuning)
- QuIP (Cornell/RelaxML): Incoherence processing for 2-bit LLM quantization
- LLM-QAT (Meta): Reusable QAT training loop with KV-cache quantization
- ParetoQ: Multi-objective ultra-low-bit quantization
- Memory-Aware MoE Routing: Long-term expert preference modeling
- BitNet b1.58 (Microsoft Research): Ternary weight quantization
- Pi-Constant Quantization: Irrational scaling factors for non-uniform quantization grids
- Logarithmic Quantization (NF4/NF3): Distribution-matched non-uniform grids (QLoRA)
- Harmonic Quantization Grids: Signal-processing-inspired spectral compression
