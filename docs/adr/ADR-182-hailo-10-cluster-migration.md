---
adr: 182
title: "Migrate cluster from Hailo-8 (vision NPU) to Hailo-10H (LLM-native NPU)"
status: proposed
date: 2026-05-05
authors: [ruvnet, claude-flow]
related: [ADR-179, ADR-180, ADR-181, ADR-176, ADR-167]
hardware: yes
---

# ADR-182 — Hailo-10H migration for the Pi 5 cluster

## Status

**Proposed (hardware-spend ADR).** Software ADRs (180, 181) max
out around ~80-100 tok/s aggregate on the existing 4× Hailo-8
cluster. ADR-182 trades hardware for the next jump: Hailo-10H
is purpose-built for transformer/LLM workloads, with onboard
8 GB DDR4 — eliminating the LPDDR4X memory-bandwidth ceiling
that bounds the entire current stack.

## Context

The current cluster (ADR-179 SOTA = 20.5 tok/s aggregate) is
**memory-bandwidth bound**, not compute bound:

- Pi 5 LPDDR4X: ~16 GB/s shared between CPU + onboard WiFi + AI HAT+
- Q4_K_M TinyLlama-1.1B: ~640 MB weights moved through that bandwidth
  per generated token
- Floor at ~640 MB / 16 GB/s = ~40 ms/token = ~25 tok/s/Pi theoretical
- Real ceiling 9 tok/s/Pi due to dequant overhead + cache effects

The Hailo-8 currently in each AI HAT+ is a **vision encoder NPU**:

- 26 TOPS INT8
- Requires HEF-compiled fixed-shape graphs
- Decoder generation (dynamic KV-cache reshape) does not compile
- Used in our cluster only for embeddings (BERT-6 encoder), not LLM decode

Hailo-10H is the LLM-native successor:

| | Hailo-8 (current) | Hailo-10H |
|---|---|---|
| Compute | 26 TOPS INT8 | ~40 TOPS INT8, first-class INT4 |
| Onboard memory | none (uses host LPDDR) | **8 GB LPDDR4 on-chip** |
| Dynamic shapes | poor compiler support | designed for KV-cache decode |
| Form factor | M.2 2242 | M.2 2280 (drop-in for AI HAT+) |
| Power | ~2.5 W | ~5-7 W |
| Price (2026) | $70-99 | $249-299 |

## Decision

Order 4 × Hailo-10H modules (~$1k total) as soon as procurement
opens. On arrival:

1. Verify Pi 5 + AI HAT+ delivers the higher per-slot power budget
   (5-7 W vs the H8's 2.5 W). If not, Pi 5 PSU upgrade may be
   needed (check the AI HAT+ rev; some early units cap at ~3 W).
2. Replace H8 in one Pi (cluster-1) first as a feasibility test.
3. Verify Hailo's compiler accepts a small Llama-class block
   (just one decoder layer) as a proof-of-compile.
4. If yes, compile full TinyLlama-1.1B → Hailo `.hef`.
5. Stage on cluster-1, bench against the Q4_K_M baseline.
6. Roll out to remaining 3 Pis if delta is ≥3×.

### What ADR-182 unlocks

| Workload | Hailo-8 cluster (today) | Hailo-10H cluster (projected) | Improvement |
|---|---:|---:|---:|
| Embedding (encoder, ADR-176) | 70 RPS/Pi NPU-bound | 100-150 RPS/Pi | 1.5-2× |
| LLM decode 1B-class | 9 tok/s/Pi (CPU + Q4_K_M) | 50-100 tok/s/Pi | 5-10× |
| LLM decode 8B-class | not feasible (Pi RAM cap) | 15-30 tok/s/Pi | new capability |
| Aggregate 4-Pi 1B | 20.5 tok/s | **~150-300 tok/s** | 7-15× |

## Risks

| Risk | Mitigation |
|---|---|
| **Hailo compiler doesn't support full LLM graph** | Phase: compile single block first, then fall back to "decoder-on-NPU + tokenizer/embed-on-CPU" hybrid if full graph fails |
| **AI HAT+ M.2 keying or power delivery incompatible** | Verify with Raspberry Pi Foundation before purchase; test cluster-1 first |
| **HailoRT version skew vs ADR-176's HailoRT 4.23** | Run two HailoRT versions in parallel during transition; embed worker on 4.23, LLM worker on whatever H10 needs |
| **Quantization scheme on Hailo-10 differs from our pi_quant** | Hailo natively supports INT4 + INT8 + their own ternary; pi_quant won't translate directly. ADR-181 work becomes redundant for the LLM path. |
| **$1k spend with no software-only fallback if it doesn't work** | Software ADRs 180 + 181 still on the H8 hardware (~80-100 tok/s aggregate target). Refund/return path on the H10s if compile fails. |

## Procurement

- **Vendor**: Hailo direct or via Sparkfun/MakerFocus distributors
- **SKU**: `Hailo-10H M.2 2280` (8 GB onboard variant)
- **Quantity**: 4 (one per cluster Pi)
- **Budget**: ~$1,000 + tax/shipping
- **Lead time**: ~2026 H1; check current availability before ordering

## Acceptance

ADR-182 ships when:

- [ ] 4 × Hailo-10H modules installed in the cluster
- [ ] Embedding worker (ADR-176) still serves on `:50051` at
      ≥70 RPS/Pi (regression gate)
- [ ] LLM worker on `:50053` runs TinyLlama-1.1B on the H10
      with **≥30 tok/s/Pi** (3× the ADR-179 SOTA)
- [ ] Aggregate 4-Pi LLM throughput **≥120 tok/s** (6× ADR-179)
- [ ] Power draw under 35 W cluster-total (vs 28 W on H8)

## Tracking

- Out-of-band ADR — no branch yet, hardware-blocked
- Procurement decision lives with the user; this ADR captures the
  expected technical envelope so the spend is justified
