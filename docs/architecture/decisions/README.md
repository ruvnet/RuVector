# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for ruvector's core technical decisions.

## Decision Index

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [ADR-001](ADR-001-core-simd-strategy.md) | Core SIMD Optimization Strategy | Accepted | AVX-512/AVX2/NEON with runtime detection for 2.5x-18x speedup |
| [ADR-002](ADR-002-hyperbolic-embeddings.md) | Hyperbolic Embeddings for Hierarchical Data | Accepted | Poincare ball model for low-distortion tree embeddings |
| [ADR-003](ADR-003-mcp-protocol.md) | MCP Server Architecture and Transport | Accepted | JSON-RPC 2.0 over stdio/SSE/WebSocket for AI agent integration |
| [ADR-004](ADR-004-rvf-format.md) | RVF Cognitive Container Format | Accepted | Self-describing vector format with WASM kernels |
| [ADR-005](ADR-005-cross-platform-bindings.md) | Cross-Platform Bindings (WASM + NAPI-RS) | Accepted | WASM-first with native acceleration via NAPI-RS |
| [ADR-006](ADR-006-sona-adaptation.md) | SONA Self-Optimizing Architecture | Accepted | Online learning for automatic parameter tuning |
| [ADR-007](ADR-007-differential-privacy.md) | Differential Privacy and Epsilon-Budget | Accepted | Privacy budget management for collective learning |
| [ADR-008](ADR-008-flash-attention.md) | Flash Attention Implementation | Accepted | Memory-efficient O(N) attention with block-sparse patterns |

## ADR Template

When creating new ADRs, use this structure:

```markdown
# ADR-XXX: Title

| Field | Value |
|-------|-------|
| **Status** | Proposed / Accepted / Deprecated / Superseded |
| **Date** | YYYY-MM-DD |
| **Authors** | Names |
| **Reviewers** | Names |
| **Supersedes** | ADR-XXX (if applicable) |
| **Related** | ADR-XXX, ADR-YYY |

## 1. Context
[Problem statement and background]

## 2. Decision
[The decision and its justification]

## 3. Rationale
[Why this decision over alternatives]

## 4. Consequences
[Benefits, costs, and tradeoffs]

## 5. Implementation
[Key implementation details]

## 6. Related Decisions
[Links to related ADRs]

## 7. References
[External references]

## 8. Revision History
[Change log]
```

## Decision Categories

### Core Performance
- **ADR-001**: SIMD optimization strategy (AVX-512/AVX2/NEON)
- **ADR-008**: Flash Attention for LLM inference

### Data Representation
- **ADR-002**: Hyperbolic embeddings (Poincare ball model)
- **ADR-004**: RVF cognitive container format

### System Architecture
- **ADR-003**: MCP protocol and transport choices
- **ADR-005**: Cross-platform bindings (WASM + NAPI-RS)
- **ADR-006**: SONA self-optimization
- **ADR-007**: Differential privacy architecture

## Quick Reference

| Topic | ADR | Key Decision |
|-------|-----|--------------|
| Vector distance | ADR-001 | Hand-optimized SIMD for 2.5-18x speedup |
| Hierarchical data | ADR-002 | Poincare ball embeddings with c=1.0 default |
| AI agent protocol | ADR-003 | MCP with stdio (local) and SSE (remote) |
| File format | ADR-004 | RVF with 15 segment types, ML-DSA-65 signatures |
| Node.js bindings | ADR-005 | NAPI-RS native with WASM fallback |
| Self-optimization | ADR-006 | Temperature tracking, pattern learning, adaptive ef |
| Privacy | ADR-007 | Gaussian mechanism, epsilon-budget, PII stripping |
| Long context | ADR-008 | Flash Attention with 128K+ support |

## Related Documentation

For more detailed ADRs, see the main `/docs/adr/` directory which contains:
- ADR-001 through ADR-088+ covering specific implementation decisions
- Coherence Engine ADRs in `/docs/adr/coherence-engine/`
- Project-specific ADRs in various crate directories

## Status Definitions

| Status | Meaning |
|--------|---------|
| **Proposed** | Under discussion, not yet implemented |
| **Accepted** | Decision made, implementation in progress or complete |
| **Deprecated** | No longer recommended, retained for history |
| **Superseded** | Replaced by a newer ADR |

## Implementation Status

| ADR | Crate | Status |
|-----|-------|--------|
| ADR-001 | `ruvector-core` | Complete |
| ADR-002 | `ruvector-core`, `ruvector-hyperbolic-hnsw` | Complete |
| ADR-003 | `mcp-brain`, `ruvector-cli` | Complete |
| ADR-004 | `rvf`, `rvf-kernel` | Complete |
| ADR-005 | `ruvector-wasm`, `ruvector-node` | Complete |
| ADR-006 | `ruvector-core` | Complete |
| ADR-007 | `pi-brain` | Complete |
| ADR-008 | `ruvllm` | Complete |
