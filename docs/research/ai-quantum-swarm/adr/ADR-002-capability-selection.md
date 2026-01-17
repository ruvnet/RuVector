# ADR-002: Capability Selection Criteria

**Status**: Accepted
**Date**: 2026-01-17
**Deciders**: Research Team

## Context

We identified many potential AI-quantum capabilities. We need criteria to prioritize which 7 capabilities to deeply research.

## Decision

We will use a **weighted scoring matrix** with the following criteria:

### Selection Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Novelty** | 15% | Is this genuinely new? Not just AI + quantum separately |
| **AI-Quantum Synergy** | 20% | Does combining AI and quantum create emergent value? |
| **Technical Feasibility** | 20% | Achievable within 1-2 years with current technology |
| **RuVector Integration** | 15% | Leverages existing crates (ruQu, mincut, attention) |
| **Real-World Impact** | 15% | Addresses healthcare, finance, security applications |
| **Verification Path** | 15% | Falsifiable tests, reproducible benchmarks, external signals |

### Research Foundation Gate

A capability **must** meet the following before it can be Tier 1 or Tier 2:
- At least 3 primary sources from the last 24 months
- At least 1 source must include open code, open data, or a clearly reproducible method
- If the gate fails, the capability is Tier 3 by default

### Scoring Consistency

Each criterion must include anchor rubrics:

| Score | Novelty | Synergy | Feasibility |
|-------|---------|---------|-------------|
| **High (13-15)** | New mechanism or theorem-level idea. Clear delta vs prior art. | Quantum and AI create value neither can do alone. | Prototype in 6 weeks, usable in 12-18 months. |
| **Mid (7-12)** | Known ideas combined in a new way. Moderate delta. | Some benefit, could be matched classically with effort. | Prototype in 12 weeks, usable in 18-24 months. |
| **Low (1-6)** | Mostly standard with minor changes. | Two parallel parts without emergent gain. | Blocked by hardware, data, or theory. |

| Score | Integration | Impact | Verification Path |
|-------|-------------|--------|-------------------|
| **High (13-15)** | Direct reuse of existing crates. Minimal new primitives. | Clear buyer, workflow, measurable win. | Falsifiable, benchmarkable, independent external signals exist. |
| **Mid (7-12)** | Some reuse, requires new core types. | Plausible value, unclear adoption path. | Benchmarks exist but weak falsifiability or single source. |
| **Low (1-6)** | Mostly standalone. | Interesting but speculative. | Hard to test, mostly narrative. |

### Scoring Matrix (Revised)

| Capability | Novelty | Synergy | Feasible | Integrate | Impact | Verify | **Total** | Gate |
|------------|---------|---------|----------|-----------|--------|--------|-----------|------|
| **NQED** | 14/15 | 19/20 | 18/20 | 15/15 | 13/15 | 14/15 | **93** | PASS |
| **AV-QKCM** | 13/15 | 18/20 | 19/20 | 15/15 | 12/15 | 15/15 | **92** | PASS |
| **QGAT-Mol** | 12/15 | 18/20 | 17/20 | 13/15 | 14/15 | 14/15 | **88** | PASS |
| **QEAR** | 15/15 | 19/20 | 13/20 | 12/15 | 13/15 | 10/15 | **82** | PASS |
| **QFLG** | 11/15 | 16/20 | 16/20 | 14/15 | 14/15 | 12/15 | **83** | PASS |
| **VQ-NAS** | 13/15 | 15/20 | 12/20 | 13/15 | 11/15 | 10/15 | **74** | PASS |
| **QARLP** | 10/15 | 14/20 | 14/20 | 10/15 | 12/15 | 9/15 | **69** | FAIL |

### Tier Classification

| Tier | Min Score | Capabilities |
|------|-----------|--------------|
| **Tier 1** | ≥88 | NQED, AV-QKCM, QGAT-Mol |
| **Tier 2** | ≥80 | QEAR, QFLG |
| **Tier 3** | ≥70 | VQ-NAS, QARLP (QARLP fails gate) |

### Tier Promotion and Demotion Rules

**Two-Week Falsification Test**
- If we cannot define a concrete falsifiable test with measurable outputs, the capability cannot be Tier 1 or Tier 2
- Test must be documented in evidence pack with pass/fail criteria

**Six-Week Prototype Test**
- If no runnable proof of concept exists by week 6, demote one tier
- Exception requires explicit approval with documented rationale

**Kill Criteria Per Capability**

| Capability | Two-Week Test | Six-Week Test | Kill Condition |
|------------|---------------|---------------|----------------|
| NQED | GNN encoder produces valid embeddings for d=5 surface code | End-to-end decode with measurable accuracy | Accuracy < MWPM baseline |
| AV-QKCM | E-value test detects synthetic drift | Monitor produces valid confidence sequences | False positive rate > 10% |
| QGAT-Mol | Attention layer processes molecular graph | WASM demo with H2O molecule | Cannot represent basic orbitals |
| QEAR | Reservoir simulation produces features | Integration with attention module | No quantum advantage signal |
| QFLG | Gradient aggregation compiles | Byzantine detection works | Privacy guarantee broken |
| VQ-NAS | Search space definition | Architecture search runs | Search degenerates |
| QARLP | Policy gradient update works | Simple environment solved | Worse than classical baseline |

## Rationale

### NQED (Score: 93)
- Highest synergy: GNN + min-cut is genuinely novel integration
- Direct ruQu integration via syndrome pipeline
- AlphaQubit proves neural decoders work; we add structural awareness
- **Verification**: Clear benchmark against MWPM/UF decoders

### AV-QKCM (Score: 92)
- Perfect ruQu fit: extends e-value framework with quantum kernels
- Anytime-valid statistics are cutting-edge (Howard et al. 2021)
- Immediate applicability to coherence monitoring
- **Verification**: Statistical properties are mathematically verifiable

### QGAT-Mol (Score: 88)
- Clear quantum advantage (molecular orbitals are quantum)
- Strong industry demand (drug discovery)
- Good ruvector-attention integration path
- **Verification**: Existing molecular benchmarks (QM9, etc.)

### QEAR (Score: 82)
- Most scientifically novel: quantum reservoir + attention fusion
- Recent breakthroughs (5-atom reservoir, Feb 2025)
- Risk: hardware requirements, but simulation viable
- **Verification**: Weaker - relies on reservoir quality claims

### QFLG (Score: 83)
- Addresses critical privacy concerns
- Natural cognitum-gate-tilezero extension
- Byzantine tolerance is relevant
- **Verification**: Privacy proofs can be formalized

### VQ-NAS (Score: 74)
- Interesting but crowded field
- Longer time to value
- Keep as exploratory
- **Verification**: Search effectiveness is measurable

### QARLP (Score: 69, Gate FAIL)
- Quantum RL is promising but early
- Limited RuVector integration points
- **Gate Failure**: Insufficient reproducible sources
- Keep as exploratory, re-evaluate quarterly

## Consequences

### Positive
- Clear prioritization with verification requirements
- Measurable criteria for progress evaluation
- Tier system with promotion/demotion prevents drift
- Kill criteria prevent sunk cost fallacy

### Negative
- Scores still contain subjective elements
- May miss breakthrough opportunities in lower-scored areas
- Two-week/six-week tests add overhead

### Mitigation
- Quarterly re-evaluation of scores with updated evidence
- Allow 10% time for capability pivots
- Cross-pollination between tiers
- Evidence packs track all scoring decisions

## Amendments (2026-01-17)

### Verification Path Criterion
Added to measure whether the capability can be validated by external signals that remain true when the system is off. This includes falsifiable tests, reproducible benchmarks, and at least one independent measurement source.

### Research Foundation Gate
Capabilities must meet minimum literature requirements before Tier 1/2 classification.

### Scoring Consistency
Added anchor rubrics to ensure different agents score consistently.

### Tier Promotion/Demotion Rules
Two-week falsification and six-week prototype requirements with automatic demotion.

## Related
- [Main Research Document](../../ai-quantum-capabilities-2025.md)
- [ADR-001: Swarm Structure](ADR-001-swarm-structure.md)
- [Capability Scorecard](../swarm-config/capability-scorecard.yaml)
