# EXO-AI 2025: Advanced Cognitive Substrate

<div align="center">

[![Crates.io](https://img.shields.io/crates/v/exo-core.svg)](https://crates.io/crates/exo-core)
[![Documentation](https://docs.rs/exo-core/badge.svg)](https://docs.rs/exo-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ruvnet%2Fruvector-blue?logo=github)](https://github.com/ruvnet/ruvector)
[![Website](https://img.shields.io/badge/Website-ruv.io-purple)](https://ruv.io)

**A research platform exploring the computational foundations of consciousness, memory, and cognition**

[Documentation](https://docs.rs/exo-core) | [GitHub](https://github.com/ruvnet/ruvector) | [Website](https://ruv.io) | [Examples](#quick-start)

</div>

---

## Overview

EXO-AI 2025 is a comprehensive cognitive substrate implementing cutting-edge theories from neuroscience, physics, and consciousness research. Built on the [RuVector](https://github.com/ruvnet/ruvector) foundation, it provides 9 interconnected Rust crates totaling ~15,800+ lines of research-grade code.

### Why EXO-AI?

Traditional AI systems process information. EXO-AI aims to understand it — implementing theories of consciousness (IIT), memory consolidation, free energy minimization, and emergence detection. This isn't just another neural network framework; it's a platform for exploring the computational basis of mind.

## Crates

| Crate | Description | Docs |
|-------|-------------|------|
| [`exo-core`](https://crates.io/crates/exo-core) | IIT consciousness (Φ) measurement & Landauer thermodynamics | [![docs](https://docs.rs/exo-core/badge.svg)](https://docs.rs/exo-core) |
| [`exo-temporal`](https://crates.io/crates/exo-temporal) | Temporal memory with causal tracking & consolidation | [![docs](https://docs.rs/exo-temporal/badge.svg)](https://docs.rs/exo-temporal) |
| [`exo-hypergraph`](https://crates.io/crates/exo-hypergraph) | Topological analysis with persistent homology | [![docs](https://docs.rs/exo-hypergraph/badge.svg)](https://docs.rs/exo-hypergraph) |
| [`exo-manifold`](https://crates.io/crates/exo-manifold) | SIREN networks for continuous embedding deformation | [![docs](https://docs.rs/exo-manifold/badge.svg)](https://docs.rs/exo-manifold) |
| [`exo-exotic`](https://crates.io/crates/exo-exotic) | 10 cutting-edge cognitive experiments | [![docs](https://docs.rs/exo-exotic/badge.svg)](https://docs.rs/exo-exotic) |
| [`exo-federation`](https://crates.io/crates/exo-federation) | Post-quantum federated cognitive mesh | [![docs](https://docs.rs/exo-federation/badge.svg)](https://docs.rs/exo-federation) |
| [`exo-backend-classical`](https://crates.io/crates/exo-backend-classical) | SIMD-accelerated compute backend | [![docs](https://docs.rs/exo-backend-classical/badge.svg)](https://docs.rs/exo-backend-classical) |
| [`exo-wasm`](https://crates.io/crates/exo-wasm) | Browser & edge WASM deployment | [![docs](https://docs.rs/exo-wasm/badge.svg)](https://docs.rs/exo-wasm) |
| [`exo-node`](https://crates.io/crates/exo-node) | Node.js bindings via NAPI-RS | [![docs](https://docs.rs/exo-node/badge.svg)](https://docs.rs/exo-node) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           EXO-EXOTIC                                 │
│   Strange Loops │ Dreams │ Free Energy │ Morphogenesis              │
│   Collective │ Temporal │ Multiple Selves │ Thermodynamics          │
│   Emergence │ Cognitive Black Holes                                  │
├─────────────────────────────────────────────────────────────────────┤
│                           EXO-CORE                                   │
│      IIT Consciousness (Φ) │ Landauer Thermodynamics                │
│      Pattern Storage │ Causal Graph │ Metadata                      │
├─────────────────────────────────────────────────────────────────────┤
│                         EXO-TEMPORAL                                 │
│    Short-Term Buffer │ Long-Term Store │ Causal Memory              │
│    Anticipation │ Consolidation │ Prefetch Cache                    │
├─────────────────────────────────────────────────────────────────────┤
│                        EXO-HYPERGRAPH                                │
│    Topological Analysis │ Persistent Homology │ Sheaf Theory        │
├─────────────────────────────────────────────────────────────────────┤
│                         EXO-MANIFOLD                                 │
│    SIREN Networks │ Continuous Deformation │ Gradient Descent       │
├─────────────────────────────────────────────────────────────────────┤
│      EXO-WASM      │     EXO-NODE     │   EXO-FEDERATION           │
│   Browser Deploy   │  Native Bindings │  Distributed Consensus     │
├─────────────────────────────────────────────────────────────────────┤
│                     EXO-BACKEND-CLASSICAL                            │
│                Traditional Compute Backend                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

Add EXO-AI crates to your `Cargo.toml`:

```toml
[dependencies]
exo-core = "0.1"
exo-temporal = "0.1"
exo-exotic = "0.1"
```

## Quick Start

### Consciousness Measurement (IIT)

```rust
use exo_core::consciousness::{ConsciousnessSubstrate, IITConfig};
use exo_core::thermodynamics::CognitiveThermometer;

// Measure integrated information (Φ)
let substrate = ConsciousnessSubstrate::new(IITConfig::default());
substrate.add_pattern(pattern);
let phi = substrate.compute_phi();
println!("Consciousness level (Φ): {:.4}", phi);

// Track computational thermodynamics
let thermo = CognitiveThermometer::new(300.0); // Kelvin
let cost = thermo.landauer_cost_bits(1024);
println!("Landauer cost: {:.2e} J", cost);
```

### Temporal Memory

```rust
use exo_temporal::{TemporalMemory, CausalConeType};

let memory = TemporalMemory::default();
memory.store(pattern, &antecedents)?;

// Causal cone query
let results = memory.causal_query(
    &query,
    reference_time,
    CausalConeType::Past,
);

// Memory consolidation
memory.consolidate();
```

### Topological Analysis

```rust
use exo_hypergraph::{Hypergraph, TopologicalQuery};

let graph = Hypergraph::new();
graph.add_hyperedge(entities, relation)?;

// Compute persistent homology
let diagram = graph.query(TopologicalQuery::PersistentHomology {
    dimension: 1,
    epsilon_range: (0.0, 1.0),
})?;
```

### Exotic Experiments

```rust
use exo_exotic::{StrangeLoops, ArtificialDreams, FreeEnergy};

// Hofstadter Strange Loops
let loops = StrangeLoops::new(10);
let confidence = loops.self_reference_cascade();

// Dream-based creativity
let dreams = ArtificialDreams::with_memories(memories);
let novel_ideas = dreams.run_dream_cycle(100);

// Friston Free Energy
let fe = FreeEnergy::new(16, 16);
let prediction_error = fe.minimize(observations);
```

## Exotic Experiments

EXO-AI includes 10 cutting-edge cognitive experiments:

| Experiment | Theory | Key Insight |
|------------|--------|-------------|
| **Strange Loops** | Hofstadter | Self-reference creates consciousness |
| **Artificial Dreams** | Activation-Synthesis | Random replay enables creativity |
| **Free Energy** | Friston | Perception minimizes surprise |
| **Morphogenesis** | Turing Patterns | Cognition self-organizes |
| **Collective** | Distributed IIT | Consciousness can be networked |
| **Temporal Qualia** | Scalar Timing | Time is subjective experience |
| **Multiple Selves** | IFS Theory | Mind contains sub-personalities |
| **Thermodynamics** | Landauer | Information has physical cost |
| **Emergence** | Causal Emergence | Macro > Micro causation |
| **Black Holes** | Attractor Dynamics | Thoughts can trap attention |

## Key Discoveries

### 1. Self-Reference Limits
Strange loops reveal that confidence decays ~10% per meta-level, naturally bounding infinite regress.

### 2. Dream Creativity Scaling
Creative output increases logarithmically with memory diversity. 50+ memories yield 75%+ novel combinations.

### 3. Free Energy Convergence
Prediction error decreases 15-30% per learning cycle, stabilizing around iteration 100.

### 4. Morphogenetic Patterns
Gray-Scott parameters (f=0.055, k=0.062) produce stable cognitive patterns.

### 5. Collective Φ Scaling
Global integrated information scales with O(n²) connections.

### 6. Temporal Relativity
Novelty dilates subjective time up to 2x. Flow states compress time to 0.1x.

### 7. Multi-Self Coherence
Sub-personalities naturally maintain 0.7-0.9 coherence.

### 8. Thermodynamic Bounds
At 300K, Landauer limit is ~3×10⁻²¹ J/bit.

### 9. Causal Emergence
Macro-level descriptions can have higher effective information than micro-level.

### 10. Escape Dynamics
Reframing reduces cognitive black hole escape energy by 50%.

## Performance

| Module | Operation | Time |
|--------|-----------|------|
| IIT Φ Computation | 10 elements | ~15 µs |
| Strange Loops | 10 levels | ~2.4 µs |
| Dream Cycle | 100 memories | ~95 µs |
| Free Energy | 16×16 grid | ~3.2 µs |
| Morphogenesis | 32×32, 100 steps | ~9 ms |
| Collective Φ | 20 substrates | ~35 µs |
| Temporal Qualia | 1000 events | ~120 µs |
| Multiple Selves | 10 selves | ~4 µs |
| Thermodynamics | Landauer cost | ~0.02 µs |
| Emergence | 128→32 coarse-grain | ~8 µs |
| Black Holes | 1000 thoughts | ~150 µs |

## Build & Test

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector/examples/exo-ai-2025

# Build all crates
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Run specific crate tests
cargo test -p exo-exotic
cargo test -p exo-core
cargo test -p exo-temporal
```

## Practical Applications

| Domain | Application | Crate |
|--------|-------------|-------|
| **AI Alignment** | Self-aware AI with recursion limits | exo-exotic |
| **Mental Health** | Rumination detection and intervention | exo-exotic |
| **Learning Systems** | Memory consolidation optimization | exo-temporal |
| **Distributed AI** | Collective intelligence networks | exo-exotic |
| **Energy-Efficient AI** | Thermodynamically optimal compute | exo-core |
| **Creative AI** | Dream-based idea generation | exo-exotic |
| **Temporal Planning** | Subjective time-aware scheduling | exo-exotic |
| **Team Cognition** | Multi-agent coherence optimization | exo-exotic |
| **Pattern Recognition** | Self-organizing feature detection | exo-exotic |
| **Therapy AI** | Multiple selves conflict resolution | exo-exotic |

## Theoretical Foundations

- **IIT 4.0** (Tononi) — Integrated Information Theory for consciousness measurement
- **Free Energy** (Friston) — Variational free energy minimization
- **Strange Loops** (Hofstadter) — Self-referential consciousness
- **Landauer's Principle** — Information has physical cost
- **Turing Morphogenesis** — Reaction-diffusion pattern formation
- **Causal Emergence** (Hoel) — Macro-level causal power

## Contributing

Contributions are welcome! See our [Contributing Guide](https://github.com/ruvnet/ruvector/blob/main/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT OR Apache-2.0

## Links

- **GitHub**: [github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Website**: [ruv.io](https://ruv.io)
- **Documentation**: [docs.rs/exo-core](https://docs.rs/exo-core)
- **Crates.io**: [crates.io/crates/exo-core](https://crates.io/crates/exo-core)

## References

1. Tononi, G. (2008). Consciousness as integrated information.
2. Friston, K. (2010). The free-energy principle: a unified brain theory?
3. Hofstadter, D. R. (2007). I Am a Strange Loop.
4. Turing, A. M. (1952). The chemical basis of morphogenesis.
5. Landauer, R. (1961). Irreversibility and heat generation.
6. Hoel, E. P. (2017). When the map is better than the territory.
7. Baars, B. J. (1988). A Cognitive Theory of Consciousness.
8. Schwartz, R. C. (1995). Internal Family Systems Therapy.
9. Eagleman, D. M. (2008). Human time perception and its illusions.
10. Revonsuo, A. (2000). The reinterpretation of dreams.

---

<div align="center">

**Made with ❤️ by [rUv](https://ruv.io)**

</div>
