# RuVector Nervous System

[![Crates.io](https://img.shields.io/crates/v/ruvector-nervous-system.svg)](https://crates.io/crates/ruvector-nervous-system)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://docs.rs/ruvector-nervous-system)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-313%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Lines of Code](https://img.shields.io/badge/lines-22.9k-blue.svg)]()

**A five-layer bio-inspired nervous system architecture for vector databases, enabling systems that survive, adapt, and cooperate.**

> *"From 'How do we make machines smarter?' to 'What kind of organism are we building?'"*

## Overview

This crate implements a complete nervous system architecture inspired by biological neural systems, targeting **100-1000× energy improvements** and **sub-millisecond latency** for vector database operations. Instead of just optimizing algorithms, we've defined a new capability class.

```
┌─────────────────────────────────────────────────────────────┐
│                    COHERENCE LAYER                          │
│  Global Workspace • Oscillatory Routing • Predictive Coding │
│                     (90-99% bandwidth reduction)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     LEARNING LAYER                          │
│     BTSP One-Shot • E-prop Online • EWC Consolidation      │
│                    (Learn in single exposure)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY LAYER                           │
│     Hopfield Networks • HDC Vectors • Pattern Separation   │
│                  (2^(d/2) exponential capacity)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      REFLEX LAYER                           │
│      K-WTA Competition • Dendritic Coincidence • Safety    │
│                       (<1μs decisions)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      SENSING LAYER                          │
│      Event Bus • Sparse Spikes • Backpressure Control      │
│                  (10,000+ events/ms throughput)             │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Hyperdimensional Computing (HDC)
- **10,000-bit binary hypervectors** with 10^40 representational capacity
- **XOR binding** in <50ns
- **Hamming similarity** in <100ns via SIMD popcount
- Associative memory with collision-resistant encoding

### Modern Hopfield Networks
- **Exponential storage**: 2^(d/2) patterns in d dimensions
- Mathematically equivalent to **transformer attention**
- Single-step retrieval via softmax-weighted sum
- <1ms retrieval for 1000 patterns in 512D

### K-Winner-Take-All (K-WTA)
- **<1μs** single winner selection for 1000 neurons
- Lateral inhibition for sparse activation
- HNSW-compatible routing decisions

### Pattern Separation
- Hippocampal-inspired **dentate gyrus** encoding
- **2-5% sparsity** matching cortical statistics
- <1% collision rate on synthetic corpora

### Dendritic Coincidence Detection
- **NMDA-like nonlinearity** with 10-50ms windows
- Plateau potentials for BTSP gating
- Reduced compartment models

### BTSP: Behavioral Timescale Plasticity
- **One-shot learning** over seconds-long windows
- Eligibility traces with 1-3 second time constants
- Bidirectional plasticity (weak→potentiate, strong→depress)

### E-prop: Eligibility Propagation
- **O(1) memory per synapse** (12 bytes)
- Online learning without backpropagation through time
- 1000+ millisecond temporal credit assignment

### Elastic Weight Consolidation (EWC)
- **45% forgetting reduction** with 2× parameter overhead
- Fisher Information diagonal approximation
- Complementary Learning Systems (hippocampus + neocortex)

### Coherence-Gated Routing
- **Kuramoto oscillators** for phase-coupled communication
- Predictive coding with **90-99% bandwidth reduction**
- Global workspace with 4-7 item capacity (Miller's law)

### Event Bus
- **Lock-free ring buffers** with <100ns push/pop
- Region-based sharding with backpressure control
- **10,000+ events/ms** sustained throughput

## Use Cases: From Practical to Exotic

### Tier 1: Immediate Practical Applications

| Application | What Changes | Key Benefit |
|-------------|--------------|-------------|
| **Anomaly Detection** | Event streams replace batch logs; reflexes fire on structural anomalies | Detection before failure, microsecond response |
| **Edge Autonomy** | Reflex arcs handle safety; policy loops only when needed | Lower power, certifiable bounded paths |
| **Medical Wearables** | Continuous sensing with sparse spikes; one-shot personalization | Adapts to person, always-on, private |

### Tier 2: Near-Term Transformative

| Application | What Changes | Key Benefit |
|-------------|--------------|-------------|
| **Self-Optimizing Software** | Watch structure and timing, not just outputs | Self-stabilizing, structural witnesses |
| **Swarm Intelligence** | Local reflexes, coherence gates for sync | Scale without fragility, emergent intelligence |
| **Digital Twins** | Low fidelity continuous, bullet-time for critical | Always warm, costs scale with relevance |

### Tier 3: Exotic But Real

| Application | What Changes | Key Benefit |
|-------------|--------------|-------------|
| **Machine Self-Awareness** | Monitor own coherence; sense failure before drops | "I am becoming unstable" |
| **Synthetic Nervous Systems** | Infrastructure as sensing fabric | Environments respond like organisms |
| **Bio-Machine Interfaces** | Adapt to biological timing; integrate with reflexes | Machines stop fighting biology |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ruvector-nervous-system = "0.1"
```

### Example: One-Shot Learning

```rust
use ruvector_nervous_system::plasticity::btsp::{BTSPLayer, BTSPAssociativeMemory};

// Create a BTSP layer with 2 second time constant
let mut layer = BTSPLayer::new(100, 2000.0);

// One-shot association: pattern -> target
let pattern = vec![0.1; 100];
layer.one_shot_associate(&pattern, 1.0);

// Immediate recall (no training iterations!)
let output = layer.forward(&pattern);
assert!((output - 1.0).abs() < 0.1);
```

### Example: Hyperdimensional Computing

```rust
use ruvector_nervous_system::hdc::{Hypervector, HdcMemory};

// Create random 10,000-bit hypervectors
let concept_a = Hypervector::random();
let concept_b = Hypervector::random();

// XOR binding (<50ns)
let bound = concept_a.bind(&concept_b);

// Similarity via Hamming distance (<100ns)
let sim = concept_a.similarity(&concept_b);

// Associative memory
let mut memory = HdcMemory::new();
memory.store("apple", concept_a.clone());
let results = memory.retrieve(&concept_a, 0.9);
```

### Example: Modern Hopfield Retrieval

```rust
use ruvector_nervous_system::hopfield::ModernHopfield;

// Create network with exponential capacity
let mut hopfield = ModernHopfield::new(512, 10.0);

// Store patterns
hopfield.store(pattern1);
hopfield.store(pattern2);

// Retrieve with noisy query (<1ms)
let retrieved = hopfield.retrieve(&noisy_query);
```

### Example: Winner-Take-All

```rust
use ruvector_nervous_system::compete::WTALayer;

// Create WTA layer
let mut wta = WTALayer::new(1000, 0.5, 0.8);

// Fast winner selection (<1μs)
if let Some(winner) = wta.compete(&activations) {
    route_to_winner(winner);
}
```

### Example: Coherence-Gated Routing

```rust
use ruvector_nervous_system::routing::{OscillatoryRouter, GlobalWorkspace};

// Kuramoto oscillators for phase coupling
let mut router = OscillatoryRouter::new(10, 40.0); // 40Hz gamma band
router.step(0.001); // 1ms step

// Communication gain based on phase coherence
let gain = router.communication_gain(sender, receiver);

// Global workspace (4-7 items max)
let mut workspace = GlobalWorkspace::new(7);
workspace.broadcast(representation);
```

## Tutorial: Building a Complete System

### Step 1: Event Sensing

```rust
use ruvector_nervous_system::eventbus::{DVSEvent, ShardedEventBus, BackpressureController};

// Sharded event bus with backpressure
let bus = ShardedEventBus::new_spatial(4, 1024);
let controller = BackpressureController::default();

// Process events sparsely
for event in stream {
    controller.update(bus.avg_fill_ratio());
    if controller.should_accept() {
        bus.push(event)?;
    }
}
```

### Step 2: Reflex Response

```rust
use ruvector_nervous_system::compete::KWTALayer;
use ruvector_nervous_system::dendrite::Dendrite;

// K-winners for sparse activation
let kwta = KWTALayer::new(1000, 50); // Top 50 winners
let winners = kwta.select(&inputs);

// Dendritic coincidence detection
let mut dendrite = Dendrite::new(10, 30.0); // 10 synapses, 30ms window
dendrite.receive_spike(synapse_id, timestamp);
if dendrite.has_plateau() {
    trigger_btsp_learning();
}
```

### Step 3: Memory and Learning

```rust
use ruvector_nervous_system::separate::DentateGyrus;
use ruvector_nervous_system::plasticity::eprop::EpropNetwork;

// Pattern separation before storage
let encoder = DentateGyrus::new(512, 10000, 500, 42); // 5% sparsity
let sparse_code = encoder.encode(&input);

// Online learning with e-prop
let mut network = EpropNetwork::new(100, 500, 10);
network.online_step(&input, &target, 0.001, 0.01);
```

### Step 4: Coherence and Coordination

```rust
use ruvector_nervous_system::routing::CoherenceGatedSystem;

// Full coherence-gated system
let mut system = CoherenceGatedSystem::new(10, 40.0, 0.5, 7);

// Route with coherence gating
let routed = system.route_with_coherence(&message, sender, 0.001);
```

## Performance Benchmarks

| Component | Target | Achieved |
|-----------|--------|----------|
| HDC Binding | <50ns | 64ns |
| HDC Similarity | <100ns | ~80ns |
| WTA Single Winner | <1μs | <1μs |
| K-WTA (k=50) | <10μs | 2.7μs |
| Hopfield Retrieval | <1ms | <1ms |
| Pattern Separation | <500μs | <500μs |
| E-prop Synapse Memory | 8-12 bytes | 12 bytes |
| Event Bus | 10K events/ms | 10K+ events/ms |

## Documentation

- [Architecture Guide](docs/nervous-system/architecture.md) - Complete crate layout and traits
- [Deployment Guide](docs/nervous-system/deployment.md) - Three-phase deployment plan
- [Test Plan](docs/nervous-system/test-plan.md) - Benchmarks and quality metrics
- [Examples](examples/README.md) - Practical to exotic use cases

## Biological References

| Component | Research Basis |
|-----------|----------------|
| HDC | Kanerva 1988, Plate 2003 |
| Modern Hopfield | Ramsauer et al. 2020 |
| Pattern Separation | Rolls 2013, Dentate Gyrus |
| Dendritic Processing | Stuart & Spruston 2015, Dendrify |
| BTSP | Bittner et al. 2017 |
| E-prop | Bellec et al. 2020 |
| EWC | Kirkpatrick et al. 2017 |
| Coherence Routing | Fries 2015 |
| Global Workspace | Baars 1988, Dehaene 2014 |

## License

MIT License - See [LICENSE](LICENSE)

## Contributing

We welcome contributions! Each module should include:
- Comprehensive unit tests
- Criterion benchmarks
- Documentation with biological context
- Examples demonstrating use cases

## What This Enables

Systems that:
- **Survive** - Graceful degradation, not catastrophic failure
- **Adapt** - Learning through use, not retraining
- **Cooperate** - Emergent coordination, not central control

This is no longer just about making machines smarter. It's about giving them nervous systems that let them exist in the world.
