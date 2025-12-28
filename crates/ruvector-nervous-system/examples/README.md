# Nervous System Examples

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

Bio-inspired nervous system architecture examples demonstrating the transition from **"How do we make machines smarter?"** to **"What kind of organism are we building?"**

## Overview

These examples show how nervous system thinking unlocks new products, markets, and research categories. The architecture enables systems that **age well** instead of breaking under complexity.

## Application Tiers

### Tier 1: Immediate Practical Applications
*Shippable with current architecture*

| Example | Domain | Key Benefit |
|---------|--------|-------------|
| [anomaly_detection](tier1/anomaly_detection.rs) | Infrastructure, Finance, Security | Detection before failure, microsecond response |
| [edge_autonomy](tier1/edge_autonomy.rs) | Drones, Vehicles, Robotics | Lower power, certified reflex paths |
| [medical_wearable](tier1/medical_wearable.rs) | Monitoring, Assistive Devices | Adapts to the person, always-on, private |

### Tier 2: Near-Term Transformative Applications
*Possible once local learning and coherence routing mature*

| Example | Domain | Key Benefit |
|---------|--------|-------------|
| [self_optimizing_systems](tier2/self_optimizing_systems.rs) | Agents Monitoring Agents | Self-stabilizing software, structural witnesses |
| [swarm_intelligence](tier2/swarm_intelligence.rs) | IoT Fleets, Sensor Meshes | Scale without fragility, emergent intelligence |
| [adaptive_simulation](tier2/adaptive_simulation.rs) | Digital Twins, Logistics | Always-warm simulation, costs scale with relevance |

### Tier 3: Exotic But Real Applications
*Technically grounded, novel research directions*

| Example | Domain | Key Benefit |
|---------|--------|-------------|
| [machine_self_awareness](tier3/machine_self_awareness.rs) | Structural Self-Sensing | Systems say "I am becoming unstable" |
| [synthetic_nervous_systems](tier3/synthetic_nervous_systems.rs) | Buildings, Factories, Cities | Environments respond like organisms |
| [bio_machine_interface](tier3/bio_machine_interface.rs) | Prosthetics, Rehabilitation | Machines stop fighting biology |

## Quick Start

```bash
# Run a Tier 1 example
cargo run --example anomaly_detection

# Run a Tier 2 example
cargo run --example swarm_intelligence

# Run a Tier 3 example
cargo run --example machine_self_awareness
```

## Architecture Principles

Each example demonstrates the same five-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    COHERENCE LAYER                          │
│  Global Workspace • Oscillatory Routing • Predictive Coding │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                     LEARNING LAYER                          │
│     BTSP One-Shot • E-prop Online • EWC Consolidation      │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY LAYER                           │
│     Hopfield Networks • HDC Vectors • Pattern Separation   │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      REFLEX LAYER                           │
│      K-WTA Competition • Dendritic Coincidence • Safety    │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      SENSING LAYER                          │
│      Event Bus • Sparse Spikes • Backpressure Control      │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts Demonstrated

### Reflex Arcs
Fast, deterministic responses with bounded execution:
- Latency: <100μs
- Certifiable: Maximum iteration counts
- Safety: Witness logging for every decision

### Homeostasis
Self-regulation instead of static thresholds:
- Adaptive learning from normal operation
- Graceful degradation under stress
- Anticipatory maintenance

### Coherence Gating
Synchronize only when needed:
- Kuramoto oscillators for phase coupling
- Communication gain based on phase coherence
- 90-99% bandwidth reduction via prediction

### One-Shot Learning
Learn immediately from single examples:
- BTSP: Seconds-scale eligibility traces
- No batch retraining required
- Personalization through use

## Tutorial: Building a Custom Application

### Step 1: Define Your Sensing Layer

```rust
use ruvector_nervous_system::eventbus::{DVSEvent, EventRingBuffer};

// Create event buffer with backpressure
let buffer = EventRingBuffer::new(1024);

// Process events sparsely
if let Some(event) = buffer.pop() {
    // Only significant changes generate events
}
```

### Step 2: Add Reflex Gates

```rust
use ruvector_nervous_system::compete::WTALayer;

// Winner-take-all for fast decisions
let mut wta = WTALayer::new(100, 0.5, 0.8);

// <1μs for 1000 neurons
if let Some(winner) = wta.compete(&inputs) {
    trigger_immediate_response(winner);
}
```

### Step 3: Implement Memory

```rust
use ruvector_nervous_system::hopfield::ModernHopfield;
use ruvector_nervous_system::hdc::Hypervector;

// Hopfield for associative retrieval
let mut hopfield = ModernHopfield::new(512, 10.0);
hopfield.store(pattern);

// HDC for ultra-fast similarity
let similarity = v1.similarity(&v2); // <100ns
```

### Step 4: Enable Learning

```rust
use ruvector_nervous_system::plasticity::btsp::BTSPSynapse;

// One-shot learning
let mut synapse = BTSPSynapse::new(0.5, 2000.0); // 2s time constant
synapse.update(presynaptic_active, plateau_signal, dt);
```

### Step 5: Add Coherence

```rust
use ruvector_nervous_system::routing::{OscillatoryRouter, GlobalWorkspace};

// Phase-coupled routing
let mut router = OscillatoryRouter::new(10, 40.0); // 40Hz gamma
let gain = router.communication_gain(sender, receiver);

// Global workspace (4-7 items)
let mut workspace = GlobalWorkspace::new(7);
workspace.broadcast(representation);
```

## Performance Targets

| Component | Latency | Throughput |
|-----------|---------|------------|
| Event Bus | <100ns push/pop | 10,000+ events/ms |
| WTA | <1μs | 1M+ decisions/sec |
| HDC Similarity | <100ns | 10M+ comparisons/sec |
| Hopfield Retrieval | <1ms | 1000+ queries/sec |
| BTSP Update | <100ns | 10M+ synapses/sec |

## From Practical to Exotic

The same architecture scales from:

1. **Practical**: Anomaly detection with microsecond response
2. **Transformative**: Self-optimizing software systems
3. **Exotic**: Machines that sense their own coherence

The difference is how much reflex, learning, and coherence you turn on.

## Further Reading

- [Architecture Documentation](../../docs/nervous-system/architecture.md)
- [Deployment Guide](../../docs/nervous-system/deployment.md)
- [Test Plan](../../docs/nervous-system/test-plan.md)
- [Main Crate Documentation](../README.md)

## Contributing

Examples welcome! Each should demonstrate:
1. A clear use case
2. The nervous system architecture
3. Performance characteristics
4. Tests and documentation

## License

MIT License - See [LICENSE](../LICENSE)
