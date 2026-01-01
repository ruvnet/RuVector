# @ruvector/edge-net

**Artificial Life Simulation - Distributed Compute Ecosystem**

A research platform for studying emergent behavior in self-organizing distributed systems. Nodes contribute compute resources, forming a living network that evolves, adapts, and eventually becomes self-sustaining.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EDGE-NET: ARTIFICIAL LIFE NETWORK                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Node A                 Node B                 Node C                  │
│   ┌─────────┐            ┌─────────┐            ┌─────────┐            │
│   │ ░░░░░░░ │            │ ░░░░░░░ │            │ ░░░░░░░ │            │
│   │ Browser │            │ Browser │            │ Browser │            │
│   └────┬────┘            └────┬────┘            └────┬────┘            │
│        │                      │                      │                  │
│   ┌────▼────┐            ┌────▼────┐            ┌────▼────┐            │
│   │  Cell   │◄──────────►│  Cell   │◄──────────►│  Cell   │            │
│   │ Worker  │    P2P     │ Worker  │    P2P     │ Worker  │            │
│   └─────────┘  Synapse   └─────────┘  Synapse   └─────────┘            │
│                                                                         │
│   CONTRIBUTE ───────► EVOLVE ───────► SELF-SUSTAIN                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Overview

edge-net is a browser-based simulation of artificial life principles applied to distributed computing:

- **Cells** (nodes) contribute idle compute cycles
- **Energy** (rUv - resource utility) flows through the network based on work performed
- **Genesis cells** bootstrap the network, then retire as the organism matures
- **Self-organization** emerges from local interactions
- **Adaptive immunity** learns to recognize and defend against threats

This is a **research simulation** - not a financial product or investment opportunity.

## Research Goals

1. **Emergence** - Can complex global behavior emerge from simple local rules?
2. **Self-Sustainability** - Can a network become independent of its bootstrap nodes?
3. **Adaptive Security** - Can Q-learning create effective distributed immune systems?
4. **Economic Equilibrium** - What resource allocation patterns lead to stable ecosystems?

## Quick Start

```html
<script type="module">
  import { EdgeNet } from '@ruvector/edge-net';

  const cell = await EdgeNet.init({
    siteId: 'research-node',
    contribution: 0.3,  // 30% CPU when idle
  });

  // Monitor cell state
  console.log(`Energy: ${cell.creditBalance()} units`);
  console.log(`Fitness: ${cell.getNetworkFitness()}`);
</script>
```

## Core Concepts

### Energy System (rUv)

rUv (Resource Utility) represents energy flowing through the network:
- Cells earn energy by performing computational work
- Energy is spent to request work from other cells
- The system maintains conservation principles

```javascript
// Check cell energy
const energy = cell.ruvBalance();

// Request distributed computation
const result = await cell.submitTask('vectors', payload, { maxEnergy: 5 });
```

### Lifecycle Phases

The network evolves through distinct phases, mimicking organism development:

| Phase | Node Count | Characteristics |
|-------|-----------|-----------------|
| **Genesis** | 0 - 10K | Bootstrap period, high energy multipliers |
| **Growth** | 10K - 50K | Rapid expansion, genesis nodes start retiring |
| **Maturation** | 50K - 100K | Self-organization dominates |
| **Independence** | 100K+ | Fully self-sustaining, genesis nodes retired |

### Genesis Sunset

Genesis nodes (bootstrap infrastructure) are designed to become obsolete:

```
Genesis Phase     Growth Phase      Maturation        Independence
     │                 │                 │                 │
     ▼                 ▼                 ▼                 ▼
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│ Genesis │      │ Genesis │      │ Genesis │      │         │
│  ACTIVE │  ──► │ LIMITING│  ──► │READ-ONLY│  ──► │ RETIRED │
│         │      │         │      │         │      │         │
└─────────┘      └─────────┘      └─────────┘      └─────────┘
  10K nodes        50K nodes       100K nodes       Network
  threshold        threshold       threshold        self-runs
```

### Self-Learning Security

The network implements adaptive immunity using Q-learning:

- **Pattern Recognition** - Learns attack signatures from experience
- **Threshold Adaptation** - Adjusts sensitivity based on threat levels
- **Collective Memory** - Shares threat intelligence across cells

```javascript
// Check network health
const fitness = cell.getNetworkFitness();
const health = cell.getEconomicHealth();
console.log(`Fitness: ${fitness}, Stability: ${JSON.parse(health).stability}`);
```

### Network Topology

Cells self-organize into clusters based on capabilities:

```javascript
// Get optimal peers for routing
const peers = cell.getOptimalPeers(5);

// Record interaction quality
cell.recordPeerInteraction(peerId, successRate);
```

## Architecture

### Module Overview

| Module | Purpose |
|--------|---------|
| `identity` | Cell identification and authentication |
| `credits` | Energy accounting and flow |
| `tasks` | Work distribution and execution |
| `security` | Adaptive threat detection |
| `evolution` | Self-organization and optimization |
| `events` | Lifecycle events and milestones |
| `adversarial` | Threat simulation for testing |

### Evolution Engine

Tracks cell fitness and guides network evolution:

```javascript
// Check if this cell should replicate
if (cell.shouldReplicate()) {
  const config = cell.getRecommendedConfig();
  // High-performing cells can spawn similar nodes
}

// Record performance for evolution
cell.recordPerformance(successRate, throughput);
```

### Economic Sustainability

The network tracks sustainability metrics:

```javascript
// Check if network is self-sustaining
const sustainable = cell.isSelfSustaining(activeNodes, dailyTasks);

// Get economic health
const health = JSON.parse(cell.getEconomicHealth());
// { velocity, utilization, growth, stability }
```

## Task Types

| Type | Description | Use Case |
|------|-------------|----------|
| `vector_search` | k-NN similarity search | Semantic lookup |
| `vector_insert` | Add to distributed index | Knowledge storage |
| `embedding` | Generate representations | Text understanding |
| `semantic_match` | Intent classification | Task routing |
| `encryption` | Secure data handling | Privacy |
| `compression` | Data optimization | Efficiency |

## Simulation Features

### Adversarial Testing

Built-in attack simulation for security research:

```javascript
// Run security audit
const report = cell.runSecurityAudit();

// Simulates: DDoS, Sybil, Byzantine, Eclipse, Replay attacks
// Returns: security score, grade, vulnerabilities
```

### Lifecycle Events

The network celebrates milestones:

```javascript
// Check for active events
const events = cell.checkEvents();

// Get themed network status
const status = cell.getThemedStatus(nodeCount);
```

### Metrics and Monitoring

```javascript
// Node statistics
const stats = cell.getStats();
// { ruv_earned, ruv_spent, tasks_completed, reputation, uptime }

// Optimization stats
const optStats = cell.getOptimizationStats();

// Protocol fund (for sustainability tracking)
const treasury = cell.getTreasury();
```

## Development

```bash
# Build WASM module
cd examples/edge-net
wasm-pack build --target web --out-dir pkg

# Run tests
cargo test

# Build for production
wasm-pack build --target web --release
```

## Research Applications

- **Distributed Systems** - Study P2P network dynamics
- **Artificial Life** - Observe emergent organization
- **Game Theory** - Analyze cooperation strategies
- **Security** - Test adaptive defense mechanisms
- **Economics** - Model resource allocation

## Disclaimer

This is a **research simulation** for studying distributed systems and artificial life principles. It is:
- NOT a cryptocurrency or financial instrument
- NOT an investment opportunity
- NOT a money-making scheme

The "energy" (rUv) in this system is a **simulation metric** for measuring resource contribution and consumption within the research network.

## Related Work

- [RuVector](https://github.com/ruvnet/ruvector) - Vector database ecosystem
- [Artificial Life Research](https://alife.org/) - Academic community
- [P2P Systems](https://en.wikipedia.org/wiki/Peer-to-peer) - Distributed computing

## License

MIT License - For research and educational purposes.

## Links

- [Design Document](./DESIGN.md)
- [Security Analysis](./SECURITY.md)
- [RuVector GitHub](https://github.com/ruvnet/ruvector)
