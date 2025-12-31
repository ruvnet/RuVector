# @ruvector/edge-net

**Distributed Compute Intelligence Network**

Contribute browser compute, earn **rUv** (Resource Utility Vouchers), access shared AI infrastructure.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     EDGE-NET: SHARED COMPUTE NETWORK                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Website A              Website B              Website C               │
│   ┌─────────┐            ┌─────────┐            ┌─────────┐            │
│   │ Visitor │            │ Visitor │            │ Visitor │            │
│   │ Browser │            │ Browser │            │ Browser │            │
│   └────┬────┘            └────┬────┘            └────┬────┘            │
│        │                      │                      │                  │
│   ┌────▼────┐            ┌────▼────┐            ┌────▼────┐            │
│   │edge-net │◄──────────►│edge-net │◄──────────►│edge-net │            │
│   │ Worker  │    P2P     │ Worker  │    P2P     │ Worker  │            │
│   └─────────┘            └─────────┘            └─────────┘            │
│                                                                         │
│   CONTRIBUTE ───────► EARN rUv VOUCHERS ───────► ACCESS COMPUTE        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```html
<script type="module">
  import { EdgeNet } from '@ruvector/edge-net';

  const node = await EdgeNet.init({
    siteId: 'my-site',
    contribution: 0.3,  // 30% CPU when idle
  });

  // Check earnings
  console.log(`Balance: ${node.creditBalance()} rUv`);
</script>
```

## Features

| Feature | Description |
|---------|-------------|
| **rUv Currency** | Resource Utility Vouchers - quantum-resistant DAG credits |
| **Contribution Curve** | Early adopters earn up to 10x multiplier |
| **Web Workers** | Non-blocking compute in background threads |
| **P2P Network** | Serverless task distribution via GUN.js |
| **Stake & Earn** | Stake rUv to participate and earn rewards |
| **Reputation System** | Quality-based ranking for task assignment |
| **Genesis Sunset** | Genesis nodes retire when network is self-sustaining |

## How It Works

### 1. Contribute Compute

When visitors browse your site, idle CPU cycles are used for distributed AI tasks:

```javascript
const node = await EdgeNet.init({
  siteId: 'your-site',
  contribution: {
    cpuLimit: 0.3,          // Max 30% CPU
    memoryLimit: 256_000_000, // 256MB
    tasks: ['vectors', 'embeddings', 'encryption'],
  },
});
```

### 2. Earn rUv (Resource Utility Vouchers)

rUv are earned based on:
- **Compute work completed** (1 rUv per task unit)
- **Uptime bonus** (0.1 rUv per hour online)
- **Early adopter multiplier** (up to 10x for first contributors)

```javascript
// Check current multiplier
const multiplier = node.getMultiplier();
console.log(`Current multiplier: ${multiplier}x`);

// Check balance
const balance = node.creditBalance();
console.log(`rUv Balance: ${balance}`);
```

### 3. Use rUv for AI Tasks

Spend earned vouchers to access distributed AI compute:

```javascript
// Submit a vector search task
const result = await node.submitTask('vector_search', {
  query: new Float32Array(128).fill(0.5),
  k: 10,
}, {
  maxRuv: 5,
});

console.log(result);
// { results: [...], cost: 2, verified: true }
```

## rUv: Resource Utility Vouchers

rUv is a quantum-resistant DAG-based credit system designed for compute resource allocation:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          rUv DAG LEDGER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌───┐   ┌───┐   ┌───┐                                               │
│    │TX1│──►│TX2│──►│TX4│                                               │
│    └───┘   └───┘   └───┘                                               │
│       ╲       ╲     ╱                                                   │
│        ╲       ╲   ╱                                                    │
│    ┌───┐ ╲   ┌───┐   ┌───┐                                             │
│    │TX3│──►──│TX5│──►│TX6│◄── Latest transactions                      │
│    └───┘     └───┘   └───┘                                             │
│                                                                         │
│    • No mining (instant finality)                                      │
│    • Zero transaction fees                                              │
│    • Quantum-resistant signatures (ML-DSA)                              │
│    • Proof-of-work spam prevention                                      │
│    • Genesis nodes sunset when network matures                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Contribution Curve

Early adopters receive bonus multipliers that decay as the network grows:

| Network Stage | Multiplier | Genesis Status |
|---------------|------------|----------------|
| Genesis | 10.0x | Genesis nodes required |
| 100K CPU-hours | 9.1x | Genesis nodes required |
| 1M CPU-hours | 4.0x | Genesis nodes optional |
| 10M+ CPU-hours | 1.0x | Network self-sustaining |

```
multiplier = 1 + 9 × e^(-network_compute / 1,000,000)
```

### Genesis Node Sunset

Genesis nodes bootstrap the network but are designed to become unnecessary:

| Threshold | Action |
|-----------|--------|
| 10K+ active nodes | Genesis nodes stop accepting new connections |
| 50K+ active nodes | Genesis nodes enter read-only mode |
| 100K+ active nodes | Genesis nodes can be safely retired |
| Self-sustaining | Pure P2P network - no central infrastructure |

### Staking

Stake rUv to participate in consensus and earn passive rewards:

```javascript
// Stake 1000 rUv
await node.stake(1000);

// Check staked amount
const staked = node.stakedAmount();

// Unstake (after lock period)
await node.unstake(500);
```

## Security

| Layer | Protection |
|-------|------------|
| Identity | Ed25519 signatures |
| Encryption | AES-256-GCM for task payloads |
| Consensus | QDAG with cumulative weight |
| Anti-Sybil | Stake + fingerprinting + rate limits |
| Verification | Redundant execution + spot-checks |

See [SECURITY.md](./SECURITY.md) for full security analysis.

## API Reference

### EdgeNetNode

```javascript
const node = await EdgeNet.init(config);

// Identity
node.nodeId()           // Unique node identifier
node.creditBalance()    // Current rUv balance
node.getMultiplier()    // Current reward multiplier
node.getStats()         // { ruv, tasks, uptime, reputation }

// Contribution
node.start()            // Start contributing
node.pause()            // Pause contribution
node.resume()           // Resume contribution
node.disconnect()       // Leave network

// Tasks
await node.submitTask(type, payload, options)
await node.processNextTask()  // For workers

// Staking
await node.stake(amount)
await node.unstake(amount)
node.stakedAmount()
```

### Configuration

```javascript
EdgeNet.init({
  // Identity
  siteId: 'my-site',

  // Contribution
  contribution: {
    cpuLimit: 0.3,              // 0.0 - 1.0
    memoryLimit: 256_000_000,   // bytes
    bandwidthLimit: 1_000_000,  // bytes/sec
    tasks: ['vectors', 'embeddings', 'encryption'],
  },

  // Idle detection
  idle: {
    minIdleTime: 5000,          // ms before contributing
    respectBattery: true,       // reduce on battery
  },

  // Network
  relays: [
    'https://gun-manhattan.herokuapp.com/gun',
  ],

  // Callbacks
  onCredit: (earned, total) => {},
  onTask: (task) => {},
  onError: (error) => {},
});
```

## Task Types

| Type | Description | Cost |
|------|-------------|------|
| `vector_search` | k-NN search in HNSW index | 1 rUv / 1K vectors |
| `vector_insert` | Add vectors to index | 0.5 rUv / 100 vectors |
| `embedding` | Generate text embeddings | 5 rUv / 100 texts |
| `semantic_match` | Task-to-agent routing | 1 rUv / 10 queries |
| `encryption` | AES encrypt/decrypt | 0.1 rUv / MB |
| `compression` | Adaptive quantization | 0.2 rUv / MB |

## Performance

| Metric | Target |
|--------|--------|
| WASM load time | < 100ms |
| Memory usage (idle) | < 50MB |
| CPU usage (active) | Configurable 10-50% |
| Task latency | < 100ms |
| Credit sync | < 1s |

## Integration with RuVector

edge-net integrates with the RuVector ecosystem:

- **ruvector-dag**: DAG-based task scheduling and critical path analysis
- **ruvector-graph**: Distributed graph database for knowledge storage
- **@ruvector/edge**: WASM modules for crypto, vectors, neural networks
- **QUDAG**: Quantum-resistant consensus from ruvector-dag

## Development

```bash
# Build WASM
cd examples/edge-net
wasm-pack build --target web --out-dir pkg

# Run tests
wasm-pack test --headless --chrome

# Bundle for CDN
cd pkg && npx esbuild edge-net.js --bundle --minify --outfile=edge-net.min.js
```

## License

MIT License

## Links

- [Design Document](./DESIGN.md)
- [Security Analysis](./SECURITY.md)
- [RuVector GitHub](https://github.com/ruvnet/ruvector)
- [npm Package](https://www.npmjs.com/package/@ruvector/edge-net)
