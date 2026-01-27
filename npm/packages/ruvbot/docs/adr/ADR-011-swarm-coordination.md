# ADR-011: Swarm Coordination (agentic-flow Integration)

## Status
Accepted

## Date
2026-01-27

## Context

Clawdbot has basic async processing. RuvBot integrates agentic-flow for:
- Multi-agent swarm coordination
- 12 specialized background workers
- Byzantine fault-tolerant consensus
- Dynamic topology switching

## Decision

### Swarm Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RuvBot Swarm Coordination                     │
├─────────────────────────────────────────────────────────────────┤
│  Topologies                                                      │
│    ├─ hierarchical       : Queen-worker (anti-drift)           │
│    ├─ mesh               : Peer-to-peer network                 │
│    ├─ hierarchical-mesh  : Hybrid for scalability              │
│    └─ adaptive           : Dynamic switching                    │
├─────────────────────────────────────────────────────────────────┤
│  Consensus Protocols                                             │
│    ├─ byzantine          : BFT (f < n/3 faulty)                 │
│    ├─ raft               : Leader-based (f < n/2)               │
│    ├─ gossip             : Eventually consistent                │
│    └─ crdt               : Conflict-free replication            │
├─────────────────────────────────────────────────────────────────┤
│  Background Workers (12)                                         │
│    ├─ ultralearn   [normal]   : Deep knowledge acquisition     │
│    ├─ optimize     [high]     : Performance optimization        │
│    ├─ consolidate  [low]      : Memory consolidation (EWC++)   │
│    ├─ predict      [normal]   : Predictive preloading           │
│    ├─ audit        [critical] : Security analysis               │
│    ├─ map          [normal]   : Codebase mapping                │
│    ├─ preload      [low]      : Resource preloading             │
│    ├─ deepdive     [normal]   : Deep code analysis              │
│    ├─ document     [normal]   : Auto-documentation              │
│    ├─ refactor     [normal]   : Refactoring suggestions         │
│    ├─ benchmark    [normal]   : Performance benchmarking        │
│    └─ testgaps     [normal]   : Test coverage analysis          │
└─────────────────────────────────────────────────────────────────┘
```

### Integration with agentic-flow

```typescript
import {
  SwarmCoordinator,
  ByzantineConsensus,
  WorkerPool
} from 'agentic-flow';

// Initialize swarm
const swarm = new SwarmCoordinator({
  topology: 'hierarchical',
  maxAgents: 8,
  strategy: 'specialized',
  consensus: 'raft'
});

// Dispatch to specialized workers
await swarm.dispatch({
  worker: 'ultralearn',
  task: { type: 'deep-analysis', content },
  priority: 'normal'
});

// Byzantine fault-tolerant consensus
const consensus = new ByzantineConsensus({
  replicas: 5,
  timeout: 30000
});
await consensus.propose(decision);
```

### Worker Configuration

```typescript
interface WorkerConfig {
  type: WorkerType;
  priority: 'low' | 'normal' | 'high' | 'critical';
  concurrency: number;
  timeout: number;
  retries: number;
  backoff: 'exponential' | 'linear';
}

const WORKER_DEFAULTS: Record<WorkerType, WorkerConfig> = {
  ultralearn: { priority: 'normal', concurrency: 2, timeout: 60000 },
  optimize: { priority: 'high', concurrency: 4, timeout: 30000 },
  consolidate: { priority: 'low', concurrency: 1, timeout: 120000 },
  audit: { priority: 'critical', concurrency: 1, timeout: 45000 },
  // ... etc
};
```

## Consequences

### Positive
- Distributed task execution
- Fault tolerance via consensus
- Specialized workers for different task types
- Dynamic scaling

### Negative
- Coordination overhead
- Complexity of distributed systems
- Network latency

### RuvBot Advantages over Clawdbot
- 12 specialized workers vs basic async
- Byzantine fault tolerance vs none
- Multi-topology support vs single-threaded
- Learning workers (ultralearn, consolidate) vs static
