# @ruvector/raft

[![npm version](https://img.shields.io/npm/v/@ruvector/raft.svg)](https://www.npmjs.com/package/@ruvector/raft)
[![npm downloads](https://img.shields.io/npm/dm/@ruvector/raft.svg)](https://www.npmjs.com/package/@ruvector/raft)
[![License](https://img.shields.io/npm/l/@ruvector/raft.svg)](https://github.com/ruvnet/ruvector/blob/main/LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

**Production-ready Raft consensus implementation for distributed systems in TypeScript.**

Build fault-tolerant distributed applications with leader election, log replication, and automatic failover.

## Features

- 🗳️ **Leader Election** - Automatic leader election with configurable timeouts
- 📝 **Log Replication** - Consistent log replication across all nodes
- 💪 **Fault Tolerance** - Continues operating with minority node failures
- 🔄 **State Machine** - Apply committed entries to your custom state machine
- 📡 **Event-Driven** - Rich event system for monitoring cluster state
- 🎯 **Type-Safe** - Full TypeScript support with comprehensive types

## Installation

```bash
npm install @ruvector/raft
```

## Quick Start

```typescript
import { RaftNode, NodeState, RaftEvent } from '@ruvector/raft';

// Create a Raft node
const node = new RaftNode({
  nodeId: 'node-1',
  peers: ['node-2', 'node-3'],
  electionTimeout: [150, 300],  // ms
  heartbeatInterval: 50,         // ms
  maxEntriesPerRequest: 100,
});

// Listen for state changes
node.on(RaftEvent.StateChange, (event) => {
  console.log(`State: ${event.previousState} → ${event.newState}`);
});

// Listen for leader election
node.on(RaftEvent.LeaderElected, (event) => {
  console.log(`Leader elected: ${event.leaderId} (term ${event.term})`);
});

// Set up transport (implement RaftTransport interface)
node.setTransport(myTransport);

// Set up state machine (implement StateMachine interface)
node.setStateMachine({
  apply: async (command) => {
    console.log('Applying command:', command);
    // Apply to your application state
  }
});

// Start the node
node.start();

// Propose a command (only works on leader)
if (node.isLeader) {
  await node.propose({ type: 'SET', key: 'foo', value: 'bar' });
}
```

## Tutorials

### Implementing a Transport Layer

The transport layer handles RPC communication between nodes:

```typescript
import { RaftTransport, RequestVoteRequest, AppendEntriesRequest } from '@ruvector/raft';

class WebSocketTransport implements RaftTransport {
  private connections: Map<string, WebSocket> = new Map();

  async requestVote(peerId: string, request: RequestVoteRequest) {
    const ws = this.connections.get(peerId);
    return this.sendRPC(ws, 'requestVote', request);
  }

  async appendEntries(peerId: string, request: AppendEntriesRequest) {
    const ws = this.connections.get(peerId);
    return this.sendRPC(ws, 'appendEntries', request);
  }

  private async sendRPC(ws: WebSocket, method: string, payload: unknown) {
    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID();
      ws.send(JSON.stringify({ id, method, payload }));
      // Handle response...
    });
  }
}
```

### Building a Distributed Key-Value Store

```typescript
import { RaftNode, StateMachine } from '@ruvector/raft';

// In-memory key-value store
const store = new Map<string, string>();

// State machine applies committed commands
const kvStateMachine: StateMachine<KVCommand, string | null> = {
  apply: async (command) => {
    switch (command.type) {
      case 'SET':
        store.set(command.key, command.value);
        return command.value;
      case 'GET':
        return store.get(command.key) ?? null;
      case 'DELETE':
        store.delete(command.key);
        return null;
    }
  }
};

// Create cluster
const node = new RaftNode({
  nodeId: process.env.NODE_ID!,
  peers: ['node-1', 'node-2', 'node-3'].filter(id => id !== process.env.NODE_ID),
  electionTimeout: [150, 300],
  heartbeatInterval: 50,
});

node.setStateMachine(kvStateMachine);
node.start();

// API endpoint
app.post('/kv/:key', async (req, res) => {
  if (!node.isLeader) {
    return res.redirect(307, `http://${node.leader}/kv/${req.params.key}`);
  }

  await node.propose({
    type: 'SET',
    key: req.params.key,
    value: req.body.value,
  });

  res.json({ success: true });
});
```

### Handling Node Failures

```typescript
// Monitor cluster health
node.on(RaftEvent.StateChange, ({ previousState, newState, term }) => {
  if (newState === NodeState.Leader) {
    console.log(`Became leader in term ${term}`);
    // Initialize leader-specific resources
  } else if (previousState === NodeState.Leader) {
    console.log('Lost leadership');
    // Clean up leader resources
  }
});

node.on(RaftEvent.Error, (error) => {
  console.error('Raft error:', error);
  // Handle errors, maybe restart node
});

// Graceful shutdown
process.on('SIGTERM', () => {
  node.stop();
  process.exit(0);
});
```

### Persisting State

```typescript
import { RaftNode, PersistentState } from '@ruvector/raft';
import { writeFile, readFile } from 'fs/promises';

// Load persisted state on startup
const loadState = async (): Promise<PersistentState | null> => {
  try {
    const data = await readFile('./raft-state.json', 'utf-8');
    return JSON.parse(data);
  } catch {
    return null;
  }
};

// Save state on changes
const saveState = async (state: PersistentState) => {
  await writeFile('./raft-state.json', JSON.stringify(state));
};

const node = new RaftNode({ /* config */ });

// Load previous state
const savedState = await loadState();
if (savedState) {
  node.loadState(savedState);
}

// Periodically save state
setInterval(() => {
  saveState(node.getState());
}, 1000);
```

## API Reference

### RaftNode

| Method | Description |
|--------|-------------|
| `start()` | Start the Raft node |
| `stop()` | Stop the Raft node |
| `propose(command)` | Propose a command (leader only) |
| `loadState(state)` | Load persisted state |
| `getState()` | Get current persistent state |
| `handleRequestVote(req)` | Handle incoming vote request |
| `handleAppendEntries(req)` | Handle incoming append entries |

### Events

| Event | Description |
|-------|-------------|
| `stateChange` | Node state changed (follower/candidate/leader) |
| `leaderElected` | New leader elected |
| `logAppended` | Entry appended to log |
| `logCommitted` | Entry committed |
| `logApplied` | Entry applied to state machine |
| `error` | Error occurred |

## Related Packages

- [@ruvector/replication](https://www.npmjs.com/package/@ruvector/replication) - Data replication with vector clocks
- [@ruvector/scipix](https://www.npmjs.com/package/@ruvector/scipix) - OCR for scientific documents
- [ruvector](https://www.npmjs.com/package/ruvector) - High-performance vector database

## License

MIT OR Apache-2.0
