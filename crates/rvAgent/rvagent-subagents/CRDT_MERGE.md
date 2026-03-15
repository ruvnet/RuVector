# CRDT State Merging for Parallel Subagents

## Overview

The CRDT (Conflict-free Replicated Data Type) merge implementation in `rvagent-subagents` provides deterministic conflict resolution when multiple subagents execute in parallel and produce overlapping state updates.

This implementation follows **ADR-103 B7** (CRDT State Merging) and uses:
- **Vector Clocks** for causal ordering
- **Last-Write-Wins (LWW) Registers** for conflict resolution
- **Deterministic tie-breaking** using node IDs

## Architecture

### Core Components

1. **VectorClock** (`crdt_merge::VectorClock`)
   - Tracks causal ordering of events across nodes
   - Enables detection of happens-before relationships
   - Merges by taking max timestamp per node

2. **LwwRegister** (`crdt_merge::LwwRegister<T>`)
   - Last-Write-Wins register with vector clock timestamps
   - Resolves conflicts by timestamp (higher wins)
   - Uses node_id as deterministic tie-breaker

3. **CrdtState** (`crdt_merge::CrdtState`)
   - Collection of LWW registers keyed by string
   - Each node has a unique node_id
   - Supports deterministic merge operations

4. **merge_subagent_results** (`crdt_merge::merge_subagent_results`)
   - Primary API for merging parallel subagent results
   - Validates unique node IDs
   - Merges all child states into parent

## Usage

### Basic Example

```rust
use rvagent_subagents::crdt_merge::{CrdtState, merge_subagent_results};

// Parent agent (node_id = 0)
let mut parent = CrdtState::new(0);
parent.set("status", b"analyzing".to_vec());

// Subagent 1 (node_id = 1)
let mut subagent1 = CrdtState::new(1);
subagent1.set("findings", b"3 issues".to_vec());

// Subagent 2 (node_id = 2)
let mut subagent2 = CrdtState::new(2);
subagent2.set("findings", b"5 issues".to_vec());

// Merge all subagent results
merge_subagent_results(&mut parent, vec![subagent1, subagent2])?;

// For concurrent writes to "findings", subagent2 wins (higher node_id)
assert_eq!(parent.get("findings"), Some(b"5 issues".as_slice()));
```

### Conflict Resolution

When multiple subagents write to the same key, conflicts are resolved by:

1. **Timestamp comparison**: Higher timestamp wins
2. **Node ID tie-breaker**: If timestamps are equal, higher node_id wins

This ensures deterministic resolution regardless of merge order.

```rust
let mut parent = CrdtState::new(0);
parent.set("key1", b"parent_value".to_vec());

let mut child1 = CrdtState::new(1);
child1.set("key1", b"child1_value".to_vec());

let mut child2 = CrdtState::new(2);
child2.set("key1", b"child2_value".to_vec());

merge_subagent_results(&mut parent, vec![child1, child2])?;

// child2 wins (higher node_id, concurrent writes)
assert_eq!(parent.get("key1"), Some(b"child2_value".as_slice()));
```

### Causal Ordering

Vector clocks preserve causal relationships:

```rust
let mut state0 = CrdtState::new(0);
state0.set("counter", b"0".to_vec());

// state1 observes state0, then writes
let mut state1 = CrdtState::new(1);
state1.clock().clone_from(state0.clock());
state1.set("counter", b"1".to_vec());

// state2 observes state1, then writes
let mut state2 = CrdtState::new(2);
state2.clock().clone_from(state1.clock());
state2.set("counter", b"2".to_vec());

state0.merge(&state1);
state0.merge(&state2);

// state2's write happened after state1's (causal ordering preserved)
assert_eq!(state0.get("counter"), Some(b"2".as_slice()));
```

## API Reference

### VectorClock

```rust
impl VectorClock {
    pub fn new(node_id: u32) -> Self;
    pub fn tick(&mut self, node_id: u32);
    pub fn merge(&mut self, other: &VectorClock);
    pub fn happens_before(&self, other: &VectorClock) -> bool;
    pub fn get(&self, node_id: u32) -> u64;
}
```

### LwwRegister

```rust
impl<T> LwwRegister<T> {
    pub fn new(value: T, timestamp: u64, node_id: u32) -> Self;
    pub fn value(&self) -> &T;
    pub fn timestamp(&self) -> u64;
    pub fn node_id(&self) -> u32;
    pub fn should_win_over(&self, other: &Self) -> bool;
}
```

### CrdtState

```rust
impl CrdtState {
    pub fn new(node_id: u32) -> Self;
    pub fn set(&mut self, key: &str, value: Vec<u8>);
    pub fn get(&self, key: &str) -> Option<&[u8]>;
    pub fn clock(&self) -> &VectorClock;
    pub fn node_id(&self) -> u32;
    pub fn keys(&self) -> impl Iterator<Item = &String>;
    pub fn merge(&mut self, other: &CrdtState);
}
```

### merge_subagent_results

```rust
pub fn merge_subagent_results(
    parent: &mut CrdtState,
    results: Vec<CrdtState>,
) -> Result<(), MergeError>;
```

**Errors:**
- `MergeError::EmptyResultSet` if results is empty
- `MergeError::DuplicateNodeId` if any two states share the same node_id

## Integration with rvagent-subagents

The CRDT merge functionality is designed to work alongside the existing state isolation mechanisms:

1. **prepare_subagent_state**: Filters parent state before passing to subagent
2. **CRDT merge**: Deterministically merges parallel subagent results
3. **merge_subagent_state**: Merges non-excluded keys back to parent

### Parallel Execution Flow

```
Parent State
    │
    ├─> prepare_subagent_state() ─> Subagent 1 ─> CrdtState (node_id=1)
    ├─> prepare_subagent_state() ─> Subagent 2 ─> CrdtState (node_id=2)
    └─> prepare_subagent_state() ─> Subagent 3 ─> CrdtState (node_id=3)
                                           │
                                           ├─> merge_subagent_results()
                                           │
                                           ▼
                                    Merged Parent State
```

## Properties

### Correctness Guarantees

1. **Determinism**: Merge result is the same regardless of merge order
2. **Commutativity**: merge(A, B) = merge(B, A)
3. **Associativity**: merge(merge(A, B), C) = merge(A, merge(B, C))
4. **Idempotence**: merge(A, A) = A
5. **Causal Consistency**: Happens-before relationships are preserved

### Performance Characteristics

- **Space**: O(n × k) where n = number of keys, k = number of nodes in vector clock
- **Time**: O(k) per key comparison, O(n × k) for full merge
- **Memory**: Each register stores value + timestamp + node_id

## Testing

The implementation includes comprehensive tests:

```bash
# Run all CRDT merge tests
cargo test --package rvagent-subagents --lib crdt_merge

# Run specific test
cargo test --package rvagent-subagents --lib crdt_merge::tests::test_merge_subagent_results_success

# Run example
cargo run --example crdt_merge_demo
```

Test coverage includes:
- Vector clock operations (tick, merge, happens-before)
- LWW register conflict resolution
- CRDT state merge with various conflict scenarios
- Causal ordering preservation
- Error cases (empty results, duplicate node IDs)
- Large-scale merges (100+ keys)

## Future Enhancements

Potential improvements for future iterations:

1. **Delta CRDTs**: Only transmit deltas instead of full state
2. **Garbage Collection**: Remove old vector clock entries
3. **Compression**: Compact representation for large states
4. **Typed Values**: Generic over value types (not just Vec<u8>)
5. **Operation-based CRDTs**: Support for counters, sets, maps
6. **Merkle Trees**: Efficient state comparison and synchronization

## References

- ADR-103: rvAgent Subagent Architecture
- ADR-097: Subagent state isolation and handoff protocol
- [Conflict-free Replicated Data Types](https://arxiv.org/abs/1805.06358)
- [Vector Clocks](https://en.wikipedia.org/wiki/Vector_clock)
- [Last-Write-Wins Semantics](https://en.wikipedia.org/wiki/Eventual_consistency)

## License

MIT OR Apache-2.0
