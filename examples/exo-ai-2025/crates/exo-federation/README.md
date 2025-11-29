# exo-federation

Federated cognitive mesh networking for EXO-AI 2025 distributed substrate.

## Overview

This crate implements a distributed federation layer for cognitive substrates with:

- **Post-quantum cryptography** (CRYSTALS-Kyber key exchange)
- **Privacy-preserving onion routing** for query intent protection
- **CRDT-based eventual consistency** across federation nodes
- **Byzantine fault-tolerant consensus** (PBFT-style)

## Architecture

```
┌─────────────────────────────────────────┐
│      FederatedMesh (Coordinator)        │
├─────────────────────────────────────────┤
│ • Local substrate instance              │
│ • Consensus coordination                │
│ • Federation gateway                    │
│ • Cryptographic identity                │
└─────────────────────────────────────────┘
         │           │           │
   ┌─────┘           │           └─────┐
   ▼                 ▼                 ▼
Handshake         Onion            CRDT
Protocol          Router      Reconciliation
```

## Modules

### `crypto.rs` (232 lines)

Post-quantum cryptographic primitives:

- `PostQuantumKeypair` - CRYSTALS-Kyber key pairs (placeholder implementation)
- `EncryptedChannel` - Secure communication channels
- `SharedSecret` - Key derivation from PQ key exchange

**Status**: Placeholder implementation. Real implementation will use `pqcrypto-kyber`.

### `handshake.rs` (280 lines)

Federation joining protocol:

- `join_federation()` - Cryptographic handshake with peers
- `FederationToken` - Access token with negotiated capabilities
- `Capability` - Feature negotiation system

**Protocol**:
1. Post-quantum key exchange
2. Establish encrypted channel
3. Exchange and negotiate capabilities
4. Issue federation token

### `onion.rs` (263 lines)

Privacy-preserving query routing:

- `onion_query()` - Multi-hop encrypted routing
- `OnionMessage` - Layered encrypted messages
- `peel_layer()` - Relay node layer decryption

**Features**:
- Query intent privacy (each relay only knows prev/next hop)
- Multiple encryption layers
- Response routing through same path

### `crdt.rs` (329 lines)

Conflict-free replicated data types:

- `GSet<T>` - Grow-only set (union merge)
- `LWWRegister<T>` - Last-writer-wins register (timestamp-based)
- `LWWMap<K,V>` - Map of LWW registers
- `reconcile_crdt()` - Merge federated query responses

**Properties**:
- Commutative, associative, idempotent merges
- Eventual consistency guarantees
- No coordination required for updates

### `consensus.rs` (340 lines)

Byzantine fault-tolerant consensus:

- `byzantine_commit()` - PBFT-style consensus protocol
- `CommitProof` - Cryptographic proof of consensus
- Byzantine threshold calculation (n = 3f + 1)

**Phases**:
1. Pre-prepare (leader proposes)
2. Prepare (nodes acknowledge, 2f+1 required)
3. Commit (nodes commit, 2f+1 required)

### `lib.rs` (286 lines)

Main federation coordinator:

- `FederatedMesh` - Main coordinator struct
- `FederationScope` - Query scope control (Local/Direct/Global)
- `FederatedResult` - Query results from peers

## Usage Example

```rust
use exo_federation::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create local substrate instance
    let substrate = SubstrateInstance {};

    // Initialize federated mesh
    let mut mesh = FederatedMesh::new(substrate)?;

    // Join federation
    let peer = PeerAddress::new(
        "peer.example.com".to_string(),
        8080,
        peer_public_key.to_vec()
    );
    let token = mesh.join_federation(&peer).await?;

    // Execute federated query
    let results = mesh.federated_query(
        query_data,
        FederationScope::Global { max_hops: 5 }
    ).await?;

    // Commit state update with consensus
    let update = StateUpdate { /* ... */ };
    let proof = mesh.byzantine_commit(update).await?;

    Ok(())
}
```

## Implementation Status

### ✅ Completed

- Core data structures and interfaces
- Module organization
- Async patterns with Tokio
- Comprehensive test coverage
- Documentation

### 🚧 Placeholder Implementations

- **Post-quantum crypto**: Currently using simplified placeholders
  - Real implementation needs `pqcrypto-kyber` integration
  - Proper key exchange protocol

- **Network layer**: Simulated message passing
  - Real implementation needs TCP/UDP networking
  - Message serialization/deserialization

- **Consensus coordination**: Single-node simulation
  - Real implementation needs distributed message collection
  - Network timeout handling

### 🔜 Future Work

1. **Real PQC Integration**
   - Integrate `pqcrypto-kyber` crate
   - Implement actual key exchange
   - Add digital signatures

2. **Network Layer**
   - TCP/QUIC transport
   - Message framing
   - Connection pooling

3. **Distributed Consensus**
   - Leader election
   - View change protocol
   - Checkpoint mechanisms

4. **Performance Optimizations**
   - Batch message processing
   - Parallel verification
   - Cache optimizations

## Security Considerations

### Implemented

- Post-quantum key exchange (placeholder)
- Message authentication codes
- Onion routing for query privacy

### TODO

- Certificate management
- Peer authentication
- Rate limiting
- DoS protection
- Audit logging

## Dependencies

```toml
exo-core = { path = "../exo-core" }
tokio = { version = "1.41", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
dashmap = "6.1"
rand = "0.8"
sha2 = "0.10"
hex = "0.4"
```

## Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test --lib crypto
cargo test --lib handshake
cargo test --lib consensus
```

## References

- **CRYSTALS-Kyber**: [pqcrypto.org](https://pqcrypto.org/)
- **PBFT**: "Practical Byzantine Fault Tolerance" by Castro & Liskov
- **CRDTs**: "A comprehensive study of CRDTs" by Shapiro et al.
- **Onion Routing**: Tor protocol design

## Integration with EXO-AI

This crate integrates with the broader EXO-AI cognitive substrate:

- **exo-core**: Core traits and types
- **exo-temporal**: Causal memory coordination
- **exo-manifold**: Distributed manifold queries
- **exo-hypergraph**: Federated topology queries

## License

MIT OR Apache-2.0
