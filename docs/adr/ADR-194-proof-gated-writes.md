# ADR-194: Proof-Gated Vector Writes with Merkle-Accumulating Witness Logs

**Status:** Proposed  
**Date:** 2026-05-24  
**Deciders:** RuVector core team  
**Tags:** security, agent-memory, integrity, merkle, vector-database, RAG-safety

---

## Context

Every major vector database (Qdrant, Milvus, Weaviate, LanceDB, FAISS) stores
vectors without cryptographic integrity guarantees on the write path. A vector
inserted at time T cannot be distinguished from one silently mutated at time
T+1. This is not a theoretical risk: the MemoryGraft attack (arxiv:2512.16962,
Dec 2025) demonstrated that all major agent memory systems accept writes without
provenance verification, enabling persistent compromise of retrieval results.

The "Mnemonic Sovereignty" survey (arxiv:2604.16548, Apr 2026) identifies
"write-path security" as the single most unaddressed gap in LLM agent memory:
agents write to shared vector stores with no ability to audit, replay, or verify
the history of writes. HONEYBEE (arxiv:2505.01538, May 2025) addresses query-
side RBAC for vector databases but explicitly leaves the write path unguarded.

RuVector is positioned as a Rust-native cognition substrate for agents. For
agent memory to be trustworthy, the write substrate must provide:
1. Tamper evidence: any post-write mutation changes a verifiable commitment.
2. Receipt issuance: each write returns a proof that can be stored independently.
3. Offline verification: integrity can be checked without the vector index.
4. WASM compatibility: no syscalls or OS-specific crypto (sha2 is pure Rust).

---

## Decision

Introduce `ruvector-proof-gate` as a new composable crate in the RuVector
workspace. It provides:

### WriteGate trait

```rust
pub trait WriteGate: Send + Sync {
    fn admit(&mut self, payload: &WritePayload) -> Result<WriteReceipt, GateError>;
    fn verify_receipt(&self, receipt: &WriteReceipt) -> bool;
    fn chain_root(&self) -> [u8; 32];
    fn len(&self) -> usize;
    fn variant(&self) -> GateVariant;
}
```

### Three production-relevant variants

**NullGate** — zero-overhead baseline. Admits all writes with no hashing.
Use only in development or testing environments where integrity is not required.

**HashChainGate** — O(1) per write. Each write computes:
```
commitment[n] = SHA256("ruvector:chain:" || commitment[n-1] || payload_hash || n)
```
where `payload_hash = SHA256(canonical_bytes(payload))`. This creates a
sequential, append-only chain. Mutating entry k breaks all commitments from k
onward, detectable by replaying the chain. Memory cost: 32 bytes per entry.
Measured throughput (release, x86_64, 10K writes of 128-dim vectors):
**253,889 writes/sec, mean latency 3.9 µs**.

**MerkleGate** — O(log n) amortized per write using a Merkle Mountain Range.
The MMR maintains a forest of perfect binary trees. Appending is O(log n)
amortized. The "bagged" root changes with every write. Unlike a linear chain,
the MMR supports membership proofs: a leaf N can be proven to be in the set
without replaying the full history. Memory cost: ~2n * 32 bytes (leaves + peaks).
Measured throughput: **128,215 writes/sec, mean latency 7.8 µs**.

### WritePayload and WriteReceipt

`WritePayload` carries: id (u64), vector (Vec<f32>), metadata (Vec<u8>),
agent_id ([u8; 16]), timestamp_ns (u64). `canonical_bytes()` produces a
length-prefixed, fixed-structure serialization that prevents length-extension
ambiguity.

`WriteReceipt` carries: sequence (u64), payload_hash ([u8; 32]),
chain_commitment ([u8; 32]), gate_variant (GateVariant). Receipts are
intentionally small (~80 bytes) and can be stored alongside vector entries or
in a separate audit table.

---

## Consequences

### Positive

- Agents can now prove "I wrote vector X at time T with agent_id Y" by
  presenting a receipt and verifying it against a stored chain root.
- The write path adds 3.9–7.8 µs overhead on commodity x86_64 hardware,
  acceptable for agent memory writes (not high-frequency trading).
- No blockchain, no external service, no network call: fully in-process.
- WASM-compatible (sha2 = 0.10 compiles to WASM with no std feature changes).
- Composable: any caller holding a `Box<dyn WriteGate>` can swap strategies.

### Negative

- HashChainGate stores 32 bytes per entry; 1M writes = 32 MB gate state. For
  very large agent memory stores, this should be checkpointed and compacted.
- MerkleGate's leaf store also grows linearly (n * 32 bytes for membership
  proofs); same checkpoint recommendation.
- Neither variant prevents a compromised write path from creating fraudulent
  receipts. The guarantees are tamper-evidence, not write authorization. For
  write authorization, combine with RBAC (see future HONEYBEE extension).
- Full chain replay (HashChain verification) is O(n) and requires re-hashing
  all payloads; this is practical for audit but not for every query.

---

## Alternatives Considered

### Blockchain-backed tamper evidence (arxiv:2511.07577)
Blockchain-RAG uses a distributed ledger for source reliability scores.
**Rejected**: prohibitive latency (~seconds per write), no WASM compatibility,
requires a network connection, massively overengineered for in-process agent
memory.

### Ed25519 write signatures
Sign each write with a private key. **Partially attractive**: provides
authorization (not just tamper-evidence). **Deferred**: key management
complexity is out of scope for the PoC; adding to `ed25519-dalek` dependency
raises the MSRV and WASM surface. Can be added as a `SignedGate` wrapper
in a follow-on crate.

### Trusted Execution Environment attestation (arxiv:2604.05480 context)
TEE-backed write proofs (Intel TDX, AMD SEV) prove the embedding model was not
tampered. **Deferred**: hardware dependency, not WASM-compatible, appropriate
for a future `ruvector-tee` crate.

### Append-only database log (e.g., redb write-ahead log)
RuVector already uses redb for persistence. The WAL could be hashed.
**Deferred**: too tightly coupled to storage; the WriteGate is intentionally
storage-agnostic so it wraps any vector store, in-memory or disk-backed.

---

## Implementation Plan

- [x] `crates/ruvector-proof-gate/` crate created and added to workspace
- [x] `WriteGate` trait with `NullGate`, `HashChainGate`, `MerkleGate`
- [x] `WritePayload` and `WriteReceipt` types
- [x] 15 unit tests, all passing
- [x] Benchmark binary with acceptance thresholds
- [x] Research document at `docs/research/nightly/2026-05-24-proof-gated-writes/`
- [ ] `serde` feature for receipt serialization to JSON/MessagePack
- [ ] `SignedGate` wrapper with Ed25519 write authorization
- [ ] Checkpoint/compaction API for long-running agent memory stores
- [ ] Integration with `ruvector-core` InsertOp pipeline
- [ ] MCP tool surface for receipt submission and verification
- [ ] WASM build target and benchmark

---

## Benchmark Evidence

Hardware: x86_64 Linux, release build (opt-level=3, LTO fat)  
Dataset: 10,000 vectors × 128 dimensions  
Rust: 1.94.1

| Variant       | Mean (µs) | p50 (µs) | p95 (µs) | Throughput     | Memory (KB) |
|---------------|-----------|----------|----------|----------------|-------------|
| NullGate      | 0.024     | 0.022    | 0.023    | 42,560,617/sec | ~0          |
| HashChainGate | 3.939     | 3.649    | 4.782    | 253,889/sec    | 312.5       |
| MerkleGate    | 7.799     | 7.713    | 9.739    | 128,215/sec    | 313.0       |

Receipt verification: PASS for all variants.  
Chain roots: non-zero and distinct for HashChain and Merkle.  
Acceptance result: **PASS**.

Cargo command:
```
cargo run --release -p ruvector-proof-gate --example benchmark
```

---

## Failure Modes

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Gate state lost (OOM, crash) | Chain root unverifiable | Checkpoint root to durable store after every N writes |
| Receipt store lost | Cannot verify past writes | Store receipts in a separate append-only log |
| Payload hash collision | False negative on verification | SHA-256 collision probability negligible (2^{-128}) |
| Sequential write bottleneck | Throughput < 50K/sec | Gate is `&mut self`; for concurrent writes, partition by namespace |
| Chain replay required at scale | O(n) verification cost | Use MerkleGate (O(log n) membership proofs) |

---

## Security Considerations

- SHA-256 is collision-resistant under NIST standards. No known practical
  preimage or collision attacks as of 2026.
- The genesis seed in HashChainGate is a fixed constant (not a secret).
  The chain provides tamper-evidence, not authentication. For authentication,
  add a `SignedGate` layer.
- `canonical_bytes()` uses explicit length prefixes to prevent payload aliasing.
  A vector `[1.0]` with metadata `[2.0 bytes...]` cannot be confused with
  `[1.0, 2.0]` with empty metadata.
- The crate has no `unsafe` code. All hash operations use the `sha2` crate
  which is `#![forbid(unsafe_code)]` internally.

---

## Migration Path

1. **Phase 1 (now):** Opt-in. Callers wrap their insert logic with a WriteGate.
   Existing RuVector code unchanged.
2. **Phase 2:** Add `WriteGate` integration to `ruvector-core` InsertOp.
   Feature-flag: `--features proof-gate`.
3. **Phase 3:** MCP tool surface: `vector_write_admit` returns receipt,
   `vector_write_verify` checks a stored receipt against the current gate state.
4. **Phase 4:** WASM build. Verify sha2 compiles correctly under
   `wasm32-unknown-unknown` and run the benchmark binary via Wasmtime.

---

## Open Questions

1. Should the gate state be persisted to redb alongside the vector data?
   If yes, what serialization format? (bincode, rkyv, or custom)
2. Should `MerkleGate` expose inclusion proof generation?
   The leaf array is already stored; proof generation is O(log n) path tracing.
3. For ruFlo workflow loops, should gate checkpoints be treated as workflow
   events (triggering audit steps automatically)?
4. Does the `agent_id` field in `WritePayload` need a verification step
   (i.e. should the agent_id be verified against an identity registry)?
5. What is the right default gate in `ruvector-core`? `NullGate` for
   backward compatibility, or `HashChainGate` for safety-first?
