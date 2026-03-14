# ruvix-proof

Proof engine with 3-tier routing for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate implements the proof engine that enforces proof-gated mutation in RuVix. Every state mutation requires a valid proof token, ensuring the kernel physically prevents unauthorized changes.

## Proof Tiers

| Tier | Name | Latency | Use Case |
|------|------|---------|----------|
| 0 | Reflex | <100ns | High-frequency vector updates |
| 1 | Standard | <100us | Graph mutations with Merkle witness |
| 2 | Deep | <10ms | Full coherence verification with mincut |

## Security Properties (ADR-087 Section 20.4)

- **Time-bounded validity**: Proofs expire after a configurable window (default 100ms)
- **Single-use nonces**: Each nonce can only be consumed once
- **Capability-gated**: PROVE rights required on target object
- **Cache limits**: Maximum 64 entries with 100ms TTL

## Components

### ProofEngine

Generates proof tokens with automatic tier routing based on operation complexity.

```rust
use ruvix_proof::{ProofEngine, ProofEngineConfig};

let mut engine = ProofEngine::new(ProofEngineConfig::default());

// Generate a Reflex-tier proof
let mutation_hash = [0u8; 32];
let token = engine.generate_reflex_proof(&mutation_hash, current_time_ns())?;
```

### ProofVerifier

Verifies proof tokens with security checks.

```rust
use ruvix_proof::{ProofVerifier, VerifierConfig};

let mut verifier = ProofVerifier::new();
let result = verifier.verify(&token, &expected_hash, current_time_ns)?;
```

### ProofCache

Manages proof caching with single-use nonce semantics.

```rust
use ruvix_proof::{ProofCache, ProofCacheConfig};

let mut cache = ProofCache::with_config(ProofCacheConfig {
    max_entries: 64,
    ttl_ns: 100_000_000, // 100ms
    evict_expired: true,
});
```

### WitnessLog

Records verified proofs as 82-byte ADR-047 compatible attestations.

```rust
use ruvix_proof::{WitnessLog, AttestationBuilder};

let mut log = WitnessLog::new();
let attestation = AttestationBuilder::from_token(&token, timestamp_ns).build();
let entry = log.append(attestation, token.tier)?;
```

## Features

- `std` (default): Enable standard library support
- `alloc`: Enable alloc crate support
- `coherence`: Enable coherence verification
- `metrics`: Enable metrics collection
- `audit-log`: Enable audit logging for proof operations

## Integration with RuVix

This crate integrates with:

- `ruvix-types`: Core type definitions (ProofToken, ProofTier, ProofPayload)
- `ruvix-cap`: Capability checking for PROVE rights
- `ruvector-verified`: ProofEnvironment for formal verification

## License

MIT OR Apache-2.0
