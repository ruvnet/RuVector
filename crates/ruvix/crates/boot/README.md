# ruvix-boot

RVF boot loading for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate provides the RVF (RuVector Format) boot loading infrastructure. The kernel boot follows a strict five-stage process with cryptographic verification at every step.

## Boot Sequence (ADR-087 Section 9.1)

| Stage | Name | Description |
|-------|------|-------------|
| **0** | Hardware Init | Platform-specific initialization (mocked in Phase A) |
| **1** | RVF Verify | Parse manifest + ML-DSA-65 signature verification |
| **2** | Object Create | Create root task, regions, queues, witness log |
| **3** | Component Mount | Mount components + distribute capabilities |
| **4** | First Attestation | Boot attestation to witness log |

```
Stage 0 ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4
Hardware    Signature   Objects     Mount       Attest
Init        Verify      Create      Components
```

## Security Properties (SEC-001)

Critical security fixes implemented:

- **Signature failure**: PANIC IMMEDIATELY, no fallback boot path
- **Root task capability drop**: After Stage 3, root task drops to minimum set
- **Witness log integrity**: Append-only, cryptographically linked

## Components

### BootLoader

Main boot orchestrator:

```rust
use ruvix_boot::{BootLoader, BootConfig};

let config = BootConfig::default();
let mut loader = BootLoader::new(config);

// Load and verify the RVF boot image
let manifest_bytes = include_bytes!("boot.rvf.manifest");
let signature = include_bytes!("boot.rvf.sig");

// This will PANIC if signature verification fails (SEC-001)
loader.boot(manifest_bytes, signature)?;
```

### RvfManifest

RVF package manifest structure:

```rust
use ruvix_boot::{RvfManifest, ComponentDecl, QueueWiring};

let manifest = RvfManifest {
    version: 1,
    components: vec![
        ComponentDecl {
            id: "sensor-hub",
            wasm_hash: [0u8; 32],
            memory: MemorySchema::default(),
        },
    ],
    queue_wirings: vec![
        QueueWiring {
            from: "sensor-hub",
            to: "inference-engine",
            queue_size: 64,
        },
    ],
    proof_policy: ProofPolicy::standard(),
};
```

### SignatureVerifier

ML-DSA-65 signature verification (NIST FIPS 204):

```rust
use ruvix_boot::{SignatureVerifier, VerifyResult};

let verifier = SignatureVerifier::new(&public_key);
match verifier.verify(&manifest_bytes, &signature) {
    VerifyResult::Valid => { /* continue boot */ },
    VerifyResult::Invalid => panic!("Boot signature invalid!"),
    VerifyResult::Error(e) => panic!("Signature verification error: {:?}", e),
}
```

### WitnessLog

Append-only witness log for boot attestation:

```rust
use ruvix_boot::{WitnessLog, WitnessLogConfig};

let config = WitnessLogConfig::default();
let mut log = WitnessLog::new(config);

// Record boot attestation
let entry = log.append_boot_attestation(&boot_proof)?;
```

### CapabilityDistribution

Post-boot capability restriction:

```rust
use ruvix_boot::{CapabilityDistribution, MinimumCapabilitySet};

// Root task drops to minimum capabilities after mount
let distribution = CapabilityDistribution::new();
distribution.restrict_root_task(root_task, MinimumCapabilitySet::default())?;
```

## Constants

```rust
use ruvix_boot::{
    ML_DSA_65_SIGNATURE_SIZE,
    ML_DSA_65_PUBLIC_KEY_SIZE,
    MAX_MANIFEST_SIZE,
    MAX_COMPONENTS,
};

assert_eq!(ML_DSA_65_SIGNATURE_SIZE, 3309);  // NIST FIPS 204
assert_eq!(ML_DSA_65_PUBLIC_KEY_SIZE, 1952);
assert_eq!(MAX_MANIFEST_SIZE, 1024 * 1024);  // 1 MiB
assert_eq!(MAX_COMPONENTS, 256);
```

## Features

- `std` (default): Enable standard library support
- `alloc`: Enable alloc crate support
- `metrics`: Enable boot metrics collection
- `verbose`: Enable verbose boot logging
- `baremetal`: Phase B bare metal (no std, no libc)

## Integration with RuVix

This crate integrates with:

- `ruvix-types`: Core type definitions
- `ruvix-cap`: Boot capability set and initial capabilities
- `ruvix-region`: Region creation for kernel objects
- `ruvix-proof`: Boot attestation generation

## License

MIT OR Apache-2.0
