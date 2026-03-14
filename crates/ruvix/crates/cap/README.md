# ruvix-cap

seL4-inspired capability management for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate provides the capability manager that enforces all access control in RuVix. Every kernel object is accessed exclusively through capabilities, following the principle: **"No syscall succeeds without an appropriate capability handle."**

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Capability** | Unforgeable kernel-managed token: object ID, type, rights, badge, epoch |
| **Derivation Tree** | Capabilities can be derived with equal or fewer rights |
| **Delegation Depth** | Maximum depth of 8 to prevent unbounded chains |
| **Epoch-based Invalidation** | Detects stale handles automatically |

## Design Principles (ADR-087 Section 6)

1. A task can only grant capabilities it holds
2. Granted rights must be equal or fewer than held rights
3. Revocation propagates through the derivation tree
4. `GRANT_ONCE` provides non-transitive delegation
5. Epoch-based invalidation detects stale handles

## Components

### CapabilityManager

Central capability management:

```rust
use ruvix_cap::{CapabilityManager, CapManagerConfig};
use ruvix_types::{ObjectType, CapRights, TaskHandle};

let config = CapManagerConfig::default();
let mut manager: CapabilityManager<64> = CapabilityManager::new(config);

// Create a root capability for a new vector store
let task = TaskHandle::new(1, 0);
let cap_handle = manager.create_root_capability(
    0x1000,  // object_id
    ObjectType::VectorStore,
    0,       // badge
    task,
)?;
```

### Capability Granting

Delegate capabilities with restricted rights:

```rust
// Grant a read-only derived capability
let derived = manager.grant(
    cap_handle,
    CapRights::READ,
    42,  // new badge
    task,
    TaskHandle::new(2, 0),  // target task
)?;
```

### Capability Revocation

Revoke capabilities and all derivatives:

```rust
// Revoke cascades through derivation tree
manager.revoke(cap_handle, task)?;
```

### DerivationTree

Track capability relationships:

```rust
use ruvix_cap::DerivationTree;

let tree = manager.derivation_tree();
let children = tree.children_of(cap_handle);
let depth = tree.depth(cap_handle);
```

## Rights Bitmap

```rust
use ruvix_types::CapRights;

// Available rights
let read = CapRights::READ;        // Read access
let write = CapRights::WRITE;      // Write access
let grant = CapRights::GRANT;      // Can grant to others
let revoke = CapRights::REVOKE;    // Can revoke grants
let prove = CapRights::PROVE;      // Can create proofs
let grant_once = CapRights::GRANT_ONCE;  // Single-use grant

// Combine rights
let read_write = CapRights::READ | CapRights::WRITE;
```

## Security Features

### Boot Signature Verification (SEC-001)

```rust
use ruvix_cap::{verify_boot_signature_or_panic, BootVerifier};

// PANICS on failure - no fallback boot path
verify_boot_signature_or_panic(&manifest, &signature, &public_key);
```

### Audit System

```rust
use ruvix_cap::{CapabilityAuditor, AuditConfig, AuditFlags};

let config = AuditConfig {
    flags: AuditFlags::GRANT | AuditFlags::REVOKE,
    depth_warning_threshold: 4,
};
let auditor = CapabilityAuditor::new(config);

// Audit operations are automatically logged
let result = auditor.audit_grant(&grant_request)?;
```

## Constants

```rust
use ruvix_cap::{DEFAULT_MAX_DELEGATION_DEPTH, DEFAULT_CAP_TABLE_CAPACITY, AUDIT_DEPTH_WARNING_THRESHOLD};

assert_eq!(DEFAULT_MAX_DELEGATION_DEPTH, 8);       // Section 20.2
assert_eq!(DEFAULT_CAP_TABLE_CAPACITY, 1024);      // Per-task capacity
assert_eq!(AUDIT_DEPTH_WARNING_THRESHOLD, 4);      // Audit warning level
```

## Features

- `std` (default): Enable standard library support
- `alloc`: Enable alloc crate support
- `audit-log`: Enable audit logging for all capability operations

## Integration with RuVix

This crate integrates with:

- `ruvix-types`: Core type definitions (`CapHandle`, `CapRights`, `Capability`)
- `ruvix-boot`: Boot capability distribution and root task setup
- `ruvix-proof`: PROVE rights checking for proof generation

## License

MIT OR Apache-2.0
