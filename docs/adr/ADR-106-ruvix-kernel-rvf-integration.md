# ADR-106: RuVix Kernel Integration with RVF

| Field         | Value                                      |
|---------------|--------------------------------------------|
| **Status**    | Proposed                                   |
| **Date**      | 2026-03-15                                 |
| **Deciders**  | RuVector Core Team                         |
| **Relates to**| ADR-087 (RuVix Cognition Kernel), ADR-031 (RVF Format) |

## Context

The RuVector project contains two major subsystems that deal with the RVF (RuVector Format):

1. **RuVix Cognition Kernel** (`crates/ruvix/`) — A bare-metal microkernel with 12 syscalls, capability-gated resources, and a 5-stage boot sequence. It is organized as a dedicated Cargo workspace with 22 internal crates (`ruvix-types`, `ruvix-nucleus`, `ruvix-boot`, `ruvix-cap`, `ruvix-proof`, `ruvix-sched`, `ruvix-region`, `ruvix-queue`, `ruvix-vecgraph`, `ruvix-hal`, `ruvix-drivers`, `ruvix-smp`, `ruvix-physmem`, `ruvix-dma`, `ruvix-dtb`, `ruvix-net`, `ruvix-fs`, `ruvix-shell`, `ruvix-cli`, etc.).

2. **RVF Format Stack** (`crates/rvf/`) — The file-format and runtime for RVF vector stores. It is organized as a dedicated Cargo workspace with 17+ sub-crates (`rvf-types`, `rvf-runtime`, `rvf-kernel`, `rvf-index`, `rvf-quant`, `rvf-wire`, `rvf-crypto`, `rvf-manifest`, `rvf-ebpf`, `rvf-launch`, `rvf-wasm`, `rvf-import`, `rvf-federation`, `rvf-node`, `rvf-server`, `rvf-adapters`, `rvf-cli`).

### Current Integration State

The dependency relationship is **uni-directional and informal**:

- **RuVix → RVF**: RuVix references RVF concepts extensively (45 source files mention "rvf"), but does so through its *own* re-implemented types (`ruvix-types::rvf::RvfMountHandle`, `RvfComponentId`, `RvfVerifyStatus`, `WitTypeId`). These are independent `#[repr(C)]`/`#[repr(transparent)]` structs that do not depend on `rvf-types`.

- **RVF → RuVix**: Zero references. The RVF stack has no knowledge of the kernel.

- **Parallel type systems**: Both stacks define kernel-related types independently:
  - `rvf-types::kernel` defines `KernelHeader`, `KernelArch`, `KernelType`, `KernelBinding`, segment flags, and wire-format constants.
  - `ruvix-types::rvf` defines `RvfMountHandle`, `RvfComponentId`, `RvfVerifyStatus` — runtime abstractions for the kernel's mount syscall.

- **`rvf-kernel`** crate builds real Linux bzImage/initramfs images and embeds them into RVF files using `rvf-types::kernel::KernelHeader`. It is a *build-time* tool, not a runtime dependency of the ruvix kernel.

### Key Integration Points

| RuVix Subsystem | RVF Subsystem | Integration Point |
|-----------------|---------------|-------------------|
| `ruvix-boot` (Stage 1: RVF Verify) | `rvf-manifest`, `rvf-crypto` | Manifest parsing + ML-DSA-65 signature verification |
| `ruvix-boot` (Stage 3: Component Mount) | `rvf-wasm` | WASM component loading from RVF segments |
| `ruvix-nucleus::Syscall::RvfMount` | `rvf-runtime` | Runtime package mounting |
| `ruvix-types::rvf` | `rvf-types::kernel` | Parallel type definitions with no shared code |
| `ruvix-nucleus::VectorStore` | `rvf-runtime::RvfStore` | Both manage vectors; kernel's is in-memory, RVF's is on-disk |
| `ruvix-boot::WitnessLog` | `rvf-runtime::witness` | Both implement witness/attestation logs independently |

### Problems

1. **Type divergence**: `ruvix-types::rvf` and `rvf-types` define the same concepts (`RvfVerifyStatus`, mount handles, component IDs) with incompatible representations. Converting between them requires manual mapping.

2. **Duplicate witness implementations**: `ruvix-boot::WitnessLog` and `rvf-runtime::witness` both implement cryptographically-linked append-only logs with no shared code.

3. **No shared manifest format**: `ruvix-boot::manifest::RvfManifest` parses a `RVF1`-prefixed manifest, while `rvf-manifest` defines the canonical RVF manifest format. These are likely incompatible.

4. **Kernel image embedding is disconnected**: `rvf-kernel` builds Linux kernel images and creates `KernelHeader` structs for embedding in RVF files, but `ruvix-boot` does not consume these headers — it has its own boot verification path.

5. **No runtime bridge**: When the ruvix kernel mounts an RVF package at runtime (`Syscall::RvfMount`), it does not use `rvf-runtime::RvfStore` to read the package. The mount implementation in `ruvix-boot::mount::RvfMount` is a standalone implementation.

## Decision

Adopt a **shared-types bridge** architecture with three layers:

### Layer 1: Shared Wire Types (`rvf-types` as the canonical source)

`rvf-types` becomes the single source of truth for all wire-format types used by both stacks. Specifically:

- `rvf-types` already provides `KernelHeader`, `KernelArch`, `KernelType`, `KernelBinding`, segment types, and flags.
- Add `RvfMountHandle`, `RvfComponentId`, `RvfVerifyStatus`, and `WitTypeId` to `rvf-types` (or a new `rvf-types::mount` module), since these are format-level concepts.
- `ruvix-types::rvf` re-exports from `rvf-types` instead of defining its own structs. This is opt-in via a `rvf-compat` feature flag so the ruvix kernel can still build in `no_std` without pulling in `rvf-types::std`.

### Layer 2: Manifest & Signature Convergence

- `ruvix-boot::manifest::RvfManifest` delegates to `rvf-manifest` for parsing the canonical manifest format. The kernel-specific boot manifest is a *subset* of the full RVF manifest.
- `ruvix-boot::signature::SignatureVerifier` delegates to `rvf-crypto` for ML-DSA-65 verification.

### Layer 3: Runtime Bridge (`rvf-runtime` adapter in ruvix)

- A new module `ruvix-nucleus::rvf_bridge` (or `ruvix-boot::rvf_bridge`) acts as an adapter between the kernel's mount syscall and `rvf-runtime::RvfStore`.
- The bridge translates kernel-internal handle types to RVF store operations.
- The bridge is feature-gated (`feature = "rvf-runtime"`) so the kernel can still run standalone (e.g., on bare metal without filesystem access).

### What Does NOT Change

- The ruvix kernel retains its own `#[no_std]`-compatible internal type system for syscall dispatch.
- `rvf-kernel` (build-time Linux kernel embedding) remains independent — it is not a runtime dependency.
- The ruvix kernel's in-memory `VectorStore` remains separate from `rvf-runtime::RvfStore` (different data planes).

## Consequences

### Positive

- **Single source of truth** for wire types eliminates the risk of format incompatibility between kernel boot images and RVF files.
- **Real manifest parsing** in the kernel boot path means ruvix can boot from actual RVF packages rather than a parallel manifest format.
- **Reduced code duplication** in witness logging and signature verification.
- **Feature-gated integration** preserves the kernel's ability to run in `no_std`/bare-metal environments.

### Negative

- **Build complexity**: `ruvix-types` gains a dependency on `rvf-types` (behind a feature flag), adding cross-workspace dependency management.
- **Version coupling**: Changes to `rvf-types` wire formats now affect the kernel. This is mitigated by `rvf-types`'s existing stability guarantees (it is at v0.2.0 and published to crates.io).
- **Migration effort**: Existing ruvix tests (45+ files) that reference `ruvix-types::rvf::*` need updating to use the re-exported types.

### Risks

- **`no_std` compatibility**: `rvf-types` must remain `no_std`-compatible (it already has `default-features = []` with `std` as opt-in). This must be verified before the ruvix kernel takes the dependency.
- **Circular workspace dependencies**: Since both live in separate Cargo workspaces within the same repo, cross-workspace `path` dependencies require careful version management for crates.io publishing.

## Implementation Plan

| Phase | Scope | Effort |
|-------|-------|--------|
| **Phase 1** | Add mount-related types to `rvf-types`. Feature-gate `ruvix-types` to re-export from `rvf-types`. Update ruvix tests. | S |
| **Phase 2** | Replace `ruvix-boot::manifest` parser with `rvf-manifest` delegation. Replace `ruvix-boot::signature` with `rvf-crypto` delegation. | M |
| **Phase 3** | Implement `rvf_bridge` adapter in ruvix-nucleus for runtime mount operations using `rvf-runtime`. | M |
| **Phase 4** | Unify witness log implementations (extract shared trait to `rvf-types`). | S |

## Alternatives Considered

### Alternative A: Full Merge

Merge `ruvix-types` and `rvf-types` into a single crate. Rejected because:
- `rvf-types` is published on crates.io and used externally.
- `ruvix-types` is `#[no_std]` first with bare-metal constraints.
- Merging would force all RVF users to pull in kernel-specific types.

### Alternative B: Status Quo (Keep Separate)

Continue with independent type systems and manual mapping. Rejected because:
- Type divergence is already causing inconsistencies (mount handle layout, verify status codes).
- Duplicate witness/signature code increases maintenance burden.
- Boot-from-real-RVF is blocked without manifest convergence.

### Alternative C: RVF as a Runtime Dependency of RuVix

Make `rvf-runtime` a direct dependency of `ruvix-nucleus`. Rejected because:
- `rvf-runtime` requires `std` (filesystem, I/O).
- The ruvix kernel must remain `no_std`-compatible for bare-metal targets.
- A bridge adapter (Layer 3) provides the same benefit with cleaner boundaries.
