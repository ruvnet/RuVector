# ADR-291: Runtime Compatibility and Version Negotiation

- **Status**: Implemented
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — compatibility-matrix.json v1 + CLI enforcement (compat.ts, FORGE_E_UNSUPPORTED_TARGET with closest-match) + contract fields in canonical manifests (rvf-forge-core) landed on feat/rvf-forge.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-285, ADR-286, ADR-287, ADR-288, ADR-289, ADR-290, ADR-292, ADR-293
- **Tags**: forge, rvm, rvf, compatibility, versioning, negotiation, packaging

## Context

The same signed RVF must run through RVM bare metal, RVM hosted mode, RVM WASM
mode, browser WASM, and the desktop RVF Reader, with identical capability,
witness, state, and lifecycle semantics across every backend. RVForge produces
packages for all of these from one canonical artifact.

That guarantee only holds when the runtime inside a package is a version that
was actually validated against the RVF's schemas. Nothing in a `.exe`, `.dmg`,
or `.deb` inherently records which RVM it embeds, which capability policy it was
built against, or which state and witness schema versions it can read. Without
that record, an installer is an opaque binary and the invariant that build
output be traceable to the input RVF, runtime version, source revision, builder,
and signing identity cannot be checked by anyone downstream.

Two failure shapes follow. A package can embed a runtime that silently
misinterprets a newer RVF, producing divergent evaluation hashes across
platforms. Or Forge can accept a target combination that no one has validated,
shipping an installer that fails on first run. Both are avoided by making the
runtime binding explicit in the package and by refusing to build combinations
that are not in a published matrix.

This ADR defines the Forge to RVM integration contract and the compatibility
gates on both sides of it. The execution semantics of a loaded RVF are ADR-284;
the trust boundary of the build that emits these packages is ADR-290; the
acceleration segments a runtime profile may select are ADR-292; the RVM-specific
output formats are ADR-293.

## Decision

Every Forge-generated package embeds a machine-readable runtime contract. Forge
refuses to build combinations absent from the published RVM compatibility
matrix, and the runtime refuses to load an RVF whose versions it does not
support.

### 1. The embedded contract

Every generated package embeds:

```json
{
  "rvfIdentity": "sha256 value",
  "rvmVersion": "semantic version",
  "rvmCommit": "source revision",
  "runtimeProfile": "wasm",
  "capabilityPolicyHash": "sha256 value",
  "stateSchemaVersion": 1,
  "witnessSchemaVersion": 1
}
```

Field meanings:

- `rvfIdentity` — SHA256 of the canonical RVF. Identical across every platform
  package produced from one build, and unchanged from the submitted artifact.
- `rvmVersion` / `rvmCommit` — the exact runtime semantic version and source
  revision embedded in this package. Both are required; the version alone does
  not identify a build.
- `runtimeProfile` — the runtime family this package was built for, drawn from
  the runtime preference given at submission: WASM, native, Linux microVM, or
  RVM.
- `capabilityPolicyHash` — SHA256 of the capability policy the package was built
  against, covering network, filesystem, devices, models, tools, and memory.
- `stateSchemaVersion` — the state delta and checkpoint schema this runtime
  reads and writes.
- `witnessSchemaVersion` — the witness record schema this runtime emits.

The contract is part of the build provenance record and is covered by the
package signature. It is readable by `npx @ruvector/forge verify` without
executing the RVF payload.

### 2. Published compatibility matrix

RVM publishes a compatibility matrix enumerating the validated combinations of
`rvmVersion`, `runtimeProfile`, `stateSchemaVersion`, `witnessSchemaVersion`,
and target platform. Forge rejects any combination absent from that matrix at
build-manifest generation time, before upload and before a worker is allocated.

Rejection is a stable machine-readable error code from the CLI, naming the
offending field and the nearest supported combination. Forge does not
approximate, downgrade, or substitute a runtime to make an unsupported request
succeed.

The matrix is versioned and hash-addressed. The build provenance record names
the matrix revision consulted, so a past build's admission decision can be
reconstructed.

### 3. Rejection at load

Independently of build-time gating, the runtime rejects incompatible RVF
versions at load. `rvm-rvf` verifies the root manifest before allocating
executable memory and refuses execution when the RVF requires unsupported
capabilities or declares schema versions this runtime does not implement.

Load-time checks, in order:

1. Root manifest signature and hash verify.
2. Declared `stateSchemaVersion` and `witnessSchemaVersion` are supported.
3. `capabilityPolicyHash` matches the policy the package was built against.
4. Required capability classes are all implementable on this host.
5. Every referenced segment verifies before it is loaded.

A failure at any step produces a witness record for the verification result and
refuses execution. Build-time gating and load-time rejection are independent
defenses; neither is permitted to be skipped because the other exists.

### 4. Runtime selection under the contract

The reader selects the strongest compatible runtime in this order:

```text
Native RVM
Operating system isolation plus WASM
WASM
Linux microVM
Unsupported
```

The exact order may be changed by signed policy. Selection is constrained by
the embedded `runtimeProfile`: the reader may select a runtime at or below the
profile the package was built and validated for, and never above it. A package
built for the WASM profile does not opportunistically promote itself to native
RVM at run time.

When no compatible runtime is available on the host, the outcome is
`Unsupported` and the reader reports it as a first-class result rather than
degrading to an unvalidated path.

### 5. Platform compatibility targets

Forge supports, and the matrix enumerates, these targets:

| Target | Notes |
|---|---|
| Windows 10 and 11, x64 | NSIS `.exe` and `.msi` |
| Windows 11, ARM64 | NSIS `.exe` and `.msi` |
| macOS 13 or later, Intel | `.app` and `.dmg`, signed and notarized |
| macOS 13 or later, Apple Silicon | `.app` and `.dmg`, signed and notarized |
| macOS universal | Intel plus ARM64 in one package |
| Ubuntu 22.04 and 24.04 | `.deb`, GPG signed |
| Debian 12 or later | `.deb`, GPG signed |
| Generic Linux | `.AppImage` |
| Browser | WebAssembly execution |
| RVM appliances | native `.rvf` output, see ADR-293 |

A target absent from this table is not buildable, not because it is technically
impossible but because no validated runtime combination exists for it.

### 6. Updates and lineage

Updates are signed, version constrained, reversible, and linked to the prior
RVF identity. An update's embedded contract is checked against the installed
package's contract before the update is applied: an update may not silently move
a package across a `stateSchemaVersion` or `witnessSchemaVersion` boundary that
would strand existing state. Cross-schema migration is an explicit, reversible
operation, not a side effect of an update.

State deltas are bound to the base RVF identity and state from an unrelated RVF
lineage is rejected; the contract's `rvfIdentity` field is what makes that check
possible on the desktop side.

## Acceptance criteria

1. One signed RVF runs unchanged on hosted Linux, Windows, macOS, QEMU, and
   bare-metal RVM, and produces identical deterministic evaluation hashes.
2. Every package produced by one build carries the same `rvfIdentity`, and that
   value equals the SHA256 of the submitted RVF.
3. A build request naming a combination absent from the published compatibility
   matrix is rejected before upload, with a stable machine-readable error code.
4. The build provenance record names the compatibility-matrix revision that
   admitted the build.
5. An RVF declaring an unsupported `stateSchemaVersion` or
   `witnessSchemaVersion` is refused at load, with a witness record for the
   refusal.
6. An RVF requiring capabilities the host cannot implement is refused at load
   rather than partially granted.
7. A modified runtime, policy, model, or state segment is rejected.
8. `forge verify` reads the embedded contract from a produced installer without
   executing the RVF payload.
9. A package built for the WASM profile never selects native RVM at run time.
10. A host with no compatible runtime reports `Unsupported` rather than
    executing on an unvalidated path.
11. An update that would cross a state or witness schema boundary is refused as
    an update and requires an explicit reversible migration.
12. Installers built for each row of the platform table complete successfully on
    a clean operating system.

## Consequences

### Positive

- Every installed package can state exactly which runtime it contains, which is
  what makes identical cross-platform evaluation hashes auditable rather than
  asserted.
- Unsupported combinations fail at manifest generation, before any compute is
  spent.
- Load-time rejection keeps the guarantee even for packages built before a
  matrix change.
- Update lineage checks prevent silent state stranding.

### Negative

- The compatibility matrix is an artifact that must be maintained, validated,
  and published on every RVM release, and it gates Forge availability.
- Refusing to approximate an unsupported target means some requests simply fail
  where a best-effort build would have produced something.
- Two independent gates duplicate some checking cost on every load.
- Capping runtime selection at the built profile forfeits opportunistic use of a
  stronger runtime that happens to be present.

## Alternatives Considered

- **Version the runtime only by semantic version**: rejected because a semantic
  version does not identify a build; `rvmCommit` is required for reproducibility.
- **Negotiate the runtime dynamically at first launch**: rejected because it
  moves compatibility decisions to end-user machines where they cannot be
  validated or witnessed in advance.
- **Gate only at build time**: rejected because packages outlive matrix
  revisions and an installed package must still refuse an incompatible RVF.
- **Gate only at load time**: rejected because it spends full build cost on
  combinations known in advance to be unsupported.
- **Allow runtime promotion above the built profile**: rejected because the
  promoted path was never validated for that package, breaking the identical
  evaluation-hash guarantee.
- **Carry the contract outside the signature**: rejected because an unsigned
  contract can be edited to claim any runtime.

## Implementation Surfaces

- `rvm-rvf` — manifest reading, version and capability rejection at load
- `rvm-launch` — `rvm inspect`, `rvm verify`
- `rvm-ffi` — `rvm_validate`, `rvm_inspect`
- `rvm-node` — bindings consumed by `@ruvector/forge`
- `@ruvector/forge` — `validate`, `build`, `submit`, `verify`
- published RVM compatibility matrix, versioned and hash-addressed
- JSON Schema for the embedded runtime contract
- build provenance record and verification report
