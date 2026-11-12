# ADR-284: RVF Execution Contract for RVM Backends

- **Status**: Accepted
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — contract side implemented on feat/rvf-forge: rvf-forge-core enforces verify-before-load, per-segment verification, unsigned-executable rejection, witness-per-verification (103 tests); Reader consumes it via path dep (113 tests). Runtime backends (rvm hosted/bare-metal/WASM execution) are cross-repo in ruvnet/rvm.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-280, ADR-283, ADR-285, ADR-286
- **Tags**: rvf, rvm, execution, wasm, verification, witness, portability

## Context

ADR-283 makes one canonical `.rvf` the input to every platform installer. That
only means something if the artifact also *behaves* the same everywhere it
runs. A signed RVF that yields different results on a desktop reader than on a
bare-metal appliance is not a portable agent; it is five agents that happen to
share a filename.

RVM already provides partitions, capabilities, witnesses, proof gates,
scheduling, memory management, measured boot, and a WASM agent lifecycle. What
it does not yet expose is a complete desktop host runtime that an `.exe`,
`.dmg`, or `.deb` package can invoke. The existing `rvm-wasm` module also caps
modules at 1 MB, which is far too small for a practical agent runtime.

There is a second, sharper problem. Packaging and inspection tools handle
untrusted RVFs constantly — Forge workers scan them, `rvm inspect` reads them,
the CLI validates them before upload. Any path where "reading" an RVF can
become "running" an RVF turns every one of those tools into an execution
surface for confidential third-party code.

## Decision

Define an **RVF execution contract**. The same signed RVF must run unchanged
through every backend:

```text
RVM bare metal
RVM hosted mode
RVM WASM mode
Browser WASM
Desktop RVF Reader
```

RVM must provide **identical capability, witness, state, and lifecycle
semantics across every backend**. A backend that cannot provide those
semantics is not a conforming backend, and Forge must not emit a package
targeting it.

### 1. RVF loading requirements

These apply to every backend and to every tool that opens an RVF:

1. **Verify the root manifest before allocating executable memory.**
   Allocation follows verification, never precedes it.
2. **Verify every referenced segment before loading it.** Root-manifest
   validity is not transitive trust for segment contents.
3. **Reject unsigned executable segments by default.**
4. Support progressive loading for large models.
5. Support encrypted segments.
6. Support architecture-specific acceleration segments.
7. **Never execute RVF content during inspection or packaging.** This is what
   makes `rvm inspect`, `forge validate`, and Forge's package scanning safe to
   point at untrusted artifacts, and it is the runtime half of ADR-283's
   "build workers never execute the submitted RVF" invariant.
8. Enforce maximum model, runtime, memory, and state sizes through signed
   policy.
9. **Produce a witness record for every verification result** — successes and
   failures alike. A rejected RVF is an auditable event, not a silent error.
10. **Refuse execution when the RVF requires unsupported capabilities.**
    Degrading to a partial capability set is not permitted; see ADR-286.

### 2. `rvm-rvf` responsibilities

A new crate, `rvm-rvf`, owns the boundary between the format and the machine:

1. Reading RVF manifests.
2. Verifying RVF signatures and hashes.
3. Resolving runtime segments.
4. Loading models, memory, policies, and WASM components.
5. Rejecting incompatible RVF versions.
6. Mapping RVF capabilities into RVM capability tables (ADR-286).
7. Preserving the canonical RVF identity.

It sits alongside `rvm-host` (per-OS adapters, ADR-285), `rvm-launch`
(lifecycle commands), `rvm-ffi` (stable C interface for Tauri and other native
hosts), `rvm-node` (Node bindings for `@ruvector/forge`), `rvm-policy`, and
`rvm-state`.

Lifecycle commands exposed by `rvm-launch`:

```bash
rvm inspect agent.rvf
rvm verify agent.rvf
rvm run agent.rvf
rvm suspend INSTANCE_ID
rvm resume INSTANCE_ID
rvm checkpoint INSTANCE_ID
rvm witness INSTANCE_ID
rvm terminate INSTANCE_ID
```

The `rvm-ffi` surface mirrors it: `rvm_validate`, `rvm_inspect`, `rvm_create`,
`rvm_start`, `rvm_suspend`, `rvm_resume`, `rvm_checkpoint`,
`rvm_export_witness`, `rvm_terminate`. Note that `inspect` and `validate` are
first-class operations distinct from `create`/`start` precisely because
requirement 7 forbids the former from implying the latter.

### 3. WASM runtime requirements

1. **Remove the fixed 1 MB production limit.**
2. Validate modules through streaming input.
3. Support multiple WASM components per RVF.
4. Support the WASM Component Model and WIT interfaces.
5. Default to no filesystem, network, environment, clock, randomness, GPU, or
   device access.
6. Provide deterministic virtual clock and seeded randomness modes — these are
   what make identical evaluation hashes achievable across backends.
7. Enforce memory, instruction, wall time, storage, and invocation quotas.
8. Support suspend, snapshot, migration, resume, and termination.
9. Record capability requests and denials in `rvm-witness`.
10. Prevent one agent component from reading another component's memory.

### 4. State semantics

The base RVF is immutable. Changes are stored in encrypted RVF delta segments,
using `CompressedCheckpoint` for execution snapshots and `WitnessDelta` for
reconstruction. Every delta is bound to the base RVF identity, and state from
an unrelated RVF lineage is rejected. State supports branch, rollback, merge,
migrate, and reset; it can be deleted without deleting the base RVF; it
supports customer-controlled encryption keys; and it must reconstruct
identically across compatible platforms.

### 5. Version negotiation

Each package embeds `rvfIdentity`, `rvmVersion`, `rvmCommit`,
`runtimeProfile`, `capabilityPolicyHash`, `stateSchemaVersion`, and
`witnessSchemaVersion`. Forge rejects combinations absent from the published
RVM compatibility matrix rather than shipping a package whose runtime
semantics are unverified.

Beyond desktop installers, conforming RVM outputs are `Agent.rvf`,
`Agent.rvm.img`, `Agent.rvm.efi`, `Agent.qemu.img`, and
`Agent.appliance.bundle`. Bare-metal outputs use the existing deterministic
seven-phase measured boot sequence provided by `rvm-boot`.

## Acceptance criteria

A release passes only when one signed RVF:

1. Runs unchanged on hosted Linux, Windows, macOS, QEMU, and bare-metal RVM.
2. Produces identical deterministic evaluation hashes.
3. Cannot access undeclared files, networks, memory, devices, or agents.
4. Suspends on one backend and resumes on another.
5. Preserves its base RVF identity.
6. Reconstructs state from checkpoint plus witness deltas.
7. Rejects modified runtime, policy, model, and state segments.
8. Produces a complete cryptographically verifiable witness chain.

## Consequences

### Positive

- Portability becomes a testable property rather than a claim: criterion 4
  (suspend on one backend, resume on another) either works or it does not.
- Inspection tooling can be pointed at hostile artifacts safely, which is what
  lets Forge scan packages without executing their payload.
- Deterministic clock and seeded randomness give reproducible evaluation
  hashes, which downstream provenance and audit depend on.
- A single `rvm-ffi` surface serves Tauri, the CLI, and Node bindings without
  three divergent integrations.

### Negative

- "Identical semantics across five backends" is a demanding contract; each new
  backend costs conformance work, not just a port.
- Verify-before-allocate and per-segment verification add startup latency that
  must fit inside the 500 ms pre-model-load reader budget.
- Removing the 1 MB WASM cap requires streaming validation and real quota
  enforcement to replace what the cap was crudely providing.
- Deterministic modes constrain how the runtime may expose time and entropy,
  which limits some optimizations.

## Alternatives Considered

- **Let each backend define its own capability and witness semantics**:
  rejected because it makes the signed artifact's behavior unpredictable and
  destroys the audit value of the witness chain.
- **Keep the 1 MB WASM limit and split large agents into many modules**:
  rejected as an artificial constraint that pushes complexity onto every
  publisher; streaming validation with policy-controlled limits is the fix.
- **Verify lazily as segments are touched**: rejected because it means
  executable memory can be allocated for content that has not been verified.
- **Allow inspection to instantiate a module for richer metadata**: rejected;
  richer metadata is not worth turning every scanner into an execution surface.
- **Degrade gracefully when a required capability is unsupported**: rejected
  in favor of refusing execution, so an agent never runs in a silently
  weakened configuration.

## Implementation Surfaces

```text
rvm-rvf      manifest reading, signature/hash verification, segment resolution
rvm-host     Windows / macOS / Linux / Browser / QEMU / bare metal adapters
rvm-launch   inspect, verify, run, suspend, resume, checkpoint, witness, terminate
rvm-ffi      stable C interface for Tauri and other native hosts
rvm-node     Node bindings used by @ruvector/forge
rvm-policy   signed size and capability policy enforcement
rvm-state    immutable base plus encrypted delta segments
rvm-wasm     streaming validation replacing the fixed 1 MB limit
rvm-witness  verification and capability-denial records
```

Implementation sequence: land this contract, implement `rvm-rvf`, replace the
WASM size limit with streaming validation, implement capability policy mapping
(ADR-286), implement immutable state deltas and checkpoint reconstruction,
implement `rvm-launch` and `rvm-ffi`, implement Linux hosted mode, add Windows
and macOS host adapters, add Node bindings, connect `@ruvector/forge`, add
bare-metal and appliance outputs, and complete independent security
validation.
