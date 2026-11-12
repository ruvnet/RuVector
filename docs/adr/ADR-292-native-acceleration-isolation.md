# ADR-292: Native Acceleration Isolation

- **Status**: Proposed
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — policy encoded (compatibility matrix acceleration/segment rules; ADR text); enforcement requires rvm partitions/sandboxes — cross-repo.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-285, ADR-286, ADR-287, ADR-288, ADR-289, ADR-290, ADR-291, ADR-293
- **Tags**: rvf, rvm, acceleration, native, isolation, wasm, segments, policy

## Context

WASM gives universal execution but not peak throughput. Practical agent runtimes
want native libraries for acceleration, and an RVF may therefore carry
architecture-specific acceleration segments alongside its portable ones.

Native code is exactly the thing WASM isolation exists to contain. A native
acceleration library loaded into the RVF Reader process shares that process's
address space, file descriptors, and host privileges. At that point the reader's
sandbox describes the WASM component only, and the capability gate that every
external operation is supposed to pass through can be bypassed by any native
segment that chooses to. Undeclared capabilities would remain reachable in
practice while appearing denied in policy.

There is a second, quieter pressure. Acceleration segments and model payloads
are large. A 5 GB installer is a real outcome of embedding them, and loading
multi-gigabyte segments eagerly conflicts with a reader startup budget measured
in hundreds of milliseconds. Whatever isolation is chosen has to coexist with
progressive loading and with encrypted segments.

This ADR governs how acceleration segments are declared, selected, loaded, and
contained. The general execution contract is ADR-284; the hosted RVM security
boundary is ADR-285; the desktop host adapters that implement per-platform
isolation are ADR-289; the compatibility contract that constrains which runtime
profile may select acceleration at all is ADR-291.

## Decision

Architecture-specific acceleration segments are a supported part of the RVF.
Native libraries never load directly into the RVF Reader process. They require
a separate sandbox or an RVM partition, and their selection is bounded by the
runtime-selection ladder and by signed policy.

### 1. Architecture-specific acceleration segments

An RVF may carry acceleration segments targeted at specific architectures
alongside its portable WASM components. `rvm-rvf` resolves runtime segments and
selects the acceleration segment matching the host architecture, or none.

Segment handling follows the general RVF loading rules without exception:

1. The root manifest is verified before any executable memory is allocated.
2. Every referenced segment is verified before it is loaded.
3. Unsigned executable segments are rejected by default; an acceleration
   segment is an executable segment.
4. RVF content is never executed during inspection or packaging.

An acceleration segment whose signature or hash fails verification is not a
degraded-performance case. It is a load failure, and it produces a witness
record like any other verification result.

### 2. Native libraries never load into the reader process

Native extensions must never load directly into the RVF Reader process. A native
acceleration segment executes in one of exactly two places:

- **A separate sandbox process** — a distinct OS process under the host
  adapter's isolation primitives, communicating with the reader over a narrow
  message interface. On Windows this means Job Objects, restricted tokens,
  filesystem restrictions, and outbound network controls; on macOS, application
  sandboxing, hardened runtime, and scoped entitlements; on Linux, namespaces,
  cgroups, seccomp, restricted mounts, and network namespaces.
- **An RVM partition** — partition memory isolation, capability tables, device
  leases, and witnessed security gates, on bare-metal or hosted RVM.

In both placements the native segment holds no ambient authority. Every external
operation it performs passes through the same `rvm-security` sequence as any
other component:

```text
Capability check -> Proof verification -> Witness recording -> Operation
```

That three-stage gate remains the only privileged execution path. A native
segment cannot read another component's memory, and one agent component cannot
read another's, regardless of which side of the sandbox boundary it runs on.

Hosted RVM must not claim bare-metal isolation when executing as a normal
desktop process; a sandboxed native segment on a desktop host is operating-system
isolation, and it is described as such. Independent escape testing is required
before RVM execution is described as hardened isolation.

### 3. Runtime selection: strongest compatible runtime

The reader selects the strongest compatible runtime in this order:

```text
Native RVM
Operating system isolation plus WASM
WASM
Linux microVM
Unsupported
```

The exact order may be changed by signed policy. Acceleration segment selection
is subordinate to this ladder, not parallel to it: the ladder picks the runtime,
and only then does the selected runtime decide whether an acceleration segment
for the host architecture is available and permitted.

Consequences of the ordering:

- Under **Native RVM**, an acceleration segment runs in a partition.
- Under **operating system isolation plus WASM**, an acceleration segment runs
  in a separate sandboxed process, or is skipped.
- Under plain **WASM**, no native acceleration segment is loaded; the portable
  path is used.
- Under **Linux microVM**, acceleration is available only where the microVM's
  isolation covers it.
- **Unsupported** is a reported outcome, never a fallback to an unisolated
  native load.

Absence of an acceleration segment for the host architecture is never an error.
The portable path is always present, and skipping acceleration changes
performance, not results.

### 4. Encrypted segments and progressive loading

RVF loading supports encrypted segments and progressive loading for large
models. Both apply to acceleration segments and to model payloads.

Progressive loading does not weaken verification order. A segment is verified
before it is loaded, and progressive loading verifies each delivered portion
against the manifest rather than deferring verification until the whole segment
has arrived. Executable memory is allocated only after the root manifest
verifies.

Progressive loading is what keeps reader startup under 500 milliseconds before
model loading: the reader reaches a running state without having materialized
multi-gigabyte segments, and pulls them as execution demands them.

Encrypted segments keep model weights and proprietary acceleration code
confidential at rest inside a distributed installer. Decryption happens inside
the isolation boundary that will execute the segment, not in the reader process
that merely routes it.

### 5. Signed policy limits

Maximum model, runtime, memory, and state sizes are enforced through signed
policy. The policy is part of the capability policy covered by
`capabilityPolicyHash` in the ADR-291 contract, so a package cannot be
repackaged with looser limits without invalidating its signature.

Limits enforced through signed policy:

| Limit | Applies to |
|---|---|
| maximum model size | model segments, before and after decryption |
| maximum runtime size | acceleration and runtime segments |
| maximum memory | per-component memory, sandbox and partition |
| maximum state size | encrypted state delta segments |

The fixed 1 MB WASM module limit is removed and replaced by streaming validation
plus these policy-controlled limits. A segment exceeding a policy limit is
refused at load, with a witness record, rather than being truncated or partially
loaded.

Quota enforcement over memory, instructions, wall time, storage, and invocations
applies to sandboxed native segments as it does to WASM components, so an
acceleration path cannot be used to escape the quotas the portable path obeys.

## Acceptance criteria

1. A signed RVF containing acceleration segments produces identical
   deterministic evaluation hashes with acceleration enabled and with the
   portable path only.
2. No native acceleration library is mapped into the RVF Reader process address
   space, verified by inspecting loaded modules of the reader process at run
   time.
3. A sandboxed native segment cannot access undeclared files, networks, memory,
   devices, or agents.
4. One component, native or WASM, cannot read another component's memory.
5. An acceleration segment with an invalid signature or hash fails the load and
   emits a witness record; it does not silently fall back.
6. An unsigned executable segment is rejected by default.
7. A host with no matching acceleration segment runs the portable path and
   produces the same evaluation hashes.
8. A package built for the WASM runtime profile loads no native acceleration
   segment.
9. Progressive loading of a large model verifies each portion against the
   manifest before use, and executable memory is allocated only after root
   manifest verification.
10. Reader startup completes under 500 milliseconds before model loading with
    progressive loading enabled on a multi-gigabyte RVF.
11. A segment exceeding a signed-policy model, runtime, memory, or state limit
    is refused at load with a witness record.
12. Editing a policy limit without re-signing invalidates the capability policy
    hash and the package fails verification.
13. Encrypted segments are decrypted inside the executing isolation boundary,
    and plaintext segment contents are not present in the reader process.
14. Capability requests and denials from sandboxed native segments are recorded
    in `rvm-witness`.

## Consequences

### Positive

- The sandbox description matches reality: no native code holds reader-process
  privileges, so denied capabilities are genuinely unreachable.
- Acceleration becomes a performance decision rather than a security decision,
  because results are identical either way.
- Progressive loading keeps startup fast without weakening verification order.
- Signed size limits close the repackaging path around resource policy.

### Negative

- Cross-process or cross-partition calls add latency that partly offsets the
  acceleration being sought.
- A separate sandbox process per accelerated component adds memory and lifecycle
  complexity to the reader.
- Per-architecture segments multiply what must be built, signed, and validated.
- Decrypting inside the isolation boundary means key material must reach that
  boundary, which constrains how sandboxes are provisioned.
- Encrypted, progressively loaded segments are harder to inspect during support
  investigations.

## Alternatives Considered

- **Load native acceleration libraries directly into the reader with a
  capability shim**: rejected because a native library in the same address space
  can bypass any in-process shim, making the capability policy unenforceable.
- **Ship only WASM and forgo native acceleration**: rejected because it forfeits
  a large throughput margin that architecture-specific segments recover, and the
  isolation problem is solvable.
- **Allow an unisolated native fallback when no sandbox is available**: rejected
  because it converts an availability problem into a silent isolation failure;
  `Unsupported` is the correct outcome.
- **Enforce size limits in the reader configuration rather than signed policy**:
  rejected because unsigned limits can be edited by whoever repackages the
  installer.
- **Keep a fixed module size limit**: rejected because 1 MB is far too small for
  practical agent runtimes; streaming validation plus policy limits replaces it.
- **Verify a progressively loaded segment only once fully received**: rejected
  because it requires buffering multi-gigabyte segments and defeats the startup
  budget progressive loading exists to meet.

## Implementation Surfaces

- `rvm-rvf` — segment resolution, architecture selection, progressive and
  encrypted segment loading
- `rvm-host` — per-platform sandbox process adapters
- `rvm-policy` — signed model, runtime, memory, and state size limits
- `rvm-cap` and `rvm-security` — capability gate for sandboxed native segments
- `rvm-witness` — verification results, capability requests and denials
- `rvf reader` — sandbox process lifecycle and narrow message interface
- escape-testing harness for the sandbox and partition boundaries
