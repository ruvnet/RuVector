# ADR-287: WASM Component Model Integration for the RVM Runtime

- **Status**: Proposed
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — entirely rvm-runtime scope (cross-repo, ruvnet/rvm): streaming validation replacing the 1MB limit, component model, quotas. This repo carries only the contract expectations (compatibility matrix wasm profile).
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-285, ADR-286, ADR-288, ADR-289
- **Tags**: rvm, wasm, component-model, wit, sandboxing, determinism, quotas, witness

## Context

RVForge converts one canonical `.rvf` artifact into installable packages for
Windows, macOS, Linux, and RVM. The same signed RVF must execute through RVM
bare metal, RVM hosted mode, RVM WASM mode, browser WASM, and the desktop RVF
Reader, with identical capability, witness, state, and lifecycle semantics on
every backend.

The current `rvm-wasm` module cannot carry that contract. It limits modules to
a fixed 1 MB, which is far below a practical agent runtime that bundles an
interpreter, tool adapters, and interface code. The limit is a compile-time
constant rather than a policy input, so a publisher cannot raise it for a
legitimately large component, and an operator cannot lower it for a hostile
tenant.

The module boundary is also the wrong unit of composition. An RVF describes an
agent, a model, memory, an interface, and policies. Expressing that as a single
flat WASM module forces every concern into one linear memory with one export
table, which makes per-concern capability scoping impossible and makes
inter-component isolation a matter of convention rather than enforcement.

Two further gaps block the acceptance test. First, deterministic evaluation
hashes across backends require that ambient nondeterminism — wall clock,
randomness, environment — is not silently available. Second, "cannot access
undeclared files, networks, memory, devices, or agents" requires that a denied
request is an observable, recorded event, not a silent failure inside guest
code.

## Decision

The RVM WASM runtime adopts the WebAssembly Component Model with WIT-described
interfaces, replaces the fixed size limit with streaming validation under
signed policy, denies all host access by default, and records every capability
decision in `rvm-witness`.

### 1. Streaming validation replaces the fixed size limit

The 1 MB production limit is removed. It is replaced by:

1. **Streaming validation.** Modules and components are validated incrementally
   as bytes arrive from the RVF segment reader. Validation does not require the
   whole artifact to be resident, so a large model-adjacent component does not
   force a proportional buffer allocation before the first structural error is
   detected.
2. **Policy-controlled limits.** Maximum component bytes, maximum decoded
   size, maximum linear-memory pages, and maximum component count are inputs
   from the signed capability policy, not constants. The policy hash is part of
   the Forge integration contract (`capabilityPolicyHash`).
3. **Fail-closed defaults.** When policy omits a limit, the runtime applies a
   conservative built-in default rather than an unbounded one. An RVF that
   needs more must say so in signed policy.

Validation completes before any executable memory is allocated for the
component, consistent with the RVF loading rule that the root manifest is
verified before executable memory allocation and every referenced segment is
verified before it is loaded.

### 2. Component Model and WIT interfaces

RVM targets the WASM Component Model rather than core modules alone.

- A single RVF may carry **multiple components** — for example an agent core, a
  tool adapter, and an interface component — each as its own verified segment.
- Component boundaries are described by **WIT interfaces**. Imports and exports
  are typed, so the host knows exactly which host functions a component asks
  for before instantiation, and can refuse instantiation when an import has no
  corresponding granted capability.
- Host capabilities are exposed only as WIT imports. There is no ambient import
  namespace that a component can reach without being declared.
- Components communicate through their declared WIT interfaces only. There is
  no shared linear memory between components.

Because imports are typed and enumerable before execution, the runtime performs
**import reconciliation** at load time: every import is matched against the
granted capability set, and an unmatched import is a load-time rejection, not a
runtime trap. This is what makes "refuse execution when the RVF requires
unsupported capabilities" a pre-execution decision.

### 3. Default-deny host surface

The runtime grants **no** filesystem, network, environment, clock, randomness,
GPU, or device access by default. Each of these is a distinct capability class
under the RVF capability schema (ADR-286) and must be explicitly granted by the
signed policy.

Every external operation passes through the existing `rvm-security` sequence
and no other path:

```text
Capability check → Proof verification → Witness recording → Operation
```

The WASM host bindings are implemented as callers of that sequence. A host
function that bypasses the gate is a defect, not an optimization; the gate
remains the only privileged execution path.

### 4. Deterministic clock and randomness modes

To satisfy "produces identical deterministic evaluation hashes" across
backends, the runtime provides two nondeterminism sources in explicitly
selectable modes:

| Source | Default | Deterministic mode | Ambient mode |
|---|---|---|---|
| Clock | denied | virtual clock advanced only by declared runtime events | host wall clock, requires the clock capability |
| Randomness | denied | seeded stream derived from the run seed | host entropy, requires the randomness capability |

In deterministic mode the virtual clock is not the host clock scaled or offset;
it advances only through runtime-defined steps, so two backends executing the
same component with the same inputs observe the same time sequence. The seeded
randomness stream is derived deterministically from a recorded seed, and the
seed is part of the witness record so a run can be reproduced.

Selecting ambient clock or ambient randomness is a capability grant, is
recorded, and disqualifies the run from deterministic-hash comparison. The
runtime labels such runs rather than silently producing divergent hashes.

### 5. Quotas

The runtime enforces, per component instance:

```text
memory      linear-memory pages and growth ceiling
instructions cumulative executed-instruction budget
wall time   maximum elapsed execution time
storage     bytes writable to persistent state
invocations number of exported-function calls accepted
```

Quotas come from signed policy with fail-closed defaults. Exhausting a quota
suspends or terminates the instance according to policy and produces a witness
record naming the exhausted quota. Quota exhaustion is a normal, observable
outcome — not a crash, and not something guest code can catch and hide.

Instruction and wall-time budgets are enforced independently because they fail
differently: an instruction budget bounds a compute-bound loop deterministically
across backends, while a wall-time budget bounds a component blocked on a
granted host call. Deterministic-hash runs are gated on the instruction budget
so that host scheduling variation does not change the outcome.

### 6. Lifecycle: suspend, snapshot, migrate, resume, terminate

The runtime supports the full instance lifecycle:

- **suspend** — halt at an instruction boundary with all guest state intact.
- **snapshot** — produce a `CompressedCheckpoint` of the suspended instance.
- **migrate** — move a snapshot to a different backend among hosted Linux,
  Windows, macOS, QEMU, and bare metal RVM.
- **resume** — reconstruct from a checkpoint plus witness deltas (ADR-288) and
  continue.
- **terminate** — destroy the instance and release its quota reservations,
  emitting a terminal witness record.

Snapshots capture linear memory, component instance state, quota consumption,
and the deterministic clock and randomness cursors. Without the cursors, a
resumed instance would restart its virtual time or its seeded stream and break
deterministic reconstruction. Snapshot identity is bound to the base RVF
identity, so a checkpoint from an unrelated lineage is rejected on resume.

Suspend-on-one-backend / resume-on-another is a required acceptance behavior,
so the snapshot format is backend-independent and versioned by
`stateSchemaVersion`.

### 7. Witnessed capability requests and denials

`rvm-witness` records **both** granted and denied capability requests. A denial
is a first-class record containing the requesting component, the requested
capability class, the policy decision, and the time (virtual or ambient, as
selected).

This matters for two acceptance items. "Undeclared capabilities must remain
inaccessible" is verified by observing denial records rather than by trusting
absence of effect. "Produces a complete cryptographically verifiable witness
chain" requires that the chain has no gaps, including for operations that never
happened because they were refused.

### 8. Inter-component memory isolation

One agent component must not read another component's memory. The runtime
enforces this structurally:

- Each component instance owns its own linear memory. No shared memory is
  configured between components in the same RVF.
- Cross-component calls pass values through WIT-typed interfaces; the runtime
  copies data across the boundary rather than aliasing it.
- Handles to host resources are not forgeable integers in a shared table; a
  component can only use resources granted to it.
- Inter-agent messaging is itself a capability class and passes through the
  same security gate as any other external operation.

Native extensions are out of scope for this ADR's isolation guarantee: they
must never load directly into the RVF Reader process and require a separate
sandbox or RVM partition (ADR-289).

## Acceptance criteria

1. A WASM component larger than 1 MB loads and executes when signed policy
   permits its size; the same component is rejected when policy sets a lower
   ceiling.
2. Streaming validation rejects a structurally invalid component without
   allocating executable memory for it.
3. An RVF carrying multiple components instantiates all of them, and each
   component's imports are reconciled against granted capabilities before
   instantiation.
4. A component importing an undeclared host interface fails at load time with a
   capability-mismatch result, not at first call.
5. With no capability grants, attempts to reach filesystem, network,
   environment, clock, randomness, GPU, or device surfaces are denied, and each
   denial appears in the witness record.
6. Two backends running the same component under deterministic clock and
   seeded randomness produce identical evaluation hashes.
7. A run that requests ambient clock or ambient randomness is recorded as such
   and is not compared against deterministic-hash baselines.
8. Exceeding the memory, instruction, wall-time, storage, or invocation quota
   suspends or terminates the instance per policy and emits a witness record
   naming the exhausted quota.
9. An instance suspends on one backend, snapshots, migrates, and resumes on
   another backend with identical reconstructed state, including virtual-clock
   and randomness cursors.
10. A checkpoint whose base RVF identity does not match the target RVF is
    rejected on resume.
11. A component attempting to read another component's linear memory fails; no
    configuration exposes shared memory between components of the same RVF.
12. The complete witness chain for a run verifies cryptographically and
    contains both granted and denied capability events.

## Consequences

### Positive

- Practical agent runtimes become expressible; the 1 MB constant no longer
  dictates architecture.
- Typed WIT imports turn capability enforcement into a load-time structural
  check instead of a runtime hope.
- Deterministic clock and randomness make cross-backend hash equality
  achievable rather than accidental.
- Denial records make "undeclared capabilities are inaccessible" auditable.
- Multi-component RVFs allow per-concern capability scoping — an interface
  component need not hold the agent core's grants.

### Negative

- The Component Model and WIT tooling are a heavier dependency than raw core
  modules, and browser WASM support requires careful target selection.
- Cross-component value copying costs more than shared memory would.
- Instruction-budget accounting adds measurable execution overhead.
- Snapshot format must be maintained as a versioned, backend-independent
  contract, which constrains future runtime internals.
- Policy-controlled limits move a class of failures from compile time to
  deployment time, so policy authoring errors become a real operational risk.

## Alternatives Considered

- **Raise the fixed limit to a larger constant**: rejected because it moves the
  wall without removing it and still gives operators no way to tighten limits
  for untrusted tenants.
- **Keep core modules and enforce isolation by convention**: rejected because
  inter-component memory isolation would be unenforceable and import
  reconciliation impossible.
- **Grant a default POSIX-like host surface and restrict it afterward**:
  rejected because it inverts the default-deny requirement and makes every
  omission a silent grant.
- **Use the host wall clock with a monotonic shim**: rejected because host
  scheduling variance leaks into evaluation hashes across backends.
- **Record only granted capability operations in the witness**: rejected
  because it leaves no evidence for the denial acceptance criteria.
- **Enforce only wall-time budgets**: rejected because wall time varies by
  backend and cannot bound a compute loop deterministically.

## Implementation Surfaces

- `rvm-wasm` — streaming validator, Component Model instantiation, quota
  enforcement, deterministic clock and randomness modes
- `rvm-rvf` — segment resolution and per-component verification before load
- `rvm-cap` — capability classes and grant reconciliation against WIT imports
- `rvm-security` — the capability/proof/witness gate wrapping every host call
- `rvm-witness` — granted and denied capability records, quota-exhaustion
  records, run seed recording
- `rvm-policy` — signed limits for size, memory, instructions, wall time,
  storage, invocations, and component count
- `rvm-state` — `CompressedCheckpoint` capture and restore including clock and
  randomness cursors
- WIT interface definitions for host capability classes
