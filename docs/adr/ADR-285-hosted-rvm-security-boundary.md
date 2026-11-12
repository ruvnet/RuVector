# ADR-285: Hosted RVM Security Boundary

- **Status**: Accepted
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — isolation-claim discipline implemented: compatibility-matrix.json encodes per-profile isolationClaim (os-sandbox+wasm never labeled bare-metal); Reader renders claims from the matrix. OS-level sandbox mechanisms themselves are cross-repo rvm-host scope.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-280, ADR-283, ADR-284, ADR-286
- **Tags**: rvm, security, isolation, sandbox, desktop, wasm, claims

## Context

ADR-284 requires the same signed RVF to run across bare-metal RVM, hosted RVM,
RVM WASM, browser WASM, and the desktop RVF Reader with identical capability,
witness, state, and lifecycle semantics. Identical *semantics* is not the same
as identical *isolation strength*, and conflating the two is the most likely
way this system ends up making a security claim it cannot support.

Bare-metal RVM has partition memory isolation, capability tables, device
leases, measured boot, and witnessed security gates — a hypervisor-class
boundary. Hosted RVM is a normal desktop process running under a normal user
account on Windows, macOS, or Linux. It inherits that user's ambient
authority. Whatever it does to constrain a guest agent is layered *inside* a
process that the operating system already trusts with the user's files.

The risk is not that hosted mode is weak. It is that hosted mode will be
described using bare-metal vocabulary, a customer will make a deployment
decision on that description, and the gap will only surface during an
incident.

## Decision

Hosted RVM uses **operating system isolation plus WASM**. It must **not claim
bare-metal isolation when executing as a normal desktop process.**

This is a constraint on both the implementation and the language used to
describe it. Documentation, marketing, the CLI, the dashboard, and support
conversations must distinguish hosted isolation from bare-metal partition
isolation. "Hardened isolation" is a term reserved by the criteria in section
4 below.

### 1. What hosted RVM may claim

Hosted mode enforces the ADR-286 capability boundary and the ADR-284 execution
contract: undeclared capabilities are inaccessible, every external operation
passes the `rvm-security` three-stage gate, capability requests and denials
are witnessed, and WASM components cannot read one another's memory. Quotas on
memory, instructions, wall time, storage, and invocations are enforced.

Hosted mode may accurately claim: default-deny capability enforcement, WASM
memory isolation between components, witnessed capability decisions,
per-platform OS confinement of the host process, and quota enforcement.

### 2. What hosted RVM may not claim

It may not claim partition memory isolation, device leases, measured boot,
witnessed security gates in the bare-metal sense, or resistance to an attacker
who has already achieved code execution as the same OS user. It may not claim
protection of the guest agent from the host user — a desktop process runs with
the user's authority and cannot hide its contents from that user.

### 3. Per-OS isolation stacks

Hosted RVM composes WASM isolation with the strongest confinement each
platform offers to a user-space process:

**Windows** — WASM isolation, Job Objects, restricted tokens, filesystem
restrictions, and outbound network controls.

**macOS** — WASM isolation, application sandboxing, hardened runtime, scoped
entitlements, and notarization.

**Linux** — WASM isolation, namespaces, cgroups, seccomp, restricted mounts,
and network namespaces.

**Bare-metal RVM** (for contrast, not for hosted mode) — partition memory
isolation, capability tables, device leases, measured boot, and witnessed
security gates.

Each host adapter lives in `rvm-host` and declares which of these mechanisms
it actually engaged. An adapter that could not apply a mechanism reports that
rather than silently running with a weaker stack, so the effective boundary is
observable instead of assumed.

### 4. Independent escape testing gates the "hardened isolation" claim

**Independent escape testing is required before describing RVM execution as
hardened isolation.** Until that testing has been performed and its results
recorded, hosted mode is described in terms of the concrete mechanisms it
engages (section 3) and nothing stronger. This gate applies per platform: an
escape test on Linux does not license the claim on Windows or macOS.

### 5. Native extensions never load into the RVF Reader process

**Native extensions must never load directly into the RVF Reader process.**
They require a separate sandbox or RVM partition.

A native extension loaded in-process has the reader's full authority and can
bypass every mechanism in section 3 — the WASM boundary, the capability gate,
and the witness record all become advisory the moment arbitrary native code
shares the address space. This is why acceleration segments (ADR-284) are
architecture-specific *data* resolved by the runtime, not a mechanism for
injecting host code into the reader.

The same reasoning drives the runtime preference order in ADR-283: Native RVM
first, then OS isolation plus WASM, then WASM, then Linux microVM. The order
is strongest-boundary-first, and it may be changed only by signed policy.

## Acceptance criteria

1. One signed RVF cannot access undeclared files, networks, memory, devices,
   or agents on hosted Linux, Windows, or macOS.
2. Every capability request and denial appears in the witness chain, and the
   resulting chain verifies cryptographically.
3. A host adapter reports which isolation mechanisms it engaged; an adapter
   that could not apply a declared mechanism fails visibly rather than running
   silently degraded.
4. No build of the RVF Reader loads a native extension into its own process;
   extensions resolve to a separate sandbox or RVM partition.
5. Product and technical documentation for hosted mode contains no bare-metal
   isolation claim, and the term "hardened isolation" appears only for
   platforms with recorded independent escape-test results.
6. A modified runtime, policy, model, or state segment is rejected rather than
   executed under a weakened boundary.

## Consequences

### Positive

- Customers can make deployment decisions against an accurate boundary
  instead of discovering the difference during an incident.
- Reserving "hardened isolation" behind independent testing keeps the phrase
  meaningful when it is eventually earned.
- Keeping native extensions out of the reader process preserves the value of
  every other control; without that rule the capability gate is decorative.
- Per-adapter mechanism reporting makes the effective boundary observable in
  the field, not just in design documents.

### Negative

- Hosted mode is a weaker boundary than bare metal, and saying so plainly is a
  competitive cost against products that are less precise.
- Out-of-process native extensions are slower and more complex than in-process
  loading, and some acceleration paths will be harder to deliver.
- Independent escape testing is external work with real cost and schedule
  impact before a claim can be made.
- Three per-OS isolation stacks mean three sets of platform-specific bugs.

## Alternatives Considered

- **Describe hosted and bare-metal isolation uniformly for simplicity**:
  rejected. Uniform language across genuinely different boundaries is a
  security misrepresentation, not a simplification.
- **Allow in-process native extensions behind a signature check**: rejected. A
  signature establishes provenance, not confinement; signed native code in the
  reader process still has the reader's full authority.
- **Rely on WASM alone and skip per-OS confinement**: rejected because the host
  process itself needs constraining, not just the guest module.
- **Self-assess escape resistance internally**: rejected; the whole value of
  the "hardened isolation" claim is that someone outside the team tried to
  break it.
- **Ship the claim now and validate later**: rejected. Once a claim is public,
  customers deploy against it, and later validation cannot un-make those
  decisions.

## Implementation Surfaces

```text
rvm-host     per-OS adapters (Windows / macOS / Linux / Browser / QEMU / bare metal)
rvm-cap      capability tables backing the default-deny boundary (ADR-286)
rvm-security capability check → proof verification → witness recording → operation
rvm-witness  capability requests, denials, and verification results
rvf reader   Tauri desktop reader — no in-process native extensions
```

Independent escape testing is the final item in the RVM implementation
sequence and gates the corresponding claim per platform.
