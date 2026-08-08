# ADR-286: RVF Capability Schema Mapping into `rvm-cap`

- **Status**: Accepted
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — schema side implemented: default-deny CapabilityManifest in CLI pack + registry objects + Reader capability card (vague-scope rejection, manual-review triggers). rvm-cap table mapping is cross-repo in ruvnet/rvm.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-280, ADR-283, ADR-284, ADR-285
- **Tags**: rvf, rvm, capabilities, security, default-deny, witness, policy

## Context

An RVF carries a capability policy covering network, filesystem, devices,
models, tools, and memory. RVM independently defines capability rights in
`rvm-cap` and routes privileged work through the `rvm-security` gate. Until
those two vocabularies are formally connected, "the RVF declares what the
agent may do" and "RVM enforces what the agent may do" are two separate
claims, and the space between them is where undeclared access lives.

The invariant that has to hold — across the desktop reader, hosted RVM, RVM
WASM, browser WASM, and bare-metal RVM alike (ADR-284) — is that **undeclared
capabilities remain inaccessible**. That is only achievable if the mapping
from RVF policy to `rvm-cap` rights is total, default-deny, and the only path
to privileged operations.

## Decision

**RVF policies map directly into `rvm-cap` rights.** The mapping is
default-deny and covers fifteen capability classes:

```text
Memory · Filesystem · Network · Model · MCP · Process · Clock · Randomness
GPU · Sensor · Display · Audio · Clipboard · Persistent state
Inter agent messaging
```

### 1. Default deny

Nothing is granted unless the RVF policy declares it. The WASM runtime
defaults to no filesystem, network, environment, clock, randomness, GPU, or
device access, and the capability mapping is what selectively re-opens
individual rights against that closed baseline. An absent declaration is a
denial, not an unspecified case to be resolved by host defaults.

The policy is covered by `capabilityPolicyHash` in the package's RVM
integration contract, so the granted set is part of the artifact's verifiable
identity rather than a runtime configuration a host can quietly widen.

### 2. The `rvm-security` gate is the only privileged path

Every external operation passes through the existing `rvm-security` sequence:

```text
Capability check → Proof verification → Witness recording → Operation
```

RVM already defines this three-stage gate, and it **remains the only
privileged execution path**. No backend, host adapter, acceleration segment,
or convenience API may reach an external resource by another route. The
ordering matters: the operation is last, and the witness record is written
before it, so a granted operation is auditable even if it subsequently fails.

Capability requests *and denials* are both recorded in `rvm-witness`. A denial
is evidence — it is how "undeclared access was attempted and refused" becomes
provable rather than merely asserted.

Because this gate is singular, the ADR-285 rule that native extensions never
load into the RVF Reader process is load-bearing here too: in-process native
code could reach resources without passing the gate at all, which would make
every capability decision above advisory.

### 3. Refusal rather than degradation

**RVM refuses execution when the RVF requires unsupported capabilities.** If a
backend cannot provide a declared capability class, the correct outcome is a
witnessed refusal, not a silent start with a reduced capability set. An agent
that quietly runs without a capability it declared as required is
indistinguishable, from the outside, from an agent that is working.

Forge applies the same rule ahead of time: it rejects combinations absent from
the published RVM compatibility matrix rather than producing a package whose
capability requirements the target runtime cannot honor.

### 4. Class-level notes

- **Memory** — quota-enforced; one agent component may never read another
  component's memory.
- **Filesystem** — declared paths only; undeclared filesystem access must be
  denied, and that denial is part of the release acceptance test.
- **Network** — declared destinations only; undeclared network access must be
  denied. Host adapters supply outbound network controls per ADR-285.
- **Model** — governs which model segments the agent may load, bounded by the
  signed maximum model size policy.
- **MCP** — governs access to MCP tool surfaces as a first-class capability
  rather than an implicit consequence of network access.
- **Process** — process creation and control.
- **Clock** and **Randomness** — deniable by default, with deterministic
  virtual clock and seeded randomness modes available; these underpin the
  identical-evaluation-hash requirement in ADR-284.
- **GPU** — declared explicitly; GPU capability brokering is deferred scope in
  ADR-283, so the class exists and defaults closed.
- **Sensor**, **Display**, **Audio**, **Clipboard** — device and user-surface
  access, each independently declared.
- **Persistent state** — governs encrypted RVF delta segments; state is bound
  to the base RVF identity and state from an unrelated RVF lineage is
  rejected.
- **Inter agent messaging** — communication with other agents is a declared
  capability, not an ambient property of running in the same runtime.

Size ceilings for model, runtime, memory, and state are enforced through
signed policy, so a capability grant cannot be used to exceed the artifact's
declared resource envelope.

## Acceptance criteria

1. Undeclared filesystem and network access is denied, on a clean operating
   system, in a package produced by the full Forge pipeline.
2. One signed RVF cannot access undeclared files, networks, memory, devices,
   or agents on any conforming backend.
3. Every capability request and denial appears in `rvm-witness`, and the
   resulting chain is cryptographically verifiable.
4. An RVF requiring a capability class the backend does not support is refused
   execution rather than started with a reduced set.
5. A modified policy segment is rejected rather than executed.
6. State from an unrelated RVF lineage is rejected under the persistent-state
   capability.
7. The granted capability set is reproducible from `capabilityPolicyHash` and
   matches what the runtime actually enforced.

## Consequences

### Positive

- "Undeclared capabilities remain inaccessible" becomes enforceable at a
  single, auditable choke point instead of being distributed across backends.
- Witnessed denials turn attempted overreach into evidence, which is what
  makes the acceptance tests meaningful rather than best-effort.
- Binding the policy hash to the artifact identity means a capability grant
  cannot be widened without changing the artifact.
- Splitting MCP, model, and inter-agent messaging into their own classes
  prevents the common failure where tool access rides in on a network grant.

### Negative

- Fifteen classes is a large surface to enforce consistently across five
  backends, and each backend must implement all of them or refuse.
- Default-deny means publishers must declare capabilities explicitly, which is
  more work than an implicit-grant model and will produce early friction.
- Refusing rather than degrading makes capability mismatches hard failures at
  install or launch time, which is a worse user experience than a partial run
  and is nonetheless the correct trade.
- Routing every external operation through one gate puts that gate on the hot
  path for all privileged work.

## Alternatives Considered

- **Coarse capability groups instead of fifteen classes**: rejected because
  coarse groups over-grant; a "devices" bucket that bundles sensor, display,
  audio, and clipboard hands an agent far more than it declared.
- **Allow hosts to supply defaults for undeclared classes**: rejected; that is
  precisely how undeclared access appears, and it makes behavior depend on the
  host rather than the signed artifact.
- **Degrade gracefully when a capability is unsupported**: rejected. A silently
  weakened agent looks like a working agent.
- **Add fast paths that bypass the gate for performance**: rejected. A second
  privileged path is a second thing to audit, and it will drift.
- **Treat MCP access as implied by network capability**: rejected; MCP tool
  surfaces are a distinct authority and are declared separately.

## Implementation Surfaces

```text
rvm-cap       capability tables and rights
rvm-security  capability check → proof verification → witness recording → operation
rvm-rvf       maps RVF policy declarations into rvm-cap rights
rvm-policy    signed size and capability policy enforcement
rvm-witness   capability requests, denials, and verification records
rvm-state     persistent-state capability, lineage binding, encrypted deltas
@ruvector/forge  capabilityPolicyHash emission and compatibility-matrix rejection
```

Capability policy mapping is the fourth item in the RVM implementation
sequence, landing after `rvm-rvf` and the WASM streaming-validation change and
before immutable state deltas.
