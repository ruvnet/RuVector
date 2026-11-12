# ADR-293: RVM Installer and Appliance Formats

- **Status**: Proposed
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — output formats declared planned in compatibility-matrix.json rvmOutputs; generation blocked on rvm-boot integration — cross-repo.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-285, ADR-286, ADR-287, ADR-288, ADR-289, ADR-290, ADR-291, ADR-292
- **Tags**: forge, rvm, appliance, boot, qemu, packaging, outputs, measured-boot

## Context

Desktop installers cover Windows, macOS, and Linux, but they do not cover the
backends where RVM is strongest. The requirement is that one signed RVF runs
through RVM bare metal, RVM hosted mode, RVM WASM mode, browser WASM, and the
desktop RVF Reader with identical capability, witness, state, and lifecycle
semantics. Two of those five backends have no packaging format among the desktop
outputs.

RVM already provides partitions, capabilities, witnesses, proof gates,
scheduling, memory management, measured boot, and WASM agent lifecycle. What is
missing is not the runtime; it is the set of artifacts that carry a signed RVF
into an appliance or a virtual machine, and the decision about which of those
artifacts are in scope now.

There is also a Linux-specific need. The RVF Reader's runtime ladder includes a
Linux microVM tier for runtime compatibility, and QEMU or KVM is the mechanism
for it. That tier needs a bootable image produced by the same build that
produces the installers, from the same canonical RVF, or the compatibility claim
does not survive contact with a real Linux host.

This ADR defines the RVM-specific outputs RVForge generates and what stays out
of scope. The runtime contract these images must carry is ADR-291; the acceleration
segments a partition may load are ADR-292; the trust boundary of the build that
emits them is ADR-290.

## Decision

In addition to desktop installers, Forge generates RVM-specific outputs. Bare
metal outputs use the existing deterministic seven-phase measured boot sequence
provided by `rvm-boot`. QEMU/KVM images serve the Linux runtime-compatibility
tier. Bare-metal install experience and hardware TEE attestation stay out of
scope.

### 1. RVM-specific outputs

```text
Agent.rvf
Agent.rvm.img
Agent.rvm.efi
Agent.qemu.img
Agent.appliance.bundle
```

| Output | Purpose |
|---|---|
| `Agent.rvf` | The original signed canonical RVF, unchanged, for hosts that already have an RVF Reader or an RVM appliance registered for `.rvf` |
| `Agent.rvm.img` | Bare-metal RVM disk image carrying the RVF and the measured boot sequence |
| `Agent.rvm.efi` | EFI-bootable RVM payload for UEFI firmware |
| `Agent.qemu.img` | QEMU/KVM image for the Linux microVM runtime tier |
| `Agent.appliance.bundle` | Appliance distribution bundle: image, runtime contract, provenance, software inventory, checksums, and witness receipt |

Every one of these is subject to the same invariants as the desktop installers.
The embedded RVF hash is identical across every platform package, including the
RVM outputs. The RVF identity, contents, policies, and signatures remain
unchanged. Each output carries its RVF identity, software inventory, build
manifest, source hash, builder identity, and witness receipt, and each embeds
the ADR-291 runtime contract naming `rvmVersion`, `rvmCommit`, `runtimeProfile`,
`capabilityPolicyHash`, `stateSchemaVersion`, and `witnessSchemaVersion`.

`Agent.rvf` is a pass-through, not a re-emission. Forge does not repack,
recompress, or re-sign it; the bytes delivered are the bytes submitted, which is
what makes the identical-hash invariant checkable by inspection.

### 2. Bare metal uses the existing measured boot

`Agent.rvm.img` and `Agent.rvm.efi` use the existing deterministic seven-phase
measured boot sequence provided by `rvm-boot`. Forge does not define a new boot
path, does not add phases, and does not alter phase ordering. It produces images
that the existing sequence consumes.

What Forge contributes is placement and binding: the signed RVF, the capability
policy, and the runtime contract are laid into the image where the measured boot
sequence expects to measure them, so that boot measurements cover the artifact
identity rather than only the runtime.

The boot sequence's determinism is what makes two images built from the same
inputs produce the same measurements. Forge preserves that by pinning dependency
versions and hashes and by producing images from a build whose provenance is
recorded, per ADR-290.

Bare-metal RVM isolation properties are the ones RVM already provides:
partition memory isolation, capability tables, device leases, measured boot, and
witnessed security gates. Forge inherits them; it does not re-implement or
weaken them.

### 3. QEMU/KVM as the Linux runtime-compatibility path

`Agent.qemu.img` is the artifact behind the Linux microVM tier of the runtime
ladder:

```text
Native RVM
Operating system isolation plus WASM
WASM
Linux microVM
Unsupported
```

QEMU or KVM provides Linux runtime compatibility where the host's native
isolation is unavailable or insufficient. The image is produced by the same
build, from the same canonical RVF, as the `.deb`, `.rpm`, and `.AppImage`
outputs, so the microVM tier is not a separately maintained artifact that can
drift from the installers beside it.

`rvm-host` provides the QEMU host adapter alongside the Windows, macOS, Linux,
browser, and bare-metal adapters, giving the microVM tier the same capability,
witness, state, and lifecycle semantics as every other backend.

### 4. Lifecycle and state parity

RVM outputs expose the same lifecycle surface as every other backend, through
`rvm-launch`:

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

State follows the same rules on an appliance as on a desktop: the base RVF is
immutable, changes live in encrypted RVF delta segments bound to the base RVF
identity, `CompressedCheckpoint` carries execution snapshots, `WitnessDelta`
carries reconstruction, and state from an unrelated RVF lineage is rejected.
This is what allows an instance to suspend on one backend and resume on another.

### 5. Deferred scope

The following stay explicitly out of scope for this release:

- **Bare-metal RVM installation** — Forge produces `Agent.rvm.img` and
  `Agent.rvm.efi`, but the experience of writing them to hardware, partitioning,
  firmware configuration, and recovery is not a Forge deliverable. The images
  are artifacts; installing them is an operator activity.
- **Hardware TEE attestation** — measured boot produces measurements; binding
  those measurements to a hardware root of trust and remotely attesting them is
  deferred.
- Microsoft Store and Apple App Store distribution.
- Android and iOS packages.
- GPU capability brokering.
- Public RVF marketplace.
- Automatic model quantization.
- Delta installer generation.
- Federated enterprise builders.

Deferring bare-metal install UX and TEE attestation does not weaken the outputs
that are in scope. It means the release does not claim an installation product
or a hardware-rooted attestation chain it has not built. Consistent with the
security requirements, RVM execution is not described as hardened isolation
until independent escape testing has been completed.

## Acceptance criteria

1. One signed RVF produces `Agent.rvf`, `Agent.rvm.img`, `Agent.rvm.efi`,
   `Agent.qemu.img`, and `Agent.appliance.bundle` from a single build.
2. The embedded RVF SHA256 in every RVM output equals the SHA256 in every
   desktop installer from the same build, and equals the submitted RVF's hash.
3. `Agent.rvf` is byte-identical to the submitted RVF.
4. `Agent.rvm.img` and `Agent.rvm.efi` boot through the existing `rvm-boot`
   seven-phase measured boot sequence with no added or reordered phases.
5. Two builds from identical inputs produce identical boot measurements.
6. Boot measurements cover the RVF identity and capability policy, not only the
   runtime.
7. One signed RVF runs unchanged on hosted Linux, Windows, macOS, QEMU, and
   bare-metal RVM, producing identical deterministic evaluation hashes.
8. An instance suspended on one backend resumes on another and reconstructs
   state from checkpoint plus witness deltas.
9. An RVM output preserves its base RVF identity across suspend, resume,
   checkpoint, and terminate.
10. Undeclared files, networks, memory, devices, and agents remain inaccessible
    from an appliance instance.
11. Modified runtime, policy, model, and state segments are rejected by an RVM
    output as they are by a desktop installer.
12. Every RVM output produces a complete cryptographically verifiable witness
    chain, and its build provenance and witness records verify independently.
13. `Agent.appliance.bundle` contains the image, runtime contract, provenance
    record, software inventory, SHA256 checksums, and witness receipt, and the
    bundle verifies as a unit.
14. Release documentation describes RVM execution as hardened isolation only
    after independent escape testing.

## Consequences

### Positive

- The RVM backends gain packaging parity with the desktop backends, so the
  one-RVF-everywhere claim is testable end to end.
- Reusing `rvm-boot` unchanged means measured-boot determinism is inherited
  rather than re-argued.
- Producing the QEMU image in the same build as the Linux installers removes the
  drift risk between the microVM tier and the packages beside it.
- Explicitly deferring install UX and TEE attestation keeps the release from
  implying guarantees it has not built.

### Negative

- Five additional outputs per build increase build time, artifact storage, and
  the verification matrix.
- Appliance images are large, and the embedded packaging mode duplicates model
  payloads across them.
- Without bare-metal install tooling, operators must handle imaging themselves,
  which limits who can consume the bare-metal outputs.
- Measured boot without TEE attestation gives local integrity measurement but no
  remote proof, which some deployments will need before adopting appliances.
- The QEMU tier adds a virtualization dependency to Linux hosts that fall
  through to it.

## Alternatives Considered

- **Ship only `Agent.rvf` and let operators build their own images**: rejected
  because it pushes the identical-hash and measured-boot binding onto every
  operator, where it cannot be verified centrally.
- **Define a Forge-specific boot sequence for appliances**: rejected because
  `rvm-boot` already provides a deterministic seven-phase measured boot, and a
  second sequence would fork the measurement semantics.
- **Use containers instead of QEMU for the Linux compatibility tier**: rejected
  because the tier exists precisely for hosts whose native isolation is
  unavailable or insufficient, which is where container isolation is also in
  question.
- **Build the QEMU image in a separate pipeline**: rejected because a separate
  build cannot guarantee the same embedded RVF hash as the installers it is
  meant to match.
- **Include bare-metal install UX in this release**: rejected as deferred scope;
  it is an installation product, not a packaging format.
- **Claim hardware-rooted attestation from measured boot alone**: rejected
  because measurement without a hardware root of trust and a remote attestation
  path is not attestation.

## Implementation Surfaces

- `rvf forge core` — RVM image assembly and appliance bundling
- `rvm-boot` — existing deterministic seven-phase measured boot, consumed unchanged
- `rvm-host` — QEMU and bare-metal RVM adapters
- `rvm-launch` — lifecycle commands for appliance instances
- `rvm-state` — encrypted delta segments, `CompressedCheckpoint`, `WitnessDelta`
- `@ruvector/forge` — RVM targets in `build`, `submit`, `download`, `verify`
- appliance bundle manifest schema and verification report
