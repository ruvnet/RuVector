# RVForge — Canonical Requirements

> Source of truth for the RVForge + RVM integration build-out (branch
> `feat/rvf-forge`). Captured 2026-08-03 from the product directive. ADRs
> ADR-283 through ADR-293 derive from this document. If an ADR and this
> document disagree, reconcile the ADR and note the change here.

## Product Summary

Build `RVForge`: an npm CLI plus an optional hosted build service that
converts one canonical RVF into signed platform installers.

```bash
npx @ruvector/forge build agent.rvf windows macos linux
```

Outputs:

```text
AgentSetup.exe
Agent.msi
Agent.dmg
Agent.deb
Agent.rpm
Agent.AppImage
Agent.rvf
```

### Architecture

1. **Canonical RVF** — The original RVF remains identical across platforms.
   It contains the agent, model, memory, interface, policies, and signatures.

2. **Rust RVF Reader** — A small Rust application verifies and executes the
   RVF using:
   - WASM for universal execution
   - Native libraries for acceleration
   - QEMU or KVM for Linux runtime compatibility
   - RVM for native coherence domains

3. **Tauri packaging layer** — Tauri already generates NSIS executables, MSI,
   DMG, macOS applications, Debian packages, RPM, and AppImage packages. It
   can also embed the RVF as an application resource.
   Ref: https://v2.tauri.app/reference/config/

4. **Build service** — The npm CLI either builds locally or submits an
   encrypted build specification to dedicated Linux, Windows, and macOS
   workers. Cross compilation alone is insufficient because DMG and Debian
   bundles require their native build environments.
   Ref: https://tauri.app/v1/guides/building/linux

```text
npm CLI
   ↓
Signed build manifest
   ↓
RVForge scheduler
   ↓
Linux worker / Windows worker / macOS worker
   ↓
Signing and notarization
   ↓
Installers plus provenance
```

GitHub Actions already provides Linux, Windows, and macOS workers, making it
suitable for the first implementation.
Ref: https://docs.github.com/actions/using-github-hosted-runners/about-github-hosted-runners

### Packaging modes

1. **Complete capsule** — The RVF and runtime are embedded inside every
   installer. Operates completely offline but duplicates large models across
   outputs.
2. **Thin installer** — The installer contains the RVF Reader and downloads
   the signed RVF during installation. May reduce a 5 GB installer to roughly
   10–30 MB.
3. **Shared reader** — Install the RVF Reader once and register `.rvf` as a
   file type. Subsequent RVFs require no platform packaging.
4. **Enterprise private build** — Build entirely inside the customer
   environment so models, prompts, data, and signing credentials never reach
   Cognitum infrastructure.

### Signing

macOS requires Apple code signing and notarization for normal external
distribution. Windows signing avoids SmartScreen warnings.
Refs: https://v2.tauri.app/distribute/sign/macos/ ,
https://v2.tauri.app/distribute/sign/windows/

The service should support:

1. Customer supplied signing identities.
2. Signing operations through KMS or HSM without exporting private keys.
3. Cognitum signing for verified marketplace packages.
4. Unsigned development builds.

Every output should include the RVF identity, software inventory, build
manifest, source hash, builder identity, and witness receipt.

### Suggested implementation layout

```text
@ruvector/forge     npm CLI and API client
rvf forge core      Rust packaging and verification library
rvf reader          Tauri desktop reader
forge service       Build scheduler, artifact registry, signing, billing
forge workers       Linux, Windows, macOS isolated builders
```

### Delivery estimate

1. Local packaging prototype: two to three weeks.
2. Hosted unsigned build matrix: another three to four weeks.
3. Signing, notarization, provenance, private builds, and billing: another
   four to six weeks.

A production version is roughly eight to twelve weeks for three engineers.
Raw compute should remain below approximately $1 for most builds. Hosted
runner pricing is about $0.006/Linux-min, $0.010/Windows-min,
$0.062/macOS-min.
Ref: https://docs.github.com/en/billing/reference/actions-runner-pricing

The largest risk is exposing proprietary RVFs and signing credentials to the
build service. The fix is ephemeral workers, encrypted uploads, network
isolation during packaging, HSM based signing, automatic deletion, and a
self hosted enterprise builder.

**Acceptance test:** submit one signed RVF and produce installable Windows,
macOS, and Linux packages. Each must execute offline, generate identical
evaluation hashes, preserve the original RVF identity, deny undeclared
capabilities, and emit verifiable build and runtime witness records.

---

# RVForge Requirements

## 1. Objective

RVForge converts one canonical `.rvf` artifact into installable packages
for Windows, macOS, Linux, and RVM.

The RVF identity, contents, policies, and signatures must remain unchanged
across every generated package.

## 2. Inputs

1. Valid `.rvf` file.
2. Application name, version, publisher, description, icon, and identifier.
3. Target operating systems and architectures.
4. Packaging mode: embedded, thin, or shared reader.
5. Runtime preference: WASM, native, Linux microVM, or RVM.
6. Capability policy covering network, filesystem, devices, models, tools,
   and memory.
7. Optional customer signing references.
8. Optional update channel and distribution URL.

## 3. Outputs

1. Windows NSIS `.exe`.
2. Windows `.msi`.
3. macOS `.app` and `.dmg`.
4. Linux `.deb`.
5. Linux `.rpm`.
6. Linux `.AppImage`.
7. Original signed `.rvf`.
8. SHA256 checksums.
9. Software inventory.
10. Build provenance record.
11. RVF witness receipt.
12. Verification report.

## 4. Core Invariants

1. The embedded RVF hash must be identical across every platform package.
2. Build workers must never execute the submitted RVF.
3. The RVF Reader must verify signatures before loading executable segments.
4. Undeclared capabilities must remain inaccessible.
5. Build output must be traceable to the input RVF, runtime version, source
   revision, builder, and signing identity.
6. Mutable state must never modify the signed base identity.
7. Build failure must produce no partially trusted artifact.

## 5. npm CLI Requirements

Package name: `@ruvector/forge`

Required commands:

```bash
npx @ruvector/forge init
npx @ruvector/forge validate agent.rvf
npx @ruvector/forge build agent.rvf
npx @ruvector/forge submit agent.rvf
npx @ruvector/forge status BUILD_ID
npx @ruvector/forge download BUILD_ID
npx @ruvector/forge verify AgentSetup.exe
```

The CLI must:

1. Operate on Node.js 20 or later.
2. Support interactive and unattended execution.
3. Validate the RVF locally before upload.
4. Generate a canonical build manifest.
5. Support local and hosted builds.
6. Display estimated build time, output size, and cost before submission.
7. Resume interrupted uploads.
8. Verify downloaded artifacts automatically.
9. Return stable machine readable error codes.
10. Never place signing secrets in command history or logs.

## 6. Packaging Requirements

### FR001 Embedded mode
The installer contains the complete RVF and RVF Reader. It must execute
without internet access.

### FR002 Thin mode
The installer contains the reader and signed RVF locator. The reader
downloads and verifies the RVF before execution.

### FR003 Shared reader mode
The installer registers `.rvf` files with the operating system. Any
compatible RVF can then be opened directly.

### FR004 Runtime selection
The reader selects the strongest compatible runtime in this order:

```text
Native RVM
Operating system isolation plus WASM
WASM
Linux microVM
Unsupported
```

The exact order may be changed by signed policy.

### FR005 State management
Runtime state must be stored as encrypted RVF delta segments separate from
the immutable base artifact.

### FR006 Updates
Updates must be signed, version constrained, reversible, and linked to the
prior RVF identity.

### FR007 Branding
Packages must support custom icons, publisher identity, application name,
interface theme, license, and installation text.

## 7. Hosted Service Requirements

Required API operations:

```text
POST /v1/builds
GET /v1/builds/{id}
GET /v1/builds/{id}/artifacts
POST /v1/builds/{id}/cancel
POST /v1/builds/{id}/verify
DELETE /v1/builds/{id}
```

The service must:

1. Run Linux, Windows, and macOS builds on native workers.
2. Isolate every build in a fresh worker.
3. Build targets concurrently.
4. Cache public dependencies by verified hash.
5. Never share private RVF data between tenants.
6. Stream structured build status and logs.
7. Support resumable multipart uploads.
8. Support customer controlled storage.
9. Support private enterprise deployment.
10. Retain artifacts according to tenant policy.

## 8. Security Requirements

1. Default deny capability policy.
2. Ed25519 or stronger RVF signature verification.
3. Per job worker isolation.
4. Encrypted upload, storage, and artifact delivery.
5. Signing keys held in an HSM, KMS, or customer controlled signing service.
6. No private signing key export.
7. Network disabled during final packaging unless notarization requires
   access.
8. Dependency versions and hashes pinned.
9. Logs stripped of tokens, paths, prompts, model data, and environment
   secrets.
10. Build workers destroyed within five minutes of job completion.
11. Uploaded private RVFs deleted within sixty minutes unless retention is
    explicitly enabled.
12. Every produced package scanned without executing its RVF payload.
13. Revoked publisher identities rejected at installation or execution.
14. All privileged service actions written to an immutable audit log.
15. Independent escape testing required before describing RVM execution as
    hardened isolation.

## 9. Signing Requirements

### Windows
1. Support organization validated and extended validation certificates.
2. Support Azure Key Vault signing.
3. Apply trusted timestamps.
4. Generate signed NSIS and MSI installers.

### macOS
1. Support Apple Developer ID signing.
2. Support Apple notarization.
3. Generate Intel, ARM64, and universal packages.
4. Staple notarization results to the application and DMG.

### Linux
1. Support GPG signed Debian and RPM packages.
2. Generate repository metadata.
3. Verify package and embedded RVF independently.

## 10. Performance Requirements

1. Local validation under two seconds for RVFs below 1 GB, excluding full
   payload hashing.
2. Cached build under three minutes per platform.
3. Uncached build under ten minutes, excluding Apple notarization.
4. Package overhead below 30 MB for the shared reader.
5. Embedded package overhead below 50 MB plus the RVF size.
6. Reader startup under 500 milliseconds before model loading.
7. Hosted service availability of at least 99.9 percent.
8. Build success rate above 99 percent for valid specifications.

## 11. Compatibility Requirements

1. Windows 10 and 11 on x64.
2. Windows 11 on ARM64.
3. macOS 13 or later on Intel and Apple Silicon.
4. Ubuntu 22.04 and 24.04.
5. Debian 12 or later.
6. Generic Linux through AppImage.
7. Browser execution through WebAssembly.
8. Native `.rvf` output for RVM appliances.

## 12. MVP Scope

The first release must include:

1. npm CLI.
2. Local validation.
3. Hosted Linux, Windows, and macOS builds.
4. Embedded and thin packaging.
5. Tauri based RVF Reader.
6. WASM execution.
7. Windows and macOS signing.
8. Linux package signing.
9. Immutable base plus encrypted state delta.
10. Build provenance, software inventory, and witness receipts.
11. Web dashboard for submission, status, and downloads.
12. Tenant isolation and automatic artifact deletion.

## 13. Deferred Scope

1. Microsoft Store distribution.
2. Apple App Store distribution.
3. Android and iOS packages.
4. Hardware TEE attestation.
5. Bare metal RVM installation.
6. GPU capability brokering.
7. Public RVF marketplace.
8. Automatic model quantization.
9. Delta installer generation.
10. Federated enterprise builders.

## 14. Primary Failure Mode

The greatest risk is compromising proprietary RVFs or publisher signing
identities during hosted builds.

The required mitigation is an enterprise private builder plus HSM based
signing. The hosted service must treat every submitted RVF as confidential
executable intellectual property.

## 15. Release Acceptance Test

Submit one signed RVF and generate `.exe`, `.msi`, `.dmg`, `.deb`, and
`.AppImage` outputs.

Release passes only when:

1. Every installer completes successfully on a clean operating system.
2. Every embedded RVF has the same SHA256 hash.
3. Offline execution succeeds in embedded mode.
4. Tampering prevents execution.
5. Undeclared filesystem and network access is denied.
6. State survives restart without modifying the base RVF.
7. Uninstallation removes the runtime and state according to policy.
8. Build provenance and witness records verify independently.

---

# RVM Integration Requirements

Applies specifically to [`ruvnet/rvm`](https://github.com/ruvnet/rvm).

## 1. Objective

RVM must become an executable backend for RVForge packages.

The same signed RVF must run through:

```text
RVM bare metal
RVM hosted mode
RVM WASM mode
Browser WASM
Desktop RVF Reader
```

RVM must provide identical capability, witness, state, and lifecycle
semantics across every backend.

## 2. Current Gap

RVM already provides partitions, capabilities, witnesses, proof gates,
scheduling, memory management, measured boot, and WASM agent lifecycle.
However, it does not yet expose a complete desktop host runtime that an
`.exe`, `.dmg`, or `.deb` package can invoke.

The current `rvm-wasm` module also limits modules to 1 MB. That is too small
for practical agent runtimes and must be replaced with streaming validation
or policy controlled limits.
Ref: https://github.com/ruvnet/rvm/blob/main/userguide/04-crate-reference.md

## 3. New Crates

```text
rvm-rvf
rvm-host
rvm-launch
rvm-ffi
rvm-node
rvm-policy
rvm-state
```

### RVM001 `rvm-rvf`

Responsible for:

1. Reading RVF manifests.
2. Verifying RVF signatures and hashes.
3. Resolving runtime segments.
4. Loading models, memory, policies, and WASM components.
5. Rejecting incompatible RVF versions.
6. Mapping RVF capabilities into RVM capability tables.
7. Preserving the canonical RVF identity.

### RVM002 `rvm-host`

Provide host adapters for:

```text
Windows / macOS / Linux / Browser / QEMU / RVM bare metal
```

Hosted RVM must use operating system isolation plus WASM. It must not claim
bare metal isolation when executing as a normal desktop process.

### RVM003 `rvm-launch`

Expose lifecycle commands:

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

### RVM004 `rvm-ffi`

Expose a stable C interface for Tauri and other native hosts:

```text
rvm_validate
rvm_inspect
rvm_create
rvm_start
rvm_suspend
rvm_resume
rvm_checkpoint
rvm_export_witness
rvm_terminate
```

### RVM005 `rvm-node`

Provide Node API bindings used by `@ruvector/forge`.

## 4. RVF Loading Requirements

1. Verify the root manifest before allocating executable memory.
2. Verify every referenced segment before loading it.
3. Reject unsigned executable segments by default.
4. Support progressive loading for large models.
5. Support encrypted segments.
6. Support architecture specific acceleration segments.
7. Never execute RVF content during inspection or packaging.
8. Enforce maximum model, runtime, memory, and state sizes through signed
   policy.
9. Produce a witness record for every verification result.
10. Refuse execution when the RVF requires unsupported capabilities.

## 5. WASM Runtime Requirements

1. Remove the fixed 1 MB production limit.
2. Validate modules through streaming input.
3. Support multiple WASM components per RVF.
4. Support the WASM Component Model and WIT interfaces.
5. Default to no filesystem, network, environment, clock, randomness, GPU,
   or device access.
6. Provide deterministic virtual clock and seeded randomness modes.
7. Enforce memory, instruction, wall time, storage, and invocation quotas.
8. Support suspend, snapshot, migration, resume, and termination.
9. Record capability requests and denials in `rvm-witness`.
10. Prevent one agent component from reading another component's memory.

## 6. Capability Mapping

RVF policies must map directly into `rvm-cap` rights.

Required capability classes:

```text
Memory · Filesystem · Network · Model · MCP · Process · Clock · Randomness
GPU · Sensor · Display · Audio · Clipboard · Persistent state
Inter agent messaging
```

Every external operation must pass through the existing `rvm-security`
sequence:

```text
Capability check → Proof verification → Witness recording → Operation
```

RVM already defines this three stage gate and should remain the only
privileged execution path. Ref: https://github.com/ruvnet/rvm

## 7. State Requirements

1. Treat the base RVF as immutable.
2. Store changes in encrypted RVF delta segments.
3. Use `CompressedCheckpoint` for execution snapshots.
4. Use `WitnessDelta` for reconstruction.
5. Bind every delta to the base RVF identity.
6. Reject state from an unrelated RVF lineage.
7. Support branch, rollback, merge, migrate, and reset.
8. Permit state deletion without deleting the base RVF.
9. Support customer controlled state encryption keys.
10. Produce identical reconstructed state across compatible platforms.

## 8. Host Isolation Requirements

### Windows
WASM isolation, Job Objects, restricted tokens, filesystem restrictions,
outbound network controls.

### macOS
WASM isolation, application sandboxing, hardened runtime, scoped
entitlements, notarization.

### Linux
WASM isolation, namespaces, cgroups, seccomp, restricted mounts, network
namespaces.

### Bare metal RVM
Partition memory isolation, capability tables, device leases, measured boot,
witnessed security gates.

Native extensions must never load directly into the RVF Reader process. They
require a separate sandbox or RVM partition.

## 9. Forge Integration Contract

Every generated package must embed:

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

Forge must reject combinations that are absent from the published RVM
compatibility matrix.

## 10. RVM Specific Outputs

In addition to desktop installers, Forge should generate:

```text
Agent.rvf
Agent.rvm.img
Agent.rvm.efi
Agent.qemu.img
Agent.appliance.bundle
```

The bare metal outputs must use the existing deterministic seven phase
measured boot sequence provided by `rvm-boot`.
Ref: https://github.com/ruvnet/rvm/blob/main/userguide/01-quickstart.md

## 11. Required ADRs

1. RVF execution contract.
2. Hosted RVM security boundary.
3. RVF capability schema mapping.
4. WASM Component Model integration.
5. Immutable base and state delta lifecycle.
6. Desktop host adapters.
7. Forge build and signing trust boundary.
8. Runtime compatibility and version negotiation.
9. Native acceleration isolation.
10. RVM installer and appliance formats.

## 12. Implementation Sequence

1. Add RVF execution contract and compatibility ADR.
2. Implement `rvm-rvf`.
3. Replace the WASM size limit with streaming validation.
4. Implement capability policy mapping.
5. Implement immutable state deltas and checkpoint reconstruction.
6. Implement `rvm-launch` and `rvm-ffi`.
7. Implement Linux hosted mode.
8. Add Windows and macOS host adapters.
9. Add Node bindings.
10. Connect `@ruvector/forge`.
11. Add bare metal and appliance outputs.
12. Complete independent security validation.

## 13. RVM Acceptance Test

A release passes only when one signed RVF:

1. Runs unchanged on hosted Linux, Windows, macOS, QEMU, and bare metal RVM.
2. Produces identical deterministic evaluation hashes.
3. Cannot access undeclared files, networks, memory, devices, or agents.
4. Suspends on one backend and resumes on another.
5. Preserves its base RVF identity.
6. Reconstructs state from checkpoint plus witness deltas.
7. Rejects modified runtime, policy, model, and state segments.
8. Produces a complete cryptographically verifiable witness chain.

---

# Product Vision & Distribution

RVForge turns an RVF agent into a normal installable application that
users can download, double tap, and run securely on almost any system.

## User experience

```text
Upload agent.rvf
Choose Windows, macOS, Linux, or RVM
Select permissions and branding
Generate installers
Distribute application
```

The customer receives:

```text
MyAgent.exe
MyAgent.dmg
MyAgent.deb
MyAgent.AppImage
MyAgent.rvf
```

When opened, the application:

1. Verifies the agent's signature and publisher.
2. Creates a quarantined runtime.
3. Loads the embedded model, memory, interface, and tools.
4. Grants only declared capabilities.
5. Runs locally, potentially without internet access.
6. Stores learned state separately from the signed application.
7. Records actions in a verifiable witness chain.

## What this enables

1. **Agents become products** — A Ruflo, MetaHarness, RuView, or Cognitum
   agent can be distributed like conventional software instead of requiring
   repositories, terminals, cloud accounts, or installation instructions.
2. **Sovereign execution** — Sensitive documents, prompts, models, and
   memory can remain entirely on the customer's laptop, server, appliance,
   or private network.
3. **One artifact across environments** — The same RVF identity can move
   between browser, desktop, cloud, edge hardware, and native RVM without
   rebuilding the agent itself.
4. **Quarantined third party agents** — Organizations can run an agent
   while denying it access to files, networks, devices, credentials, or
   other applications unless explicitly approved.
5. **Offline agents** — An RVF containing its model and knowledge can
   operate in hospitals, factories, vehicles, remote locations, regulated
   environments, and disconnected systems.
6. **Persistent portable intelligence** — The agent can retain memory,
   suspend, resume, migrate, branch, and roll back while preserving its
   cryptographic identity and history.
7. **White label partner distribution** — Xunison, Arista, Netgear, Eero,
   healthcare partners, and other channels could receive branded installers
   or appliance images containing specific Cognitum capabilities.
8. **Agent marketplace** — Developers could publish signed RVFs with
   declared permissions, pricing, licensing, resource requirements,
   evaluation results, and publisher identity.
9. **Enterprise governance** — Security teams receive a concrete artifact
   to approve. They can inspect its model, permissions, software inventory,
   data access, tests, and witness records before deployment.
10. **Commercial licensing** — RVFs could enforce licenses for:

```text
Per device · Per user · Per execution · Subscription
Offline enterprise · Partner appliance · Feature capability
```

## Example products

```text
RuView Home        Local spatial intelligence application
Cognitum Analyst   Offline document and data analysis agent
Ruflo Developer    Quarantined autonomous coding environment
HearMusica         Private audio intelligence application
Partner Appliance  Signed RVF bundle for router or edge hardware
```

## Operational impact

Today, distributing one agent across three operating systems and two
processor architectures can require six builds, multiple signing pipelines,
separate update systems, and weeks of integration work.

RVForge reduces that to one agent artifact and an automated five to ten
minute packaging process. The platform specific wrappers change, but the
agent identity and behavior remain constant.

The largest limitation is that arbitrary Python or Node applications will
not instantly become universally portable. Their dependencies must be
embedded, compiled to WASM, or executed inside a Linux microVM. **The
initial secure product should therefore prioritize Rust and WASM agents**,
with broader compatibility added later.

## End-user acceptance test

A nontechnical user downloads the generated application, installs it
without developer tools, runs it offline, confirms denied host access,
moves its encrypted state to another platform, and resumes with the same
verified identity and behavior.

---

# RVForge Platform — Store, Reader, Publisher, Registry, Enterprise

RVForge is an independent agentic application store, runtime, package
registry, and trust system built around RVF and RVM. It is closer to Steam
plus npm plus an enterprise application catalog than to the Apple App
Store.

```text
Developers publish intelligence
Users install agents
RVM quarantines execution
RVF preserves identity and state
RVForge governs trust, licensing, and updates
```

## P1. Product Definition

RVForge consists of five products:

1. **RVForge Store** — Public marketplace for discovering, purchasing, and
   installing agents.
2. **RVForge Reader** — Desktop application that installs and runs RVFs
   through WASM, operating system isolation, or RVM.
3. **RVForge Publisher** — Web console and npm CLI for building, testing,
   signing, and publishing RVFs.
4. **RVForge Registry** — Content addressed package registry containing
   releases, manifests, signatures, evaluations, and provenance.
5. **RVForge Enterprise** — Private agent stores with organizational
   approval, policy enforcement, deployment, and audit controls.

## P2. Distribution Model

Users install RVForge once through a signed `.exe`, `.dmg`, `.deb`, `.rpm`,
or `.AppImage`. Agents are then distributed directly as signed `.rvf`
files.

```text
Install RVForge → Browse agents → Review capabilities →
Install signed RVF → Run inside quarantine → Store encrypted state locally
```

RVForge does not require a new operating system installer for every agent.
Standalone installers remain available for branded consumer and partner
applications.

This works on Windows, macOS, Linux, browsers with restrictions, and native
RVM systems. It cannot fully replace Apple's iPhone and iPad App Store
because Apple restricts downloaded executable functionality.
Ref: https://developer.apple.com/app-store/review/guidelines/ (2.5.2)

## P3. Core Marketplace Objects

```text
Publisher · Organization · RVF Package · Release · Capability Manifest
Runtime Profile · Model Manifest · Evaluation Report · Security Report
License · Entitlement · Installation · State Capsule · Update
Witness Receipt · Revocation
```

Every release is immutable. A new version creates a new signed release
linked to its predecessor.

## P4. Publisher UX

### Create

```bash
npx @ruvector/rvforge init
```

Publisher provides: agent name; description and category; icon and
screenshots; pricing model; support information; privacy policy; runtime
requirements; publisher identity.

### Package

```bash
npx @ruvector/rvforge pack agent.rvf
```

RVForge validates:

```text
RVF structure · Publisher signature · Executable segments
Model provenance · Capability policy · Runtime compatibility
Memory requirements · External services · Software inventory
License compatibility
```

### Test

```bash
npx @ruvector/rvforge test agent.rvf
```

Tests include: clean installation; deterministic evaluations; capability
denials; network monitoring; filesystem escape attempts; resource
exhaustion; malformed inputs; state checkpoint and recovery; update and
rollback; witness verification.

### Publish

```bash
npx @ruvector/rvforge publish agent.rvf
```

The publisher sees:

```text
Validation passed
Security profile: restricted
Evaluation score: 94 percent
Supported runtimes: WASM and RVM
Supported systems: Windows, macOS, Linux
Requested capabilities: selected files
Network access: none
Ready to publish
```

## P5. Store UX

Home sections: Featured Agents · Verified Publishers · Runs Entirely
Locally · Enterprise Ready · Spatial Intelligence · Developer Tools ·
Healthcare · Audio Intelligence · Recently Updated · Free and Open Source.

Search filters: Category · Price · Publisher · Local or cloud model ·
Runtime · Operating system · Capability level · Open source · Enterprise
approved · Offline support · Evaluation score.

Every listing displays: agent name, publisher identity, version, price,
purpose, screenshots, runtime requirements, model location, data handling,
capabilities, evaluation results, security findings, software inventory,
release history, user reviews.

Primary trust card (conceptual):

```text
Cognitum Analyst

Verified publisher
Runs locally
No internet access
Reads only files you select
Encrypted persistent memory
WASM and RVM isolation
Evaluation score: 94 percent
No critical security findings
```

Primary actions: Install · Try in Temporary Session · Review Capabilities ·
View Source · Purchase · Add to Organization.

## P6. Installation UX

Before installation, RVForge presents the exact capability contract:

```text
This agent requests:
  Selected document access · 512 MB memory · Local model execution
  Encrypted persistent state

This agent cannot:
  Access the internet · Read other folders · Use the microphone
  Run background processes · Contact external model providers
```

Actions: Install · Customize Permissions · Cancel.

Broad permission descriptions such as "access your computer" are
prohibited.

## P7. Library UX

Library states: Installed · Running · Paused · Updates · Quarantined ·
Organization Managed · Archived.

Each agent exposes: Open · Pause · Terminate · Clone · Reset · Export
State · Import State · Review Activity · Change Permissions · Verify ·
Uninstall.

## P8. Runtime UX

While an agent is active, RVForge displays: current task, runtime type,
model activity, CPU usage, memory usage, network connections, filesystem
access, tool calls, recent actions, witness status, estimated execution
cost.

Emergency controls remain visible: Pause · Terminate · Disconnect
Network · Revoke Capabilities · Rollback State.

## P9. Update UX

Every update displays a semantic permission difference:

```text
Version 1.3 changes:
  Adds PDF processing
  Requests access to selected folders
  Introduces optional OpenAI connectivity
  Changes memory schema from 2 to 3
```

Users must approve any capability expansion. Updates containing only code
fixes within the existing contract may follow organizational update
policy.

Rollback must remain available until state migration makes rollback
unsafe. That condition must be declared before installation.

## P10. Trust Levels

1. **Published** — Identity verified and package structurally valid.
2. **Tested** — Automated runtime, security, and evaluation tests passed.
3. **Reviewed** — Human security and capability review completed.
4. **Enterprise Approved** — Approved by the customer's security or
   governance team.

Trust levels must describe evidence, not imply that software is
universally safe.

## P11. Review Pipeline

```text
Upload → Signature verification → Static inspection →
Malware and dependency scanning → Quarantined execution →
Capability testing → Behavioral evaluation → Publisher review →
Publish or reject
```

Manual review is required when an agent requests: unrestricted filesystem
access; arbitrary network access; process creation; native code;
credentials; background execution; financial transactions; health
decisions; physical device control; inter agent delegation.

## P12. Security Model

1. Publisher signs the RVF.
2. RVForge verifies and countersigns the release record.
3. RVM verifies both before execution.
4. Every capability defaults to denied.
5. Every privileged operation passes through `rvm-security`.
6. Runtime actions produce witness records.
7. The registry maintains a public transparency log.
8. Compromised packages can be revoked.
9. Installed packages can be quarantined without deleting user state.
10. Enterprise administrators can override public store availability.

RVForge must never silently revoke or delete locally owned RVFs.
Revocation blocks execution by policy while preserving export and forensic
access.

## P13. Commercial Model

Supported licensing:

```text
Free · Open source · One time purchase · Subscription · Per user
Per device · Per organization · Per execution · Usage metered
Private enterprise license · Partner appliance license
```

A reasonable initial marketplace fee is 10 percent, excluding model and
compute costs.

The customer may choose: embedded local model; publisher supplied
inference; Cognitum Meta LLM; customer supplied model credentials;
enterprise private inference. All external inference costs must be
disclosed before execution.

## P14. Enterprise UX

Administrators can: approve or deny agents; create private catalogs; set
capability ceilings; require local inference; restrict network domains;
control updates; assign licenses; deploy agents; revoke capabilities;
inspect witnesses; export audit evidence; set jurisdiction rules.

Organizational policy always overrides publisher requested permissions.

## P15. MVP

The first RVForge release should include:

1. Windows, macOS, and Linux Reader.
2. Public RVF registry.
3. npm publisher CLI.
4. Publisher identity and signing.
5. Search and agent listings.
6. Free agent installation.
7. Capability cards.
8. WASM quarantine.
9. RVM integration.
10. Updates and rollback.
11. Witness viewer.
12. Revocation.
13. Private organization catalogs.

Paid applications, revenue sharing, mobile support, and advanced
evaluations follow. A functional marketplace MVP requires approximately
twelve to sixteen weeks with five engineers; a production commercial store
with payments, enterprise controls, moderation, and independent security
validation is closer to six months.

## P16. Primary Differentiation

Conventional stores distribute applications. RVForge distributes governed
intelligence:

```text
Code · Model · Memory · Policy · Runtime · Evaluation · Identity
Lineage · Witness
```

The biggest failure mode is becoming another untrusted agent marketplace
filled with unverifiable wrappers and exaggerated claims. The fix is
making capability disclosure, reproducible evaluations, publisher
identity, and witnessed execution mandatory rather than optional badges.

## Platform acceptance test

A publisher uploads one signed RVF, automated review detects its exact
capabilities, a user installs it without developer tools, RVM denies
undeclared access, the agent runs offline, an update requests fresh
permission, and every build, installation, and privileged action verifies
through the public witness record.

---

# RVForge Agent Dock

A horizontal pill that shows persistent agents without opening the full
application — a **security and control surface, not a decorative chat
widget**. It is a continuous trust indicator differentiating RVForge from
conventional agent interfaces that hide autonomous activity in the
background.

## D1. Collapsed state

```text
Agent icon · Agent name · Current task · Progress · Runtime status ·
Pause · Terminate · Expand
```

Example:

```text
Cognitum Analyst   Reviewing 42 documents   68%   Pause   Stop
```

## D2. Expanded state

1. Current objective.
2. Recent actions.
3. Pending approvals.
4. Model and token usage.
5. CPU and memory consumption.
6. Network activity.
7. Capability grants.
8. Witness status.
9. Estimated cost.
10. Text or voice instruction field.

## D3. Agent states

Unmistakable colors and icons:

```text
Idle · Running · Waiting for approval · Paused · Capability denied ·
Error · Quarantined · Completed
```

The user must always be able to pause or terminate an agent with one
action.

## D4. Multiple agents

Do not place every agent in the bar. Show:

```text
Active agent · Two secondary agent icons · Additional agent count ·
Aggregate resource usage · Approval count
```

Selecting the count opens the full swarm view.

## D5. Platform placement

1. macOS menu bar or floating notch style dock.
2. Windows system tray plus optional floating dock.
3. Linux panel or floating dock.
4. Browser toolbar inside RVForge.
5. Mobile Live Activity for status and approvals.
6. RVM appliance dashboard for active coherence domains.

## D6. Security requirements

The dock chrome must be controlled by RVForge, **never by the agent**.
Agents may provide task text and progress but cannot alter the pause
button, trust badge, network indicator, or permission state.

System messages and agent generated content must be visually distinct.
Otherwise a malicious agent could display a fake approval or claim it has
stopped when it is still executing.

## D7. Key capability card

Expanding the agent icon immediately shows:

```text
Verified publisher · RVM or WASM isolated · Local model ·
Network disabled · Selected folder access · Encrypted memory ·
Witness chain valid
```

## D8. Noise control

The largest UX risk is excessive noise from perpetual agents. The fix is
event thresholds: show only approvals, policy violations, cost limits,
failures, and meaningful milestones.

## Dock acceptance test

From any application, the user can identify the active agent, understand
what it is doing, inspect its permissions, and terminate it within five
seconds and two interactions.
