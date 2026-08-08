# ADR-283: RVForge — One Canonical RVF to Signed Platform Installers

- **Status**: Accepted
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — implementation in progress: CLI (validate/build/verify, embedded+thin, inventory, witness chains — 137 tests), rvf-forge-core (103 tests), Reader scaffold+dock (90 tests), 3-OS CI landed; hosted build service + Tauri bundling of real installers pending.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-280, ADR-284, ADR-285, ADR-286
- **Tags**: rvf, forge, packaging, tauri, installers, signing, provenance, ci

## Context

An `.rvf` artifact already carries an agent, its model, its memory, its
interface, its policies, and its signatures as one durable self-contained
file (ADR-280). What it does not have is a way to reach a person who expects
to double-click an installer. Distributing an agent today means hand-building
per-platform packages, which reintroduces exactly the divergence the format
was designed to eliminate: three build trees, three signing stories, and no
guarantee that the agent inside the `.dmg` is bit-identical to the agent
inside the `.msi`.

Cross compilation alone does not close the gap. DMG and Debian bundles
require their native build environments, so a real pipeline needs Linux,
Windows, and macOS workers. Tauri already emits NSIS executables, MSI, DMG,
macOS applications, Debian packages, RPM, and AppImage, and can embed an
arbitrary resource into each — which makes it the natural packaging layer
rather than something to rebuild.

The largest risk in a hosted build service is the opposite of a build
failure: a successful build that leaked a proprietary RVF or a publisher's
signing identity. A submitted RVF is confidential executable intellectual
property, and the pipeline has to be designed around that assumption instead
of adding confidentiality later.

The product goal is that an RVF agent becomes a normal installable
application a nontechnical user can download, double tap, and run securely —
offline-capable, quarantined, with learned state stored separately from the
signed application and every action recorded in a witness chain. The full
product vision (white-label partner distribution, agent marketplace,
enterprise governance, commercial licensing modes) and the end-user
acceptance test live in `docs/research/rvf-forge/requirements.md` §"Product
Vision & Distribution". Because arbitrary Python or Node applications cannot
instantly become universally portable, the initial secure product
prioritizes Rust and WASM agents, with broader compatibility added later.

## Decision

Build **RVForge**: an npm CLI (`@ruvector/forge`) plus an optional hosted
build service that converts one canonical `.rvf` into installable packages for
Windows, macOS, Linux, and RVM.

```bash
npx @ruvector/forge build agent.rvf windows macos linux
```

```text
AgentSetup.exe   Agent.msi     Agent.dmg
Agent.deb        Agent.rpm     Agent.AppImage
Agent.rvf
```

### 1. Pipeline

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

The canonical RVF is the input and also one of the outputs. Its identity,
contents, policies, and signatures remain unchanged across every generated
package. A small Rust **RVF Reader** verifies and executes the artifact using
WASM for universal execution, native libraries for acceleration, QEMU or KVM
for Linux runtime compatibility, and RVM for native coherence domains
(ADR-284). **Tauri** is the packaging layer and embeds the RVF as an
application resource.

**GitHub Actions is the first worker implementation.** It already provides
Linux, Windows, and macOS runners with ephemeral per-job machines, which
matches the isolation requirement without standing up a fleet first. Hosted
runner pricing is roughly $0.006/Linux-min, $0.010/Windows-min, and
$0.062/macOS-min, keeping raw compute below about $1 for most builds.

### 2. Core invariants

These are non-negotiable and every implementation surface below exists to
uphold one of them:

1. **The embedded RVF hash is identical across every platform package.**
2. **Build workers never execute the submitted RVF.** Packaging and scanning
   are inspection-only operations.
3. The RVF Reader verifies signatures before loading executable segments.
4. Undeclared capabilities remain inaccessible (ADR-286).
5. Build output is traceable to the input RVF, runtime version, source
   revision, builder, and signing identity.
6. Mutable state never modifies the signed base identity.
7. **Build failure produces no partially trusted artifact.** A failed job
   emits diagnostics, never a half-signed or unsigned-but-shipped installer.

### 3. Packaging modes

| Mode | Contents | Trade-off |
|---|---|---|
| **Embedded** (FR001) | Complete RVF plus RVF Reader in every installer | Executes with no internet access; duplicates large models across outputs |
| **Thin** (FR002) | Reader plus a signed RVF locator; reader downloads and verifies before execution | Can reduce a 5 GB installer to roughly 10–30 MB |
| **Shared reader** (FR003) | Reader installed once, `.rvf` registered as a file type | Subsequent RVFs need no platform packaging at all |
| **Enterprise private build** | Entire build runs inside the customer environment | Models, prompts, data, and signing credentials never reach Cognitum infrastructure |

The reader selects the strongest compatible runtime in the order **Native RVM
→ OS isolation plus WASM → WASM → Linux microVM → Unsupported** (FR004). That
order may be changed only by signed policy.

Runtime state is stored as encrypted RVF delta segments separate from the
immutable base artifact (FR005). Updates are signed, version constrained,
reversible, and linked to the prior RVF identity (FR006). Packages support
custom icons, publisher identity, application name, interface theme, license,
and installation text (FR007).

### 4. CLI command surface

Package name `@ruvector/forge`, Node.js 20 or later:

```bash
npx @ruvector/forge init
npx @ruvector/forge validate agent.rvf
npx @ruvector/forge build agent.rvf
npx @ruvector/forge submit agent.rvf
npx @ruvector/forge status BUILD_ID
npx @ruvector/forge download BUILD_ID
npx @ruvector/forge verify AgentSetup.exe
```

The CLI supports interactive and unattended execution, validates the RVF
locally before upload, generates a canonical build manifest, and supports both
local and hosted builds. Before submission it displays estimated build time,
output size, and cost. It resumes interrupted uploads, verifies downloaded
artifacts automatically, returns stable machine-readable error codes, and
never places signing secrets in command history or logs.

### 5. Hosted API surface

```text
POST   /v1/builds
GET    /v1/builds/{id}
GET    /v1/builds/{id}/artifacts
POST   /v1/builds/{id}/cancel
POST   /v1/builds/{id}/verify
DELETE /v1/builds/{id}
```

The service runs Linux, Windows, and macOS builds on native workers, isolates
every build in a fresh worker, builds targets concurrently, caches public
dependencies by verified hash, and never shares private RVF data between
tenants. It streams structured build status and logs, supports resumable
multipart uploads and customer-controlled storage, supports private enterprise
deployment, and retains artifacts according to tenant policy.

### 6. Security posture

Default-deny capability policy; Ed25519 or stronger RVF signature
verification; per-job worker isolation; encrypted upload, storage, and
artifact delivery. Signing keys live in an HSM, KMS, or customer-controlled
signing service and private keys are never exported. Network is disabled
during final packaging unless notarization requires access. Dependency
versions and hashes are pinned. Logs are stripped of tokens, paths, prompts,
model data, and environment secrets. Build workers are destroyed within five
minutes of job completion, and uploaded private RVFs are deleted within sixty
minutes unless retention is explicitly enabled. Every produced package is
scanned without executing its RVF payload, revoked publisher identities are
rejected at installation or execution, and all privileged service actions are
written to an immutable audit log.

### 7. Signing and provenance

Four signing paths are supported: customer-supplied signing identities;
signing through KMS or HSM without exporting private keys; Cognitum signing
for verified marketplace packages; and unsigned development builds.

- **Windows** — organization-validated and extended-validation certificates,
  Azure Key Vault signing, trusted timestamps, signed NSIS and MSI installers.
- **macOS** — Apple Developer ID signing, Apple notarization, Intel/ARM64/
  universal packages, notarization results stapled to the app and the DMG.
- **Linux** — GPG-signed Debian and RPM packages, repository metadata, and
  independent verification of the package and the embedded RVF.

Every output carries the RVF identity, software inventory, build manifest,
source hash, builder identity, and witness receipt. Alongside the installers,
each build emits SHA256 checksums, a build provenance record, an RVF witness
receipt, and a verification report.

Every generated package embeds the RVM integration contract:

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

Forge rejects combinations absent from the published RVM compatibility matrix.

### 8. Scope

**MVP**: npm CLI; local validation; hosted Linux, Windows, and macOS builds;
embedded and thin packaging; Tauri-based RVF Reader; WASM execution; Windows
and macOS signing; Linux package signing; immutable base plus encrypted state
delta; build provenance, software inventory, and witness receipts; a web
dashboard for submission, status, and downloads; tenant isolation and
automatic artifact deletion.

**Deferred**: Microsoft Store distribution; Apple App Store distribution;
Android and iOS packages; hardware TEE attestation; bare-metal RVM
installation; GPU capability brokering; a public RVF marketplace; automatic
model quantization; delta installer generation; federated enterprise builders.

**Performance targets**: local validation under two seconds for RVFs below
1 GB excluding full payload hashing; cached build under three minutes per
platform; uncached build under ten minutes excluding Apple notarization;
shared-reader package overhead below 30 MB; embedded package overhead below
50 MB plus RVF size; reader startup under 500 ms before model loading; hosted
availability at least 99.9 percent; build success above 99 percent for valid
specifications.

**Compatibility**: Windows 10/11 x64, Windows 11 ARM64, macOS 13+ on Intel and
Apple Silicon, Ubuntu 22.04 and 24.04, Debian 12+, generic Linux via AppImage,
browser execution via WebAssembly, and native `.rvf` output for RVM
appliances.

## Acceptance criteria

Submit one signed RVF and generate `.exe`, `.msi`, `.dmg`, `.deb`, and
`.AppImage` outputs. The release passes only when:

1. Every installer completes successfully on a clean operating system.
2. Every embedded RVF has the same SHA256 hash.
3. Offline execution succeeds in embedded mode.
4. Tampering prevents execution.
5. Undeclared filesystem and network access is denied.
6. State survives restart without modifying the base RVF.
7. Uninstallation removes the runtime and state according to policy.
8. Build provenance and witness records verify independently.

## Consequences

### Positive

- One artifact, one identity: the agent a user installs on Windows is
  provably the agent a reviewer audited on Linux.
- Publishers get platform reach without maintaining three build trees.
- Shared-reader mode makes subsequent RVFs a zero-packaging distribution.
- Enterprise private builds keep proprietary models and signing credentials
  off shared infrastructure entirely.

### Negative

- A hosted service that handles confidential RVFs and signing material is a
  high-value target and carries permanent operational security cost.
- macOS notarization introduces an external dependency with latency outside
  our control, which is why it is excluded from the build-time target.
- Embedded mode duplicates large models across six output formats.
- Native workers for three operating systems cost more than cross compilation
  would, and that cost is structural rather than an optimization gap.

## Alternatives Considered

- **Cross-compile everything from Linux**: rejected because DMG and Debian
  bundles require their native build environments.
- **Ship a per-platform SDK and let publishers package**: rejected because it
  reintroduces divergent artifacts and gives up the identical-hash invariant.
- **Let workers run the RVF to smoke-test the package**: rejected outright.
  Executing submitted confidential code on shared build infrastructure is the
  primary failure mode this design exists to prevent; packages are scanned
  without executing their payload.
- **Persistent build machines for speed**: rejected in favor of fresh
  per-job workers destroyed within five minutes.

## Implementation Surfaces

```text
@ruvector/forge     npm CLI and API client
rvf forge core      Rust packaging and verification library
rvf reader          Tauri desktop reader
forge service       Build scheduler, artifact registry, signing, billing
forge workers       Linux, Windows, macOS isolated builders
```

Delivery estimate: local packaging prototype two to three weeks; hosted
unsigned build matrix another three to four weeks; signing, notarization,
provenance, private builds, and billing another four to six weeks. Roughly
eight to twelve weeks for three engineers.
