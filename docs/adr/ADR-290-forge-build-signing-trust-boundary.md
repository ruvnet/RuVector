# ADR-290: Forge Build and Signing Trust Boundary

- **Status**: Proposed
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — partial groundwork: CLI signing-key hygiene landed (no key material in logs, world-readable keys refused), witness receipts + provenance on every build/verify/publish. Hosted build service, HSM/KMS signing, and worker lifecycle remain unimplemented.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-285, ADR-286, ADR-287, ADR-288, ADR-289, ADR-291, ADR-292, ADR-293
- **Tags**: forge, rvf, build-service, signing, notarization, provenance, isolation, security

## Context

RVForge converts one canonical `.rvf` artifact into installable packages for
Windows, macOS, Linux, and RVM. Cross compilation alone is insufficient: DMG and
Debian bundles require their native build environments, so the hosted service
must run Linux, Windows, and macOS builds on native workers.

That requirement creates the project's primary failure mode. A submitted RVF
contains the agent, model, memory, interface, policies, and signatures — it is
confidential executable intellectual property, and a hosted build necessarily
places it on infrastructure the customer does not own. The same build path also
handles publisher signing identities, because macOS requires Apple code signing
and notarization for normal external distribution and Windows signing avoids
SmartScreen warnings.

The greatest risk is therefore compromising proprietary RVFs or publisher
signing identities during hosted builds. Neither risk is addressed by build
correctness. Both are addressed by where data is allowed to live, how long it
lives there, what the worker is allowed to reach, and whether the customer has
the option to not use the hosted service at all.

This ADR fixes the trust boundary around the hosted build service and the
signing path. The execution contract for the produced packages is ADR-284; the
hosted RVM security boundary is ADR-285; the compatibility contract that
constrains which runtime a build may target is ADR-291.

## Decision

The hosted service treats every submitted RVF as confidential executable
intellectual property. The build and signing path is constructed so that a
compromise of a single worker exposes neither a customer's RVF beyond that
job's lifetime nor any private signing key at all.

### 1. Every submitted RVF is confidential

Submitted RVFs, build manifests, capability policies, prompts, and model data
are tenant-private by default. Private RVF data is never shared between tenants.
Only public dependencies are cached, and they are cached by verified hash.

Build workers never execute the submitted RVF. Inspection, validation, packaging,
and scanning all operate on the RVF as inert data. Every produced package is
scanned without executing its RVF payload.

### 2. Ephemeral per-job workers

Every build is isolated in a fresh worker. Workers are single-use: a worker
serves exactly one job and is destroyed within five minutes of job completion,
including on failure and cancellation. Build targets still run concurrently,
each on its own worker.

Reuse of a worker across jobs, across tenants, or across retries of the same job
is prohibited. A retry is a new job on a new worker.

### 3. Data lifetime

- Upload, storage, and artifact delivery are encrypted.
- Uploaded private RVFs are deleted within sixty minutes of upload unless
  retention is explicitly enabled by tenant policy.
- Artifacts are retained according to tenant policy, and the service supports
  customer-controlled storage for tenants that decline service-side retention.
- Uploads are resumable multipart; a partially uploaded RVF that is never
  completed is subject to the same sixty-minute deletion clock.

### 4. Network posture during packaging

Network access is disabled during final packaging. The single exception is
Apple notarization, which requires access to Apple's service and is permitted
only for the notarization step and only to that destination.

Dependency versions and hashes are pinned. Dependency hydration happens before
the network is closed, from pinned lockfiles against the verified-hash public
cache.

### 5. Signing without key export

Signing keys are held in an HSM, a KMS, or a customer-controlled signing
service. Private signing keys are never exported, and no signing key material
is present in a worker's memory or filesystem. The worker submits a digest to
the signing service and receives a signature.

The service supports four signing postures:

1. Customer-supplied signing identities.
2. Signing operations through KMS or HSM without exporting private keys.
3. Cognitum signing for verified marketplace packages.
4. Unsigned development builds, which are labeled as such in the provenance
   record and are not eligible for marketplace distribution.

Platform-specific signing requirements:

- **Windows** — organization-validated and extended-validation certificates,
  Azure Key Vault signing, trusted timestamps, and signed NSIS and MSI
  installers.
- **macOS** — Apple Developer ID signing, Apple notarization, Intel, ARM64, and
  universal packages, with notarization results stapled to both the application
  and the DMG.
- **Linux** — GPG-signed Debian and RPM packages, generated repository metadata,
  and independent verification of the package and the embedded RVF.

Revoked publisher identities are rejected at installation or execution.

### 6. Logs and audit

Logs are stripped of tokens, paths, prompts, model data, and environment
secrets before they leave the worker. The CLI never places signing secrets in
command history or logs. Structured build status and logs are streamed to the
submitting tenant only.

All privileged service actions are written to an immutable audit log. Privileged
actions include job admission, worker provisioning and destruction, signing
invocations, artifact release, retention changes, and deletion.

### 7. Enterprise private builder is the mitigation

The required mitigation for the primary failure mode is an enterprise private
builder plus HSM-based signing. In this mode the build runs entirely inside the
customer environment, so models, prompts, data, and signing credentials never
reach Cognitum infrastructure. The hosted service is a convenience tier; the
private builder is the answer for customers whose RVF or signing identity
cannot leave their perimeter.

The service supports private enterprise deployment as a first-class
configuration, not as a fork. Federated enterprise builders remain deferred.

### 8. Provenance carried by every output

Every output includes:

```text
RVF identity
software inventory
build manifest
source hash
builder identity
witness receipt
```

Build output is traceable to the input RVF, runtime version, source revision,
builder, and signing identity. The embedded RVF hash is identical across every
platform package. Alongside the installers, each build emits SHA256 checksums,
a build provenance record, an RVF witness receipt, and a verification report.

A build failure produces no partially trusted artifact. Artifacts are released
only after the complete build, sign, and verify sequence succeeds.

### 9. Trust boundary summary

```text
customer environment          | hosted service                | signing service
------------------------------|-------------------------------|-----------------
RVF authored and signed       | encrypted upload              |
build manifest generated      | fresh single-use worker       |
local validation              | packaging, network disabled   |
                              | digest submitted              -> HSM/KMS signs
                              | signature returned            <- no key export
                              | notarization (network, Apple) |
                              | provenance + witness emitted  |
verify downloaded artifact    | worker destroyed <= 5 min     |
                              | RVF deleted <= 60 min         |
```

Enterprise private builder collapses the middle column into the first.

## Acceptance criteria

1. A submitted signed RVF produces installable Windows, macOS, and Linux
   packages, each of which executes offline, preserves the original RVF
   identity, denies undeclared capabilities, and emits verifiable build and
   runtime witness records.
2. Every embedded RVF across `.exe`, `.msi`, `.dmg`, `.deb`, and `.AppImage`
   has the same SHA256 hash.
3. No build worker executes the submitted RVF at any point in inspection,
   packaging, or scanning.
4. A worker is unreachable and destroyed within five minutes of job completion,
   verified for successful, failed, and cancelled jobs.
5. An uploaded private RVF is unrecoverable sixty minutes after upload when
   retention is not explicitly enabled.
6. Network egress attempts during final packaging fail, except the notarization
   step's traffic to Apple.
7. No private signing key is present in worker memory or filesystem, and no
   code path can export one from the HSM, KMS, or customer signing service.
8. Streamed and stored logs contain no tokens, absolute customer paths, prompts,
   model data, or environment secrets, verified against a redaction fixture.
9. Every privileged service action appears in the immutable audit log with the
   acting identity and job identifier.
10. Build provenance and witness records verify independently of the service
    that produced them.
11. A failed build releases no artifact that carries a signature or provenance
    record.
12. An enterprise private build completes with no network call to Cognitum
    infrastructure carrying RVF, prompt, model, or credential data.
13. A package signed by a revoked publisher identity is rejected at installation
    or execution.
14. Tampering with a produced package prevents execution.

## Consequences

### Positive

- A single-worker compromise exposes at most one job's RVF, and no signing key.
- Customers whose IP cannot leave their perimeter have a supported path.
- Provenance is uniform across platforms, so verification tooling is one
  implementation rather than six.
- Deleting build data on a clock removes an accumulating breach surface.

### Negative

- Single-use workers forfeit warm-cache speed; the verified-hash public
  dependency cache only partially recovers it.
- HSM/KMS signing adds latency and an external dependency to every signed build.
- Closing the network during packaging means any missing dependency is a hard
  build failure rather than a silent fetch.
- Sixty-minute deletion makes post-hoc debugging of customer builds difficult
  without explicit retention opt-in.
- Supporting an enterprise private builder doubles the deployment surface that
  must be tested and released.

## Alternatives Considered

- **Shared long-lived builders with per-job cleanup**: rejected because
  cleanup correctness becomes the entire security argument, and cross-tenant
  residue is unfalsifiable in practice.
- **Cross compilation from a single Linux fleet**: rejected because DMG and
  Debian bundles require native build environments.
- **Uploading signing certificates to the worker**: rejected outright; it
  converts a build compromise into a publisher-identity compromise.
- **Keeping the network open during packaging for convenience**: rejected
  because it is the natural exfiltration path for a compromised build step.
- **Indefinite retention of uploaded RVFs to speed rebuilds**: rejected as the
  default; available only as explicit tenant-configured retention.
- **Treating the enterprise private builder as a later phase**: rejected
  because it is the required mitigation for the primary failure mode, not an
  enhancement.

## Implementation Surfaces

- `forge service` — scheduler, artifact registry, signing broker, audit log
- `forge workers` — Linux, Windows, macOS single-use builders
- `rvf forge core` — Rust packaging and verification library
- `@ruvector/forge` — CLI submit, status, download, verify
- `POST /v1/builds`, `GET /v1/builds/{id}`, `GET /v1/builds/{id}/artifacts`,
  `POST /v1/builds/{id}/cancel`, `POST /v1/builds/{id}/verify`,
  `DELETE /v1/builds/{id}`
- provenance record, software inventory, and witness receipt schemas
- log redaction filters and immutable audit storage
- enterprise private builder deployment package
