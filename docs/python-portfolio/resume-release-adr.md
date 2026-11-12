# ADR: Recover the frozen Python portfolio release

Status: Proposed
Date: October 12, 2026
Scope: RuVector companion distributions at version 0.2.0

![Release recovery overview](recovery-2026-10-12.svg)

## Context

The initial portfolio release partially reached PyPI. A new-project HTTP 429 interrupted publication. Package names, versions and actual published files must be inspected again before any upload. A successful wheel upload does not imply its source distribution or another package was uploaded.

The original companion artifacts require `ruvector==0.1.3`. Current main changes the same companion version to depend on runtime 0.1.4. Rebuilding 0.2.0 from current main would silently mix release identities and cannot repair the original release faithfully.

The stopped historical supervisor had a deadline ending October 11 at 22:35 UTC. Its receipt is useful evidence, not proof that no other publisher is running. Recovery must establish one current publisher owner from live process state, workflow runs, private Slack claims and existing receipts.

## Decision

Recover only the original immutable artifacts. Do not add another Python library while this portfolio delivery is unfinished.

The frozen release identity is:

| Field | Exact value |
| --- | --- |
| Source commit | `fccf36e9a76d35bce6750dfd3b25802cae3d683b` |
| Companion version | `0.2.0` |
| Required native runtime | `0.1.3` |
| GitHub Actions run | `38169271248` |
| Artifact ID | `11713787141` |
| Artifact archive SHA256 | `b4c92cae86c753305abef7aad395fdd9d91c7e84c3c887342defa7a3e4607241` |
| Expected companion artifacts | 22 files across 11 distributions |

The [frozen inventory](../../python/portfolio/frozen-release-0.2.0.json) records each expected filename, byte size and SHA256. It provides release identity; it does not authorize publication.

A recovery implementation must:

1. Confirm no live publisher already owns this release; claim the scoped branch and package work in the existing private Slack thread.
2. Download the original artifact by immutable run and artifact identity. Verify the archive checksum and all 22 member checksums against the committed inventory. Reject path traversal, extra members, missing members and mismatched bytes.
3. Fetch the current PyPI JSON for every expected distribution/version. An existing expected filename must match its frozen SHA256. Unexpected release filenames or mismatches stop recovery.
4. Stage only files missing from PyPI. Stage nothing when all expected files are already present. Avoid `skip-existing` as a substitute for preflight identity verification.
5. Require independent review and repository release checks before using the protected `pypi` environment. Publish through the established GitHub Actions identity without broadening secrets, changing protection or bypassing approval.
6. Stop on HTTP 429. There is no confirmed upload cooldown here. Do not infer that the separate documented Trusted Publisher registration limit applies to this upload. Reinspect state and receipts before a later authorized attempt.
7. Requery PyPI after publication, verify every expected filename and hash, and execute installed-artifact consumer tests. Record implementation, checks, merge, publication and verification separately.

A transport failure or ambiguous upload response requires read-only reconciliation first. A manifest cannot establish whether a network side effect completed.

## Architecture

```mermaid
flowchart TD
    A["Frozen artifact and ledger"] --> B["Archive and file verification"]
    C["Current PyPI inventory"] --> D["Identity and missing file check"]
    B --> D
    D --> E{"Safe to publish?"}
    E -->|Mismatch or 429| F["Stop and record blocker"]
    E -->|Reviewed missing files| G["Protected publishing"]
    G --> H["PyPI and consumer verification"]
```

## Tutorial acceptance

The companion [attention tutorial](../../python/portfolio/tutorials/README.md) uses only published `ruvector-attention==0.2.0`, runtime `ruvector==0.1.3` and NumPy 2.2.6. Its exact code passed a clean PyPI install on CPython 3.10.22, Linux x86_64 and glibc 2.39 on October 12, 2026. It checks the native weighted sum against a float64 reference and rejects invalid dimensions and nonfinite queries.

This smoke test does not prove all portfolio files are published, benchmark performance or establish fresh macOS/Windows execution. The example does not download models, generate text or train parameters.

## Consequences and alternatives

Frozen-file recovery preserves original published wheel metadata and avoids duplicate version content. Original artifacts must remain available and independently verified. If the archive becomes unavailable or its checksum differs, recovery stops; a reviewed new version is preferable to rebuilding historical bytes.

Rebuilding from current main was rejected because the runtime dependency changed without changing the companion version. Blindly repeating the supervisor was rejected because it could duplicate a publisher and obscure partial side effects. Ignoring HTTP 429 or modifying credentials was rejected because neither repairs the release gate.

Rollback means stopping promotion and preserving the recorded inventory. PyPI files are not overwritten. Any later defect requires a reviewed release correction under package policy; successful publication is not treated as reversible file replacement.

## Evidence and references

- [Original portfolio PR #1188](https://github.com/ruvnet/RuVector/pull/1188)
- [Frozen source](https://github.com/ruvnet/RuVector/tree/fccf36e9a76d35bce6750dfd3b25802cae3d683b/python/portfolio)
- [Original artifact run](https://github.com/ruvnet/RuVector/actions/runs/38169271248)
- [Security boundaries](../../python/portfolio/SECURITY.md)
- [Frozen file inventory](../../python/portfolio/frozen-release-0.2.0.json)
