# ADR: Focused Python packages over a shared RuVector runtime

Status: Proposed, implemented for review
Date: 2026-10-11

## Context

The existing ruvector-py extension already binds graph, attention and clustering Rust crates. Rebuilding the same extension separately for each Python distribution would multiply wheel builds and native patch obligations. The first portfolio should provide real task APIs, not empty packages created only for search traffic.

## Decision

Create ruvector-graph, ruvector-cluster and ruvector-rerank with independent import namespaces, a coordinated 0.1.0 version and an exact dependency on ruvector 0.1.2. Add bounded graph traversal, structured clustering results, and stable attention ranking. Native algorithms stay in the existing PyO3 module. Document that attention ordering is equivalent to dot product, and no trained relevance improvement is implied.

Keep the existing ruvector native wheel pipeline. Add a separate portfolio workflow that tests source-built native bindings and the released runtime, rebuilds sdists, audits Python and Rust dependencies and publishes tested distributions using a configured PyPI token or OIDC. Version changes merged into main publish automatically after gates pass; ordinary edits without a version change do not publish. Matching release tags with main ancestry are also supported. Published packages are then installed and tested on Linux, macOS and Windows. Token publishing does not produce OIDC attestations. Configure publisher identities out of band; never transfer long-lived tokens into chat.

## Consequences

Three portable Python wheels reuse one native runtime rather than adding fifteen native platform builds for three packages across five platforms. Installation footprint stays equal to the shared runtime plus small Python wrappers. Exact pinning avoids silent ABI/API drift but couples upgrades. This is not a portfolio of independent slim Rust extensions. Split binaries only after measured installation cost or independent release needs justify the extra maintenance.

Search metadata describes actual functions; it makes no performance, adoption or ranking promises. PyPI uploads are nontransactional. Broken releases are yanked and fixed under a new version.

## Acceptance

Build wheels and sdists; install in clean environments; run examples and independent numeric/graph invariants against the real native extension. Require audit gates, exact version metadata, and successful tag ancestry checks before publishing. Verify PyPI installation on supported platforms after publication. Do not equate source tests with cross-platform wheel verification.
