# ADR-353: npm constellation and skills discovery

- Status: Proposed; inventory and adapter implementation under validation
- Date: 2026-10-11
- Updated: 2026-10-11
- Related: RuVector ADR-065, ADR-305, ADR-306, ADR-313, ADR-335; Ruflo ADR-150 and ADR-322A (different repository namespaces)

## Context and observed evidence

The initial public npm maintainer sweep enumerated **398 packages** and **486 internal dependency edges**. All returned current maintainers include `ruvnet`; this is not proof that the current token may publish each package. The authenticated package-access endpoint returned HTTP 403. Registry search covers public indexed packages, not private packages or an authoritative account export.

The source audit found 557 tracked manifests in 25 canonical local repositories, 450 named non-private candidates, 357 unique names, and 89 duplicate names. Examples, platform packages, bundled internals and stale checkouts are not independent release targets. See [source audit](../ecosystem/source-audit-2026-10-11.md) and [registry snapshot](../ecosystem/snapshot-2026-10-11/inventory.json).

The live graph includes a runtime cycle `strange-loops` ↔ `sublinear-time-solver`; the merged local manifest graph did not. The published graph is therefore a separate compatibility surface that must be retained and tested. Latest direct package versions returned no advisories from npm's bulk endpoint on this scan; no transitive or source-security claim follows.

## Decision

Create a focused `@ruvector/constellation` adapter and CLI. Its registry sweep records version, maintainer, source coordinates, platform/engine requirements, lifecycle hooks, integrity, attestation presence, typed dependency edges, public advisory response and explicit incomplete/error states. Preserve all leaf/platform packages in coverage; group them for remediation rather than removing them from denominators.

Layer the ecosystem:

1. Native and WASM leaves: RuVector primitives and platform packages. No Ruflo or MetaHarness hard dependency.
2. Storage and SDKs: core, RVF, graph, attention, SONA, AgentDB. Existing integration contracts remain authoritative.
3. Adapters: constellation and existing RuVector MetaHarness bridge. Optional peer dependencies and lazy imports for optional capabilities.
4. Applications: `ruvector`, `ruflo`, `metaharness`, agentic-flow. Compose SDKs here; adding a dependency requires an exercised consumer operation and a missing-optional test.
5. Agent surfaces: focused MCP tools, host plugin manifests and discoverable skills. These are distinct install surfaces; installing a skill does not register an MCP or install its runtime.

The constellation MCP reads a fixed report and exposes package, edge and improvement queries. No arbitrary shell, install, mutation, credentials or publish tool. Report metadata is untrusted data. Client configuration accepts explicit installed executables. Existing Ruflo MCP handles coordination and memory; constellation does not pretend that bookkeeping executes work.

Use the real `@metaharness/router` API only with finite dimension-consistent embeddings, real quality labels, complete prices and an enforced quality threshold. No synthetic labels derived from downloads or metadata coverage. Missing data means a deterministic priority queue, explicitly labeled heuristic. Optional peers avoid making the package graph a forced installation graph.

## Skills and visibility

Keep `skills/ruvector-constellation/SKILL.md` as the canonical discovery entry, package a copy into the npm tarball, and test `npx skills add <source> --list` plus installation into clean Codex/Claude agent directories. Advertise canonical repository coordinates, exact package names, runtime requirements, practical examples and source links. Measure successful discovery/install/first operation, not number of dependency edges or keyword stuffing. Do not fabricate marketplace listing or industry ranking results.

One focused skill covers package selection and integration; specialized skills should be added only for distinct workflows with working examples. MCP and plugin registration require real host tests before claiming coverage. Native modules need their own OS/architecture test matrix; the cloud observer does not certify them.

## Compatibility and security

Preserve edge kinds: dependency, optional dependency, optional/nonoptional peer, development dependency. Detect strongly connected components only on runtime installation edges, while keeping peer edges visible. Never bulk rewrite semver ranges. Track both source SHA and published version; immutable npm versions cannot be replaced.

A missing advisory result is unknown, not zero vulnerabilities. Registry attestation presence is not attestation verification. Public metadata scans cannot certify tarball contents, token scope, transitive dependencies or exploitability. Source-to-package mapping and release allowlists must be independently reviewed before publication.

## Acceptance and rollout

- Fully paginated discovery; deduplication and coverage/error counters.
- Deterministic typed graph; cycle, missing metadata and network failure tests.
- Actual MCP initialize/list/call smoke; bounded inputs and outputs.
- Skill discoverable through the upstream skills CLI; tarball contains skill and license.
- Optional routing tests prove missing dependency and below-quality failure behavior.
- Lockfile-backed install and security audit of the new adapter.
- Existing packages change only through focused compatibility-tested batches.

Start with the observer/adapter and release proof. Next batches: repair the published cycle after source ownership verification; align source coordinates and license metadata; test existing Ruflo/MetaHarness/RuVector integration edges; expand measured SDK/MCP/skill coverage. Do not independently publish Ruflo internals: its stable train is `@claude-flow/cli` → `claude-flow` → `ruflo`.

## Alternatives and consequences

Hard-depending every package on orchestration would increase cold-start/install cost, expose native leaves to unrelated failures and introduce cycles. A single mega-plugin would hide package-specific compatibility. A registry-only crawler cannot prove actual integrations, so its findings produce candidate work, not release approval. The adapter adds one package and an explicit maintenance obligation, justified by reusable CLI, MCP and scheduled reporting surfaces.

## References

- [Skills CLI](https://github.com/vercel-labs/skills)
- [npm provenance](https://docs.npmjs.com/generating-provenance-statements/)
- [Companion weekly control loop](ADR-354-weekly-cloud-run-npm-improvement.md)
