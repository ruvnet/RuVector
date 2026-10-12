# Source ecosystem audit — 2026-10-11

Read-only inspection of tracked package manifests in canonical project directories (a `.git` directory; alternate worktrees excluded). Raw inventory: `/tmp/ecosystem-manifests.json`. No registry ownership, publish permission, artifact validity, or security-cleared status inferred from manifests. No repository mutation or publication performed.

## Inventory

557 manifests; 450 named, non-private manifest candidates; 357 distinct candidate names. 89 names have multiple local manifest sources. Examples, stubs, generated manifests and internal bundled components remain in these candidate counts. They MUST NOT be automatically promoted to a release allowlist.

| Checkout | All manifests | Non-private named candidates |
|---|---:|---:|
| LatentMesh-1 | 4 | 0 |
| RuView | 9 | 4 |
| agent-harness-generator | 71 | 66 |
| agents-of-the-dawm | 3 | 1 |
| api | 3 | 1 |
| autogenous | 1 | 1 |
| cognitum-slack | 1 | 1 |
| console-scratch | 24 | 4 |
| core-memory | 1 | 0 |
| discovery | 3 | 1 |
| edge-chat | 1 | 0 |
| latentmesh | 4 | 0 |
| meta-llm | 3 | 0 |
| metaharness | 87 | 79 |
| myrtc-core | 1 | 1 |
| neural-trader | 76 | 70 |
| ruClip | 1 | 1 |
| ruflo | 55 | 48 |
| ruflo-dashboard | 1 | 0 |
| ruos-desktop | 1 | 0 |
| ruos-desktop-1 | 2 | 0 |
| ruvector | 176 | 160 |
| ruvnet-cognitum | 14 | 10 |
| sdks | 1 | 1 |
| website | 14 | 1 |

## Critical release/source findings

- `ruflo/AGENTS.md` fixes the stable public train to `@claude-flow/cli`, then `claude-flow`, then `ruflo`; internal @claude-flow components are bundled, not normal standalone publish targets. Clean reviewed source, umbrella version audit, registry verification, and latest/alpha/v3alpha alignment are required. CLI alone receives helper-signing configuration. Source HEAD inspected: `9fa701a466159242dfec06379ce1278c849a3e34`.
- `metaharness` HEAD `d5833dc6512ac1adeeef91a331c29055cd8a4dbb` and `agent-harness-generator` HEAD `213c8b0bbd27a746b5c785fa9f5bca4669056935` have overlapping package names at different versions: metaharness 0.4.16 vs 0.3.1; router 0.4.0 vs 0.3.2; flywheel 0.1.11 vs 0.1.3. Resolve authoritative source by registry repository + release history, not directory name.
- `metaharness/packages/agent-harness-generator-lib/package.json` pins metaharness 0.1.5 although local CLI is 0.4.16. Updating needs export compatibility testing, not a blanket version bump.
- Ruflo stable workflow uses an explicit stable tag, actor guard, immutable checkout and serialized release train. Some actions use mutable major tags (`actions/checkout@v4`, `actions/setup-node@v4`, `pnpm/action-setup@v6`): harden to reviewed SHA pins.
- MetaHarness `.github/workflows/publish.yml` uses pinned action SHAs and explicit tag SHA binding. Native build failures are tolerated and comments explicitly state native platform publishing is TODO; do not claim all native variants are release-verified. It pipes a remote wasm-pack installer into sh and falls back from npm ci to npm install: reproducibility/security hardening candidates.

## Real dependency constellation

- `ruflo/ruflo/package.json`: ruflo 3.54.1 depends on @claude-flow/cli ^3.54.1.
- `ruflo/v3/@claude-flow/cli/package.json`: optional @metaharness/darwin ~0.10.2, flywheel ~0.1.10, radio ~0.1.0, turn-credit ~0.1.0, ruvector ^0.2.27, agentdb ^3.0.0-alpha.17; optional peers router ^0.4.0, metaharness ^0.4.1 and @ruvector/typesafe ^0.1.0. Integration already exists; improve contract tests instead of adding redundant deps.
- `metaharness/packages/kernel-js/package.json`: direct @ruvector/emergent-time ^0.1.0, optional peer @ruvector/rvf ^0.2.0 and platform optional packages at 0.1.0.
- `metaharness/packages/router/package.json`: optional peer @ruvector/tiny-dancer ^0.1.21.
- `metaharness/packages/create-agent-harness/package.json`: direct darwin, weight-eft, redblue, flywheel, turn-credit, avo; optional @ruvector/ruvllm ^2.5.6 and optional peer kernel ^0.1.0.
- `metaharness/packages/jujutsu/package.json`: agenticow ^0.2.0 and agentic-jujutsu ^2.3.6; `autogenous/packages/radio-moe/package.json`: @metaharness/radio ^0.1.0.
- Graph scan merging candidate name sources found no cycles among dependencies+optionalDependencies. Including peers exposes CLI→codex→CLI optional-peer backedge, which is not a runtime hard-dependency cycle. Avoid adding Ruflo/MetaHarness as hard dependencies into low-level RuVector packages; compose them in adapters/plugins.
- Floating constraints worth review: @claude-flow/performance uses @ruvector/attention and @ruvector/sona latest; @metaharness/oo-agents uses @metaharness/radio *.

## Discovery and skills

- Ruflo has 379 tracked SKILL.md files and 52 plugin.json manifests; MetaHarness 16 / 2; agent-harness-generator 14 / 1. Counts include duplicated host variants and fixtures; not unique capability counts. No tracked `server.json` in these three repos; MCP registry publication readiness remains unverified.
- Ruflo root `SKILL.md` already documents `npx skills add ruvnet/ruflo --skill ruflo --yes` and `--all` at lines 68/71. Preserve curated selection, validate installation from clean temporary directories.
- MetaHarness skills are under `.claude-plugin/skills/*/SKILL.md`; CLI npm files list only dist/templates/genomes/README/LICENSE. Plugin skills are not explicitly shipped as npm CLI content. Need separate repository skill discovery and npm artifact verification; do not assume npm install makes skills discoverable.
- MetaHarness plugin identifies agent-harness-generator version 0.1.0, while CLI is 0.4.16; homepage/repository largely retain old agent-harness-generator coordinates. Router points to metaharness repository. Establish canonical identity and intentional independent version policy.
- Ruflo public shim package files are bin/src/dist/README/LICENSE; root claude-flow includes .claude-plugin/.claude; CLI includes .claude/plugins/catalog-manifest. Tests should inspect exact packed artifacts to prove each advertised discovery entry exists.

## Working programmatic seams and constraints

### MetaHarness router

`/home/ruvultra/projects/metaharness/packages/router/src/index.ts`: `Router.fromExamples(rows, prices, { k, qualityBar })`, rows `{ embedding:number[], scores:Record<string,number> }`; `.route(embedding)` returns `{id,predictedQuality,costPerMTok,metBar}`. Constructor requires nonempty candidates. Empty dataset throws, missing price defaults zero, embedding dimensions and numeric finiteness are not enforced. Caller must validate finite nonnegative complete prices, fixed finite embedding dimensions, label bounds, and enforce metBar. The router does not execute models or create labeled data. Never invent training labels from package metadata or treat predictions as proof of improvement.

### MetaHarness flywheel

`/home/ruvultra/projects/metaharness/packages/flywheel/src/index.ts` exports `runFlywheelGenerations`, `meetsPromotionRule`, `makeSigner`, `verifyReceipt`, `verifyReplayBundle`, `InMemoryLineageStore`. `run.ts` requires rootPolicy (Record<string,string>), proposer(base,target), evaluator(policy,suite), holdout, maxGenerations, signer. Evaluator returns `{primary,noopRate,costPerWin,regressed}`. Optional anchor provides independent non-regression check. `gate.ts` requires strictly improved noopRate and non-worsening primary/cost/security. Default gate is a candidate gate, never an npm publication authorization.

Safety constraints: validate all numeric values (comparisons alone do not reject every NaN); budget is checked once per generation, so an evaluator/proposer needs its own limit; callback checkpoint errors are swallowed; default lineage storage is volatile. Persist immutable replay artifacts separately. `makeSigner()` creates a fresh per-process Ed25519 key; embedded-key verification proves consistency but not trusted author identity. Production needs a configured Signer plus pinned public-key trust and artifact hashes. Only enable evaluation cache for deterministic evaluations.

### Ruflo

`/home/ruvultra/projects/ruflo/v3/@claude-flow/cli/package.json` exports `./ruvector` and `./mcp-tools`; `src/ruvector/index.ts` exports `coverageRouterTools`, `hooksCoverageRoute`, `hooksCoverageSuggest`, `hooksCoverageGaps`, `isRuvectorAvailable`. `src/ruvector/coverage-tools.ts` defines handler-based MCP tools; hooksCoverageRoute.handler({task, projectRoot?, threshold?, useRuvector?}) delegates to coverageRoute. These relate to code test coverage, not npm market visibility. Source inspection is not a runtime smoke pass. Prefer existing configured MCP for coordination and memory; a registration record is not execution.

## Recommended weekly acceptance

Collect registry names + versions + ownership, deduplicate authoritative sources; inspect exact tarballs; build typed hard/optional/peer dependency graph; test changed edges and expected missing-optional fallback; run advisories and artifact secret scans; validate `npx skills` discovery in temporary host directories; evaluate candidate changes against frozen installation/compatibility/security anchors; emit source-bound ADR/replay/report. Publish only changed packages in established release trains with immutable source, artifact digest and clean consumer smoke. Measure install success, discovery success, dependent compatibility, vulnerability deltas, latency/size; never use package-count/dependency-count inflation as quality.

## Registry-to-source crosswalk

The [398-package crosswalk](source-crosswalk.json) maps every published name to local manifest candidates: 197 have one candidate, 77 have multiple, and 124 have no matching named manifest in the inspected checkouts. A missing local match can be a generated platform manifest or a repository not checked out; it is not evidence that source does not exist. Repository metadata also includes third-party/co-maintained packages and placeholder URLs, so maintenance membership alone must not choose a release source. Publish access remains unverified per package despite the successful `ruvnet` identity check.
