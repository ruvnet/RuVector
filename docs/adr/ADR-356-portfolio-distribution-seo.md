# ADR-356: Portfolio distribution and SEO standard for the ruvnet npm packages

- **Status**: Accepted
- **Date**: 2026-11-08
- **Deciders**: ruv (owner), RuVector Architecture Team
- **Related**: ADR-355 (ruvector-weaver); skill `.claude/skills/ecosystem-weaver/SKILL.md`
- **Tags**: ecosystem, npm, seo, distribution, metadata, release-safety
- **Source plan**: owner distribution plan, pinned snapshot
  `ruvnet/ruvnet@5f46f8cf7b8db17a28f602be3eaceae99cd70d08`
  (`data/snapshots/2026-11-08/npm-inventory.json`, 405 package IDs, and
  `data/snapshots/2026-11-08/npm-downloads.json`)

## Context

ruvnet publishes 405 npm packages (the verified inventory of 2026-11-08).
Distribution is very uneven: about 40 packages pass 100k downloads/month and
about 235 sit under 1k. Many have not been published in over 180 days. The
metadata is inconsistent. Homepages point at GitHub READMEs or other sites.
`agentdb` carries 64 keywords and `agentic-flow` 67, which is keyword
stuffing; npm search ranks on quality, popularity and maintenance, and does
not reward it. Most READMEs never say what works with what.

The owner asked to use the popular packages to give the less popular ones
reach, to fix the old packages' README and topic SEO, and to point every
homepage at `https://ruv.io`. The obvious shortcut is to add the small
packages as dependencies of the large ones. This ADR rejects that shortcut
for anything the code does not use at runtime.

## Decision

### 1. The metadata standard

Every npm manifest, including generated platform manifests (set at the
source template, never hand-published):

| Field | Rule |
|---|---|
| `homepage` | `https://ruv.io`, the root. ruv.io is a catch-all SPA (every path returns 200 with the same title), so per-package paths are not used until a route proves distinct per-package content. |
| `repository` | Canonical source: `{type: git, url: git+https://github.com/<owner>/<repo>.git, directory: <path>}`. `directory` is required for monorepo packages. Never ruv.io. |
| `bugs` | `https://github.com/<owner>/<repo>/issues`. |
| `keywords` | 8 to 20 (proposals aim for 16), every one from the taxonomy (below) or the package's own name. |
| `description` | At most 200 characters, containing the package's primary topic phrase. |

The GitHub **Website** of each owning public repo is `https://ruv.io`.
Repo-level GitHub **topics** come from the taxonomy, at most 20, and a
proposal replaces the list rather than appending to it.

### 2. The README contract

Each package README must have:

1. **One job in the first paragraph.** What the package does, in one sentence.
2. **Supported environments.** Node versions, OS/architecture, and native or
   WASM fallback, stated only where true.
3. **A tested quickstart with expected output.** Every documented command is
   run in a clean `mktemp -d` install. The real captured output is pasted in
   as "Expected output". Commands that cannot be run are not documented as
   a quickstart.
4. **One meaningful link to the next package.** A real next step, not a list.
5. **"Works with / Used by"**, generated from the real dependency graph:
   runtime dependencies inside the fabric, and published packages that depend
   on it. A related-by-topic link is never presented as a dependency.
6. **A footer linking https://ruv.io.**
7. **For flagship and entry-point READMEs, one small animated SVG.** It uses
   SMIL or CSS animation, has no external fonts or scripts and no `@import`,
   and stays under 20 KB. It is committed to the repo and referenced by its
   `raw.githubusercontent.com` URL so it also renders on npmjs.com. It must
   show the package's real flow (for example query → HNSW index → top-k
   results), not decoration. Any number inside it comes from a ledger
   measurement. `weaver readme --lint` enforces the ADR-355 rule that numeric
   claims appear only in the ledger-rendered block.

Historic usage notes are preserved. Generated blocks sit between
`<!-- weaver:seo:* -->` markers, so a re-run replaces them in place.

### 3. Topic taxonomy, measured rather than guessed

`config/weaver/topics.toml` is the canonical taxonomy. A topic is admitted
only with recorded demand: the npm keyword search total
(`registry.npmjs.org/-/v1/search?text=keywords:<t>`), the top-5 results'
monthly downloads, and the GitHub topic repo count
(`search/repositories?q=topic:<t>`). The floor is 50 npm packages **or** 500
GitHub repos, and ruvnet packages must genuinely ship the capability.
Measurements are recorded in the ledger (`seo:topic:<slug>`) and refreshed
with `weaver seo demand --taxonomy --record`. "Trending" is never claimed
without these numbers.

Two kinds of terms are exempt from the floor:

- **Brand terms** such as `ruvector` and `agentdb`. They are legal
  keywords but never primary topics.
- **Niche terms**, long-tail names of a verified feature. Each pilot search
  theme is one. They are measured and recorded, legal as keywords, never
  primary.

Several suggested themes measured below the floor on 2026-11-08 and are
**not** promoted as topics: `swe-bench` (35 npm / 400 GitHub),
`quantum-simulation` (4 / 415), `link-prediction` (2 / 411) and
`graph-embeddings` (0 / 91). The last two are kept as niche terms because the
`@ruvector/kge` pilot validates that theme.

### 4. Dependency edges and the three pilots

Visibility without a dependency comes from metadata, the "Works with / Used
by" cross-links, shared topics and the homepage. A dependency edge is added
only where code really uses it at runtime, with a test. Two rules apply:

- **A required dependency must be exercised by the default workflow.**
- **`optionalDependencies` is not opt-in.** npm installs optional dependencies
  by default. A truly opt-in feature is a separate recipe, an adapter or an
  explicit install.

Before any edge is added, check for cycles, duplicate packages, version
ranges, native-binary fallback and license compatibility.

| Pilot | Theme | Lane | Disposition |
|---|---|---|---|
| `@ruvector/kge` | graph embeddings and link prediction | RuVector | A runnable link-prediction demo, reached from the flagship as a recipe (`npx`/explicit install). It is **not** a required dependency of `ruvector`, because the default `ruvector` workflow does not exercise it. |
| `@ruvector/rvf-mcp-server` | local vector search via MCP | RuVector (server); Ruflo (recipe) | The `@ruvector/rvf` edge already exists. The work is a tested MCP quickstart. The Ruflo-side recipe belongs to the ruflo lane. |
| `@metaharness/host-github-actions` | agent harness evaluation in CI | MetaHarness lane | Handed off. This lane does not edit the metaharness repo. |

**Acceptance gates for each pilot:**

- First check whether an equivalent integration already exists.
- Measure cold install size and median install time over 5 fresh installs
  against the pinned prior version. A regression above **5%** fails the pilot.
- Inspect the lockfile for unexpected packages.
- Run `npm audit`. Critical and high findings go through a private advisory.
- Pack the release with `npm pack`, then read back the published version,
  homepage, keywords and README from the registry.
- After a clean install, the exact documented command must run.

### 5. Two separate measures

- **Reach** counts npm download events, search impressions and guide
  visits. Downloads are registry events, not users. October 2026 has known
  reporting gaps.
- **Adoption** counts unique external repositories that explicitly declare
  or import a package, or that voluntarily report a successful quickstart.
  Transitive lockfile entries and repeated CI installs do not count. The
  30-day pilot target is 10 external adopters across the three pilots, and
  evidence coverage is logged.

No default telemetry is added. Reach is never reported as adoption.

### 6. Aliases and migration

A suspected alias (for example `ruvector-attention-wasm` or `ruvector-sona`)
gets a **migration audit**. The audit checks who installs the package, what
it re-exports, and whether a supported replacement covers the same API.
**A package is never deprecated on downloads or name alone.** A deprecation
candidate must also fail the registry install/import gate **and** have no
published consumer. The owner then runs the deprecation with 2FA. Low
downloads do not override maintenance or compatibility needs.

### 7. Rollback

If a new integration regresses install behaviour (size, time or a broken
native fallback), the edge is removed in a patch release and the discovery
path is kept: README links, topics and keywords. Repo-level metadata
changes record the old homepage and topics in the ledger before the edit,
including the exact `gh repo edit` command that reverts them.

### 8. Implementation (`weaver seo`)

| Command | Role |
|---|---|
| `seo demand [--taxonomy] [--record]` | Measures topic demand and records it in the ledger. |
| `seo audit --all [--smoke-stale]` | Reports homepage ≠ ruv.io; keyword count outside 8–20 or keywords off the taxonomy; description missing, too long or without its phrase; missing repository, bugs or `directory`; README missing install, Works with / Used by, or the ruv.io link; repo topics missing, over 20 or off the taxonomy; stale over 180 days; and import-gate status (from ADR-355 gate reports, otherwise `unknown`). Ranked by monthly downloads × issue count. Writes `seo-audit.json`. |
| `seo baseline --inventory <405.json> --plan <plan.md>` | Builds one row per inventory ID: version, homepage, keywords, README present, dependency edges, unpacked size, deprecated, downloads (from api.npmjs.org), maintainer, canonical repo and directory from the published manifest, the owner's `class_hint` checked against the manifest, and a disposition. Unknowns are the literal `"unknown"`. Graph packages outside the inventory are listed as reconcile items. |
| `seo propose <pkg> \| --repo r \| --all` | Prints the exact diff for package.json (format-preserving), the README blocks, the flow SVG and the repo topic list. |
| `seo apply --repo r [--repo-level] [--packages …\|--top N]` | Repo metadata goes out through `gh repo edit` and is read back. Package changes become a branch and PR with a patch bump. **`apply` never publishes.** Publishing goes through the repo's own release path and the ADR-355 release gates. Repos and manifests owned by other lanes (`hands_off_*` in topics.toml) are refused. |

### 9. Rollout

1. Baseline all 405 packages and reconcile new ones every wave.
2. Entry points and pilots go first.
3. The rest is grouped by verified owning repo and shared release template,
   in reviewable batches of roughly 20 to 40 manifests.
4. Platform packages get the standard from the source template and link to
   the parent quickstart.
5. Each adapter or example gets a runnable path from its flagship.
6. Each feature candidate gets one concrete integration or a documented
   no-fit disposition.
7. Publishes are staggered at no more than about 10 per hour.

**Portfolio acceptance:**

- All 405 IDs are accounted for, each with a verified disposition and a
  homepage readback, or a recorded blocker.
- Every promoted workflow has a passing clean install.
- There are no dependency cycles and no unused required edges.

Not every package must become a dependency.

## Consequences

- Reach work cannot inflate the dependency graph. Every edge is real and
  tested, so install size and supply-chain surface grow only with use.
- Keyword lists shrink from 60+ to about 16 high-signal terms with recorded
  demand.
- Metadata changes to a package take effect on npm only after a republish,
  because npm renders the tarball README. They therefore ride normal patch
  releases and are not pushed out as a burst.
- Lanes that own other repos (ruflo, metaharness) get proposals, not edits.

## Alternatives considered

- **Add small packages as dependencies of popular ones.** Rejected. It adds
  install weight and supply-chain risk to every downstream user, and it
  fails the "exercised by the default workflow" rule.
- **Per-package ruv.io paths.** Rejected for now. The site is a catch-all
  SPA, so those paths would be soft 404s.
- **Keep keyword stuffing.** Rejected. It does not help npm ranking, and it
  makes the package look low quality.
