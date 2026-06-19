---
adr: 257
title: "Extract ruqu and rvdna into standalone ruvnet repos referenced as git submodules"
status: proposed
date: 2026-06-17
authors: [ruvnet, claude-flow]
related: [ADR-025, ADR-071, ADR-QE-002]
tags: [monorepo, submodules, ruqu, rvdna, quantum, genomics, cargo-workspace, npm-workspace, packaging, extraction]
---

# ADR-257 — Extract `ruqu` and `rvdna` into standalone repos (git submodules)

> **Decision in one line.** Split `ruqu` (both clusters) and `rvdna` out of the
> ruvector monorepo into standalone `ruvnet/ruqu` and `ruvnet/rvdna` repos, and
> re-reference them as **git submodules** mounted at consolidated paths. This is
> feasible and keeps the superproject building. `rvdna` and the `ruQu` min-cut
> crate depend on `ruvector-*` crates, but **all of those are published on
> crates.io** (the full closure is at `2.2.3` as of 2026-06-17 — see Update). So
> the standalone repos **can build independently** once the extraction rewrites
> their intra-workspace `path` deps to crates.io `version` deps — the one
> required migration step for standalone buildability.

## Context

Request: publish `ruqu` and `rvdna` as standalone `ruvnet` repos and reference
them back here via submodules. A full reference sweep (every `Cargo.toml`,
workflow, script, npm manifest, doc, and `use` site) produced the coupling map
below. The naming is ambiguous and the coupling is asymmetric, so the decision
hinges on facts, not the names.

### What "ruqu" and "rvdna" actually are (evidence H)

There are **two distinct `ruqu` projects**, plus `rvdna`:

| Unit | Path(s) | Package name | Nature | Outbound deps |
|---|---|---|---|---|
| **Quantum-sim cluster** | `crates/ruqu-core`, `crates/ruqu-algorithms`, `crates/ruqu-exotic`, `crates/ruqu-wasm`, `npm/packages/ruqu-wasm` | `ruqu-core` (2.0.5) etc. | Pure-Rust quantum circuit simulator | **Self-contained** — only depend on each other |
| **min-cut `ruQu`** | `crates/ruQu` | `ruqu` (2.2.3) | "Classical nervous system for quantum machines" (dynamic min-cut) | `ruvector-mincut`, `ruvector-mincut-gated-transformer`, `cognitum-gate-tilezero` (all **optional**) |
| **rvdna** | `examples/dna` (Rust crate `rvdna` 0.3.0) + `npm/packages/rvdna` (`@ruvector/rvdna` 0.3.0) | `rvdna` | Quantum/AI genomics engine + `.rvdna` format | **9 `ruvector-*` crates** via path deps (core, attention, gnn, graph, dag, math, filter, collections, solver) |

### Coupling that constrains the design (evidence H)

1. **Outbound (now resolved).** `rvdna` (`examples/dna`) path-depends on nine
   `ruvector-*` crates; `ruQu` path-depends on `ruvector-mincut` (+1). These deps
   are declared in dual `{ version = "…", path = "…" }` form. **All of the
   `ruvector-*` deps are published on crates.io** (the rvdna closure —
   core/attention/gnn/graph/dag/math/filter/collections/solver + transitive
   cluster/raft/replication — is at `2.2.3`; see Update). So a standalone repo
   builds in isolation **once its `path =` keys are dropped** (leaving the
   crates.io `version`), which the migration must do. (An earlier draft claimed
   these were unpublished; that was a crates.io **API rate-limit** misread —
   corrected here.)
2. **Inbound.** `examples/OSpipe/Cargo.toml` and `examples/rvf/Cargo.toml`
   path-depend on `ruqu-core`/`ruqu-algorithms`. These stay in ruvector and must
   keep resolving the extracted crates.
3. **Code spans two trees.** ruqu lives under both `crates/` and
   `npm/packages/`; rvdna under both `examples/dna` and `npm/packages/rvdna`. A
   single submodule mounts at exactly one path, so one repo cannot be restored to
   multiple scattered original paths simultaneously.
4. **npm wrappers cross the boundary.** `npm/packages/rvdna` builds via
   `napi … --cargo-cwd ../../../examples/dna`; `npm/packages/ruqu-wasm` wraps
   `crates/ruqu-wasm`. The npm package and its Rust crate must live in the same
   standalone repo, or the build path breaks.
5. **CI.** `.github/workflows/ci.yml` has a dedicated `ruqu-quantum` nextest
   shard (`-p ruqu -p ruqu-algorithms -p ruqu-core -p ruqu-exotic -p ruqu-wasm`)
   and excludes those from the catch-all shard.
6. **Dead cross-repo links.** `examples/dna/adr/ADR-002` and `ADR-010` link to
   `../../crates/ruQu/docs/adr/...` — broken once the two land in different repos.

## Decision

Per the chosen scope (**both ruqu clusters**) and mechanism (**git submodules**):

1. **Create two standalone repos:**
   - **`ruvnet/ruqu`** — contains the quantum-sim cluster (`ruqu-core`,
     `ruqu-algorithms`, `ruqu-exotic`, `ruqu-wasm`), the min-cut `ruqu` crate
     (from `crates/ruQu`), and the `npm/packages/ruqu-wasm` wrapper, preserving an
     internal `crates/…` + `npm/…` layout.
   - **`ruvnet/rvdna`** — contains the `rvdna` Rust crate (from `examples/dna`)
     and the `npm/packages/rvdna` wrapper.
   Extract **with history** via `git filter-repo --path …` per repo.

2. **Mount each repo as ONE submodule at a single consolidated path**, not at the
   original scattered paths (impossible for one repo → many paths):
   - `ruvnet/ruqu` → `external/ruqu/`
   - `ruvnet/rvdna` → `external/rvdna/`
   Rewrite the superproject references to the new paths (workspace `members`,
   the `OSpipe`/`rvf` path deps, the npm `workspaces` glob, CI shard paths).

3. **Make the submodules build standalone** by rewriting their intra-workspace
   `ruvector-*`/`ruqu-*` deps from `{ version, path }` to **version-only**
   (crates.io). Since the full closure is published at `2.2.3` (and rvdna pins
   compatible `^2.0`), the extracted repos resolve their deps from crates.io and
   `cargo build` on their own. Inside the superproject the pinned submodule
   commit still builds reproducibly.

4. **Accept the standalone-build limitation explicitly.** `ruvnet/rvdna` and the
   `ruvnet/ruqu` min-cut crate will not `cargo build` on their own until their
   `ruvector-*` dependencies are published to crates.io (tracked as follow-up,
   out of scope here). The quantum-sim cluster in `ruvnet/ruqu` **does** build
   standalone today. This trade-off is the direct consequence of choosing
   submodules over publishing-to-registry first; it is recorded, not hidden.

5. **Fix the breakage enumerated above** as part of the cutover: workspace
   members, `OSpipe`/`rvf` path deps → `external/ruqu/…`, npm workspace glob, CI
   `ruqu-quantum` shard paths, `package.json` `repository.directory`/`homepage`,
   the `--cargo-cwd` napi path, the two dead cross-repo ADR links, and regenerate
   `Cargo.lock` + `npm/package-lock.json`.

6. **Gate the irreversible step.** Creating the public repos and pushing filtered
   history is outward-facing and hard to undo. The migration is encoded as an
   idempotent script (`scripts/extract-ruqu-rvdna-submodules.sh`) reviewed via
   this ADR/PR **before** it is run against GitHub.

## Consequences

### Positive
- `ruqu` and `rvdna` get their own issue trackers, release cadence, and stars;
  the monorepo shrinks and its CI matrix simplifies.
- The quantum-sim `ruqu` repo is immediately a clean, independently-buildable
  open-source artifact.
- Submodules pin an exact commit, so the superproject's build is reproducible.

### Negative
- **Submodule friction**: contributors must `git clone --recursive` (or
  `git submodule update --init`); a plain clone yields empty `external/…` dirs
  and a failing workspace. CI must add a submodule-checkout step.
- **Dep deduplication**: the extracted repos pull `ruvector-*` from crates.io;
  in-workspace edits to those crates won't reach a submodule pinned to a
  published version until a new version is cut. (Acceptable — they are released
  deps now, not local path deps.)
- **Path churn**: moving from `crates/ruqu-*` to `external/ruqu/crates/ruqu-*`
  touches many references (workspace, CI, docs, two example crates).
- Two-step commits across repos (change submodule, then bump the pointer here)
  add overhead for cross-cutting edits.

### Neutral / Alternatives considered
- **Publish to crates.io/npm and depend by version** (no submodule) — the
  idiomatic Rust/npm split; avoids submodule friction and yields truly standalone
  repos, but requires publishing the ruvector dependency crates first. Rejected
  here only because submodules were explicitly requested; recommended as the
  eventual end-state and the prerequisite for standalone builds.
- **One repo per crate** (keep original paths, many submodules) — preserves
  paths but fragments `ruqu` into 5+ repos, contradicting "one ruqu repo."
- **`git subtree`** instead of submodule — vendors a copy with no pinned link;
  loses the "single source of truth" the request implies.

## Migration procedure (executed by `scripts/extract-ruqu-rvdna-submodules.sh`)

1. **Pre-flight**: clean tree; confirm `git-filter-repo` available; confirm no
   `ruvnet/ruqu` / `ruvnet/rvdna` exist yet.
2. **Extract with history** into temp clones via
   `git filter-repo --path crates/ruqu-core --path … --path npm/packages/ruqu-wasm`
   (and the rvdna paths), rewriting to the target internal layout.
3. **Create repos** `gh repo create ruvnet/ruqu --public` / `…/rvdna --public`;
   push the filtered histories.
4. **In a branch of this repo**: `git rm -r` the original dirs; `git submodule add`
   each new repo at `external/ruqu` / `external/rvdna`; apply the reference
   rewrites (§Decision.5); `cargo metadata` to confirm the workspace resolves;
   `npm install` to regenerate the lockfile.
5. **Verify**: `cargo check -p ruqu-core` (and the two consumers `ospipe`,
   `rvf-examples`) resolve through the submodule; `npm run build` for the two
   wrappers; CI submodule-checkout step added.
6. **Open PR** bumping the submodule pointers.

## Rollback
- Before repos exist: delete the branch — fully reversible.
- After repos exist but pre-merge: `git submodule deinit`, restore the dirs from
  history, delete the (empty-of-external-consumers) repos. The filtered repos can
  be archived rather than deleted.
- Post-merge: revert the cutover commit and re-vendor from the submodule commit.

## Update (2026-06-17) — dependency closure published to crates.io

The `rvdna` Rust-crate dependency closure was synced to **`2.2.3`** on crates.io,
removing the only real blocker to standalone builds:

| Status | Crates |
|---|---|
| Published this change | `ruvector-collections`, `-filter`, `-math`, `-dag`, `-cluster`, `-raft`, `-replication`, `-gnn`, `-attention` (all `2.2.3`) |
| Already at 2.2.3 | `ruvector-solver`, `ruvector-core`, `ruvector-graph` |

All 12 crates in the closure are now at `2.2.3` (verified via the crates.io
sparse index). Each was published **with its existing README** (cargo bundles
`README.md` automatically). Remaining for standalone `rvdna`/`ruQu` builds: the
mechanical `path =`→`version` rewrite in the extracted repos (migration step).

## Links
- Reference sweep: root `Cargo.toml` (members 97, 109–112, 118),
  `examples/OSpipe/Cargo.toml:39`, `examples/rvf/Cargo.toml:27-28`,
  `.github/workflows/ci.yml:135-141,264-268`,
  `npm/packages/rvdna/package.json` (`build:napi --cargo-cwd ../../../examples/dna`),
  `npm/packages/ruqu-wasm/package.json:34,36`, `examples/dna/Cargo.toml:16-53`
  (9 ruvector path deps), `crates/ruQu/Cargo.toml:16,23`.
- Prior art: ADR-025 (exo-AI multiparadigm), ADR-071 (npx ecosystem gap),
  ADR-QE-002 (quantum-engine crate structure).
