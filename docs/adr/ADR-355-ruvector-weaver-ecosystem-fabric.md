# ADR-355: ruvector-weaver, a self-improving harness for the ruvnet package fabric

- **Status**: Accepted
- **Date**: 2026-11-05
- **Deciders**: ruv (owner), RuVector Architecture Team
- **Related**: ADR-026 (model routing), ADR-354; skill `.claude/skills/ecosystem-weaver/SKILL.md`
- **Tags**: ecosystem, npm, integration, release-safety, autopilot, loop

## Context

ruvnet maintains about 400 npm packages and about 500 crates across roughly
80 repositories. Cross-integration has been manual and lossy. During one
session the following were found by hand: consumers probing npm names that
404 (`@ruvector/mincut`, `@ruvector/ast|diff|coverage`), real packages imported
but never declared, fallbacks reimplementing capabilities that already exist
as packages (mincut, PageRank, LoRA), open-ended `>=` ranges, hundreds of open
PRs from automated routines, and a release regression: `@ruvector/ruvllm`
4.0.0/4.0.1 were published from `ruvnet/ruvllm` over the 3.x line from
`ruvnet/RuVector`. A 137-file runtime was replaced by a 16-file harness that
has nothing to do with it, and `ruvector`'s `>=2.0.0` range picked it up.

The goal is a standing loop that finds this work, ranks it, hands it to agents,
and learns from the outcomes. It must not spam the registry or fabricate
claims.

## Decision

Split the loop into a **deterministic Rust core** and **LLM lanes**.

`crates/ruvector-weaver` (binary `weaver`, `publish = false`) owns everything
that can be computed without judgement. All artifacts go under a work dir
(default `/data/scratch/weaver`). Config and the ledger live in
`config/weaver/`.

| Command | Output | Role |
|---|---|---|
| `graph` | `graph.json` | npm `maintainer:ruvnet` (paginated) + crates.io owner; version, last publish, monthly downloads (`api.npmjs.org` bulk/per-scope with 429 backoff via `--fresh-downloads`), runtime/optional/peer edges, repository, unpacked size, runtime dep count, open ranges; consumer-repo set. Platform NAPI binaries stay as edges and are excluded from ranking. |
| `scan <repo>` | `scan/<repo>.json` | Shallow clone into the cache. Signals with file:line evidence: 404 imports, undeclared and unused ecosystem deps (nearest `package.json`), stub/TODO/simulated/fallback markers near domain keywords mapped through `capabilities.toml`, npm-name collisions, open ranges, and a security lane (`npm audit`, `cargo deny`). |
| `prs <repo>` | `prs/<repo>.json` | Open PRs classified as `land` (fix/security/perf/deps/release), `close_with_ledger` (research with a REJECT/negative verdict) or other. |
| `release` | `scan/_release.json` | Per package: repository changed, file count / unpacked size / export surface drop vs the previous version and the last release of the previous major (rename-aware via GitHub redirects). |
| `gate <pkg>` | `gates/<pkg>.json` | Candidate health: recent publish, non-empty tarball with real entrypoints, tarball leak scan (`.claude*`, `.claude-flow/`, `.swarm/`, `.db`, `.env*`, `.npmrc`, keys), dependency cycle against the consumer, smoke import in a temp npm project (exports become the "verified API"), release checks. `--dir` runs the pre-publish checks; `--new --dir` adds the source gates for new names. Exits 2 when it rejects. |
| `queue` | `queue.json` | Ranks by `ln(10 + consumer downloads) × signal confidence × occurrence boost × candidate health × signal-type weight`. Dedupes against the ledger and open PRs, takes one lane per repo (at most 8) with batching of independent items, and attaches a self-contained brief per item. |
| `ledger record\|show\|import\|snapshot` | live `<work>/ledger.jsonl` | Append-only outcomes outside the repo. `config/weaver/ledger.jsonl` is the seed snapshot. |
| `readme <pkg>` | README + banner | SEO README skeleton plus an animated SVG banner in the `assets/ruvector/packages/*.svg` convention, and `--lint`. |
| `propose-new` | `create` items in `queue.json` | Capability gaps (a 404 import that no capability covers) become create work. |

**Self-improvement.**
- Ledger state per (repo, subject or candidate) decides suppression: shipped, rejected, queued or in-progress work is never re-proposed. A pair stops being proposed after `max_failures`.
- The weight for each signal type is its base weight × 2 × the Laplace-smoothed hit rate `(shipped + 1) / (shipped + rejected + failed + 2)`.
- A rejection whose reason is a heavy dependency tree is re-queued automatically once the candidate ships a newer release under the size/dep bar (`[gate] lean_max_runtime_deps`, `lean_max_unpacked_bytes`). The first case is sublinear-time-solver's "package diet". Its waiting consumers are agentdb's SparsificationService PPR fallback, claude-flow's `simplePersonalizedPageRank`, and ruvector extras.

**Lanes.** There is one active lane per repo, because two agents in one repo
collide on versions and lockfiles. Up to 8 lanes run concurrently. A lane may
batch independent items (distinct candidate, no shared evidence file, public
disclosure) into one PR or release. Integrations are not capped per tick. The
queue covers every scanned repo of the whole graph.

**Open PRs.** These are a signal source. Fix/security/deps PRs become `land`
items. Research PRs with a negative verdict become `close_with_ledger` items:
the result goes into the ledger, then the PR is closed. Any other item whose
candidate or subject an open PR already covers is suppressed, so lanes never
reimplement an open PR. The open-PR snapshot from 2026-11-05 is seeded in the
ledger as `queued`. Where live PR data exists it replaces the snapshot.

## The gates

1. **New npm names: autopilot behind hard gates** (user decision, 2026-11-05).
   The first design held every new package name in a human approval queue,
   because npm treats bulk name creation as spam and stat-gaming and the
   publish token cannot unpublish. The owner chose full autopilot instead.
   New names are limited only by quality gates, each recorded on the item or
   in the ledger:
   - (a) **Gap evidence.** At least one consumer has file:line evidence for
     the gap, and the same lane wires the new package into that consumer. No
     orphan packages.
   - (b) **No near-duplicate.** Neither the capability table nor a registry
     search of ruvnet packages already covers the gap. No renames.
   - (c) **Real implementation.** Not a stub and not a re-export only, the
     package's tests pass, the tarball is non-empty with a working
     entrypoint, the leak scan is clean, and a smoke import from the registry
     succeeds after publish.
   - (d) **README via `weaver readme`,** with numbers from the ledger only.
   - (e) **Optional rate limits.** `max_new_packages_per_tick` and
     `max_new_packages_per_day` default to unset (unlimited) and stay
     available as knobs.

   Version bumps of existing packages are allowed after `weaver gate`.
2. **Ledger-only numeric claims.** A README, banner, PR body or changelog may
   state a performance number only when it was measured in the lane and
   recorded with `ledger record --measure`. `weaver readme` renders those
   numbers inside a delimited measured block, drops numeric clauses from
   registry descriptions, and `--lint` fails on any unit-bearing or
   comparative number outside that block.
3. **Release-regression gate** (from the `@ruvector/ruvllm` incident).
   `weaver gate` refuses a publish when any of these trip, unless the ledger
   holds an explicit `override` entry for `release:<check>:<pkg>`:
   - the repository changed between consecutive versions or majors;
   - the file count, unpacked size or export surface dropped sharply;
   - the npm name is declared by two repos;
   - an ecosystem dependency has an open-ended range.

   These findings are high-priority queue items (weight 1.6).
4. **Disclosure hygiene, which is not an approval step.** First-party
   critical/high findings go to private GitHub security advisories, never to
   public PRs. Upstream advisories (npm audit, RustSec) are already public
   and ship as dependency bumps. Audit items of unknown severity are triaged
   first.

## Consequences

- Recurring discovery becomes a deterministic, tested binary (26 offline
  tests on fixture repos and fakes, no network in unit tests). The LLM budget
  goes to the work itself.
- The live ledger lives outside the repo (`/data/scratch/weaver/ledger.jsonl`).
  Recording outcomes therefore never needs a PR. A per-tick ledger push had
  restarted CI on the harness PR on every push. The in-repo copy is a seed
  snapshot, refreshed occasionally with `weaver ledger snapshot`.
- Rejections can name blockers (`blocked_on: [subjects]`). The queue
  re-opens them once every blocker is shipped, which replaces hand re-queueing.
- Before handing out lanes, weaver reads each repo's queued and in-progress
  Actions runs. Saturated repos (account-wide runner caps) are deferred.
  Superseded runs on loop-owned branches become direct `cancel-superseded`
  items, which never touch other branches.
- Security fixes must be reachable. `fix_unreachable` queues a bump
  wherever a dependent's declared range excludes the patched version. This
  came out of GHSA-mjwg-5fm3-fx2p: `/darwin` 0.13.0 was
  excluded by `^0.9.1`, `^0.12.0` and `~0.12.0`. `zero_caret` flags 0.x
  carets that already pin a dependent away from the dep's current release.
- Autopilot creation can still publish a weak package that passes every
  gate. That risk is accepted by the owner. It is mitigated by the
  no-duplicate and consumer-wiring gates and the post-publish smoke import,
  and deprecation (not unpublish) is the remedy.
- Heuristic detectors produce noise (stub markers especially). Signal-type
  reweighting from the ledger and the one-lane-per-repo cap keep that noise
  from turning into churn.

## Alternatives considered

- **Pure LLM discovery each tick.** Rejected: it is expensive,
  nondeterministic and keeps no memory.
- **Human approval queue for new names.** Implemented first, then replaced at
  the owner's request (gate 1).
- **Mirroring the ledger into an RVF store.** Deferred. JSONL keeps the PR
  small and diffable. An RVF mirror via the rvf crates is future work.

## Seeded state (2026-11-05)

- **Shipped:** ruvector 0.3.10/0.3.11 (`@ruvector/rvf` ^0.4.2).
- **In progress:**
  - ruvector graph-mincut → `@ruvector/mincut-wasm` (0.3.12);
  - claude-flow: learning-wasm, mincut, ast/diff/coverage, diskann;
  - agentic-flow: aidefence, agent-booster, router;
  - agentdb: mincut-wasm, graph-transformer;
  - 16 audit items in ruvector;
  - the `@ruvector/ruvllm` 4.1.0 restore and the `@ruvector/ruvllm-gpu-loop`
    rename;
  - the sublinear-time-solver package diet.
- **Rejected:**
  - midstreamer → agentic-flow (dependency cycle);
  - sublinear-time-solver in ruvector main deps (heavy tree; re-queues on a
    lean release).
- **Queued:**
  - 7 audit backlog items;
  - the ruvector `@ruvector/ruvllm >=2.0.0` range;
  - the open ruvector PRs classified land / close-with-ledger.
