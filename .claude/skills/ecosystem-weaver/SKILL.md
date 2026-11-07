---
name: ecosystem-weaver
version: 0.1.0
description: Self-improving /loop runbook that weaves the ruvnet package fabric. Each tick ingests lane reports into the weaver ledger, refreshes the package graph, repo scans, open PRs and release checks, ranks a one-lane-per-repo queue, and dispatches up to 8 lanes that integrate, fix, land, create and publish packages behind hard gates. Use for ecosystem cross-integration, release-regression repair, open-PR triage, and autopilot package creation across ruvnet repos.
category: ecosystem
tags: [weaver, loop, ecosystem, npm, integration, release, autopilot]
requires:
  - cargo (builds crates/ruvector-weaver)
  - gh (authenticated)
  - npm, node
  - gcloud (publish tokens only, at publish time)
---

# Ecosystem Weaver

`weaver` (crate `crates/ruvector-weaver`, ADR-355) is the deterministic half
of the loop. It produces `queue.json`. LLM lanes consume it and report back
through `weaver ledger record`. The ledger is the loop's memory: shipped,
rejected, queued and in-flight pairs are never re-proposed, and hit rates
per signal type reweight the next ranking.

## Setup (once per machine)

```bash
cd /home/ruvultra/projects/ruvector            # or a dedicated worktree
git worktree add --detach ../wt-weaver-loop origin/main
cd ../wt-weaver-loop
cargo build --release -p ruvector-weaver
export W="$PWD/target/release/weaver --config-dir $PWD/config/weaver"
# work dir defaults to /data/scratch/weaver (graph.json, scan/, prs/, gates/, repos/, queue.json)
# LIVE ledger: /data/scratch/weaver/ledger.jsonl (override with WEAVER_LEDGER)
```

`--config-dir` supplies `weaver.toml` and `capabilities.toml`. The live
ledger sits outside the repo, so recording an outcome never needs a PR. On
first use it is seeded from the in-repo snapshot
`config/weaver/ledger.jsonl`. To refresh that snapshot occasionally, run
` ledger snapshot` and open one small PR. To apply entries collected
elsewhere, run ` ledger import <file>`; entries that are already present
are skipped.

## The tick (run in this order)

1. **Ingest reports.** For every lane that finished since the last tick,
   run the `weaver ledger record ...` command its brief asked it to return
   (outcome `shipped|rejected|failed`, PR URL, `pkg@version`, one-line reason,
   `--measure metric=value:unit@baseline` for every measured delta). A lane
   that ended without a report is recorded `failed` with the reason it
   stopped. A research PR closed as REJECT is recorded `rejected` with its
   verdict BEFORE it is closed.
2. **Refresh signals.**
   ```bash
   $W graph                      # npm maintainer:ruvnet + crates.io owner; 12 h HTTP cache
   $W release                    # repo change / surface drop / open ranges per package
   $W prs --all                  # open PRs per repo (add --checks for CI rollups)
   $W resolve --top 15           # fresh installs + npm ls --all: invalid_resolution
   $W scan <repos touched since last tick>   # daily: $W scan --all
   ```
3. **Rank.** `$W queue --gate-top 20` writes `queue.json`: `lanes` (≤ 8, one
   per repo), `items` (each with a self-contained `brief`), `suppressed`
   (with the reason: ledger state, gate rejection, open PR already covering
   it, PR no longer open), `busy_repos`.
0. **Run `direct` items first.** `queue.direct` lists items the tick runs
   itself, with no agent and no lane. Today the only kind is
   `infra:cancel-superseded`: its brief holds the exact `gh run cancel` loop,
   which only touches runs on loop-owned branches.
4. **Claim, then dispatch.** For each lane in `queue.lanes`:
   - record every item in the lane as `in_progress` first
     (`$W ledger record --repo R --candidate C --subject S --kind K --signal SK --outcome in_progress`)
     so the repo stays busy and no later tick double-dispatches it;
   - spawn ONE background agent per lane (Agent tool, `run_in_background: true`,
     `isolation: "worktree"` where the lane edits a repo), all lanes in one
     message, with the concatenated `brief` of the lane's `active` + `batched`
     items. Batched items are independent (distinct candidate, no shared
     evidence file) and ship in the same PR / release.
5. **Persist and re-arm.** The live ledger is already durable (appended +
   fsynced outside the repo); snapshot it into the repo only occasionally,
   never once per tick (each push would restart CI). Store the tick summary
   (lanes dispatched, items suppressed, top-15) in memory
   (`memory_store` namespace `weaver`); re-arm with Monitor on the lane
   agents plus a ScheduleWakeup safety net (1200-1800 s). Do not poll.

## Hard rules every lane follows

- One active lane per repo; never two agents in one repo.
- **Branch names.** Lanes push to `weaver/<lane>-<topic>` (or `feat/weaver-*`).
  Those are the only branches whose superseded CI runs weaver ever cancels
  (`[loop] owned_branch_prefixes`).
- **Actions saturation.** `weaver prs` also records each repo's queued and
  in-progress run counts. A repo over `max_active_runs_per_repo` (default 20)
  is listed in `busy_repos` as `gha-saturated` and gets no lane that tick.
- **Blocked rejections.** Record a rejection that waits on other work with
  `--outcome rejected --blocked-on <subject>` (repeatable). The queue
  re-opens it automatically once every blocker's latest entry is `shipped`.
- **A red main is a baseline, not a blocker.** `weaver prs` records the
  workflows already failing on each repo's default branch. Each brief lists
  them, and the repo gets its own `main_red` hygiene item. Compare your PR
  against that baseline; do not stall on failures that were already there.
  Hanging (never-finishing) jobs are not detected; record those as ledger items.
- Never re-implement an open PR (the queue already suppresses those; if a
  lane finds one, stop and report `rejected: covered by #N`).
- **Numbers only from the ledger.** No numeric performance claim in a
  README, PR body or changelog unless it was measured in that lane and
  recorded with `--measure`. `weaver readme <pkg>` renders the measured
  block; `weaver readme --lint README.md` must pass.
- **Autopilot creation** (user decision 2026-11-05): a `create` item may
  create and publish a NEW npm name with no human approval, only when every
  gate passes: gap evidence with file:line in a consumer, no near-duplicate
  in the fabric, real implementation + passing tests
  (`$W gate <name> --new --dir <pkg> --run-tests`), non-empty tarball with a
  working entrypoint and a clean leak scan, smoke import from the registry
  after publish (`$W gate <name> --consumer <repo>`), README via
  `weaver readme`, and the same lane wires the package into the consumer
  (no orphans). Caps per tick/day are optional knobs in `weaver.toml`
  (unset = unlimited).
- **Gate scope.** Pass `--consumer-package <pkg>` to check a cycle from the
  exact package that takes the dependency; without it, siblings published
  from the candidate's own repo are ignored. Paths a package ships on purpose
  go in `[gate.leak_allow]`. Leak paths the previous version already carried
  are reported, and only NEW ones fail.
- **Release gate.** Before any publish of an existing name:
  `$W gate <pkg> --dir <pkg dir>`. It refuses (exit 2) on repository change,
  sharp file/size/export drop, or open-ended ecosystem ranges, unless the
  ledger holds `--outcome override --subject release:<check>:<pkg>` with a reason.
- **Evidence before filing.** Before reporting "X@latest is vulnerable",
  run `$W vuln-evidence X --dir <consumer> --affected "<range>"`. It resolves
  every installed copy (`npm ls --all`) and exits 2 when none falls inside
  the affected range. A nested stale copy is a dependency bump in the parent
  package, not an advisory against latest.
- **Disclosure.** First-party critical/high security findings go to a
  private GitHub security advisory, never a public PR or issue. Upstream
  advisories (npm audit / RustSec) are already public and ship as a bump PR.
- **Tokens.** `gcloud secrets versions access latest --secret=NPM_TOKEN|CARGO_REGISTRY_TOKEN --project=cognitum-20260110 --account=ruv@ruv.net`
  only at publish time, piped into the publish command; never printed,
  logged, or written to disk. `npm unpublish` is not available: gate first.
- **Scratch directories.** Each lane creates its own temp dir with
  `mktemp -d` (under its scratchpad or `/data/scratch`) and deletes ONLY that
  dir, by its exact path, at the end. Never `rm -rf` a shared parent such as
  `/data/scratch`, `/tmp/*` or another lane's worktree, and never use glob or
  age-based sweeps. Two over-broad cleanups on 2026-11-05 removed other
  lanes' work.
- Explicit `git add <paths>` only. Commits end with
  `Co-Authored-By: claude-flow <ruv@ruv.net>`.

## Reading the queue

```bash
jq '.lanes' /data/scratch/weaver/queue.json
jq -r '.items[] | select(.status=="active") | .brief' /data/scratch/weaver/queue.json
jq '.suppressed | group_by(.reason | split(":")[0]) | map({reason: .[0].reason | split(":")[0], n: length})' /data/scratch/weaver/queue.json
$W ledger show            # history + hit rates + busy repos
```

Signal types and work kinds: `missing_package` (404 import) -> replace, or
create when no capability covers it; `undeclared_import` / `unused_declared`
-> hygiene; `stub_marker` -> replace; `security` -> security; `open_pr` ->
land / close_with_ledger; `release_regression` -> release_fix. A rejection
whose reason is a heavy dependency tree is re-queued automatically once the
candidate ships a release under the size/dep bar in `weaver.toml [gate]`.
`weaver resolve` installs the top packages fresh and runs `npm ls --all`.
It emits `invalid_resolution` for every dependency whose resolved version
breaks a declared range, and the lane goes to the consumer whose range won
hoisting (for example, metaharness `^2.5.6` beat ruvector `^4.1.0` for
`@ruvector/ruvllm`).
`weaver release` also emits `fix_unreachable` (a dependent's declared range
excludes the version carrying a shipped security fix in the ledger; queue a
bump) and `zero_caret` (a `^0.y` / `~0.y` range that already misses the
dep's current release, because 0.x carets act as pins).
