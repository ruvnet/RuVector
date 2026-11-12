# Ecosystem Weaver — agent runbook (Codex and other non-Claude agents)

This repo has no Codex-specific convention, so this file mirrors
`SKILL.md` in this directory for any agent that reads `AGENTS.md`.
Design and gates: `docs/adr/ADR-355-ruvector-weaver-ecosystem-fabric.md`.

## Build

```bash
cargo build --release -p ruvector-weaver
W="target/release/weaver --config-dir config/weaver"   # run from a ruvector checkout
```

## One tick

1. Ingest finished lane reports: run each `weaver ledger record ...` line a
   lane returned (`--outcome shipped|rejected|failed`, `--pr`, `--version`,
   `--reason`, `--measure metric=value:unit@baseline`). Lanes without a
   report are recorded `failed`.
2. Refresh: `$W graph`, `$W release`, `$W prs --all`, `$W resolve --top 15`, `$W scan <repos>`
   (`--all` daily).
3. Rank: `$W queue --gate-top 20` -> `/data/scratch/weaver/queue.json`.
3b. Run every id in `.direct` yourself first (agent-free; e.g. cancel superseded
   runs on loop-owned branches with the `gh run cancel` loop in its brief).
4. For each entry in `.lanes` (at most 8, one per repo): record its items as
   `in_progress`, then hand ONE worker the `brief` of every item in that lane.
5. The live ledger (`/data/scratch/weaver/ledger.jsonl`, or `$WEAVER_LEDGER`)
   is outside the repo and needs no commit. Use `weaver ledger snapshot` only
   occasionally, for a small seed PR. Schedule the next tick.

## Rules for a lane worker

- Work only in the lane's repo; one PR (or release) per lane.
- Push to a `weaver/<lane>-<topic>` branch; record waiting rejections with
  `--blocked-on <subject>` so they re-open automatically.
- Stop if `weaver gate <candidate> --consumer <repo>` fails; report `rejected`
  with the failing gate.
- No numeric performance claims unless measured and recorded with
  `--measure`; README numbers come only from `weaver readme`'s measured block,
  and `weaver readme --lint <README.md>` must pass.
- New npm names (autopilot): every gate in `weaver gate <name> --new --dir <pkg> --run-tests`
  must pass before publish, the registry smoke gate after publish, and the
  same lane wires the package into the consumer.
- Before publishing an existing name: `weaver gate <pkg> --dir <pkg dir>`
  (refuses on repository change, file/size/export drop, open ranges unless
  the ledger has an `override` for `release:<check>:<pkg>`).
- First-party critical/high security findings: private advisory only.
- Before filing a vulnerability against "latest": `weaver vuln-evidence <pkg> --dir <consumer> --affected "<range>"`
  must show an affected resolved copy (nested stale copies are a parent bump).
- Publish tokens only via `gcloud secrets versions access latest --secret=NPM_TOKEN --project=cognitum-20260110 --account=ruv@ruv.net`
  at publish time; never print or store them.
- Workflows already red on main (listed under "Main baseline" in the brief) are
  pre-existing: do not block on them; only the `main_red` item fixes them.
- Scratch: create your own dir with `mktemp -d` and delete only that exact path;
  never sweep shared dirs (`/data/scratch`, `/tmp/*`) or other worktrees.
- Explicit `git add <paths>`; commit trailer `Co-Authored-By: claude-flow <ruv@ruv.net>`.
- Report back with the exact `weaver ledger record` command from the brief.
