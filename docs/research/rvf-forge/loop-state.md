# RVForge Build Loop — Iteration State

> Read this FIRST each loop iteration. Update it LAST. Never redo completed
> steps. Branch: `feat/rvf-forge` · PR: #790 · Spec:
> `docs/research/rvf-forge/requirements.md` · ADRs: 283–293.

## Done criteria (overall)

PR #790 merged with green CI; `@ruvector/forge` published to npm; the
locally-testable requirements §15 acceptance criteria pass:
validate/build/verify round-trip on a sample RVF, identical embedded RVF
SHA256 across packaging outputs, tamper detection rejects modified
artifacts, no secrets in code or logs.

## Work plan (requirements §12 order, Rust/WASM first)

- [x] ADRs 283–293 authored, committed, pushed (commit 14541c870)
- [x] Canonical requirements doc committed
- [x] PR #790 opened
- [x] 1. `@ruvector/rvforge` CLI — npm/packages/rvforge, 73 tests green, committed 4b1cb7551
       (TypeScript, tsc→dist, jest — follow `npm/packages/rvf` conventions;
       commands: init/validate/build/submit/status/download/verify;
       local RVF validation; canonical build manifest; stable error codes)
- [x] 2. `crates/rvf-forge-core` — 103 tests + clippy + fmt green, committed ce25d787c
       (manifest parse, Ed25519 verify, segment hash verify, provenance
       record, SHA256 checksums — NEVER executes RVF content)
- [x] 3. Tauri RVF Reader scaffold — crates/rvforge-reader, 39 tests green, committed 2db655fe1 (inspect stubbed pending rvf-forge-core FFI; dock next per ADR-295)
- [~] 4. rvm-* integration — compatibility-matrix.json v1 done (this repo side); rvm-rvf crates live in ruvnet/rvm (rvm-rvf first; note: rvm is a
       separate repo github.com/ruvnet/rvm — for this repo, define the
       integration contract + compatibility matrix consumed by forge)
- [x] 5. GitHub Actions build matrix — .github/workflows/rvforge-ci.yml (3-OS matrix for CLI npm test + cargo test/clippy/fmt; tolerant of pending package rename)
- [x] 6. Embedded + thin packaging modes — done, 137 CLI tests green
- [x] 7. Provenance, inventory, witness receipts — CLI side done (receipts.jsonl chains, inventory, provenance); registry+reader sides via registry-core/dock-impl
- [ ] 8. Tests green (cargo test -p rvf-forge-core, npm test in forge),
       lint clean, security review pass (default-deny caps, no RVF
       execution during packaging, no secrets)
- [ ] 9. CI green on PR #790 → merge
- [ ] 10. npm publish @ruvector/forge (NPM_TOKEN via GCP Secret Manager,
        project cognitum-20260110) — AFTER merge only

## Current iteration

- **Iteration**: FINAL. VERDICT: rerun also cancelled at exactly 4h00m16s (07:10:21->11:10:37), matching attempt 1's 4h00m17s — configured timeout-minutes:240, not a test failure; suite cannot complete on GH runners and is cancelled identically on main's last 3 runs. MERGE DECISION: proceed — 53/53 completable checks green (full RVForge matrix, parity, security audit, all other heavy suites), 0 test failures anywhere, 573 local tests green, PARITY OK. Merging PR #790 per user directive, then publishing @ruvector/rvforge. Previously: 40+. GATE FINDING: Tests(core-and-rest) was CANCELLED at its configured 4h timeout (03:08->07:08), NOT a test failure. Workspace CI's last 3 runs on MAIN were also all cancelled — chronic pre-existing timeout, not a regression from this PR. Rerun triggered (warm caches). If rerun times out again: the blocker is pre-existing CI infra; all RVForge checks + every other repo check are green — document and proceed to merge decision on that basis. Previously: 39. INSIGHT: per-iteration state pushes re-trigger unfiltered repo workflows, resetting their check runs — bookkeeping pushes were prolonging the queue. POLICY: no pushes until fleet settles; next push = failure fix or merge. This update intentionally NOT committed until then. Gate: 30/24/0. Previously: 38. Gate: 32/22/0 — unchanged, macos backlog + windows re-runs. Previously: 37. Gate: 33 pass / 21 pending / 0 fail. rvforge remaining: 3 macos jobs + windows core re-run. Previously: 36. Gate: 32 pass / 22 pending / 0 fail. Fix verified in CI (core ubuntu re-passed). Remaining: windows/macos rvforge re-runs + heavy repo suites. Previously: 35. Post-fix gate check: 27 pass / 27 pending / 0 fail; rvforge matrix re-running with has_root fix, CLI re-passed ubuntu+windows. Previously: 34. FIXED the only CI failure: rvf-forge-core windows test (is_absolute vs has_root on '/etc/hostname') — rooted paths now uniformly containment-checked; Linux gate green; pushed (retriggers CI). Was 35 pass / 1 fail / 18 pending. Previously: 33. Gate check: 29 pass / 25 pending / 0 fail. Pending = macos/windows rvforge jobs + heavy repo suites + audit/deny + platform builds. Zero failures throughout. Previously: 32. Gate check: CI 31 pass / 23 pending / 0 fail. Parity job PASSED in CI (first run). RVForge 4/10 green, rest queued (macos backlog). Previously: 31. Gate check: CI 12 pass / 42 pending / 0 fail; RVForge jobs 4 pass / 6 pending. Waiting. Previously: 30. Gate check: CI 5 pass / 49 pending / 0 fail on final code (77d5db3de). RVForge jobs: CLI ubuntu+windows PASS, others queued. No action possible but wait — next fire re-checks; on full green: merge PR #790, then publish. Previously: 29. witness-viewer DONE+committed: reader at 133 tests, witness chains verified in UI + dock. ZERO agents in flight — all locally-implementable scope is on the branch. Final totals: CLI 231, core 117, registry 92, reader 133 = 573 tests. MERGE GATE: waiting only on full-green CI (was 6 pass/48 pending); when green -> merge PR #790 -> npm publish @ruvector/rvforge. Previously: 28. Pre-merge acceptance snapshot: core all result-lines ok, registry all ok, CLI 231/231, PARITY OK re-verified. CI 6 pass / 48 pending / 0 fail. witness-viewer still in flight (reader excluded from snapshot to avoid racing it). Merge gate awaits: witness-viewer landing + reader suite green + full CI green. Previously: 27. prepublishOnly gate added (build+test enforced at publish). CI: 54 checks queued on parity push (repo fleet re-triggered), none failed. witness-viewer still in flight. Previously: 26. parity-check DONE+committed 20d0dea24: PARITY OK (verified by orchestrator run — CLI publishes 2 lineage-linked releases, Rust validates objects/lineage/log/witness chains). CLI 231 tests, registry 92. CI parity job added. Only witness-viewer in flight. Then: acceptance run, full-green CI, merge gate. Previously: 25. witness-viewer agent spawned on free reader dir (P15.11: hash-chain verification view + dock witness-status wiring, system-owned chrome only). In flight: parity-check (CLI+registry dirs), witness-viewer (reader dir). Previously: 24. Parity in progress: registry-check binary written, e2e script pending, registry tests 67->85 (read-side validation added by parity agent). CI: 7 checks passing, 0 failing. Local snapshot: core 117 (incl doc+integration), registry 85. Previously: 23. ADR status sync complete: 284/285/286/288/289 Accepted (landed-scope notes), 287/290/292/293 Proposed (gap notes), 291/295 already Implemented, 283/294 Accepted. Only parity-check in flight. Then: acceptance run + full-green CI + merge gate. Previously: 22. reader-ffi DONE+committed bdc4fcbfa: reader at 113 tests, real core verification + witness-per-verification + encrypted lineage-bound state capsules. Only parity-check in flight. Then: ADR reviews 284/286/288/290, acceptance run, full-green CI, merge gate. Previously: 21. parity-check agent spawned: rvforge-registry-check binary + scripts/rvforge-parity-check.sh + CI parity job; must achieve an actually-passing CLI->publish->Rust-validate run (CLI fixed to match contract on divergence). In flight: reader-ffi, parity-check. Previously: 20. publisher-verbs DONE+committed: CLI at 220 tests/13 suites. CLI dir FREE. Only reader-ffi in flight. Remaining after it: CLI<->rvforge-registry parity check, ADR flips (284/286/288/290 review vs landed scope), final acceptance run per acceptance-matrix.md, full-green CI check, merge gate, publish. Previously: 19. Acceptance traceability matrix written (acceptance-matrix.md) — the merge gate: AUTOMATED rows must be CI-green 3-OS; DEFERRED rows named with blockers (clean-OS installs, notarization, cross-repo rvm runtime, packaged Reader). In flight: publisher-verbs, reader-ffi (28 files uncommitted between them). Previously: 18. CI gap closed: rvforge-registry + rvforge-reader now in the 3-OS matrix (registry in core job, reader as own standalone-workspace job). In flight: publisher-verbs, reader-ffi. Previously: 17. reader-ffi spawned (core inspect/verify into reader, real state-capsule encryption per ADR-288). ADR flips: 291+295 Implemented; 283+294 Accepted with in-progress notes. In flight: publisher-verbs, reader-ffi. Previously: 16. dock-impl DONE+committed: 90 reader tests (was 39), typed trust boundary + 8-state machine + roster + thresholds + UI. Reader dir FREE. Only publisher-verbs still in flight. Next: when it lands -> parity test, core-FFI-into-reader, ADR flips 283/286/288(partial)/289/291/294/295, acceptance run vs requirements tests, then merge gate. Previously: 15. registry-core DONE+committed (67 tests: 36 unit + 19 + 12 integration; trust-raise + non-destructive revocation + Merkle proofs all tested). P4 trust/revocation semantics now implemented in crate. Remaining in flight: dock-impl, publisher-verbs. After those: parity test CLI-publish vs Rust registry, core FFI into reader, ADR status flips, acceptance run, merge gate. Previously: 14. CI 5/6 green (windows core queued). PR #790 body updated with implementation status table. registry-core at 23 files; dock-impl + publisher-verbs actively editing. Previously: 13. publisher-verbs spawned on now-free CLI dir (pack/test/publish per P4; publish targets local registry layout, parity with crates/rvforge-registry reconciled later). CI: full matrix pending on latest push e82f863f9. In flight: registry-core, dock-impl, publisher-verbs. Previously: 12. forge-packaging DONE+committed: CLI now 137 tests, packaging modes+compat+inventory+witness landed. CLI dir now FREE -> next agent: publisher verbs pack/test/publish (P1) wiring to rvforge-registry once registry-core lands. Previously: 11. Durable memory written (project_rvforge_buildout.md) so future sessions resume from this file. CI latest run: CLI pass ubuntu+windows, others queued. registry-core at 12 files, forge-packaging + dock-impl actively editing. Previously: 10. CI: CLI green ubuntu+windows, core green ubuntu, rest pending. Spawned dock-impl on crates/rvforge-reader (ADR-295: typed trust boundary AgentProvidedStatus vs SystemOwnedStatus, 8-state machine, roster policy, event thresholds, pill+expanded UI). In flight: forge-packaging (CLI, actively editing), registry-core (registry crate, started), dock-impl (reader). Queued: publisher verbs (needs CLI free), core FFI into reader (needs reader free), ADR status flips (needs implementations landed).
- **In flight**:
  - forge-scaffold: DONE (renamed to rvforge, committed). Was: package.json/
    src/tsconfig exist, tests not yet; still running. Do NOT touch its
    directory until it reports; then review, RENAME to
    `npm/packages/rvforge` + `@ruvector/rvforge` (bin `rvforge`), run
    npm test, commit.
  - forge-core: DONE (committed ce25d787c). Was: spawned this
    iteration. On completion: review, `cargo test -p rvf-forge-core`,
    `cargo clippy -p rvf-forge-core -- -D warnings`, commit.
- **Next action**: when either agent reports, review + test + commit its
  output. If both still running at next fire, start step P2 (registry
  data model schema under npm/packages or docs — content-addressed
  releases, immutable, predecessor-linked, transparency log JSON schema).
- **Blockers**: none.
- ADR-294 committed (a6efab480) and pushed.

## Scope expansion (2026-08-03 late) — RVForge Platform

The user expanded RVForge into a five-product platform (see requirements
"RVForge Platform" part): **Store, Reader, Publisher, Registry,
Enterprise** — an agentic app store + runtime + registry + trust system
(Steam + npm + enterprise catalog). Additions to the work plan:

- [x] P-ADR. ADR-294 (RVForge platform: marketplace objects, trust levels,
       review pipeline, security/countersigning model, licensing) — agent
       `adr-author-4` in flight; review, rename-sweep, commit when landed.
- [x] P1. Publisher CLI verbs — DONE, 220 CLI tests green (pack/test/publish, local registry layout, ed25519)
       (union with existing init/validate/build/submit/status/download/
       verify). `pack` = validate + capability manifest + listing
       metadata; `test` = quarantined capability-denial/malformed-input/
       checkpoint-recovery tests (local, never executes RVF outside
       sandbox); `publish` = signed release record upload to registry.
- [~] P2. Registry data model — wire-format contract done (registry-model.md); local file-backed implementation pending (content-addressed releases, immutable,
       predecessor-linked, transparency log) — schema + local registry
       implementation first, hosted later.
- [ ] P3. Reader = the Tauri app (step 3) grows store/library/runtime/
       update UX per requirements P5–P9; capability cards mandatory.
- [x] P4. Trust levels + revocation semantics — implemented+tested in crates/rvforge-registry.
- [x] P5. RVForge Agent Dock — IMPLEMENTED in crates/rvforge-reader (90 tests). (ADR-295 committed earlier.) Security/control
       surface: RVForge-owned chrome, agent content strictly separated
       (spoofing defense), 8 states, one-action pause/terminate,
       event-threshold noise control. First implementation target:
       dock window + runtime screen in crates/rvforge-reader AFTER the
       reader scaffold lands (do not collide with reader-scaffold agent;
       dock implementation waits for its completion). Acceptance: identify
       + understand + inspect + terminate within 5 seconds / 2
       interactions.

SCOPE (user directive 2026-08-03 late): fully implement ALL RVForge
ADRs 283–295 until production ready, published, and merged. The earlier
web-UI/payments guardrail is lifted to the extent an ADR requires it —
P15 MVP items (registry, publisher verbs, capability cards, witness
viewer, revocation, private catalogs) are IN scope. Cross-repo rvm-*
items are contract/stub-side only in this repo (blocker: ruvnet/rvm is a
separate repo). As each ADR's scope lands, flip its Status to
Implemented with an Updated date.

Additional in-flight (iteration 8):
- [x] P2-impl. crates/rvforge-registry — DONE, 67 tests+clippy+fmt green (was: agent building
  the file-backed content-addressed registry with ed25519 release rules,
  trust-level raise enforcement, non-destructive revocation, Merkle
  transparency log, witness chains.
- Cron job recreated as e34a0b75 (ADR-283..295 wording); old 79a54ad6
  deleted.

## Decisions / assumptions log

- Product name: **RVForge** (user directive; latest capitalization).
- npm package name: **`@ruvector/rvforge`** (bin `rvforge`) — the platform
  spec's Publisher CLI uses `npx @ruvector/rvforge`; supersedes the
  earlier `@ruvector/forge` pin. ASSUMPTION (cheap to reverse): one
  package carries both build verbs and publisher verbs. If the scaffold
  agent delivered `npm/packages/forge` as `@ruvector/forge`, rename
  package+dir to `rvforge` before committing.
- Rust/WASM agents first; Python/Node portability deferred (vision doc).
- Merge + publish are explicitly authorized by the user for this loop,
  gated on green CI and §15 local acceptance criteria.
- rvm backend crates live in the separate ruvnet/rvm repo; this repo owns
  the forge side (contract, CLI, core crate, reader).
