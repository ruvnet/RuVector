# RVForge Acceptance Traceability Matrix

> Merge gate for PR #790. Every locally-testable acceptance criterion
> from requirements §15 (release), RVM §13, the platform test, and the
> dock test, mapped to its automated evidence. Criteria that require
> environments this repo cannot host (clean-OS installs, notarization,
> hosted RVM backends, bare metal) are marked DEFERRED with the blocker —
> they gate the *release*, not this PR's merge.

## Release acceptance (§15)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Every installer completes on a clean OS | DEFERRED | needs real Tauri bundling + clean VMs; hosted build service pending (ADR-283 note) |
| 2 | Every embedded RVF has the same SHA256 | AUTOMATED | CLI: embedded-mode identical-hash invariant tests (build fails on divergence) |
| 3 | Offline execution succeeds in embedded mode | DEFERRED | needs packaged Reader binary; Reader verify path is offline-only by construction (no network code — enforced in reader tests) |
| 4 | Tampering prevents execution | AUTOMATED | core: tampered-bytes verify fails; CLI: tampered artifact FORGE_E_VERIFY_FAILED; reader: tampered fixture → no capability card, no load |
| 5 | Undeclared filesystem/network access denied | PARTIAL | default-deny in capability types + manifest denials tests (CLI, registry, reader); runtime enforcement needs RVM/WASM host (cross-repo rvm) |
| 6 | State survives restart without modifying base RVF | AUTOMATED (reader) | state-capsule round-trip + lineage-binding tests; base immutability by construction (capsule separate dir) |
| 7 | Uninstall removes runtime+state per policy | DEFERRED | needs packaged installers |
| 8 | Build provenance and witness records verify independently | AUTOMATED | CLI witness chain build→verify continuity + broken-chain detection; registry receipt chains; provenance recheck in verify |

## RVM acceptance (§13) — cross-repo

Items 1–8 depend on rvm-* crates in ruvnet/rvm. This repo's side (contract,
compatibility matrix, capability schema, state-delta contract) is
implemented and tested; runtime execution semantics are DEFERRED to the
rvm repo. Blocker recorded in loop-state.md.

## Platform acceptance

| Criterion | Status | Evidence |
|---|---|---|
| Publisher uploads one signed RVF | AUTOMATED | CLI publish round-trip tests (keygen → pack → publish; content addresses verified) |
| Automated review detects exact capabilities | AUTOMATED | pack: capability manifest extraction + vague-scope manual-review flagging tests |
| User installs without developer tools | DEFERRED | needs packaged Reader |
| RVM denies undeclared access | DEFERRED | cross-repo rvm |
| Agent runs offline | DEFERRED | needs packaged Reader |
| Update requests fresh permission | AUTOMATED (model) | registry: predecessor lineage + release immutability; reader: capability-card re-render per manifest (update-diff UX pending) |
| Build/install/privileged actions verify via witness record | AUTOMATED | witness chains across CLI + registry + reader (per-component); unified cross-component viewer pending |

## Dock acceptance (ADR-295)

| Criterion | Status | Evidence |
|---|---|---|
| Identify active agent | AUTOMATED | roster attention-priority selection tests |
| Understand what it is doing | AUTOMATED | sanitized agent-provided task text tests (spoofing rejected) |
| Inspect its permissions | AUTOMATED | capability-card-from-manifest tests |
| Terminate within 5s / 2 interactions | AUTOMATED | acceptance-path test (expand + terminate ≤ 2 API interactions) |
| Agent cannot alter chrome/trust state | AUTOMATED | typed AgentProvidedStatus/SystemOwnedStatus separation + sanitizer tests |

## Merge-gate summary

PR #790 merges when: (a) all AUTOMATED rows are green in CI on all three
OSes, (b) no PARTIAL row has regressed, (c) DEFERRED rows are each named
in loop-state.md with their blocker. Publish (`@ruvector/rvforge` to npm)
follows merge, never precedes it.
