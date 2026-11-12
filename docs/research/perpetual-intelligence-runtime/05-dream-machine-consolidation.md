# Dream Machine Consolidation Position (PIR WP2)

**Status**: Delivered with the first WP2 slice · **Tracking**: ruvector#838, dream-machine#22 · **Decision basis**: ADR-306

Four independent implementations of "nightly, evidence-gated evaluation" exist across the org.
Leaving four live is the outcome WP2 exists to prevent. This document states the position for each:
keep/adapt/deprecate, which layer it owns, and how it is wired.

## Consolidation table

| # | Implementation | Position | Layer it owns | Wiring |
|---|---|---|---|---|
| 1 | `ruvnet/dream-machine` engine (npm `dream-machine` v0.1.1, ADR-0001 Accepted) | **Keep — adopt as evaluation core** | Orchestration + record: verdict vocabulary (`ACCEPT`/`REJECT`/`INCONCLUSIVE`), 10-column ledger, witness stamp (`sha256(sha256(report)+commit)`), schedule/compile, learning signals | Consumed by ruvector via `crates/ruvector-sota-bench/harness/src/dreamMachine.ts` (this slice). Engine is invoked through its published programmatic API (`run(argv, io)`) with an in-memory IO — no fork, no vendoring. |
| 2 | ruflo dream cycle (`ruvnet/ruflo`) | **Deprecate as an implementation; keep its contracts** | Historically: nightly cycle prototype. Going forward: none. Its ADR-322A/322B evaluation↔promotion transaction model and ADR-381 sequential-evidence statistics remain the *contracts* PIR adopts | dream-machine ADR-0001 explicitly names it as a subsumed prior instance. New dream-cycle work lands in `ruvnet/dream-machine`; ruflo keeps only the transaction/witness contracts (ADR-322C) that verdicts are logged against. |
| 3 | metaharness `docs/dream-cycle/` (prior reference instance; note: there is **no** metaharness "ADR-251") | **Deprecate — documentation-only prior art** | None going forward; it was the documented nightly-routine reference that dream-machine productized | Superseded by dream-machine ADR-0001 (which names it as subsumed). No code to migrate; its ledger format lives on as `@dream-machine/ledger`'s 10-column schema. |
| 4 | ruvector research-gate + sota-bench harness (ADR-282: `scripts/research-gate/`, `harness/src/{statistics,vetoes,flywheel,darwin}.ts`) | **Keep — adapt as the statistical + promotion-recommendation layer** | Paired-bootstrap significance (`statistics.ts`), hard vetoes (`vetoes.ts`), Ed25519-signed replay bundles + promotion rule (`flywheel.ts`), Darwin mutation proposals (`darwin.ts`), CI wiring (`.github/workflows/research-*.yml`) | dream-machine verdicts enter this layer as veto-provider input (`dreamMachineVetoProvider`), feeding the existing `ruvectorPromotionRule`. research-gate's `verify_promotion.py` and the proof gate remain unchanged downstream authorities. |

## The integration contract (who owns what — no duplicated authority)

- **Paired-bootstrap significance**: owned by ruvector `statistics.ts` (ADR-282). The engine never
  recomputes it; the adapter maps `PairedDecision.outcome` → engine verdict via `verdictFromDecision`.
- **Verdict vocabulary, ledger record, witness stamp**: owned by the dream-machine engine. The
  adapter drives the engine (`witness stamp` → `ledger append` → `ledger verify` → `witness verify`)
  and returns a frozen `DreamMachineVerdict`.
- **Hard vetoes and the promotion recommendation**: owned by ruvector `vetoes.ts`/`flywheel.ts`.
  Verdicts are conjunctive veto input: `ACCEPT` adds no credit; `REJECT`/`INCONCLUSIVE` block.
  No mutation reaches the proof gate without passing a verdict.
- **Signed replay bundle**: owned by ruvector `flywheel.ts` (`@metaharness/flywheel` signer),
  unchanged by this slice.
- **Promotion**: owned by a human. The adapter exports no promote/merge function (test-enforced);
  a verdict is an input to `ruvector-proof-gate`/`rvm-proof`, never a bypass of it.

## Wiring diagram

```
Darwin proposals (darwin.ts)
        │
        ▼
benchmark.ts observations ──► statistics.ts PairedDecision ─┐
                                                            ▼
                              dreamMachine.ts (engine: verdict + ledger + witness)
                                                            │ vetoesFromVerdict
                                                            ▼
capabilityVetoProvider ─────────────────────────► vetoes (conjunctive)
                                                            │
                                                            ▼
                              flywheel.ts ruvectorPromotionRule → recommendation only
                                                            │
                                                            ▼
                    research-gate / proof gate / HUMAN merge (unchanged)
```

## Out of scope for this slice (tracked in #838)

- `@metaharness/*` version-drift resolution (darwin 0.9.1 / flywheel 0.1.10 vs pinned 0.8.0 / 0.1.7) — WP0b (#846).
  Note: the published `dream-machine` npm package is dependency-free (bundled dist), so this slice does not collide with the pins.
- Witness-logging verdicts against ruflo ADR-322C's JCS/Ed25519 contract (program-wide, ADR-312).
- Cloud Run control plane in `ruv-dev`/`us-central1`.
- SONA dream-replay pre-filter feeding candidates into this pipeline.
