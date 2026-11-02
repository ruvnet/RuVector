# Retrieval contract qualification for issue #1313

This is a **test plan only**. It does not claim current-main failures or a fix.

## Provenance
Historical Titan/RuVector CPU campaign: October 24, 2026; PDF consolidation November 2, 2026. Tested npm versions were `ruvector@0.3.3`, `@ruvector/core@0.1.33`, native `0.1.31`. The repository binary candidate was `c44d7337c0c5ad2ecd2294178475387facb96516`. The historical benchmark is not a current-main test.

## Reproduction matrix
| Contract | Setup | Expected exact path | Expected ANN path |
| --- | --- | --- | --- |
| Filtered top-k | 10k vectors, 384D, 1%, 10%, 50%, 100% eligible, k=10 | Exactly min(k, eligible_count) unique eligible IDs; 100% tie-aware oracle recall | Exactly min(k, eligible_count) only if explicitly guaranteed; measure recall independently |
| Namespace isolation | Mixed tenant metadata; filter tenant A | Zero tenant B IDs | Zero tenant B IDs |
| Sparse eligible | 0, 1, 9 eligible, k=10 | Return 0, 1, 9 | Explicit documented behavior |
| Duplicate ID | Insert ID, insert same ID with new vector, search and restart | Either reject second insert or atomic upsert; no duplicate IDs; scores agree with returned vectors | Same contract where supported |
| Nonfinite vectors | Insert/query with NaN, Infinity and -Infinity | Deterministic rejection where finite input required | Same |
| Mutations | Insert, update, delete, restart | No deleted IDs; updated scores/vectors consistent | Same |

## Implementation protocol
1. Pin main SHA, native binary hash, package versions, CPU, operating system and runtime.
2. Record an actual failing regression on main before changing implementation. If already passing, document that outcome and avoid a redundant fix.
3. Implement the smallest fix in the affected package, linking #1313 and existing #1134, #931, #430, #1150 as relevant.
4. Run package-specific unit/integration checks and report commands, exit codes and exact failures.
5. Benchmark 10k, 100k and 1M with fixed seeds, held-out queries, exact oracle, matched payload, p50/p95/p99, recall@10, completeness, RSS, and build/update costs. Distinguish ACORN from core metadata postfiltering.
6. Keep CI and human review as merge gates.

## Invariants
```text
for each result r: predicate(r.metadata) == true
unique(result.ids).length == result.length
exact_result.length == min(k, eligible_count)
for each result r: score(r.vector, query) agrees with r.reported_score within defined numeric tolerance
no returned id is deleted
```

Do not interpret underfilled top-k as tenant leakage. Do not interpret zero vectors as universally invalid. Do not claim production readiness or a measured performance improvement from this plan.
