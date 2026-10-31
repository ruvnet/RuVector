# ADR-355: Coherence-Gated ReasoningBank Retrieval — Rejected, Normalization Root Cause Identified

## Status

Rejected (for promotion into `ReasoningBank`). Experimental, additive module
(`ruvector-sona::coherence_gate`) retained in-tree as evidence and reference
implementation, not wired into `ReasoningBank::find_similar` or any default
retrieval path. No existing public API changed.

## Context

SONA's `ReasoningBank` (`crates/sona/src/reasoning_bank.rs`) stores patterns
extracted from trajectory clusters and retrieves them by
`find_similar(query, k)`: raw cosine similarity to every stored pattern,
sorted, top-k. This mirrors the retrieval step described for Google/Yale/
UIUC's ReasoningBank agent-memory framework (arXiv:2509.25140), which SONA's
module is named after and modeled on.

Pure cosine ranking cannot distinguish a pattern several independent
trajectories corroborate from a pattern that happens to sit close to the
query by chance — or one placed there deliberately. Related RAG-retrieval
literature addresses exactly this: CAGE (arXiv:2609.04647) blends a
passage's query-similarity with its similarity to the rest of the retrieved
set; GRADA builds a document-similarity graph and penalizes documents that
are query-relevant but structurally unsupported by it, specifically to catch
adversarial documents. Neither idea had been tried against ReasoningBank's
pattern store before this run.

## Hypothesis

```text
Given a SONA ReasoningBank-style pattern store containing both a
well-corroborated true-cluster pattern (6 members, mutually >= 0.90 cosine
similarity) and an adversarial singleton pattern placed deliberately near a
query vector by chance (isolated from every true cluster below the edge
threshold, mimicking a spurious or injected match),

when pattern retrieval re-ranks candidates by blending raw cosine similarity
with a coherence gate (alpha=0.5; gate = each pattern's count of other
patterns at cosine similarity >= 0.90, normalized by (bank_size - 1))
instead of ranking by raw cosine similarity alone,

then mean precision@10 (fraction of top-10 results belonging to the query's
own true cluster, averaged over 3 corpus seeds x 10 query draws per seed)
improves by >= 15 percentage points over the cosine-only baseline,

subject to: (a) a quality-weighted ablation using each pattern's existing,
here label-uninformative, avg_quality field shows no comparable
improvement, and (b) retrieval latency stays within 2x of baseline for bank
sizes up to 2000 patterns.
```

Full methodology, raw benchmark output, calibration diagnostics, and the
decoy-specific instrumentation are in
`docs/research/nightly/2026_10_31_coherence_gated_reasoning_bank/README.md`.

## Decision

1. Add `ruvector-sona::coherence_gate` (`CoherenceConfig`,
   `PatternCoherenceGraph`, `find_similar_coherence_gated`,
   `find_similar_quality_weighted`): a pure, additive module operating on
   `LearnedPattern` snapshots, with no changes to `ReasoningBank` or
   `types.rs`. No feature flag needed — no new dependency, compiles in the
   default build.
2. **Do not wire this into `ReasoningBank::find_similar` or expose it as a
   recommended retrieval path.** The measured evidence (below) falsifies
   the hypothesis: the coherence gate, as specified, produced a measured
   precision lift of +0.00 to +0.67 percentage points across all tested
   bank sizes (200/1000/2000/5000 patterns) — roughly two orders of
   magnitude below the pre-registered 15pp threshold — with an identified,
   specific root cause (see "Evidence").
3. Keep the module in-tree as a working, tested reference implementation
   and as retained negative evidence, per this repository's nightly-process
   evidence-retention convention (see ADR-345 for the comparable precedent
   of a rejected-but-retained structural-signal experiment).

## Evidence

| Gate | Threshold | Measured (n=200/1000/2000/5000) | Result |
|---|---|---|---|
| Precision@10 lift (gated vs. baseline) | >= 15.00 pp | +0.33 / +0.00 / +0.67 / +0.00 pp | **FAIL** |
| Quality-weighted ablation: no comparable lift | qualifying condition | -34.33 / -39.33 / -37.67 / -35.00 pp (large negative — no lift to confound) | holds |
| Latency ratio (gated / baseline, mean), n <= 2000 | <= 2.00x | 1.204x / 1.396x / 1.443x | **PASS** |

Full raw benchmark output and the decoy-specific diagnostic table (showing
the injected adversarial decoy present in the baseline's top-10 in 30/30
trials at every scale, and demoted by gating in 0/30 trials at every scale)
are in the nightly README and `raw_benchmark_output.txt` (linked above).

**Root cause** (measured, not inferred): `gate(id) = degree(id) /
(bank_size - 1)`. A true cluster's member degree is capped by its own
cluster size (5 siblings in this corpus) independent of bank size, while
the denominator grows with the whole store. At `alpha = 0.5`, the gate
term's maximum possible contribution to the score (`0.5 * 5/199 ~= 0.0125`
at the smallest tested bank, `n = 200`) is far smaller than the raw-
similarity gap between the decoy and a true cluster member
(`0.5 * (0.99 - 0.90) ~= 0.045`). The gate is numerically negligible at
every tested scale, and provably gets smaller, not larger, as the bank
grows — the opposite of what would be needed for the mechanism to work at
realistic-to-large scale.

## Consequences

- `ruvector-sona` gains a working, tested (if unpromoted) reference for how
  a pattern-pattern corroboration graph would integrate with
  `ReasoningBank`'s pattern type, useful to whoever next attempts a
  locally-normalized version (see "Open Questions").
- No existing behavior changes: `ReasoningBank::find_similar` and every
  other existing public API are untouched. `coherence_gate` is purely
  additive and not called from anywhere in the default build.
- A specific, falsifiable follow-up (local rather than global degree
  normalization) is documented in the nightly README's "Next Research" but
  not implemented in this run — implementing and re-benchmarking it now,
  after already knowing it would likely fix the measured problem, would be
  exactly the after-the-fact threshold-chasing this repository's nightly
  process rules forbid.

## Alternatives Considered

- **Normalize degree by a local candidate-pool size instead of whole-bank
  size** (e.g. count only among the top-M cosine candidates already being
  considered, or among patterns within some multiple of the edge
  threshold's radius). Not implemented in this run — this is the primary
  identified fix, deliberately left for a future, independently-benchmarked
  pass rather than retrofitted here to pass the pre-registered threshold.
- **Weighted degree (sum of similarities above threshold) instead of a
  plain count.** Would not address the root cause: the normalization
  denominator is the actual problem, not whether numerator uses count or
  weighted-sum.
- **Rebuild the coherence graph only among the top-k candidates per query**
  (query-time, not corpus-time). Would sidestep the whole-bank denominator
  issue by construction (denominator = k, not n), but changes the
  amortization story (graph build would move from "once per corpus change"
  to "once per query"), which the latency sub-criterion's current framing
  does not account for — flagged as a separate design question, not
  attempted here.
- **Ship the mechanism with `alpha` tuned per-bank-size to compensate for
  the shrinking gate.** Rejected outright: this is curve-fitting to a known
  flaw rather than fixing it, and it would silently break again at any bank
  size not specifically tuned for.

## Implementation Plan

Already implemented in this PR:

- `crates/sona/src/coherence_gate.rs` (new module, ~280 lines + tests)
- `crates/sona/src/lib.rs`: added `pub mod coherence_gate;` and re-exports
- `crates/sona/examples/coherence_gate_bench.rs` (benchmark harness)
- 8 unit tests in `coherence_gate.rs` (correctness, malformed input,
  concurrency, scale — see nightly README "Testing")

No further implementation is planned under this ADR; the locally-normalized
follow-up is explicitly out of scope here (see "Open Questions" and the
nightly README's "Next Research").

## API Shape

```rust
pub struct CoherenceConfig {
    pub edge_threshold: f32,
    pub alpha: f32,
}

pub struct PatternCoherenceGraph { /* opaque */ }

impl PatternCoherenceGraph {
    pub fn build(patterns: &[LearnedPattern], edge_threshold: f32) -> Self;
    pub fn gate(&self, id: u64) -> f32;
    pub fn edge_threshold(&self) -> f32;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}

pub fn find_similar_coherence_gated(
    patterns: &[LearnedPattern],
    graph: &PatternCoherenceGraph,
    query: &[f32],
    k: usize,
    alpha: f32,
) -> Vec<(u64, f32)>;

pub fn find_similar_quality_weighted(
    patterns: &[LearnedPattern],
    query: &[f32],
    k: usize,
    alpha: f32,
) -> Vec<(u64, f32)>;
```

## Feature Flags

None added. `coherence_gate` has no new dependency and compiles in the
default build; it is simply not called from any default code path.

## Benchmark Evidence

See "Evidence" above and the linked nightly README and
`raw_benchmark_output.txt` for full raw output, exact commands, and the
calibration/decoy diagnostics.

## Security

- No new cryptographic primitive, I/O, or external dependency.
- **This mechanism, even corrected for the normalization bug, is not a
  security control.** Degree-based corroboration is reported in adjacent
  GraphRAG-poisoning literature as itself an attack surface: an adversary
  (or a buggy upstream pipeline) that writes several mutually-similar fake
  patterns, rather than one isolated decoy, would be rewarded by the exact
  mechanism meant to demote isolated spurious matches. This run only tested
  the single-decoy case. Any future revival of this mechanism must be
  described as a ranking refinement against *accidental* lucky matches, not
  a defense against a motivated writer to the bank — see the nightly
  README's "Security and Governance" for the full discussion and citations.
- `find_similar_coherence_gated`/`find_similar_quality_weighted` perform no
  independent input validation beyond what `LearnedPattern::similarity`
  already provides (returns `0.0` on dimension mismatch); this matches
  `ReasoningBank::find_similar`'s existing contract and is not a new gap.

## Governance

Follows this repository's established pattern (e.g. ADR-345) for a rejected
hypothesis with a specific, measured root cause: retain the implementation
and evidence in-tree, do not wire it into any default path, and record the
falsifiable next step rather than silently abandoning the thread.

## Failure Modes

1. **Normalization vanishing at scale** (the measured root cause): the gate
   signal shrinks as the bank grows, so the mechanism degrades exactly in
   the direction opposite to what production scale would need. Documented
   above; not patched in this run.
2. **Single-decoy methodology blind spot**: this run's benchmark only
   injects one isolated adversarial pattern per query. A coordinated
   multi-pattern injection was not tested and, per the cited literature,
   would likely defeat even a corrected version of this mechanism (see
   "Security").
3. **`O(n^2)` graph build cost**: ~0.96s measured at `n = 5000`. Amortized
   per corpus-change (not per query) in this design, but still a real,
   unaddressed scaling concern if this module were ever revived for larger
   pattern stores.

## Migration

None — no existing API changed, nothing to migrate.

## Rollback

None needed — the module is not called from any default path. Removing
`crates/sona/src/coherence_gate.rs` and its `lib.rs` re-exports would fully
revert this change with no other side effects.

## Rejection Criteria (restated)

ACCEPT required precision@10 lift >= 15pp at every tested bank size, the
quality-weighted ablation showing no comparable lift, and latency <= 2x
baseline at bank sizes up to 2000. The precision criterion failed by two
orders of magnitude at every tested scale. **Rejected.**

## Open Questions

1. Does a locally-normalized gate (denominator = candidate pool size or a
   fixed neighborhood size, not whole-bank size) clear the same 15pp
   threshold? Unanswered — the natural next experiment, deliberately not
   attempted in this run.
2. Does the effect (if any, once correctly normalized) survive a
   coordinated multi-decoy adversarial injection, or only the single-decoy
   case tested here? Unanswered.
3. Is `ReasoningBank`'s real default scale (`k_clusters = 5`, typically a
   handful of live patterns) different enough from this benchmark's
   200-5000-pattern range that a small-n-specific experiment is warranted
   before drawing any conclusion about default-configuration behavior?
   Flagged as a representativeness gap, not resolved here.
