# Coherence-Gated ReasoningBank Retrieval: A Degree-Normalization Pitfall

**Date**: 2026-10-31
**Crate**: `ruvector-sona` (new module `coherence_gate`)
**Result**: **REJECT** (precision-lift criterion decisively fails at every
tested bank size; root cause identified and isolated)

## Summary

SONA's `ReasoningBank` (`crates/sona/src/reasoning_bank.rs`) retrieves stored
reasoning patterns by raw cosine similarity alone (`find_similar`) — the same
retrieval primitive described for Google/Yale/UIUC's ReasoningBank agent-memory
framework (arXiv:2509.25140). Nothing in that ranking distinguishes a pattern
many independent trajectories corroborate from one that merely landed close to
the query by chance, or was planted there. This run implemented and benchmarked
a **coherence gate**: blend each pattern's raw similarity with a
corroboration signal derived from a pattern-pattern similarity graph (edges
above a cosine threshold; gate = normalized degree), mirroring the
coherence-aware reranking idea in CAGE (arXiv:2609.04647) and the
graph-based adversarial-document defense in GRADA, applied here to
ReasoningBank's pattern store instead of a RAG passage set.

The pre-registered hypothesis (precision@10 lift >= 15 percentage points over
cosine-only baseline, with a quality-metadata ablation showing no comparable
lift) is **REJECTED**: the measured lift was between +0.00 and +0.67
percentage points across all four tested bank sizes (200 / 1000 / 2000 /
5000 patterns), three corpus seeds, and 30 query trials per size. A
diagnostic instrumentation pass shows *why*: the coherence gate as specified
— degree normalized by the **whole-bank size** `n - 1` — is numerically
negligible whenever the bank holds more than a few hundred patterns, because
a genuinely corroborated cluster's absolute degree (a handful of sibling
patterns) is swamped by a denominator counting every other pattern in the
store. The adversarial decoy pattern (an isolated near-duplicate of the
query) was present in the baseline's top-10 in 30/30 trials at every scale,
and gating demoted it from the top-10 in **0/30** trials. The latency
sub-criterion (coherence-gated retrieval within 2x baseline latency) *did*
hold at every tested scale (1.12x-1.52x), but that is moot given the primary
criterion's outright failure.

This is documented as a clean negative result with an identified, specific,
fixable flaw (see "Next Research"), not implemented further in this run per
the house rule against re-tuning parameters until something passes.

## Hypothesis (fixed before benchmarking, not altered after seeing results)

```text
Given a SONA ReasoningBank-style pattern store containing both a
well-corroborated true-cluster pattern (6 members, mutually >= 0.90 cosine
similarity, i.e. an edge in the coherence graph) and an adversarial singleton
pattern placed deliberately near a query vector by chance (sigma_adv=0.03,
isolated from every true cluster below the edge threshold, mimicking a
spurious or injected match),

when pattern retrieval re-ranks candidates by blending raw cosine similarity
with a coherence gate (alpha=0.5; gate = each pattern's count of other
patterns at cosine similarity >= 0.90, normalized by (bank_size - 1)) instead
of ranking by raw cosine similarity alone,

then mean precision@10 (fraction of the top-10 retrieved patterns that are
members of the query's own true cluster, averaged over 3 corpus seeds x 10
query draws per seed = 30 trials per bank size) improves by >= 15 percentage
points over the cosine-only baseline,

subject to: (a) a quality-weighted ablation — blending cosine with each
pattern's existing, here label-uninformative, avg_quality field at the same
alpha — shows no comparable improvement over baseline (isolating that graph
structure, not reused metadata, would be driving any gain), and (b)
end-to-end per-query retrieval latency (graph build amortized once per corpus
change, not charged per query) stays within 2x of the cosine-only baseline's
per-query latency for bank sizes up to 2000 patterns.
```

Acceptance table, fixed thresholds vs. measured results:

| Criterion | Threshold | Measured (n=200/1000/2000/5000) | Result |
|---|---|---|---|
| Precision@10 lift (coherence-gated vs. baseline) | >= 15.00 pp | +0.33 / +0.00 / +0.67 / +0.00 pp | **FAIL** |
| Quality-weighted ablation shows no comparable lift | n/a (qualifying condition) | -34.33 / -39.33 / -37.67 / -35.00 pp (large *negative*, not comparable to any lift) | holds (vacuously — there is no lift to confound with) |
| Latency ratio (coherence-gated / baseline, mean) | <= 2.00x, n <= 2000 | 1.204x / 1.396x / 1.443x / (1.522x, n=5000 beyond the committed range) | **PASS** at all tested n |

**Overall acceptance call: REJECT.** The primary criterion fails by a wide
margin at every scale, including the largest bank size (5000) tested, which
is beyond the 2000-pattern range the latency sub-criterion was scoped to.
The latency sub-criterion passing is irrelevant once the primary criterion
fails outright.

## SOTA Context (real sources, read 2026-10-31)

- **ReasoningBank** (arXiv:2509.25140, Google/Yale/UIUC, 2025, revised
  2026-03): the agent-memory framework SONA's `reasoning_bank.rs` models
  itself on. Confirmed via web search: it stores reasoning strategies
  (not raw trajectories), retrieves via embedding similarity search
  (reported as Gemini-Embedding-001 + cosine) to pull top-k memory items
  injected into the agent's context, and writes new entries by LLM-judge
  pass/fail labeling. The paper reports consistent gains over no-memory and
  raw-trajectory-store baselines — but its retrieval step, as described, is
  cosine-only, the same gap this run targeted.
- **CAGE: Coherence-Aware Graph Encoding for Retrieval-Augmented Generation**
  (arXiv:2609.04647): scores each retrieved passage by its embedding
  similarity to the *other retrieved passages*, then blends that coherence
  score with query relevance via a tunable weight — structurally the same
  alpha-blend this run's candidate A implements, applied to RAG passages
  rather than ReasoningBank patterns. The direct precedent for the blending
  formula used here.
- **GRADA (Graph-based Reranker against Adversarial Documents Attack)**:
  builds a graph over retrieved documents weighted by document-to-document
  similarity and penalizes similarity using each document's coherence with
  the rest of the retrieved set, on the premise that adversarial documents
  can look query-relevant yet lack coherence with the rest of the set —
  exactly the failure mode this run's adversarial decoy pattern was
  constructed to probe.
- **GraphER: An Efficient Graph-Based Enrichment and Reranking Method for
  RAG** (arXiv:2603.24925): its "Graph Cohesive Smoothing" step exists
  specifically to reduce bias toward hub nodes in a similarity graph — a
  direct, relevant caution against naive degree-based scoring, which this
  run's root-cause finding independently reproduces from a different angle
  (degree-based scoring here failed not by over-rewarding hubs, but by being
  numerically swamped at scale; both are symptoms of using *raw or
  globally-normalized* degree as a trust signal without controlling for
  bank size or local density).
- **GRAGPoison / degree-centrality vulnerability** (summarized via secondary
  sources reporting on 2025-2026 GraphRAG poisoning literature, not verified
  against a single primary paper in this pass): reports that adding
  supporting relations to a knowledge/document graph raises poisoning attack
  success 40-60%, i.e. raw degree-as-corroboration is itself an attack
  surface when an adversary can insert multiple mutually-corroborating fake
  patterns rather than one isolated one. This directly bounds how far this
  run's approach could be pushed even if the normalization bug were fixed —
  see "Security and Governance" below.

## Architecture

New module: `crates/sona/src/coherence_gate.rs` (~280 lines of
implementation + tests), additive — no changes to `reasoning_bank.rs` or
`types.rs`. Operates on `&[LearnedPattern]` snapshots so it composes with
`ReasoningBank::get_all_patterns()` without requiring any change to the
bank's internal K-means/trajectory bookkeeping.

- `CoherenceConfig { edge_threshold: f32, alpha: f32 }` — tunable parameters,
  `Default` = `{ edge_threshold: 0.75, alpha: 0.5 }` (the benchmark uses
  `edge_threshold = 0.90`, chosen via the calibration step below, not the
  struct default — see "Adversarial Self-Review").
- `PatternCoherenceGraph::build(patterns, edge_threshold)` — `O(n^2)`
  pairwise cosine, one edge per pair at or above threshold, gate per pattern
  = `degree / max(n - 1, 1)`, clamped to `[0, 1]`. Built once per corpus
  change (analogous to rebuilding after `ReasoningBank::extract_patterns` or
  `consolidate`), not per query.
- `find_similar_coherence_gated(patterns, graph, query, k, alpha)`
  (**candidate A**): `score = alpha * cosine + (1 - alpha) * gate`, top-k,
  deterministic tie-break by ascending id.
- `find_similar_quality_weighted(patterns, query, k, alpha)`
  (**candidate B**, ablation): `score = alpha * cosine + (1 - alpha) *
  normalized(avg_quality)`, using existing `LearnedPattern::avg_quality`
  metadata instead of any graph signal. Exists purely to test whether
  reused bookkeeping fields already capture what the graph captures.
- **Baseline** (not a separate module — the benchmark's
  `find_similar_baseline` reimplements exactly `ReasoningBank::find_similar`'s
  algorithm directly over the pattern slice, for apples-to-apples timing
  without K-means/trajectory overhead unrelated to this hypothesis).

## Implementation

- `crates/sona/src/coherence_gate.rs`: `CoherenceConfig`,
  `PatternCoherenceGraph`, `find_similar_coherence_gated`,
  `find_similar_quality_weighted`, plus 8 unit tests (see "Testing").
- `crates/sona/src/lib.rs`: added `pub mod coherence_gate;` and re-exported
  `CoherenceConfig`, `PatternCoherenceGraph`, `find_similar_coherence_gated`,
  `find_similar_quality_weighted`.
- `crates/sona/examples/coherence_gate_bench.rs`: the benchmark harness
  (corpus generation, calibration diagnostic, precision/latency
  measurement, decoy-specific diagnostic).

No existing public API changed; no existing test modified or weakened; no
feature flag needed (pure-Rust, no new dependency — reuses the crate's
existing `rand = "0.8"` dependency, already present for other examples).

## Testing

`cargo test -p ruvector-sona --lib` (full crate): **122 passed, 0 failed, 0
ignored** (114 pre-existing + 8 new). No regression in any existing test.

The 8 new tests in `coherence_gate.rs`:

1. `isolated_pattern_gates_to_zero` — a singleton pattern has no possible
   corroboration.
2. `corroborated_cluster_gates_higher_than_isolated_outlier` — basic
   correctness of the degree computation.
3. `coherence_gated_demotes_isolated_near_duplicate_of_query` — the
   core hypothesis-relevant unit case, hand-constructed with angles on the
   unit circle (cluster at 24/25/26 degrees from the query, isolated decoy
   at 2 degrees) so that the *unnormalized-by-n* gate (here n=4, so the
   normalization bug does not yet bite) correctly demotes the decoy. This
   is the proof that the mechanism is correct in principle; the benchmark
   is what shows it breaks down under the chosen whole-bank normalization
   at realistic-to-large n.
4. `quality_weighted_ablation_does_not_use_graph_structure` — confirms the
   ablation, with uniform quality, behaves like the cosine baseline (a
   sanity check on the ablation's own correctness, not the hypothesis).
5. `malformed_input_does_not_panic` — **adversarial/edge case**: NaN in a
   centroid, mismatched centroid dimensions vs. the query, an empty bank,
   `k = 0`, `k > n`. No panic in any case.
6. `alpha_is_clamped` — out-of-range `alpha` (5.0, -5.0) does not panic and
   produces a finite score.
7. `scale_does_not_panic_and_gate_stays_bounded` — **scale test**: 2000
   patterns, confirms every gate value stays in `[0, 1]`.
8. `concurrent_reads_are_consistent` — **concurrency test**: 8 threads
   concurrently calling `find_similar_coherence_gated` against the same
   `Arc`-shared, immutable pattern slice and graph all agree with a
   single-threaded reference result. (Concurrent *reads* are the relevant
   concurrency scenario here: the graph and pattern snapshot are immutable
   once built — there is no mutable shared state for concurrent *writers*
   to race on in this module's API.)

## Benchmark

Exact commands (raw verbatim output in `raw_benchmark_output.txt` in this
directory):

```bash
cargo test -p ruvector-sona --lib
cargo build --release -p ruvector-sona --example coherence_gate_bench
cargo run --release -p ruvector-sona --example coherence_gate_bench
cargo test -p ruvector-sona coherence_gate --lib
```

Methodology: for each of 3 fixed seeds (1, 2, 3) and 4 bank sizes (200,
1000, 2000, 5000 patterns), a synthetic corpus is built in 64 dimensions:

- 8 true clusters x 6 members each (48 patterns), each member a small
  perturbation (`sigma_member = 0.05`) of a random unit centroid, giving a
  tight intra-cluster cosine similarity (~0.95, confirmed by the
  calibration step below).
- 10 queries per seed, each a *larger* perturbation (`sigma_query = 0.45`)
  of one cluster's centroid (not identical to any stored member) — cycling
  through all 8 clusters.
- 10 adversarial decoys per seed, one per query, each a *tiny* perturbation
  of its own query (`sigma_adv = 0.03`): a near-perfect cosine match for
  that query, isolated from every true cluster (confirmed below threshold
  by the calibration step), and carrying the *same* randomly-drawn
  `avg_quality` range as everything else (no label leak through metadata).
- The remainder padded with independent random unit noise singletons to
  reach the target bank size.

`avg_quality` is drawn independently and uniformly for every pattern,
regardless of true/noise/decoy label — deliberately uninformative, so
candidate B's ablation result isolates whether the *graph structure* (not
reused metadata) explains any effect.

A calibration diagnostic (printed, used only to verify the embedding
geometry actually separates "corroborated member" similarity from "isolated
decoy" similarity at the chosen `edge_threshold = 0.90` — not used to tune
the hypothesis after seeing its outcome) confirmed:

```
intra_cluster_sim=0.9492  adv_to_member_sim=0.3401  query_to_adv_sim=0.9895
cluster_above_threshold=true  adv_below_threshold=true
```

Each query trial: run baseline / coherence-gated / quality-weighted
retrieval (`Instant`-timed, after 3 warmup queries per corpus that are not
counted), record precision@10 (fraction of top-10 results in the query's
own true cluster — ground truth, not a label visible to the retrieval
functions) and per-call latency. Graph build time is reported separately
(amortized once per corpus, not charged against query latency, matching how
a `ReasoningBank` would rebuild its coherence graph after
`extract_patterns`/`consolidate`, not on every query).

### Results

| Bank size | Baseline P@10 | Gated P@10 | Quality P@10 | Lift (gated) | Lift (quality) | Baseline lat. (mean us) | Gated lat. (mean us) | Lat. ratio | Graph build (mean us) |
|---|---|---|---|---|---|---|---|---|---|
| 200  | 0.5767 | 0.5800 | 0.2333 | +0.33pp | -34.33pp | 21.84  | 26.29  | 1.204x | 1,518 |
| 1000 | 0.5467 | 0.5467 | 0.1533 | +0.00pp | -39.33pp | 105.91 | 147.86 | 1.396x | 37,792 |
| 2000 | 0.5200 | 0.5267 | 0.1433 | +0.67pp | -37.67pp | 214.50 | 309.62 | 1.443x | 151,762 |
| 5000 | 0.4767 | 0.4767 | 0.1267 | +0.00pp | -35.00pp | 555.63 | 845.65 | 1.522x | 959,717 |

Decoy-specific diagnostic (30 trials per bank size, one decoy per query,
always present somewhere in the bank):

| Bank size | Decoy in baseline top-10 | Decoy in gated top-10 | Decoy demoted by gating |
|---|---|---|---|
| 200  | 30/30 | 30/30 | 0/30 |
| 1000 | 30/30 | 30/30 | 0/30 |
| 2000 | 30/30 | 30/30 | 0/30 |
| 5000 | 30/30 | 30/30 | 0/30 |

The decoy — by construction a near-perfect cosine match for its query and
isolated from every true cluster — appears in the baseline's top-10 in
*every single trial* at every scale, and coherence gating **never** removes
it, at any scale tested.

### Root cause

`PatternCoherenceGraph::gate(id) = degree(id) / (n - 1)`, where `n` is the
**entire bank size**, not a local neighborhood size. A true cluster member's
absolute degree is bounded by its cluster size (5 other members, in this
corpus) regardless of how large the bank grows. So:

- at `n = 200`: `gate_member ~= 5 / 199 ~= 0.0251`
- at `n = 1000`: `gate_member ~= 5 / 999 ~= 0.0050`
- at `n = 2000`: `gate_member ~= 5 / 1999 ~= 0.0025`
- at `n = 5000`: `gate_member ~= 5 / 4999 ~= 0.0010`

At `alpha = 0.5`, the gate's contribution to the final score is
`0.5 * gate_member`, i.e. at most ~0.0125 even at the smallest tested bank
(`n=200`) — far too small to overcome the decoy's ~0.49 raw-similarity-only
score (`0.5 * ~0.99`) advantage over a true member's ~0.45
(`0.5 * ~0.90`) raw-similarity term. The gate term is, numerically, noise.
This is not a benchmark artifact or an unlucky corpus — it is a direct,
provable consequence of normalizing a local structural signal (a pattern's
corroboration by *its own* cluster) by a global quantity (the size of the
*entire* store) that grows independently of cluster size. The effect gets
*worse*, not better, as the bank grows — exactly the regime (hundreds to
thousands of patterns) a production `ReasoningBank` deployment would
eventually reach, even though its default `PatternConfig` keeps `k_clusters`
(and so typical pattern counts) small by default (5).

## Adversarial Self-Review

- **Could this result be gamed to produce a false ACCEPT?** No — the
  thresholds (15pp lift, 2x latency) were fixed in this document's
  hypothesis section before the benchmark was run, and were not changed
  after seeing the +0.00 to +0.67pp measured lift. The honest call given
  those fixed thresholds is REJECT, and that is what is reported.
- **Is the benchmark representative of real ReasoningBank usage?** Partially
  not, in a way that matters: `PatternConfig::default()`'s `k_clusters = 5`
  means a *real* `ReasoningBank`, as currently used, typically holds only a
  handful of extracted patterns at any time — not the 200-5000 patterns
  benchmarked here. The larger bank sizes were chosen to stress-test the
  mechanism's generality (and were explicitly bounded to <= 2000 in the
  hypothesis's latency sub-criterion for that reason), but they also happen
  to be exactly the regime where the normalization bug is most visible.
  At a more realistic few-dozen-pattern bank, the `n - 1` denominator is
  smaller and the gate signal would carry more relative weight — this run
  did not benchmark that specific small-n regime with the decoy-injection
  methodology, which is a real gap; see "Next Research".
- **Does it survive scale, deletes, or concurrency?** Scale: yes, tested up
  to `n = 5000` without panics (the `scale_does_not_panic_and_gate_stays_bounded`
  unit test and the benchmark itself both exercise this; the hypothesis's
  negative result also *holds*, i.e. gets worse, not better, at larger
  scale, so there is no scale regime tested here where the mechanism
  recovers). Deletes: not applicable to this module in isolation — it
  operates on an immutable snapshot and has no delete API of its own
  (deletions would happen in `ReasoningBank::prune_patterns`/`consolidate`,
  upstream of this module, followed by a graph rebuild). Concurrency: tested
  for concurrent reads (`concurrent_reads_are_consistent`), the only
  concurrency scenario this module's pure, read-only API exposes.
- **Is there a simpler alternative that would have worked?** Likely yes, and
  it is the natural next step rather than something to retroactively swap
  in now: normalize degree by the number of patterns actually *within
  retrieval range* of the query (e.g. the candidate pool already being
  scored, or a k-nearest-neighbor count in the coherence graph itself)
  instead of the whole bank. This was not implemented or tested in this run
  — doing so now, after seeing that it would likely fix the measured
  problem, would be exactly the kind of after-the-fact threshold-chasing the
  nightly process's rules forbid. It is recorded here as the specific,
  falsifiable next step instead.
- **Could the coherence-gate mechanism itself be gamed by an adversary
  (distinct from "could this benchmark result be gamed")?** Yes, and this is
  a real limitation even if the normalization bug above were fixed:
  degree-based corroboration is, per the GRAGPoison-adjacent literature
  cited above, itself attackable by an adversary who inserts *multiple*
  mutually-similar fake patterns rather than one isolated decoy (exactly
  the attack this run's single-decoy methodology does not probe). A
  corrected, locally-normalized gate would still reward a coordinated block
  of fake corroborating patterns exactly as it would reward a real cluster.
  See "Security and Governance".

## Security and Governance

- No new cryptographic primitive, no new I/O, no new external dependency.
  `coherence_gate.rs` is pure, allocation-only Rust operating on data already
  owned by the caller.
- **The mechanism this run tested — even if the normalization bug were
  fixed — is not a security control.** Raw pairwise-similarity degree, as a
  trust/corroboration signal, is vulnerable to coordinated injection: an
  adversary (or a buggy upstream pipeline that writes many near-duplicate
  low-quality trajectories) who can insert several mutually-similar fake
  patterns gets rewarded by the same mechanism intended to demote isolated
  spurious matches. This is a documented property of degree-based
  corroboration in the adjacent RAG-poisoning literature, not a novel
  finding of this run, but it directly bounds how this module should be
  described if ever revisited: a **ranking refinement to reduce lucky
  false positives**, never a defense against a motivated adversary able to
  write to the bank.
- `find_similar_coherence_gated` and `find_similar_quality_weighted` perform
  no input sanitization of their own beyond the NaN/dimension-mismatch
  handling inherited from `LearnedPattern::similarity` (which already
  returns `0.0` for mismatched dimensions) — this is intentional: they are
  pure scoring functions over data the caller is trusted to have validated,
  matching `ReasoningBank::find_similar`'s own existing contract.

## Ecosystem Fit

- **Agent memory / SONA**: directly relevant — this module augments
  `ReasoningBank`'s retrieval path. Not wired into `ReasoningBank` itself in
  this run (it is additive, operating on a pattern snapshot) because the
  measured result does not support promoting it.
- **Coherence scoring**: applies the "corroboration graph" idea already
  proven useful elsewhere in this repository
  (`ruvector-temporal-coherence`'s `CoherenceGraph`, 2026-06-13 nightly) to a
  structurally different data source (SONA's learned patterns rather than
  raw memory vectors) — and finds the same *idea* does not trivially
  transfer without correcting for how the normalization constant scales.
- **Vector search**: the pattern-pattern coherence graph is itself a small
  (here, up to 5000-node) similarity graph; at larger scale, building it
  with a real ANN index instead of the `O(n^2)` brute-force pairwise scan
  used here would be the natural optimization, not attempted in this run
  since the primary hypothesis failed before reaching that concern.
- **RVF/RVM, witness/provenance, WASM/edge, MCP**: not materially relevant
  to this specific module — it is an in-memory scoring refinement with no
  persistence, signing, or cross-process surface. SONA already ships WASM
  bindings for its core engine (`sona/src/wasm.rs`); `coherence_gate.rs` adds
  no WASM-incompatible dependency (pure `Vec`/`HashMap` arithmetic) and
  would compile under the existing `wasm` feature if ever exposed there, but
  this was not benchmarked under an actual WASM runtime in this pass.

## Practical Applications (if the normalization were fixed)

1. Demoting isolated, possibly-malformed or anomalous trajectory clusters
   from ReasoningBank retrieval results without needing an LLM judge call
   per retrieval.
2. A general pattern for any RuVector pattern/centroid store (not just
   SONA's) where "is this result corroborated by anything else nearby" is a
   cheap, embedding-only signal worth computing once per index rebuild.
3. A building block for a future locally-normalized version (see "Next
   Research") that could be benchmarked against the same decoy-injection
   methodology this run already built and validated.

## Long Horizon Applications

1. **Thesis**: any graph-corroboration signal layered onto embedding
   retrieval must normalize by a *local* quantity (neighborhood size,
   candidate pool size) rather than global store size, or it silently
   vanishes exactly as the store grows — a general caution for every
   RuVector crate (not just `sona`) that might reach for "add a coherence
   graph" as a retrieval improvement.
   - Required advances: a principled local-normalization scheme (e.g.
     degree relative to the k-NN candidate pool already retrieved, not the
     whole bank) with its own falsifiable benchmark.
   - RuVector's role: this repository already has at least two other
     coherence-graph implementations (`ruvector-temporal-coherence`,
     `ruvector-coherence`); a shared, correctly-normalized primitive could
     serve all of them instead of each reinventing (and each independently
     risking) the same normalization mistake.
   - Why this experiment matters: it is a concrete, measured instance of the
     mistake, with an isolated root cause, not a hypothetical concern.
   - Primary uncertainty: whether a locally-normalized version clears the
     *same* 15pp threshold, or whether the effect size is smaller still once
     corrected (untested).
   - Falsification path: rerun this exact benchmark harness
     (`coherence_gate_bench.rs`) with a locally-normalized gate; if the
     decoy is still never demoted, the mechanism itself (not just this
     normalization choice) is unsound for this task.
2. **Adversarial-memory robustness for autonomous agents**: as agents write
   their own memory (ReasoningBank-style or otherwise) with less human
   review, cheap structural checks against lucky/spurious retrieval matter
   more, but (per "Security and Governance" above) must be designed knowing
   degree-based trust is itself gameable by a motivated writer.
   - Required advances: a corroboration signal robust to coordinated
     injection, not just isolated decoys (e.g. weighting by write-time
     diversity or provenance, not just embedding proximity).
   - RuVector's role: the ecosystem's existing witness/provenance machinery
     (`ruvector-mincut`'s witness module, `ruvector-retrieval-receipt`)
     could in principle supply exactly that provenance signal to a future
     version of this mechanism.
   - Why this experiment matters: it is the first measured attempt, in this
     repository, at a non-provenance-based corroboration signal for agent
     memory, and it failed in a way that points directly at why
     provenance-based signals (already built elsewhere in this ecosystem)
     are the more defensible direction.
   - Primary uncertainty: whether a provenance-weighted version is cheap
     enough to run per-query at SONA's sub-millisecond adaptation budget.
   - Falsification path: benchmark a provenance-weighted variant against the
     same coordinated-injection attack this run did not test.

## Falsification Criteria (restated from the fixed hypothesis)

ACCEPT would have required: precision@10 lift >= 15pp at every tested bank
size, the quality-weighted ablation showing no comparable lift, and latency
<= 2x baseline for bank sizes up to 2000. The first criterion failed
decisively (+0.00 to +0.67pp, two orders of magnitude below threshold) at
every scale, including scales beyond the 2000-pattern range the latency
criterion was scoped to. **REJECT.**

## Limitations

- The benchmark's bank sizes (200-5000 patterns) are larger than a default
  `ReasoningBank` (`k_clusters = 5`) would typically hold; see "Adversarial
  Self-Review" for why this cuts both ways (stresses generality, but also
  does not test the specific small-n regime where the gate signal would
  carry more relative weight).
- Only one decoy per query was tested; a coordinated multi-decoy injection
  (several mutually-corroborating fake patterns) was not tested and, per the
  cited literature, would likely defeat even a correctly-normalized version
  of this mechanism.
- `PatternCoherenceGraph::build` is `O(n^2)`; at `n = 5000` it took ~0.96s
  (measured), which is already a meaningful one-time cost at that scale
  (though amortized across all future queries until the next rebuild, not
  charged per-query).
- The synthetic corpus's clusters are isotropic Gaussian perturbations of a
  single centroid; real SONA trajectory embeddings (derived from reward-
  weighted activation averages, per `reasoning_bank.rs::compute_embedding`)
  may have a different, non-isotropic geometry this benchmark does not
  reproduce.

## Next Research

1. Re-run this exact harness with degree normalized by a local quantity
   (e.g. the number of patterns within some fixed radius multiple of
   `edge_threshold`, or the size of the top-k candidate pool already being
   scored) instead of whole-bank size, as a direct, falsifiable follow-up —
   not attempted here per the rule against re-tuning until something passes.
2. Test a coordinated multi-decoy injection (the realistic adversarial case
   per the cited poisoning literature) once a locally-normalized version
   exists, rather than only the single-decoy case this run covers.
3. Benchmark at the actual default `ReasoningBank` scale (`k_clusters = 5`,
   so typically single-digit-to-low-dozens of live patterns) with the same
   decoy-injection methodology, to close the representativeness gap noted
   in "Limitations".
4. If a locally-normalized version clears the threshold, consider wiring it
   into `ReasoningBank` itself as an opt-in `find_similar_coherence_gated`
   method (not a default), following the same "additive, feature-gated,
   evidence-retained" pattern this repository's mincut-adjacent nightlies
   (e.g. ADR-345) already established for a comparable negative result.
