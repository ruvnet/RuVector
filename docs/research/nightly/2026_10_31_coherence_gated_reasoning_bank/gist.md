# Why a corroboration graph vanished at scale: a ReasoningBank retrieval experiment

## The problem

Agent-memory frameworks that retrieve stored reasoning strategies by
embedding similarity — Google/Yale/UIUC's ReasoningBank (arXiv:2509.25140)
among them — share a blind spot: cosine similarity cannot tell the
difference between a pattern that many independent trajectories agree is
the right strategy for a region of embedding space, and a pattern that
simply landed close to the query by chance (or was placed there
deliberately). Both rank identically if their raw similarity to the query
is identical.

SONA's `ReasoningBank` (`crates/sona/src/reasoning_bank.rs` in this
repository) has exactly this retrieval shape: `find_similar` scores every
stored pattern by cosine similarity to the query alone, sorts, and returns
the top-k. It is simple and fast, and it has no way to notice that one of
its top-10 results is corroborated by five siblings while another is alone
in its region of the embedding space.

## The idea: blend similarity with corroboration

The fix other systems reach for is well precedented. CAGE
(arXiv:2609.04647) reranks RAG passages by blending each passage's
similarity to the query with its similarity to the *other retrieved
passages* — a coherence score. GRADA builds a similarity graph over
retrieved documents and penalizes documents that are query-relevant but
structurally unsupported by the rest of the set, specifically to catch
adversarial documents that "look relevant but lack coherence."

The natural translation to ReasoningBank: build a graph over stored
*patterns* (not retrieved results), with an edge between any two patterns
whose centroids are mutually similar above some threshold. A pattern's
*gate* value is its degree in that graph, normalized to `[0, 1]`. Final
retrieval score: `alpha * cosine_similarity + (1 - alpha) * gate`. A
pattern corroborated by several siblings gets a boost; an isolated pattern
— however close to the query — does not.

This is what `crates/sona/src/coherence_gate.rs` implements, as an additive
module alongside (not inside) `ReasoningBank`:

```rust
pub struct PatternCoherenceGraph { /* degree per pattern id, normalized */ }

impl PatternCoherenceGraph {
    pub fn build(patterns: &[LearnedPattern], edge_threshold: f32) -> Self {
        // O(n^2) pairwise cosine; edge if similarity >= edge_threshold.
        // gate(id) = degree(id) / (n - 1)
    }
}

pub fn find_similar_coherence_gated(
    patterns: &[LearnedPattern],
    graph: &PatternCoherenceGraph,
    query: &[f32],
    k: usize,
    alpha: f32,
) -> Vec<(u64, f32)> {
    // score = alpha * cosine(pattern, query) + (1 - alpha) * graph.gate(pattern.id)
}
```

A second function, `find_similar_quality_weighted`, is an ablation: it
blends cosine similarity with each pattern's existing (here deliberately
uninformative) `avg_quality` metadata field instead of any graph signal,
to check whether the graph is doing something a reused bookkeeping field
would not.

## The experiment

Build a synthetic 64-dimensional pattern bank: 8 true clusters of 6 tightly
clustered patterns each (mutual cosine similarity ~0.95), padded with
random noise singletons up to a target bank size (200 to 5000 patterns
tested). For each of 30 query trials (3 seeds x 10 queries), draw a query
near one true cluster's centroid, and inject one adversarial decoy: a
pattern that is a near-perfect cosine match for *that specific query*
(similarity ~0.99) but isolated — below the edge threshold — from every
true cluster. The decoy models exactly the GRADA failure mode: query-
relevant, structurally unsupported.

Pre-registered, fixed-before-running thresholds: coherence-gated retrieval
must beat cosine-only baseline by at least 15 percentage points of
precision@10 (fraction of the top-10 results that are genuinely members of
the query's own cluster), the quality-weighted ablation must show no
comparable improvement, and gated retrieval's latency must stay within 2x
of baseline at bank sizes up to 2000.

## The result

The decoy appeared in the baseline's top-10 in 30 of 30 trials, at every
bank size tested. Coherence gating demoted it from the top-10 in 0 of 30
trials, at every bank size. Measured precision lift ranged from +0.00 to
+0.67 percentage points — nowhere near the 15-point bar, and, if anything,
*shrinking* as the bank grew (from the already-tiny +0.33pp at n=200 to
+0.00pp at n=5000). Latency stayed within 1.2x-1.52x of baseline at every
scale — comfortably under the 2x bound — but that is irrelevant once the
primary criterion fails this badly.

## Why: the denominator ate the signal

`gate(id) = degree(id) / (n - 1)`. A true cluster member's degree is capped
at 5 (its cluster has 6 members total) no matter how large the bank gets.
At `n = 200`, that's `gate ~= 5/199 ~= 0.025`. At `n = 5000`, it's
`5/4999 ~= 0.001`. At `alpha = 0.5`, the gate term contributes at most
`0.5 * 0.025 = 0.0125` to the final score even at the *smallest* bank
tested — nowhere near enough to overcome the decoy's `0.5 * 0.99 ~= 0.495`
versus a true member's `0.5 * 0.90 ~= 0.45` raw-similarity gap. The gate
term is numerically noise. And the problem gets strictly worse as the bank
grows, because the denominator (whole-bank size) grows while a cluster's
absolute degree does not.

This is not a tuning accident specific to this corpus — it is the direct,
provable consequence of normalizing a *local* structural signal
(corroboration by one's own cluster) by a *global* quantity (the size of
the entire store). Any coherence-graph gate built this way will behave the
same way: fine on a tiny bank where `n - 1` is itself small, and vanishing
exactly as the bank reaches the sizes where a corroboration signal would
matter most.

## What this doesn't mean

It doesn't mean graph-based corroboration is a dead end for ReasoningBank-
style retrieval — CAGE and GRADA's results in the broader RAG-reranking
literature suggest otherwise. It means *this specific normalization choice*
fails, in a way that is cheap to diagnose (an instrumented decoy-tracking
pass immediately shows the gate is near-zero for everyone) and that points
directly at the fix: normalize degree by a local quantity — the retrieval
candidate pool, or a fixed neighborhood size — rather than the whole bank.
That fix was deliberately not implemented in this pass, to avoid the
appearance (or the reality) of tuning parameters until a pre-registered
threshold passes. It is the first item in this write-up's "Next Research."

It also doesn't mean degree-based corroboration, even corrected, would be a
*security* mechanism. The adjacent literature on GraphRAG poisoning reports
that adding supporting relations to a document graph can raise an attacker's
success rate 40-60% — i.e., an adversary who can write several mutually-
corroborating fake patterns, rather than one isolated decoy, would get
rewarded by exactly the mechanism meant to catch isolated spurious matches.
This run only tested the single-decoy case. A corrected gate would still
need a provenance signal, not just an embedding-proximity one, to resist a
motivated writer — and this ecosystem already has witness/provenance
machinery elsewhere (`ruvector-mincut`'s witness module,
`ruvector-retrieval-receipt`) that would be the more defensible direction
for that specific threat model.

## Takeaway

A coherence gate for pattern retrieval is a reasonable idea with real
precedent (CAGE, GRADA) — but "reasonable idea" and "correctly normalized
implementation" are different claims, and only measuring the second one
honestly is what turned a plausible-sounding mechanism into a documented,
specific, falsifiable bug report instead of an unverified assumption
shipped into `ReasoningBank`.
