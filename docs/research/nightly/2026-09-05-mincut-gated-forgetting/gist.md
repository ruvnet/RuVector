# Why I Rejected My Own Idea: Mincut-Gated Forgetting for Agent Memory

## Problem

RuVector's agent-memory crate compacts a store by scoring every memory
independently — recency, access frequency, and similarity to a recent query
context — and keeping the top-scoring subset. That works well on average,
but it has a blind spot: a memory that is the *only* semantic link between
two otherwise unrelated topics can score badly on every one of those terms
(it's old, rarely touched, and off-topic relative to whatever the agent is
currently focused on) and get evicted, quietly severing a connection the
rest of the store depended on.

The workspace this crate lives in already has a general-purpose dynamic
minimum-cut graph engine sitting right next to it. The obvious question: can
you point that engine at the memory store's similarity graph, find the
vertices sitting on the graph's structural "seams," and protect them from
eviction? I built it, benchmarked it honestly, and it doesn't work — for two
independent, measured reasons. This is the writeup of why.

## Technical Design

Two new compaction-policy variants, both built on top of the existing
scalar scoring function:

- **Soft**: build a k-nearest-neighbor cosine-similarity graph over the
  candidate memories, hand it to the mincut engine's `partition()` call,
  flag every vertex with an edge crossing that partition as "structurally
  load-bearing," and add a fixed bonus to its existing scalar score before
  ranking.
- **Hard**: same boundary detection, but instead of a bonus, reserve a
  fraction of the eviction budget specifically for the highest-scoring
  boundary vertices, then fill the rest normally.

Alongside this, a second, independent piece: an eviction *witness*. The
crate already had tamper-evident, hash-chained records for admitting a new
memory and (in a sibling crate) for serving a retrieval — but nothing
recorded *deletions*. `compact_witnessed` closes that gap by emitting one
chained record per evicted id before the store is actually mutated, reusing
the exact same record format and chaining scheme the admission path already
uses.

## Implementation

Both pieces are real, tested Rust, not a sketch:

- The similarity graph is built with the crate's existing cosine-similarity
  function — no new math.
- The graph is handed to the workspace's existing min-cut library through
  its own public, from-vectors constructor — not reimplemented.
- The witness chain reuses the existing record format, hashing function, and
  "no witness, no mutation" invariant the admission path already enforces —
  one new tag added to an existing extensible enum, nothing new invented.

## Actual Benchmark Evidence

Before running anything, I fixed a falsifiable test: compact an 84-memory
synthetic corpus (six topic clusters plus twelve deliberately placed "bridge"
memories connecting pairs of clusters) down to 50%, and check whether the
mincut-aware policies keep meaningfully more bridges than the plain scalar
baseline, without costing more than 100x the baseline's compaction time.

```text
Policy                           Bridge Surv.    Recall@10  Compaction (us)
----------------------------------------------------------------------------
CoherenceWeighted (baseline)            66.7%       100.0%               47
MincutGatedForgetting-Soft              66.7%       100.0%            86188
MincutGatedForgetting-Hard              66.7%       100.0%            85658

Tamper-detection: 20/20 single-byte-flip tampers detected.

Acceptance:
  Bridge-survival gap: +0.0pp (need >=15pp)  -> FAIL
  Compaction slowdown: ~1,800-2,700x (need <=100x) -> FAIL
  Tamper detection: 20/20 -> PASS
```

Bridge survival is *identical* to the baseline. And the reason isn't subtle
— it's a hard performance wall. Before running that benchmark, I profiled
the mincut engine's core `partition()` call on its own, on graphs with
nothing else going on:

```text
n=50   -> 76.8ms
n=100  -> 481.3ms
n=200  -> 2,712.9ms
n=400  -> 11,415.0ms
```

That is worse-than-quadratic scaling, and it rules out using this call
per-compaction on any corpus bigger than a few hundred vectors — the
originally-planned 2,000-memory experiment was simply not runnable in a
reasonable amount of time, which is itself why the corpus above shrank to 84.

On top of that, a second problem: I built a tiny, hand-crafted 19-vertex
graph with a mathematically provable unique weakest link (a single vertex
whose only two edges are the sole bridge between two dense 9-vertex
cliques), and called the same partition function on it, unchanged, 30 times
in a row:

```text
30 calls, avg 841ms/call
15/30 (50%) returned no usable result at all
```

Same input, same function, half the time it finds nothing. I didn't find
any explicit randomness in the library's relevant code paths, so my best
guess is this comes from hash-map iteration order feeding into internal
tie-breaking rather than an intentional randomized algorithm — but either
way, a caller can't treat a single call's result as reliable.

## Limitations

- I only ran the effectiveness benchmark at one corpus size (84 memories) —
  the size the performance wall forced me down to. I don't know whether the
  "no measured benefit" result holds at a larger, faster-computed scale,
  because no such scale was reachable in this session.
- The 19-vertex non-determinism reproduction is a single, deliberately
  simple topology. I didn't verify how often the same failure shows up on
  denser, more realistic graphs beyond noting that boundary-set sizes did
  visibly vary call-to-call at n=84 too.
- I didn't try the mincut library's lower-level API surface directly
  (bypassing the convenience wrapper I used) — that's the most promising
  next step, not a dead end.

## Production Relevance

None, today. The witness half of this work (auditable, tamper-evident
eviction records) is genuinely production-ready and independent of the
rejected hypothesis — it ships, off the back of this same PR, as an
always-available function any existing compaction policy can use. The
mincut half does not ship as anything more than an off-by-default,
feature-gated experiment kept around specifically so the evidence isn't
lost and so a future attempt doesn't have to rediscover these two failure
modes from scratch.

## RuVector Ecosystem Implications

This is a two-crate integration test between an existing graph-algorithms
crate and an existing agent-memory crate, and the honest result is: the
integration point I tried isn't ready. That's useful information for anyone
who owns the graph-algorithms crate — it now has a concrete, reproducible
latency table and a concrete, reproducible non-determinism repro case,
neither of which existed before this session, and both runnable via checked-
in example scripts.

## Future Direction

1. Redo this exact experiment against the mincut library's more primitive,
   presumably-faster APIs instead of its convenience wrapper.
2. Root-cause and fix the non-determinism inside that library.
3. Only then, re-run this unmodified benchmark and see if the picture
   changes.
4. Independently of all of the above, wire a real digital signature into the
   eviction-witness chain this work already built — it's useful on its own
   and doesn't need the mincut question resolved first.

## References

- `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md` — full
  methodology, complete raw output, and the "why RuVector" ecosystem
  analysis.
- `docs/adr/ADR-345-mincut-gated-forgetting.md` — the architectural decision
  record.
- `docs/research/nightly/2026-06-14-agent-memory-compaction/README.md` — the
  original scalar-scoring baseline this work measured against.
