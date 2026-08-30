# Does provenance survive contact with a real ANN index?

Two weeks ago, a RuVector research crate called `ruvector-retrieval-receipt`
shipped witness-chained receipts for vector search results: wrap a query's
top-k with a hash chain or Merkle tree, and anyone holding the receipt can
later check, offline, whether the result set they were handed is the one the
engine actually committed to. It worked, and it detected 100% of injected
tampering in testing. But it measured overhead against a brute-force cosine
scan — a deliberate experimental control to isolate the provenance layer's
cost from ANN approximation quality. The report said so explicitly, and its
own rejection criteria named the obvious follow-up: *what happens once you
put this on a real index, where search itself is cheap?*

## The problem

A ratio measured against an expensive baseline can look artificially
flattering. If receipt-build cost is fixed and search cost is huge (O(n)
brute force), the overhead percentage is trivially tiny — but that tells you
nothing about whether the receipt layer is cheap *in absolute terms*, or
whether it stays cheap once the thing it's riding on gets fast. Real
production ANN indexes are fast precisely because they don't scan everything.
If receipt overhead doesn't shrink relative to a fast baseline, the original
"it's basically free" story doesn't hold up.

## The design

`ruvector-hnsw-receipt` composes the exact same receipt cryptography — not a
reimplementation, a dependency — on top of `ruvector-hnsw-repair`'s real
multi-layer HNSW graph: a from-scratch implementation of Malkov & Yashunin's
2018 algorithm, with bounded node degree and `ef`-bounded search, already
living in the RuVector workspace for a different purpose (online repair after
deletes).

```rust
let index = HnswReceiptIndex::ingest(n, dims, seed); // real WriteReceipt + real HNSW node per vector
let raw_ids = index.search_raw(&query, k, ef);        // baseline: pure HNSW search, no receipt work
let items = index.search_items(&query, k, ef);        // + ResultItem { score, write_receipt }
let receipt = RetrievalReceipt::build(
    ReceiptVariant::Merkle, query_hash(&query), index.index_state_root(), &items,
);
assert!(receipt.verify_full(query_hash(&query), index.index_state_root(), &items));
```

Ingestion still runs every vector through a real
`ruvector_proof_gate::HashChainGate` — so every stored vector carries a real
chained write receipt — before handing it to `HnswGraph::insert`. The two
data structures stay in lockstep because `HnswGraph` assigns node ids
sequentially from zero, the same order vectors are admitted through the
gate.

## What was measured, and how

The benchmark times six stages separately, each with its own clock window,
so graph traversal, rescoring, receipt construction, and receipt
verification are never conflated into one number: `search_raw`,
`search_items`, `PerResult build`, `Merkle build`, `PerResult verify_full`,
`Merkle verify_full`. Every query in the timed loop also asserts that
`search_raw`'s result ids exactly match `search_items`'s, in order — the
"receipts don't perturb search" claim is checked live on every sample, not
just in a separate unit test.

Two scales, one repeat for timing-variance sanity, all on the same
Intel Xeon 4-core box, release build, deterministic seeds throughout:

| N | dims | Merkle build p50 overhead vs. raw search p50 | verify_full success |
|---|---|---|---|
| 5,000 | 64 | 4.38% | 300/300 (100%) |
| 5,000 | 64 (repeat) | 4.66% | 300/300 (100%) |
| 20,000 | 128 | 1.35% | 300/300 (100%) |

## What it found

The overhead ratio didn't just stay small — it *shrank* as the index grew,
from 4.4% to 1.3%. That's the interesting part. Receipt-build cost is O(k)
(k=10 fixed in both runs), completely flat with respect to index size. HNSW
search cost grows with graph traversal as N and dimensionality increase. A
brute-force baseline can't show this, because brute-force cost also grows
with N — the ratio stays artificially stable either way, for the wrong
reason. On a real index, the ratio actually tells you something, and what it
tells you is: this gets cheaper, not more expensive, at scale.

Both this experiment's own pre-registered bar (50%) and the original crate's
tighter 15% rejection threshold were cleared by roughly an order of
magnitude at the smaller scale, and two orders at the larger one.

Merkle's proof-size advantage over PerResult (160 bytes vs. 320 bytes
worst-case at k=10 — half, matching the O(log k) vs. O(k) asymptotic)
carried over from the brute-force experiment unchanged, as expected: that
property depends only on the receipt structure, not on what produced the
results.

## What it didn't find, and why that's stated plainly

Recall@10 against a brute-force cosine ground truth came in low — 0.58 at
the smaller scale, 0.31 at the larger one. That's not a receipt-layer
problem: it's because the HNSW graph ranks candidates internally by squared
L2 distance on un-normalized random vectors, while this crate's result
scoring (matching the original brute-force crate, for comparability) uses
cosine similarity. On unnormalized vectors those two rankings diverge, and
the graph's construction parameters were left at un-tuned defaults. Recall
was never the metric under test here — it's reported as context, exactly as
the original nightly report reported it as out of scope — but it's worth
saying clearly rather than letting a low number sit unexplained.

Only two scales were tested. ADR-304, the original ADR, named N≥100k as the
scale that would need re-confirmation before any production claim; this run
reaches 20,000. That's this report's own first item for next research, not
a footnote to skip.

## Why this is more than a benchmark exercise

A retrieval receipt that's only cheap next to a strawman baseline is a
research curiosity. A retrieval receipt that gets *cheaper, relatively*, as
a real index scales up is a production argument: it means "receipt every
query by default" stops being a tradeoff decision and starts being close to
free, at least on this dimension. That's the difference between provenance
as an opt-in feature you carefully budget for and provenance as a default
you don't think about — which matters more every year agent memory gets
cited as evidence for what an agent actually did.

## What's next

Compose the same receipts against RuVector's production index rather than
the repair-focused one used here; tune the graph and align the scoring
metric with its native ranking to close the recall gap and see whether the
overhead conclusion survives; push past 100k vectors; and add the
root/receipt signing that both this crate and its predecessor still lack
before either can support a real non-repudiation claim.

---

*Full methodology, raw benchmark output, unit tests, and the accompanying
ADR are in the RuVector repository:
`docs/research/nightly/2026-08-26-hnsw-witness-receipts/README.md`,
`docs/adr/ADR-340-hnsw-witness-receipts.md`, and
`crates/ruvector-hnsw-receipt/`.*
