# ADR-013: Retrieval-Based Species Identification

## Status
Accepted

## Date
2026-08-03

## Context

The root `README.md` documents `POST /api/identify` returning ranked species with
confidence scores. No such route exists, and no classifier exists to serve it. Grepping
the workspace for a classification head, a label set, a logits tensor, or a
species→index mapping returns nothing.

This is not an oversight in the API layer. The model choice makes it structural: Perch
2.0 as integrated here is an **embedding model**. The ONNX session in
`onnx_inference.rs` has one input and one output, and that output is the 1536-D vector.
There are no logits to decode.

Meanwhile `sevensense-core/src/config.rs` defaults `model_name` to `"birdnet-v2.4"` at
1024 dimensions, contradicting the Perch 2.0 / 1536-D constants used everywhere else.
Whoever wires the API to the real crates will hit this.

Three options were available: add a classifier head, swap to BirdNET, or identify by
retrieval. This ADR chooses retrieval.

## Decision

Species identification is **k-nearest-neighbour retrieval against a labelled reference
index**, not classification. There is no classifier head and no fixed label set.

### 1. Why retrieval

- **Open-set by construction.** A classifier can only emit labels in its training set.
  A retrieval system returns "nearest known neighbours, all far away", which is the
  correct answer for an unknown species — and detecting the unknown is a stated goal of
  the anomaly-detection use case.
- **Extensible without retraining.** Adding a species means adding labelled reference
  embeddings. Adding a class to a classifier means retraining and redeploying.
- **Taxon-agnostic.** `TaxonId` is already an open newtype over a string with no fixed
  vocabulary. Nothing in the pipeline is bird-specific; Perch 2.0 is multi-taxa with
  documented marine transfer. Frogs, bats, and insects need reference recordings, not
  code changes.
- **Evidence-backed.** A confidence number from a softmax is not explainable. "These
  five labelled recordings are nearest, at these distances, from these locations" is
  auditable — and it is exactly what the RAB evidence-pack design (ADR-006) already
  assumes.

### 2. Scoring

Cosine similarity over L2-normalized embeddings, aggregated per taxon over the k
nearest neighbours (default k = 25):

```
score(taxon) = Σ_{i ∈ knn, label(i) = taxon} w(d_i)
w(d) = exp(−d² / 2σ²)                        σ = 0.35
```

Scores are normalized across candidate taxa to sum to 1. Distance weighting matters:
unweighted vote counting lets 12 mediocre matches outrank 3 excellent ones, which is
the wrong answer whenever the reference corpus is unbalanced — and reference corpora
are always unbalanced.

**These normalized scores are not calibrated probabilities and must not be presented as
such.** They are reported as `score`, never `probability` or `confidence`, and the
response carries the raw neighbour distances so a caller can judge for itself.

### 3. Open-set rejection

If the nearest neighbour's cosine distance exceeds `unknown_threshold` (default 0.55),
the result is `Unknown` with the neighbours still attached. Returning a confident label
for a sound unlike anything in the index is the failure mode most likely to cause a
false conservation record, and it is worth an explicit branch.

### 4. Response

```jsonc
{
  "segment_id": "seg_01J…",
  "status": "identified",            // identified | unknown | insufficient_quality
  "candidates": [
    { "taxon": "Anthus_trivialis", "common_name": "Tree pipit",
      "score": 0.62, "n_neighbors": 14, "min_distance": 0.19 }
  ],
  "neighbors": [
    { "segment_id": "ref_…", "taxon": "Anthus_trivialis",
      "distance": 0.19, "source": "xeno-canto:XC123456" }
  ],
  "features": { "centroid_hz": 4200.0, "tonality": 0.72 },
  "quality": "high",
  "latency_ms": 31
}
```

Every identification carries its evidence. The `neighbors` array is what makes the
result checkable, so it is not optional.

### 5. Quality gating

Segments failing ADR-011 quality checks return `insufficient_quality` without running
inference — wind (high energy, low tonality, centroid below 500 Hz), clipping (crest
→ 1), or energy below the gate. This is both a correctness and a cost measure: it
keeps garbage out of the index and saves the inference budget for real calls.

### 6. Reference index

Reference embeddings are labelled segments in the HNSW index, tagged
`is_reference = true`. The index is seeded from public corpora (xeno-canto, iNaturalist)
and grows as users label clusters via the existing
`PUT /api/v1/clusters/:id/label`. That endpoint already exists and is currently the
only route by which a species name can enter the system; this design makes it the
intended one rather than an accident.

Provenance (`source`) is mandatory on reference entries. An unattributed reference
recording cannot be audited, and an index of unauditable references is not evidence.

### 7. Config correction

`sevensense-core/src/config.rs` is corrected to Perch 2.0 at 1536 dimensions. The
1024-D `birdnet-v2.4` default is wrong for the integrated model and would silently
produce dimension-mismatched vectors on first real use.

## Consequences

### Positive
- Open-set: unknowns are detectable rather than forced into the nearest class.
- New taxa need data, not retraining.
- Every result is auditable against specific reference recordings.
- No label set to maintain, version, or disagree with a taxonomy revision.

### Negative
- Accuracy depends entirely on reference coverage. A species absent from the index is
  unidentifiable — whereas a trained classifier would at least know its own classes.
  This is the central trade of the design.
- Latency includes a k-NN query. HNSW keeps this at single-digit milliseconds, so it is
  not the bottleneck; inference is.
- Scores are not calibrated. Naming them `score` mitigates but does not eliminate the
  risk that a downstream consumer treats them as probabilities.

### Risks
- Reference-corpus bias propagates directly into results, and unbalanced corpora bias
  toward well-sampled taxa. Distance weighting reduces this; it does not remove it.
  Per-taxon reference counts are exposed via `n_neighbors` so the bias is visible.

## Alternatives Considered

**Add a classifier head on Perch embeddings.** A linear probe over 1536-D features is
cheap to train and would give calibrated probabilities. Rejected as the *primary*
mechanism because it reintroduces a closed label set and a retraining cycle. It remains
a reasonable future addition *alongside* retrieval, where the two can cross-check —
disagreement between a probe and k-NN is itself a useful signal.

**Switch to BirdNET.** Ships ~6000 avian classes with calibrated outputs and would work
immediately. Rejected: it is bird-only, which forecloses the multi-taxa direction, and
it provides no embedding space for the similarity search and manifold view that are the
platform's distinguishing features.

**Hyperbolic k-NN over the Poincaré ball.** Promising for taxonomic hierarchy and
already implemented in `sevensense-vector::hyperbolic`. Deferred: it needs embeddings
trained in hyperbolic space to be meaningful, and Perch outputs Euclidean vectors.

## References
- ADR-006 (data architecture / evidence packs), ADR-007 (inference), ADR-010, ADR-011
